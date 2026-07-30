from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.experiments.answer_utility.config import (
    AnswerSupervisionView,
    AnswerUtilityExperimentVariant,
    answer_utility_experiment_profile,
)
from tgvf_rl.representation.experiments.answer_utility.native_pipeline import (
    Qwen3AnswerUtilityGroupBuilder,
)
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.native_pipeline import (
    REPRESENTATION_PROMPT_IDENTITY,
    REPRESENTATION_PROMPT_SCHEMA_VERSION,
    Qwen3NativeRepresentationGroupBuilder,
    RepresentationPromptConfig,
)
from tgvf_rl.representation.training.runtime import (
    QWEN3_REPRESENTATION_BRANCH_LAYERS,
    create_qwen3_representation_runtime,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


_IMAGE_TOKEN = "<|image_pad|>"
_IM_END_TOKEN = "<|im_end|>"
_ASSISTANT_PREFILL = "<|im_start|>assistant\n<think>\n"
_EVIDENCE_SENTINEL = "GOLD_EVIDENCE_SENTINEL_MUST_NOT_ENTER_CLEAN_CONTEXT"


class _Tokenizer:
    """Char-level fixture with native image/EOS tokens kept indivisible."""

    is_fast = True
    chat_template = "tiny-answer-utility-template-v1"

    def __init__(self) -> None:
        self.name_or_path = "/tiny-answer-utility-qwen3"
        self._piece_to_id = {_IMAGE_TOKEN: 7, _IM_END_TOKEN: 8}
        self._id_to_piece = {7: _IMAGE_TOKEN, 8: _IM_END_TOKEN}
        self._next_id = 9

    def __len__(self) -> int:
        return 1024

    def _id(self, piece: str) -> int:
        if piece not in self._piece_to_id:
            self._piece_to_id[piece] = self._next_id
            self._id_to_piece[self._next_id] = piece
            self._next_id += 1
        return self._piece_to_id[piece]

    def _tokenize(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        pieces: list[str] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            special = next(
                (
                    token
                    for token in (_IMAGE_TOKEN, _IM_END_TOKEN)
                    if text.startswith(token, cursor)
                ),
                None,
            )
            end = cursor + (len(special) if special is not None else 1)
            pieces.append(text[cursor:end])
            offsets.append((cursor, end))
            cursor = end
        return pieces, offsets

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        pieces, _offsets = self._tokenize(text)
        return [self._id(piece) for piece in pieces]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> dict[str, list[object]]:
        assert not add_special_tokens and return_offsets_mapping and not truncation
        pieces, offsets = self._tokenize(text)
        return {
            "input_ids": [self._id(piece) for piece in pieces],
            "offset_mapping": offsets,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._id(token)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._id_to_piece[token_id]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return "".join(self._id_to_piece[int(token_id)] for token_id in token_ids)


class _Processor:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template
        self.image_processor = SimpleNamespace(merge_size=2, patch_size=16)

    @staticmethod
    def _user(messages: list[dict[str, object]]) -> str:
        content = messages[0]["content"]
        assert isinstance(content, (tuple, list))
        prompt = next(
            item["text"]
            for item in content
            if isinstance(item, dict) and item["type"] == "text"
        )
        has_image = any(
            isinstance(item, dict) and item["type"] == "image" for item in content
        )
        return f"<user>{_IMAGE_TOKEN if has_image else ''}{prompt}</user>\n"

    @staticmethod
    def _call(messages: list[dict[str, object]]) -> str:
        tool_calls = messages[1]["tool_calls"]
        assert isinstance(tool_calls, (tuple, list))
        function = tool_calls[0]["function"]
        payload = json.dumps(
            {"name": function["name"], "arguments": function["arguments"]},
            ensure_ascii=False,
        )
        return f"<tool_call>\n{payload}\n</tool_call>"

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tools[0]["function"]["name"] == "tgvf_focus_tool"
        assert not tokenize
        user = self._user(messages)
        if len(messages) == 1:
            assert add_generation_prompt
            return user + _ASSISTANT_PREFILL
        call = self._call(messages)
        reasoning = messages[1]["reasoning_content"]
        history = (
            user
            + _ASSISTANT_PREFILL
            + str(reasoning)
            + "\n</think>\n\n"
            + call
            + _IM_END_TOKEN
            + "\n"
        )
        if len(messages) == 2:
            assert not add_generation_prompt
            return history
        history += f"<tool_response>{_IMAGE_TOKEN}</tool_response>\n"
        if len(messages) == 3:
            assert add_generation_prompt
            return history + _ASSISTANT_PREFILL
        assert len(messages) == 4 and not add_generation_prompt
        final = messages[-1]
        return (
            history
            + _ASSISTANT_PREFILL
            + str(final["reasoning_content"])
            + "\n</think>\n\n"
            + str(final["content"])
            + _IM_END_TOKEN
            + "\n"
        )

    def __call__(self, *, text, images, padding, return_tensors):
        assert len(text) == 1 and not padding and return_tensors == "pt"
        canonical_ids = self.tokenizer.encode(text[0], add_special_tokens=False)
        visual_id = self.tokenizer.convert_tokens_to_ids(_IMAGE_TOKEN)
        assert canonical_ids.count(visual_id) == len(images)
        return {
            "input_ids": torch.tensor([canonical_ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(canonical_ids), dtype=torch.long),
            "pixel_values": torch.arange(len(images) * 4 * 3, dtype=torch.float32).view(
                len(images) * 4, 3
            ),
            "image_grid_thw": torch.tensor(
                [[1, 2, 2] for _ in images], dtype=torch.long
            ),
        }


class _Merger(nn.Module):
    def __init__(self, vision_width: int, language_width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(vision_width * 4, language_width, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states.reshape(-1, hidden_states.shape[-1] * 4))


class _VisionTower(nn.Module):
    def __init__(self, *, vision_width: int, language_width: int) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(3, vision_width, bias=False)
        self.merger = _Merger(vision_width, language_width)
        self.deepstack_merger_list = nn.ModuleList(
            [_Merger(vision_width, language_width) for _ in range(3)]
        )

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor):
        del grid_thw
        hidden = self.patch_projection(pixel_values)
        return self.merger(hidden), [
            merger(hidden + index + 1)
            for index, merger in enumerate(self.deepstack_merger_list)
        ]


class _LanguageModel(nn.Module):
    def __init__(self, vocabulary: int, width: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocabulary, width)

    def get_input_embeddings(self):
        return self.embed_tokens


class _Core(nn.Module):
    def __init__(self, visual: _VisionTower, language_model: _LanguageModel) -> None:
        super().__init__()
        self.visual = visual
        self.language_model = language_model

    def get_rope_index(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw,
        attention_mask: torch.Tensor,
    ):
        del attention_mask
        assert video_grid_thw is None
        assert image_grid_thw.shape in {(1, 3), (2, 3)}
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return (
            positions.view(1, 1, -1).expand(3, 1, -1),
            torch.zeros(1, 1, dtype=torch.long),
        )


class _TinyQwen3(nn.Module):
    def __init__(self, tokenizer_length: int) -> None:
        super().__init__()
        vision_width = 4
        language_width = 6
        self.model = _Core(
            _VisionTower(vision_width=vision_width, language_width=language_width),
            _LanguageModel(tokenizer_length + 4, language_width),
        )
        self.lm_head = nn.Linear(language_width, tokenizer_length + 4, bias=False)
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            _name_or_path="/tiny-answer-utility-qwen3",
            vision_config=SimpleNamespace(
                hidden_size=vision_width,
                out_hidden_size=language_width,
                spatial_merge_size=2,
                deepstack_visual_indexes=QWEN3_REPRESENTATION_BRANCH_LAYERS,
            ),
            text_config=SimpleNamespace(hidden_size=language_width),
        )


def _prompt() -> RepresentationPromptConfig:
    template = "{question}"
    return RepresentationPromptConfig(
        identity=REPRESENTATION_PROMPT_IDENTITY,
        template=template,
        expected_sha256=hashlib.sha256(template.encode()).hexdigest(),
        schema_version=REPRESENTATION_PROMPT_SCHEMA_VERSION,
    )


def _sample(image: Path, index: int) -> RepresentationTrainingSample:
    answers = ("OPEN", "CLOSED")
    return RepresentationTrainingSample(
        sample_id=f"answer-utility-{index}",
        image=str(image),
        image_id="shared-answer-utility-image",
        question="What is written on the selected label?",
        target=f"label section {index}",
        evidence_description=(
            _EVIDENCE_SENTINEL if index == 0 else "SECOND_GOLD_EVIDENCE_SENTINEL"
        ),
        short_answer=answers[index],
    )


def _runtime():
    tokenizer = _Tokenizer()
    processor = _Processor(tokenizer)
    identity = ModelIdentity(
        family="qwen3_vl",
        model_name="tiny-answer-utility-qwen3",
        revision_or_path="/tiny-answer-utility-qwen3",
        tokenizer_length=len(tokenizer),
        chat_template_sha256=hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
    )
    return create_qwen3_representation_runtime(
        model=_TinyQwen3(len(tokenizer)),
        processor=processor,
        model_identity=identity,
        conditioning_config=TargetConditioningConfig(
            provider=TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING,
            embedding_identity=(
                "/tiny-answer-utility-qwen3::language_model.input_embeddings"
            ),
        ),
        adapter_dtype=torch.float32,
        fixture_mode=True,
    )


def _builders(tmp_path: Path, variant: AnswerUtilityExperimentVariant):
    image = tmp_path / "shared-image.bin"
    image.write_bytes(b"answer-utility-image-fixture")
    runtime = _runtime()
    prompt = _prompt()
    base = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=Qwen3VLAdapter(),
        prompt=prompt,
        image_loader=lambda path: Path(path).read_bytes(),
    )
    wrapped = Qwen3AnswerUtilityGroupBuilder(
        base_builder=base,
        runtime=runtime,
        prompt=prompt,
        profile=answer_utility_experiment_profile(variant),
    )
    return runtime, base, wrapped, (_sample(image, 0), _sample(image, 1))


def _owned_labels(labels: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(label for label in labels if label != EVIDENCE_IGNORE_INDEX)


def test_clean_d_only_transcript_excludes_source_and_gold_evidence_and_labels_answer_eos(
    tmp_path: Path,
) -> None:
    runtime, _base, builder, samples = _builders(
        tmp_path, AnswerUtilityExperimentVariant.E4
    )

    group = builder(
        samples,
        runtime.adapter,
        collective_candidate_count=len(samples),
    )

    assert group.supervision_view is AnswerSupervisionView.CLEAN_D_ONLY
    assert tuple(item.sample_id for item in group.answer_supervisions) == tuple(
        sample.sample_id for sample in samples
    )
    for sample, supervision, control in zip(
        samples,
        group.answer_supervisions,
        group.controls,
        strict=True,
    ):
        assert supervision.context_kind == "clean_d_only"
        assert supervision.evidence_field_injected is False
        assert supervision.source_positions == ()
        assert len(supervision.d_positions) == 1
        assert supervision.transcript.text.count(_IMAGE_TOKEN) == 1
        assert sample.evidence_description not in supervision.transcript.text
        assert _EVIDENCE_SENTINEL not in supervision.transcript.text
        expected = tuple(
            runtime.tokenizer.encode(sample.short_answer, add_special_tokens=False)
        ) + (runtime.tokenizer.convert_tokens_to_ids(_IM_END_TOKEN),)
        assert _owned_labels(supervision.labels) == expected
        assert supervision.supervised_positions == (
            *supervision.answer_positions,
            *supervision.eos_positions,
        )
        request = supervision.request(observation=control.correct)
        assert request.use_cache is False
        assert tuple(block.kind for block in request.visual_blocks) == ("focused_d",)
        assert request.visual_blocks[0].positions == supervision.d_positions


def test_gold_evidence_view_preserves_legacy_labels_and_declares_leakage(
    tmp_path: Path,
) -> None:
    runtime, base, builder, samples = _builders(
        tmp_path, AnswerUtilityExperimentVariant.E3
    )
    baseline = base(
        samples,
        runtime.adapter,
        collective_candidate_count=len(samples),
    )
    baseline_labels = tuple(row.supervision.labels for row in baseline.rows)
    baseline_positions = tuple(
        row.supervision.evidence_token_positions for row in baseline.rows
    )

    group = builder(
        samples,
        runtime.adapter,
        collective_candidate_count=len(samples),
    )

    assert tuple(row.supervision.labels for row in group.legacy.rows) == baseline_labels
    assert (
        tuple(row.supervision.evidence_token_positions for row in group.legacy.rows)
        == baseline_positions
    )
    for sample, legacy_row, supervision, control in zip(
        samples,
        group.legacy.rows,
        group.answer_supervisions,
        group.controls,
        strict=True,
    ):
        assert supervision.context_kind == "gold_evidence"
        assert supervision.evidence_field_injected is True
        assert supervision.source_positions == legacy_row.source_positions
        assert supervision.d_positions == legacy_row.d_positions
        assert torch.equal(supervision.input_ids, legacy_row.input_ids)
        assert sample.evidence_description in supervision.transcript.text
        expected = tuple(
            runtime.tokenizer.encode(sample.short_answer, add_special_tokens=False)
        ) + (runtime.tokenizer.convert_tokens_to_ids(_IM_END_TOKEN),)
        assert _owned_labels(supervision.labels) == expected
        assert supervision.labels != legacy_row.supervision.labels
        request = supervision.request(
            observation=control.correct,
            source=group.legacy.source_visual,
        )
        assert request.use_cache is False
        assert tuple(block.kind for block in request.visual_blocks) == (
            "source_image",
            "focused_d",
        )


def test_none_view_is_a_true_noop_over_the_legacy_builder(tmp_path: Path) -> None:
    runtime, _base, builder, samples = _builders(
        tmp_path, AnswerUtilityExperimentVariant.E0
    )

    group = builder(
        samples,
        runtime.adapter,
        collective_candidate_count=len(samples),
    )

    assert group.supervision_view is AnswerSupervisionView.NONE
    assert group.answer_supervisions == ()
    assert group.controls == ()
    assert tuple(row.sample_id for row in group.legacy.rows) == tuple(
        sample.sample_id for sample in samples
    )


def test_answer_suffix_location_cannot_match_inside_native_im_end(
    tmp_path: Path,
) -> None:
    runtime, _base, builder, samples = _builders(
        tmp_path, AnswerUtilityExperimentVariant.E2
    )
    samples = (replace(samples[0], short_answer="end"), samples[1])

    group = builder(
        samples,
        runtime.adapter,
        collective_candidate_count=len(samples),
    )

    supervision = group.answer_supervisions[0]
    expected = tuple(runtime.tokenizer.encode("end", add_special_tokens=False)) + (
        runtime.tokenizer.convert_tokens_to_ids(_IM_END_TOKEN),
    )
    assert _owned_labels(supervision.labels) == expected

    invalid = (replace(samples[0], short_answer=_IM_END_TOKEN), samples[1])
    with pytest.raises(ValueError, match="native (protocol )?control"):
        builder(
            invalid,
            runtime.adapter,
            collective_candidate_count=len(invalid),
        )
