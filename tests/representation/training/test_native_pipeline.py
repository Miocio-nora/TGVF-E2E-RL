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
from tgvf_rl.qwen.base import resolve_language_model
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.internal_evaluation import (
    NativeCausalValueFlipRequest,
    NativeFreeContinuationRequest,
    create_injected_native_counterfactual_evaluator,
)
from tgvf_rl.representation.training.native_pipeline import (
    REPRESENTATION_PROMPT_IDENTITY,
    REPRESENTATION_PROMPT_SCHEMA_VERSION,
    Qwen3NativeRepresentationGroupBuilder,
    RepresentationPromptConfig,
    RepresentationReadoutLossSupervisionBinding,
    _bind_all_ones_attention_mask,
    _expand_native_visual_placeholders,
    _processor_batch,
    _render_native_action_targets_batch,
    _minimal_overlapping_token_positions,
    _single_visual_expansion_count,
    _validate_single_input,
    build_native_representation_messages,
    render_native_action_target,
)
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.readout import (
    HISTORICAL_READOUT_LOSS_SUPERVISION_POLICY_IDENTITY,
    RepresentationReadoutLossSupervision,
)
from tgvf_rl.representation.training.runtime import (
    QWEN3_REPRESENTATION_BRANCH_LAYERS,
    create_qwen3_representation_runtime,
)
from tgvf_rl.representation.training.qwen3_counterfactual import (
    Qwen3CounterfactualCaseBuilder,
    build_qwen3_d_only_messages,
    load_qwen3_counterfactual_manifest,
)
from tgvf_rl.representation.training.schema import (
    RepresentationChoice,
    RepresentationTrainingSample,
)
from tgvf_rl.representation.training.streaming import (
    score_streaming_same_image_group,
)
from tgvf_rl.representation.training.transcript import (
    CanonicalEvidenceSupervision,
    ModelEvidenceSupervision,
    NATIVE_REPRESENTATION_PRE_REASONING,
    render_native_evidence_labels,
)


_IMAGE_TOKEN = "<|image_pad|>"
_ASSISTANT_PREFILL = "<|im_start|>assistant\n<think>\n"


@pytest.mark.parametrize(
    "offsets, expected",
    (
        (((0, 2), (2, 4), (4, 7), (7, 10)), (1, 2)),
        (((0, 3), (3, 6), (6, 8), (8, 10)), (1, 2)),
        (((0, 2), (2, 4), (4, 6), (6, 8), (8, 10)), (1, 2, 3)),
    ),
)
def test_native_target_positions_use_minimal_overlapping_token_cover(
    offsets: tuple[tuple[int, int], ...],
    expected: tuple[int, ...],
) -> None:
    assert (
        _minimal_overlapping_token_positions(
            "0123456789",
            offsets,
            span_start=3,
            span_end=7,
            name="native action target",
        )
        == expected
    )


def test_bound_all_ones_mask_rejects_mutation_replacement_and_false_cpu_mask() -> None:
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    proof = _bind_all_ones_attention_mask(attention_mask)

    _validate_single_input(
        input_ids,
        attention_mask,
        attention_mask_proof=proof,
    )
    with pytest.raises(ValueError, match="does not bind this tensor state"):
        _validate_single_input(
            input_ids,
            attention_mask.clone(),
            attention_mask_proof=proof,
        )
    attention_mask[0, 0] = 0
    with pytest.raises(ValueError, match="does not bind this tensor state"):
        _validate_single_input(
            input_ids,
            attention_mask,
            attention_mask_proof=proof,
        )
    with pytest.raises(ValueError, match="cannot contain masked tokens"):
        _bind_all_ones_attention_mask(torch.tensor([[1, 0]], dtype=torch.long))


def test_materialized_action_revalidates_bound_ids_without_content_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING)
    sample = _sample(tmp_path / "unused.png", 0)
    messages = build_native_representation_messages(sample, _prompt())
    canonical = render_native_action_target(runtime, messages)
    builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=Qwen3VLAdapter(),
        prompt=_prompt(),
        image_loader=lambda _path: b"image",
    )
    action = builder._materialize_action(canonical, b"image")

    def forbidden_tolist(*_args: object, **_kwargs: object) -> list[object]:
        raise AssertionError("bound action validation must not copy tensor contents")

    monkeypatch.setattr(torch.Tensor, "tolist", forbidden_tolist)
    action.assert_bound_invariants()
    action.input_ids[0, 0] += 1
    with pytest.raises(ValueError, match="does not bind this tensor state"):
        action.assert_bound_invariants()


class _Tokenizer:
    is_fast = True
    chat_template = "tiny-native-representation-template-v1"

    def __init__(self) -> None:
        self.name_or_path = "/tiny-native-qwen3"
        self._piece_to_id = {_IMAGE_TOKEN: 7}
        self._id_to_piece = {7: _IMAGE_TOKEN}
        self._next_id = 8

    def __len__(self) -> int:
        return 1024

    def _id(self, piece: str) -> int:
        if piece not in self._piece_to_id:
            self._piece_to_id[piece] = self._next_id
            self._id_to_piece[self._next_id] = piece
            self._next_id += 1
        return self._piece_to_id[piece]

    def _tokenize(self, text: str):
        pieces = []
        offsets = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(_IMAGE_TOKEN, cursor):
                end = cursor + len(_IMAGE_TOKEN)
            else:
                end = cursor + 1
            pieces.append(text[cursor:end])
            offsets.append((cursor, end))
            cursor = end
        return pieces, offsets

    def encode(self, text: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        pieces, _ = self._tokenize(text)
        return [self._id(piece) for piece in pieces]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ):
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
        token_ids,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return "".join(
            self._id_to_piece.get(int(token_id), f"<id:{int(token_id)}>")
            for token_id in token_ids
        )


class _Processor:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template
        self.image_processor = SimpleNamespace(merge_size=2, patch_size=16)
        self.visual_tokens_per_image = 1
        self.calls: list[tuple[str, int]] = []

    @staticmethod
    def _user(messages) -> str:
        prompt = next(
            item["text"] for item in messages[0]["content"] if item["type"] == "text"
        )
        has_image = any(item["type"] == "image" for item in messages[0]["content"])
        return f"<user>{_IMAGE_TOKEN if has_image else ''}{prompt}</user>\n"

    @staticmethod
    def _call(messages) -> str:
        function = messages[1]["tool_calls"][0]["function"]
        payload = json.dumps(
            {"name": function["name"], "arguments": function["arguments"]},
            ensure_ascii=False,
        )
        return f"<tool_call>\n{payload}\n</tool_call>"

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
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
        if len(messages) == 2:
            assert not add_generation_prompt
            return (
                user
                + _ASSISTANT_PREFILL
                + messages[1]["reasoning_content"]
                + "\n</think>\n\n"
                + call
                + "<|im_end|>\n"
            )
        history = (
            user
            + _ASSISTANT_PREFILL
            + messages[1]["reasoning_content"]
            + "\n</think>\n\n"
            + call
            + "<|im_end|>\n"
        )
        history += f"<tool_response>{_IMAGE_TOKEN}</tool_response>\n"
        if len(messages) == 3:
            assert add_generation_prompt
            return history + _ASSISTANT_PREFILL
        assert len(messages) == 4 and not add_generation_prompt
        return (
            history
            + _ASSISTANT_PREFILL
            + messages[-1]["reasoning_content"]
            + "\n</think>\n\n"
            + messages[-1]["content"]
            + "<|im_end|>\n"
        )

    def __call__(self, *, text, images, padding, return_tensors):
        assert len(text) == 1 and not padding and return_tensors == "pt"
        self.calls.append((text[0], len(images)))
        canonical_ids = self.tokenizer.encode(text[0], add_special_tokens=False)
        visual_id = self.tokenizer.convert_tokens_to_ids(_IMAGE_TOKEN)
        assert canonical_ids.count(visual_id) == len(images)
        expanded = []
        for token_id in canonical_ids:
            expanded.extend(
                [token_id] * self.visual_tokens_per_image
                if token_id == visual_id
                else [token_id]
            )
        return {
            "input_ids": torch.tensor([expanded], dtype=torch.long),
            "attention_mask": torch.ones(1, len(expanded), dtype=torch.long),
            "pixel_values": torch.arange(len(images) * 4 * 3, dtype=torch.float32).view(
                len(images) * 4, 3
            )
            % 12,
            "image_grid_thw": torch.tensor(
                [[1, 2, 2] for _ in images], dtype=torch.long
            ),
        }


class _ImageCapRecordingProcessor:
    def __init__(self, *, shortest_edge: object = 3136) -> None:
        self.image_processor = SimpleNamespace(size={"shortest_edge": shortest_edge})
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        image_count = len(kwargs["images"])
        return {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": torch.ones(image_count, 3),
            "image_grid_thw": torch.ones(image_count, 3, dtype=torch.long),
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
        self.forward_calls = 0

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor):
        self.forward_calls += 1
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

    def forward(
        self,
        *,
        inputs_embeds,
        visual_pos_masks,
        deepstack_visual_embeds,
        **kwargs,
    ):
        if inputs_embeds is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                raise ValueError("input_ids are required when inputs_embeds is absent")
            inputs_embeds = self.embed_tokens(input_ids)
        hidden = inputs_embeds.clone()
        if deepstack_visual_embeds is not None:
            for branch in deepstack_visual_embeds:
                hidden = hidden.clone()
                hidden[visual_pos_masks] += branch
        running_hidden = hidden.cumsum(dim=1)
        past_key_values = kwargs.get("past_key_values")
        if past_key_values is not None:
            running_hidden = running_hidden + past_key_values[0]
        output_hidden = hidden + running_hidden * 0.01
        next_cache = (
            (running_hidden[:, -1:].detach(),)
            if kwargs.get("use_cache", False)
            else None
        )
        return SimpleNamespace(
            last_hidden_state=output_hidden,
            past_key_values=next_cache,
        )


class _Core(nn.Module):
    def __init__(self, visual: _VisionTower, language_model: _LanguageModel) -> None:
        super().__init__()
        self.visual = visual
        self.language_model = language_model

    def get_rope_index(
        self,
        *,
        input_ids,
        image_grid_thw,
        video_grid_thw,
        attention_mask,
    ):
        assert video_grid_thw is None and image_grid_thw.shape in {(1, 3), (2, 3)}
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.view(1, 1, -1).expand(3, 1, -1)
        return positions, torch.zeros(1, 1, dtype=torch.long)


class _TinyQwen3(nn.Module):
    def __init__(self, tokenizer_length: int) -> None:
        super().__init__()
        vision_width = 4
        language_width = 6
        self.model = _Core(
            _VisionTower(
                vision_width=vision_width,
                language_width=language_width,
            ),
            _LanguageModel(tokenizer_length + 4, language_width),
        )
        self.lm_head = nn.Linear(language_width, tokenizer_length + 4, bias=False)
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            _name_or_path="/tiny-native-qwen3",
            vision_config=SimpleNamespace(
                hidden_size=vision_width,
                out_hidden_size=language_width,
                spatial_merge_size=2,
                deepstack_visual_indexes=QWEN3_REPRESENTATION_BRANCH_LAYERS,
            ),
            text_config=SimpleNamespace(hidden_size=language_width),
        )

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        output_hidden_states,
        use_cache,
        return_dict,
    ):
        assert output_hidden_states and not use_cache and return_dict
        assert pixel_values.shape[0] == int(image_grid_thw.prod().item())
        main, deepstack = self.model.visual(
            pixel_values,
            grid_thw=image_grid_thw,
        )
        hidden = resolve_language_model(self).get_input_embeddings()(input_ids).clone()
        visual_mask = input_ids == 7
        hidden[visual_mask] = main
        output = self.model.language_model(
            inputs_embeds=hidden,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
            attention_mask=attention_mask,
        )
        return SimpleNamespace(
            hidden_states=(hidden, output.last_hidden_state),
        )


def _sample(image: Path, index: int) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=f"sample-{index}",
        image=str(image),
        image_id="shared-image",
        question="What is written on the label?",
        target=f"label section {index}",
        evidence_description=f"Section {index} reads OPEN.",
        short_answer="OPEN",
    )


def _counterfactual_sample(
    image: Path,
    *,
    sample_id: str,
    image_id: str,
    target: str,
    evidence: str,
) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=sample_id,
        image=str(image),
        image_id=image_id,
        question="What is written on the status label?",
        target=target,
        evidence_description=evidence,
        short_answer=evidence.rsplit(" ", 1)[-1].rstrip("."),
    )


def _prompt() -> RepresentationPromptConfig:
    template = "{question}"
    return RepresentationPromptConfig(
        identity=REPRESENTATION_PROMPT_IDENTITY,
        template=template,
        expected_sha256=hashlib.sha256(template.encode()).hexdigest(),
        schema_version=REPRESENTATION_PROMPT_SCHEMA_VERSION,
    )


class _SparseAnswerBearingReadoutFactory:
    identity = "test-explicit-answer-bearing-readout-policy-v1"

    def __init__(self) -> None:
        self.sample_ids: list[str] = []

    def __call__(
        self,
        sample: RepresentationTrainingSample,
        canonical: CanonicalEvidenceSupervision,
        model: ModelEvidenceSupervision,
    ) -> RepresentationReadoutLossSupervision:
        self.sample_ids.append(sample.sample_id)
        value_start = canonical.evidence_char_start + sample.evidence_description.index(
            sample.short_answer
        )
        value_end = value_start + len(sample.short_answer)
        answer_start = canonical.transcript.text.index(
            sample.short_answer,
            canonical.evidence_char_end,
        )
        answer_end = answer_start + len(sample.short_answer)
        value_canonical_positions = _minimal_overlapping_token_positions(
            canonical.transcript.text,
            canonical.token_offsets,
            span_start=value_start,
            span_end=value_end,
            name="test evidence value",
        )
        answer_canonical_positions = _minimal_overlapping_token_positions(
            canonical.transcript.text,
            canonical.token_offsets,
            span_start=answer_start,
            span_end=answer_end,
            name="test final answer",
        )
        value_positions = tuple(
            position
            for canonical_position in value_canonical_positions
            for position in model.canonical_to_model_positions[canonical_position]
        )
        answer_positions = tuple(
            position
            for canonical_position in answer_canonical_positions
            for position in model.canonical_to_model_positions[canonical_position]
        )
        supervised_positions = tuple(sorted((*value_positions, *answer_positions)))
        supervised = set(supervised_positions)
        labels = tuple(
            token_id if position in supervised else EVIDENCE_IGNORE_INDEX
            for position, token_id in enumerate(model.model_token_ids)
        )
        return RepresentationReadoutLossSupervision(
            policy_identity=self.identity,
            identity=f"{self.identity}:{sample.sample_id}",
            labels=labels,
            supervised_token_positions=supervised_positions,
            evidence_value_token_positions=value_positions,
            answer_token_positions=answer_positions,
            source_image_block_query_start=model.evidence_token_positions[0] - 1,
            source_image_block_query_end=answer_positions[-1],
        )


def _runtime(provider: TargetConditioningProviderKind):
    tokenizer = _Tokenizer()
    processor = _Processor(tokenizer)
    identity = ModelIdentity(
        family="qwen3_vl",
        model_name="tiny-native-qwen3",
        revision_or_path="/tiny-native-qwen3",
        tokenizer_length=len(tokenizer),
        chat_template_sha256=hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
    )
    config = (
        TargetConditioningConfig(provider=provider, hidden_layer=-1)
        if provider is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
        else TargetConditioningConfig(
            provider=provider,
            embedding_identity=("/tiny-native-qwen3::language_model.input_embeddings"),
        )
    )
    return create_qwen3_representation_runtime(
        model=_TinyQwen3(len(tokenizer)),
        processor=processor,
        model_identity=identity,
        conditioning_config=config,
        adapter_dtype=torch.float32,
        fixture_mode=True,
    )


def test_prompt_hash_and_fields_are_explicit() -> None:
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        RepresentationPromptConfig(
            identity="bad",
            template="Question: {question}",
            expected_sha256="0" * 64,
        )
    template = "Target: {target}"
    with pytest.raises(ValueError, match="requires template exactly"):
        RepresentationPromptConfig(
            identity="bad",
            template=template,
            expected_sha256=hashlib.sha256(template.encode()).hexdigest(),
        )
    non_native_template = "Question: {question}"
    with pytest.raises(ValueError, match="requires template exactly"):
        RepresentationPromptConfig(
            identity="bad-v1",
            template=non_native_template,
            expected_sha256=hashlib.sha256(non_native_template.encode()).hexdigest(),
            schema_version=REPRESENTATION_PROMPT_SCHEMA_VERSION,
        )


def test_initial_representation_message_contract_hides_target_from_user(
    tmp_path: Path,
) -> None:
    sample = replace(
        _sample(tmp_path / "unused.png", 0),
        choices=(RepresentationChoice(label="A", text="choice-only text"),),
    )

    native = build_native_representation_messages(sample, _prompt())
    assert native[0] == {
        "role": "user",
        "content": (
            {"type": "image"},
            {"type": "text", "text": sample.question},
        ),
    }
    assert sample.target not in native[0]["content"][1]["text"]
    assert sample.choices[0].text not in native[0]["content"][1]["text"]
    assert native[1]["reasoning_content"] == NATIVE_REPRESENTATION_PRE_REASONING
    assert native[1]["content"] == ""
    assert native[1]["tool_calls"][0]["function"]["arguments"] == {
        "target": sample.target
    }
    assert native[2] == {"role": "tool", "content": ({"type": "image"},)}
    assert native[3] == {
        "role": "assistant",
        "reasoning_content": sample.evidence_description,
        "content": sample.short_answer,
    }
    d_only = build_qwen3_d_only_messages(sample, _prompt())
    assert d_only[1]["reasoning_content"] == NATIVE_REPRESENTATION_PRE_REASONING


def test_processor_batch_forwards_bounded_smart_resize_without_overriding_minimum() -> (
    None
):
    processor = _ImageCapRecordingProcessor(shortest_edge=3136)
    images = (object(), object())

    _processor_batch(
        processor,
        text="native transcript",
        images=images,
        image_max_pixels=262144,
    )

    assert processor.calls == [
        {
            "text": ["native transcript"],
            "images": list(images),
            "padding": False,
            "return_tensors": "pt",
            "images_kwargs": {
                "size": {
                    "shortest_edge": 3136,
                    "longest_edge": 262144,
                }
            },
        }
    ]


def test_processor_batch_omits_image_kwargs_without_a_cap() -> None:
    processor = _ImageCapRecordingProcessor()

    _processor_batch(
        processor,
        text="native transcript",
        images=(object(),),
        image_max_pixels=None,
    )

    assert "images_kwargs" not in processor.calls[0]


@pytest.mark.parametrize("image_max_pixels", [True, 1.5, "262144", 0, -1, 3135])
def test_processor_batch_rejects_invalid_or_subminimum_image_caps(
    image_max_pixels: object,
) -> None:
    processor = _ImageCapRecordingProcessor(shortest_edge=3136)

    with pytest.raises(
        (TypeError, ValueError), match="image_max_pixels|pixel|shortest"
    ):
        _processor_batch(
            processor,
            text="native transcript",
            images=(object(),),
            image_max_pixels=image_max_pixels,  # type: ignore[arg-type]
        )
    assert processor.calls == []


def test_native_action_target_uses_strict_raw_tool_span(tmp_path: Path) -> None:
    runtime = _runtime(TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE)
    sample = _sample(tmp_path / "unused.png", 0)
    messages = build_native_representation_messages(sample, _prompt())
    target = render_native_action_target(runtime, messages)

    assert target.target_text == sample.target
    assert target.sampled_turn.sampled_text.startswith(
        NATIVE_REPRESENTATION_PRE_REASONING + "\n</think>"
    )
    assert target.sampled_turn.sampled_text.endswith("</tool_call>")
    assert (
        target.transcript.token_ids[
            target.canonical_target_span.start : target.canonical_target_span.end
        ]
        == target.canonical_target_token_ids
    )


def test_batched_action_render_preserves_scalar_order_spans_and_hashes(
    tmp_path: Path,
) -> None:
    runtime = _runtime(TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING)
    samples = (_sample(tmp_path / "unused.png", 0), _sample(tmp_path / "unused.png", 1))
    messages_batch = tuple(
        build_native_representation_messages(sample, _prompt()) for sample in samples
    )
    scalar = tuple(
        render_native_action_target(runtime, messages) for messages in messages_batch
    )

    batched = _render_native_action_targets_batch(runtime, messages_batch)

    assert batched == scalar
    assert tuple(item.target_text for item in batched) == tuple(
        sample.target for sample in samples
    )
    assert tuple(item.canonical_target_span for item in batched) == tuple(
        item.canonical_target_span for item in scalar
    )
    assert tuple(item.transcript.text_sha256 for item in batched) == tuple(
        item.transcript.text_sha256 for item in scalar
    )
    assert tuple(item.transcript.token_ids_sha256 for item in batched) == tuple(
        item.transcript.token_ids_sha256 for item in scalar
    )


def test_shared_visual_expansion_matches_processor_for_action_and_readout(
    tmp_path: Path,
) -> None:
    runtime = _runtime(TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING)
    runtime.processor.visual_tokens_per_image = 3
    image = b"one-processed-image"
    sample = _sample(tmp_path / "unused.png", 0)
    messages = build_native_representation_messages(sample, _prompt())
    action = render_native_action_target(runtime, messages)
    action_batch = _processor_batch(
        runtime.processor,
        text=action.transcript.text,
        images=(image,),
    )
    derived_action_ids, action_expansion = _expand_native_visual_placeholders(
        runtime,
        action.transcript.token_ids,
        visual_token_counts=(3,),
    )

    assert derived_action_ids == tuple(
        int(value) for value in action_batch["input_ids"][0]
    )
    assert _single_visual_expansion_count(action_expansion) == 3

    canonical_readout = render_native_evidence_labels(
        runtime.renderer,
        messages,
        evidence_description=sample.evidence_description,
    )
    readout_batch = _processor_batch(
        runtime.processor,
        text=canonical_readout.transcript.text,
        images=(image, image),
    )
    derived_readout_ids, readout_expansion = _expand_native_visual_placeholders(
        runtime,
        canonical_readout.transcript.token_ids,
        visual_token_counts=(3, 3),
    )

    assert derived_readout_ids == tuple(
        int(value) for value in readout_batch["input_ids"][0]
    )
    assert tuple(
        len(mapped)
        for canonical_id, mapped in zip(
            readout_expansion.canonical_token_ids,
            readout_expansion.canonical_to_model_positions,
            strict=True,
        )
        if canonical_id == runtime.tokenizer.convert_tokens_to_ids(_IMAGE_TOKEN)
    ) == (3, 3)


@pytest.mark.parametrize(
    "provider",
    [
        TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE,
        TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING,
    ],
)
def test_real_group_builder_contract_supports_both_providers(
    tmp_path: Path,
    provider: TargetConditioningProviderKind,
) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"immutable-image-fixture")
    runtime = _runtime(provider)
    family = Qwen3VLAdapter()
    samples = (_sample(image, 0), _sample(image, 1))
    builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=family,
        prompt=_prompt(),
        image_loader=lambda path: Path(path).read_bytes(),
    )

    group = builder(
        samples,
        runtime.adapter,
        collective_candidate_count=len(samples) + 1,
    )

    assert len(runtime.processor.calls) == 1
    assert runtime.processor.calls[0][1] == 1
    assert tuple(row.sample_id for row in group.rows) == (
        "sample-0",
        "sample-1",
    )
    assert all(len(row.source_positions) == 1 for row in group.rows)
    assert all(len(row.d_positions) == 1 for row in group.rows)
    assert all(
        candidate.target_conditioning_provider is provider
        for candidate in group.candidates
    )
    assert all(candidate.visual.main.requires_grad for candidate in group.candidates)
    assert all(candidate.attention is not None for candidate in group.candidates)
    assert all(
        candidate.attention is not None
        and not candidate.attention.main.requires_grad
        and len(candidate.attention.deepstack) == 3
        and all(
            not attention.requires_grad for attention in candidate.attention.deepstack
        )
        for candidate in group.candidates
    )
    assert group.collective_candidate_count == 3
    assert (
        group.loss_supervision_policy_identity
        == HISTORICAL_READOUT_LOSS_SUPERVISION_POLICY_IDENTITY
    )
    assert all(row.loss_supervision is None for row in group.rows)
    assert len(group.collective_padding) == 1
    assert all(
        tensor.requires_grad
        for tensor in (
            group.collective_padding[0].main,
            *group.collective_padding[0].deepstack,
        )
    )

    scores = score_streaming_same_image_group(family, runtime.model, group)
    assert scores.score_matrix.shape == (2, 2)


def test_explicit_sparse_readout_binding_materializes_real_native_rows(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"immutable-image-fixture")
    runtime = _runtime(TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING)
    family = Qwen3VLAdapter()
    samples = (_sample(image, 0), _sample(image, 1))
    factory = _SparseAnswerBearingReadoutFactory()
    binding = RepresentationReadoutLossSupervisionBinding(
        identity=factory.identity,
        factory=factory,
    )
    builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=family,
        prompt=_prompt(),
        image_loader=lambda path: Path(path).read_bytes(),
        readout_loss_supervision=binding,
    )

    group = builder(
        samples,
        runtime.adapter,
        collective_candidate_count=len(samples),
    )

    assert factory.sample_ids == [sample.sample_id for sample in samples]
    assert group.loss_supervision_policy_identity == factory.identity
    for row in group.rows:
        sparse = row.loss_supervision
        assert sparse is not None
        assert sparse.policy_identity == factory.identity
        assert sparse.identity == f"{factory.identity}:{row.sample_id}"
        assert len(sparse.evidence_value_token_positions) == len("OPEN")
        assert len(sparse.answer_token_positions) == len("OPEN")
        assert row.loss_labels == sparse.labels
        assert row.loss_supervised_token_positions == sparse.supervised_token_positions
        assert len(row.loss_supervised_token_positions) == 2 * len("OPEN")

    scores = score_streaming_same_image_group(family, runtime.model, group)
    assert torch.equal(scores.evidence_token_counts, torch.tensor([8, 8]))


def test_sparse_readout_binding_rejects_policy_identity_drift(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"immutable-image-fixture")
    runtime = _runtime(TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING)
    factory = _SparseAnswerBearingReadoutFactory()

    def mismatched_factory(
        sample: RepresentationTrainingSample,
        canonical: CanonicalEvidenceSupervision,
        model: ModelEvidenceSupervision,
    ) -> RepresentationReadoutLossSupervision:
        return replace(
            factory(sample, canonical, model),
            policy_identity="different-policy-v1",
        )

    builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=Qwen3VLAdapter(),
        prompt=_prompt(),
        image_loader=lambda path: Path(path).read_bytes(),
        readout_loss_supervision=RepresentationReadoutLossSupervisionBinding(
            identity=factory.identity,
            factory=mismatched_factory,
        ),
    )

    with pytest.raises(ValueError, match="configured policy identity"):
        builder(
            (_sample(image, 0), _sample(image, 1)),
            runtime.adapter,
            collective_candidate_count=2,
        )


def test_sparse_readout_requires_an_explicit_typed_binding() -> None:
    runtime = _runtime(TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING)
    with pytest.raises(TypeError, match="explicit typed binding"):
        Qwen3NativeRepresentationGroupBuilder(
            runtime=runtime,
            family_adapter=Qwen3VLAdapter(),
            prompt=_prompt(),
            image_loader=lambda _path: b"image",
            readout_loss_supervision=lambda *_args: None,  # type: ignore[arg-type]
        )


def test_contextual_builder_reuses_preencoded_vision_with_exact_output(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"immutable-image-fixture")
    runtime = _runtime(TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE)
    family = Qwen3VLAdapter()
    samples = (_sample(image, 0), _sample(image, 1))
    common = {
        "runtime": runtime,
        "family_adapter": family,
        "prompt": _prompt(),
        "image_loader": lambda path: Path(path).read_bytes(),
    }
    legacy = Qwen3NativeRepresentationGroupBuilder(**common)
    preencoded = Qwen3NativeRepresentationGroupBuilder(
        **common,
        reuse_preencoded_vision_for_contextual_conditioning=True,
    )

    before = runtime.vision_tower.forward_calls
    legacy_group = legacy(
        samples,
        runtime.adapter,
        collective_candidate_count=len(samples),
    )
    legacy_calls = runtime.vision_tower.forward_calls - before
    before = runtime.vision_tower.forward_calls
    preencoded_group = preencoded(
        samples,
        runtime.adapter,
        collective_candidate_count=len(samples),
    )
    preencoded_calls = runtime.vision_tower.forward_calls - before

    assert legacy_calls == 3
    assert preencoded_calls == 1
    for legacy_candidate, preencoded_candidate in zip(
        legacy_group.candidates,
        preencoded_group.candidates,
        strict=True,
    ):
        torch.testing.assert_close(
            legacy_candidate.visual.main,
            preencoded_candidate.visual.main,
            rtol=0,
            atol=0,
        )
        for legacy_branch, preencoded_branch in zip(
            legacy_candidate.visual.deepstack,
            preencoded_candidate.visual.deepstack,
            strict=True,
        ):
            torch.testing.assert_close(
                legacy_branch,
                preencoded_branch,
                rtol=0,
                atol=0,
            )


def test_group_builder_checks_runtime_invariants_only_at_group_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"immutable-image-fixture")
    runtime = _runtime(TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING)
    builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=Qwen3VLAdapter(),
        prompt=_prompt(),
        image_loader=lambda path: Path(path).read_bytes(),
    )
    original_assert = runtime.assert_bound_invariants
    boundary_checks = 0

    def counted_assert() -> None:
        nonlocal boundary_checks
        boundary_checks += 1
        original_assert()

    monkeypatch.setattr(runtime, "assert_bound_invariants", counted_assert)

    group = builder(
        (_sample(image, 0), _sample(image, 1)),
        runtime.adapter,
        collective_candidate_count=2,
    )

    assert len(group.candidates) == 2
    assert boundary_checks == 2


def test_group_builder_exit_check_rejects_mid_group_runtime_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"immutable-image-fixture")
    runtime = _runtime(TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING)
    original_assert = runtime.assert_bound_invariants
    boundary_checks = 0

    def counted_assert() -> None:
        nonlocal boundary_checks
        boundary_checks += 1
        original_assert()

    def mutating_image_loader(path: str) -> bytes:
        runtime.model.config._name_or_path = "/mutated-mid-group"
        return Path(path).read_bytes()

    monkeypatch.setattr(runtime, "assert_bound_invariants", counted_assert)
    builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=Qwen3VLAdapter(),
        prompt=_prompt(),
        image_loader=mutating_image_loader,
    )

    with pytest.raises(ValueError, match="model path differs"):
        builder(
            (_sample(image, 0), _sample(image, 1)),
            runtime.adapter,
            collective_candidate_count=2,
        )

    assert boundary_checks == 2


def test_group_builder_rejects_same_key_with_different_image_paths(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    runtime = _runtime(TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING)
    builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=Qwen3VLAdapter(),
        prompt=_prompt(),
        image_loader=lambda path: Path(path).read_bytes(),
    )
    samples = (_sample(first, 0), _sample(second, 1))

    with pytest.raises(ValueError, match="one exact source image"):
        builder(
            samples,
            runtime.adapter,
            collective_candidate_count=len(samples),
        )


def test_qwen3_d_only_counterfactual_executes_actual_cpu_forward(
    tmp_path: Path,
) -> None:
    image_a = tmp_path / "counterfactual-a.bin"
    image_b = tmp_path / "counterfactual-b.bin"
    image_a.write_bytes(b"counterfactual-image-a")
    image_b.write_bytes(b"counterfactual-image-b")
    runtime = _runtime(TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING)
    family = Qwen3VLAdapter()
    prompt = _prompt()

    def image_loader(path: str) -> bytes:
        return Path(path).read_bytes()

    group_builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=family,
        prompt=prompt,
        image_loader=image_loader,
    )
    primary_a = _counterfactual_sample(
        image_a,
        sample_id="counterfactual-a",
        image_id="counterfactual-image-a",
        target="status label",
        evidence="The status label reads OPEN.",
    )
    primary_b = _counterfactual_sample(
        image_b,
        sample_id="counterfactual-b",
        image_id="counterfactual-image-b",
        target="status label",
        evidence="The status label reads CLOSED.",
    )
    secondary_a = _counterfactual_sample(
        image_a,
        sample_id="counterfactual-a-secondary",
        image_id="counterfactual-image-a",
        target="serial label",
        evidence="The serial label reads A-01.",
    )
    secondary_b = _counterfactual_sample(
        image_b,
        sample_id="counterfactual-b-secondary",
        image_id="counterfactual-image-b",
        target="serial label",
        evidence="The serial label reads B-01.",
    )
    group_a = group_builder(
        (primary_a, secondary_a),
        runtime.adapter,
        collective_candidate_count=2,
    )
    group_b = group_builder(
        (primary_b, secondary_b),
        runtime.adapter,
        collective_candidate_count=2,
    )
    candidates = {
        primary_a.sample_id: group_a.candidates[0],
        primary_b.sample_id: group_b.candidates[0],
    }
    manifest_path = (
        Path(__file__).parents[2]
        / "fixtures"
        / "qwen3_counterfactual_smoke_manifest_v1.json"
    )
    manifest = load_qwen3_counterfactual_manifest(manifest_path)
    assert (
        manifest.content_sha256
        == load_qwen3_counterfactual_manifest(manifest_path).content_sha256
    )
    case_builder = Qwen3CounterfactualCaseBuilder(
        runtime=runtime,
        prompt=prompt,
    )
    with pytest.raises(ValueError, match="supplied dataset"):
        case_builder.build(
            manifest=manifest,
            data_manifest_sha256="0" * 64,
            samples=(primary_a, primary_b),
            observations=candidates,
        )
    built = case_builder.build(
        manifest=manifest,
        data_manifest_sha256=manifest.source_data_manifest_sha256,
        samples=(primary_a, primary_b),
        observations=candidates,
    )
    case = built.cases[0]
    context = case.context
    assert context.source_image_positions == ()
    assert context.image_grid_thw == ((1, 2, 2),)
    assert context.position_ids.shape == (3, 1, context.input_ids.shape[1])
    assert len(context.d_positions) == 1
    assert not case.observation_a.main.requires_grad
    value_ids = built.materializer.value_token_ids(context, "OPEN")
    teacher = built.materializer.teacher_forced(
        context=context,
        observation=case.observation_a,
        continuation_token_ids=value_ids,
    )
    assert torch.equal(
        teacher.request.position_ids[..., : context.input_ids.shape[1]],
        context.position_ids,
    )
    assert teacher.request.use_cache is False
    assert tuple(block.kind for block in teacher.request.visual_blocks) == (
        "focused_d",
    )
    assert len(teacher.request.visual_blocks[0].deepstack) == 3
    with pytest.raises(ValueError, match="identity/content drifted"):
        built.materializer.generation_step(
            context=replace(context, image_grid_thw=((1, 4, 2),)),
            observation=case.observation_a,
            generated_token_ids=(),
        )

    with torch.no_grad():
        runtime.model.lm_head.weight.zero_()
    vision_calls = runtime.vision_tower.forward_calls
    evaluator = create_injected_native_counterfactual_evaluator(
        model=runtime.model,
        family_adapter=family,
        materializer=built.materializer,
        eos_token_ids=(1023,),
        max_new_tokens=2,
    )
    causal = evaluator.causal_value_flip(NativeCausalValueFlipRequest(case=case))
    continuation = evaluator.free_continuation(
        NativeFreeContinuationRequest(
            case_id=case.case_id,
            variant="value_a",
            expected_value=case.expected_value_a,
            context=context,
            observation_identity=case.observation_a_identity,
            observation=case.observation_a,
        )
    )
    assert causal.case_id == case.case_id
    assert continuation.generated_token_ids == (0, 0)
    assert continuation.stop_reason == "length_cap"
    assert runtime.vision_tower.forward_calls == vision_calls
    assert len(runtime.tokenizer) == 1024
