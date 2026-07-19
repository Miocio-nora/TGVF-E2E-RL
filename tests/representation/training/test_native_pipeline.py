from __future__ import annotations

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
from tgvf_rl.representation.training.native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
    RepresentationPromptConfig,
    build_native_representation_messages,
    render_native_action_target,
)
from tgvf_rl.representation.training.runtime import (
    QWEN3_REPRESENTATION_BRANCH_LAYERS,
    create_qwen3_representation_runtime,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.streaming import (
    score_streaming_same_image_group,
)


_IMAGE_TOKEN = "<|image_pad|>"
_ASSISTANT_PREFILL = "<|im_start|>assistant\n<think>\n"
_EVIDENCE_SUFFIX = "\n</think>\n\n<|im_end|>\n"


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


class _Processor:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template

    @staticmethod
    def _user(messages) -> str:
        prompt = next(
            item["text"]
            for item in messages[0]["content"]
            if item["type"] == "text"
        )
        return f"<user>{_IMAGE_TOKEN}{prompt}</user>\n"

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
                + "\n</think>\n\n"
                + call
                + "<|im_end|>\n"
            )
        history = user + "<assistant>" + call + "</assistant>\n"
        history += f"<tool_response>{_IMAGE_TOKEN}</tool_response>\n"
        if len(messages) == 3:
            assert add_generation_prompt
            return history + _ASSISTANT_PREFILL
        assert len(messages) == 4 and not add_generation_prompt
        return (
            history
            + _ASSISTANT_PREFILL
            + messages[-1]["reasoning_content"]
            + _EVIDENCE_SUFFIX
        )

    def __call__(self, *, text, images, padding, return_tensors):
        assert len(text) == 1 and not padding and return_tensors == "pt"
        canonical_ids = self.tokenizer.encode(text[0], add_special_tokens=False)
        visual_id = self.tokenizer.convert_tokens_to_ids(_IMAGE_TOKEN)
        assert canonical_ids.count(visual_id) == len(images)
        expanded = []
        for token_id in canonical_ids:
            expanded.append(token_id)
        return {
            "input_ids": torch.tensor([expanded], dtype=torch.long),
            "attention_mask": torch.ones(1, len(expanded), dtype=torch.long),
            "pixel_values": torch.arange(
                len(images) * 4 * 3, dtype=torch.float32
            ).view(len(images) * 4, 3)
            % 12,
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
        hidden = inputs_embeds.clone()
        for branch in deepstack_visual_embeds:
            hidden = hidden.clone()
            hidden[visual_pos_masks] += branch
        hidden = hidden + hidden.cumsum(dim=1) * 0.01
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


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
        assert video_grid_thw is None and image_grid_thw.shape == (2, 3)
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
        hidden = resolve_language_model(self).get_input_embeddings()(input_ids)
        prefix = hidden.cumsum(dim=1) * 0.01
        return SimpleNamespace(hidden_states=(hidden, hidden + prefix))


def _sample(image: Path, index: int) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=f"sample-{index}",
        image=str(image),
        image_id="shared-image",
        question="What is written on the label?",
        target=f"label section {index}",
        evidence_description=f"Section {index} reads OPEN.",
    )


def _prompt(template: str = "Question: {question}\nInspect local evidence."):
    return RepresentationPromptConfig(
        identity="tiny-native-prompt-v1",
        template=template,
        expected_sha256=hashlib.sha256(template.encode()).hexdigest(),
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
            embedding_identity=(
                "/tiny-native-qwen3::language_model.input_embeddings"
            ),
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
    with pytest.raises(ValueError, match=r"\{question\}"):
        RepresentationPromptConfig(
            identity="bad",
            template=template,
            expected_sha256=hashlib.sha256(template.encode()).hexdigest(),
        )


def test_native_action_target_uses_strict_raw_tool_span(tmp_path: Path) -> None:
    runtime = _runtime(TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE)
    sample = _sample(tmp_path / "unused.png", 0)
    messages = build_native_representation_messages(sample, _prompt())
    target = render_native_action_target(runtime, messages)

    assert target.target_text == sample.target
    assert target.sampled_turn.sampled_text.startswith("\n</think>")
    assert target.sampled_turn.sampled_text.endswith("</tool_call>")
    assert target.transcript.token_ids[
        target.canonical_target_span.start : target.canonical_target_span.end
    ] == target.canonical_target_token_ids


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
    assert all(
        candidate.visual.main.requires_grad for candidate in group.candidates
    )
    assert group.collective_candidate_count == 3
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
