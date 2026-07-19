from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.qwen.qwen25_vl import Qwen25VLAdapter
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.transcript import (
    _build_visual_token_expansion,
    render_native_evidence_labels,
)
from tgvf_rl.protocol.native import NativeProtocolRenderer


_ASSISTANT_PREFILL = "<|im_start|>assistant\n<think>\n"
_COMPLETION_SUFFIX = "\n</think>\n\n<|im_end|>\n"


class _OffsetTokenizer:
    is_fast = True
    chat_template = "synthetic-native-qwen-thinking-template-v1"

    def __init__(self, *, offset_fault: str | None = None) -> None:
        self.offset_fault = offset_fault
        self._pieces_to_ids: dict[str, int] = {"<|image_pad|>": 7}
        self._next_id = 8

    def __len__(self) -> int:
        return 4096

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_id(token)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return next(
            piece
            for piece, candidate_id in self._pieces_to_ids.items()
            if candidate_id == token_id
        )

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        pieces, _ = self._tokenize(text)
        return [self._token_id(piece) for piece in pieces]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> dict[str, object]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        assert truncation is False
        pieces, offsets = self._tokenize(text)
        evidence = "The sign reads OPEN."
        evidence_start = text.rfind(evidence)
        if self.offset_fault == "prefix_cross":
            index = next(
                index
                for index, (start, _) in enumerate(offsets)
                if start == evidence_start
            )
            _, end = offsets[index]
            offsets[index] = (evidence_start - 1, end)
        elif self.offset_fault == "non_whitespace_suffix":
            evidence_end = evidence_start + len(evidence)
            index = next(
                index
                for index, (start, _) in enumerate(offsets)
                if start == evidence_end - 1
            )
            offsets[index] = (evidence_end - 1, evidence_end + 2)
        input_ids = [self._token_id(piece) for piece in pieces]
        if self.offset_fault == "id_mismatch":
            input_ids[-1] += 1
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def _token_id(self, piece: str) -> int:
        if piece not in self._pieces_to_ids:
            self._pieces_to_ids[piece] = self._next_id
            self._next_id += 1
        return self._pieces_to_ids[piece]

    @staticmethod
    def _tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
        pieces: list[str] = []
        offsets: list[tuple[int, int]] = []
        position = 0
        while position < len(text):
            if text.startswith("<|image_pad|>", position):
                end = position + len("<|image_pad|>")
            elif text.startswith(".\n", position):
                end = position + 2
            else:
                end = position + 1
            pieces.append(text[position:end])
            offsets.append((position, end))
            position = end
        return pieces, offsets


class _SlowTokenizer(_OffsetTokenizer):
    is_fast = False


class _Processor:
    chat_template = _OffsetTokenizer.chat_template

    def __init__(self, tokenizer: _OffsetTokenizer) -> None:
        self.tokenizer = tokenizer

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tools[0]["function"]["name"] == "tgvf_focus_tool"
        assert tokenize is False
        history = (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "The sign reads OPEN.<|im_end|>\n"
            "<|im_start|>assistant\n<tool_call>\ncall\n</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n<|vision_start|>"
            "<|image_pad|><|vision_end|>\n</tool_response><|im_end|>\n"
        )
        prefill = history + _ASSISTANT_PREFILL
        if add_generation_prompt:
            return prefill
        evidence = messages[-1]["reasoning_content"].strip("\n")
        return prefill + evidence + _COMPLETION_SUFFIX


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(4096, 4)

    def get_input_embeddings(self):
        return self.embed_tokens


class _TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(language_model=_TinyLanguageModel())
        self.lm_head = nn.Linear(4, 4096, bias=False)


def _messages(*, evidence: str = "The sign reads OPEN.", content: str = ""):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "The sign reads OPEN."},
            ],
        },
        {
            "role": "assistant",
            "reasoning_content": "",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "tgvf_focus_tool",
                        "arguments": {"target": "the sign"},
                    },
                }
            ],
        },
        {"role": "tool", "content": [{"type": "image"}]},
        {
            "role": "assistant",
            "reasoning_content": evidence,
            "content": content,
        },
    ]


def _canonical(tokenizer: _OffsetTokenizer | None = None):
    tokenizer = tokenizer or _OffsetTokenizer()
    renderer = NativeProtocolRenderer(
        _Processor(tokenizer), expected_tokenizer_length=4096
    )
    return tokenizer, render_native_evidence_labels(
        renderer,
        _messages(),
        evidence_description="The sign reads OPEN.",
    )


def test_only_final_post_tool_evidence_tokens_receive_canonical_labels() -> None:
    tokenizer, supervision = _canonical()

    assert supervision.transcript.text == (
        supervision.generation_prefill.text
        + supervision.evidence_text
        + _COMPLETION_SUFFIX
    )
    assert supervision.transcript.text.count(supervision.evidence_text) == 2
    assert supervision.evidence_char_start == len(supervision.generation_prefill.text)
    assert supervision.evidence_byte_end - supervision.evidence_byte_start == len(
        supervision.evidence_text.encode("utf-8")
    )
    owned = set(supervision.evidence_token_positions)
    assert owned
    for position, (label, token_id) in enumerate(
        zip(
            supervision.canonical_labels,
            supervision.transcript.token_ids,
            strict=True,
        )
    ):
        assert label == (token_id if position in owned else EVIDENCE_IGNORE_INDEX)

    _, offsets = tokenizer._tokenize(supervision.transcript.text)
    final_start, final_end = offsets[supervision.evidence_token_positions[-1]]
    assert supervision.transcript.text[final_start:final_end] == ".\n"
    assert final_start < supervision.evidence_char_end < final_end


def test_visual_expansion_maps_canonical_labels_to_model_positions() -> None:
    tokenizer, canonical = _canonical()
    visual_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    model_ids: list[int] = []
    visual_occurrence = 0
    for token_id in canonical.transcript.token_ids:
        if token_id == visual_id:
            visual_occurrence += 1
            model_ids.extend([token_id] * (visual_occurrence + 1))
        else:
            model_ids.append(token_id)

    result = Qwen3VLAdapter().materialize_representation_supervision(
        _TinyQwen(),
        tokenizer,
        canonical,
        torch.tensor([model_ids], dtype=torch.long),
    )

    assert len(result.visual_model_positions) == 5
    assert tuple(map(len, result.visual_expansion_blocks)) == (2, 3)
    assert not set(result.visual_model_positions).intersection(
        result.evidence_token_positions
    )
    assert result.evidence_token_positions[0] > result.visual_model_positions[-1]
    for position in result.visual_model_positions:
        assert result.labels[position] == EVIDENCE_IGNORE_INDEX
    for position in result.evidence_token_positions:
        assert result.labels[position] == result.model_token_ids[position]


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("prefix_cross", "crosses into evidence"),
        ("non_whitespace_suffix", "non-whitespace template"),
        ("id_mismatch", "differs from rendered transcript"),
    ],
)
def test_ambiguous_or_mismatched_offset_mapping_fails_closed(
    fault: str, match: str
) -> None:
    tokenizer = _OffsetTokenizer(offset_fault=fault)
    renderer = NativeProtocolRenderer(
        _Processor(tokenizer), expected_tokenizer_length=4096
    )

    with pytest.raises(ValueError, match=match):
        render_native_evidence_labels(
            renderer,
            _messages(),
            evidence_description="The sign reads OPEN.",
        )


def test_slow_tokenizer_and_nonempty_answer_content_fail_closed() -> None:
    slow = _SlowTokenizer()
    with pytest.raises(TypeError, match="fast tokenizer"):
        render_native_evidence_labels(
            NativeProtocolRenderer(_Processor(slow), expected_tokenizer_length=4096),
            _messages(),
            evidence_description="The sign reads OPEN.",
        )

    tokenizer = _OffsetTokenizer()
    fabricated = _messages()
    fabricated[-3]["reasoning_content"] = "I will inspect the target."
    with pytest.raises(ValueError, match="cannot fabricate reasoning"):
        render_native_evidence_labels(
            NativeProtocolRenderer(
                _Processor(tokenizer), expected_tokenizer_length=4096
            ),
            fabricated,
            evidence_description="The sign reads OPEN.",
        )

    with pytest.raises(ValueError, match="empty answer content"):
        render_native_evidence_labels(
            NativeProtocolRenderer(
                _Processor(tokenizer), expected_tokenizer_length=4096
            ),
            _messages(content="The answer is OPEN."),
            evidence_description="The sign reads OPEN.",
        )


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        ("\nThe sign reads OPEN.", "leading/trailing newlines"),
        ("The sign says <|im_end|>.", "native control tags"),
    ],
)
def test_evidence_boundary_or_native_control_injection_fails_closed(
    evidence: str, match: str
) -> None:
    tokenizer = _OffsetTokenizer()
    with pytest.raises(ValueError, match=match):
        render_native_evidence_labels(
            NativeProtocolRenderer(
                _Processor(tokenizer), expected_tokenizer_length=4096
            ),
            _messages(evidence=evidence),
            evidence_description=evidence,
        )


def test_visual_expansion_and_qwen25_family_slot_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-visual model token"):
        _build_visual_token_expansion(
            family="qwen3_vl",
            canonical_token_ids=(1, 7, 2),
            model_token_ids=(1, 7, 7, 3),
            visual_placeholder_token_id=7,
        )
    with pytest.raises(ValueError, match="no accepted representation"):
        _build_visual_token_expansion(
            family="qwen2_5_vl",
            canonical_token_ids=(1, 7, 2),
            model_token_ids=(1, 7, 7, 2),
            visual_placeholder_token_id=7,
        )

    _, canonical = _canonical()
    with pytest.raises(NotImplementedError, match="family-specific native transcript"):
        Qwen25VLAdapter().materialize_representation_supervision(
            _TinyQwen(),
            object(),
            canonical,
            torch.tensor([[1]], dtype=torch.long),
        )


def test_renderer_rejects_effective_chat_template_mutation() -> None:
    tokenizer = _OffsetTokenizer()
    processor = _Processor(tokenizer)
    renderer = NativeProtocolRenderer(processor, expected_tokenizer_length=4096)
    processor.chat_template = "mutated-template"

    with pytest.raises(ValueError, match="chat template changed"):
        renderer.render(_messages(), add_generation_prompt=False)
