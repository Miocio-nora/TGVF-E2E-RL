"""Pinned Qwen chat-template renderer for environment-owned transcript bytes."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schema import TGVF_FOCUS_TOOL_SCHEMA_SHA256, build_tgvf_focus_tool_schema


@dataclass(frozen=True, slots=True)
class RenderedTranscript:
    text: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    text_sha256: str
    chat_template_sha256: str
    tool_schema_sha256: str
    tokenizer_length: int


class NativeProtocolRenderer:
    """Uses the model's own template and forbids tokenizer growth."""

    def __init__(self, processor: Any, *, expected_tokenizer_length: int) -> None:
        tokenizer = getattr(processor, "tokenizer", processor)
        if not hasattr(processor, "apply_chat_template"):
            raise TypeError("processor must expose apply_chat_template")
        if not hasattr(tokenizer, "encode"):
            raise TypeError("processor tokenizer must expose encode")
        if len(tokenizer) != expected_tokenizer_length:
            raise ValueError(
                f"tokenizer length mismatch: expected={expected_tokenizer_length} actual={len(tokenizer)}"
            )
        template = getattr(processor, "chat_template", None) or getattr(
            tokenizer, "chat_template", None
        )
        if not isinstance(template, str) or not template:
            raise ValueError("processor chat template must be explicit")
        self.processor = processor
        self.tokenizer = tokenizer
        self.expected_tokenizer_length = expected_tokenizer_length
        self.chat_template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> RenderedTranscript:
        if len(self.tokenizer) != self.expected_tokenizer_length:
            raise ValueError("tokenizer grew after renderer construction")
        text = self.processor.apply_chat_template(
            list(messages),
            tools=[build_tgvf_focus_tool_schema()],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(text, str):
            raise TypeError("chat template did not return text")
        token_ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        raw_ids = b"".join(struct.pack("<I", token_id) for token_id in token_ids)
        return RenderedTranscript(
            text=text,
            token_ids=token_ids,
            token_ids_sha256=hashlib.sha256(raw_ids).hexdigest(),
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            chat_template_sha256=self.chat_template_sha256,
            tool_schema_sha256=TGVF_FOCUS_TOOL_SCHEMA_SHA256,
            tokenizer_length=len(self.tokenizer),
        )

    @staticmethod
    def assert_generation_prefill(
        transcript: RenderedTranscript, tokenizer: Any
    ) -> None:
        expected = tuple(
            tokenizer.encode(
                "<|im_start|>assistant\n<think>\n", add_special_tokens=False
            )
        )
        if not expected or transcript.token_ids[-len(expected) :] != expected:
            raise ValueError(
                "native Qwen Thinking generation prefill differs from the accepted contract"
            )
