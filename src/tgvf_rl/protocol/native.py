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
        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        text = self.processor.apply_chat_template(
            list(messages),
            tools=[build_tgvf_focus_tool_schema()],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        if not isinstance(text, str):
            raise TypeError("chat template did not return text")
        token_ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
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

    def assert_tokenizer_length(self) -> None:
        """Fail if any processor/tokenizer operation changed vocabulary size."""

        actual = len(self.tokenizer)
        if actual != self.expected_tokenizer_length:
            raise ValueError(
                "tokenizer length changed after renderer construction: "
                f"expected={self.expected_tokenizer_length} actual={actual}"
            )

    def assert_chat_template_identity(self) -> None:
        """Fail if the processor's effective template changed after pinning."""

        template = getattr(self.processor, "chat_template", None) or getattr(
            self.tokenizer, "chat_template", None
        )
        if not isinstance(template, str) or not template:
            raise ValueError("processor chat template is no longer explicit")
        actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if actual != self.chat_template_sha256:
            raise ValueError(
                "processor chat template changed after renderer construction"
            )

    @staticmethod
    def assert_generation_prefill(
        transcript: RenderedTranscript, tokenizer: Any
    ) -> None:
        expected_text = "<|im_start|>assistant\n<think>\n"
        if not transcript.text.endswith(expected_text):
            raise ValueError(
                "native Qwen Thinking generation prefill text differs from the accepted contract"
            )
        final_assistant = transcript.text.rsplit("<|im_start|>assistant\n", maxsplit=1)[
            -1
        ]
        if final_assistant != "<think>\n":
            raise ValueError(
                "native Qwen Thinking prefill has a duplicate or extra assistant opener"
            )
        expected = tuple(tokenizer.encode(expected_text, add_special_tokens=False))
        if not expected or transcript.token_ids[-len(expected) :] != expected:
            raise ValueError(
                "native Qwen Thinking generation prefill differs from the accepted contract"
            )
