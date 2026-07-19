"""Pinned Qwen chat-template renderer for environment-owned transcript bytes."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tgvf_rl.tokenizer_invariants import effective_tokenizer_length

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
        actual_tokenizer_length = effective_tokenizer_length(tokenizer)
        if actual_tokenizer_length != expected_tokenizer_length:
            raise ValueError(
                "tokenizer length mismatch: "
                f"expected={expected_tokenizer_length} actual={actual_tokenizer_length}"
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
        return self._build_rendered_transcript(text, token_ids)

    def render_many(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, Any]]],
        *,
        add_generation_prompt: bool,
    ) -> tuple[RenderedTranscript, ...]:
        """Render independent conversations with exact scalar-render semantics.

        Processors and tokenizers with native batch support are invoked once per
        batch.  Implementations that reject or return a non-batch-shaped result
        fall back to their scalar APIs without changing transcript bytes.
        """

        conversations = tuple(tuple(messages) for messages in messages_batch)
        if not conversations:
            self.assert_tokenizer_length()
            self.assert_chat_template_identity()
            return ()

        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        texts = self._render_texts_many(
            conversations,
            add_generation_prompt=add_generation_prompt,
        )
        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        token_ids_batch = self._encode_texts_many(texts)
        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        return tuple(
            self._build_rendered_transcript(text, token_ids)
            for text, token_ids in zip(texts, token_ids_batch, strict=True)
        )

    def _render_texts_many(
        self,
        conversations: tuple[tuple[Mapping[str, Any], ...], ...],
        *,
        add_generation_prompt: bool,
    ) -> tuple[str, ...]:
        try:
            batch_result = self.processor.apply_chat_template(
                [list(messages) for messages in conversations],
                tools=[build_tgvf_focus_tool_schema()],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            self.assert_tokenizer_length()
            self.assert_chat_template_identity()
        else:
            self.assert_tokenizer_length()
            self.assert_chat_template_identity()
            texts = self._coerce_text_batch(batch_result, len(conversations))
            if texts is not None:
                return texts

        texts = []
        for messages in conversations:
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
            texts.append(text)
        return tuple(texts)

    def _encode_texts_many(self, texts: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
        try:
            batch_result = self.tokenizer(
                list(texts),
                add_special_tokens=False,
            )
        except Exception:
            self.assert_tokenizer_length()
            self.assert_chat_template_identity()
        else:
            self.assert_tokenizer_length()
            self.assert_chat_template_identity()
            token_ids_batch = self._coerce_token_id_batch(batch_result, len(texts))
            if token_ids_batch is not None:
                return token_ids_batch

        token_ids_batch = []
        for text in texts:
            token_ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
            self.assert_tokenizer_length()
            self.assert_chat_template_identity()
            token_ids_batch.append(token_ids)
        return tuple(token_ids_batch)

    @staticmethod
    def _coerce_text_batch(value: Any, expected_size: int) -> tuple[str, ...] | None:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return None
        texts = tuple(value)
        if len(texts) != expected_size or not all(
            isinstance(text, str) for text in texts
        ):
            return None
        return texts

    @staticmethod
    def _coerce_token_id_batch(
        value: Any, expected_size: int
    ) -> tuple[tuple[int, ...], ...] | None:
        if isinstance(value, Mapping):
            value = value.get("input_ids")
        else:
            value = getattr(value, "input_ids", None)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return None
        rows = tuple(value)
        if len(rows) != expected_size:
            return None
        token_ids_batch: list[tuple[int, ...]] = []
        for row in rows:
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                return None
            token_ids = tuple(row)
            try:
                for token_id in token_ids:
                    struct.pack("<I", token_id)
            except (struct.error, TypeError):
                return None
            token_ids_batch.append(token_ids)
        return tuple(token_ids_batch)

    def _build_rendered_transcript(
        self, text: str, token_ids: tuple[int, ...]
    ) -> RenderedTranscript:
        raw_ids = b"".join(struct.pack("<I", token_id) for token_id in token_ids)
        return RenderedTranscript(
            text=text,
            token_ids=token_ids,
            token_ids_sha256=hashlib.sha256(raw_ids).hexdigest(),
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            chat_template_sha256=self.chat_template_sha256,
            tool_schema_sha256=TGVF_FOCUS_TOOL_SCHEMA_SHA256,
            tokenizer_length=effective_tokenizer_length(self.tokenizer),
        )

    def assert_tokenizer_length(self) -> None:
        """Fail if any processor/tokenizer operation changed vocabulary size."""

        actual = effective_tokenizer_length(self.tokenizer)
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
