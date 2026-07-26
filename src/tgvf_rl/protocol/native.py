"""Pinned Qwen chat-template renderer for environment-owned transcript bytes."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from tgvf_rl.tokenizer_invariants import effective_tokenizer_length

from .schema import (
    TGVF_FOCUS_TOOL_NAME,
    build_native_tool_schemas,
    native_tool_schemas_sha256,
)


@dataclass(frozen=True, slots=True)
class RenderedTranscript:
    text: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    text_sha256: str
    chat_template_sha256: str
    tool_schema_sha256: str
    tokenizer_length: int


class NativeAssistantDialect(str, Enum):
    """Checkpoint-bound assistant serialization owned by Qwen's template."""

    QWEN3_VL_THINKING = "qwen3-vl-thinking-v1"
    QWEN3_VL_INSTRUCT = "qwen3-vl-instruct-v1"

    @property
    def generation_prefill_text(self) -> str:
        if self is NativeAssistantDialect.QWEN3_VL_THINKING:
            return "<|im_start|>assistant\n<think>\n"
        return "<|im_start|>assistant\n"

    @property
    def template_owns_think_opener(self) -> bool:
        return self is NativeAssistantDialect.QWEN3_VL_THINKING


def native_assistant_dialect_for_model(model_name: str) -> NativeAssistantDialect:
    """Resolve an exact supported Qwen3 edition without text-based inference."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be non-empty")
    basename = Path(model_name).name
    if basename == "Qwen3-VL-8B-Thinking":
        return NativeAssistantDialect.QWEN3_VL_THINKING
    if basename == "Qwen3-VL-8B-Instruct":
        return NativeAssistantDialect.QWEN3_VL_INSTRUCT
    if basename == "fixture" or basename.startswith("tiny-"):
        # Existing synthetic fixtures encode the historical Thinking template.
        return NativeAssistantDialect.QWEN3_VL_THINKING
    raise ValueError(f"unsupported native assistant model edition: {model_name!r}")


class NativeProtocolRenderer:
    """Uses the model's own template and forbids tokenizer growth."""

    def __init__(
        self,
        processor: Any,
        *,
        expected_tokenizer_length: int,
        tool_names: tuple[str, ...] = (TGVF_FOCUS_TOOL_NAME,),
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
        assistant_dialect: NativeAssistantDialect = (
            NativeAssistantDialect.QWEN3_VL_THINKING
        ),
    ) -> None:
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
        if not isinstance(assistant_dialect, NativeAssistantDialect):
            raise TypeError("assistant_dialect must be NativeAssistantDialect")
        template = getattr(processor, "chat_template", None) or getattr(
            tokenizer, "chat_template", None
        )
        if not isinstance(template, str) or not template:
            raise ValueError("processor chat template must be explicit")
        self.processor = processor
        self.tokenizer = tokenizer
        self.expected_tokenizer_length = expected_tokenizer_length
        self.assistant_dialect = assistant_dialect
        self.tool_names = tuple(tool_names)
        supplied_schemas = (
            tuple(build_native_tool_schemas(self.tool_names))
            if tool_schemas is None
            else tuple(tool_schemas)
        )
        if not supplied_schemas:
            raise ValueError("tool_schemas must be non-empty")
        schemas = tuple(_copy_json_mapping(schema) for schema in supplied_schemas)
        schema_names: list[str] = []
        for schema in schemas:
            if not isinstance(schema, Mapping):
                raise TypeError("each tool schema must be a mapping")
            function = schema.get("function")
            if not isinstance(function, Mapping):
                raise ValueError("each tool schema must contain a function mapping")
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("each tool schema must contain a function name")
            schema_names.append(name)
        if tuple(schema_names) != self.tool_names:
            raise ValueError("tool schema names differ from ordered tool_names")
        self.tool_schemas = schemas
        self.tool_schema_sha256 = native_tool_schemas_sha256(schemas)
        self.chat_template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> RenderedTranscript:
        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        self.assert_tool_schema_identity()
        text = self.processor.apply_chat_template(
            list(messages),
            tools=self._fresh_tool_schemas(),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        self.assert_tool_schema_identity()
        if not isinstance(text, str):
            raise TypeError("chat template did not return text")
        token_ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        self.assert_tool_schema_identity()
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
            self.assert_tool_schema_identity()
            return ()

        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        self.assert_tool_schema_identity()
        texts = self._render_texts_many(
            conversations,
            add_generation_prompt=add_generation_prompt,
        )
        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        self.assert_tool_schema_identity()
        token_ids_batch = self._encode_texts_many(texts)
        self.assert_tokenizer_length()
        self.assert_chat_template_identity()
        self.assert_tool_schema_identity()
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
                tools=self._fresh_tool_schemas(),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            self.assert_tokenizer_length()
            self.assert_chat_template_identity()
            self.assert_tool_schema_identity()
        else:
            self.assert_tokenizer_length()
            self.assert_chat_template_identity()
            self.assert_tool_schema_identity()
            texts = self._coerce_text_batch(batch_result, len(conversations))
            if texts is not None:
                return texts

        texts = []
        for messages in conversations:
            text = self.processor.apply_chat_template(
                list(messages),
                tools=self._fresh_tool_schemas(),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            self.assert_tokenizer_length()
            self.assert_chat_template_identity()
            self.assert_tool_schema_identity()
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
            tool_schema_sha256=self.tool_schema_sha256,
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

    def assert_tool_schema_identity(self) -> None:
        """Fail if the exact registered schemas changed after pinning."""

        if native_tool_schemas_sha256(self.tool_schemas) != self.tool_schema_sha256:
            raise ValueError("tool schemas changed after renderer construction")

    def _fresh_tool_schemas(self) -> list[dict[str, Any]]:
        self.assert_tool_schema_identity()
        return [_copy_json_mapping(schema) for schema in self.tool_schemas]

    def assert_generation_prefill(
        self, transcript: RenderedTranscript, tokenizer: Any
    ) -> None:
        expected_text = self.assistant_dialect.generation_prefill_text
        if not transcript.text.endswith(expected_text):
            raise ValueError(
                "native Qwen generation prefill text differs from the selected "
                "assistant-dialect contract"
            )
        final_assistant = transcript.text.rsplit("<|im_start|>assistant\n", maxsplit=1)[
            -1
        ]
        expected_assistant_body = (
            "<think>\n"
            if self.assistant_dialect.template_owns_think_opener
            else ""
        )
        if final_assistant != expected_assistant_body:
            raise ValueError(
                "native Qwen prefill has a duplicate or extra assistant opener"
            )
        expected = tuple(tokenizer.encode(expected_text, add_special_tokens=False))
        if not expected or transcript.token_ids[-len(expected) :] != expected:
            raise ValueError(
                "native Qwen generation prefill differs from the selected contract"
            )


def _copy_json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one schema without sharing nested mutable values with a processor."""

    if not isinstance(value, Mapping):
        raise TypeError("each tool schema must be a mapping")

    def copy_value(item: Any) -> Any:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("tool schema object keys must be strings")
            return {key: copy_value(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [copy_value(child) for child in item]
        if item is None or type(item) in {str, int, float, bool}:
            return item
        raise TypeError("tool schemas must contain only JSON-compatible values")

    return copy_value(value)
