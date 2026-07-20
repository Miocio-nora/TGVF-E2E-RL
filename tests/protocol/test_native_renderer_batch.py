from __future__ import annotations

import json
from typing import Any

import pytest

from tgvf_rl.protocol.native import NativeProtocolRenderer
from tgvf_rl.protocol.schema import POLICY_RL_TOOL_NAMES, native_tool_set_sha256


class _BatchTokenizer:
    chat_template = "synthetic-native-template-v1"

    def __init__(self, *, supports_batch: bool = True) -> None:
        self.supports_batch = supports_batch
        self.batch_calls = 0
        self.scalar_calls = 0

    def __len__(self) -> int:
        return 4096

    @staticmethod
    def _ids(text: str) -> list[int]:
        return [byte + 1 for byte in text.encode("utf-8")]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.scalar_calls += 1
        return self._ids(text)

    def __call__(
        self, texts: list[str], *, add_special_tokens: bool
    ) -> dict[str, list[list[int]]]:
        assert add_special_tokens is False
        self.batch_calls += 1
        if not self.supports_batch:
            raise TypeError("batch tokenization is unavailable")
        return {"input_ids": [self._ids(text) for text in texts]}


class _BatchProcessor:
    chat_template = _BatchTokenizer.chat_template

    def __init__(
        self, tokenizer: _BatchTokenizer, *, supports_batch: bool = True
    ) -> None:
        self.tokenizer = tokenizer
        self.supports_batch = supports_batch
        self.batch_calls = 0
        self.scalar_calls = 0

    @staticmethod
    def _render(messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
        payload = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        suffix = "<assistant-prefill>" if add_generation_prompt else "<complete>"
        return payload + suffix

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        assert tools[0]["function"]["name"] == "tgvf_focus_tool"
        assert tokenize is False
        if messages and isinstance(messages[0], list):
            self.batch_calls += 1
            if not self.supports_batch:
                raise TypeError("batch rendering is unavailable")
            return [
                self._render(conversation, add_generation_prompt)
                for conversation in messages
            ]
        self.scalar_calls += 1
        return self._render(messages, add_generation_prompt)


def _messages() -> tuple[list[dict[str, Any]], ...]:
    return (
        [{"role": "user", "content": "first target"}],
        [
            {"role": "user", "content": "second target"},
            {"role": "assistant", "content": "blue"},
        ],
        [{"role": "user", "content": "日本語の target"}],
    )


@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_render_many_is_byte_and_hash_identical_to_scalar_rendering(
    add_generation_prompt: bool,
) -> None:
    batch_tokenizer = _BatchTokenizer()
    batch_processor = _BatchProcessor(batch_tokenizer)
    batch_renderer = NativeProtocolRenderer(
        batch_processor, expected_tokenizer_length=4096
    )

    scalar_tokenizer = _BatchTokenizer()
    scalar_processor = _BatchProcessor(scalar_tokenizer)
    scalar_renderer = NativeProtocolRenderer(
        scalar_processor, expected_tokenizer_length=4096
    )
    expected = tuple(
        scalar_renderer.render(
            messages,
            add_generation_prompt=add_generation_prompt,
        )
        for messages in _messages()
    )

    actual = batch_renderer.render_many(
        _messages(),
        add_generation_prompt=add_generation_prompt,
    )

    assert actual == expected
    assert batch_processor.batch_calls == 1
    assert batch_processor.scalar_calls == 0
    assert batch_tokenizer.batch_calls == 1
    assert batch_tokenizer.scalar_calls == 0


def test_render_many_falls_back_to_exact_scalar_apis() -> None:
    tokenizer = _BatchTokenizer(supports_batch=False)
    processor = _BatchProcessor(tokenizer, supports_batch=False)
    renderer = NativeProtocolRenderer(processor, expected_tokenizer_length=4096)

    actual = renderer.render_many(_messages(), add_generation_prompt=False)

    verifier_tokenizer = _BatchTokenizer()
    verifier = NativeProtocolRenderer(
        _BatchProcessor(verifier_tokenizer), expected_tokenizer_length=4096
    )
    expected = tuple(
        verifier.render(messages, add_generation_prompt=False)
        for messages in _messages()
    )
    assert actual == expected
    assert processor.batch_calls == 1
    assert processor.scalar_calls == len(_messages())
    assert tokenizer.batch_calls == 1
    assert tokenizer.scalar_calls == len(_messages())


def test_render_many_empty_batch_is_a_no_op() -> None:
    tokenizer = _BatchTokenizer()
    processor = _BatchProcessor(tokenizer)
    renderer = NativeProtocolRenderer(processor, expected_tokenizer_length=4096)

    assert renderer.render_many((), add_generation_prompt=False) == ()
    assert processor.batch_calls == 0
    assert processor.scalar_calls == 0
    assert tokenizer.batch_calls == 0
    assert tokenizer.scalar_calls == 0


def test_policy_rl_renderer_supplies_both_tools_in_declared_order() -> None:
    class CapturingProcessor(_BatchProcessor):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self.tool_names = ()

        def apply_chat_template(
            self, messages, *, tools, tokenize, add_generation_prompt
        ):
            self.tool_names = tuple(item["function"]["name"] for item in tools)
            return super().apply_chat_template(
                messages,
                tools=tools,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )

    tokenizer = _BatchTokenizer()
    processor = CapturingProcessor(tokenizer)
    renderer = NativeProtocolRenderer(
        processor,
        expected_tokenizer_length=4096,
        tool_names=POLICY_RL_TOOL_NAMES,
    )
    result = renderer.render(_messages()[0], add_generation_prompt=False)
    assert processor.tool_names == ("tgvf_focus_tool", "image_zoom_in_tool")
    assert result.tool_schema_sha256 == native_tool_set_sha256(POLICY_RL_TOOL_NAMES)
