from __future__ import annotations

from hashlib import sha256

import pytest

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import SampledPolicyTurn
from tgvf_rl.environment.native_appender import (
    QWEN_NATIVE_IMAGE_PLACEHOLDER,
    QwenNativeToolObservationAppender,
)
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol import NativeProtocolRenderer, TokenByteSpan
from tgvf_rl.protocol.schema import StandardToolError


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


class _Registrar:
    def __init__(self) -> None:
        self.calls = []

    def register_tool_turn(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _sampled(text: str, token_ids: tuple[int, ...]) -> SampledPolicyTurn:
    policy = PolicyVersion("native-appender-test", 0, "1" * 64)
    sampling = SamplingIdentity(
        policy_version=policy,
        backend="vllm",
        backend_version="0.12.0",
        seed=42,
        rng_state_sha256="2" * 64,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    spans = tuple(
        TokenByteSpan(index, token_id, 0, 0) for index, token_id in enumerate(token_ids)
    )
    return SampledPolicyTurn(
        text=text,
        token_ids=token_ids,
        token_byte_spans=spans,
        behavior_logprobs=tuple(-1.0 for _ in token_ids),
        sampling=sampling,
        think_token_span=TokenSpan(0, 1),
        stop_reason="tool_call_stop",
        backend_request_sha256="3" * 64,
        backend_response_sha256="4" * 64,
    )


def test_success_and_error_append_only_environment_owned_native_suffix() -> None:
    tokenizer = _CharacterTokenizer()
    registrar = _Registrar()
    appender = QwenNativeToolObservationAppender(
        tokenizer=tokenizer, registrar=registrar
    )
    sampled_text = "inspect\n</think>\n<tool_call>{}</tool_call>"
    sampled = _sampled(
        sampled_text, tuple(tokenizer.encode(sampled_text, add_special_tokens=False))
    )
    prompt = (7, 8)

    updated, suffix = appender.append(
        prompt,
        sampled,
        ObservationHandle("obs-0", "5" * 64),
        call_index=0,
    )
    assert updated == prompt + sampled.token_ids + suffix
    assert QWEN_NATIVE_IMAGE_PLACEHOLDER in "".join(map(chr, suffix))
    assert registrar.calls[-1]["updated_prompt_token_ids"] == updated

    error = StandardToolError(
        code="tool_execution_failed",
        message="failed",
        attempt_index=1,
        recoverable=True,
        maximum_tool_calls=4,
    )
    _, error_suffix = appender.append(prompt, sampled, error, call_index=1)
    rendered_error = "".join(map(chr, error_suffix))
    assert error.canonical_json in rendered_error
    assert QWEN_NATIVE_IMAGE_PLACEHOLDER not in rendered_error


def test_qwen3_appended_tokens_equal_native_chat_template() -> None:
    transformers = pytest.importorskip("transformers")
    model_path = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"
    processor = transformers.AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    renderer = NativeProtocolRenderer(processor, expected_tokenizer_length=151_669)
    user = {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "What color?"}],
    }
    call = {
        "role": "assistant",
        "reasoning_content": "I should inspect.",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "tgvf_focus_tool",
                    "arguments": {"target": "label"},
                },
            }
        ],
    }
    prompt = renderer.render([user], add_generation_prompt=True)
    completed_call = renderer.render([user, call], add_generation_prompt=False)
    next_prompt = renderer.render(
        [user, call, {"role": "tool", "content": [{"type": "image"}]}],
        add_generation_prompt=True,
    )
    close_ids = tuple(
        processor.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    )
    assert completed_call.token_ids[: len(prompt.token_ids)] == prompt.token_ids
    assert completed_call.token_ids[-len(close_ids) :] == close_ids
    sampled_ids = completed_call.token_ids[len(prompt.token_ids) : -len(close_ids)]
    sampled_text = completed_call.text[len(prompt.text) : -len("<|im_end|>\n")]
    sampled = _sampled(sampled_text, sampled_ids)
    registrar = _Registrar()
    appender = QwenNativeToolObservationAppender(
        tokenizer=processor.tokenizer, registrar=registrar
    )

    updated, _ = appender.append(
        prompt.token_ids,
        sampled,
        ObservationHandle("obs-0", sha256(b"obs").hexdigest()),
        call_index=0,
    )

    assert updated == next_prompt.token_ids
    assert updated.count(processor.tokenizer.convert_tokens_to_ids("<think>")) == 2
