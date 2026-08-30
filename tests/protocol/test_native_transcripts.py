from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import SampledPolicyTurn
from tgvf_rl.environment.native_appender import (
    NativeSuccessObservationContract,
    QwenNativeToolObservationAppender,
    render_qwen_native_success_environment_text,
)
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol.native import NativeAssistantDialect, NativeProtocolRenderer
from tgvf_rl.protocol.observation_contract import (
    NativeSuccessObservationProtocolId,
)
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import (
    NativeToolCapabilityProfile,
    TokenByteSpan,
    build_native_tool_schemas,
)
from tgvf_rl.protocol.tool_prompts import (
    NATIVE_SHARED_USER_TEXT_TEMPLATE,
    NATIVE_SHARED_USER_TEXT_TEMPLATE_SHA256,
    SHARED_USER_PROMPT_TEMPLATE_SHA256,
    TGVF_VISUAL_TOOL_PROMPTS_VERSION,
    build_visual_tool_prompt_messages,
    native_policy_messages_sha256,
    render_successful_visual_tool_response,
    visual_tool_prompt_identity,
)


FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "qwen3_visual_tool_prompts_v3.json"
)

_PROFILE_BY_FIXTURE_NAME = {
    "tgvf_only": NativeToolCapabilityProfile.TGVF_ONLY,
    "crop_only": NativeToolCapabilityProfile.CROP_ONLY,
    "tgvf_crop": NativeToolCapabilityProfile.CROP_TGVF,
}
_FIRST_TARGET = "the small label's background color for identifying its color"
_SECOND_TARGET = (
    "the lower half of the small label's background color for resolving the "
    "ambiguous shade"
)


@pytest.fixture(scope="module")
def qwen3_processor_and_fixture():
    fixture = json.loads(FIXTURE_PATH.read_text())
    model_path = Path(fixture["model_path"])
    if not model_path.is_dir():
        pytest.skip("accepted local Qwen3 processor is unavailable")
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    return processor, fixture


def _renderer(
    processor: Any,
    profile: NativeToolCapabilityProfile,
    *,
    tokenizer_length: int,
) -> NativeProtocolRenderer:
    schemas = build_native_tool_schemas(profile.tool_names)
    return NativeProtocolRenderer(
        processor,
        expected_tokenizer_length=tokenizer_length,
        tool_names=profile.tool_names,
        tool_schemas=schemas,
    )


def _assistant_tool_call(reasoning: str, target: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "reasoning_content": reasoning,
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "tgvf_focus_tool",
                    "arguments": {"target": target},
                },
            }
        ],
    }


def _successful_focus_response(target: str) -> dict[str, Any]:
    response_text = render_successful_visual_tool_response(
        "tgvf_focus_tool",
        {"target": target},
    )
    return {
        "role": "tool",
        "content": [
            {"type": "text", "text": response_text + "\n"},
            {"type": "image"},
        ],
    }


def _focus_transcript_cases(question: str) -> dict[str, tuple[dict[str, Any], ...]]:
    base = build_visual_tool_prompt_messages(
        question,
        tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
    )
    call1 = _assistant_tool_call(
        "I should inspect the label closely.",
        _FIRST_TARGET,
    )
    call2 = _assistant_tool_call(
        "The first observation is ambiguous, so I should inspect the lower half.",
        _SECOND_TARGET,
    )
    answer = {
        "role": "assistant",
        "reasoning_content": "The focused evidence shows a blue background.",
        "content": "The label is blue.",
    }
    return {
        "direct": base + (answer,),
        "one_call": base + (call1, _successful_focus_response(_FIRST_TARGET), answer),
        "two_repeated_calls": base
        + (
            call1,
            _successful_focus_response(_FIRST_TARGET),
            call2,
            _successful_focus_response(_SECOND_TARGET),
            answer,
        ),
    }


def _assert_golden(rendered, expected: dict[str, Any]) -> None:
    assert len(rendered.token_ids) == expected["length"]
    assert rendered.token_ids_sha256 == expected["token_ids_sha256"]
    assert rendered.text_sha256 == expected["text_sha256"]


def _assert_decode_round_trip(processor: Any, rendered) -> None:
    decoded = processor.tokenizer.decode(
        rendered.token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert decoded == rendered.text


def test_qwen3_visual_tool_prompt_v3_initial_generation_goldens(
    qwen3_processor_and_fixture,
) -> None:
    processor, fixture = qwen3_processor_and_fixture
    assert fixture["contract_version"] == TGVF_VISUAL_TOOL_PROMPTS_VERSION
    assert fixture["tokenizer_length"] == 151_669
    assert len(processor.tokenizer) == fixture["tokenizer_length"]
    assert (
        fixture["shared_user_prompt_template_sha256"]
        == SHARED_USER_PROMPT_TEMPLATE_SHA256
    )
    assert (
        fixture["native_shared_user_text_template_sha256"]
        == NATIVE_SHARED_USER_TEXT_TEMPLATE_SHA256
    )

    question = fixture["question"]
    native_user_fragment = NATIVE_SHARED_USER_TEXT_TEMPLATE.format(question=question)
    for fixture_name, profile in _PROFILE_BY_FIXTURE_NAME.items():
        expected = fixture["profiles"][fixture_name]
        messages = build_visual_tool_prompt_messages(
            question,
            tool_profile=profile,
        )
        identity = visual_tool_prompt_identity(profile)
        renderer = _renderer(
            processor,
            profile,
            tokenizer_length=fixture["tokenizer_length"],
        )
        rendered = renderer.render(messages, add_generation_prompt=True)

        assert expected["profile_value"] == profile.value
        assert expected["tool_name"] == profile.tool_names[0]
        assert expected["schema_sha256"] == renderer.tool_schema_sha256
        assert expected["prompt_bundle_sha256"] == identity.bundle_sha256
        assert expected["messages_sha256"] == native_policy_messages_sha256(messages)
        assert renderer.chat_template_sha256 == fixture["chat_template_sha256"]
        assert rendered.tool_schema_sha256 == expected["schema_sha256"]
        _assert_golden(rendered, expected["initial_generation_prompt"])
        _assert_decode_round_trip(processor, rendered)
        renderer.assert_generation_prefill(rendered, processor.tokenizer)

        assert messages[1]["content"][1]["text"] == native_user_fragment
        assert native_user_fragment in rendered.text
        assert rendered.text.count("<think>") == 1
        assert rendered.tokenizer_length == fixture["tokenizer_length"]

    assert len(processor.tokenizer) == fixture["tokenizer_length"]


def test_qwen3_focus_direct_one_call_and_repeated_call_goldens(
    qwen3_processor_and_fixture,
) -> None:
    processor, fixture = qwen3_processor_and_fixture
    renderer = _renderer(
        processor,
        NativeToolCapabilityProfile.TGVF_ONLY,
        tokenizer_length=fixture["tokenizer_length"],
    )
    cases = _focus_transcript_cases(fixture["question"])
    expected_turns = {"direct": 1, "one_call": 2, "two_repeated_calls": 3}
    expected_responses = {
        "direct": (),
        "one_call": (_FIRST_TARGET,),
        "two_repeated_calls": (_FIRST_TARGET, _SECOND_TARGET),
    }

    for name, messages in cases.items():
        expected = fixture["focus_transcripts"][name]
        rendered = renderer.render(messages, add_generation_prompt=False)
        assert expected["messages_sha256"] == native_policy_messages_sha256(messages)
        _assert_golden(rendered, expected)
        _assert_decode_round_trip(processor, rendered)
        assert rendered.text.count("<think>") == expected_turns[name]
        assert rendered.text.count("<tool_response>") == len(expected_responses[name])
        for target in expected_responses[name]:
            accepted_response = render_successful_visual_tool_response(
                "tgvf_focus_tool",
                {"target": target},
            )
            assert accepted_response in rendered.text

    repeated = cases["two_repeated_calls"]
    repeated_names = tuple(
        message["tool_calls"][0]["function"]["name"]
        for message in repeated
        if message["role"] == "assistant" and message.get("tool_calls")
    )
    assert repeated_names == ("tgvf_focus_tool", "tgvf_focus_tool")
    assert len(processor.tokenizer) == fixture["tokenizer_length"]


class _Registrar:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def register_tool_turn(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _sampled_policy_turn(
    tokenizer: Any,
    *,
    text: str,
    token_ids: tuple[int, ...],
) -> SampledPolicyTurn:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    assert tuple(encoded["input_ids"]) == token_ids
    byte_boundaries = [0]
    for character in text:
        byte_boundaries.append(byte_boundaries[-1] + len(character.encode("utf-8")))
    spans = tuple(
        TokenByteSpan(
            token_index=index,
            token_id=token_id,
            byte_start=byte_boundaries[start],
            byte_end=byte_boundaries[end],
        )
        for index, (token_id, (start, end)) in enumerate(
            zip(token_ids, encoded["offset_mapping"], strict=True)
        )
    )
    policy = PolicyVersion("qwen3-visual-tool-prompt-golden", 0, "1" * 64)
    sampling = SamplingIdentity(
        policy_version=policy,
        backend="vllm",
        backend_version="golden-fixture-only",
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


def test_manual_success_appender_matches_full_qwen_rerender_exactly(
    qwen3_processor_and_fixture,
) -> None:
    processor, fixture = qwen3_processor_and_fixture
    renderer = _renderer(
        processor,
        NativeToolCapabilityProfile.TGVF_ONLY,
        tokenizer_length=fixture["tokenizer_length"],
    )
    base = build_visual_tool_prompt_messages(
        fixture["question"],
        tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
    )
    call = _assistant_tool_call(
        "I should inspect the label closely.",
        _FIRST_TARGET,
    )
    initial = renderer.render(base, add_generation_prompt=True)
    completed_call = renderer.render(base + (call,), add_generation_prompt=False)
    close_text = "<|im_end|>\n"
    close_ids = tuple(processor.tokenizer.encode(close_text, add_special_tokens=False))
    assert completed_call.token_ids[: len(initial.token_ids)] == initial.token_ids
    assert completed_call.token_ids[-len(close_ids) :] == close_ids
    sampled_ids = completed_call.token_ids[len(initial.token_ids) : -len(close_ids)]
    sampled_text = completed_call.text[len(initial.text) : -len(close_text)]
    sampled = _sampled_policy_turn(
        processor.tokenizer,
        text=sampled_text,
        token_ids=sampled_ids,
    )
    parsed = StrictToolCallParser(
        enabled_tool_names=NativeToolCapabilityProfile.TGVF_ONLY.tool_names
    ).parse(sampled.parser_turn())
    registrar = _Registrar()
    appender = QwenNativeToolObservationAppender(
        tokenizer=processor.tokenizer,
        registrar=registrar,
        observation_contract=NativeSuccessObservationContract(
            protocol_id=NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1,
            tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_THINKING,
        ),
    )

    rerendered = renderer.render(
        base + (call, _successful_focus_response(_FIRST_TARGET)),
        add_generation_prompt=True,
    )
    updated, _environment_ids = appender.append(
        initial.token_ids,
        sampled,
        ObservationHandle("golden-observation", sha256(b"golden").hexdigest()),
        call_index=0,
        parsed_call=parsed,
    )

    manual_text = (
        initial.text
        + sampled.text
        + render_qwen_native_success_environment_text(parsed)
    )
    assert manual_text == rerendered.text
    assert updated == rerendered.token_ids
    assert registrar.calls[-1]["updated_prompt_token_ids"] == updated
    assert len(processor.tokenizer) == fixture["tokenizer_length"]
