from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import SampledPolicyTurn
from tgvf_rl.environment.native_appender import (
    NativeSuccessObservationContract,
    QWEN_NATIVE_IMAGE_PLACEHOLDER,
    QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX,
    QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT,
    QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT_SHA256,
    QWEN_NATIVE_MATCHED_ATOMIC_SUCCESS_TEXT,
    QWEN_NATIVE_MATCHED_ATOMIC_SUCCESS_TEXT_SHA256,
    QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT,
    QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256,
    QWEN_NATIVE_MATCHED_TGVF_SUCCESS_TEXT,
    QWEN_NATIVE_MATCHED_TGVF_SUCCESS_TEXT_SHA256,
    QWEN_NATIVE_RESPONSE_SUFFIX,
    QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX,
    QwenNativeToolObservationAppender,
    render_qwen_native_success_payload,
)
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol import (
    NativeAssistantDialect,
    NativeProtocolRenderer,
    NativeSuccessObservationProtocolId,
    StrictToolCallParser,
    TokenByteSpan,
)
from tgvf_rl.protocol.schema import StandardToolError
from tgvf_rl.protocol.schema import NativeToolCapabilityProfile


_QWEN3_VL_INSTRUCT_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct")
_QWEN3_VL_THINKING_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking")


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


class _CanonicalizingTokenizer(_CharacterTokenizer):
    """Tokenizer whose fresh encode merges one already-sampled sequence."""

    def __init__(self, sampled_text: str) -> None:
        self.sampled_text = sampled_text
        self.encoded_texts: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encoded_texts.append(text)
        if text == self.sampled_text:
            return [999_999]
        return super().encode(text, add_special_tokens=add_special_tokens)


class _ImagePadTokenizer(_CharacterTokenizer):
    image_token_id = 9876

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result: list[int] = []
        cursor = 0
        token = "<|image_pad|>"
        while cursor < len(text):
            if text.startswith(token, cursor):
                result.append(self.image_token_id)
                cursor += len(token)
            else:
                result.append(ord(text[cursor]))
                cursor += 1
        return result

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|image_pad|>"
        return self.image_token_id


class _VisualTokenCountStore:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.calls: list[ObservationHandle] = []

    def resolve_visual_token_count(self, observation: ObservationHandle) -> int:
        self.calls.append(observation)
        return self.counts[observation.observation_id]


class _Registrar:
    def __init__(self) -> None:
        self.calls = []

    def register_tool_turn(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _generic_contract(
    tool_profile: NativeToolCapabilityProfile = (NativeToolCapabilityProfile.TGVF_ONLY),
    *,
    assistant_dialect: NativeAssistantDialect = (
        NativeAssistantDialect.QWEN3_VL_THINKING
    ),
) -> NativeSuccessObservationContract:
    return NativeSuccessObservationContract(
        protocol_id=NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1,
        tool_profile=tool_profile,
        assistant_dialect=assistant_dialect,
    )


def _matched_crop_contract() -> NativeSuccessObservationContract:
    return NativeSuccessObservationContract(
        protocol_id=(NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1),
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )


def _legacy_crop_contract() -> NativeSuccessObservationContract:
    return NativeSuccessObservationContract(
        protocol_id=(NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1),
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )


def _matched_no_echo_contract(
    profile: NativeToolCapabilityProfile,
) -> NativeSuccessObservationContract:
    protocol_id = {
        NativeToolCapabilityProfile.TGVF_ONLY: (
            NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1
        ),
        NativeToolCapabilityProfile.CROP_TGVF: (
            NativeSuccessObservationProtocolId.DEEPEYES_ATOMIC_MATCHED_V1
        ),
    }[profile]
    return NativeSuccessObservationContract(
        protocol_id=protocol_id,
        tool_profile=profile,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )


def _sampled(
    text: str,
    token_ids: tuple[int, ...],
    *,
    token_byte_spans: tuple[TokenByteSpan, ...] | None = None,
) -> SampledPolicyTurn:
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
    spans = token_byte_spans or tuple(
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


def _ascii_sampled_call(tool_name: str, arguments: dict[str, object]):
    payload = json.dumps(
        {"name": tool_name, "arguments": arguments},
        separators=(",", ":"),
    )
    text = f"inspect\n</think>\n<tool_call>{payload}</tool_call>"
    token_ids = tuple(ord(character) for character in text)
    spans = tuple(
        TokenByteSpan(index, token_id, index, index + 1)
        for index, token_id in enumerate(token_ids)
    )
    sampled = _sampled(text, token_ids, token_byte_spans=spans)
    return sampled, StrictToolCallParser().parse(sampled.parser_turn())


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_text"),
    (
        (
            "tgvf_focus_tool",
            {"target": "the gauge needle position"},
            'Focused visual observation for target:\n"the gauge needle position"',
        ),
        (
            "tgvf_crop_tool",
            {
                "bbox_2d": [1, 2, 30, 40],
                "target": "the gauge needle position",
            },
            'Target-conditioned crop for:\n"the gauge needle position"',
        ),
    ),
)
def test_success_appends_exact_profile_text_then_one_image_placeholder(
    tool_name: str,
    arguments: dict[str, object],
    expected_text: str,
) -> None:
    tokenizer = _CharacterTokenizer()
    registrar = _Registrar()
    appender = QwenNativeToolObservationAppender(
        tokenizer=tokenizer,
        registrar=registrar,
        observation_contract=_generic_contract(
            NativeToolCapabilityProfile.TGVF_ONLY
            if tool_name == "tgvf_focus_tool"
            else NativeToolCapabilityProfile.CROP_TGVF
        ),
    )
    sampled, parsed = _ascii_sampled_call(tool_name, arguments)
    prompt = (7, 8)

    updated, suffix = appender.append(
        prompt,
        sampled,
        ObservationHandle("obs-0", "5" * 64),
        call_index=0,
        parsed_call=parsed,
    )
    assert updated == prompt + sampled.token_ids + suffix
    assert "".join(map(chr, suffix)) == (
        QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX
        + expected_text
        + "\n"
        + QWEN_NATIVE_IMAGE_PLACEHOLDER
        + QWEN_NATIVE_RESPONSE_SUFFIX
    )
    assert render_qwen_native_success_payload(parsed) == (
        expected_text + "\n" + QWEN_NATIVE_IMAGE_PLACEHOLDER
    )
    assert registrar.calls[-1]["updated_prompt_token_ids"] == updated


def test_instruct_tool_response_starts_next_assistant_without_think() -> None:
    tokenizer = _CharacterTokenizer()
    appender = QwenNativeToolObservationAppender(
        tokenizer=tokenizer,
        registrar=_Registrar(),
        observation_contract=_generic_contract(
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
        ),
    )
    sampled, parsed = _ascii_sampled_call("tgvf_focus_tool", {"target": "the gauge"})

    _updated, suffix = appender.append(
        (7, 8),
        sampled,
        ObservationHandle("obs-instruct", "5" * 64),
        call_index=0,
        parsed_call=parsed,
    )

    suffix_text = "".join(map(chr, suffix))
    assert suffix_text.endswith(QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX)
    assert suffix_text.index(QWEN_NATIVE_IMAGE_PLACEHOLDER) < suffix_text.index(
        "Think first"
    )
    next_assistant = suffix_text.rsplit("<|im_start|>assistant\n", 1)[1]
    assert "<think>" not in next_assistant


def test_crop_protocol_requires_explicit_canonical_or_legacy_identity() -> None:
    with pytest.raises(ValueError, match="explicit matched or legacy Crop protocol"):
        NativeSuccessObservationContract(
            protocol_id=NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1,
            tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
        )

    with pytest.raises(ValueError, match="requires Qwen3-VL Instruct"):
        NativeSuccessObservationContract(
            protocol_id=(NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1),
            tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_THINKING,
        )

    legacy_thinking = NativeSuccessObservationContract(
        protocol_id=(
            NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC_THINKING_V1
        ),
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_THINKING,
    )
    _sampled_turn, parsed = _ascii_sampled_call(
        "image_zoom_in_tool", {"bbox_2d": [1, 2, 30, 40]}
    )
    rendered = legacy_thinking.render(parsed)
    assert "Zoomed-in visual observation" in rendered
    assert rendered.endswith(QWEN_NATIVE_RESPONSE_SUFFIX)

    with pytest.raises(ValueError, match="requires Qwen3-VL Thinking"):
        NativeSuccessObservationContract(
            protocol_id=(
                NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC_THINKING_V1
            ),
            tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
        )


def test_canonical_and_legacy_crop_contracts_are_distinct_and_byte_locked() -> None:
    sampled, parsed = _ascii_sampled_call(
        "image_zoom_in_tool",
        {"bbox_2d": [1, 2, 30, 40], "label": "the small gauge"},
    )
    assert sampled.text == parsed.sampled_text

    matched = _matched_crop_contract().render(parsed)
    legacy = _legacy_crop_contract().render(parsed)

    assert matched == QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT
    assert legacy == QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT
    assert matched != legacy
    assert sha256(matched.encode("utf-8")).hexdigest() == (
        QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256
    )
    assert QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256 == (
        "f745fa6cfcc3ba9eb27125a49581fd823fb5930b7b0a51b28e51982999fa2d0a"
    )
    assert sha256(legacy.encode("utf-8")).hexdigest() == (
        QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT_SHA256
    )
    assert "Zoomed-in visual observation" not in matched
    assert "Zoomed-in visual observation" in legacy
    assert matched.count(QWEN_NATIVE_IMAGE_PLACEHOLDER) == 1
    assert legacy.count(QWEN_NATIVE_IMAGE_PLACEHOLDER) == 1


@pytest.mark.parametrize(
    ("profile", "tool_name", "arguments", "expected_text", "expected_sha256"),
    (
        (
            NativeToolCapabilityProfile.TGVF_ONLY,
            "tgvf_focus_tool",
            {"target": "runtime-specific-target-sentinel-tgvf"},
            QWEN_NATIVE_MATCHED_TGVF_SUCCESS_TEXT,
            "62275af22c3d399f4fffa25bc2a722104fd7a485f90b9b48f55e56919c9c9f87",
        ),
        (
            NativeToolCapabilityProfile.CROP_TGVF,
            "tgvf_crop_tool",
            {
                "bbox_2d": [1, 2, 30, 40],
                "target": "runtime-specific-target-sentinel-atomic",
            },
            QWEN_NATIVE_MATCHED_ATOMIC_SUCCESS_TEXT,
            "116b57898845be6e784a1d5f2e23304bcf595146cc2d4f9d85028074f16aec1e",
        ),
    ),
)
def test_matched_tgvf_and_atomic_observations_never_echo_sampled_target(
    profile: NativeToolCapabilityProfile,
    tool_name: str,
    arguments: dict[str, object],
    expected_text: str,
    expected_sha256: str,
) -> None:
    sampled_turn, parsed = _ascii_sampled_call(tool_name, arguments)
    contract = _matched_no_echo_contract(profile)
    rendered = contract.render(parsed)

    assert rendered == expected_text
    assert arguments["target"] not in rendered
    assert rendered.count(QWEN_NATIVE_IMAGE_PLACEHOLDER) == 1
    assert sha256(rendered.encode("utf-8")).hexdigest() == expected_sha256
    assert (
        QWEN_NATIVE_MATCHED_TGVF_SUCCESS_TEXT_SHA256
        if profile is NativeToolCapabilityProfile.TGVF_ONLY
        else QWEN_NATIVE_MATCHED_ATOMIC_SUCCESS_TEXT_SHA256
    ) == expected_sha256

    appender = QwenNativeToolObservationAppender(
        tokenizer=_CharacterTokenizer(),
        registrar=_Registrar(),
        observation_contract=contract,
    )
    _updated, environment_ids = appender.append(
        (7, 8),
        sampled_turn,
        ObservationHandle(f"obs-{profile.value}", "5" * 64),
        call_index=0,
        parsed_call=parsed,
    )
    environment_text = "".join(map(chr, environment_ids))
    assert environment_text == rendered
    assert arguments["target"] not in environment_text

    alternative_arguments = dict(arguments)
    alternative_arguments["target"] = "a-completely-different-runtime-target"
    _alternative_turn, alternative = _ascii_sampled_call(
        tool_name, alternative_arguments
    )
    assert _matched_no_echo_contract(profile).render(alternative) == rendered


def test_no_tool_observation_identity_cannot_append_any_tool_turn() -> None:
    contract = NativeSuccessObservationContract(
        protocol_id=NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1,
        tool_profile=NativeToolCapabilityProfile.NO_TOOL,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    sampled, parsed = _ascii_sampled_call(
        "tgvf_focus_tool", {"target": "must-never-execute"}
    )
    appender = QwenNativeToolObservationAppender(
        tokenizer=_CharacterTokenizer(),
        registrar=_Registrar(),
        observation_contract=contract,
    )

    with pytest.raises(RuntimeError, match="forbids appending tool turns"):
        appender.append(
            (7, 8),
            sampled,
            ObservationHandle("obs-no-tool", "5" * 64),
            call_index=0,
            parsed_call=parsed,
        )


@pytest.mark.parametrize(
    ("contract", "expected_text"),
    (
        (_matched_crop_contract(), QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT),
        (
            _legacy_crop_contract(),
            QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT,
        ),
    ),
)
def test_crop_appender_uses_only_its_explicit_contract(
    contract: NativeSuccessObservationContract,
    expected_text: str,
) -> None:
    tokenizer = _CharacterTokenizer()
    appender = QwenNativeToolObservationAppender(
        tokenizer=tokenizer,
        registrar=_Registrar(),
        observation_contract=contract,
    )
    sampled, parsed = _ascii_sampled_call(
        "image_zoom_in_tool", {"bbox_2d": [1, 2, 30, 40]}
    )

    _updated, suffix = appender.append(
        (7, 8),
        sampled,
        ObservationHandle("obs-explicit-crop", "5" * 64),
        call_index=0,
        parsed_call=parsed,
    )

    assert "".join(map(chr, suffix)) == expected_text


@pytest.mark.skipif(
    not _QWEN3_VL_INSTRUCT_PATH.is_dir(),
    reason="pinned local Qwen3-VL Instruct tokenizer is absent",
)
def test_real_qwen_tokenizer_preserves_matched60_and_legacy_generic86() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        _QWEN3_VL_INSTRUCT_PATH,
        local_files_only=True,
        trust_remote_code=False,
    )

    matched_ids = tokenizer.encode(
        QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT,
        add_special_tokens=False,
    )
    legacy_ids = tokenizer.encode(
        QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT,
        add_special_tokens=False,
    )

    assert len(matched_ids) == 60
    assert len(legacy_ids) == 86


def test_error_append_uses_canonical_json_without_visual_placeholder() -> None:
    tokenizer = _CharacterTokenizer()
    registrar = _Registrar()
    appender = QwenNativeToolObservationAppender(
        tokenizer=tokenizer,
        registrar=registrar,
        observation_contract=_generic_contract(),
    )
    sampled, _parsed = _ascii_sampled_call(
        "tgvf_focus_tool", {"target": "the red label text"}
    )
    prompt = (7, 8)

    error = StandardToolError(
        code="tool_execution_failed",
        message="failed",
        attempt_index=1,
        recoverable=True,
        maximum_tool_calls=4,
    )
    _, error_suffix = appender.append(
        prompt,
        sampled,
        error,
        call_index=1,
        parsed_call=None,
    )
    rendered_error = "".join(map(chr, error_suffix))
    assert error.canonical_json in rendered_error
    assert QWEN_NATIVE_IMAGE_PLACEHOLDER not in rendered_error

    with pytest.raises(ValueError, match="requires its parsed call"):
        appender.append(
            prompt,
            sampled,
            ObservationHandle("obs-0", "5" * 64),
            call_index=0,
            parsed_call=None,
        )


def test_policy_sampled_tokens_are_appended_verbatim_without_reencoding() -> None:
    sampled, parsed = _ascii_sampled_call(
        "tgvf_focus_tool", {"target": "the red label text"}
    )
    tokenizer = _CanonicalizingTokenizer(sampled.text)
    registrar = _Registrar()
    appender = QwenNativeToolObservationAppender(
        tokenizer=tokenizer,
        registrar=registrar,
        observation_contract=_generic_contract(),
    )

    updated, environment_ids = appender.append(
        (7, 8),
        sampled,
        ObservationHandle("obs-noncanonical", "5" * 64),
        call_index=0,
        parsed_call=parsed,
    )

    assert tokenizer.sampled_text not in tokenizer.encoded_texts
    assert updated == (7, 8) + sampled.token_ids + environment_ids
    assert updated[2 : 2 + len(sampled.token_ids)] == sampled.token_ids


def test_success_expands_placeholder_to_store_resolved_visual_token_count() -> None:
    tokenizer = _ImagePadTokenizer()
    registrar = _Registrar()
    store = _VisualTokenCountStore({"obs-expanded": 6})
    appender = QwenNativeToolObservationAppender(
        tokenizer=tokenizer,
        registrar=registrar,
        observation_contract=_generic_contract(),
        visual_token_count_resolver=store,
    )
    sampled, parsed = _ascii_sampled_call(
        "tgvf_focus_tool", {"target": "the red label text"}
    )
    handle = ObservationHandle("obs-expanded", "5" * 64)

    updated, environment_ids = appender.append(
        (7, 8),
        sampled,
        handle,
        call_index=0,
        parsed_call=parsed,
    )

    assert environment_ids.count(tokenizer.image_token_id) == 6
    assert store.calls == [handle]
    assert updated == (7, 8) + sampled.token_ids + environment_ids
    assert registrar.calls[-1]["updated_prompt_token_ids"] == updated


@pytest.mark.skipif(
    not _QWEN3_VL_THINKING_PATH.is_dir(),
    reason="pinned local Qwen3-VL Thinking processor is absent",
)
def test_qwen3_appended_tokens_equal_native_chat_template() -> None:
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        _QWEN3_VL_THINKING_PATH,
        local_files_only=True,
        trust_remote_code=False,
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
    close_ids = tuple(
        processor.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    )
    assert completed_call.token_ids[: len(prompt.token_ids)] == prompt.token_ids
    assert completed_call.token_ids[-len(close_ids) :] == close_ids
    sampled_ids = completed_call.token_ids[len(prompt.token_ids) : -len(close_ids)]
    sampled_text = completed_call.text[len(prompt.text) : -len("<|im_end|>\n")]
    spans = _fast_token_spans(processor.tokenizer, sampled_text, sampled_ids)
    sampled = _sampled(
        sampled_text,
        sampled_ids,
        token_byte_spans=spans,
    )
    parsed = StrictToolCallParser().parse(sampled.parser_turn())
    registrar = _Registrar()
    appender = QwenNativeToolObservationAppender(
        tokenizer=processor.tokenizer,
        registrar=registrar,
        observation_contract=_generic_contract(),
    )

    response_text = 'Focused visual observation for target:\n"label"'
    next_prompt = renderer.render(
        [
            user,
            call,
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": response_text + "\n"},
                    {"type": "image"},
                ],
            },
        ],
        add_generation_prompt=True,
    )
    updated, _ = appender.append(
        prompt.token_ids,
        sampled,
        ObservationHandle("obs-0", sha256(b"obs").hexdigest()),
        call_index=0,
        parsed_call=parsed,
    )

    assert updated == next_prompt.token_ids
    assert updated.count(processor.tokenizer.convert_tokens_to_ids("<think>")) == 2


def _fast_token_spans(tokenizer, text: str, token_ids: tuple[int, ...]):
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
    spans = []
    for index, (token_id, offsets) in enumerate(
        zip(token_ids, encoded["offset_mapping"], strict=True)
    ):
        start, end = offsets
        spans.append(
            TokenByteSpan(
                index,
                token_id,
                byte_boundaries[start],
                byte_boundaries[end],
            )
        )
    return tuple(spans)
