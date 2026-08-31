import pytest

from tgvf_rl.protocol import (
    NativeAssistantDialect,
    NativeSuccessObservationProtocolId,
    NativeToolCapabilityProfile,
    validate_success_observation_protocol,
)


def test_success_observation_protocol_ids_are_stable_and_explicit() -> None:
    assert tuple(item.value for item in NativeSuccessObservationProtocolId) == (
        "qwen-native-success-v1",
        "qwen3-vl-instruct-no-tool-no-execution-v1",
        "qwen3-vl-instruct-deepeyes-crop-matched-v1",
        "qwen3-vl-instruct-deepeyes-tgvf-matched-v1",
        "qwen3-vl-instruct-deepeyes-atomic-matched-v1",
        "qwen3-vl-instruct-crop-generic86-legacy-v1",
        "qwen3-vl-thinking-crop-generic-legacy-v1",
    )


def test_success_observation_protocol_validation_is_fail_closed() -> None:
    matched = validate_success_observation_protocol(
        NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    assert matched is NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1

    expected_profiles = {
        NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1: (
            NativeToolCapabilityProfile.NO_TOOL
        ),
        NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1: (
            NativeToolCapabilityProfile.TGVF_ONLY
        ),
        NativeSuccessObservationProtocolId.DEEPEYES_ATOMIC_MATCHED_V1: (
            NativeToolCapabilityProfile.CROP_TGVF
        ),
    }
    for protocol_id, profile in expected_profiles.items():
        assert (
            validate_success_observation_protocol(
                protocol_id,
                tool_profile=profile,
                assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
            )
            is protocol_id
        )
        with pytest.raises(ValueError, match="requires"):
            validate_success_observation_protocol(
                protocol_id,
                tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
                assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
            )
        with pytest.raises(ValueError, match="requires Qwen3-VL Instruct"):
            validate_success_observation_protocol(
                protocol_id,
                tool_profile=profile,
                assistant_dialect=NativeAssistantDialect.QWEN3_VL_THINKING,
            )

    with pytest.raises(ValueError, match="plain Crop requires"):
        validate_success_observation_protocol(
            NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1,
            tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
        )
    with pytest.raises(ValueError, match="requires Qwen3-VL Instruct"):
        validate_success_observation_protocol(
            NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1,
            tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_THINKING,
        )
