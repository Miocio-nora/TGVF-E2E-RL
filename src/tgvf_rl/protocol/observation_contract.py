"""Identity-only contract for environment-owned visual-tool observations."""

from __future__ import annotations

from enum import Enum

from .native import NativeAssistantDialect
from .schema import NativeToolCapabilityProfile


class NativeSuccessObservationProtocolId(str, Enum):
    """Stable identity for one exact success-continuation byte protocol."""

    GENERIC_NATIVE_V1 = "qwen-native-success-v1"
    NO_TOOL_NO_EXECUTION_V1 = "qwen3-vl-instruct-no-tool-no-execution-v1"
    DEEPEYES_CROP_MATCHED_V1 = "qwen3-vl-instruct-deepeyes-crop-matched-v1"
    DEEPEYES_TGVF_MATCHED_V1 = "qwen3-vl-instruct-deepeyes-tgvf-matched-v1"
    DEEPEYES_ATOMIC_MATCHED_V1 = "qwen3-vl-instruct-deepeyes-atomic-matched-v1"
    LEGACY_CROP_GENERIC86_V1 = "qwen3-vl-instruct-crop-generic86-legacy-v1"
    LEGACY_CROP_GENERIC_THINKING_V1 = "qwen3-vl-thinking-crop-generic-legacy-v1"


def validate_success_observation_protocol(
    protocol_id: NativeSuccessObservationProtocolId | str,
    *,
    tool_profile: NativeToolCapabilityProfile,
    assistant_dialect: NativeAssistantDialect,
) -> NativeSuccessObservationProtocolId:
    """Validate one explicit success-observation identity without rendering.

    This lives in the protocol layer so run/evaluation configuration can bind
    the byte contract without importing the environment implementation.
    """

    try:
        selected = NativeSuccessObservationProtocolId(protocol_id)
    except (TypeError, ValueError) as error:
        raise ValueError("success observation protocol ID is invalid") from error
    if not isinstance(tool_profile, NativeToolCapabilityProfile):
        raise TypeError("tool_profile must be NativeToolCapabilityProfile")
    if not isinstance(assistant_dialect, NativeAssistantDialect):
        raise TypeError("assistant_dialect must be NativeAssistantDialect")

    matched_instruct_profiles = {
        NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1: (
            NativeToolCapabilityProfile.CROP_ONLY
        ),
        NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1: (
            NativeToolCapabilityProfile.TGVF_ONLY
        ),
        NativeSuccessObservationProtocolId.DEEPEYES_ATOMIC_MATCHED_V1: (
            NativeToolCapabilityProfile.CROP_TGVF
        ),
    }
    if selected is NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1:
        if tool_profile is not NativeToolCapabilityProfile.NO_TOOL:
            raise ValueError("no-tool observation identity requires no_tool")
        if assistant_dialect is not NativeAssistantDialect.QWEN3_VL_INSTRUCT:
            raise ValueError("no-tool observation identity requires Qwen3-VL Instruct")
        return selected
    if selected in matched_instruct_profiles:
        if tool_profile is not matched_instruct_profiles[selected]:
            raise ValueError(
                f"{selected.value} requires {matched_instruct_profiles[selected].value}"
            )
        if assistant_dialect is not NativeAssistantDialect.QWEN3_VL_INSTRUCT:
            raise ValueError("matched DeepEyes observation requires Qwen3-VL Instruct")
        return selected

    legacy_crop_protocols = {
        NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1,
        NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC_THINKING_V1,
    }
    if (
        selected in legacy_crop_protocols
        and tool_profile is not NativeToolCapabilityProfile.CROP_ONLY
    ):
        raise ValueError("Crop observation protocols require crop_only")
    if selected is NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1:
        if tool_profile is NativeToolCapabilityProfile.CROP_ONLY:
            raise ValueError(
                "plain Crop requires an explicit matched or legacy Crop protocol"
            )
        return selected
    if selected is NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1:
        if assistant_dialect is not NativeAssistantDialect.QWEN3_VL_INSTRUCT:
            raise ValueError("generic86 Crop requires Qwen3-VL Instruct")
        return selected
    if assistant_dialect is not NativeAssistantDialect.QWEN3_VL_THINKING:
        raise ValueError("legacy Thinking Crop requires Qwen3-VL Thinking")
    return selected


__all__ = [
    "NativeSuccessObservationProtocolId",
    "validate_success_observation_protocol",
]
