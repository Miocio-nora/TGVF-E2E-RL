"""Native DeepEyes Crop reward manager with the current T-free equation.

This module deliberately leaves the proven PRL14 answer extraction, Qwen2.5
72B judging, ThinkLite math fallback, and native-Crop audit normalization in
``DeepEyesOfficialRewardManager`` unchanged.  It replaces only the final
scalar composition with the same pure ``Stage3ShapedRewardKernel`` used by
the RP67 T-free policy runs.

The exact equation is::

    2 * answer_correct
    - 0.05 * max(0, tool_attempt_count - 1)
    - 1[invalid final protocol or any Crop/tool error]

Tool-utility labels, positive successful-Crop bonuses, Focus reward, and
Grounding reward are all disabled.  The five Stage3 components and their
evidence remain in the reward-extra audit surface for every source family.
"""

from __future__ import annotations

from collections.abc import Mapping
import json

from .deepeyes_verl_reward import DeepEyesOfficialRewardManager
from .stage3_shaped import (
    STAGE3_SHAPED_REWARD_VERSION,
    Stage3ShapedRewardFacts,
    Stage3ShapedRewardKernel,
)


DEEPEYES_CROP_TFREE_VERL_REWARD_SCHEMA = (
    "tgvf.deepeyes-crop-tfree-verl-reward-manager.v1"
)
DEEPEYES_CROP_TFREE_VERL_REWARD_MANAGER_CLASS = (
    "tgvf_rl.rewards.deepeyes_crop_tfree_verl_reward.DeepEyesCropTFreeRewardManager"
)

_STAGE3_COMPONENT_FIELD_NAMES = (
    "stage3_answer_reward",
    "stage3_tool_reward",
    "stage3_focus_reward",
    "stage3_grounding_reward",
    "stage3_protocol_reward",
)


def _strict_nonnegative_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"native Crop T-free {name} must be a non-negative integer")
    return value


def _strict_binary_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise ValueError(f"native Crop T-free {name} must be 0 or 1")
    return value


def _tool_attempt_count(result: Mapping[str, object]) -> int:
    """Recover all attempts, including parser/dispatch failures.

    ``crop_call_count`` counts calls that reached ``execute``.  Every tool
    attempt renders one native observation span, including failures before
    execute, so the span count is the lossless current-reward equivalent of
    ``len(trajectory.tool_calls) + len(trajectory.tool_errors)``.
    """

    crop_call_count = _strict_nonnegative_integer(
        result.get("crop_call_count"), name="crop_call_count"
    )
    spans = result.get("crop_observation_token_spans")
    if not isinstance(spans, list):
        raise TypeError("native Crop observation spans must remain a list")
    if len(spans) < crop_call_count:
        raise ValueError("native Crop rendered observations are fewer than calls")
    return len(spans)


def _protocol_errors(result: Mapping[str, object]) -> tuple[str, ...]:
    """Derive the one-bit protocol penalty from lossless native audit facts.

    Calls that reach ``NativeDeepEyesCropTool.execute`` are represented by
    ``crop_call_count``.  A parser/tool-dispatch failure may instead render an
    error observation without reaching ``execute``; the native agent records
    that response in ``crop_observation_token_spans``.  The latter therefore
    closes the only error gap in ``crop_error_count`` without changing the
    repeated-call term, whose scientific contract is explicitly based on
    ``crop_call_count``.
    """

    errors: list[str] = []
    format_penalty = result.get("format_penalty")
    if format_penalty not in {0, -1}:
        raise ValueError("native Crop T-free format_penalty must be 0 or -1")
    if format_penalty == -1:
        errors.append("protocol_invalid")

    crop_call_count = _strict_nonnegative_integer(
        result.get("crop_call_count"), name="crop_call_count"
    )
    crop_error_count = _strict_nonnegative_integer(
        result.get("crop_error_count"), name="crop_error_count"
    )
    if crop_error_count > crop_call_count:
        raise ValueError("native Crop errors exceed calls reaching execute")
    if crop_error_count:
        errors.append("crop_tool_error")

    tool_attempt_count = _tool_attempt_count(result)
    if tool_attempt_count > crop_call_count:
        errors.append("tool_parse_or_dispatch_error")
    overflow = _strict_binary_integer(
        result.get("decoder_context_overflow"), name="decoder_context_overflow"
    )
    if overflow:
        errors.append("decoder_context_overflow")
    return tuple(errors)


def _compose_tfree_reward(
    historical: Mapping[str, object],
    *,
    kernel: Stage3ShapedRewardKernel,
) -> dict[str, object]:
    """Replace only the historical scalar with the exact T-free composition."""

    if not isinstance(historical, Mapping):
        raise TypeError("native Crop answer scorer result must be a mapping")
    if not isinstance(kernel, Stage3ShapedRewardKernel):
        raise TypeError("native Crop T-free kernel has the wrong type")

    accuracy = _strict_binary_integer(historical.get("acc"), name="acc")
    crop_call_count = _strict_nonnegative_integer(
        historical.get("crop_call_count"), name="crop_call_count"
    )
    tool_attempt_count = _tool_attempt_count(historical)
    crop_count = _strict_nonnegative_integer(
        historical.get("successful_crop_count"), name="successful_crop_count"
    )
    if crop_count > crop_call_count:
        raise ValueError("successful native Crops exceed calls reaching execute")
    protocol_errors = _protocol_errors(historical)

    shaped = kernel.score(
        Stage3ShapedRewardFacts(
            answer_correct=bool(accuracy),
            tool_label=None,
            tool_call_count=tool_attempt_count,
            successful_tgvf_observation_count=crop_count,
            quality_rewards_enabled=False,
            label_confidence=None,
            tool_utility_reward_enabled=False,
            protocol_errors=protocol_errors,
        )
    )
    components = tuple(shaped.components)
    if len(components) != len(_STAGE3_COMPONENT_FIELD_NAMES):
        raise RuntimeError("native Crop T-free kernel component count differs")

    result = dict(historical)
    result.update(
        {
            "score": shaped.total,
            # Preserve the legacy field for stable metric columns while making
            # explicit that this equation contains no conditional tool bonus.
            "conditional_tool": 0,
            "reward_profile": STAGE3_SHAPED_REWARD_VERSION,
            "stage3_reward_schema": DEEPEYES_CROP_TFREE_VERL_REWARD_SCHEMA,
            "stage3_answer_gated": int(shaped.answer_gated),
            "stage3_tool_attempt_count": tool_attempt_count,
            "stage3_quality_judge_applicable": int(shaped.quality_judge_applicable),
            "stage3_quality_judge_covered": int(shaped.quality_judge_covered),
            "stage3_protocol_errors": json.dumps(
                protocol_errors,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "stage3_reward_components": json.dumps(
                [
                    {
                        "name": component.name.value,
                        "score": component.score,
                        "evidence": component.evidence,
                    }
                    for component in components
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    )
    result.update(
        {
            field_name: component.score
            for field_name, component in zip(
                _STAGE3_COMPONENT_FIELD_NAMES, components, strict=True
            )
        }
    )
    return result


class DeepEyesCropTFreeRewardManager(DeepEyesOfficialRewardManager):
    """Hydra-importable native Crop manager with current T-free composition."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.stage3_kernel = Stage3ShapedRewardKernel()

    async def _score_visual(self, **value: object) -> dict[str, object]:
        audit = value.get("audit")
        if not isinstance(audit, Mapping):
            raise TypeError("native Crop T-free trajectory audit must be a mapping")
        overflow = _strict_binary_integer(
            audit.get("decoder_context_overflow"), name="decoder_context_overflow"
        )
        if overflow:
            historical = self._result(
                score=-1.0,
                accuracy=0,
                format_penalty=-1,
                conditional_tool=0,
                answer="",
                crop_count=int(value["crop_count"]),
                trajectory_id=str(value["trajectory_id"]),
                judge=None,
                visual_requested=0,
                thinklite_requested=0,
                route="decoder_context_overflow_no_answer",
                audit=audit,
            )
        else:
            historical = await super()._score_visual(**value)
        return _compose_tfree_reward(historical, kernel=self.stage3_kernel)

    async def _score_thinklite(self, **value: object) -> dict[str, object]:
        historical = await super()._score_thinklite(**value)
        return _compose_tfree_reward(historical, kernel=self.stage3_kernel)


__all__ = [
    "DEEPEYES_CROP_TFREE_VERL_REWARD_MANAGER_CLASS",
    "DEEPEYES_CROP_TFREE_VERL_REWARD_SCHEMA",
    "DeepEyesCropTFreeRewardManager",
]
