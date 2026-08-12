"""Async Stage3-shaped scoring over the proven DeepEyes answer transport.

This adapter changes only the scalar composition.  It delegates answer
extraction and Qwen2.5-72B API judging to the same asynchronous scorer used by
the matched Crop/TGVF baseline, then combines that verified answer fact with
the immutable tool-utility label and the Stage3-shaped kernel.  Visual
Focus/Grounding rewards are intentionally disabled for this execution path.
"""

from __future__ import annotations

from typing import Protocol

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.data.tgvf_tool_utility import TGVFToolUtilityRuntimeBinding
from tgvf_rl.trajectories.schema import TrajectoryRecord

from .stage3_shaped import (
    Stage3ShapedRewardFacts,
    Stage3ShapedRewardKernel,
    ToolNecessityLabel,
)
from .stage3_verl_adapter import (
    Stage3ShapedRewardSpec,
    Stage3VerlTrajectoryReward,
)
from .verl_adapter import PilotVerlTrajectoryReward


class AsyncAnswerTrajectoryScorer(Protocol):
    """The already-proven matched baseline answer-scoring boundary."""

    async def score_async(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
    ) -> PilotVerlTrajectoryReward: ...


class AsyncStage3ShapedTGVFTrajectoryRewardScorer:
    """Compose Answer/Tool/Protocol rewards without a visual judge call."""

    def __init__(
        self,
        *,
        answer_scorer: AsyncAnswerTrajectoryScorer,
        spec: Stage3ShapedRewardSpec,
        tool_utility: TGVFToolUtilityRuntimeBinding | None,
        kernel: Stage3ShapedRewardKernel | None = None,
    ) -> None:
        if not callable(getattr(answer_scorer, "score_async", None)):
            raise TypeError("answer_scorer must implement score_async()")
        if not isinstance(spec, Stage3ShapedRewardSpec):
            raise TypeError("spec must be Stage3ShapedRewardSpec")
        if spec.visual_quality_enabled:
            raise ValueError("async no-visual scorer requires disabled visual quality")
        if spec.tool_utility_reward_enabled:
            if not isinstance(tool_utility, TGVFToolUtilityRuntimeBinding):
                raise TypeError(
                    "enabled tool-utility reward requires a verified runtime binding"
                )
        elif tool_utility is not None:
            raise ValueError(
                "disabled tool-utility reward cannot bind a utility sidecar"
            )
        if tool_utility is not None:
            if tool_utility.sidecar_sha256 != spec.tool_utility_sidecar_sha256:
                raise IdentityMismatchError("Stage3 tool sidecar identity differs")
            if tool_utility.manifest_sha256 != spec.tool_utility_manifest_sha256:
                raise IdentityMismatchError("Stage3 tool sidecar manifest differs")
        self.answer_scorer = answer_scorer
        self.spec = spec
        self.tool_utility = tool_utility
        self.kernel = kernel or Stage3ShapedRewardKernel()

    async def score_async(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
    ) -> Stage3VerlTrajectoryReward:
        if not isinstance(trajectory, TrajectoryRecord):
            raise TypeError("trajectory must be TrajectoryRecord")
        if getattr(request, "identity", None) != trajectory.identity:
            raise IdentityMismatchError(
                "Stage3 reward request and trajectory identities differ"
            )
        answer_reward = await self.answer_scorer.score_async(
            request=request,
            trajectory=trajectory,
        )
        if not isinstance(answer_reward, PilotVerlTrajectoryReward):
            raise TypeError("answer scorer returned an invalid Pilot reward")
        context = answer_reward.context
        verification = answer_reward.result.answer_verification
        label = (
            None
            if self.tool_utility is None
            else self.tool_utility.label_for_sample(context.sample_id)
        )
        protocol_errors = tuple(
            dict.fromkeys(
                (
                    *(('protocol_invalid',) if not context.protocol_valid else ()),
                    *context.tool_error_codes,
                )
            )
        )
        result = self.kernel.score(
            Stage3ShapedRewardFacts(
                answer_correct=verification.correct,
                tool_label=(
                    None
                    if label is None
                    else ToolNecessityLabel(label.utility_label)
                ),
                tool_call_count=context.tool_call_count,
                successful_tgvf_observation_count=(
                    context.successful_tgvf_observation_count
                ),
                quality_rewards_enabled=False,
                label_confidence=None if label is None else label.confidence,
                tool_utility_reward_enabled=(
                    self.spec.tool_utility_reward_enabled
                ),
                protocol_errors=protocol_errors,
            )
        )
        return Stage3VerlTrajectoryReward(
            trajectory_id=trajectory.identity.canonical_id,
            group_uid=trajectory.identity.group_id,
            rollout_index=trajectory.identity.rollout_index,
            context=context,
            answer_verification=verification,
            tool_label=label,
            spec=self.spec,
            result=result,
        )


__all__ = [
    "AsyncAnswerTrajectoryScorer",
    "AsyncStage3ShapedTGVFTrajectoryRewardScorer",
]
