"""Async Stage3-shaped scoring over the proven DeepEyes/API transports.

This adapter changes only the scalar composition.  It delegates answer
extraction and Qwen2.5-72B API judging to the same asynchronous scorer used by
the matched Crop/TGVF baseline, then combines that verified answer fact with
the immutable tool-utility label and the Stage3-shaped kernel.  An optional
gold-free asynchronous visual seam supplies Focus/Grounding without changing
the answer-verification path.
"""

from __future__ import annotations

from typing import Protocol

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.data.tgvf_tool_utility import TGVFToolUtilityRuntimeBinding
from tgvf_rl.trajectories.schema import TrajectoryRecord

from .stage3_shaped import (
    QualityJudgeScore,
    Stage3ShapedRewardFacts,
    Stage3ShapedRewardKernel,
    ToolNecessityLabel,
)
from .stage3_verl_adapter import (
    Stage3ShapedRewardSpec,
    Stage3VerlTrajectoryReward,
    Stage3VisualJudgeSampleFailure,
    Stage3VisualQualityJudgement,
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


class AsyncStage3VisualQualityJudge(Protocol):
    """Gold-free visual-quality boundary for one completed trajectory."""

    async def judge_async(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
        context: object,
    ) -> Stage3VisualQualityJudgement: ...


class AsyncStage3ShapedTGVFTrajectoryRewardScorer:
    """Compose Stage3 components while retaining the async answer path."""

    def __init__(
        self,
        *,
        answer_scorer: AsyncAnswerTrajectoryScorer,
        spec: Stage3ShapedRewardSpec,
        tool_utility: TGVFToolUtilityRuntimeBinding | None,
        visual_quality_judge: AsyncStage3VisualQualityJudge | None = None,
        kernel: Stage3ShapedRewardKernel | None = None,
    ) -> None:
        if not callable(getattr(answer_scorer, "score_async", None)):
            raise TypeError("answer_scorer must implement score_async()")
        if not isinstance(spec, Stage3ShapedRewardSpec):
            raise TypeError("spec must be Stage3ShapedRewardSpec")
        if spec.visual_quality_enabled:
            if not callable(getattr(visual_quality_judge, "judge_async", None)):
                raise TypeError("enabled async visual quality requires judge_async()")
        elif visual_quality_judge is not None:
            raise ValueError("disabled async visual quality cannot bind a visual judge")
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
        self.visual_quality_judge = visual_quality_judge
        self.kernel = kernel or Stage3ShapedRewardKernel(
            protocol_error_penalty=spec.protocol_error_penalty
        )
        if self.kernel.protocol_error_penalty != spec.protocol_error_penalty:
            raise IdentityMismatchError(
                "Stage3 kernel protocol penalty differs from its reward spec"
            )

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
                    *(("protocol_invalid",) if not context.protocol_valid else ()),
                    *context.tool_error_codes,
                )
            )
        )
        focus_score: QualityJudgeScore | None = None
        grounding_score: QualityJudgeScore | None = None
        quality_judge_failure: str | None = None
        visual_judge_usage = None
        if (
            self.spec.visual_quality_enabled
            and context.successful_tgvf_observation_count >= 1
        ):
            assert self.visual_quality_judge is not None
            try:
                judgement = await self.visual_quality_judge.judge_async(
                    request=request,
                    trajectory=trajectory,
                    context=context,
                )
            except Stage3VisualJudgeSampleFailure as error:
                quality_judge_failure = error.code
                visual_judge_usage = error.usage
            else:
                if not isinstance(judgement, Stage3VisualQualityJudgement):
                    raise TypeError(
                        "async visual_quality_judge returned the wrong result type"
                    )
                if (
                    judgement.trajectory_id != trajectory.identity.canonical_id
                    or judgement.sample_id != context.sample_id
                    or judgement.successful_observation_count
                    != context.successful_tgvf_observation_count
                    or judgement.judge_identity != self.spec.visual_judge_identity
                ):
                    raise IdentityMismatchError(
                        "async visual-quality judgement identity differs"
                    )
                focus_score = judgement.focus_score
                grounding_score = judgement.grounding_score
                visual_judge_usage = judgement.usage
        result = self.kernel.score(
            Stage3ShapedRewardFacts(
                answer_correct=verification.correct,
                tool_label=(
                    None if label is None else ToolNecessityLabel(label.utility_label)
                ),
                tool_call_count=context.tool_call_count,
                successful_tgvf_observation_count=(
                    context.successful_tgvf_observation_count
                ),
                focus_score=focus_score,
                grounding_score=grounding_score,
                quality_judge_failure=quality_judge_failure,
                quality_rewards_enabled=self.spec.visual_quality_enabled,
                label_confidence=None if label is None else label.confidence,
                tool_utility_reward_enabled=(self.spec.tool_utility_reward_enabled),
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
            visual_judge_usage=visual_judge_usage,
        )


__all__ = [
    "AsyncAnswerTrajectoryScorer",
    "AsyncStage3VisualQualityJudge",
    "AsyncStage3ShapedTGVFTrajectoryRewardScorer",
]
