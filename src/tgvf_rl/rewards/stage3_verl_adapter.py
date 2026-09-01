"""Identity-bound ``stage3-shaped-v1`` trajectory reward bridge.

The bridge is intentionally parallel to the legacy Pilot-v1 adapter.  It
turns an immutable trajectory, a counterfactual tool-utility sidecar row, and
an injected visual-quality judgement into the pure five-component kernel.
No visual judge implementation lives here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from typing import Callable, Protocol

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.data.tgvf_tool_utility import (
    TGVFToolUtilityLabelBinding,
    TGVFToolUtilityRuntimeBinding,
)
from tgvf_rl.judges.base import JudgeUsage
from tgvf_rl.trajectories.schema import TrajectoryRecord

from .schema import (
    AnswerVerificationResult,
    AnswerVerifier,
    RewardContext,
)
from .stage3_shaped import (
    QualityJudgeScore,
    STAGE3_ANSWER_REWARD_SCALE,
    STAGE3_PROTOCOL_ERROR_PENALTY,
    STAGE3_REPEATED_CALL_PENALTY,
    STAGE3_SHAPED_REWARD_VERSION,
    Stage3ShapedRewardFacts,
    Stage3ShapedRewardKernel,
    Stage3ShapedRewardResult,
    ToolNecessityLabel,
)
from .verl_adapter import (
    PILOT_VERL_ANSWER_ROUTE_FIELD,
    PILOT_VERL_JUDGE_USAGE_FIELD,
    PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD,
    PILOT_VERL_REWARD_COMPONENTS_FIELD,
    PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD,
    PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD,
    PilotRewardContextProvider,
)


STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION = "tgvf-stage3-shaped-verl-reward-bridge-v1"
STAGE3_VERL_TOOL_LABEL_FIELD = "tgvf_stage3_tool_necessity_label"
STAGE3_VERL_TOOL_LABEL_CONFIDENCE_FIELD = "tgvf_stage3_tool_label_confidence"
STAGE3_VERL_TOOL_LABEL_ROW_SHA256_FIELD = "tgvf_stage3_tool_label_row_sha256"
STAGE3_VERL_TOOL_SIDECAR_SHA256_FIELD = "tgvf_stage3_tool_sidecar_sha256"
STAGE3_VERL_QUALITY_APPLICABLE_FIELD = "tgvf_stage3_quality_judge_applicable"
STAGE3_VERL_QUALITY_COVERED_FIELD = "tgvf_stage3_quality_judge_covered"
STAGE3_VERL_QUALITY_FAILURE_FIELD = "tgvf_stage3_quality_judge_failure"
STAGE3_VERL_VISUAL_JUDGE_USAGE_FIELD = "tgvf_stage3_visual_judge_usage"
_STAGE3_SAMPLE_LOCAL_VISUAL_FAILURE_CODES = frozenset({"transport", "malformed_output"})


@dataclass(frozen=True, slots=True)
class Stage3ShapedRewardSpec:
    """Versioned identities accepted by one Stage3-shaped reward scorer."""

    pipeline_identity: ArtifactIdentity
    answer_verifier_identity: ArtifactIdentity
    visual_judge_identity: ArtifactIdentity | None
    tool_utility_sidecar_sha256: str | None
    tool_utility_manifest_sha256: str | None
    visual_quality_enabled: bool = True
    tool_utility_reward_enabled: bool = True
    answer_reward_scale: float = STAGE3_ANSWER_REWARD_SCALE
    repeated_call_penalty: float = STAGE3_REPEATED_CALL_PENALTY
    protocol_error_penalty: float = STAGE3_PROTOCOL_ERROR_PENALTY
    profile: str = STAGE3_SHAPED_REWARD_VERSION

    def __post_init__(self) -> None:
        if self.profile != STAGE3_SHAPED_REWARD_VERSION:
            raise ValueError("unexpected Stage3-shaped reward profile")
        for field_name in ("pipeline_identity", "answer_verifier_identity"):
            if not isinstance(getattr(self, field_name), ArtifactIdentity):
                raise TypeError(f"{field_name} must be ArtifactIdentity")
        if type(self.visual_quality_enabled) is not bool:
            raise TypeError("visual_quality_enabled must be bool")
        if type(self.tool_utility_reward_enabled) is not bool:
            raise TypeError("tool_utility_reward_enabled must be bool")
        for field_name, value in (
            ("answer_reward_scale", self.answer_reward_scale),
            ("repeated_call_penalty", self.repeated_call_penalty),
            ("protocol_error_penalty", self.protocol_error_penalty),
        ):
            if type(value) is not float:
                raise TypeError(f"{field_name} must be float")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.visual_quality_enabled:
            if not isinstance(self.visual_judge_identity, ArtifactIdentity):
                raise TypeError("enabled visual quality requires an ArtifactIdentity")
        elif self.visual_judge_identity is not None:
            raise ValueError(
                "disabled visual quality cannot bind a visual judge identity"
            )
        utility_hashes = (
            self.tool_utility_sidecar_sha256,
            self.tool_utility_manifest_sha256,
        )
        if self.tool_utility_reward_enabled:
            for field_name, value in zip(
                (
                    "tool_utility_sidecar_sha256",
                    "tool_utility_manifest_sha256",
                ),
                utility_hashes,
                strict=True,
            ):
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"{field_name} must be a lowercase SHA-256")
        elif utility_hashes != (None, None):
            raise ValueError(
                "disabled tool-utility reward cannot bind utility identities"
            )


@dataclass(frozen=True, slots=True)
class Stage3VisualQualityJudgement:
    """Successful provider result tied to one exact trajectory and D count."""

    trajectory_id: str
    sample_id: str
    successful_observation_count: int
    focus_score: QualityJudgeScore
    grounding_score: QualityJudgeScore
    judge_identity: ArtifactIdentity
    usage: JudgeUsage | None = None

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.sample_id:
            raise ValueError("visual-quality result identities must be non-empty")
        if (
            type(self.successful_observation_count) is not int
            or self.successful_observation_count < 1
        ):
            raise ValueError(
                "visual-quality result requires a successful observation count"
            )
        if type(self.focus_score) is not QualityJudgeScore:
            raise TypeError("visual-quality focus_score has the wrong type")
        if type(self.grounding_score) is not QualityJudgeScore:
            raise TypeError("visual-quality grounding_score has the wrong type")
        if not isinstance(self.judge_identity, ArtifactIdentity):
            raise TypeError("visual-quality judge_identity has the wrong type")
        if self.usage is not None and not isinstance(self.usage, JudgeUsage):
            raise TypeError("visual-quality usage has the wrong type")


class Stage3VisualJudgeSampleFailure(RuntimeError):
    """One local transport/response failure that may fall back to F=G=0."""

    def __init__(self, code: str, *, usage: JudgeUsage | None = None) -> None:
        if (
            type(code) is not str
            or code not in _STAGE3_SAMPLE_LOCAL_VISUAL_FAILURE_CODES
        ):
            raise ValueError(
                "sample-local visual judge failure must be transport or "
                "malformed_output"
            )
        self.code = code
        if usage is not None and not isinstance(usage, JudgeUsage):
            raise TypeError("sample-local visual judge failure usage is invalid")
        self.usage = usage
        super().__init__(code)


class Stage3VisualQualityJudge(Protocol):
    """Provider seam; structural/identity exceptions must not be downgraded."""

    def judge(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
        context: RewardContext,
    ) -> Stage3VisualQualityJudgement: ...


@dataclass(frozen=True, slots=True)
class Stage3VerlTrajectoryReward:
    """One Stage3-shaped scalar plus all runtime identities and coverage."""

    trajectory_id: str
    group_uid: str
    rollout_index: int
    context: RewardContext
    answer_verification: AnswerVerificationResult
    tool_label: TGVFToolUtilityLabelBinding | None
    spec: Stage3ShapedRewardSpec
    result: Stage3ShapedRewardResult
    visual_judge_usage: JudgeUsage | None = None

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.group_uid:
            raise ValueError("Stage3 reward trajectory/group identities are required")
        if type(self.rollout_index) is not int or self.rollout_index < 0:
            raise ValueError("Stage3 reward rollout_index must be non-negative")
        if not isinstance(self.context, RewardContext):
            raise TypeError("Stage3 reward context has the wrong type")
        if not isinstance(self.answer_verification, AnswerVerificationResult):
            raise TypeError("Stage3 answer verification has the wrong type")
        if not isinstance(self.spec, Stage3ShapedRewardSpec):
            raise TypeError("Stage3 reward spec has the wrong type")
        if self.spec.tool_utility_reward_enabled:
            if not isinstance(self.tool_label, TGVFToolUtilityLabelBinding):
                raise TypeError("enabled Stage3 utility reward requires a tool label")
        elif self.tool_label is not None:
            raise ValueError("disabled Stage3 utility reward cannot carry a tool label")
        if not isinstance(self.result, Stage3ShapedRewardResult):
            raise TypeError("Stage3 reward result has the wrong type")
        if self.visual_judge_usage is not None and not isinstance(
            self.visual_judge_usage, JudgeUsage
        ):
            raise TypeError("Stage3 visual judge usage has the wrong type")
        if (
            self.tool_label is not None
            and self.tool_label.sample_id != self.context.sample_id
        ):
            raise IdentityMismatchError("Stage3 tool label belongs to another sample")
        if not math.isfinite(self.result.total):
            raise ValueError("Stage3 trajectory reward must be finite")

    @property
    def total(self) -> float:
        return float(self.result.total)

    @property
    def pipeline_sha256(self) -> str:
        return self.spec.pipeline_identity.sha256

    @property
    def raw_components(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (component.name.value, float(component.score))
            for component in self.result.components
        )

    def reward_sidecars(self) -> dict[str, object]:
        usage = self.answer_verification.judge_usage
        visual_usage = self.visual_judge_usage
        return {
            PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD: (
                STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION
            ),
            PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD: self.pipeline_sha256,
            PILOT_VERL_REWARD_COMPONENTS_FIELD: self.raw_components,
            PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD: self.trajectory_id,
            PILOT_VERL_ANSWER_ROUTE_FIELD: self.answer_verification.route,
            PILOT_VERL_JUDGE_USAGE_FIELD: (
                None
                if usage is None
                else (
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                    usage.cost_usd,
                )
            ),
            STAGE3_VERL_TOOL_LABEL_FIELD: (
                None if self.tool_label is None else self.tool_label.utility_label
            ),
            STAGE3_VERL_TOOL_LABEL_CONFIDENCE_FIELD: (
                None if self.tool_label is None else self.tool_label.confidence
            ),
            STAGE3_VERL_TOOL_LABEL_ROW_SHA256_FIELD: (
                None if self.tool_label is None else self.tool_label.row_sha256
            ),
            STAGE3_VERL_TOOL_SIDECAR_SHA256_FIELD: (
                self.spec.tool_utility_sidecar_sha256
            ),
            STAGE3_VERL_QUALITY_APPLICABLE_FIELD: (
                self.result.quality_judge_applicable
            ),
            STAGE3_VERL_QUALITY_COVERED_FIELD: self.result.quality_judge_covered,
            STAGE3_VERL_QUALITY_FAILURE_FIELD: self.result.quality_judge_failure,
            STAGE3_VERL_VISUAL_JUDGE_USAGE_FIELD: (
                None
                if visual_usage is None
                else (
                    visual_usage.prompt_tokens,
                    visual_usage.completion_tokens,
                    visual_usage.total_tokens,
                    visual_usage.cost_usd,
                )
            ),
        }

    def reward_extra_info(self) -> dict[str, object]:
        components = dict(self.raw_components)
        usage = self.answer_verification.judge_usage
        visual_usage = self.visual_judge_usage
        return {
            "tgvf_exact_trajectory_reward": self.total,
            **{f"stage3_{name}_reward": score for name, score in components.items()},
            "reward_pipeline_sha256": self.pipeline_sha256,
            "stage3_quality_judge_applicable": int(
                self.result.quality_judge_applicable
            ),
            "stage3_quality_judge_covered": int(self.result.quality_judge_covered),
            "stage3_quality_judge_failed": int(
                self.result.quality_judge_failure is not None
            ),
            "answer_judge_calls": int(usage is not None),
            "answer_judge_prompt_tokens": 0 if usage is None else usage.prompt_tokens,
            "answer_judge_completion_tokens": (
                0 if usage is None else usage.completion_tokens
            ),
            "answer_judge_cost_usd": 0.0 if usage is None else usage.cost_usd,
            "visual_judge_calls": int(self.result.quality_judge_applicable),
            "visual_judge_prompt_tokens": (
                0 if visual_usage is None else visual_usage.prompt_tokens
            ),
            "visual_judge_completion_tokens": (
                0 if visual_usage is None else visual_usage.completion_tokens
            ),
            "visual_judge_cost_usd": (
                0.0 if visual_usage is None else visual_usage.cost_usd
            ),
        }


class Stage3VerlTrajectoryRewardScorer:
    """Score one trajectory without changing legacy Pilot-v1 semantics."""

    def __init__(
        self,
        *,
        spec: Stage3ShapedRewardSpec,
        answer_verifier: AnswerVerifier,
        context_provider: PilotRewardContextProvider,
        tool_utility: TGVFToolUtilityRuntimeBinding | None,
        visual_quality_judge: Stage3VisualQualityJudge | None,
        kernel: Stage3ShapedRewardKernel | None = None,
        audit_sink: Callable[[TrajectoryRecord, Stage3VerlTrajectoryReward], None]
        | None = None,
    ) -> None:
        if not isinstance(spec, Stage3ShapedRewardSpec):
            raise TypeError("spec must be Stage3ShapedRewardSpec")
        if not callable(getattr(answer_verifier, "verify", None)):
            raise TypeError("answer_verifier must implement verify()")
        if not callable(getattr(context_provider, "build", None)):
            raise TypeError("context_provider must implement build()")
        if spec.tool_utility_reward_enabled:
            if not isinstance(tool_utility, TGVFToolUtilityRuntimeBinding):
                raise TypeError(
                    "enabled tool-utility reward requires a verified runtime binding"
                )
        elif tool_utility is not None:
            raise ValueError(
                "disabled tool-utility reward cannot bind a utility sidecar"
            )
        if spec.visual_quality_enabled:
            if not callable(getattr(visual_quality_judge, "judge", None)):
                raise TypeError("visual_quality_judge must implement judge()")
        elif visual_quality_judge is not None:
            raise ValueError(
                "disabled visual quality cannot bind a visual judge provider"
            )
        if tool_utility is not None:
            if tool_utility.sidecar_sha256 != spec.tool_utility_sidecar_sha256:
                raise IdentityMismatchError("Stage3 tool sidecar identity differs")
            if tool_utility.manifest_sha256 != spec.tool_utility_manifest_sha256:
                raise IdentityMismatchError("Stage3 tool sidecar manifest differs")
        rule_identity = getattr(answer_verifier, "rule_identity", None)
        if rule_identity is not None and rule_identity != spec.answer_verifier_identity:
            raise IdentityMismatchError("Stage3 answer rule identity differs")
        judge_identity = getattr(answer_verifier, "judge_model_identity", None)
        if judge_identity is not None and not isinstance(
            judge_identity, ArtifactIdentity
        ):
            raise TypeError(
                "answer verifier judge_model_identity must be ArtifactIdentity"
            )
        self.spec = spec
        self.answer_verifier = answer_verifier
        self.answer_judge_identity = judge_identity
        judge_route = getattr(answer_verifier, "judge_route", None)
        if judge_route is None and judge_identity is not None:
            judge_route = getattr(answer_verifier, "route", None)
        if judge_route is not None and (
            not isinstance(judge_route, str) or not judge_route.strip()
        ):
            raise TypeError("answer verifier judge route must be non-empty text")
        self.answer_judge_route = judge_route
        self.context_provider = context_provider
        self.tool_utility = tool_utility
        self.visual_quality_judge = visual_quality_judge
        self.kernel = kernel or Stage3ShapedRewardKernel(
            answer_reward_scale=spec.answer_reward_scale,
            repeated_call_penalty=spec.repeated_call_penalty,
            protocol_error_penalty=spec.protocol_error_penalty,
        )
        if (
            self.kernel.answer_reward_scale != spec.answer_reward_scale
            or self.kernel.repeated_call_penalty != spec.repeated_call_penalty
            or self.kernel.protocol_error_penalty != spec.protocol_error_penalty
        ):
            raise IdentityMismatchError(
                "Stage3 kernel coefficients differ from its reward spec"
            )
        self.audit_sink = audit_sink

    def score(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
    ) -> Stage3VerlTrajectoryReward:
        context, label = self._prepare(
            request=request,
            trajectory=trajectory,
        )
        verification = self._verify_answer(context)
        visual = self._judge_visual(
            request=request,
            trajectory=trajectory,
            context=context,
        )
        reward = self._finish_score(
            trajectory=trajectory,
            context=context,
            label=label,
            verification=verification,
            visual=visual,
        )
        if self.audit_sink is not None:
            self.audit_sink(trajectory, reward)
        return reward

    async def score_async(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
    ) -> Stage3VerlTrajectoryReward:
        """Await independent answer/visual judges and preserve the same kernel."""

        context, label = self._prepare(
            request=request,
            trajectory=trajectory,
        )
        verification, visual = await asyncio.gather(
            self._verify_answer_async(context),
            self._judge_visual_async(
                request=request,
                trajectory=trajectory,
                context=context,
            ),
        )
        reward = self._finish_score(
            trajectory=trajectory,
            context=context,
            label=label,
            verification=verification,
            visual=visual,
        )
        if self.audit_sink is not None:
            await asyncio.to_thread(self.audit_sink, trajectory, reward)
        return reward

    def _prepare(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
    ) -> tuple[RewardContext, TGVFToolUtilityLabelBinding | None]:
        if not isinstance(trajectory, TrajectoryRecord):
            raise TypeError("trajectory must be TrajectoryRecord")
        if getattr(request, "identity", None) != trajectory.identity:
            raise IdentityMismatchError(
                "Stage3 reward request and trajectory identities differ"
            )
        context = self.context_provider.build(request=request, trajectory=trajectory)
        if not isinstance(context, RewardContext):
            raise TypeError("context_provider must return RewardContext")
        if context.sample_id != trajectory.identity.sample_id:
            raise IdentityMismatchError(
                "Stage3 reward context belongs to another sample"
            )
        label = (
            None
            if self.tool_utility is None
            else self.tool_utility.label_for_sample(context.sample_id)
        )
        return context, label

    def _verify_answer(self, context: RewardContext) -> AnswerVerificationResult:
        verification = self.answer_verifier.verify(context)
        return self._validate_answer_verification(verification)

    async def _verify_answer_async(
        self, context: RewardContext
    ) -> AnswerVerificationResult:
        verify_async = getattr(self.answer_verifier, "verify_async", None)
        if callable(verify_async):
            verification = await verify_async(context)
        else:
            verification = await asyncio.to_thread(self.answer_verifier.verify, context)
        return self._validate_answer_verification(verification)

    def _validate_answer_verification(
        self, verification: object
    ) -> AnswerVerificationResult:
        if not isinstance(verification, AnswerVerificationResult):
            raise TypeError("answer_verifier returned the wrong result type")
        # RuleFirstAnswerVerifier intentionally reports the configured judge
        # model for semantic fallback and the rule identity otherwise.  Bind
        # the route to the exact identity, rather than merely accepting either
        # identity for every route.
        used_semantic_fallback = (
            self.answer_judge_identity is not None
            and verification.verifier_identity == self.answer_judge_identity
        )
        expected_verifier_identity = (
            self.answer_judge_identity
            if used_semantic_fallback
            else self.spec.answer_verifier_identity
        )
        if (
            expected_verifier_identity is None
            or verification.verifier_identity != expected_verifier_identity
        ):
            raise IdentityMismatchError(
                "Stage3 answer verifier route and configured identity differ"
            )
        if used_semantic_fallback:
            if self.answer_judge_route is None or not (
                verification.route == self.answer_judge_route
                or verification.route.startswith(self.answer_judge_route + "_")
            ):
                raise IdentityMismatchError(
                    "Stage3 semantic fallback route differs from its model binding"
                )
        elif self.answer_judge_route is not None and (
            verification.route == self.answer_judge_route
            or verification.route.startswith(self.answer_judge_route + "_")
        ):
            raise IdentityMismatchError(
                "Stage3 rule result claimed the configured judge route"
            )
        return verification

    def _judge_visual(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
        context: RewardContext,
    ) -> tuple[
        QualityJudgeScore | None,
        QualityJudgeScore | None,
        str | None,
        JudgeUsage | None,
    ]:
        if not (
            self.spec.visual_quality_enabled
            and context.successful_tgvf_observation_count >= 1
        ):
            return None, None, None, None
        assert self.visual_quality_judge is not None
        try:
            judgement = self.visual_quality_judge.judge(
                request=request,
                trajectory=trajectory,
                context=context,
            )
        except Stage3VisualJudgeSampleFailure as error:
            return None, None, error.code, error.usage
        return self._validated_visual_judgement(
            judgement,
            trajectory=trajectory,
            context=context,
        )

    async def _judge_visual_async(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
        context: RewardContext,
    ) -> tuple[
        QualityJudgeScore | None,
        QualityJudgeScore | None,
        str | None,
        JudgeUsage | None,
    ]:
        if not (
            self.spec.visual_quality_enabled
            and context.successful_tgvf_observation_count >= 1
        ):
            return None, None, None, None
        assert self.visual_quality_judge is not None
        try:
            judge_async = getattr(self.visual_quality_judge, "judge_async", None)
            if callable(judge_async):
                judgement = await judge_async(
                    request=request,
                    trajectory=trajectory,
                    context=context,
                )
            else:
                judgement = await asyncio.to_thread(
                    self.visual_quality_judge.judge,
                    request=request,
                    trajectory=trajectory,
                    context=context,
                )
        except Stage3VisualJudgeSampleFailure as error:
            return None, None, error.code, error.usage
        return self._validated_visual_judgement(
            judgement,
            trajectory=trajectory,
            context=context,
        )

    def _validated_visual_judgement(
        self,
        judgement: object,
        *,
        trajectory: TrajectoryRecord,
        context: RewardContext,
    ) -> tuple[
        QualityJudgeScore,
        QualityJudgeScore,
        None,
        JudgeUsage | None,
    ]:
        if not isinstance(judgement, Stage3VisualQualityJudgement):
            raise TypeError("visual_quality_judge returned the wrong result type")
        if (
            judgement.trajectory_id != trajectory.identity.canonical_id
            or judgement.sample_id != context.sample_id
            or judgement.successful_observation_count
            != context.successful_tgvf_observation_count
            or judgement.judge_identity != self.spec.visual_judge_identity
        ):
            raise IdentityMismatchError("visual-quality judgement identity differs")
        return (
            judgement.focus_score,
            judgement.grounding_score,
            None,
            judgement.usage,
        )

    def _finish_score(
        self,
        *,
        trajectory: TrajectoryRecord,
        context: RewardContext,
        label: TGVFToolUtilityLabelBinding | None,
        verification: AnswerVerificationResult,
        visual: tuple[
            QualityJudgeScore | None,
            QualityJudgeScore | None,
            str | None,
            JudgeUsage | None,
        ],
    ) -> Stage3VerlTrajectoryReward:
        (
            focus_score,
            grounding_score,
            quality_judge_failure,
            visual_judge_usage,
        ) = visual

        protocol_errors = tuple(
            dict.fromkeys(
                (
                    *(("protocol_invalid",) if not context.protocol_valid else ()),
                    *context.tool_error_codes,
                )
            )
        )
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
        reward = Stage3VerlTrajectoryReward(
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
        return reward


__all__ = [
    "STAGE3_VERL_QUALITY_APPLICABLE_FIELD",
    "STAGE3_VERL_QUALITY_COVERED_FIELD",
    "STAGE3_VERL_QUALITY_FAILURE_FIELD",
    "STAGE3_VERL_REWARD_BRIDGE_SCHEMA_VERSION",
    "STAGE3_VERL_TOOL_LABEL_CONFIDENCE_FIELD",
    "STAGE3_VERL_TOOL_LABEL_FIELD",
    "STAGE3_VERL_TOOL_LABEL_ROW_SHA256_FIELD",
    "STAGE3_VERL_TOOL_SIDECAR_SHA256_FIELD",
    "STAGE3_VERL_VISUAL_JUDGE_USAGE_FIELD",
    "Stage3ShapedRewardSpec",
    "Stage3VerlTrajectoryReward",
    "Stage3VerlTrajectoryRewardScorer",
    "Stage3VisualJudgeSampleFailure",
    "Stage3VisualQualityJudge",
    "Stage3VisualQualityJudgement",
]
