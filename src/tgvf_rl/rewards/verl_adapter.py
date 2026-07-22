"""Fail-closed trajectory reward records for the upstream veRL boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol

from tgvf_rl.trajectories.schema import TrajectoryRecord

from .pipeline import PilotRewardPipeline
from .schema import AnswerTaskKind, RewardContext, RewardResult


PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION = "tgvf-pilot-verl-reward-bridge-v1"
PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD = "tgvf_reward_bridge_schema_version"
PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD = "tgvf_reward_pipeline_sha256"
PILOT_VERL_REWARD_COMPONENTS_FIELD = "tgvf_reward_components"
PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD = "tgvf_reward_trajectory_id"

_COMPONENT_NAMES = (
    "answer_reward",
    "format_reward",
    "conditional_tool_reward",
)
_COMPONENT_WEIGHTS = (0.8, 0.2, 1.2)


class PilotRewardContextProvider(Protocol):
    def build(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
    ) -> RewardContext: ...


@dataclass(frozen=True, slots=True)
class PilotVerlTrajectoryReward:
    """One trajectory reward plus row identities carried through DataProto."""

    trajectory_id: str
    group_uid: str
    rollout_index: int
    context: RewardContext
    result: RewardResult

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("reward trajectory_id must be non-empty")
        if not isinstance(self.group_uid, str) or not self.group_uid:
            raise ValueError("reward group_uid must be non-empty")
        if type(self.rollout_index) is not int or self.rollout_index < 0:
            raise ValueError("reward rollout_index must be non-negative")
        if not isinstance(self.context, RewardContext):
            raise TypeError("reward context must be RewardContext")
        if not isinstance(self.result, RewardResult):
            raise TypeError("reward result must be RewardResult")
        _validate_pilot_reward_result(self.context, self.result)

    @property
    def total(self) -> float:
        return float(self.result.total)

    @property
    def raw_components(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (component.name, float(component.raw_score))
            for component in self.result.components
        )

    @property
    def pipeline_sha256(self) -> str:
        return self.result.pipeline_identity.sha256

    def reward_sidecars(self) -> dict[str, object]:
        """Return JSON/object-array-safe fields flattened by veRL AgentLoop."""

        return {
            PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD: (
                PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION
            ),
            PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD: self.pipeline_sha256,
            PILOT_VERL_REWARD_COMPONENTS_FIELD: self.raw_components,
            PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD: self.trajectory_id,
        }

    def reward_extra_info(self) -> dict[str, object]:
        """Expose decomposed metrics without delegating scoring to veRL."""

        components = dict(self.raw_components)
        return {
            "tgvf_exact_trajectory_reward": self.total,
            "answer_reward": components["answer_reward"],
            "format_reward": components["format_reward"],
            "conditional_tool_reward": components["conditional_tool_reward"],
            "reward_pipeline_sha256": self.pipeline_sha256,
        }


class PilotVerlTrajectoryRewardScorer:
    """Score a completed trajectory before it crosses AgentLoopOutput."""

    def __init__(
        self,
        *,
        pipeline: PilotRewardPipeline,
        context_provider: PilotRewardContextProvider,
        audit_sink: Callable[
            [TrajectoryRecord, PilotVerlTrajectoryReward], None
        ]
        | None = None,
    ) -> None:
        if not isinstance(pipeline, PilotRewardPipeline):
            raise TypeError("pipeline must be PilotRewardPipeline")
        if not callable(getattr(context_provider, "build", None)):
            raise TypeError("context_provider must implement build()")
        if (
            pipeline.spec.answer_weight,
            pipeline.spec.format_weight,
            pipeline.spec.conditional_tool_weight,
        ) != _COMPONENT_WEIGHTS:
            raise ValueError("veRL Pilot reward requires fixed weights 0.8/0.2/1.2")
        self.pipeline = pipeline
        self.context_provider = context_provider
        self.audit_sink = audit_sink

    def score(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
    ) -> PilotVerlTrajectoryReward:
        if not isinstance(trajectory, TrajectoryRecord):
            raise TypeError("trajectory must be TrajectoryRecord")
        request_identity = getattr(request, "identity", None)
        if request_identity != trajectory.identity:
            raise ValueError("reward request and trajectory identities differ")
        context = self.context_provider.build(
            request=request,
            trajectory=trajectory,
        )
        if not isinstance(context, RewardContext):
            raise TypeError("context_provider must return RewardContext")
        if context.sample_id != trajectory.identity.sample_id:
            raise ValueError("reward context sample differs from trajectory")
        result = self.pipeline.score(context)
        if result.pipeline_identity != self.pipeline.spec.pipeline_identity:
            raise ValueError("reward pipeline identity changed while scoring")
        reward = PilotVerlTrajectoryReward(
            trajectory_id=trajectory.identity.canonical_id,
            group_uid=trajectory.identity.group_id,
            rollout_index=trajectory.identity.rollout_index,
            context=context,
            result=result,
        )
        if context.task_kind is AnswerTaskKind.MULTIPLE_CHOICE:
            answer = result.components[0]
            expected_route = (
                "multiple_choice_rule"
                if context.has_valid_final_answer
                else "missing_final_answer"
            )
            if not answer.evidence.startswith(f"route={expected_route};"):
                raise ValueError(
                    "MCQ Pilot reward used an unexpected deterministic rule route"
                )
            if answer.verifier_identity != self.pipeline.spec.answer_verifier_identity:
                raise ValueError("MCQ Pilot reward unexpectedly used a judge identity")
            if not context.has_valid_final_answer and answer.raw_score != 0.0:
                raise ValueError(
                    "unanswered MCQ trajectory must receive zero answer reward"
                )
        if self.audit_sink is not None:
            self.audit_sink(trajectory, reward)
        return reward


def _validate_pilot_reward_result(
    context: RewardContext,
    result: RewardResult,
) -> None:
    if not math.isfinite(result.total):
        raise ValueError("Pilot trajectory reward must be finite")
    components = tuple(result.components)
    names = tuple(component.name for component in components)
    if names != _COMPONENT_NAMES:
        raise ValueError("Pilot reward components differ from the accepted equation")
    raw = tuple(float(component.raw_score) for component in components)
    if raw[0] not in {0.0, 1.0}:
        raise ValueError("answer_reward must be binary")
    if raw[1] not in {-1.0, 0.0}:
        raise ValueError("format_reward must be zero or negative one")
    if raw[2] not in {0.0, 1.0}:
        raise ValueError("conditional_tool_reward must be binary")
    if raw[2] > raw[0]:
        raise ValueError("conditional tool reward requires a correct answer")
    if raw[2] and context.successful_tgvf_observation_count < 1:
        raise ValueError("conditional tool reward requires a successful observation")
    expected_weighted = tuple(
        score * weight for score, weight in zip(raw, _COMPONENT_WEIGHTS, strict=True)
    )
    actual_weighted = tuple(float(component.weighted_score) for component in components)
    for actual, expected in zip(actual_weighted, expected_weighted, strict=True):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Pilot reward component weight differs from 0.8/0.2/1.2")
    expected_total = sum(expected_weighted)
    if not math.isclose(
        float(result.total), expected_total, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("Pilot total reward differs from the accepted equation")


__all__ = [
    "PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD",
    "PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION",
    "PILOT_VERL_REWARD_COMPONENTS_FIELD",
    "PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD",
    "PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD",
    "PilotRewardContextProvider",
    "PilotVerlTrajectoryReward",
    "PilotVerlTrajectoryRewardScorer",
]
