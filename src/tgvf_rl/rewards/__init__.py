"""Versioned, decomposed reward pipeline."""

from .pipeline import ExactTextVerifier, PilotRewardPipeline, RewardPipeline
from .context import reward_context_from_trajectory
from .schema import (
    AnswerTaskKind,
    AnswerVerificationResult,
    PilotRewardSpec,
    RewardContext,
    RewardPipelineSpec,
    RewardResult,
)
from .verifiers import RuleFirstAnswerVerifier
from .verl_adapter import (
    PILOT_VERL_ANSWER_ROUTE_FIELD,
    PILOT_VERL_JUDGE_USAGE_FIELD,
    PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD,
    PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION,
    PILOT_VERL_REWARD_COMPONENTS_FIELD,
    PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD,
    PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD,
    PilotRewardContextProvider,
    PilotVerlTrajectoryReward,
    PilotVerlTrajectoryRewardScorer,
)

__all__ = [
    "AnswerTaskKind",
    "AnswerVerificationResult",
    "ExactTextVerifier",
    "PilotRewardPipeline",
    "PilotRewardSpec",
    "PilotRewardContextProvider",
    "PilotVerlTrajectoryReward",
    "PilotVerlTrajectoryRewardScorer",
    "PILOT_VERL_REWARD_BRIDGE_SCHEMA_FIELD",
    "PILOT_VERL_ANSWER_ROUTE_FIELD",
    "PILOT_VERL_JUDGE_USAGE_FIELD",
    "PILOT_VERL_REWARD_BRIDGE_SCHEMA_VERSION",
    "PILOT_VERL_REWARD_COMPONENTS_FIELD",
    "PILOT_VERL_REWARD_PIPELINE_SHA256_FIELD",
    "PILOT_VERL_REWARD_TRAJECTORY_ID_FIELD",
    "RewardContext",
    "RewardPipeline",
    "RewardPipelineSpec",
    "RewardResult",
    "RuleFirstAnswerVerifier",
    "reward_context_from_trajectory",
]
