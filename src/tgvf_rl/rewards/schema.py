"""Typed reward and verifier contracts for policy-RL trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Protocol

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.judges.base import JudgeUsage


PILOT_REWARD_LEGACY_WEIGHTS = (0.8, 0.2, 1.2)
PILOT_REWARD_ANSWER_PRIMARY_WEIGHTS = (0.8, 0.2, 0.2)
PILOT_REWARD_DEEPEYES_MATH_WEIGHTS = (1.2, 0.4, 0.0)
PILOT_REWARD_EQUATION_LEGACY = "pilot-legacy-v1"
PILOT_REWARD_EQUATION_ANSWER_PRIMARY = "pilot-answer-primary-v1"
PILOT_REWARD_EQUATION_DEEPEYES_VISUAL = "deepeyes-visual-v1"
PILOT_REWARD_EQUATION_DEEPEYES_MATH = "deepeyes-math-v1"
PILOT_REWARD_WEIGHT_PROFILES = MappingProxyType(
    {
        "legacy": PILOT_REWARD_LEGACY_WEIGHTS,
        "answer-primary": PILOT_REWARD_ANSWER_PRIMARY_WEIGHTS,
    }
)
PILOT_REWARD_WEIGHTS_BY_EQUATION = MappingProxyType(
    {
        PILOT_REWARD_EQUATION_LEGACY: PILOT_REWARD_LEGACY_WEIGHTS,
        PILOT_REWARD_EQUATION_ANSWER_PRIMARY: PILOT_REWARD_ANSWER_PRIMARY_WEIGHTS,
        PILOT_REWARD_EQUATION_DEEPEYES_VISUAL: PILOT_REWARD_LEGACY_WEIGHTS,
        PILOT_REWARD_EQUATION_DEEPEYES_MATH: PILOT_REWARD_DEEPEYES_MATH_WEIGHTS,
    }
)

_DEEPEYES_VISUAL_DATA_SOURCES = frozenset({"vstar", "vl_agent", "chart", "arxivqa"})
_DEEPEYES_MATH_DATA_SOURCES = frozenset({"thinklite", "thinklite_eureka", "xince"})


def deepeyes_reward_equation_for_data_source(
    data_source: object,
) -> tuple[str, tuple[float, float, float]]:
    """Resolve the executable DeepEyes equation from its exact dataset source."""

    if data_source in _DEEPEYES_VISUAL_DATA_SOURCES:
        route = PILOT_REWARD_EQUATION_DEEPEYES_VISUAL
    elif data_source in _DEEPEYES_MATH_DATA_SOURCES:
        route = PILOT_REWARD_EQUATION_DEEPEYES_MATH
    else:
        raise ValueError(
            "DeepEyes source-aware reward received an unsupported data_source: "
            f"{data_source!r}"
        )
    return route, PILOT_REWARD_WEIGHTS_BY_EQUATION[route]


def pilot_reward_weight_profile_name(weights: tuple[float, float, float], /) -> str:
    """Return the exact accepted Pilot reward profile for ``weights``."""

    for name, accepted in PILOT_REWARD_WEIGHT_PROFILES.items():
        if weights == accepted:
            return name
    raise ValueError(
        "Policy Pilot reward weights must match an accepted profile: "
        "legacy=0.8/0.2/1.2 or answer-primary=0.8/0.2/0.2"
    )


class AnswerTaskKind(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    MATH = "math"
    OPEN_VQA = "open_vqa"


@dataclass(frozen=True, slots=True)
class RewardContext:
    sample_id: str
    question: str
    candidate_answer: str
    expected_answer: str | None
    tool_call_count: int
    task_kind: AnswerTaskKind = AnswerTaskKind.OPEN_VQA
    protocol_valid: bool = True
    has_valid_final_answer: bool = True
    successful_tgvf_observation_count: int = 0
    tool_error_codes: tuple[str, ...] = ()
    data_source: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.question:
            raise ValueError("reward sample identity and question must be non-empty")
        if not isinstance(self.task_kind, AnswerTaskKind):
            raise TypeError("task_kind must be AnswerTaskKind")
        if self.tool_call_count < 0 or self.successful_tgvf_observation_count < 0:
            raise ValueError("tool counts must be non-negative")
        if self.successful_tgvf_observation_count > self.tool_call_count:
            raise ValueError("successful TGVF observations cannot exceed tool calls")
        if any(not code for code in self.tool_error_codes):
            raise ValueError("tool error codes must be non-empty")
        if self.data_source is not None and (
            not isinstance(self.data_source, str) or not self.data_source.strip()
        ):
            raise ValueError("reward data_source must be non-empty text when present")


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    strip: bool
    casefold: bool
    collapse_whitespace: bool


@dataclass(frozen=True, slots=True)
class RewardComponentSpec:
    name: str
    weight: float
    verifier_identity: ArtifactIdentity
    minimum_score: float = 0.0
    maximum_score: float = 1.0

    def __post_init__(self) -> None:
        values = (self.weight, self.minimum_score, self.maximum_score)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("reward component values must be finite")
        if self.minimum_score > self.maximum_score:
            raise ValueError("reward component score bounds are reversed")


@dataclass(frozen=True, slots=True)
class RewardPipelineSpec:
    identity: ArtifactIdentity
    components: tuple[RewardComponentSpec, ...]

    def __post_init__(self) -> None:
        names = tuple(component.name for component in self.components)
        if not names or len(names) != len(set(names)):
            raise ValueError("reward components must be non-empty and unique")
        if any(not component.name for component in self.components):
            raise ValueError("reward component names must be non-empty")


@dataclass(frozen=True, slots=True)
class RewardComponentResult:
    name: str
    raw_score: float
    weighted_score: float
    verifier_identity: ArtifactIdentity
    evidence: str


@dataclass(frozen=True, slots=True)
class RewardResult:
    total: float
    components: tuple[RewardComponentResult, ...]
    pipeline_identity: ArtifactIdentity
    answer_verification: AnswerVerificationResult | None = None


@dataclass(frozen=True, slots=True)
class AnswerVerificationResult:
    correct: bool
    route: str
    evidence: str
    verifier_identity: ArtifactIdentity
    judge_usage: JudgeUsage | None = None

    def __post_init__(self) -> None:
        if not self.route or not self.evidence:
            raise ValueError("answer verification route/evidence must be non-empty")
        if self.judge_usage is not None and not isinstance(
            self.judge_usage, JudgeUsage
        ):
            raise TypeError("answer verification judge_usage has the wrong type")


@dataclass(frozen=True, slots=True)
class PilotRewardSpec:
    """Accepted Policy Pilot scalar reward contract."""

    pipeline_identity: ArtifactIdentity
    answer_verifier_identity: ArtifactIdentity
    format_verifier_identity: ArtifactIdentity
    tool_verifier_identity: ArtifactIdentity
    answer_weight: float = 0.8
    format_weight: float = 0.2
    conditional_tool_weight: float = 1.2
    deepeyes_source_aware: bool = False

    def __post_init__(self) -> None:
        if type(self.deepeyes_source_aware) is not bool:
            raise TypeError("deepeyes_source_aware must be bool")
        pilot_reward_weight_profile_name(
            (
                self.answer_weight,
                self.format_weight,
                self.conditional_tool_weight,
            )
        )

    @property
    def weights(self) -> tuple[float, float, float]:
        return (
            self.answer_weight,
            self.format_weight,
            self.conditional_tool_weight,
        )

    @property
    def weight_profile_name(self) -> str:
        return pilot_reward_weight_profile_name(self.weights)

    def weights_for_context(self, context: RewardContext) -> tuple[float, float, float]:
        return self.equation_for_context(context)[1]

    def equation_for_context(
        self, context: RewardContext
    ) -> tuple[str, tuple[float, float, float]]:
        if not isinstance(context, RewardContext):
            raise TypeError("context must be RewardContext")
        if not self.deepeyes_source_aware:
            route = {
                "legacy": PILOT_REWARD_EQUATION_LEGACY,
                "answer-primary": PILOT_REWARD_EQUATION_ANSWER_PRIMARY,
            }[self.weight_profile_name]
            return route, self.weights

        return deepeyes_reward_equation_for_data_source(context.data_source)


class RewardComponent(Protocol):
    def score(self, context: RewardContext) -> tuple[float, str]: ...


class AnswerVerifier(Protocol):
    def verify(self, context: RewardContext) -> AnswerVerificationResult: ...
