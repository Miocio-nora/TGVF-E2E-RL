"""Typed reward and verifier contracts for policy-RL trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

from tgvf_rl.contracts.identity import ArtifactIdentity


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


@dataclass(frozen=True, slots=True)
class AnswerVerificationResult:
    correct: bool
    route: str
    evidence: str
    verifier_identity: ArtifactIdentity

    def __post_init__(self) -> None:
        if not self.route or not self.evidence:
            raise ValueError("answer verification route/evidence must be non-empty")


@dataclass(frozen=True, slots=True)
class PilotRewardSpec:
    """Accepted Policy Pilot v1 scalar reward contract."""

    pipeline_identity: ArtifactIdentity
    answer_verifier_identity: ArtifactIdentity
    format_verifier_identity: ArtifactIdentity
    tool_verifier_identity: ArtifactIdentity
    answer_weight: float = 0.8
    format_weight: float = 0.2
    conditional_tool_weight: float = 1.2

    def __post_init__(self) -> None:
        if (
            self.answer_weight,
            self.format_weight,
            self.conditional_tool_weight,
        ) != (0.8, 0.2, 1.2):
            raise ValueError("Policy Pilot v1 reward weights must be 0.8/0.2/1.2")


class RewardComponent(Protocol):
    def score(self, context: RewardContext) -> tuple[float, str]: ...


class AnswerVerifier(Protocol):
    def verify(self, context: RewardContext) -> AnswerVerificationResult: ...
