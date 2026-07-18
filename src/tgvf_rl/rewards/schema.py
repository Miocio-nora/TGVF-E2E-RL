"""Reward contracts; no real coefficients or dataset are selected here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tgvf_rl.contracts.identity import ArtifactIdentity


@dataclass(frozen=True, slots=True)
class RewardContext:
    sample_id: str
    question: str
    candidate_answer: str
    expected_answer: str | None
    tool_call_count: int


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


class RewardComponent(Protocol):
    def score(self, context: RewardContext) -> tuple[float, str]: ...
