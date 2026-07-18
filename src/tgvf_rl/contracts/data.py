"""Data identities without selecting a production dataset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceImageRef:
    uri: str
    sha256: str


@dataclass(frozen=True, slots=True)
class CanonicalSample:
    sample_id: str
    image: SourceImageRef
    question: str
    expected_answer: str | None
    split: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.question or not self.split:
            raise ValueError("sample identity, question, and split must be non-empty")


@dataclass(frozen=True, slots=True)
class DataManifestIdentity:
    name: str
    version: str
    sha256: str
    sample_count: int

    def __post_init__(self) -> None:
        if not self.name or not self.version or self.sample_count <= 0:
            raise ValueError("data manifest identity must be complete")


@dataclass(frozen=True, slots=True)
class GroupIdentity:
    group_id: str
    sample_id: str
    rollout_count: int

    def __post_init__(self) -> None:
        if self.rollout_count <= 1:
            raise ValueError("an RL group must contain more than one rollout")
