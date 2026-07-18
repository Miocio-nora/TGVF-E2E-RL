"""Explicit identities used across rollout, replay, and checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ComponentRole(str, Enum):
    BEHAVIOR = "behavior"
    PROXIMAL_OLD = "proximal_old"
    CURRENT = "current"
    REFERENCE = "reference"
    TEACHER = "teacher"
    JUDGE = "judge"


class SupportLevel(str, Enum):
    UNSUPPORTED = "unsupported"
    SCHEMA = "schema"
    SYNTHETIC = "synthetic"
    EXECUTABLE = "executable"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    namespace: str
    name: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        if not all((self.namespace, self.name, self.version)):
            raise ValueError("artifact namespace, name, and version must be non-empty")
        _validate_sha256(self.sha256)


@dataclass(frozen=True, slots=True)
class CodeIdentity:
    repository: str
    commit: str
    dirty_state_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.repository or not self.commit:
            raise ValueError("repository and commit must be non-empty")
        if self.dirty_state_sha256 is not None:
            _validate_sha256(self.dirty_state_sha256)


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    family: str
    model_name: str
    revision_or_path: str
    tokenizer_length: int
    chat_template_sha256: str

    def __post_init__(self) -> None:
        if not self.family or not self.model_name or not self.revision_or_path:
            raise ValueError("model identity fields must be non-empty")
        if self.tokenizer_length <= 0:
            raise ValueError("tokenizer_length must be positive")
        _validate_sha256(self.chat_template_sha256)


@dataclass(frozen=True, slots=True)
class ComponentIdentity:
    role: ComponentRole
    artifact: ArtifactIdentity
    code: CodeIdentity


@dataclass(frozen=True, order=True, slots=True)
class PolicyVersion:
    run_id: str
    optimizer_step: int
    weights_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.optimizer_step < 0:
            raise ValueError("optimizer_step must be non-negative")
        _validate_sha256(self.weights_sha256)


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"expected lowercase SHA256, got {value!r}")
