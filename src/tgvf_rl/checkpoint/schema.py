"""Checkpoint manifest types kept separate from distributed I/O."""

from __future__ import annotations

from dataclasses import dataclass

from tgvf_rl.contracts.identity import CodeIdentity


@dataclass(frozen=True, slots=True)
class CheckpointSection:
    name: str
    version: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class ProjectCheckpointManifest:
    schema_version: str
    run_id: str
    optimizer_step: int
    code: CodeIdentity
    sections: tuple[CheckpointSection, ...]
    rollout_policy_version: str
    sampling_backend: str

    def __post_init__(self) -> None:
        if (
            not self.schema_version
            or not self.run_id
            or not self.rollout_policy_version
        ):
            raise ValueError("checkpoint identity fields must be non-empty")
        if self.optimizer_step < 0:
            raise ValueError("optimizer_step must be non-negative")
        if self.sampling_backend.lower() != "vllm":
            raise ValueError("checkpoint sampling backend must be vLLM")
        names = tuple(section.name for section in self.sections)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("checkpoint sections must be unique and sorted")


@dataclass(frozen=True, slots=True)
class CheckpointBundle:
    manifest: ProjectCheckpointManifest
    state: dict[str, object]


@dataclass(frozen=True, slots=True)
class ResumeValidationResult:
    exact: bool
    validated_sections: tuple[str, ...]
    next_optimizer_step: int
