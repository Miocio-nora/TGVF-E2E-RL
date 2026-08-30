"""Artifact manifests and payload envelopes for representation checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tgvf_rl.objectives.base import spec_identity_sha256
from tgvf_rl.public_api_compat import rebind_public_class, rebind_public_function

from .checkpoint_identity import RepresentationRunIdentity, _validate_run_identity
from .checkpoint_integrity import (
    _non_empty_text,
    _non_negative_int,
    _sha256,
    _validate_tensor_manifest_contract,
)


REPRESENTATION_ADAPTER_ARTIFACT_SCHEMA_VERSION = "representation-adapter-artifact-v2"
REPRESENTATION_TRAINING_CHECKPOINT_SCHEMA_VERSION = (
    "representation-training-checkpoint-v2"
)
REPRESENTATION_RNG_STATE_SCHEMA_VERSION = "representation-rng-state-v2"


@dataclass(frozen=True, slots=True)
class RepresentationTensorManifestEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        _non_empty_text(self.name, field_name="tensor name")
        if not self.shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.shape
        ):
            raise ValueError("tensor shape must contain non-negative integer sizes")
        _non_empty_text(self.dtype, field_name="tensor dtype")
        _sha256(self.tensor_sha256, field_name="tensor_sha256")


@dataclass(frozen=True, slots=True)
class RepresentationAdapterArtifactManifest:
    run_identity: RepresentationRunIdentity
    run_identity_sha256: str
    global_step: int
    adapter_state_sha256: str
    tensors: tuple[RepresentationTensorManifestEntry, ...]
    schema_version: str = REPRESENTATION_ADAPTER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_run_identity(self.run_identity)
        _sha256(self.run_identity_sha256, field_name="run_identity_sha256")
        if self.run_identity_sha256 != self.run_identity.identity_sha256:
            raise ValueError("artifact run identity digest mismatch")
        _non_negative_int(self.global_step, field_name="global_step")
        _sha256(self.adapter_state_sha256, field_name="adapter_state_sha256")
        _validate_tensor_manifest(self.tensors)
        if self.schema_version != REPRESENTATION_ADAPTER_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("representation Adapter artifact schema mismatch")

    @property
    def artifact_identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class RepresentationAdapterArtifact:
    manifest: RepresentationAdapterArtifactManifest
    adapter_state: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class RepresentationTrainingCheckpointManifest:
    run_identity: RepresentationRunIdentity
    run_identity_sha256: str
    global_step: int
    accumulation_microstep: int
    adapter_state_sha256: str
    adapter_tensors: tuple[RepresentationTensorManifestEntry, ...]
    optimizer_type: str
    optimizer_parameter_names_by_group: tuple[tuple[str, ...], ...]
    optimizer_identity_sha256: str
    optimizer_state_sha256: str
    scheduler_type: str | None
    scheduler_identity_sha256: str | None
    scheduler_state_sha256: str | None
    sampler_type: str
    sampler_contract_identity_sha256: str
    sampler_identity_sha256: str
    sampler_state_sha256: str
    trainer_execution_identity_sha256: str
    initialization_identity_sha256: str
    rng_state_sha256: str
    schema_version: str = REPRESENTATION_TRAINING_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_run_identity(self.run_identity)
        _sha256(self.run_identity_sha256, field_name="run_identity_sha256")
        if self.run_identity_sha256 != self.run_identity.identity_sha256:
            raise ValueError("checkpoint run identity digest mismatch")
        _non_negative_int(self.global_step, field_name="global_step")
        if self.accumulation_microstep != 0:
            raise ValueError(
                "representation checkpoints cannot contain partial accumulation"
            )
        _sha256(self.adapter_state_sha256, field_name="adapter_state_sha256")
        _validate_tensor_manifest(self.adapter_tensors)
        _non_empty_text(self.optimizer_type, field_name="optimizer_type")
        if not self.optimizer_parameter_names_by_group or any(
            not group for group in self.optimizer_parameter_names_by_group
        ):
            raise ValueError("optimizer parameter-name groups must be non-empty")
        flattened = tuple(
            name for group in self.optimizer_parameter_names_by_group for name in group
        )
        if any(not isinstance(name, str) or not name for name in flattened):
            raise ValueError("optimizer parameter names must be non-empty strings")
        if len(flattened) != len(set(flattened)):
            raise ValueError("optimizer parameters must occur exactly once")
        _sha256(self.optimizer_identity_sha256, field_name="optimizer_identity_sha256")
        if (
            self.optimizer_identity_sha256
            != self.run_identity.optimizer.identity_sha256
        ):
            raise ValueError("checkpoint optimizer identity digest mismatch")
        _sha256(self.optimizer_state_sha256, field_name="optimizer_state_sha256")
        scheduler_fields = (
            self.scheduler_type,
            self.scheduler_identity_sha256,
            self.scheduler_state_sha256,
        )
        if any(value is None for value in scheduler_fields) != all(
            value is None for value in scheduler_fields
        ):
            raise ValueError("scheduler type, identity, and state presence must align")
        if self.scheduler_type is not None:
            _non_empty_text(self.scheduler_type, field_name="scheduler_type")
            _sha256(
                self.scheduler_identity_sha256,
                field_name="scheduler_identity_sha256",
            )
            _sha256(
                self.scheduler_state_sha256,
                field_name="scheduler_state_sha256",
            )
        if (
            self.scheduler_identity_sha256
            != self.run_identity.scheduler_identity_sha256
        ):
            raise ValueError("checkpoint scheduler identity digest mismatch")
        _non_empty_text(self.sampler_type, field_name="sampler_type")
        _sha256(
            self.sampler_contract_identity_sha256,
            field_name="sampler_contract_identity_sha256",
        )
        if (
            self.sampler_contract_identity_sha256
            != self.run_identity.sampler_contract.identity_sha256
        ):
            raise ValueError("checkpoint sampler contract identity digest mismatch")
        _sha256(self.sampler_identity_sha256, field_name="sampler_identity_sha256")
        _sha256(self.sampler_state_sha256, field_name="sampler_state_sha256")
        _sha256(
            self.trainer_execution_identity_sha256,
            field_name="trainer_execution_identity_sha256",
        )
        if (
            self.trainer_execution_identity_sha256
            != self.run_identity.trainer_execution.identity_sha256
        ):
            raise ValueError("checkpoint trainer execution identity digest mismatch")
        _sha256(
            self.initialization_identity_sha256,
            field_name="initialization_identity_sha256",
        )
        if (
            self.initialization_identity_sha256
            != self.run_identity.initialization.identity_sha256
        ):
            raise ValueError("checkpoint initialization identity digest mismatch")
        _sha256(self.rng_state_sha256, field_name="rng_state_sha256")
        if self.schema_version != REPRESENTATION_TRAINING_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("representation training checkpoint schema mismatch")

    @property
    def checkpoint_identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class RepresentationTrainingCheckpoint:
    manifest: RepresentationTrainingCheckpointManifest
    adapter_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, object]
    scheduler_state: dict[str, object] | None
    sampler_state: dict[str, object]
    rng_state: dict[str, object]


@dataclass(frozen=True, slots=True)
class RepresentationResumeResult:
    global_step: int
    next_global_step: int
    run_identity_sha256: str
    checkpoint_identity_sha256: str
    exact: bool = True


def _validate_tensor_manifest(
    entries: tuple[RepresentationTensorManifestEntry, ...],
) -> None:
    _validate_tensor_manifest_contract(
        entries, entry_type=RepresentationTensorManifestEntry
    )


_CHECKPOINT_SCHEMA_TYPES = (
    RepresentationTensorManifestEntry,
    RepresentationAdapterArtifactManifest,
    RepresentationAdapterArtifact,
    RepresentationTrainingCheckpointManifest,
    RepresentationTrainingCheckpoint,
    RepresentationResumeResult,
)

for _schema_type in _CHECKPOINT_SCHEMA_TYPES:
    rebind_public_class(
        _schema_type,
        implementation_module=__name__,
        public_module="tgvf_rl.representation.training.checkpoint",
    )
del _schema_type

rebind_public_function(
    _validate_tensor_manifest,
    implementation_module=__name__,
    public_module="tgvf_rl.representation.training.checkpoint",
)
