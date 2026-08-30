"""Strict artifacts and resumable checkpoints for the representation phase.

The deployable artifact contains only tensors owned by :class:`TGVFAdapter`.
Frozen Qwen projection ports are bound by identity but are never serialized.
Training checkpoints are deliberately optimizer-step-boundary checkpoints:
there are no partially accumulated gradients whose identity could be lost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass as dataclass
from enum import Enum as Enum
import hashlib as hashlib
import json as json
import math as math
import os as os
from pathlib import Path
import random
import tempfile as tempfile
from typing import Protocol

import torch

# Retained here so get_type_hints() keeps resolving the historical public types.
from tgvf_rl.conditioning.base import (
    TargetConditioningConfig as TargetConditioningConfig,
)
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import (
    CodeIdentity as CodeIdentity,
    ModelIdentity as ModelIdentity,
)
from tgvf_rl.objectives.base import spec_identity_sha256 as spec_identity_sha256
from tgvf_rl.observations.store import tensor_checksum as tensor_checksum
from tgvf_rl.representation.adapter import (
    TGVFAdapter,
    TGVFAdapterVariant as TGVFAdapterVariant,
)

from .checkpoint_integrity import (
    _HEX as _HEX,
    _adapter_state_to_cpu as _adapter_state_to_cpu,
    _finite_ratio as _finite_ratio,
    _integer as _integer,
    _mapping_key as _mapping_key,
    _non_empty_text as _non_empty_text,
    _non_negative_finite_float as _non_negative_finite_float,
    _non_negative_int as _non_negative_int,
    _plain_cpu_state as _plain_cpu_state,
    _positive_finite_float as _positive_finite_float,
    _positive_int as _positive_int,
    _qualified_type as _qualified_type,
    _require_adapter as _require_adapter,
    _runtime_bool as _runtime_bool,
    _runtime_float as _runtime_float,
    _runtime_optional_bool as _runtime_optional_bool,
    _save_atomic as _save_atomic,
    _sha256 as _sha256,
    _state_digest as _state_digest,
    _strictly_increasing_non_negative_ints as _strictly_increasing_non_negative_ints,
    _tensor_checksum as _tensor_checksum,
    _torch_load as _torch_load,
    _update_state_digest as _update_state_digest,
)
from .checkpoint_identity import (
    L_GEN_GLOBAL_REDUCTION as L_GEN_GLOBAL_REDUCTION,
    MATRIX_CE_GLOBAL_REDUCTION as MATRIX_CE_GLOBAL_REDUCTION,
    REPRESENTATION_ACCUMULATION_SCHEMA_VERSION as REPRESENTATION_ACCUMULATION_SCHEMA_VERSION,
    REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2 as REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2,
    REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION as REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION,
    REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION_V2 as REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION_V2,
    REPRESENTATION_INITIALIZATION_SCHEMA_VERSION as REPRESENTATION_INITIALIZATION_SCHEMA_VERSION,
    REPRESENTATION_OPTIMIZER_IDENTITY_SCHEMA_VERSION as REPRESENTATION_OPTIMIZER_IDENTITY_SCHEMA_VERSION,
    REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION as REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION,
    REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V2 as REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V2,
    REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3 as REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3,
    REPRESENTATION_SAMPLER_CONTRACT_SCHEMA_VERSION as REPRESENTATION_SAMPLER_CONTRACT_SCHEMA_VERSION,
    REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION as REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION,
    REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION_V2 as REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION_V2,
    REPRESENTATION_TRAINER_EXECUTION_SCHEMA_VERSION as REPRESENTATION_TRAINER_EXECUTION_SCHEMA_VERSION,
    RepresentationAccumulationIdentity,
    RepresentationAccumulationIdentityV2 as RepresentationAccumulationIdentityV2,
    RepresentationAdapterContractIdentity as RepresentationAdapterContractIdentity,
    RepresentationAdapterContractIdentityV2 as RepresentationAdapterContractIdentityV2,
    RepresentationInitializationIdentity as RepresentationInitializationIdentity,
    RepresentationOptimizerIdentity,
    RepresentationRunIdentity,
    RepresentationRunIdentityV3 as RepresentationRunIdentityV3,
    RepresentationSamplerContractIdentity,
    RepresentationSchedulerIdentity,
    RepresentationSchedulerIdentityV2 as RepresentationSchedulerIdentityV2,
    RepresentationTrainerExecutionIdentity,
    _validate_accumulation_identity as _validate_accumulation_identity,
    _validate_run_identity as _validate_run_identity,
    representation_adapter_contract_identity as representation_adapter_contract_identity,
)
from .checkpoint_schema import (
    REPRESENTATION_ADAPTER_ARTIFACT_SCHEMA_VERSION as REPRESENTATION_ADAPTER_ARTIFACT_SCHEMA_VERSION,
    REPRESENTATION_RNG_STATE_SCHEMA_VERSION,
    REPRESENTATION_TRAINING_CHECKPOINT_SCHEMA_VERSION as REPRESENTATION_TRAINING_CHECKPOINT_SCHEMA_VERSION,
    RepresentationAdapterArtifact,
    RepresentationAdapterArtifactManifest,
    RepresentationResumeResult,
    RepresentationTensorManifestEntry,
    RepresentationTrainingCheckpoint,
    RepresentationTrainingCheckpointManifest,
    _validate_tensor_manifest as _validate_tensor_manifest,
)
from .objective import RepresentationObjectiveConfig as RepresentationObjectiveConfig
from .sampling import (
    SAMPLER_IDENTITY_SCHEMA_VERSION as SAMPLER_IDENTITY_SCHEMA_VERSION,
    SAMPLER_STATE_SCHEMA_VERSION as SAMPLER_STATE_SCHEMA_VERSION,
    SameImageBatchSampler,
)
from .validation_identity import (
    RepresentationValidationDataIdentity as RepresentationValidationDataIdentity,
)


_BORROWED_QWEN_PREFIXES = (
    "main_projection.",
    "d_deepstack_projections.",
)


class _Stateful(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> object: ...


def save_representation_adapter_artifact_atomic(
    path: str | Path,
    *,
    adapter: TGVFAdapter,
    run_identity: RepresentationRunIdentity,
    global_step: int,
) -> RepresentationAdapterArtifactManifest:
    """Atomically write a deployable Adapter-only artifact."""

    _validate_runtime_identity(run_identity, adapter=adapter)
    _non_negative_int(global_step, field_name="global_step")
    _validate_initial_state_at_step_zero(run_identity, adapter, global_step)
    adapter_state = _adapter_state_to_cpu(adapter)
    manifest = RepresentationAdapterArtifactManifest(
        run_identity=run_identity,
        run_identity_sha256=run_identity.identity_sha256,
        global_step=global_step,
        adapter_state_sha256=_state_digest(adapter_state),
        tensors=_tensor_manifest(adapter_state),
    )
    artifact = RepresentationAdapterArtifact(manifest, adapter_state)
    _validate_adapter_artifact(artifact)
    _save_atomic(artifact, path)
    return manifest


def load_representation_adapter_artifact(
    path: str | Path,
) -> RepresentationAdapterArtifact:
    value = _torch_load(path)
    if not isinstance(value, RepresentationAdapterArtifact):
        raise ReplayMismatchError("file is not a representation Adapter artifact")
    _validate_adapter_artifact(value)
    return value


def restore_representation_adapter_artifact(
    path: str | Path,
    *,
    adapter: TGVFAdapter,
    expected_run_identity: RepresentationRunIdentity,
) -> RepresentationAdapterArtifactManifest:
    """Strictly restore owned tensors while leaving Qwen projection state intact."""

    _validate_runtime_identity(expected_run_identity, adapter=adapter)
    artifact = load_representation_adapter_artifact(path)
    _assert_same_run_identity(artifact.manifest.run_identity, expected_run_identity)
    adapter.load_artifact_state_dict(artifact.adapter_state)
    return artifact.manifest


def save_representation_training_checkpoint_atomic(
    path: str | Path,
    *,
    adapter: TGVFAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: _Stateful | None,
    sampler: SameImageBatchSampler,
    run_identity: RepresentationRunIdentity,
    accumulation: RepresentationAccumulationIdentity,
    trainer_execution: RepresentationTrainerExecutionIdentity,
    global_step: int,
) -> RepresentationTrainingCheckpointManifest:
    """Save exact training state at an optimizer-step boundary."""

    _validate_runtime_identity(
        run_identity,
        adapter=adapter,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulation=accumulation,
        trainer_execution=trainer_execution,
    )
    _non_negative_int(global_step, field_name="global_step")
    _validate_initial_state_at_step_zero(run_identity, adapter, global_step)
    parameter_names = _optimizer_parameter_names_by_group(adapter, optimizer)
    _validate_scheduler(scheduler, optimizer=optimizer)

    adapter_state = _adapter_state_to_cpu(adapter)
    optimizer_state = _plain_cpu_state(optimizer.state_dict())
    scheduler_state = (
        None if scheduler is None else _plain_cpu_state(scheduler.state_dict())
    )
    sampler_state = _plain_cpu_state(sampler.state_dict())
    rng_state = capture_representation_rng_state()
    if not isinstance(optimizer_state, dict):
        raise RuntimeError("optimizer state_dict must be a mapping")
    if scheduler_state is not None and not isinstance(scheduler_state, dict):
        raise RuntimeError("scheduler state_dict must be a mapping")
    if not isinstance(sampler_state, dict):
        raise RuntimeError("sampler state_dict must be a mapping")

    manifest = RepresentationTrainingCheckpointManifest(
        run_identity=run_identity,
        run_identity_sha256=run_identity.identity_sha256,
        global_step=global_step,
        accumulation_microstep=0,
        adapter_state_sha256=_state_digest(adapter_state),
        adapter_tensors=_tensor_manifest(adapter_state),
        optimizer_type=_qualified_type(optimizer),
        optimizer_parameter_names_by_group=parameter_names,
        optimizer_identity_sha256=run_identity.optimizer.identity_sha256,
        optimizer_state_sha256=_state_digest(optimizer_state),
        scheduler_type=None if scheduler is None else _qualified_type(scheduler),
        scheduler_identity_sha256=run_identity.scheduler_identity_sha256,
        scheduler_state_sha256=(
            None if scheduler_state is None else _state_digest(scheduler_state)
        ),
        sampler_type=_qualified_type(sampler),
        sampler_contract_identity_sha256=(
            run_identity.sampler_contract.identity_sha256
        ),
        sampler_identity_sha256=sampler.identity_sha256,
        sampler_state_sha256=_state_digest(sampler_state),
        trainer_execution_identity_sha256=(
            run_identity.trainer_execution.identity_sha256
        ),
        initialization_identity_sha256=run_identity.initialization.identity_sha256,
        rng_state_sha256=_state_digest(rng_state),
    )
    checkpoint = RepresentationTrainingCheckpoint(
        manifest=manifest,
        adapter_state=adapter_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        sampler_state=sampler_state,
        rng_state=rng_state,
    )
    _validate_training_checkpoint(checkpoint)
    _save_atomic(checkpoint, path)
    return manifest


def load_representation_training_checkpoint(
    path: str | Path,
) -> RepresentationTrainingCheckpoint:
    value = _torch_load(path)
    if not isinstance(value, RepresentationTrainingCheckpoint):
        raise ReplayMismatchError("file is not a representation training checkpoint")
    _validate_training_checkpoint(value)
    return value


def restore_representation_training_checkpoint(
    path: str | Path,
    *,
    adapter: TGVFAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: _Stateful | None,
    sampler: SameImageBatchSampler,
    expected_run_identity: RepresentationRunIdentity,
    accumulation: RepresentationAccumulationIdentity,
    trainer_execution: RepresentationTrainerExecutionIdentity,
) -> RepresentationResumeResult:
    """Validate all state before restoring, rolling back on an apply failure."""

    _validate_runtime_identity(
        expected_run_identity,
        adapter=adapter,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulation=accumulation,
        trainer_execution=trainer_execution,
    )
    checkpoint = load_representation_training_checkpoint(path)
    manifest = checkpoint.manifest
    _assert_same_run_identity(manifest.run_identity, expected_run_identity)
    if manifest.optimizer_type != _qualified_type(optimizer):
        raise IdentityMismatchError("optimizer type differs from checkpoint")
    current_names = _optimizer_parameter_names_by_group(adapter, optimizer)
    if current_names != manifest.optimizer_parameter_names_by_group:
        raise IdentityMismatchError(
            "optimizer parameter-name topology differs from checkpoint"
        )
    _validate_optimizer_group_shape(optimizer, checkpoint.optimizer_state)
    _validate_scheduler(scheduler, optimizer=optimizer)
    actual_scheduler_type = None if scheduler is None else _qualified_type(scheduler)
    if actual_scheduler_type != manifest.scheduler_type:
        raise IdentityMismatchError("scheduler presence/type differs from checkpoint")
    if manifest.sampler_type != _qualified_type(sampler):
        raise IdentityMismatchError("sampler type differs from checkpoint")
    if manifest.sampler_identity_sha256 != sampler.identity_sha256:
        raise IdentityMismatchError("sampler identity differs from checkpoint")

    rollback_adapter = _adapter_state_to_cpu(adapter)
    rollback_optimizer = deepcopy(optimizer.state_dict())
    rollback_scheduler = None if scheduler is None else deepcopy(scheduler.state_dict())
    rollback_sampler = deepcopy(sampler.state_dict())
    rollback_rng = capture_representation_rng_state()
    try:
        adapter.load_artifact_state_dict(checkpoint.adapter_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        if scheduler is not None:
            if checkpoint.scheduler_state is None:
                raise ReplayMismatchError("checkpoint scheduler state is missing")
            scheduler.load_state_dict(checkpoint.scheduler_state)
        sampler.load_state_dict(checkpoint.sampler_state)
        restore_representation_rng_state(checkpoint.rng_state)
    except Exception:
        adapter.load_artifact_state_dict(rollback_adapter)
        optimizer.load_state_dict(rollback_optimizer)
        if scheduler is not None and rollback_scheduler is not None:
            scheduler.load_state_dict(rollback_scheduler)
        sampler.load_state_dict(rollback_sampler)
        restore_representation_rng_state(rollback_rng)
        raise

    return RepresentationResumeResult(
        global_step=manifest.global_step,
        next_global_step=manifest.global_step + 1,
        run_identity_sha256=manifest.run_identity_sha256,
        checkpoint_identity_sha256=manifest.checkpoint_identity_sha256,
    )


def capture_representation_rng_state() -> dict[str, object]:
    """Capture Python, CPU torch, and the current local CUDA generator."""

    state: dict[str, object] = {
        "schema_version": REPRESENTATION_RNG_STATE_SCHEMA_VERSION,
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state().cpu().clone(),
    }
    # A CPU-only save must not initialize CUDA merely to inspect its generators.
    if torch.cuda.is_initialized():
        device_index = torch.cuda.current_device()
        visible_device_count = torch.cuda.device_count()
        if device_index < 0 or device_index >= visible_device_count:
            raise RuntimeError(
                "current CUDA device index is outside the visible device topology"
            )
        state["torch_cuda"] = {
            "device_index": device_index,
            "visible_device_count": visible_device_count,
            "state": torch.cuda.get_rng_state(device_index).cpu().clone(),
        }
    return state


def restore_representation_rng_state(state: Mapping[str, object]) -> None:
    """Strictly restore the represented CPU and current local CUDA generators."""

    _validate_rng_state(state)
    python_state = state["python"]
    cpu_state = state["torch_cpu"]
    random.setstate(python_state)  # type: ignore[arg-type]
    torch.random.set_rng_state(cpu_state)  # type: ignore[arg-type]
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        torch.cuda.set_rng_state(
            cuda_state["state"],  # type: ignore[index]
            device=cuda_state["device_index"],  # type: ignore[index]
        )


def _validate_adapter_artifact(artifact: RepresentationAdapterArtifact) -> None:
    if not isinstance(artifact.manifest, RepresentationAdapterArtifactManifest):
        raise ReplayMismatchError("malformed representation artifact manifest")
    _validate_run_identity(artifact.manifest.run_identity)
    artifact.manifest.__post_init__()
    state = _validate_adapter_state(artifact.adapter_state)
    if _state_digest(state) != artifact.manifest.adapter_state_sha256:
        raise ReplayMismatchError("representation artifact state digest mismatch")
    if _tensor_manifest(state) != artifact.manifest.tensors:
        raise ReplayMismatchError("representation artifact tensor manifest mismatch")


def _validate_training_checkpoint(
    checkpoint: RepresentationTrainingCheckpoint,
) -> None:
    manifest = checkpoint.manifest
    if not isinstance(manifest, RepresentationTrainingCheckpointManifest):
        raise ReplayMismatchError("malformed representation checkpoint manifest")
    _validate_run_identity(manifest.run_identity)
    manifest.__post_init__()
    adapter_state = _validate_adapter_state(checkpoint.adapter_state)
    if _state_digest(adapter_state) != manifest.adapter_state_sha256:
        raise ReplayMismatchError("checkpoint Adapter state digest mismatch")
    if _tensor_manifest(adapter_state) != manifest.adapter_tensors:
        raise ReplayMismatchError("checkpoint Adapter tensor manifest mismatch")
    if not isinstance(checkpoint.optimizer_state, Mapping):
        raise ReplayMismatchError("checkpoint optimizer state is not a mapping")
    if _state_digest(checkpoint.optimizer_state) != manifest.optimizer_state_sha256:
        raise ReplayMismatchError("checkpoint optimizer state digest mismatch")
    if (checkpoint.scheduler_state is None) != (manifest.scheduler_type is None):
        raise ReplayMismatchError("checkpoint scheduler presence mismatch")
    if checkpoint.scheduler_state is not None:
        if not isinstance(checkpoint.scheduler_state, Mapping):
            raise ReplayMismatchError("checkpoint scheduler state is not a mapping")
        if _state_digest(checkpoint.scheduler_state) != manifest.scheduler_state_sha256:
            raise ReplayMismatchError("checkpoint scheduler state digest mismatch")
    _validate_global_step_state(
        manifest.global_step,
        optimizer_state=checkpoint.optimizer_state,
        scheduler_state=checkpoint.scheduler_state,
    )
    if not isinstance(checkpoint.sampler_state, Mapping):
        raise ReplayMismatchError("checkpoint sampler state is not a mapping")
    if _state_digest(checkpoint.sampler_state) != manifest.sampler_state_sha256:
        raise ReplayMismatchError("checkpoint sampler state digest mismatch")
    if not isinstance(checkpoint.rng_state, Mapping):
        raise ReplayMismatchError("checkpoint RNG state is not a mapping")
    _validate_rng_state(checkpoint.rng_state)
    if _state_digest(checkpoint.rng_state) != manifest.rng_state_sha256:
        raise ReplayMismatchError("checkpoint RNG state digest mismatch")


def _validate_runtime_identity(
    identity: RepresentationRunIdentity,
    *,
    adapter: TGVFAdapter,
    sampler: SameImageBatchSampler | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: _Stateful | None = None,
    accumulation: RepresentationAccumulationIdentity | None = None,
    trainer_execution: RepresentationTrainerExecutionIdentity | None = None,
) -> None:
    _validate_run_identity(identity)
    identity.adapter_contract.assert_matches(adapter)
    if sampler is not None:
        if not isinstance(sampler, SameImageBatchSampler):
            raise TypeError("sampler must be a SameImageBatchSampler")
        if sampler.data_manifest_sha256 != identity.data_manifest_sha256:
            raise IdentityMismatchError(
                "sampler data manifest differs from representation run identity"
            )
        if sampler.world_size != identity.accumulation.data_parallel_world_size:
            raise IdentityMismatchError(
                "sampler world size differs from accumulation identity"
            )
        if (
            RepresentationSamplerContractIdentity.from_sampler(sampler)
            != identity.sampler_contract
        ):
            raise IdentityMismatchError(
                "same-image sampler contract differs from run identity"
            )
    if optimizer is not None:
        if (
            RepresentationOptimizerIdentity.from_optimizer(optimizer)
            != identity.optimizer
        ):
            raise IdentityMismatchError(
                "optimizer hyperparameters differ from representation run identity"
            )
        _validate_scheduler(scheduler, optimizer=optimizer)
        _validate_scheduler_identity(scheduler, identity.scheduler)
    elif scheduler is not None:
        raise ValueError("scheduler validation requires its optimizer")
    if accumulation is not None:
        _validate_accumulation_identity(accumulation)
        if accumulation != identity.accumulation:
            raise IdentityMismatchError(
                "runtime accumulation differs from representation run identity"
            )
    if trainer_execution is not None:
        if not isinstance(trainer_execution, RepresentationTrainerExecutionIdentity):
            raise TypeError(
                "trainer_execution must be a RepresentationTrainerExecutionIdentity"
            )
        if trainer_execution != identity.trainer_execution:
            raise IdentityMismatchError(
                "precision/gradient clipping differs from representation run identity"
            )


def _validate_initial_state_at_step_zero(
    identity: RepresentationRunIdentity,
    adapter: TGVFAdapter,
    global_step: int,
) -> None:
    if global_step != 0:
        return
    current = _state_digest(_adapter_state_to_cpu(adapter))
    if current != identity.initialization.initial_adapter_state_sha256:
        raise IdentityMismatchError(
            "global-step-zero Adapter state differs from initialization identity"
        )


def _assert_same_run_identity(
    recorded: RepresentationRunIdentity,
    expected: RepresentationRunIdentity,
) -> None:
    if recorded.identity_sha256 != expected.identity_sha256 or recorded != expected:
        raise IdentityMismatchError(
            "representation checkpoint/artifact run identity mismatch"
        )


def _validate_adapter_state(
    state: object,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ReplayMismatchError("Adapter state must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in state):
        raise ReplayMismatchError("Adapter state keys must be non-empty strings")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ReplayMismatchError("Adapter state values must all be tensors")
    if any(
        name.startswith(_BORROWED_QWEN_PREFIXES)  # type: ignore[union-attr]
        for name in state
    ):
        raise ReplayMismatchError(
            "borrowed Qwen projection state is forbidden in Adapter artifacts"
        )
    return state  # type: ignore[return-value]


def _tensor_manifest(
    state: Mapping[str, torch.Tensor],
) -> tuple[RepresentationTensorManifestEntry, ...]:
    return tuple(
        RepresentationTensorManifestEntry(
            name=name,
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            tensor_sha256=_tensor_checksum(value),
        )
        for name, value in sorted(state.items())
    )


def _optimizer_parameter_names_by_group(
    adapter: TGVFAdapter, optimizer: torch.optim.Optimizer
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    artifact_keys = set(adapter.artifact_state_dict())
    owned_parameters = {
        id(parameter): name
        for name, parameter in adapter.named_parameters()
        if name in artifact_keys and parameter.requires_grad
    }
    if not owned_parameters:
        raise ValueError("TGVF Adapter has no trainable owned parameters")
    groups: list[tuple[str, ...]] = []
    actual_ids: list[int] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            parameter_id = id(parameter)
            name = owned_parameters.get(parameter_id)
            if name is None:
                raise ValueError(
                    "optimizer contains a frozen, borrowed-Qwen, or external parameter"
                )
            actual_ids.append(parameter_id)
            names.append(name)
        if not names:
            raise ValueError("optimizer parameter groups cannot be empty")
        groups.append(tuple(names))
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("optimizer contains a duplicate parameter")
    if set(actual_ids) != set(owned_parameters):
        missing = sorted(
            owned_parameters[item] for item in set(owned_parameters) - set(actual_ids)
        )
        raise ValueError(f"optimizer omits trainable Adapter parameters: {missing}")
    return tuple(groups)


def _validate_optimizer_group_shape(
    optimizer: torch.optim.Optimizer, saved_state: Mapping[str, object]
) -> None:
    groups = saved_state.get("param_groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise ReplayMismatchError("optimizer checkpoint param_groups are malformed")
    if len(groups) != len(optimizer.param_groups):
        raise IdentityMismatchError("optimizer parameter-group count mismatch")
    for current, saved in zip(optimizer.param_groups, groups, strict=True):
        if not isinstance(saved, Mapping) or not isinstance(saved.get("params"), list):
            raise ReplayMismatchError(
                "optimizer checkpoint parameter group is malformed"
            )
        if len(current["params"]) != len(saved["params"]):
            raise IdentityMismatchError("optimizer parameter-group width mismatch")


def _validate_global_step_state(
    global_step: int,
    *,
    optimizer_state: Mapping[str, object],
    scheduler_state: Mapping[str, object] | None,
) -> None:
    state_rows = optimizer_state.get("state")
    if not isinstance(state_rows, Mapping):
        raise ReplayMismatchError("optimizer checkpoint state rows are malformed")
    optimizer_steps: list[int] = []
    for row in state_rows.values():
        if not isinstance(row, Mapping) or "step" not in row:
            raise ReplayMismatchError("AdamW checkpoint row has no step counter")
        optimizer_steps.append(_checkpoint_counter(row["step"], name="AdamW step"))
    if global_step > 0 and not optimizer_steps:
        raise ReplayMismatchError("nonzero global step has empty AdamW state")
    if any(step != global_step for step in optimizer_steps):
        raise ReplayMismatchError("AdamW step counters differ from global_step")
    if scheduler_state is not None:
        last_epoch = _checkpoint_counter(
            scheduler_state.get("last_epoch"), name="scheduler last_epoch"
        )
        step_count = _checkpoint_counter(
            scheduler_state.get("_step_count"), name="scheduler _step_count"
        )
        if last_epoch != global_step or step_count != global_step + 1:
            raise ReplayMismatchError(
                "scheduler counters differ from representation global_step"
            )


def _checkpoint_counter(value: object, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ReplayMismatchError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayMismatchError(f"{name} must be numeric")
    resolved = int(value)
    if resolved < 0 or float(resolved) != float(value):
        raise ReplayMismatchError(f"{name} must be a non-negative integer")
    return resolved


def _validate_scheduler(
    scheduler: _Stateful | None, *, optimizer: torch.optim.Optimizer
) -> None:
    if scheduler is None:
        return
    if not callable(getattr(scheduler, "state_dict", None)) or not callable(
        getattr(scheduler, "load_state_dict", None)
    ):
        raise TypeError("scheduler must provide state_dict/load_state_dict")
    bound_optimizer = getattr(scheduler, "optimizer", None)
    if bound_optimizer is not optimizer:
        raise ValueError("scheduler must be bound to the checkpointed optimizer")


def _validate_scheduler_identity(
    scheduler: _Stateful | None,
    identity: RepresentationSchedulerIdentity | None,
) -> None:
    if scheduler is None:
        if identity is not None:
            raise IdentityMismatchError(
                "run identity requires a scheduler but runtime has none"
            )
        return
    if identity is None:
        raise IdentityMismatchError(
            "runtime scheduler is absent from representation run identity"
        )
    if _qualified_type(scheduler) != identity.scheduler_type:
        raise IdentityMismatchError("scheduler type differs from run identity")
    lr_lambdas = getattr(scheduler, "lr_lambdas", None)
    if not isinstance(lr_lambdas, list) or not lr_lambdas:
        raise IdentityMismatchError(
            "project scheduler does not expose its LambdaLR construction"
        )
    for multiplier in lr_lambdas:
        closure = getattr(multiplier, "__closure__", None)
        if not closure:
            raise IdentityMismatchError(
                "project scheduler LambdaLR has no configuration closure"
            )
        matched = False
        for cell in closure:
            config = cell.cell_contents
            kind = getattr(config, "kind", None)
            kind_value = getattr(kind, "value", kind)
            if (
                kind_value == identity.kind
                and getattr(config, "total_steps", None) == identity.total_steps
                and getattr(config, "warmup_steps", None) == identity.warmup_steps
                and getattr(config, "min_lr_ratio", None)
                == getattr(identity, "min_lr_ratio", None)
            ):
                matched = True
                break
        if not matched:
            raise IdentityMismatchError(
                "scheduler construction differs from representation run identity"
            )


def _validate_rng_state(state: Mapping[str, object]) -> None:
    if not isinstance(state, Mapping):
        raise ReplayMismatchError("RNG state must be a mapping")
    allowed = {"schema_version", "python", "torch_cpu", "torch_cuda"}
    required = {"schema_version", "python", "torch_cpu"}
    keys = set(state)
    if not required <= keys or not keys <= allowed:
        raise ReplayMismatchError("RNG state fields do not match the v2 schema")
    if state["schema_version"] != REPRESENTATION_RNG_STATE_SCHEMA_VERSION:
        raise ReplayMismatchError("RNG state schema mismatch")
    validator = random.Random()
    try:
        validator.setstate(state["python"])  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError("Python RNG state is malformed") from error
    cpu_state = state["torch_cpu"]
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.device.type != "cpu"
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
    ):
        raise ReplayMismatchError("torch CPU RNG state is malformed")
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not isinstance(cuda_state, Mapping) or set(cuda_state) != {
            "device_index",
            "visible_device_count",
            "state",
        }:
            raise ReplayMismatchError("torch CUDA RNG state is malformed")
        device_index = cuda_state["device_index"]
        visible_device_count = cuda_state["visible_device_count"]
        generator_state = cuda_state["state"]
        if (
            isinstance(device_index, bool)
            or not isinstance(device_index, int)
            or device_index < 0
            or isinstance(visible_device_count, bool)
            or not isinstance(visible_device_count, int)
            or visible_device_count <= 0
            or device_index >= visible_device_count
            or not isinstance(generator_state, torch.Tensor)
            or generator_state.device.type != "cpu"
            or generator_state.dtype != torch.uint8
            or generator_state.ndim != 1
        ):
            raise ReplayMismatchError("torch CUDA RNG state is malformed")
        if not torch.cuda.is_available():
            raise ReplayMismatchError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        if not torch.cuda.is_initialized():
            raise ReplayMismatchError(
                "checkpoint contains CUDA RNG state but CUDA is not initialized"
            )
        if visible_device_count != torch.cuda.device_count():
            raise ReplayMismatchError(
                "visible CUDA device count differs from checkpoint"
            )
        if device_index != torch.cuda.current_device():
            raise ReplayMismatchError(
                "current CUDA device index differs from checkpoint"
            )
