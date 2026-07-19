"""FSDP2-safe distributed checkpoints for representation-phase training.

Model and optimizer shards use the public ``torch.distributed.checkpoint``
state-dict APIs. Rank-local sampler, scheduler, and RNG state is stored in a
small integrity-checked sidecar inside the atomically committed checkpoint
directory. Borrowed Qwen merger state is never part of the DCP payload.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Protocol, TypeVar, cast
from uuid import uuid4

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.observations.store import tensor_checksum

from .checkpoint import (
    REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION,
    REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3,
    RepresentationAccumulationIdentity,
    RepresentationOptimizerIdentity,
    RepresentationRunIdentity,
    RepresentationRunIdentityV3,
    RepresentationSamplerContractIdentity,
    RepresentationSchedulerIdentity,
    RepresentationTrainerExecutionIdentity,
    capture_representation_rng_state,
    restore_representation_rng_state,
)
from .fsdp2 import RepresentationFSDP2Binding
from .history import RepresentationMetricsHistoryIdentity
from .sampling import SameImageBatchSampler


DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION = (
    "distributed-representation-checkpoint-v1"
)
DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2 = (
    "distributed-representation-checkpoint-v2"
)
DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION = (
    "distributed-representation-rank-state-v1"
)
RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION = "rank-zero-adapter-export-v1"

_DCP_DIRECTORY_NAME = "dcp"
_METADATA_FILE_NAME = "representation_metadata.pt"
_METADATA_DIGEST_FILE_NAME = "representation_metadata.sha256"
_BORROWED_QWEN_PREFIXES = (
    "main_projection.",
    "d_deepstack_projections.",
)
_EXPECTED_TORCH_MAJOR_MINOR = (2, 9)
_HEX = frozenset("0123456789abcdef")
_COLLECTIVE_OUTCOME_KIND = "distributed-representation-collective-outcome-v1"


_T = TypeVar("_T")


class _Stateful(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> object: ...


@dataclass(frozen=True, slots=True)
class DistributedRepresentationRankState:
    rank: int
    sampler_identity_sha256: str
    sampler_state: dict[str, object]
    sampler_state_sha256: str
    rng_state: dict[str, object]
    rng_state_sha256: str
    scheduler_type: str | None
    scheduler_state: dict[str, object] | None
    scheduler_state_sha256: str | None
    schema_version: str = DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_negative_int(self.rank, field_name="rank")
        _sha256(self.sampler_identity_sha256, field_name="sampler_identity_sha256")
        if not isinstance(self.sampler_state, dict):
            raise TypeError("rank sampler state must be a dict")
        _sha256(self.sampler_state_sha256, field_name="sampler_state_sha256")
        if state_digest(self.sampler_state) != self.sampler_state_sha256:
            raise ValueError("rank sampler state digest mismatch")
        if self.sampler_state.get("identity_sha256") != self.sampler_identity_sha256:
            raise ValueError("rank sampler state carries a different identity")
        if not isinstance(self.rng_state, dict):
            raise TypeError("rank RNG state must be a dict")
        _sha256(self.rng_state_sha256, field_name="rng_state_sha256")
        if state_digest(self.rng_state) != self.rng_state_sha256:
            raise ValueError("rank RNG state digest mismatch")
        if (self.scheduler_type is None) != (self.scheduler_state is None):
            raise ValueError("rank scheduler type/state presence must align")
        if (self.scheduler_type is None) != (self.scheduler_state_sha256 is None):
            raise ValueError("rank scheduler type/digest presence must align")
        if self.scheduler_type is not None:
            _non_empty_text(self.scheduler_type, field_name="scheduler_type")
            if not isinstance(self.scheduler_state, dict):
                raise TypeError("rank scheduler state must be a dict")
            _sha256(
                self.scheduler_state_sha256,
                field_name="scheduler_state_sha256",
            )
            if state_digest(self.scheduler_state) != self.scheduler_state_sha256:
                raise ValueError("rank scheduler state digest mismatch")
        if self.schema_version != DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION:
            raise ValueError("distributed rank-state schema mismatch")


@dataclass(frozen=True, slots=True)
class DistributedRepresentationCheckpointManifest:
    run_identity: RepresentationRunIdentity
    run_identity_sha256: str
    global_step: int
    world_size: int
    fsdp_reshard_after_forward: bool
    owned_state_names: tuple[str, ...]
    optimizer_type: str
    optimizer_identity_sha256: str
    accumulation_identity_sha256: str
    trainer_execution_identity_sha256: str
    sampler_contract_identity_sha256: str
    scheduler_identity_sha256: str | None
    rank_state_sha256: tuple[str, ...]
    model_local_shard_sha256: tuple[str, ...]
    optimizer_local_shard_sha256: tuple[str, ...]
    torch_version: str
    schema_version: str = DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION
    metrics_history: RepresentationMetricsHistoryIdentity | None = None
    metrics_history_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_run_identity(self.run_identity)
        _sha256(self.run_identity_sha256, field_name="run_identity_sha256")
        if self.run_identity_sha256 != self.run_identity.identity_sha256:
            raise ValueError("distributed checkpoint run identity digest mismatch")
        _non_negative_int(self.global_step, field_name="global_step")
        _positive_int(self.world_size, field_name="world_size")
        if not isinstance(self.fsdp_reshard_after_forward, bool):
            raise TypeError("fsdp_reshard_after_forward must be bool")
        _sorted_unique_names(self.owned_state_names, field_name="owned_state_names")
        if any(
            name.startswith(_BORROWED_QWEN_PREFIXES) for name in self.owned_state_names
        ):
            raise ValueError("borrowed Qwen state is forbidden in DCP manifest")
        _non_empty_text(self.optimizer_type, field_name="optimizer_type")
        _sha256(self.optimizer_identity_sha256, field_name="optimizer_identity_sha256")
        if (
            self.optimizer_identity_sha256
            != self.run_identity.optimizer.identity_sha256
        ):
            raise ValueError("distributed optimizer identity digest mismatch")
        _sha256(
            self.accumulation_identity_sha256,
            field_name="accumulation_identity_sha256",
        )
        if (
            self.accumulation_identity_sha256
            != self.run_identity.accumulation.identity_sha256
        ):
            raise ValueError("distributed accumulation identity digest mismatch")
        _sha256(
            self.trainer_execution_identity_sha256,
            field_name="trainer_execution_identity_sha256",
        )
        if (
            self.trainer_execution_identity_sha256
            != self.run_identity.trainer_execution.identity_sha256
        ):
            raise ValueError("distributed trainer execution identity digest mismatch")
        _sha256(
            self.sampler_contract_identity_sha256,
            field_name="sampler_contract_identity_sha256",
        )
        if (
            self.sampler_contract_identity_sha256
            != self.run_identity.sampler_contract.identity_sha256
        ):
            raise ValueError("distributed sampler contract identity digest mismatch")
        if (
            self.scheduler_identity_sha256
            != self.run_identity.scheduler_identity_sha256
        ):
            raise ValueError("distributed scheduler identity digest mismatch")
        if len(self.rank_state_sha256) != self.world_size:
            raise ValueError("one rank-state digest is required per FSDP rank")
        for digest in self.rank_state_sha256:
            _sha256(digest, field_name="rank_state_sha256")
        if len(set(self.rank_state_sha256)) != len(self.rank_state_sha256):
            raise ValueError("rank-state digests must be rank-specific")
        for field_name, digests in (
            ("model_local_shard_sha256", self.model_local_shard_sha256),
            ("optimizer_local_shard_sha256", self.optimizer_local_shard_sha256),
        ):
            if not isinstance(digests, tuple) or len(digests) != self.world_size:
                raise ValueError(f"{field_name} must contain one digest per FSDP rank")
            for digest in digests:
                _sha256(digest, field_name=field_name)
        _non_empty_text(self.torch_version, field_name="torch_version")
        metrics_history = getattr(self, "metrics_history", None)
        metrics_history_identity_sha256 = getattr(
            self,
            "metrics_history_identity_sha256",
            None,
        )
        if self.schema_version == DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION:
            if (
                metrics_history is not None
                or metrics_history_identity_sha256 is not None
            ):
                raise ValueError(
                    "distributed checkpoint v1 cannot bind metrics history"
                )
        elif (
            self.schema_version
            == DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2
        ):
            _validate_metrics_history_binding(
                metrics_history,
                expected_run_identity=self.run_identity,
                expected_global_step=self.global_step,
            )
            assert isinstance(metrics_history, RepresentationMetricsHistoryIdentity)
            _sha256(
                metrics_history_identity_sha256,
                field_name="metrics_history_identity_sha256",
            )
            if metrics_history_identity_sha256 != metrics_history.identity_sha256:
                raise ValueError("distributed metrics-history digest mismatch")
        else:
            raise ValueError("distributed representation checkpoint schema mismatch")


@dataclass(frozen=True, slots=True)
class DistributedRepresentationMetadata:
    manifest: DistributedRepresentationCheckpointManifest
    rank_states: tuple[DistributedRepresentationRankState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, DistributedRepresentationCheckpointManifest):
            raise TypeError("distributed metadata manifest has the wrong type")
        self.manifest.__post_init__()
        for record in self.rank_states:
            if not isinstance(record, DistributedRepresentationRankState):
                raise TypeError("distributed metadata rank state has the wrong type")
            record.__post_init__()
        if len(self.rank_states) != self.manifest.world_size:
            raise ValueError("distributed metadata rank-state count mismatch")
        ranks = tuple(record.rank for record in self.rank_states)
        if ranks != tuple(range(self.manifest.world_size)):
            raise ValueError("distributed rank states must be sorted and complete")
        digests = tuple(_rank_state_digest(record) for record in self.rank_states)
        if digests != self.manifest.rank_state_sha256:
            raise ValueError("distributed rank-state manifest digest mismatch")
        sampler_identities = tuple(
            record.sampler_identity_sha256 for record in self.rank_states
        )
        if len(set(sampler_identities)) != len(sampler_identities):
            raise ValueError("rank-local sampler identities must be unique")
        scheduler_digests = {
            record.scheduler_state_sha256 for record in self.rank_states
        }
        if len(scheduler_digests) != 1:
            raise ValueError("scheduler state must agree across all ranks")


@dataclass(frozen=True, slots=True)
class DistributedRepresentationResumeResult:
    global_step: int
    next_global_step: int
    run_identity_sha256: str
    exact: bool = True
    next_validation_event_index: int | None = None


@dataclass(frozen=True, slots=True)
class RankZeroAdapterOwnedStateManifest:
    run_identity: RepresentationRunIdentity
    run_identity_sha256: str
    global_step: int
    tensor_names: tuple[str, ...]
    tensor_shapes: tuple[tuple[int, ...], ...]
    tensor_dtypes: tuple[str, ...]
    tensor_sha256: tuple[str, ...]
    schema_version: str = RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_run_identity(self.run_identity)
        _sha256(self.run_identity_sha256, field_name="run_identity_sha256")
        if self.run_identity_sha256 != self.run_identity.identity_sha256:
            raise ValueError("rank-zero export run identity digest mismatch")
        _non_negative_int(self.global_step, field_name="global_step")
        _sorted_unique_names(self.tensor_names, field_name="tensor_names")
        count = len(self.tensor_names)
        if not (
            len(self.tensor_shapes)
            == len(self.tensor_dtypes)
            == len(self.tensor_sha256)
            == count
        ):
            raise ValueError("rank-zero export tensor manifest fields must align")
        for shape in self.tensor_shapes:
            if not shape or any(size < 0 for size in shape):
                raise ValueError("rank-zero export tensor shape is invalid")
        for dtype in self.tensor_dtypes:
            _non_empty_text(dtype, field_name="tensor dtype")
        for digest in self.tensor_sha256:
            _sha256(digest, field_name="tensor_sha256")
        if self.schema_version != RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION:
            raise ValueError("rank-zero Adapter export schema mismatch")


@dataclass(frozen=True, slots=True)
class RankZeroAdapterOwnedStateExport:
    manifest: RankZeroAdapterOwnedStateManifest
    state: dict[str, torch.Tensor] | None
    writer_rank: int = 0

    @property
    def is_writer(self) -> bool:
        return self.state is not None


@dataclass(frozen=True, slots=True)
class _DistributedCheckpointAPI:
    get_model_state_dict: Callable[..., dict[str, object]]
    get_optimizer_state_dict: Callable[..., dict[str, object]]
    set_model_state_dict: Callable[..., Any]
    set_optimizer_state_dict: Callable[..., None]
    dcp_save: Callable[..., object]
    dcp_load: Callable[..., None]
    state_dict_options_type: type
    fsdp_module_type: type


@dataclass(frozen=True, slots=True)
class _DistributedContext:
    rank: int
    world_size: int
    process_group: Any


def save_distributed_representation_checkpoint_atomic(
    path: str | Path,
    *,
    binding: RepresentationFSDP2Binding,
    optimizer: torch.optim.Optimizer,
    scheduler: _Stateful | None,
    sampler: SameImageBatchSampler,
    run_identity: RepresentationRunIdentity,
    accumulation: RepresentationAccumulationIdentity,
    trainer_execution: RepresentationTrainerExecutionIdentity,
    global_step: int,
    metrics_history: RepresentationMetricsHistoryIdentity | None = None,
    process_group: Any = None,
) -> DistributedRepresentationCheckpointManifest:
    """Collectively save sharded Adapter/optimizer plus exact rank-local state."""

    api = _load_distributed_checkpoint_api()
    context = _distributed_context(process_group)

    def prepare_local_state() -> tuple[dict[str, object], dict[str, object], str, str]:
        _assert_distributed_fsdp2(
            binding=binding,
            optimizer=optimizer,
            process_group=process_group,
            api=api,
        )
        _validate_runtime_identity(
            binding=binding,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            run_identity=run_identity,
            accumulation=accumulation,
            trainer_execution=trainer_execution,
            world_size=context.world_size,
        )
        _non_negative_int(global_step, field_name="global_step")
        if metrics_history is not None:
            _validate_metrics_history_binding(
                metrics_history,
                expected_run_identity=run_identity,
                expected_global_step=global_step,
            )
        local_model_state, local_optimizer_state = _get_sharded_state(
            binding=binding,
            optimizer=optimizer,
            api=api,
        )
        return (
            local_model_state,
            local_optimizer_state,
            _local_shard_state_digest(local_model_state),
            _local_shard_state_digest(local_optimizer_state),
        )

    model_state, optimizer_state, model_digest, optimizer_digest = (
        _collective_local_call(
            context=context,
            phase="distributed checkpoint save preflight",
            callback=prepare_local_state,
        )
    )
    rank_states, model_digests, optimizer_digests = _gather_rank_states(
        context=context,
        sampler=sampler,
        scheduler=scheduler,
        global_step=global_step,
        run_identity=run_identity,
        metrics_history_identity_sha256=(
            None if metrics_history is None else metrics_history.identity_sha256
        ),
        model_local_shard_sha256=model_digest,
        optimizer_local_shard_sha256=optimizer_digest,
    )
    manifest = DistributedRepresentationCheckpointManifest(
        run_identity=run_identity,
        run_identity_sha256=run_identity.identity_sha256,
        global_step=global_step,
        world_size=context.world_size,
        fsdp_reshard_after_forward=binding.config.reshard_after_forward,
        owned_state_names=tuple(sorted(model_state)),
        optimizer_type=_qualified_type(optimizer),
        optimizer_identity_sha256=run_identity.optimizer.identity_sha256,
        accumulation_identity_sha256=accumulation.identity_sha256,
        trainer_execution_identity_sha256=trainer_execution.identity_sha256,
        sampler_contract_identity_sha256=(
            run_identity.sampler_contract.identity_sha256
        ),
        scheduler_identity_sha256=run_identity.scheduler_identity_sha256,
        rank_state_sha256=tuple(_rank_state_digest(record) for record in rank_states),
        model_local_shard_sha256=model_digests,
        optimizer_local_shard_sha256=optimizer_digests,
        torch_version=torch.__version__,
        schema_version=(
            DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION
            if metrics_history is None
            else DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2
        ),
        metrics_history=metrics_history,
        metrics_history_identity_sha256=(
            None if metrics_history is None else metrics_history.identity_sha256
        ),
    )
    metadata = DistributedRepresentationMetadata(manifest, rank_states)
    destination, temporary = _prepare_collective_destination(path, context=context)
    try:
        _collective_local_call(
            context=context,
            phase="distributed checkpoint DCP save",
            callback=lambda: api.dcp_save(
                {"adapter": model_state, "optimizer": optimizer_state},
                checkpoint_id=temporary / _DCP_DIRECTORY_NAME,
                process_group=context.process_group,
            ),
        )
        outcome: list[str | None] = [None]
        if context.rank == 0:
            try:
                _write_metadata(temporary, metadata)
                os.replace(temporary, destination)
                _fsync_directory(destination.parent)
            except Exception as error:  # propagated collectively below
                outcome[0] = f"{type(error).__name__}: {error}"
        _broadcast_from_group_rank_zero(outcome, context=context)
        if outcome[0] is not None:
            raise RuntimeError(
                f"rank zero failed to commit distributed checkpoint: {outcome[0]}"
            )
    finally:
        if context.rank == 0 and temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def restore_distributed_representation_checkpoint(
    path: str | Path,
    *,
    binding: RepresentationFSDP2Binding,
    optimizer: torch.optim.Optimizer,
    scheduler: _Stateful | None,
    sampler: SameImageBatchSampler,
    expected_run_identity: RepresentationRunIdentity,
    accumulation: RepresentationAccumulationIdentity,
    trainer_execution: RepresentationTrainerExecutionIdentity,
    expected_metrics_history: RepresentationMetricsHistoryIdentity | None = None,
    process_group: Any = None,
) -> DistributedRepresentationResumeResult:
    """Collectively restore an exact sharded checkpoint into an existing FSDP2 graph."""

    api = _load_distributed_checkpoint_api()
    context = _distributed_context(process_group)

    def validate_runtime() -> None:
        _assert_distributed_fsdp2(
            binding=binding,
            optimizer=optimizer,
            process_group=process_group,
            api=api,
        )
        _validate_runtime_identity(
            binding=binding,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            run_identity=expected_run_identity,
            accumulation=accumulation,
            trainer_execution=trainer_execution,
            world_size=context.world_size,
        )

    _collective_local_call(
        context=context,
        phase="distributed checkpoint restore runtime preflight",
        callback=validate_runtime,
    )
    destination = Path(path)
    metadata = _load_metadata_collective(destination, context=context)

    def prepare_load_state() -> tuple[dict[str, object], dict[str, object]]:
        _validate_restore_metadata(
            metadata,
            binding=binding,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            expected_run_identity=expected_run_identity,
            accumulation=accumulation,
            trainer_execution=trainer_execution,
            expected_metrics_history=expected_metrics_history,
            context=context,
        )
        local_model_state, local_optimizer_state = _get_sharded_state(
            binding=binding,
            optimizer=optimizer,
            api=api,
        )
        if tuple(sorted(local_model_state)) != metadata.manifest.owned_state_names:
            raise IdentityMismatchError(
                "runtime FSDP2 owned state names differ from checkpoint"
            )
        return local_model_state, local_optimizer_state

    model_state, optimizer_state = _collective_local_call(
        context=context,
        phase="distributed checkpoint restore metadata preflight",
        callback=prepare_load_state,
    )
    state = {"adapter": model_state, "optimizer": optimizer_state}
    _collective_local_call(
        context=context,
        phase="distributed checkpoint DCP load",
        callback=lambda: api.dcp_load(
            state,
            checkpoint_id=destination / _DCP_DIRECTORY_NAME,
            process_group=context.process_group,
        ),
    )

    def verify_loaded_state() -> None:
        expected_model_digest = metadata.manifest.model_local_shard_sha256[context.rank]
        expected_optimizer_digest = metadata.manifest.optimizer_local_shard_sha256[
            context.rank
        ]
        actual_model_digest = _local_shard_state_digest(state["adapter"])
        actual_optimizer_digest = _local_shard_state_digest(state["optimizer"])
        if actual_model_digest != expected_model_digest:
            raise ReplayMismatchError("DCP Adapter local-shard content digest mismatch")
        if actual_optimizer_digest != expected_optimizer_digest:
            raise ReplayMismatchError(
                "DCP optimizer local-shard content digest mismatch"
            )

    _collective_local_call(
        context=context,
        phase="distributed checkpoint loaded-content verification",
        callback=verify_loaded_state,
    )
    local = metadata.rank_states[context.rank]

    def apply_loaded_state() -> None:
        set_options = _state_dict_options(api, strict=False)
        incompatible = api.set_model_state_dict(
            binding.adapter,
            state["adapter"],
            options=set_options,
        )
        _validate_adapter_subset_load(binding, incompatible)
        api.set_optimizer_state_dict(
            binding.adapter,
            optimizer,
            state["optimizer"],
            options=_state_dict_options(api, strict=True),
        )
        if scheduler is not None:
            if local.scheduler_state is None:
                raise ReplayMismatchError("rank scheduler state is missing")
            scheduler.load_state_dict(deepcopy(local.scheduler_state))
        sampler.load_state_dict(deepcopy(local.sampler_state))
        restore_representation_rng_state(deepcopy(local.rng_state))

    _collective_local_call(
        context=context,
        phase="distributed checkpoint restore apply",
        callback=apply_loaded_state,
    )
    return DistributedRepresentationResumeResult(
        global_step=metadata.manifest.global_step,
        next_global_step=metadata.manifest.global_step + 1,
        run_identity_sha256=metadata.manifest.run_identity_sha256,
        next_validation_event_index=(
            None
            if getattr(metadata.manifest, "metrics_history", None) is None
            else metadata.manifest.metrics_history.next_validation_event_index
        ),
    )


def gather_rank_zero_full_adapter_owned_state(
    *,
    binding: RepresentationFSDP2Binding,
    run_identity: RepresentationRunIdentity,
    global_step: int,
    process_group: Any = None,
) -> RankZeroAdapterOwnedStateExport:
    """Collect a full plain CPU Adapter-only state on rank zero.

    This is an export primitive, not the sharded training-checkpoint path. Any
    borrowed Qwen prefix returned by torch is rejected rather than filtered.
    """

    api = _load_distributed_checkpoint_api()
    context = _distributed_context(process_group)

    def validate_export_runtime() -> None:
        _assert_distributed_fsdp2(
            binding=binding,
            optimizer=None,
            process_group=process_group,
            api=api,
        )
        _validate_run_identity(run_identity)
        run_identity.adapter_contract.assert_matches(binding.adapter)
        _non_negative_int(global_step, field_name="global_step")

    _collective_local_call(
        context=context,
        phase="rank-zero Adapter export preflight",
        callback=validate_export_runtime,
    )
    options = api.state_dict_options_type(
        full_state_dict=True,
        cpu_offload=True,
        ignore_frozen_params=True,
        keep_submodule_prefixes=True,
        strict=True,
        broadcast_from_rank0=False,
        flatten_optimizer_state_dict=False,
    )
    state = _collective_local_call(
        context=context,
        phase="rank-zero Adapter full-state gather",
        callback=lambda: api.get_model_state_dict(binding.adapter, options=options),
    )
    message: list[object | None] = [None]
    manifest: RankZeroAdapterOwnedStateManifest | None = None
    exported: dict[str, torch.Tensor] | None = None
    if context.rank == 0:
        try:
            if any(name.startswith(_BORROWED_QWEN_PREFIXES) for name in state):
                raise ReplayMismatchError(
                    "rank-zero full state contains a borrowed Qwen prefix"
                )
            _validate_owned_model_state(state, binding=binding)
            exported = _plain_cpu_tensor_state(state)
            names = tuple(sorted(exported))
            manifest = RankZeroAdapterOwnedStateManifest(
                run_identity=run_identity,
                run_identity_sha256=run_identity.identity_sha256,
                global_step=global_step,
                tensor_names=names,
                tensor_shapes=tuple(tuple(exported[name].shape) for name in names),
                tensor_dtypes=tuple(str(exported[name].dtype) for name in names),
                tensor_sha256=tuple(_tensor_checksum(exported[name]) for name in names),
            )
            message[0] = manifest
        except Exception as error:
            message[0] = f"{type(error).__name__}: {error}"
    _broadcast_from_group_rank_zero(message, context=context)
    if isinstance(message[0], str):
        raise ReplayMismatchError(f"rank-zero Adapter export failed: {message[0]}")
    if not isinstance(message[0], RankZeroAdapterOwnedStateManifest):
        raise ReplayMismatchError("rank-zero Adapter export manifest is missing")
    return RankZeroAdapterOwnedStateExport(
        manifest=message[0],
        state=exported if context.rank == 0 else None,
    )


def save_rank_zero_adapter_owned_state_export_atomic(
    path: str | Path,
    export: RankZeroAdapterOwnedStateExport,
) -> bool:
    """Atomically publish one gathered deployable export without overwriting.

    Every rank may call this function. Non-writer exports return ``False``
    before touching the filesystem; only rank zero's export creates the file.
    """

    _validate_rank_zero_export(export, require_state=False)
    if not export.is_writer:
        return False
    _validate_rank_zero_export(export, require_state=True)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("rank-zero Adapter exports never overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(export, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard-link publish is atomic and fails if another
        # writer already created the no-overwrite destination.
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return True


def load_rank_zero_adapter_owned_state_export(
    path: str | Path,
    *,
    expected_run_identity: RepresentationRunIdentity | None = None,
) -> RankZeroAdapterOwnedStateExport:
    """Load and integrity-check a deployable rank-zero Adapter export."""

    try:
        value = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError) as error:
        raise ReplayMismatchError(
            f"cannot load rank-zero Adapter export: {error}"
        ) from error
    if not isinstance(value, RankZeroAdapterOwnedStateExport):
        raise ReplayMismatchError("file is not a rank-zero Adapter export")
    _validate_rank_zero_export(value, require_state=True)
    if expected_run_identity is not None and (
        value.manifest.run_identity != expected_run_identity
        or value.manifest.run_identity_sha256 != expected_run_identity.identity_sha256
    ):
        raise IdentityMismatchError("rank-zero Adapter export run identity mismatch")
    return value


def _get_sharded_state(
    *,
    binding: RepresentationFSDP2Binding,
    optimizer: torch.optim.Optimizer,
    api: _DistributedCheckpointAPI,
) -> tuple[dict[str, object], dict[str, object]]:
    options = _state_dict_options(api, strict=True)
    full_model_state = api.get_model_state_dict(binding.adapter, options=options)
    model_state = _adapter_owned_subset(full_model_state, binding=binding)
    optimizer_state = api.get_optimizer_state_dict(
        binding.adapter,
        optimizer,
        options=options,
    )
    if not isinstance(optimizer_state, dict) or not optimizer_state:
        raise ReplayMismatchError("FSDP2 optimizer state dict is empty or malformed")
    return model_state, optimizer_state


def _adapter_owned_subset(
    state: Mapping[str, object],
    *,
    binding: RepresentationFSDP2Binding,
) -> dict[str, object]:
    if not isinstance(state, Mapping):
        raise TypeError("FSDP2 model state must be a mapping")
    borrowed_parameters = set(binding.plan.borrowed_parameter_names)
    leaked_parameters = borrowed_parameters & set(state)
    if leaked_parameters:
        raise ReplayMismatchError(
            f"ignore_frozen_params leaked borrowed Qwen parameters: {sorted(leaked_parameters)}"
        )
    borrowed_buffers = {
        name
        for name, _ in binding.adapter.named_buffers()
        if name.startswith(_BORROWED_QWEN_PREFIXES)
    }
    unknown_borrowed = {
        name
        for name in state
        if name.startswith(_BORROWED_QWEN_PREFIXES) and name not in borrowed_buffers
    }
    if unknown_borrowed:
        raise ReplayMismatchError(
            f"unexpected borrowed Qwen state appeared in DCP: {sorted(unknown_borrowed)}"
        )
    filtered = {
        name: value for name, value in state.items() if name not in borrowed_buffers
    }
    _validate_owned_model_state(filtered, binding=binding)
    return filtered


def _validate_owned_model_state(
    state: Mapping[str, object],
    *,
    binding: RepresentationFSDP2Binding,
) -> None:
    owned_buffers = tuple(
        name
        for name, _ in binding.adapter.named_buffers()
        if not name.startswith(_BORROWED_QWEN_PREFIXES)
    )
    expected = tuple(sorted((*binding.plan.owned_parameter_names, *owned_buffers)))
    actual = tuple(sorted(state))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ReplayMismatchError(
            f"FSDP2 Adapter-owned state mismatch: missing={missing} unexpected={unexpected}"
        )


def _validate_adapter_subset_load(
    binding: RepresentationFSDP2Binding,
    incompatible: object,
) -> None:
    missing = tuple(getattr(incompatible, "missing_keys", ()))
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    expected_missing = tuple(
        sorted(
            name
            for name, _ in (
                *tuple(binding.adapter.named_parameters()),
                *tuple(binding.adapter.named_buffers()),
            )
            if name.startswith(_BORROWED_QWEN_PREFIXES)
        )
    )
    if tuple(sorted(missing)) != expected_missing or unexpected:
        raise ReplayMismatchError(
            "strict Adapter-owned subset load produced unexpected incompatibilities"
        )


def _state_dict_options(api: _DistributedCheckpointAPI, *, strict: bool) -> object:
    return api.state_dict_options_type(
        full_state_dict=False,
        cpu_offload=False,
        ignore_frozen_params=True,
        keep_submodule_prefixes=True,
        strict=strict,
        broadcast_from_rank0=False,
        flatten_optimizer_state_dict=False,
    )


def _gather_rank_states(
    *,
    context: _DistributedContext,
    sampler: SameImageBatchSampler,
    scheduler: _Stateful | None,
    global_step: int,
    run_identity: RepresentationRunIdentity,
    metrics_history_identity_sha256: str | None,
    model_local_shard_sha256: str,
    optimizer_local_shard_sha256: str,
) -> tuple[
    tuple[DistributedRepresentationRankState, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    _sha256(
        model_local_shard_sha256,
        field_name="model_local_shard_sha256",
    )
    _sha256(
        optimizer_local_shard_sha256,
        field_name="optimizer_local_shard_sha256",
    )
    if metrics_history_identity_sha256 is not None:
        _sha256(
            metrics_history_identity_sha256,
            field_name="metrics_history_identity_sha256",
        )

    def capture_local_rank_state() -> dict[str, object]:
        sampler_state = deepcopy(sampler.state_dict())
        rng_state = capture_representation_rng_state()
        scheduler_state = (
            None if scheduler is None else deepcopy(scheduler.state_dict())
        )
        return {
            "rank": context.rank,
            "global_step": global_step,
            "run_identity_sha256": run_identity.identity_sha256,
            "metrics_history_identity_sha256": metrics_history_identity_sha256,
            "sampler_identity_sha256": sampler.identity_sha256,
            "sampler_state": sampler_state,
            "rng_state": rng_state,
            "scheduler_type": (
                None if scheduler is None else _qualified_type(scheduler)
            ),
            "scheduler_state": scheduler_state,
            "model_local_shard_sha256": model_local_shard_sha256,
            "optimizer_local_shard_sha256": optimizer_local_shard_sha256,
        }

    payload = _collective_local_call(
        context=context,
        phase="distributed checkpoint rank-state capture",
        callback=capture_local_rank_state,
    )
    gathered: list[object] = [None] * context.world_size
    torch.distributed.all_gather_object(gathered, payload, group=context.process_group)
    records = []
    model_digests = []
    optimizer_digests = []
    for expected_rank, item in enumerate(gathered):
        if not isinstance(item, Mapping):
            raise ReplayMismatchError("distributed rank-state payload is malformed")
        if item.get("rank") != expected_rank:
            raise ReplayMismatchError("distributed rank-state order is malformed")
        if item.get("global_step") != global_step:
            raise IdentityMismatchError("global_step differs across FSDP2 ranks")
        if item.get("run_identity_sha256") != run_identity.identity_sha256:
            raise IdentityMismatchError("run identity differs across FSDP2 ranks")
        if (
            item.get("metrics_history_identity_sha256")
            != metrics_history_identity_sha256
        ):
            raise IdentityMismatchError(
                "metrics-history identity differs across FSDP2 ranks"
            )
        sampler_payload = item.get("sampler_state")
        rng_payload = item.get("rng_state")
        scheduler_payload = item.get("scheduler_state")
        model_digest = item.get("model_local_shard_sha256")
        optimizer_digest = item.get("optimizer_local_shard_sha256")
        if not isinstance(sampler_payload, dict) or not isinstance(rng_payload, dict):
            raise ReplayMismatchError("rank-local sampler/RNG payload is malformed")
        if scheduler_payload is not None and not isinstance(scheduler_payload, dict):
            raise ReplayMismatchError("rank-local scheduler payload is malformed")
        _sha256(model_digest, field_name="model_local_shard_sha256")
        _sha256(optimizer_digest, field_name="optimizer_local_shard_sha256")
        records.append(
            DistributedRepresentationRankState(
                rank=expected_rank,
                sampler_identity_sha256=item["sampler_identity_sha256"],
                sampler_state=sampler_payload,
                sampler_state_sha256=state_digest(sampler_payload),
                rng_state=rng_payload,
                rng_state_sha256=state_digest(rng_payload),
                scheduler_type=item.get("scheduler_type"),
                scheduler_state=scheduler_payload,
                scheduler_state_sha256=(
                    None
                    if scheduler_payload is None
                    else state_digest(scheduler_payload)
                ),
            )
        )
        model_digests.append(cast(str, model_digest))
        optimizer_digests.append(cast(str, optimizer_digest))
    return tuple(records), tuple(model_digests), tuple(optimizer_digests)


def _validate_runtime_identity(
    *,
    binding: RepresentationFSDP2Binding,
    optimizer: torch.optim.Optimizer,
    scheduler: _Stateful | None,
    sampler: SameImageBatchSampler,
    run_identity: RepresentationRunIdentity,
    accumulation: RepresentationAccumulationIdentity,
    trainer_execution: RepresentationTrainerExecutionIdentity,
    world_size: int,
) -> None:
    _validate_run_identity(run_identity)
    run_identity.adapter_contract.assert_matches(binding.adapter)
    if (
        RepresentationOptimizerIdentity.from_optimizer(optimizer)
        != run_identity.optimizer
    ):
        raise IdentityMismatchError("distributed optimizer identity mismatch")
    if accumulation != run_identity.accumulation:
        raise IdentityMismatchError("distributed accumulation identity mismatch")
    if trainer_execution != run_identity.trainer_execution:
        raise IdentityMismatchError("distributed trainer execution identity mismatch")
    sampler_contract = RepresentationSamplerContractIdentity.from_sampler(sampler)
    if sampler_contract != run_identity.sampler_contract:
        raise IdentityMismatchError("distributed sampler contract identity mismatch")
    if accumulation.data_parallel_world_size != world_size:
        raise IdentityMismatchError(
            "accumulation world size differs from process group"
        )
    _validate_scheduler_runtime(scheduler, run_identity.scheduler, optimizer=optimizer)


def _validate_scheduler_runtime(
    scheduler: _Stateful | None,
    identity: RepresentationSchedulerIdentity | None,
    *,
    optimizer: torch.optim.Optimizer,
) -> None:
    if scheduler is None:
        if identity is not None:
            raise IdentityMismatchError("run identity requires a scheduler")
        return
    if identity is None:
        raise IdentityMismatchError("runtime scheduler is absent from run identity")
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise ValueError("scheduler must be bound to the distributed optimizer")
    if _qualified_type(scheduler) != identity.scheduler_type:
        raise IdentityMismatchError("distributed scheduler type mismatch")
    lambdas = getattr(scheduler, "lr_lambdas", None)
    if not isinstance(lambdas, list) or not lambdas:
        raise IdentityMismatchError("distributed scheduler is not project LambdaLR")
    for multiplier in lambdas:
        closure = getattr(multiplier, "__closure__", None)
        matched = False
        for cell in closure or ():
            config = cell.cell_contents
            kind = getattr(getattr(config, "kind", None), "value", None)
            if (
                kind == identity.kind
                and getattr(config, "total_steps", None) == identity.total_steps
                and getattr(config, "warmup_steps", None) == identity.warmup_steps
                and getattr(config, "min_lr_ratio", None)
                == getattr(identity, "min_lr_ratio", None)
            ):
                matched = True
                break
        if not matched:
            raise IdentityMismatchError("distributed scheduler construction mismatch")


def _distributed_context(process_group: Any) -> _DistributedContext:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "distributed checkpoint requires initialized torch.distributed"
        )
    return _DistributedContext(
        rank=torch.distributed.get_rank(process_group),
        world_size=torch.distributed.get_world_size(process_group),
        process_group=process_group,
    )


def _collective_local_call(
    *,
    context: _DistributedContext,
    phase: str,
    callback: Callable[[], _T],
) -> _T:
    """Run local work, then make every rank agree before any next collective.

    The callback itself must not contain an unmatched project collective. Public
    DCP calls are wrapped here so a backend error that returns on every rank, or
    returns on some ranks and raises on others, is agreed before the next phase.
    """

    _non_empty_text(phase, field_name="collective phase")
    local_error: Exception | None = None
    result: _T | None = None
    try:
        result = callback()
    except Exception as error:  # converted to one all-rank phase outcome below
        local_error = error
    payload = {
        "kind": _COLLECTIVE_OUTCOME_KIND,
        "rank": context.rank,
        "phase": phase,
        "error_type": (None if local_error is None else _qualified_type(local_error)),
        "error_message": None if local_error is None else str(local_error),
    }
    gathered: list[object] = [None] * context.world_size
    torch.distributed.all_gather_object(
        gathered,
        payload,
        group=context.process_group,
    )
    failures = []
    for expected_rank, item in enumerate(gathered):
        if not isinstance(item, Mapping):
            raise RuntimeError("collective phase outcome is malformed")
        if (
            item.get("kind") != _COLLECTIVE_OUTCOME_KIND
            or item.get("rank") != expected_rank
            or item.get("phase") != phase
        ):
            raise RuntimeError("collective phase outcome identity is malformed")
        error_type = item.get("error_type")
        error_message = item.get("error_message")
        if (error_type is None) != (error_message is None):
            raise RuntimeError("collective phase error fields are malformed")
        if error_type is not None:
            if not isinstance(error_type, str) or not isinstance(error_message, str):
                raise RuntimeError("collective phase error payload is malformed")
            failures.append((expected_rank, error_type, error_message))
    if failures:
        if local_error is not None:
            raise local_error
        details = "; ".join(
            f"rank {rank}: {error_type}: {message}"
            for rank, error_type, message in failures
        )
        raise RuntimeError(f"{phase} failed on another rank: {details}")
    return cast(_T, result)


def _assert_distributed_fsdp2(
    *,
    binding: RepresentationFSDP2Binding,
    optimizer: torch.optim.Optimizer | None,
    process_group: Any,
    api: _DistributedCheckpointAPI,
) -> _DistributedContext:
    if not isinstance(binding, RepresentationFSDP2Binding):
        raise TypeError("binding must be a RepresentationFSDP2Binding")
    context = _distributed_context(process_group)
    if not isinstance(binding.adapter, api.fsdp_module_type):
        raise RuntimeError("distributed checkpoint requires an FSDP2 Adapter root")
    if any(
        not isinstance(module, api.fsdp_module_type)
        for module in binding.owned_group_modules
    ):
        raise RuntimeError("every Adapter-owned group must have composable FSDP2 state")
    if context.world_size != binding.config.world_size:
        raise IdentityMismatchError(
            "FSDP2 binding world size differs from process group"
        )
    if optimizer is not None:
        binding.assert_optimizer_ownership(optimizer)
    return context


def _validate_restore_metadata(
    metadata: DistributedRepresentationMetadata,
    *,
    binding: RepresentationFSDP2Binding,
    optimizer: torch.optim.Optimizer,
    scheduler: _Stateful | None,
    sampler: SameImageBatchSampler,
    expected_run_identity: RepresentationRunIdentity,
    accumulation: RepresentationAccumulationIdentity,
    trainer_execution: RepresentationTrainerExecutionIdentity,
    expected_metrics_history: RepresentationMetricsHistoryIdentity | None,
    context: _DistributedContext,
) -> None:
    metadata.__post_init__()
    manifest = metadata.manifest
    if (
        manifest.run_identity_sha256 != expected_run_identity.identity_sha256
        or manifest.run_identity != expected_run_identity
    ):
        raise IdentityMismatchError("distributed checkpoint run identity mismatch")
    if manifest.world_size != context.world_size:
        raise IdentityMismatchError("distributed checkpoint world size mismatch")
    if manifest.fsdp_reshard_after_forward != binding.config.reshard_after_forward:
        raise IdentityMismatchError("FSDP2 reshard policy differs from checkpoint")
    if manifest.optimizer_type != _qualified_type(optimizer):
        raise IdentityMismatchError("distributed checkpoint optimizer type mismatch")
    if manifest.accumulation_identity_sha256 != accumulation.identity_sha256:
        raise IdentityMismatchError("distributed checkpoint accumulation mismatch")
    if manifest.trainer_execution_identity_sha256 != trainer_execution.identity_sha256:
        raise IdentityMismatchError("distributed checkpoint trainer execution mismatch")
    if manifest.torch_version != torch.__version__:
        raise IdentityMismatchError("torch version differs from distributed checkpoint")
    _validate_expected_metrics_history(
        manifest,
        expected_metrics_history=expected_metrics_history,
        expected_run_identity=expected_run_identity,
    )
    local = metadata.rank_states[context.rank]
    if local.sampler_identity_sha256 != sampler.identity_sha256:
        raise IdentityMismatchError("rank-local sampler identity mismatch")
    runtime_scheduler_type = None if scheduler is None else _qualified_type(scheduler)
    if local.scheduler_type != runtime_scheduler_type:
        raise IdentityMismatchError("rank-local scheduler type mismatch")


def _prepare_collective_destination(
    path: str | Path,
    *,
    context: _DistributedContext,
) -> tuple[Path, Path]:
    destination = Path(path)
    payload: list[object | None] = [None]
    if context.rank == 0:
        try:
            if destination.exists():
                raise FileExistsError(
                    "distributed checkpoints never overwrite an existing directory"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.parent / (
                f".{destination.name}.incomplete-{uuid4().hex}"
            )
            temporary.mkdir()
            payload[0] = str(temporary)
        except Exception as error:
            payload[0] = f"ERROR:{type(error).__name__}: {error}"
    _broadcast_from_group_rank_zero(payload, context=context)
    if not isinstance(payload[0], str):
        raise RuntimeError("rank zero did not provide a checkpoint staging path")
    if payload[0].startswith("ERROR:"):
        raise RuntimeError(payload[0])
    return destination, Path(payload[0])


def _write_metadata(path: Path, metadata: DistributedRepresentationMetadata) -> None:
    buffer = BytesIO()
    torch.save(metadata, buffer)
    payload = buffer.getvalue()
    digest = sha256(payload).hexdigest()
    _write_bytes_fsync(path / _METADATA_FILE_NAME, payload)
    _write_bytes_fsync(
        path / _METADATA_DIGEST_FILE_NAME,
        f"{digest}\n".encode("ascii"),
    )
    _fsync_directory(path)


def _broadcast_from_group_rank_zero(
    values: list[object | None],
    *,
    context: _DistributedContext,
) -> None:
    if context.process_group is None:
        torch.distributed.broadcast_object_list(values, src=0)
    else:
        torch.distributed.broadcast_object_list(
            values,
            group=context.process_group,
            group_src=0,
        )


def _load_metadata_collective(
    path: Path,
    *,
    context: _DistributedContext,
) -> DistributedRepresentationMetadata:
    payload: list[object | None] = [None]
    if context.rank == 0:
        try:
            payload[0] = _read_metadata(path)
        except Exception as error:
            payload[0] = f"ERROR:{type(error).__name__}: {error}"
    _broadcast_from_group_rank_zero(payload, context=context)
    if isinstance(payload[0], str):
        raise ReplayMismatchError(payload[0])
    if not isinstance(payload[0], DistributedRepresentationMetadata):
        raise ReplayMismatchError("distributed checkpoint metadata is missing")
    return payload[0]


def load_distributed_representation_checkpoint_metadata(
    path: str | Path,
) -> DistributedRepresentationMetadata:
    """Read and fully validate a committed DCP sidecar without distributed init."""

    return _read_metadata(Path(path))


def _read_metadata(path: Path) -> DistributedRepresentationMetadata:
    if not path.is_dir():
        raise FileNotFoundError("distributed checkpoint directory does not exist")
    metadata_path = path / _METADATA_FILE_NAME
    digest_path = path / _METADATA_DIGEST_FILE_NAME
    payload = metadata_path.read_bytes()
    expected = digest_path.read_text(encoding="ascii").strip()
    _sha256(expected, field_name="metadata file digest")
    if sha256(payload).hexdigest() != expected:
        raise ReplayMismatchError("distributed checkpoint metadata digest mismatch")
    value = torch.load(BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(value, DistributedRepresentationMetadata):
        raise ReplayMismatchError("distributed checkpoint metadata type mismatch")
    _normalize_legacy_manifest_defaults(value.manifest)
    value.__post_init__()
    return value


def _normalize_legacy_manifest_defaults(
    manifest: object,
) -> None:
    """Populate fields absent from a pre-v2 frozen-slots pickle."""

    if not isinstance(manifest, DistributedRepresentationCheckpointManifest):
        return
    for field_name in (
        "metrics_history",
        "metrics_history_identity_sha256",
    ):
        if not hasattr(manifest, field_name):
            object.__setattr__(manifest, field_name, None)


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _plain_cpu_tensor_state(state: Mapping[str, object]) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"full Adapter state {name!r} is not a plain tensor")
        if value.device.type != "cpu":
            raise ValueError("rank-zero full Adapter state must be CPU-offloaded")
        result[name] = value.detach().clone()
    return result


def _validate_rank_zero_export(
    export: RankZeroAdapterOwnedStateExport,
    *,
    require_state: bool,
) -> None:
    if not isinstance(export, RankZeroAdapterOwnedStateExport):
        raise TypeError("export must be a RankZeroAdapterOwnedStateExport")
    if export.writer_rank != 0:
        raise ValueError("rank-zero Adapter export writer rank must be zero")
    if not isinstance(export.manifest, RankZeroAdapterOwnedStateManifest):
        raise TypeError("rank-zero Adapter export manifest has the wrong type")
    export.manifest.__post_init__()
    if export.state is None:
        if require_state:
            raise ReplayMismatchError(
                "rank-zero Adapter export tensor state is missing"
            )
        return
    if not isinstance(export.state, dict):
        raise TypeError("rank-zero Adapter export state must be a dict")
    state = _plain_cpu_tensor_state(export.state)
    manifest = export.manifest
    names = tuple(sorted(state))
    if names != manifest.tensor_names:
        raise ReplayMismatchError("rank-zero Adapter export tensor names mismatch")
    shapes = tuple(tuple(state[name].shape) for name in names)
    dtypes = tuple(str(state[name].dtype) for name in names)
    digests = tuple(_tensor_checksum(state[name]) for name in names)
    if shapes != manifest.tensor_shapes:
        raise ReplayMismatchError("rank-zero Adapter export tensor shapes mismatch")
    if dtypes != manifest.tensor_dtypes:
        raise ReplayMismatchError("rank-zero Adapter export tensor dtypes mismatch")
    if digests != manifest.tensor_sha256:
        raise ReplayMismatchError("rank-zero Adapter export tensor checksum mismatch")
    if any(name.startswith(_BORROWED_QWEN_PREFIXES) for name in names):
        raise ReplayMismatchError(
            "rank-zero Adapter export contains borrowed Qwen state"
        )


def _rank_state_digest(record: DistributedRepresentationRankState) -> str:
    payload = {
        "schema_version": record.schema_version,
        "rank": record.rank,
        "sampler_identity_sha256": record.sampler_identity_sha256,
        "sampler_state_sha256": record.sampler_state_sha256,
        "rng_state_sha256": record.rng_state_sha256,
        "scheduler_type": record.scheduler_type,
        "scheduler_state_sha256": record.scheduler_state_sha256,
    }
    return state_digest(payload)


def _local_shard_state_digest(value: object) -> str:
    """Hash only rank-local tensor storage plus deterministic state structure."""

    try:
        from torch.distributed.tensor import DTensor
    except (ImportError, AttributeError):  # pragma: no cover - torch pin fails earlier
        DTensor = ()  # type: ignore[assignment,misc]
    hasher = sha256()
    _update_local_shard_digest(hasher, value, dtensor_type=DTensor)
    return hasher.hexdigest()


def _update_local_shard_digest(
    hasher: Any,
    value: object,
    *,
    dtensor_type: object,
) -> None:
    if isinstance(value, dtensor_type):
        local = value.to_local()
        if not isinstance(local, torch.Tensor):
            raise TypeError("DTensor local checkpoint shard must be a Tensor")
        _digest_text(hasher, "dtensor")
        _digest_text(hasher, repr(tuple(value.shape)))
        _digest_text(hasher, repr(tuple(value.stride())))
        _digest_text(hasher, str(value.dtype))
        _digest_text(hasher, str(value.layout))
        _digest_text(hasher, repr(tuple(map(str, value.placements))))
        _digest_text(hasher, repr(tuple(value.device_mesh.shape)))
        _digest_text(hasher, repr(value.device_mesh.mesh_dim_names))
        _digest_tensor_storage(hasher, local)
        return
    if isinstance(value, torch.Tensor):
        _digest_text(hasher, "tensor")
        _digest_tensor_storage(hasher, value)
        return
    if isinstance(value, Mapping):
        _digest_text(hasher, "mapping")
        keys = sorted(
            value,
            key=lambda item: (
                type(item).__module__,
                type(item).__qualname__,
                repr(item),
            ),
        )
        for key in keys:
            _update_local_shard_digest(hasher, key, dtensor_type=dtensor_type)
            _update_local_shard_digest(
                hasher,
                value[key],
                dtensor_type=dtensor_type,
            )
        return
    if isinstance(value, (tuple, list)):
        _digest_text(hasher, "tuple" if isinstance(value, tuple) else "list")
        for item in value:
            _update_local_shard_digest(hasher, item, dtensor_type=dtensor_type)
        return
    if isinstance(value, bytes):
        _digest_text(hasher, "bytes")
        hasher.update(len(value).to_bytes(8, byteorder="big"))
        hasher.update(value)
        return
    if isinstance(value, (str, int, float, bool)) or value is None:
        _digest_text(
            hasher,
            f"{type(value).__module__}.{type(value).__qualname__}:{value!r}",
        )
        return
    if isinstance(value, (torch.dtype, torch.device)):
        _digest_text(hasher, f"{type(value).__qualname__}:{value}")
        return
    raise TypeError(
        "unsupported distributed local-shard checkpoint state type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _digest_tensor_storage(hasher: Any, value: torch.Tensor) -> None:
    if value.layout is not torch.strided:
        raise TypeError("distributed checkpoint digest requires strided tensors")
    _digest_text(hasher, repr(tuple(value.shape)))
    _digest_text(hasher, repr(tuple(value.stride())))
    _digest_text(hasher, str(value.dtype))
    _digest_text(hasher, _tensor_checksum(value))


def _digest_text(hasher: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="big"))
    hasher.update(encoded)


def _validate_run_identity(identity: object) -> None:
    if not isinstance(identity, RepresentationRunIdentity):
        raise TypeError("run identity must be a RepresentationRunIdentity")
    if type(identity) is RepresentationRunIdentity:
        expected_schema_version = REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION
    elif type(identity) is RepresentationRunIdentityV3:
        expected_schema_version = REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3
    else:
        raise TypeError("unsupported representation run identity type")
    if identity.schema_version != expected_schema_version:
        raise ValueError("representation run identity schema mismatch")
    identity.__post_init__()
    # Re-hashing traverses every nested identity and fails on unsupported drift.
    _sha256(identity.identity_sha256, field_name="run identity digest")


def _validate_metrics_history_binding(
    metrics_history: object,
    *,
    expected_run_identity: RepresentationRunIdentity,
    expected_global_step: int,
) -> None:
    if not isinstance(metrics_history, RepresentationMetricsHistoryIdentity):
        raise TypeError(
            "metrics_history must be a RepresentationMetricsHistoryIdentity"
        )
    metrics_history.__post_init__()
    if metrics_history.run_id != expected_run_identity.run_id:
        raise IdentityMismatchError("metrics history run_id mismatch")
    if metrics_history.run_identity_sha256 != expected_run_identity.identity_sha256:
        raise IdentityMismatchError("metrics history run identity mismatch")
    if metrics_history.checkpoint_global_step != expected_global_step:
        raise IdentityMismatchError("metrics history checkpoint step mismatch")


def _validate_expected_metrics_history(
    manifest: DistributedRepresentationCheckpointManifest,
    *,
    expected_metrics_history: RepresentationMetricsHistoryIdentity | None,
    expected_run_identity: RepresentationRunIdentity,
) -> None:
    recorded = getattr(manifest, "metrics_history", None)
    recorded_digest = getattr(manifest, "metrics_history_identity_sha256", None)
    if manifest.schema_version == DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION:
        if expected_metrics_history is not None:
            raise IdentityMismatchError(
                "distributed checkpoint v1 has no metrics-history binding"
            )
        return
    if (
        manifest.schema_version
        != DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2
    ):
        raise ValueError("distributed representation checkpoint schema mismatch")
    if expected_metrics_history is None:
        raise ReplayMismatchError(
            "distributed checkpoint v2 requires expected metrics history"
        )
    _validate_metrics_history_binding(
        expected_metrics_history,
        expected_run_identity=expected_run_identity,
        expected_global_step=manifest.global_step,
    )
    if (
        recorded != expected_metrics_history
        or recorded_digest != expected_metrics_history.identity_sha256
    ):
        raise ReplayMismatchError("distributed checkpoint metrics history mismatch")


def _load_distributed_checkpoint_api() -> _DistributedCheckpointAPI:
    version = torch.__version__.split("+", 1)[0].split(".")
    try:
        major_minor = (int(version[0]), int(version[1]))
    except (IndexError, ValueError) as error:
        raise RuntimeError(
            f"cannot parse torch version {torch.__version__!r}"
        ) from error
    if major_minor != _EXPECTED_TORCH_MAJOR_MINOR:
        raise RuntimeError(
            "representation distributed checkpoint requires pinned torch 2.9"
        )
    from torch.distributed.checkpoint import load, save
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.fsdp import FSDPModule

    return _DistributedCheckpointAPI(
        get_model_state_dict=get_model_state_dict,
        get_optimizer_state_dict=get_optimizer_state_dict,
        set_model_state_dict=set_model_state_dict,
        set_optimizer_state_dict=set_optimizer_state_dict,
        dcp_save=save,
        dcp_load=load,
        state_dict_options_type=StateDictOptions,
        fsdp_module_type=FSDPModule,
    )


def _qualified_type(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _tensor_checksum(value: torch.Tensor) -> str:
    canonical = value if value.ndim else value.reshape(1)
    return tensor_checksum(canonical)


def _non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _sha256(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
        raise ValueError(f"{field_name} must be a lowercase SHA256")


def _positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _non_negative_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _sorted_unique_names(values: object, *, field_name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be sorted and unique")
