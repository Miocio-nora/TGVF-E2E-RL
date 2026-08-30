"""FSDP2-safe distributed checkpoints for representation-phase training.

Model and optimizer shards use the public ``torch.distributed.checkpoint``
state-dict APIs. Rank-local sampler, scheduler, and RNG state is stored in a
small integrity-checked sidecar inside the atomically committed checkpoint
directory. Borrowed Qwen merger state is never part of the DCP payload.
"""

from __future__ import annotations

from collections.abc import Callable as Callable, Mapping as Mapping
from copy import deepcopy as deepcopy
from dataclasses import dataclass as dataclass
from hashlib import sha256 as sha256
import inspect as inspect
from io import BytesIO as BytesIO
import os as os
from pathlib import Path
import shutil as shutil
import tempfile as tempfile
from typing import Any, Protocol as Protocol, TypeVar as TypeVar, cast as cast
from uuid import uuid4 as uuid4

import torch as torch

from tgvf_rl.checkpoint.coordinator import state_digest as state_digest
from tgvf_rl.contracts.errors import (
    IdentityMismatchError as IdentityMismatchError,
    ReplayMismatchError as ReplayMismatchError,
)
from tgvf_rl.observations.store import tensor_checksum as tensor_checksum

from .checkpoint import (
    REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION as REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION,
    REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3 as REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3,
    RepresentationAccumulationIdentity,
    RepresentationOptimizerIdentity as RepresentationOptimizerIdentity,
    RepresentationRunIdentity,
    RepresentationRunIdentityV3 as RepresentationRunIdentityV3,
    RepresentationSamplerContractIdentity as RepresentationSamplerContractIdentity,
    RepresentationSchedulerIdentity as RepresentationSchedulerIdentity,
    RepresentationTrainerExecutionIdentity,
    capture_representation_rng_state,
    restore_representation_rng_state,
)
from .distributed_checkpoint_export import (
    RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION as RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION,
    RankZeroAdapterOwnedStateExport,
    RankZeroAdapterOwnedStateManifest,
    _validate_rank_zero_export as _validate_rank_zero_export,
    load_rank_zero_adapter_owned_state_export as load_rank_zero_adapter_owned_state_export,
    save_rank_zero_adapter_owned_state_export_atomic as save_rank_zero_adapter_owned_state_export_atomic,
)
from .distributed_checkpoint_integrity import (
    _COLLECTIVE_OUTCOME_KIND as _COLLECTIVE_OUTCOME_KIND,
    _DCP_DIRECTORY_NAME,
    _DistributedCheckpointAPI as _DistributedCheckpointAPI,
    _DistributedContext,
    _Stateful,
    _adapter_owned_subset as _adapter_owned_subset,
    _assert_distributed_fsdp2,
    _assert_public_signature as _assert_public_signature,
    _broadcast_from_group_rank_zero,
    _collective_local_call,
    _digest_tensor_storage as _digest_tensor_storage,
    _digest_text as _digest_text,
    _distributed_context,
    _fsync_directory,
    _get_sharded_state,
    _load_distributed_checkpoint_api,
    _load_metadata_collective,
    _local_shard_state_digest,
    _normalize_legacy_manifest_defaults as _normalize_legacy_manifest_defaults,
    _plain_cpu_tensor_state,
    _prepare_collective_destination,
    _qualified_type,
    _read_metadata as _read_metadata,
    _state_dict_options,
    _tensor_checksum,
    _update_local_shard_digest as _update_local_shard_digest,
    _validate_adapter_subset_load,
    _validate_owned_model_state,
    _validate_restore_metadata,
    _write_bytes_fsync as _write_bytes_fsync,
    _write_metadata,
    load_distributed_representation_checkpoint_metadata as load_distributed_representation_checkpoint_metadata,
)
from .distributed_checkpoint_schema import (
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION,
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2,
    DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION as DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION,
    DistributedRepresentationCheckpointManifest,
    DistributedRepresentationMetadata,
    DistributedRepresentationRankState,
    DistributedRepresentationResumeResult,
    _BORROWED_QWEN_PREFIXES,
    _non_empty_text as _non_empty_text,
    _non_negative_int,
    _positive_int as _positive_int,
    _rank_state_digest,
    _sha256,
    _sorted_unique_names as _sorted_unique_names,
    _validate_expected_metrics_history as _validate_expected_metrics_history,
    _validate_metrics_history_binding,
    _validate_run_identity,
)
from .fsdp2 import RepresentationFSDP2Binding
from .history import RepresentationMetricsHistoryIdentity
from .sampling import SameImageBatchSampler


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
