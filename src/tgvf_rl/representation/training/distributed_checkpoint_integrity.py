"""Collective, metadata, and digest helpers for distributed checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import inspect
from io import BytesIO
import os
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast
from uuid import uuid4

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.observations.store import tensor_checksum
from tgvf_rl.public_api_compat import (
    freeze_public_class_annotations,
    rebind_public_class,
    rebind_public_function,
)

from .checkpoint import (
    RepresentationAccumulationIdentity,
    RepresentationRunIdentity,
    RepresentationTrainerExecutionIdentity,
)
from .distributed_checkpoint_schema import (
    DistributedRepresentationCheckpointManifest,
    DistributedRepresentationMetadata,
    _BORROWED_QWEN_PREFIXES,
    _non_empty_text,
    _sha256,
    _validate_expected_metrics_history,
)
from .fsdp2 import RepresentationFSDP2Binding, _require_supported_torch_identity
from .history import RepresentationMetricsHistoryIdentity
from .sampling import SameImageBatchSampler


_DCP_DIRECTORY_NAME = "dcp"
_METADATA_FILE_NAME = "representation_metadata.pt"
_METADATA_DIGEST_FILE_NAME = "representation_metadata.sha256"
_EXPECTED_STATE_DICT_OPTIONS_PARAMETERS = (
    "full_state_dict",
    "cpu_offload",
    "ignore_frozen_params",
    "keep_submodule_prefixes",
    "strict",
    "broadcast_from_rank0",
    "flatten_optimizer_state_dict",
    "dsd_fqn_modifiers",
)
_EXPECTED_DCP_PUBLIC_PARAMETERS = {
    "get_model_state_dict": ("model", "submodules", "options"),
    "get_optimizer_state_dict": (
        "model",
        "optimizers",
        "submodules",
        "options",
    ),
    "set_model_state_dict": ("model", "model_state_dict", "options"),
    "set_optimizer_state_dict": (
        "model",
        "optimizers",
        "optim_state_dict",
        "options",
    ),
    "save": (
        "state_dict",
        "checkpoint_id",
        "storage_writer",
        "planner",
        "process_group",
        "no_dist",
        "use_collectives",
    ),
    "load": (
        "state_dict",
        "checkpoint_id",
        "storage_reader",
        "planner",
        "process_group",
        "no_dist",
    ),
}
_COLLECTIVE_OUTCOME_KIND = "distributed-representation-collective-outcome-v1"
_T = TypeVar("_T")


class _Stateful(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> object: ...


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


def _load_distributed_checkpoint_api() -> _DistributedCheckpointAPI:
    _require_supported_torch_identity(api_name="distributed checkpoint")
    try:
        from torch.distributed.checkpoint import load, save
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_optimizer_state_dict,
            set_model_state_dict,
            set_optimizer_state_dict,
        )
        from torch.distributed.fsdp import FSDPModule
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "torch distributed checkpoint public APIs are unavailable"
        ) from error

    public_functions = {
        "get_model_state_dict": get_model_state_dict,
        "get_optimizer_state_dict": get_optimizer_state_dict,
        "set_model_state_dict": set_model_state_dict,
        "set_optimizer_state_dict": set_optimizer_state_dict,
        "save": save,
        "load": load,
    }
    for api_name, value in public_functions.items():
        _assert_public_signature(
            value,
            api_name=api_name,
            expected_parameters=_EXPECTED_DCP_PUBLIC_PARAMETERS[api_name],
        )
    _assert_public_signature(
        StateDictOptions,
        api_name="StateDictOptions",
        expected_parameters=_EXPECTED_STATE_DICT_OPTIONS_PARAMETERS,
    )
    if (
        not inspect.isclass(StateDictOptions)
        or StateDictOptions.__name__ != "StateDictOptions"
        or StateDictOptions.__module__ != "torch.distributed.checkpoint.state_dict"
    ):
        raise RuntimeError("torch StateDictOptions public class identity drifted")
    if (
        not inspect.isclass(FSDPModule)
        or FSDPModule.__name__ != "FSDPModule"
        or FSDPModule.__module__ != "torch.distributed.fsdp"
    ):
        raise RuntimeError("torch FSDPModule public class identity drifted")

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


def _assert_public_signature(
    value: object,
    *,
    api_name: str,
    expected_parameters: tuple[str, ...],
) -> None:
    try:
        actual_parameters = tuple(inspect.signature(value).parameters)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"cannot inspect torch {api_name} signature") from error
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"torch {api_name} public signature drifted: "
            f"expected={expected_parameters} actual={actual_parameters}"
        )


def _qualified_type(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _tensor_checksum(value: torch.Tensor) -> str:
    canonical = value if value.ndim else value.reshape(1)
    return tensor_checksum(canonical)


_DISTRIBUTED_CHECKPOINT_INTEGRITY_TYPES = (
    _Stateful,
    _DistributedCheckpointAPI,
    _DistributedContext,
)
_DISTRIBUTED_CHECKPOINT_INTEGRITY_FUNCTIONS = (
    _get_sharded_state,
    _adapter_owned_subset,
    _validate_owned_model_state,
    _validate_adapter_subset_load,
    _state_dict_options,
    _distributed_context,
    _collective_local_call,
    _assert_distributed_fsdp2,
    _validate_restore_metadata,
    _prepare_collective_destination,
    _write_metadata,
    _broadcast_from_group_rank_zero,
    _load_metadata_collective,
    load_distributed_representation_checkpoint_metadata,
    _read_metadata,
    _normalize_legacy_manifest_defaults,
    _write_bytes_fsync,
    _fsync_directory,
    _plain_cpu_tensor_state,
    _local_shard_state_digest,
    _update_local_shard_digest,
    _digest_tensor_storage,
    _digest_text,
    _load_distributed_checkpoint_api,
    _assert_public_signature,
    _qualified_type,
    _tensor_checksum,
)
_LEGACY_PUBLIC_MODULE = "tgvf_rl.representation.training.distributed_checkpoint"

for _integrity_type in _DISTRIBUTED_CHECKPOINT_INTEGRITY_TYPES:
    freeze_public_class_annotations(
        _integrity_type,
        implementation_globals=globals(),
    )
    rebind_public_class(
        _integrity_type,
        implementation_module=__name__,
        public_module=_LEGACY_PUBLIC_MODULE,
    )
del _integrity_type

for _integrity_function in _DISTRIBUTED_CHECKPOINT_INTEGRITY_FUNCTIONS:
    rebind_public_function(
        _integrity_function,
        implementation_module=__name__,
        public_module=_LEGACY_PUBLIC_MODULE,
    )
del _integrity_function

__all__ = ["load_distributed_representation_checkpoint_metadata"]
