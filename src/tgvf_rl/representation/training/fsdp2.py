"""Composable FSDP2 ownership boundary for representation-phase TGVF training.

The TGVF Adapter registers four frozen projection modules borrowed from Qwen.
Calling ``fully_shard(adapter)`` without an ignore set would incorrectly move
those shared parameters into the Adapter's FSDP ownership domain. The Adapter
also returns a custom dataclass, so a parameter-owning FSDP root cannot install
the tensor pre-backward hook required after resharding. This module instead
forms one FSDP group from the 52 fixed Adapter-owned tensor-returning leaves,
then installs a parameterless orchestration root over the Adapter. Every
borrowed merger parameter is explicitly ignored in both calls.

The returned borrowed-parameter set must also be ignored if the frozen Qwen
model is independently wrapped.  A Qwen FSDP root cannot own these parameters:
the TGVF Adapter calls each merger directly, outside a Qwen-root forward hook.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import torch
from torch import nn

from tgvf_rl.compatibility_stack import AUDITED_COMPATIBILITY_STACKS
from tgvf_rl.representation.adapter import TGVFAdapter


_BORROWED_PROJECTION_PREFIXES = (
    "main_projection.",
    "d_deepstack_projections.",
)
SUPPORTED_REPRESENTATION_TORCH_IDENTITIES = (
    *(
        (stack.torch_distribution_version, stack.torch_runtime_version)
        for stack in AUDITED_COMPATIBILITY_STACKS.values()
    ),
)
_FSDP_MESH_DIM_NAME = "fsdp"
_EXPECTED_FULLY_SHARD_PARAMETERS = (
    "module",
    "mesh",
    "reshard_after_forward",
    "shard_placement_fn",
    "mp_policy",
    "offload_policy",
    "ignored_params",
)
_EXPECTED_MIXED_PRECISION_POLICY_PARAMETERS = (
    "param_dtype",
    "reduce_dtype",
    "output_dtype",
    "cast_forward_inputs",
)
_EXPECTED_SET_REQUIRES_GRADIENT_SYNC_PARAMETERS = (
    "self",
    "requires_gradient_sync",
    "recurse",
)
_EXPECTED_SET_RESHARD_AFTER_BACKWARD_PARAMETERS = (
    "self",
    "reshard_after_backward",
    "recurse",
)
_EXPECTED_SET_IS_LAST_BACKWARD_PARAMETERS = ("self", "is_last_backward")
_OWNED_ATTENTION_LEAF_NAMES = (
    "target_norm",
    "target_proj",
    "visual_norm",
    "visual_proj",
    "target_q_proj",
    "visual_k_proj",
    "visual_v_proj",
    "enriched_target_norm",
    "visual_q_proj",
    "target_k_proj",
    "target_v_proj",
    "context_to_delta",
    "gate_proj",
)


def _require_supported_torch_identity(*, api_name: str) -> tuple[str, str]:
    """Require an exact installed-distribution/runtime pair before API access."""

    try:
        distribution_version = metadata.version("torch")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"representation {api_name} cannot prove the torch distribution identity"
        ) from error
    runtime_version = str(torch.__version__)
    identity = (distribution_version, runtime_version)
    if identity not in SUPPORTED_REPRESENTATION_TORCH_IDENTITIES:
        supported = ", ".join(
            f"distribution={distribution!r}/runtime={runtime!r}"
            for distribution, runtime in SUPPORTED_REPRESENTATION_TORCH_IDENTITIES
        )
        raise RuntimeError(
            f"representation {api_name} requires an exact audited torch identity; "
            f"accepted: {supported}; observed distribution={distribution_version!r}/"
            f"runtime={runtime_version!r}"
        )
    return identity


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


def _assert_public_class(
    value: object,
    *,
    api_name: str,
    expected_module: str,
) -> None:
    if (
        not inspect.isclass(value)
        or value.__name__ != api_name
        or value.__module__ != expected_module
    ):
        raise RuntimeError(f"torch {api_name} public class identity drifted")


@dataclass(frozen=True, slots=True)
class RepresentationFSDP2Config:
    """Explicit topology choices for the trainable TGVF Adapter."""

    world_size: int
    reshard_after_forward: bool

    def __post_init__(self) -> None:
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise TypeError("FSDP2 world_size must be an integer")
        if self.world_size < 2:
            raise ValueError("representation FSDP2 requires at least two ranks")
        if not isinstance(self.reshard_after_forward, bool):
            raise TypeError("reshard_after_forward must be an explicit bool")


@dataclass(frozen=True, slots=True)
class RepresentationFSDP2Plan:
    """Read-only parameter inventory; safe to build on CPU or meta devices."""

    owned_parameter_names: tuple[str, ...]
    borrowed_parameter_names: tuple[str, ...]
    borrowed_buffer_names: tuple[str, ...]
    owned_group_module_names: tuple[str, ...]
    owned_parameter_numel: int
    borrowed_parameter_numel: int

    def __post_init__(self) -> None:
        if not self.owned_parameter_names:
            raise ValueError("FSDP2 plan must contain Adapter-owned parameters")
        if not self.borrowed_parameter_names:
            raise ValueError("FSDP2 plan must contain borrowed Qwen mergers")
        if len(set(self.owned_parameter_names)) != len(self.owned_parameter_names):
            raise ValueError("Adapter-owned FSDP2 parameter names must be unique")
        if len(set(self.borrowed_parameter_names)) != len(
            self.borrowed_parameter_names
        ):
            raise ValueError("borrowed FSDP2 parameter names must be unique")
        if set(self.owned_parameter_names) & set(self.borrowed_parameter_names):
            raise ValueError("owned and borrowed FSDP2 names cannot overlap")
        if len(self.owned_group_module_names) != 52 or len(
            set(self.owned_group_module_names)
        ) != len(self.owned_group_module_names):
            raise ValueError("FSDP2 requires 52 unique Adapter-owned leaf modules")
        if self.owned_parameter_numel <= 0 or self.borrowed_parameter_numel <= 0:
            raise ValueError("FSDP2 parameter inventories must be non-empty")


@dataclass(frozen=True, slots=True)
class RepresentationFSDP2Binding:
    """Post-shard contract used to construct and audit the optimizer."""

    adapter: TGVFAdapter
    config: RepresentationFSDP2Config
    mesh: Any
    plan: RepresentationFSDP2Plan
    _borrowed_qwen_merger_parameters: tuple[nn.Parameter, ...]
    _borrowed_qwen_merger_buffers: tuple[torch.Tensor, ...]
    _owned_group_modules: tuple[nn.Module, ...]

    @property
    def borrowed_qwen_merger_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters that both Adapter and frozen-Qwen FSDP must ignore."""

        return self._borrowed_qwen_merger_parameters

    @property
    def borrowed_qwen_merger_buffers(self) -> tuple[torch.Tensor, ...]:
        return self._borrowed_qwen_merger_buffers

    @property
    def owned_group_modules(self) -> tuple[nn.Module, ...]:
        return self._owned_group_modules

    def optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return every and only current FSDP-managed trainable Adapter parameter."""

        current = _partition_adapter_parameters(self.adapter)
        if tuple(name for name, _ in current.owned) != self.plan.owned_parameter_names:
            raise RuntimeError("FSDP2 changed Adapter-owned parameter names")
        if (
            tuple(name for name, _ in current.borrowed)
            != self.plan.borrowed_parameter_names
        ):
            raise RuntimeError("FSDP2 changed borrowed Qwen parameter names")
        if tuple(id(parameter) for _, parameter in current.borrowed) != tuple(
            id(parameter) for parameter in self._borrowed_qwen_merger_parameters
        ):
            raise RuntimeError("FSDP2 replaced a borrowed Qwen merger parameter")
        return tuple(parameter for _, parameter in current.owned)

    def assert_optimizer_ownership(self, optimizer: torch.optim.Optimizer) -> None:
        """Reject optimizers containing borrowed, stale, missing, or duplicate state."""

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        actual = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group.get("params", ())
        )
        if any(not isinstance(parameter, nn.Parameter) for parameter in actual):
            raise TypeError("optimizer parameter groups must contain Parameters")
        actual_ids = tuple(id(parameter) for parameter in actual)
        expected_ids = tuple(id(parameter) for parameter in self.optimizer_parameters())
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError("optimizer contains duplicate Adapter parameters")
        if set(actual_ids) != set(expected_ids):
            raise ValueError(
                "optimizer must contain every and only FSDP2 Adapter-owned parameter"
            )
        borrowed_ids = {
            id(parameter) for parameter in self._borrowed_qwen_merger_parameters
        }
        if borrowed_ids & set(actual_ids):
            raise ValueError("optimizer contains a borrowed Qwen merger parameter")

    def begin_microstep(self, *, index: int, count: int) -> None:
        """Configure FSDP2 for one exact gradient-accumulation microstep.

        Non-final microsteps retain unsharded Adapter parameters and accumulate
        unreduced gradients locally.  The final microstep performs the one
        required reduce-scatter, matching DDP ``no_sync`` accumulation without
        changing the global loss normalization.
        """

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("FSDP2 accumulation microstep index must be an integer")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("FSDP2 accumulation microstep count must be an integer")
        if count <= 0 or index < 0 or index >= count:
            raise ValueError("FSDP2 accumulation microstep lies outside its window")
        final = index == count - 1
        _set_fsdp2_accumulation_state(
            self.adapter,
            requires_gradient_sync=final,
            reshard_after_backward=final,
            is_last_backward=final,
        )

    def finish_window(self) -> None:
        """Restore synchronized policy flags after an accumulation window.

        This is not an abort primitive: a partially failed window may retain
        private FSDP accumulation state, so the trainer is deliberately
        fail-stop after any exception.
        """

        _set_fsdp2_accumulation_state(
            self.adapter,
            requires_gradient_sync=True,
            reshard_after_backward=True,
            is_last_backward=True,
        )


def _set_fsdp2_accumulation_state(
    module: nn.Module,
    *,
    requires_gradient_sync: bool,
    reshard_after_backward: bool,
    is_last_backward: bool,
) -> None:
    """Apply the three composable-FSDP accumulation controls consistently."""

    values = (
        requires_gradient_sync,
        reshard_after_backward,
        is_last_backward,
    )
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("FSDP2 accumulation controls must be bool")
    methods = tuple(
        getattr(module, name, None)
        for name in (
            "set_requires_gradient_sync",
            "set_reshard_after_backward",
            "set_is_last_backward",
        )
    )
    if any(not callable(method) for method in methods):
        raise TypeError("bound Adapter does not expose composable-FSDP controls")
    set_sync, set_reshard, set_last = methods
    set_sync(requires_gradient_sync, recurse=True)
    set_reshard(reshard_after_backward, recurse=True)
    set_last(is_last_backward)


@dataclass(frozen=True, slots=True)
class _ParameterPartition:
    owned: tuple[tuple[str, nn.Parameter], ...]
    borrowed: tuple[tuple[str, nn.Parameter], ...]


@dataclass(frozen=True, slots=True)
class _BorrowedBufferInventory:
    names: tuple[str, ...]
    tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class _FSDP2API:
    fully_shard: Callable[..., Any]
    fsdp_module_type: type
    mixed_precision_policy_type: type
    offload_policy_type: type
    device_mesh_type: type
    dtensor_type: type

    def is_fully_sharded(self, module: nn.Module) -> bool:
        return isinstance(module, self.fsdp_module_type)


def build_representation_fsdp2_plan(
    adapter: TGVFAdapter,
    qwen_model: nn.Module,
) -> RepresentationFSDP2Plan:
    """Audit shared parameter ownership without initializing distributed state."""

    partition, buffers = _audit_adapter_qwen_ownership(adapter, qwen_model)
    group_names, _ = _owned_group_modules(adapter, partition)
    return RepresentationFSDP2Plan(
        owned_parameter_names=tuple(name for name, _ in partition.owned),
        borrowed_parameter_names=tuple(name for name, _ in partition.borrowed),
        borrowed_buffer_names=buffers.names,
        owned_group_module_names=group_names,
        owned_parameter_numel=sum(
            parameter.numel() for _, parameter in partition.owned
        ),
        borrowed_parameter_numel=sum(
            parameter.numel() for _, parameter in partition.borrowed
        ),
    )


def apply_representation_fsdp2(
    *,
    adapter: TGVFAdapter,
    qwen_model: nn.Module,
    mesh: Any,
    config: RepresentationFSDP2Config,
    mixed_precision_policy: Any,
    offload_policy: Any,
) -> RepresentationFSDP2Binding:
    """Apply a supported torch composable FSDP2 to Adapter-owned parameters only.

    The optimizer must be created *after* this function and audited with
    :meth:`RepresentationFSDP2Binding.assert_optimizer_ownership`.
    """

    if not isinstance(config, RepresentationFSDP2Config):
        raise TypeError("config must be RepresentationFSDP2Config")
    api = _load_fsdp2_api()
    if not isinstance(mixed_precision_policy, api.mixed_precision_policy_type):
        raise TypeError("mixed_precision_policy must be torch FSDP2 policy")
    if not isinstance(offload_policy, api.offload_policy_type):
        raise TypeError("offload_policy must be torch FSDP2 policy")
    if any(api.is_fully_sharded(module) for module in adapter.modules()):
        raise ValueError("TGVF Adapter already contains composable FSDP state")

    partition, buffers = _audit_adapter_qwen_ownership(adapter, qwen_model)
    if any(
        isinstance(parameter, api.dtensor_type)
        for _, parameter in (*partition.owned, *partition.borrowed)
    ):
        raise ValueError("Adapter parameters were already DTensor-sharded")
    _assert_distributed_prerequisites(
        adapter=adapter,
        mesh=mesh,
        config=config,
        api=api,
    )
    plan = build_representation_fsdp2_plan(adapter, qwen_model)
    group_names, owned_group_modules = _owned_group_modules(adapter, partition)
    if group_names != plan.owned_group_module_names:
        raise RuntimeError("Adapter-owned FSDP2 module inventory changed")
    borrowed_parameters = tuple(parameter for _, parameter in partition.borrowed)
    borrowed_parameter_ids = tuple(id(parameter) for parameter in borrowed_parameters)
    borrowed_buffer_ids = tuple(id(buffer) for buffer in buffers.tensors)
    borrowed_buffer_devices = tuple(buffer.device for buffer in buffers.tensors)

    group_argument = list(owned_group_modules)
    group_result = api.fully_shard(
        group_argument,
        mesh=mesh,
        reshard_after_forward=config.reshard_after_forward,
        mp_policy=mixed_precision_policy,
        offload_policy=offload_policy,
        ignored_params=set(borrowed_parameters),
    )
    if group_result is not group_argument:
        raise RuntimeError("composable fully_shard must mutate the owned module list")
    if any(not api.is_fully_sharded(module) for module in owned_group_modules):
        raise RuntimeError("composable fully_shard did not attach owned-group state")

    # This second call establishes the single root required by composable FSDP.
    # All owned parameters already belong to the leaf group and every borrowed
    # parameter is ignored, so the Adapter root owns no parameter group itself.
    result = api.fully_shard(
        adapter,
        mesh=mesh,
        reshard_after_forward=config.reshard_after_forward,
        mp_policy=mixed_precision_policy,
        offload_policy=offload_policy,
        ignored_params=set(borrowed_parameters),
    )
    if result is not adapter:
        raise RuntimeError("composable fully_shard must mutate and return the Adapter")
    if not api.is_fully_sharded(adapter):
        raise RuntimeError("composable fully_shard did not attach FSDP state")

    after, after_buffers = _audit_adapter_qwen_ownership(adapter, qwen_model)
    if tuple(name for name, _ in after.owned) != plan.owned_parameter_names:
        raise RuntimeError("FSDP2 changed Adapter-owned parameter names")
    if tuple(name for name, _ in after.borrowed) != plan.borrowed_parameter_names:
        raise RuntimeError("FSDP2 changed borrowed Qwen parameter names")
    after_borrowed = tuple(parameter for _, parameter in after.borrowed)
    if tuple(id(parameter) for parameter in after_borrowed) != borrowed_parameter_ids:
        raise RuntimeError("FSDP2 sharded or replaced a borrowed Qwen parameter")
    if after_buffers.names != buffers.names:
        raise RuntimeError("FSDP2 changed borrowed Qwen buffer names")
    if tuple(id(buffer) for buffer in after_buffers.tensors) != borrowed_buffer_ids:
        raise RuntimeError("FSDP2 replaced a borrowed Qwen buffer")
    if (
        tuple(buffer.device for buffer in after_buffers.tensors)
        != borrowed_buffer_devices
    ):
        raise RuntimeError("FSDP2 moved a borrowed Qwen buffer")

    return RepresentationFSDP2Binding(
        adapter=adapter,
        config=config,
        mesh=mesh,
        plan=plan,
        _borrowed_qwen_merger_parameters=after_borrowed,
        _borrowed_qwen_merger_buffers=after_buffers.tensors,
        _owned_group_modules=owned_group_modules,
    )


def _audit_adapter_qwen_ownership(
    adapter: TGVFAdapter,
    qwen_model: nn.Module,
) -> tuple[_ParameterPartition, _BorrowedBufferInventory]:
    if not isinstance(adapter, TGVFAdapter):
        raise TypeError("adapter must be TGVFAdapter")
    if not isinstance(qwen_model, nn.Module):
        raise TypeError("qwen_model must be an nn.Module")
    if qwen_model.training or any(module.training for module in qwen_model.modules()):
        raise ValueError("frozen Qwen must remain entirely in eval mode")
    qwen_parameters = tuple(qwen_model.parameters())
    if not qwen_parameters:
        raise ValueError("qwen_model must own the borrowed merger parameters")
    if any(parameter.requires_grad for parameter in qwen_parameters):
        raise ValueError("every Qwen parameter must remain frozen")

    partition = _partition_adapter_parameters(adapter)
    owned_ids = {id(parameter) for _, parameter in partition.owned}
    borrowed_ids = {id(parameter) for _, parameter in partition.borrowed}
    qwen_ids = {id(parameter) for parameter in qwen_parameters}
    if owned_ids & borrowed_ids:
        raise RuntimeError("Adapter-owned and borrowed parameter identities overlap")
    if owned_ids & qwen_ids:
        raise ValueError("Adapter-owned parameter is incorrectly registered by Qwen")
    if not borrowed_ids.issubset(qwen_ids):
        raise ValueError("borrowed Adapter merger is not owned by the selected Qwen")

    merger_modules = _borrowed_merger_modules(adapter)
    merger_ids = {
        id(parameter) for module in merger_modules for parameter in module.parameters()
    }
    if merger_ids != borrowed_ids:
        raise RuntimeError("borrowed parameter prefixes differ from merger ownership")
    if any(parameter.requires_grad for _, parameter in partition.borrowed):
        raise ValueError("borrowed Qwen merger parameters must remain frozen")
    if any(not parameter.requires_grad for _, parameter in partition.owned):
        frozen = tuple(
            name for name, parameter in partition.owned if not parameter.requires_grad
        )
        raise ValueError(f"Adapter-owned parameters unexpectedly frozen: {frozen}")
    owned_dtypes = {parameter.dtype for _, parameter in partition.owned}
    if len(owned_dtypes) != 1 or any(
        not dtype.is_floating_point for dtype in owned_dtypes
    ):
        raise ValueError(
            "Adapter-owned parameters must share one floating dtype before FSDP2"
        )
    if any(
        not parameter.dtype.is_floating_point for _, parameter in partition.borrowed
    ):
        raise ValueError("borrowed Qwen merger parameters must be floating point")
    if any(
        isinstance(module, nn.Dropout) and module.p != 0.0
        for module in adapter.modules()
    ):
        raise ValueError("representation Adapter dropout must be zero")
    return partition, _borrowed_buffers(merger_modules)


def _partition_adapter_parameters(adapter: TGVFAdapter) -> _ParameterPartition:
    named = tuple(adapter.named_parameters())
    if not named:
        raise ValueError("TGVF Adapter must expose parameters")
    owned = tuple(
        (name, parameter)
        for name, parameter in named
        if not name.startswith(_BORROWED_PROJECTION_PREFIXES)
    )
    borrowed = tuple(
        (name, parameter)
        for name, parameter in named
        if name.startswith(_BORROWED_PROJECTION_PREFIXES)
    )
    if not owned:
        raise ValueError("TGVF Adapter has no trainable-owned parameters")
    if not borrowed:
        raise ValueError("TGVF Adapter has no borrowed Qwen merger parameters")
    return _ParameterPartition(owned=owned, borrowed=borrowed)


def _owned_group_modules(
    adapter: TGVFAdapter,
    partition: _ParameterPartition,
) -> tuple[tuple[str, ...], tuple[nn.Module, ...]]:
    prefixes = ("",) + tuple(
        f"d_deepstack_branch_adapters.{layer}."
        for layer in adapter.d_deepstack_branch_layers
    )
    expected_names = tuple(
        f"{prefix}{leaf}" for prefix in prefixes for leaf in _OWNED_ATTENTION_LEAF_NAMES
    )
    named_modules = dict(adapter.named_modules())
    if any(name not in named_modules for name in expected_names):
        missing = tuple(name for name in expected_names if name not in named_modules)
        raise RuntimeError(f"Adapter-owned FSDP2 leaf modules are missing: {missing}")
    modules = tuple(named_modules[name] for name in expected_names)
    if len({id(module) for module in modules}) != len(modules):
        raise RuntimeError("Adapter-owned FSDP2 leaf modules must be distinct")
    if any(not isinstance(module, (nn.Linear, nn.LayerNorm)) for module in modules):
        raise TypeError("Adapter-owned FSDP2 leaves must return plain tensors")

    owned_ids = {id(parameter) for _, parameter in partition.owned}
    group_ids = {
        id(parameter)
        for module in modules
        for parameter in module.parameters(recurse=False)
    }
    if group_ids != owned_ids:
        raise RuntimeError(
            "Adapter-owned FSDP2 leaves do not cover exactly the trainable parameters"
        )
    return expected_names, modules


def _borrowed_merger_modules(adapter: TGVFAdapter) -> tuple[nn.Module, ...]:
    modules = (
        adapter.main_projection.projection,
        *tuple(port.projection for port in adapter.d_deepstack_projections.projections),
    )
    if len(modules) != 4 or len({id(module) for module in modules}) != 4:
        raise ValueError("TGVF Adapter requires four distinct borrowed Qwen mergers")
    return modules


def _borrowed_buffers(modules: tuple[nn.Module, ...]) -> _BorrowedBufferInventory:
    names: list[str] = []
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()
    for index, module in enumerate(modules):
        for name, buffer in module.named_buffers():
            if id(buffer) in seen:
                continue
            seen.add(id(buffer))
            names.append(f"merger.{index}.{name}")
            tensors.append(buffer)
    return _BorrowedBufferInventory(names=tuple(names), tensors=tuple(tensors))


def _assert_distributed_prerequisites(
    *,
    adapter: TGVFAdapter,
    mesh: Any,
    config: RepresentationFSDP2Config,
    api: _FSDP2API,
) -> None:
    distributed = torch.distributed
    if not distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable; FSDP2 cannot start")
    if not distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before FSDP2")
    actual_world_size = distributed.get_world_size()
    if actual_world_size != config.world_size:
        raise ValueError(
            "distributed world size differs from representation FSDP2 config: "
            f"expected={config.world_size} actual={actual_world_size}"
        )
    if not isinstance(mesh, api.device_mesh_type):
        raise TypeError("mesh must be a torch DeviceMesh")
    if mesh.ndim != 1:
        raise ValueError("representation FSDP2 requires a one-dimensional mesh")
    if mesh.device_type != "cuda":
        raise ValueError("representation FSDP2 production mesh must use CUDA")
    if mesh.mesh_dim_names != (_FSDP_MESH_DIM_NAME,):
        raise ValueError("representation FSDP2 mesh dimension must be named 'fsdp'")
    if mesh.size(0) != config.world_size:
        raise ValueError("FSDP2 mesh size differs from configured world size")
    group = mesh.get_group(0)
    if distributed.get_backend(group) != "nccl":
        raise ValueError("representation FSDP2 requires an NCCL process group")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; representation FSDP2 cannot start")

    devices = {parameter.device for parameter in adapter.parameters()}
    if len(devices) != 1:
        raise ValueError(
            "all Adapter and borrowed merger parameters must share a device"
        )
    (device,) = tuple(devices)
    if device.type != "cuda" or device.index != torch.cuda.current_device():
        raise ValueError("Adapter parameters must be on the current CUDA rank device")


def _load_fsdp2_api() -> _FSDP2API:
    _require_supported_torch_identity(api_name="FSDP2 binding")
    try:
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.fsdp import (
            FSDPModule,
            MixedPrecisionPolicy,
            OffloadPolicy,
            fully_shard,
        )
        from torch.distributed.tensor import DTensor
    except (ImportError, AttributeError) as error:
        raise RuntimeError("torch composable FSDP2 APIs are unavailable") from error

    _assert_public_signature(
        fully_shard,
        api_name="fully_shard",
        expected_parameters=_EXPECTED_FULLY_SHARD_PARAMETERS,
    )
    _assert_public_signature(
        MixedPrecisionPolicy,
        api_name="MixedPrecisionPolicy",
        expected_parameters=_EXPECTED_MIXED_PRECISION_POLICY_PARAMETERS,
    )
    _assert_public_signature(
        OffloadPolicy,
        api_name="OffloadPolicy",
        expected_parameters=(),
    )
    _assert_public_signature(
        FSDPModule.set_requires_gradient_sync,
        api_name="FSDPModule.set_requires_gradient_sync",
        expected_parameters=_EXPECTED_SET_REQUIRES_GRADIENT_SYNC_PARAMETERS,
    )
    _assert_public_signature(
        FSDPModule.set_reshard_after_backward,
        api_name="FSDPModule.set_reshard_after_backward",
        expected_parameters=_EXPECTED_SET_RESHARD_AFTER_BACKWARD_PARAMETERS,
    )
    _assert_public_signature(
        FSDPModule.set_is_last_backward,
        api_name="FSDPModule.set_is_last_backward",
        expected_parameters=_EXPECTED_SET_IS_LAST_BACKWARD_PARAMETERS,
    )
    for value, api_name, expected_module in (
        (FSDPModule, "FSDPModule", "torch.distributed.fsdp"),
        (MixedPrecisionPolicy, "MixedPrecisionPolicy", "torch.distributed.fsdp"),
        (OffloadPolicy, "OffloadPolicy", "torch.distributed.fsdp"),
        (DeviceMesh, "DeviceMesh", "torch.distributed.device_mesh"),
        (DTensor, "DTensor", "torch.distributed.tensor"),
    ):
        _assert_public_class(
            value,
            api_name=api_name,
            expected_module=expected_module,
        )
    return _FSDP2API(
        fully_shard=fully_shard,
        fsdp_module_type=FSDPModule,
        mixed_precision_policy_type=MixedPrecisionPolicy,
        offload_policy_type=OffloadPolicy,
        device_mesh_type=DeviceMesh,
        dtensor_type=DTensor,
    )


__all__ = [
    "SUPPORTED_REPRESENTATION_TORCH_IDENTITIES",
    "RepresentationFSDP2Binding",
    "RepresentationFSDP2Config",
    "RepresentationFSDP2Plan",
    "apply_representation_fsdp2",
    "build_representation_fsdp2_plan",
]
