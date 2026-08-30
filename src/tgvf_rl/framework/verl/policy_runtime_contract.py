"""Acyclic contracts shared by Policy runtime composition layers.

The process-local factory imports the concrete Qwen live builder lazily, while
the concrete builder consumes the immutable context and product contracts
defined here.  Keeping these contracts in a neutral leaf prevents the concrete
builder from importing its factory and preserves one-way runtime composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.policy.run_config import PolicyE2ESmokeRunConfig
from tgvf_rl.public_api_compat import rebind_public_class

if TYPE_CHECKING:
    from .native_agent_loop import VerlNativeTrajectoryComponentsPort
    from .policy_weight_sync import PolicyLoRASnapshot, PolicyWeightSyncState


@dataclass(frozen=True, slots=True)
class PolicyAgentLoopWorkerPlacement:
    """Deterministic upstream-worker to configured GPU assignment."""

    worker_index: int
    logical_gpu_id: int
    physical_gpu_id: int
    world_size: int

    def __post_init__(self) -> None:
        for name in ("worker_index", "logical_gpu_id", "physical_gpu_id"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.world_size) is not int or self.world_size != 4:
            raise ValueError("Policy Pilot worker placement requires world_size=4")
        if self.worker_index >= self.world_size:
            raise ValueError("AgentLoop worker index lies outside the four-GPU world")

    @property
    def torch_device(self) -> torch.device:
        """Logical CUDA device used under the run-bound visible-device order."""

        return torch.device("cuda", self.logical_gpu_id)


class PolicyLoRASnapshotConsumer(Protocol):
    """Install and prove the exact LoRA state used for local behavior forwards."""

    def apply_policy_lora_snapshot(
        self, snapshot: "PolicyLoRASnapshot", /
    ) -> PolicyVersion: ...


@dataclass(frozen=True, slots=True)
class PolicyE2ERuntimeBuildContext:
    """Identity-complete inputs supplied to one live runtime builder."""

    config: PolicyE2ESmokeRunConfig
    placement: PolicyAgentLoopWorkerPlacement
    initial_snapshot: "PolicyLoRASnapshot"
    weight_sync_state: "PolicyWeightSyncState"
    trainer_config: object
    server_manager: object
    tokenizer: object
    processor: object
    dataset_cls: object
    data_config: object

    def __post_init__(self) -> None:
        if not isinstance(self.config, PolicyE2ESmokeRunConfig):
            raise TypeError("runtime context requires PolicyE2ESmokeRunConfig")
        if not isinstance(self.placement, PolicyAgentLoopWorkerPlacement):
            raise TypeError("runtime context requires a worker placement")
        if self.initial_snapshot.policy_version.run_id != self.config.run_id:
            raise IdentityMismatchError(
                "initial local LoRA snapshot belongs to another run"
            )


@dataclass(frozen=True, slots=True)
class PolicyE2ERuntimeProduct:
    """Real trajectory components and the local exact-weight consumer."""

    trajectory_components: VerlNativeTrajectoryComponentsPort
    snapshot_consumer: PolicyLoRASnapshotConsumer

    def __post_init__(self) -> None:
        sync_builder = getattr(
            self.trajectory_components, "build_trajectory_components", None
        )
        async_builder = getattr(
            self.trajectory_components, "build_trajectory_components_async", None
        )
        if not callable(sync_builder) and not callable(async_builder):
            raise TypeError(
                "runtime product must provide a sync or async trajectory builder"
            )
        if not callable(
            getattr(self.snapshot_consumer, "apply_policy_lora_snapshot", None)
        ):
            raise TypeError("runtime product must provide apply_policy_lora_snapshot()")


# These contracts historically lived in ``policy_runtime``.  Keep the public
# module identity so existing pickles and fully-qualified introspection remain
# valid while that module re-exports these exact objects.
_LEGACY_PUBLIC_MODULE = "tgvf_rl.framework.verl.policy_runtime"
for _public_contract in (
    PolicyAgentLoopWorkerPlacement,
    PolicyLoRASnapshotConsumer,
    PolicyE2ERuntimeBuildContext,
    PolicyE2ERuntimeProduct,
):
    rebind_public_class(
        _public_contract,
        implementation_module=__name__,
        public_module=_LEGACY_PUBLIC_MODULE,
    )
del _public_contract


__all__ = [
    "PolicyAgentLoopWorkerPlacement",
    "PolicyE2ERuntimeBuildContext",
    "PolicyE2ERuntimeProduct",
    "PolicyLoRASnapshotConsumer",
]
