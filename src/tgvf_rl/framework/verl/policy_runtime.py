"""Process-local composition boundary for the Policy RL native agent loop.

Pinned veRL instantiates the Hydra ``invocation_factory`` partial once for
every configured agent-loop entry.  That construction surface must not create
another local Qwen model, reset the n=8 rollout counter, or forget the exact
LoRA version already served by vLLM.  This module therefore owns one
process-local runtime per run identity and upstream AgentLoop worker.

The boundary deliberately does not synthesize visual observations or behavior
log probabilities.  A live builder must provide the existing trajectory
components port and an exact LoRA snapshot consumer.  Until that builder is
registered, construction fails with a precise error rather than running a
text-only or stale-policy substitute.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import hmac
import os
from pathlib import Path
import re
from threading import RLock
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.framework.vllm import (
    VLLMOutputDecodingContract,
    VLLMTerminationOutcome,
    VLLMTurnTerminationContract,
    qwen3_vl_final_turn_outcomes,
)
from tgvf_rl.policy.run_config import (
    POLICY_E2E_CROP_TFREE_EXACT_MATCHED_RUN_CONFIG_SCHEMA,
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)

from .native_agent_loop import (
    BoundVerlNativeAgentLoopInvocationFactory,
    VerlNativeAgentLoopInvocation,
    VerlNativeTrajectoryComponentsPort,
)

if TYPE_CHECKING:
    from tgvf_rl.framework.verl.policy_weight_sync import (
        PolicyFullQwenSyncReceipt,
        PolicyLoRASnapshot,
        PolicyWeightSyncState,
    )


POLICY_E2E_RUNTIME_SCHEMA = "tgvf-policy-e2e-agent-loop-runtime-v1"
POLICY_AGENT_LOOP_WORKER_INDEX_ENV = "TGVF_POLICY_AGENT_LOOP_WORKER_INDEX"
_AGENT_LOOP_WORKER_NAME = re.compile(
    r"^agent_loop_worker_(?P<index>[0-9]+)(?:_[0-9a-fA-F]{8})?$"
)


class PolicyE2ELiveRuntimeUnavailableError(RuntimeError):
    """Raised when the real model/tool/replay composition is not installed."""


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
        if type(self.world_size) is not int or self.world_size <= 0:
            raise ValueError("Policy worker placement requires positive world_size")
        if self.worker_index >= self.world_size:
            raise ValueError("AgentLoop worker index lies outside the configured world")

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
    initial_snapshot: "PolicyLoRASnapshot | PolicyFullQwenSyncReceipt"
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
                "initial local policy version belongs to another run"
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
            raise TypeError(
                "runtime product must provide apply_policy_lora_snapshot()"
            )


class PolicyE2ERuntimeBuilder(Protocol):
    """Heavy process-local model/tool/replay composer."""

    @property
    def singleton_identity(self) -> str: ...

    def build(self, context: PolicyE2ERuntimeBuildContext, /) -> PolicyE2ERuntimeProduct: ...


class PeftPolicyLoRASnapshotConsumer:
    """Apply upstream's PEFT state-dict view and verify it byte-for-byte.

    The snapshot is captured from the same ``get_peft_model_state_dict`` stream
    sent to vLLM.  Loading and reading through PEFT's matching public helpers
    avoids guessing whether adapter-name segments are present in model-owned
    parameter names.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        adapter_name: str = "default",
        model_lock: RLock | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("PEFT snapshot consumer requires an nn.Module")
        if not getattr(model, "peft_config", None):
            raise ValueError("PEFT snapshot consumer requires an attached adapter")
        if not isinstance(adapter_name, str) or not adapter_name:
            raise ValueError("PEFT adapter_name must be non-empty")
        if adapter_name not in model.peft_config:
            raise ValueError("selected PEFT adapter is not attached to the model")
        self.model = model
        self.adapter_name = adapter_name
        self._lock = model_lock or RLock()

    def apply_policy_lora_snapshot(
        self, snapshot: "PolicyLoRASnapshot", /
    ) -> PolicyVersion:
        _require_policy_lora_snapshot(snapshot)
        try:
            from peft.utils.save_and_load import (
                get_peft_model_state_dict,
                set_peft_model_state_dict,
            )
        except ImportError as error:  # pragma: no cover - accepted env owns PEFT
            raise RuntimeError("exact local LoRA loading requires PEFT") from error

        source = {name: tensor.detach().cpu() for name, tensor in snapshot.tensors.items()}
        with self._lock, torch.no_grad():
            set_peft_model_state_dict(
                self.model,
                source,
                adapter_name=self.adapter_name,
            )
            installed = get_peft_model_state_dict(
                self.model,
                adapter_name=self.adapter_name,
            )
            installed = {
                name: tensor.detach().to(device="cpu").contiguous()
                for name, tensor in installed.items()
            }
        if set(installed) != set(source):
            raise ReplayMismatchError(
                "local PEFT state keys differ from the exact vLLM LoRA snapshot"
            )
        for name in sorted(source):
            expected = source[name]
            actual = installed[name]
            if (
                actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not torch.equal(actual, expected)
            ):
                raise ReplayMismatchError(
                    f"local PEFT tensor {name!r} differs from the exact snapshot"
                )
        from .policy_weight_sync import lora_parameter_mapping_sha256

        if lora_parameter_mapping_sha256(installed) != snapshot.policy_version.weights_sha256:
            raise ReplayMismatchError(
                "local PEFT state digest differs from the served behavior policy"
            )
        return snapshot.policy_version


class ExactLoRASnapshotPolicyVersionPort:
    """Refresh local behavior weights from the latest committed exact snapshot."""

    def __init__(
        self,
        *,
        state: "PolicyWeightSyncState",
        consumer: PolicyLoRASnapshotConsumer,
        initial_snapshot: "PolicyLoRASnapshot",
        snapshot_loader: Callable[..., "PolicyLoRASnapshot"] | None = None,
    ) -> None:
        if not callable(getattr(consumer, "apply_policy_lora_snapshot", None)):
            raise TypeError("snapshot consumer must implement apply_policy_lora_snapshot()")
        _require_policy_lora_snapshot(initial_snapshot)
        if initial_snapshot.policy_version.run_id != state.run_id:
            raise IdentityMismatchError("initial snapshot and weight-sync run differ")
        if initial_snapshot.run_identity_sha256 != state.run_identity_sha256:
            raise IdentityMismatchError("initial snapshot and run identities differ")
        self.state = state
        self.consumer = consumer
        self.snapshot_loader = snapshot_loader or _load_latest_snapshot
        self._lock = RLock()
        self._current = self._apply(initial_snapshot)
        self._latest_pointer_signature = _latest_pointer_signature(state)

    def current_policy_version(self) -> PolicyVersion:
        with self._lock:
            pointer_signature = _latest_pointer_signature(self.state)
            if (
                pointer_signature is not None
                and pointer_signature == self._latest_pointer_signature
            ):
                return self._current
            snapshot = self.snapshot_loader(self.state)
            _require_policy_lora_snapshot(snapshot)
            candidate = snapshot.policy_version
            if candidate.run_id != self.state.run_id:
                raise IdentityMismatchError("latest LoRA snapshot belongs to another run")
            if snapshot.run_identity_sha256 != self.state.run_identity_sha256:
                raise IdentityMismatchError("latest LoRA snapshot run identity changed")
            if candidate.optimizer_step < self._current.optimizer_step:
                raise ReplayMismatchError("latest LoRA snapshot regressed in optimizer step")
            if candidate != self._current:
                if candidate.optimizer_step == self._current.optimizer_step:
                    raise ReplayMismatchError(
                        "one optimizer step was published with two weight identities"
                    )
                self._current = self._apply(snapshot)
            self._latest_pointer_signature = pointer_signature
            return self._current

    def _apply(self, snapshot: "PolicyLoRASnapshot") -> PolicyVersion:
        installed = self.consumer.apply_policy_lora_snapshot(snapshot)
        if not isinstance(installed, PolicyVersion):
            raise TypeError("snapshot consumer must return PolicyVersion")
        if installed != snapshot.policy_version:
            raise IdentityMismatchError(
                "snapshot consumer did not prove the exact served policy version"
            )
        return installed


class ExactFullQwenSyncPolicyVersionPort:
    """Follow only completed upstream full-model synchronization receipts."""

    def __init__(
        self,
        *,
        state: "PolicyWeightSyncState",
        initial_receipt: "PolicyFullQwenSyncReceipt",
        receipt_loader: Callable[..., "PolicyFullQwenSyncReceipt"] | None = None,
    ) -> None:
        _require_full_qwen_sync_receipt(initial_receipt)
        if initial_receipt.policy_version.run_id != state.run_id:
            raise IdentityMismatchError("initial full-Qwen receipt and run differ")
        if initial_receipt.run_identity_sha256 != state.run_identity_sha256:
            raise IdentityMismatchError(
                "initial full-Qwen receipt run identity differs"
            )
        self.state = state
        self.receipt_loader = receipt_loader or _load_latest_full_qwen_receipt
        self.base_weights_sha256 = initial_receipt.base_weights_sha256
        self._lock = RLock()
        self._current = initial_receipt.policy_version
        self._latest_pointer_signature = _latest_full_qwen_pointer_signature(state)

    def current_policy_version(self) -> PolicyVersion:
        with self._lock:
            pointer_signature = _latest_full_qwen_pointer_signature(self.state)
            if (
                pointer_signature is not None
                and pointer_signature == self._latest_pointer_signature
            ):
                return self._current
            receipt = self.receipt_loader(
                self.state,
                expected_base_weights_sha256=self.base_weights_sha256,
            )
            _require_full_qwen_sync_receipt(receipt)
            candidate = receipt.policy_version
            if candidate.run_id != self.state.run_id:
                raise IdentityMismatchError(
                    "latest full-Qwen receipt belongs to another run"
                )
            if receipt.run_identity_sha256 != self.state.run_identity_sha256:
                raise IdentityMismatchError(
                    "latest full-Qwen receipt run identity changed"
                )
            if candidate.optimizer_step < self._current.optimizer_step:
                raise ReplayMismatchError(
                    "latest full-Qwen receipt regressed in optimizer step"
                )
            if candidate != self._current:
                if candidate.optimizer_step == self._current.optimizer_step:
                    raise ReplayMismatchError(
                        "one optimizer step published two full-Qwen identities"
                    )
                self._current = candidate
            self._latest_pointer_signature = pointer_signature
            return self._current


@dataclass(frozen=True, slots=True)
class PolicyE2ERuntimeIdentity:
    schema_version: str
    run_id: str
    run_identity_sha256: str
    worker_placement: PolicyAgentLoopWorkerPlacement
    builder_identity: str


@dataclass(slots=True)
class _ProcessRuntimeCore:
    identity: PolicyE2ERuntimeIdentity
    bound_factory: BoundVerlNativeAgentLoopInvocationFactory
    policy_version: ExactLoRASnapshotPolicyVersionPort | ExactFullQwenSyncPolicyVersionPort
    construction_fingerprint: tuple[int, ...]


_PROCESS_RUNTIME_LOCK = RLock()
_PROCESS_RUNTIMES: dict[tuple[int, str, int], _ProcessRuntimeCore] = {}
_REGISTERED_LIVE_BUILDER: PolicyE2ERuntimeBuilder | None = None


def register_policy_e2e_live_runtime_builder(
    builder: PolicyE2ERuntimeBuilder,
) -> None:
    """Register the production composition exactly once before Hydra use."""

    if not callable(getattr(builder, "build", None)):
        raise TypeError("live runtime builder must implement build()")
    _builder_identity(builder)
    global _REGISTERED_LIVE_BUILDER
    with _PROCESS_RUNTIME_LOCK:
        if _PROCESS_RUNTIMES:
            raise RuntimeError("cannot register a live builder after runtime construction")
        if _REGISTERED_LIVE_BUILDER is not None and _REGISTERED_LIVE_BUILDER is not builder:
            raise RuntimeError("a different Policy live runtime builder is already registered")
        _REGISTERED_LIVE_BUILDER = builder


class PolicyE2ERuntimeInvocationFactory:
    """Hydra-instantiable process singleton delegating to the bound native factory."""

    def __init__(
        self,
        *,
        run_config_path: str | Path,
        expected_run_identity_sha256: str,
        trainer_config: object,
        server_manager: object,
        tokenizer: object,
        processor: object,
        dataset_cls: object,
        data_config: object,
        runtime_builder: PolicyE2ERuntimeBuilder | None = None,
        environment: Mapping[str, str] | None = None,
        actor_name: str | None = None,
        worker_index: int | None = None,
        config_loader: Callable[[str | Path], PolicyE2ESmokeRunConfig] = (
            load_policy_e2e_smoke_run_config
        ),
        snapshot_loader: Callable[..., "PolicyLoRASnapshot"] | None = None,
        full_qwen_receipt_loader: Callable[..., "PolicyFullQwenSyncReceipt"]
        | None = None,
    ) -> None:
        if not callable(config_loader):
            raise TypeError("config_loader must be callable")
        config = config_loader(run_config_path)
        if not isinstance(config, PolicyE2ESmokeRunConfig):
            raise TypeError("config loader must return PolicyE2ESmokeRunConfig")
        _require_sha256(expected_run_identity_sha256, "expected run identity")
        if not hmac.compare_digest(
            config.identity_sha256, expected_run_identity_sha256
        ):
            raise IdentityMismatchError("Policy runtime run-config identity differs")
        _validate_trainer_runtime_identity(trainer_config, config)
        values = dict(os.environ if environment is None else environment)
        placement = resolve_policy_agent_loop_worker_placement(
            config,
            environment=values,
            actor_name=actor_name,
            worker_index=worker_index,
        )
        state = _weight_sync_state(values)
        if state.run_id != config.run_id:
            raise IdentityMismatchError("weight-sync state belongs to another run")
        if not hmac.compare_digest(
            state.run_identity_sha256, config.identity_sha256
        ):
            raise IdentityMismatchError("weight-sync state run identity differs")
        selected_builder = (
            runtime_builder
            or _REGISTERED_LIVE_BUILDER
            or _default_policy_e2e_live_runtime_builder()
        )
        builder_identity = _builder_identity(selected_builder)
        key = (os.getpid(), config.identity_sha256, placement.worker_index)
        fingerprint = tuple(
            id(value)
            for value in (
                server_manager,
                tokenizer,
                processor,
                dataset_cls,
            )
        )
        with _PROCESS_RUNTIME_LOCK:
            core = _PROCESS_RUNTIMES.get(key)
            if core is None:
                full_qwen = (
                    config.schema_version
                    == POLICY_E2E_CROP_TFREE_EXACT_MATCHED_RUN_CONFIG_SCHEMA
                )
                if full_qwen:
                    from .exact_replay_engine import (
                        _operational_base_identity_sha256,
                    )

                    initial_snapshot = (
                        full_qwen_receipt_loader
                        or _load_latest_full_qwen_receipt
                    )(
                        state,
                        expected_base_weights_sha256=(
                            _operational_base_identity_sha256(config.model)
                        ),
                    )
                    _require_full_qwen_sync_receipt(initial_snapshot)
                else:
                    initial_snapshot = (snapshot_loader or _load_latest_snapshot)(
                        state
                    )
                    _require_policy_lora_snapshot(initial_snapshot)
                context = PolicyE2ERuntimeBuildContext(
                    config=config,
                    placement=placement,
                    initial_snapshot=initial_snapshot,
                    weight_sync_state=state,
                    trainer_config=trainer_config,
                    server_manager=server_manager,
                    tokenizer=tokenizer,
                    processor=processor,
                    dataset_cls=dataset_cls,
                    data_config=data_config,
                )
                product = selected_builder.build(context)
                if not isinstance(product, PolicyE2ERuntimeProduct):
                    raise TypeError("live runtime builder must return PolicyE2ERuntimeProduct")
                if full_qwen:
                    policy_version = ExactFullQwenSyncPolicyVersionPort(
                        state=state,
                        initial_receipt=initial_snapshot,
                        receipt_loader=full_qwen_receipt_loader,
                    )
                else:
                    policy_version = ExactLoRASnapshotPolicyVersionPort(
                        state=state,
                        consumer=product.snapshot_consumer,
                        initial_snapshot=initial_snapshot,
                        snapshot_loader=snapshot_loader,
                    )
                bound = BoundVerlNativeAgentLoopInvocationFactory(
                    run_id=config.run_id,
                    model=config.model,
                    sampling_contract=config.policy.sampling,
                    policy_version=policy_version,
                    trajectory_components=product.trajectory_components,
                    decoding=_policy_decoding_contract(),
                    termination=_policy_termination_contract(config),
                    rollout_master_seed=config.rollout_rng.master_seed,
                    max_model_len=config.capacity.vllm_max_model_len,
                    rollouts_per_prompt=config.policy.sampling.trajectories_per_prompt,
                )
                identity = PolicyE2ERuntimeIdentity(
                    POLICY_E2E_RUNTIME_SCHEMA,
                    config.run_id,
                    config.identity_sha256,
                    placement,
                    builder_identity,
                )
                core = _ProcessRuntimeCore(
                    identity=identity,
                    bound_factory=bound,
                    policy_version=policy_version,
                    construction_fingerprint=fingerprint,
                )
                _PROCESS_RUNTIMES[key] = core
            else:
                if core.identity.builder_identity != builder_identity:
                    raise IdentityMismatchError(
                        "one worker attempted to reuse the run with another live builder"
                    )
                if core.construction_fingerprint != fingerprint:
                    raise IdentityMismatchError(
                        "Hydra attempted to reuse the process runtime with different "
                        "veRL worker-owned dependencies"
                    )
        self.config = config
        self.identity = core.identity
        self.bound_factory = core.bound_factory
        self.policy_version = core.policy_version

    def build(
        self,
        *,
        sampling_params: Mapping[str, object],
        sample_fields: Mapping[str, object],
    ) -> VerlNativeAgentLoopInvocation:
        return self.bound_factory.build(
            sampling_params=sampling_params,
            sample_fields=sample_fields,
        )

    async def build_async(
        self,
        *,
        sampling_params: Mapping[str, object],
        sample_fields: Mapping[str, object],
    ) -> VerlNativeAgentLoopInvocation:
        return await self.bound_factory.build_async(
            sampling_params=sampling_params,
            sample_fields=sample_fields,
        )


def resolve_policy_agent_loop_worker_placement(
    config: PolicyE2ESmokeRunConfig,
    *,
    environment: Mapping[str, str] | None = None,
    actor_name: str | None = None,
    worker_index: int | None = None,
) -> PolicyAgentLoopWorkerPlacement:
    """Resolve pinned veRL's ``agent_loop_worker_<i>_<uuid>`` identity."""

    if not isinstance(config, PolicyE2ESmokeRunConfig):
        raise TypeError("worker placement requires PolicyE2ESmokeRunConfig")
    values = os.environ if environment is None else environment
    explicit = values.get(POLICY_AGENT_LOOP_WORKER_INDEX_ENV)
    choices = sum(value is not None for value in (worker_index, explicit, actor_name))
    if choices > 1:
        raise ValueError("worker placement has more than one explicit identity source")
    if worker_index is None and explicit is not None:
        if not explicit.isdecimal():
            raise ValueError("TGVF_POLICY_AGENT_LOOP_WORKER_INDEX must be decimal")
        worker_index = int(explicit)
    if worker_index is None:
        selected_name = actor_name or _ray_actor_name()
        match = _AGENT_LOOP_WORKER_NAME.fullmatch(selected_name)
        if match is None:
            raise IdentityMismatchError(
                "Ray actor name does not match pinned veRL AgentLoop worker naming"
            )
        worker_index = int(match.group("index"))
    if type(worker_index) is not int or worker_index < 0:
        raise ValueError("AgentLoop worker index must be a non-negative integer")
    distributed = config.distributed
    if worker_index >= distributed.world_size:
        raise IdentityMismatchError("AgentLoop worker index exceeds configured world")
    logical = distributed.logical_gpu_ids[worker_index]
    if logical != worker_index:
        raise IdentityMismatchError(
            "Policy runtime requires pinned worker/logical-GPU index parity"
        )
    physical = distributed.physical_gpu_ids[logical]
    return PolicyAgentLoopWorkerPlacement(
        worker_index=worker_index,
        logical_gpu_id=logical,
        physical_gpu_id=physical,
        world_size=distributed.world_size,
    )


def _policy_decoding_contract() -> VLLMOutputDecodingContract:
    return VLLMOutputDecodingContract(
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        output_kind="final_only",
    )


def _policy_termination_contract(
    config: PolicyE2ESmokeRunConfig,
) -> VLLMTurnTerminationContract:
    sampling = config.policy.sampling
    if not sampling.is_run_bound:
        raise ValueError("Policy runtime requires a run-bound sampling contract")
    stop_strings = tuple(sampling.stop_strings or ())
    stop_token_ids = tuple(sampling.stop_token_ids or ())
    if "</tool_call>" not in stop_strings:
        raise ValueError("native multi-turn execution requires </tool_call> stop")
    final_outcomes = qwen3_vl_final_turn_outcomes(stop_token_ids)
    return VLLMTurnTerminationContract(
        required_request_stop_strings=stop_strings,
        required_request_stop_token_ids=stop_token_ids,
        include_stop_str_in_output=bool(sampling.include_stop_str_in_output),
        tool_call_terminal_suffixes=("",),
        tool_call_outcomes=(
            VLLMTerminationOutcome("stop", "</tool_call>"),
        ),
        final_turn_outcomes=final_outcomes,
    )


def _validate_trainer_runtime_identity(
    trainer_config: object, config: PolicyE2ESmokeRunConfig
) -> None:
    custom = _nested_value(
        trainer_config,
        ("actor_rollout_ref", "rollout", "custom"),
    )
    run_id = _nested_value(custom, ("run_id",))
    identity = _nested_value(custom, ("run_identity_sha256",))
    if run_id != config.run_id or identity != config.identity_sha256:
        raise IdentityMismatchError(
            "veRL trainer custom runtime identity differs from the run config"
        )


def _nested_value(root: object, path: tuple[str, ...]) -> object:
    value = root
    for name in path:
        if isinstance(value, Mapping):
            if name not in value:
                raise IdentityMismatchError(
                    f"veRL trainer config omitted runtime field {'.'.join(path)!r}"
                )
            value = value[name]
            continue
        if not hasattr(value, name):
            raise IdentityMismatchError(
                f"veRL trainer config omitted runtime field {'.'.join(path)!r}"
            )
        value = getattr(value, name)
    return value


def _builder_identity(builder: object) -> str:
    identity = getattr(builder, "singleton_identity", None)
    if not isinstance(identity, str) or not identity:
        raise ValueError("live runtime builder requires a non-empty singleton_identity")
    return identity


def _weight_sync_state(environment: Mapping[str, str]) -> "PolicyWeightSyncState":
    from .policy_weight_sync import PolicyWeightSyncState

    return PolicyWeightSyncState.from_environment(environment)


def _load_latest_snapshot(state: "PolicyWeightSyncState") -> "PolicyLoRASnapshot":
    from .policy_weight_sync import load_latest_lora_snapshot

    return load_latest_lora_snapshot(state)


def _load_latest_full_qwen_receipt(
    state: "PolicyWeightSyncState", **kwargs: object
) -> "PolicyFullQwenSyncReceipt":
    from .policy_weight_sync import load_latest_full_qwen_sync_receipt

    return load_latest_full_qwen_sync_receipt(state, **kwargs)


def _latest_pointer_signature(state: "PolicyWeightSyncState") -> str | None:
    """Cheaply avoid reloading the same LoRA tensors for every n-way rollout."""

    try:
        payload = state.latest_path.read_bytes()
    except FileNotFoundError:
        return None
    return hashlib.sha256(payload).hexdigest()


def _latest_full_qwen_pointer_signature(
    state: "PolicyWeightSyncState",
) -> str | None:
    try:
        payload = state.full_qwen_latest_path.read_bytes()
    except FileNotFoundError:
        return None
    return hashlib.sha256(payload).hexdigest()


def _default_policy_e2e_live_runtime_builder() -> PolicyE2ERuntimeBuilder:
    """Load the real composer lazily so contract-only imports stay lightweight."""

    try:
        from .policy_live_runtime import Qwen3PolicyE2ELiveRuntimeBuilder
    except ImportError as error:  # pragma: no cover - accepted live env owns deps
        raise PolicyE2ELiveRuntimeUnavailableError(
            "the concrete Qwen3 Policy E2E runtime could not be imported"
        ) from error
    return Qwen3PolicyE2ELiveRuntimeBuilder()


def _require_policy_lora_snapshot(value: object) -> None:
    from .policy_weight_sync import PolicyLoRASnapshot

    if not isinstance(value, PolicyLoRASnapshot):
        raise TypeError("snapshot loader must return PolicyLoRASnapshot")


def _require_full_qwen_sync_receipt(value: object) -> None:
    from .policy_weight_sync import PolicyFullQwenSyncReceipt

    if not isinstance(value, PolicyFullQwenSyncReceipt):
        raise TypeError(
            "full-Qwen receipt loader must return PolicyFullQwenSyncReceipt"
        )


def _ray_actor_name() -> str:
    try:
        import ray
    except ImportError as error:  # pragma: no cover - live-only dependency
        raise RuntimeError(
            "AgentLoop worker placement requires Ray actor identity"
        ) from error
    context = ray.get_runtime_context()
    getter = getattr(context, "get_actor_name", None)
    name = getter() if callable(getter) else None
    if not isinstance(name, str) or not name:
        raise RuntimeError("Ray runtime did not expose an AgentLoop actor name")
    return name


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _reset_policy_e2e_runtime_singletons_for_tests() -> None:
    """Test-only reset; live code must never reset rollout group state."""

    global _REGISTERED_LIVE_BUILDER
    with _PROCESS_RUNTIME_LOCK:
        _PROCESS_RUNTIMES.clear()
        _REGISTERED_LIVE_BUILDER = None


__all__ = [
    "ExactFullQwenSyncPolicyVersionPort",
    "ExactLoRASnapshotPolicyVersionPort",
    "POLICY_AGENT_LOOP_WORKER_INDEX_ENV",
    "POLICY_E2E_RUNTIME_SCHEMA",
    "PeftPolicyLoRASnapshotConsumer",
    "PolicyAgentLoopWorkerPlacement",
    "PolicyE2ELiveRuntimeUnavailableError",
    "PolicyE2ERuntimeBuildContext",
    "PolicyE2ERuntimeBuilder",
    "PolicyE2ERuntimeIdentity",
    "PolicyE2ERuntimeInvocationFactory",
    "PolicyE2ERuntimeProduct",
    "PolicyLoRASnapshotConsumer",
    "register_policy_e2e_live_runtime_builder",
    "resolve_policy_agent_loop_worker_placement",
]
