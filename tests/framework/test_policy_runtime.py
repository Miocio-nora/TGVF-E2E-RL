from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.framework.verl.policy_runtime import (
    ExactLoRASnapshotPolicyVersionPort,
    ExactPolicyBehaviorSnapshotVersionPort,
    PolicyAgentLoopWorkerPlacement,
    PolicyE2ERuntimeInvocationFactory,
    PolicyE2ERuntimeProduct,
    _reset_policy_e2e_runtime_singletons_for_tests,
    resolve_policy_agent_loop_worker_placement,
)
from tgvf_rl.framework.verl.policy_behavior_version import (
    FullQwenSyncReceipt,
    PolicyBehaviorPayload,
    PolicyBehaviorSnapshot,
    publish_policy_behavior_snapshot,
)
from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyLoRASnapshot,
    PolicyWeightSyncState,
    publish_policy_weight_sync_request,
    wrap_lora_parameter_stream_for_snapshot,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config
from tgvf_rl.policy.config import PolicyMethodProfile
from tgvf_rl.policy.no_tool_rl_protocol import NO_TOOL_RL_PROMPT_IDENTITY
from tgvf_rl.protocol import (
    NativeSuccessObservationProtocolId,
    NativeToolCapabilityProfile,
)
from tests.policy.test_method_run_config import method_config_factory
from tests.policy.test_run_config import _write_config


@pytest.fixture(autouse=True)
def _reset_process_runtime() -> None:
    _reset_policy_e2e_runtime_singletons_for_tests()
    yield
    _reset_policy_e2e_runtime_singletons_for_tests()


class _TrajectoryComponents:
    def __init__(self) -> None:
        self.close_calls = 0

    def build_trajectory_components(self, **kwargs: object) -> object:
        raise AssertionError("unit fixture does not execute a live trajectory")

    def close(self) -> None:
        self.close_calls += 1


class _SnapshotConsumer:
    def __init__(self, *, return_exact: bool = True) -> None:
        self.return_exact = return_exact
        self.applied: list[PolicyVersion] = []

    def apply_policy_lora_snapshot(
        self, snapshot: PolicyLoRASnapshot, /
    ) -> PolicyVersion:
        self.applied.append(snapshot.policy_version)
        if self.return_exact:
            return snapshot.policy_version
        return PolicyVersion(
            snapshot.policy_version.run_id,
            snapshot.policy_version.optimizer_step,
            "f" * 64,
        )

    def apply_policy_behavior_snapshot(
        self, snapshot: PolicyBehaviorSnapshot, /
    ) -> PolicyVersion:
        self.applied.append(snapshot.policy_version)
        if self.return_exact:
            return snapshot.policy_version
        return PolicyVersion(
            snapshot.policy_version.run_id,
            snapshot.policy_version.optimizer_step,
            "f" * 64,
        )


class _Builder:
    singleton_identity = "tests.policy-runtime-builder-v1"

    def __init__(self) -> None:
        self.contexts: list[object] = []
        self.consumer = _SnapshotConsumer()
        self.components = _TrajectoryComponents()

    def build(self, context: object, /) -> PolicyE2ERuntimeProduct:
        self.contexts.append(context)
        return PolicyE2ERuntimeProduct(
            self.components,
            self.consumer,
            self.consumer,
        )


def _environment(tmp_path: Path, *, run_id: str, run_identity: str) -> dict[str, str]:
    return {
        "TGVF_POLICY_STATE_DIR": str((tmp_path / "policy-state").resolve()),
        "TGVF_POLICY_RUN_ID": run_id,
        "TGVF_POLICY_RUN_IDENTITY_SHA256": run_identity,
        "RANK": "0",
        "WORLD_SIZE": "4",
    }


def _publish_snapshot(environment: dict[str, str], *, step: int) -> PolicyLoRASnapshot:
    state = PolicyWeightSyncState.from_environment(environment)
    publish_policy_weight_sync_request(state, step, nonce=f"step-{step}")
    stream = (
        (
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            torch.tensor([[1.0 + step, 2.0]], dtype=torch.bfloat16),
        ),
        (
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
            torch.tensor([[3.0], [4.0 + step]], dtype=torch.bfloat16),
        ),
    )
    list(
        wrap_lora_parameter_stream_for_snapshot(
            stream,
            base_sync_done=True,
            rank=0,
            world_size=4,
            global_steps=step,
            environment=environment,
        )
    )
    from tgvf_rl.framework.verl.policy_weight_sync import load_latest_lora_snapshot

    return load_latest_lora_snapshot(state, expected_optimizer_step=step)


def _publish_behavior(
    environment: dict[str, str], *, step: int
) -> PolicyBehaviorSnapshot:
    state = PolicyWeightSyncState.from_environment(environment)
    request = publish_policy_weight_sync_request(
        state, step, nonce=f"behavior-step-{step}"
    )
    return publish_policy_behavior_snapshot(
        state,
        full_qwen=FullQwenSyncReceipt.from_acknowledged_request(request),
        payload=PolicyBehaviorPayload.FULL_QWEN,
    )


def _write_no_tool_method_config(tmp_path: Path) -> Path:
    factory = method_config_factory.__wrapped__(tmp_path)
    path, _ = factory(
        profile=PolicyMethodProfile.NO_TOOL,
        tool_profile=NativeToolCapabilityProfile.NO_TOOL,
        prompt_sha256=NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1,
    )
    return path


def _trainer_config(run_id: str, identity: str) -> dict[str, object]:
    return {
        "actor_rollout_ref": {
            "rollout": {
                "custom": {
                    "run_id": run_id,
                    "run_identity_sha256": identity,
                }
            }
        }
    }


def test_worker_name_maps_deterministically_to_configured_physical_gpu(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)

    placements = tuple(
        resolve_policy_agent_loop_worker_placement(
            config,
            environment={},
            actor_name=f"agent_loop_worker_{index}_12ab34cd",
        )
        for index in range(4)
    )

    assert tuple(item.worker_index for item in placements) == (0, 1, 2, 3)
    assert tuple(item.logical_gpu_id for item in placements) == (0, 1, 2, 3)
    assert tuple(item.physical_gpu_id for item in placements) == (0, 1, 2, 3)
    assert tuple(str(item.torch_device) for item in placements) == (
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
    )

    with pytest.raises(IdentityMismatchError, match="exceeds configured world"):
        resolve_policy_agent_loop_worker_placement(
            config, environment={}, actor_name="agent_loop_worker_4_12ab34cd"
        )
    with pytest.raises(IdentityMismatchError, match="does not match"):
        resolve_policy_agent_loop_worker_placement(
            config, environment={}, actor_name="some-other-ray-actor"
        )


@pytest.mark.parametrize(
    ("worker_index", "logical_gpu_id", "physical_gpu_id", "world_size"),
    [(0, 0, 7, 1), (1, 1, 6, 2)],
)
def test_worker_placement_accepts_any_positive_generic_world_size(
    worker_index: int,
    logical_gpu_id: int,
    physical_gpu_id: int,
    world_size: int,
) -> None:
    placement = PolicyAgentLoopWorkerPlacement(
        worker_index=worker_index,
        logical_gpu_id=logical_gpu_id,
        physical_gpu_id=physical_gpu_id,
        world_size=world_size,
    )

    assert placement.world_size == world_size
    assert placement.torch_device == torch.device("cuda", logical_gpu_id)


def test_worker_placement_rejects_nonpositive_and_out_of_world_values() -> None:
    with pytest.raises(ValueError, match="positive world_size"):
        PolicyAgentLoopWorkerPlacement(0, 0, 7, 0)
    with pytest.raises(ValueError, match="worker index lies outside"):
        PolicyAgentLoopWorkerPlacement(2, 1, 7, 2)
    with pytest.raises(ValueError, match="logical GPU ID lies outside"):
        PolicyAgentLoopWorkerPlacement(1, 2, 7, 2)


def test_exact_snapshot_port_installs_initial_and_each_new_committed_version(
    tmp_path: Path,
) -> None:
    environment = _environment(
        tmp_path,
        run_id="policy-runtime-test",
        run_identity="a" * 64,
    )
    initial = _publish_snapshot(environment, step=0)
    state = PolicyWeightSyncState.from_environment(environment)
    consumer = _SnapshotConsumer()
    port = ExactLoRASnapshotPolicyVersionPort(
        state=state,
        consumer=consumer,
        initial_snapshot=initial,
    )

    assert port.current_policy_version() == initial.policy_version
    assert consumer.applied == [initial.policy_version]
    updated = _publish_snapshot(environment, step=1)
    assert port.current_policy_version() == updated.policy_version
    assert consumer.applied == [initial.policy_version, updated.policy_version]


def test_snapshot_port_rejects_consumer_that_does_not_prove_exact_weights(
    tmp_path: Path,
) -> None:
    environment = _environment(
        tmp_path,
        run_id="policy-runtime-test",
        run_identity="b" * 64,
    )
    initial = _publish_snapshot(environment, step=0)

    with pytest.raises(IdentityMismatchError, match="exact served policy"):
        ExactLoRASnapshotPolicyVersionPort(
            state=PolicyWeightSyncState.from_environment(environment),
            consumer=_SnapshotConsumer(return_exact=False),
            initial_snapshot=initial,
        )


def test_full_qwen_behavior_port_never_requires_a_lora_pointer(
    tmp_path: Path,
) -> None:
    environment = _environment(
        tmp_path,
        run_id="policy-method-runtime-test",
        run_identity="c" * 64,
    )
    initial = _publish_behavior(environment, step=0)
    state = PolicyWeightSyncState.from_environment(environment)
    consumer = _SnapshotConsumer()
    port = ExactPolicyBehaviorSnapshotVersionPort(
        state=state,
        consumer=consumer,
        initial_snapshot=initial,
    )

    assert not state.latest_path.exists()
    assert port.current_policy_version() == initial.policy_version
    updated = _publish_behavior(environment, step=1)
    assert port.current_policy_version() == updated.policy_version
    assert consumer.applied == [initial.policy_version, updated.policy_version]


def test_hydra_factory_reuses_one_bound_runtime_and_n8_counter_owner(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    environment = _environment(
        tmp_path,
        run_id=config.run_id,
        run_identity=config.identity_sha256,
    )
    initial = _publish_snapshot(environment, step=0)
    builder = _Builder()
    trainer_config = _trainer_config(config.run_id, config.identity_sha256)
    server_manager = object()
    tokenizer = object()
    processor = object()
    dataset_cls = object()
    dependencies = {
        "trainer_config": trainer_config,
        "server_manager": server_manager,
        "tokenizer": tokenizer,
        "processor": processor,
        "dataset_cls": dataset_cls,
        "data_config": object(),
    }

    first = PolicyE2ERuntimeInvocationFactory(
        run_config_path=path,
        expected_run_identity_sha256=config.identity_sha256,
        runtime_builder=builder,
        environment=environment,
        worker_index=2,
        **dependencies,
    )
    second = PolicyE2ERuntimeInvocationFactory(
        run_config_path=path,
        expected_run_identity_sha256=config.identity_sha256,
        runtime_builder=builder,
        environment=environment,
        worker_index=2,
        # Pinned veRL creates fresh trainer/data wrapper objects for every
        # per-row Hydra instantiation; they are validated by value, not address.
        trainer_config=_trainer_config(config.run_id, config.identity_sha256),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=dataset_cls,
        data_config=object(),
    )

    assert first.bound_factory is second.bound_factory
    assert first.policy_version is second.policy_version
    assert len(builder.contexts) == 1
    assert builder.consumer.applied == [initial.policy_version]
    assert first.identity.worker_placement.worker_index == 2
    assert first.identity.worker_placement.physical_gpu_id == 2


def test_process_runtime_owns_and_closes_shared_components_exactly_once(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    environment = _environment(
        tmp_path,
        run_id=config.run_id,
        run_identity=config.identity_sha256,
    )
    _publish_snapshot(environment, step=0)
    builder = _Builder()
    dependencies = {
        "trainer_config": _trainer_config(config.run_id, config.identity_sha256),
        "server_manager": object(),
        "tokenizer": object(),
        "processor": object(),
        "dataset_cls": object(),
        "data_config": object(),
    }

    first = PolicyE2ERuntimeInvocationFactory(
        run_config_path=path,
        expected_run_identity_sha256=config.identity_sha256,
        runtime_builder=builder,
        environment=environment,
        worker_index=0,
        **dependencies,
    )
    second = PolicyE2ERuntimeInvocationFactory(
        run_config_path=path,
        expected_run_identity_sha256=config.identity_sha256,
        runtime_builder=builder,
        environment=environment,
        worker_index=0,
        trainer_config=_trainer_config(config.run_id, config.identity_sha256),
        server_manager=dependencies["server_manager"],
        tokenizer=dependencies["tokenizer"],
        processor=dependencies["processor"],
        dataset_cls=dependencies["dataset_cls"],
        data_config=object(),
    )

    assert first.bound_factory is second.bound_factory
    assert len(builder.contexts) == 1
    assert builder.components.close_calls == 0

    _reset_policy_e2e_runtime_singletons_for_tests()
    _reset_policy_e2e_runtime_singletons_for_tests()

    assert builder.components.close_calls == 1


def test_hydra_factory_uses_concrete_default_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, _ = _write_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    environment = _environment(
        tmp_path,
        run_id=config.run_id,
        run_identity=config.identity_sha256,
    )

    _publish_snapshot(environment, step=0)
    builder = _Builder()
    monkeypatch.setattr(
        "tgvf_rl.framework.verl.policy_runtime."
        "_default_policy_e2e_live_runtime_builder",
        lambda: builder,
    )
    runtime = PolicyE2ERuntimeInvocationFactory(
        run_config_path=path,
        expected_run_identity_sha256=config.identity_sha256,
        trainer_config=_trainer_config(config.run_id, config.identity_sha256),
        server_manager=object(),
        tokenizer=object(),
        processor=object(),
        dataset_cls=object(),
        data_config=object(),
        environment=environment,
        worker_index=0,
    )

    assert runtime.identity.builder_identity == builder.singleton_identity
    assert len(builder.contexts) == 1


def test_method_factory_loads_full_qwen_behavior_without_lora(
    tmp_path: Path,
) -> None:
    path = _write_no_tool_method_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    environment = _environment(
        tmp_path,
        run_id=config.run_id,
        run_identity=config.identity_sha256,
    )
    behavior = _publish_behavior(environment, step=0)
    builder = _Builder()

    runtime = PolicyE2ERuntimeInvocationFactory(
        run_config_path=path,
        expected_run_identity_sha256=config.identity_sha256,
        trainer_config=_trainer_config(config.run_id, config.identity_sha256),
        server_manager=object(),
        tokenizer=object(),
        processor=object(),
        dataset_cls=object(),
        data_config=object(),
        runtime_builder=builder,
        environment=environment,
        worker_index=0,
        snapshot_loader=lambda _state: (_ for _ in ()).throw(
            AssertionError("method runtime must not load latest-lora-snapshot")
        ),
    )

    assert config.method is not None
    assert runtime.policy_version.current_policy_version() == (behavior.policy_version)
    assert builder.consumer.applied == [behavior.policy_version]
    assert not PolicyWeightSyncState.from_environment(environment).latest_path.exists()


def test_hydra_factory_rejects_config_and_trainer_identity_drift(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    environment = _environment(
        tmp_path,
        run_id=config.run_id,
        run_identity=config.identity_sha256,
    )

    with pytest.raises(IdentityMismatchError, match="run-config identity"):
        PolicyE2ERuntimeInvocationFactory(
            run_config_path=path,
            expected_run_identity_sha256="d" * 64,
            trainer_config=_trainer_config(config.run_id, config.identity_sha256),
            server_manager=object(),
            tokenizer=object(),
            processor=object(),
            dataset_cls=object(),
            data_config=object(),
            runtime_builder=_Builder(),
            environment=environment,
            worker_index=0,
        )

    with pytest.raises(IdentityMismatchError, match="trainer custom runtime"):
        PolicyE2ERuntimeInvocationFactory(
            run_config_path=path,
            expected_run_identity_sha256=config.identity_sha256,
            trainer_config=_trainer_config(config.run_id, "e" * 64),
            server_manager=object(),
            tokenizer=object(),
            processor=object(),
            dataset_cls=object(),
            data_config=object(),
            runtime_builder=_Builder(),
            environment=environment,
            worker_index=0,
        )
