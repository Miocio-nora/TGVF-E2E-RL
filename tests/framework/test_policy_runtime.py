from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.framework.verl.policy_runtime import (
    ExactFullQwenSyncPolicyVersionPort,
    ExactLoRASnapshotPolicyVersionPort,
    PolicyE2ERuntimeInvocationFactory,
    PolicyE2ERuntimeProduct,
    _reset_policy_e2e_runtime_singletons_for_tests,
    resolve_policy_agent_loop_worker_placement,
)
from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyLoRASnapshot,
    PolicyWeightSyncState,
    publish_full_qwen_sync_receipt,
    publish_policy_weight_sync_request,
    wrap_lora_parameter_stream_for_snapshot,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config
from tests.policy.test_run_config import _write_config


@pytest.fixture(autouse=True)
def _reset_process_runtime() -> None:
    _reset_policy_e2e_runtime_singletons_for_tests()
    yield
    _reset_policy_e2e_runtime_singletons_for_tests()


class _TrajectoryComponents:
    def build_trajectory_components(self, **kwargs: object) -> object:
        raise AssertionError("unit fixture does not execute a live trajectory")


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


class _Builder:
    singleton_identity = "tests.policy-runtime-builder-v1"

    def __init__(self) -> None:
        self.contexts: list[object] = []
        self.consumer = _SnapshotConsumer()

    def build(self, context: object, /) -> PolicyE2ERuntimeProduct:
        self.contexts.append(context)
        return PolicyE2ERuntimeProduct(_TrajectoryComponents(), self.consumer)


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


def test_full_qwen_version_port_follows_completed_sync_receipts(
    tmp_path: Path,
) -> None:
    environment = _environment(
        tmp_path,
        run_id="full-qwen-runtime-test",
        run_identity="c" * 64,
    )
    state = PolicyWeightSyncState.from_environment(environment)
    base = "d" * 64
    initial = publish_full_qwen_sync_receipt(
        state, optimizer_step=0, base_weights_sha256=base
    )
    port = ExactFullQwenSyncPolicyVersionPort(
        state=state, initial_receipt=initial
    )

    assert port.current_policy_version() == initial.policy_version
    updated = publish_full_qwen_sync_receipt(
        state, optimizer_step=1, base_weights_sha256=base
    )
    assert port.current_policy_version() == updated.policy_version


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
