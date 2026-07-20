from __future__ import annotations

from dataclasses import asdict
import hashlib
from threading import Thread

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
)
from tgvf_rl.environment import FocusExecutionLedger
from tgvf_rl.framework.verl import DataProtoPayload, to_verl_data_proto
from tgvf_rl.observations import ObservationHandle, ObservationStore
from tgvf_rl.policy import (
    PolicyBatchLifecycle,
    PolicyBatchLifecycleManager,
    PolicyBatchMilestone,
    PolicyBatchState,
)
from tgvf_rl.trajectories import BehaviorTraceStore, VLLMBehaviorRecorder


SHA = "0" * 64
POLICY = PolicyVersion("lifecycle-test", 0, SHA)


def _manager():
    observation_store = ObservationStore()
    behavior_store = BehaviorTraceStore()
    ledger = FocusExecutionLedger()
    manager = PolicyBatchLifecycleManager(
        observation_store=observation_store,
        behavior_store=behavior_store,
        focus_execution_ledger=ledger,
    )
    return manager, observation_store, behavior_store, ledger


def _payload(name: str) -> DataProtoPayload:
    return DataProtoPayload(
        tensor_batch={"responses": torch.tensor([[1, 2]])},
        non_tensor_batch={"exact_replay_bundle": object(), "batch": name},
        meta_info={"name": name},
    )


class _FakeDataProto:
    def __init__(self, batch, non_tensor_batch, meta_info):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    @classmethod
    def from_dict(cls, *, tensors, non_tensors, meta_info):
        return cls(tensors, non_tensors, meta_info)


def _record_behavior(store: BehaviorTraceStore, *, trajectory_id: str, seed: int):
    tokens = OwnedTokenSequence((10 + seed,), (TokenOwnership.POLICY_SAMPLED,))
    sampling = SamplingIdentity(
        policy_version=POLICY,
        backend="vllm",
        backend_version="lifecycle-fixture",
        seed=seed,
        rng_state_sha256=hashlib.sha256(f"rng-{seed}".encode()).hexdigest(),
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    return VLLMBehaviorRecorder(store).record(
        trajectory_id=trajectory_id,
        assistant_turn_index=0,
        tokens=tokens,
        actual_sampled_logprobs=(-0.5,),
        sampling=sampling,
        behavior_policy=POLICY,
        backend_request_sha256=hashlib.sha256(f"request-{seed}".encode()).hexdigest(),
        backend_response_sha256=hashlib.sha256(f"response-{seed}".encode()).hexdigest(),
    )


def _populate_batch(
    batch: PolicyBatchLifecycle,
    *,
    store: ObservationStore,
    behavior_store: BehaviorTraceStore,
    ledger: FocusExecutionLedger,
    seed: int,
):
    trajectory_id = batch.trajectory_ids[0]
    tensor = store.put_tensor(
        f"batch-{seed}.main-d",
        torch.tensor([1.0, 2.0]),
        trajectory_id=trajectory_id,
    )
    behavior = _record_behavior(behavior_store, trajectory_id=trajectory_id, seed=seed)
    ledger_handle = ledger.execute_once(
        key=(trajectory_id, 0),
        fingerprint=f"fingerprint-{seed}",
        operation=lambda: ObservationHandle(f"observation-{seed}", SHA),
    )
    payload = _payload(batch.batch_id)
    batch.attach_data_proto(payload)
    return tensor, behavior, ledger_handle, payload


def _complete_update(batch: PolicyBatchLifecycle) -> None:
    for milestone in PolicyBatchMilestone:
        with batch.consume(milestone):
            pass


def _complete_through_loss(batch: PolicyBatchLifecycle) -> None:
    for milestone in (
        PolicyBatchMilestone.BEHAVIOR_REPLAY,
        PolicyBatchMilestone.CURRENT_REPLAY,
        PolicyBatchMilestone.REFERENCE_REPLAY,
        PolicyBatchMilestone.LOSS_BACKWARD,
    ):
        with batch.consume(milestone):
            pass


def test_normal_close_is_barrier_guarded_selective_and_idempotent() -> None:
    manager, store, behavior_store, ledger = _manager()
    trajectory_id = "run/sample/0/group"
    batch = manager.open_batch(batch_id="batch-0", trajectory_ids=(trajectory_id,))
    tensor, behavior, ledger_handle, payload = _populate_batch(
        batch,
        store=store,
        behavior_store=behavior_store,
        ledger=ledger,
        seed=0,
    )

    with pytest.raises(RuntimeError, match="complete update barrier"):
        batch.close()
    torch.testing.assert_close(
        store.resolve_verified_for_trajectory(tensor, trajectory_id=trajectory_id),
        torch.tensor([1.0, 2.0]),
    )
    assert behavior_store.resolve(behavior).trajectory_id == trajectory_id
    assert (
        ledger.execute_once(
            key=(trajectory_id, 0),
            fingerprint="fingerprint-0",
            operation=lambda: pytest.fail("execute-once result was lost"),
        )
        == ledger_handle
    )

    _complete_update(batch)
    report = batch.close()
    assert report.state is PolicyBatchState.CLOSED
    assert report.ledger_entries == report.behavior_traces == report.tensors == 1
    assert report.data_proto_sidecar_fields == 2
    assert batch.close() is report
    assert manager.outstanding_batch_count == 0
    assert not payload.non_tensor_batch
    assert payload.sidecars_released

    with pytest.raises(RuntimeError, match="already closed"):
        batch.assert_open()
    with pytest.raises(RuntimeError, match="released"):
        ledger.execute_once(
            key=(trajectory_id, 0),
            fingerprint="fingerprint-0",
            operation=lambda: ledger_handle,
        )
    with pytest.raises(RuntimeError, match="released"):
        payload.assert_sidecars_available()
    with pytest.raises(RuntimeError, match="cannot be reused"):
        manager.open_batch(batch_id="batch-0", trajectory_ids=("run/another/0/group",))
    with pytest.raises(RuntimeError, match="trajectory identities cannot be reused"):
        manager.open_batch(batch_id="batch-0-retry", trajectory_ids=(trajectory_id,))
    with pytest.raises(ReplayMismatchError, match="released|missing"):
        store.resolve_verified_for_trajectory(tensor, trajectory_id=trajectory_id)

    with manager.checkpoint_barrier() as barrier:
        assert asdict(barrier) == {
            "asynchronous_staleness_steps": 0,
            "outstanding_rollout_count": 0,
        }
    checkpoint_store_state = store.checkpoint_state()
    assert checkpoint_store_state["records"] == {}
    assert checkpoint_store_state["replays"] == {}
    assert checkpoint_store_state["tensors"] == {}


def test_close_releases_sidecars_from_converted_local_verl_dataproto() -> None:
    manager, store, behavior_store, ledger = _manager()
    batch = manager.open_batch(
        batch_id="converted-dataproto",
        trajectory_ids=("run/converted/0/group",),
    )
    _tensor, _behavior, _ledger_handle, payload = _populate_batch(
        batch,
        store=store,
        behavior_store=behavior_store,
        ledger=ledger,
        seed=11,
    )
    data = to_verl_data_proto(payload, data_proto_cls=_FakeDataProto)
    data.non_tensor_batch["worker_metric"] = "preserve"

    _complete_update(batch)
    batch.close()

    assert data.non_tensor_batch == {"worker_metric": "preserve"}
    assert payload.sidecars_released


def test_exception_unwinds_consumer_before_abort_and_does_not_mark_success() -> None:
    manager, store, behavior_store, ledger = _manager()
    batch = manager.open_batch(
        batch_id="exception-batch",
        trajectory_ids=("run/sample/1/group",),
    )
    _tensor, _behavior, _ledger_handle, payload = _populate_batch(
        batch,
        store=store,
        behavior_store=behavior_store,
        ledger=ledger,
        seed=1,
    )

    with pytest.raises(ValueError, match="replay failed"):
        with batch.consume(PolicyBatchMilestone.BEHAVIOR_REPLAY):
            with pytest.raises(RuntimeError, match="consumer is active"):
                batch.abort()
            raise ValueError("replay failed")

    assert not batch.completed_milestones
    report = batch.abort()
    assert report.state is PolicyBatchState.ABORTED
    assert batch.abort() is report
    assert payload.sidecars_released
    manager.assert_quiescent()
    with pytest.raises(RuntimeError, match="trajectory identities cannot be reused"):
        manager.open_batch(
            batch_id="exception-batch-retry",
            trajectory_ids=("run/sample/1/group",),
        )


def test_same_milestone_cannot_have_overlapping_consumers() -> None:
    manager, _store, _behavior_store, _ledger = _manager()
    batch = manager.open_batch(
        batch_id="inflight-batch",
        trajectory_ids=("run/inflight/0/group",),
    )

    with batch.consume(PolicyBatchMilestone.BEHAVIOR_REPLAY):
        with pytest.raises(RuntimeError, match="already in flight"):
            with batch.consume(PolicyBatchMilestone.BEHAVIOR_REPLAY):
                pytest.fail("duplicate milestone consumer was admitted")
        with pytest.raises(RuntimeError, match="missing=.*behavior_replay"):
            with batch.consume(PolicyBatchMilestone.CURRENT_REPLAY):
                pytest.fail("an in-flight dependency counted as complete")
        with pytest.raises(RuntimeError, match="consumer is active"):
            batch.abort()

    assert batch.completed_milestones == frozenset(
        {PolicyBatchMilestone.BEHAVIOR_REPLAY}
    )
    batch.abort()


def test_two_batches_release_only_their_own_resources() -> None:
    manager, store, behavior_store, ledger = _manager()
    first_id = "run/sample/2/group"
    second_id = "run/sample/3/group"
    first = manager.open_batch(batch_id="batch-a", trajectory_ids=(first_id,))
    second = manager.open_batch(batch_id="batch-b", trajectory_ids=(second_id,))
    first_tensor, first_behavior, _, first_payload = _populate_batch(
        first,
        store=store,
        behavior_store=behavior_store,
        ledger=ledger,
        seed=2,
    )
    second_tensor, second_behavior, _, second_payload = _populate_batch(
        second,
        store=store,
        behavior_store=behavior_store,
        ledger=ledger,
        seed=3,
    )
    assert first_tensor.address.digest == second_tensor.address.digest
    with pytest.raises(RuntimeError, match="outstanding batches"):
        with manager.checkpoint_barrier():
            pytest.fail("checkpoint gate admitted outstanding batches")

    _complete_through_loss(first)
    _complete_through_loss(second)
    with pytest.raises(RuntimeError, match="no other open batch"):
        with first.consume(PolicyBatchMilestone.OPTIMIZER_STEP):
            pytest.fail("optimizer admitted an overlapping behavior cohort")
    first.abort()
    assert first_payload.sidecars_released
    assert not second_payload.sidecars_released
    torch.testing.assert_close(
        store.resolve_verified_for_trajectory(second_tensor, trajectory_id=second_id),
        torch.tensor([1.0, 2.0]),
    )
    assert behavior_store.resolve(second_behavior).trajectory_id == second_id
    with pytest.raises(ReplayMismatchError, match="released|unknown"):
        behavior_store.resolve(first_behavior)
    assert manager.outstanding_batch_ids == ("batch-b",)

    with second.consume(PolicyBatchMilestone.OPTIMIZER_STEP):
        pass
    with second.consume(PolicyBatchMilestone.ZERO_STALENESS_BARRIER):
        pass
    second.close()
    assert second_payload.sidecars_released
    manager.assert_quiescent()
    assert store.resource_counts().tensors == 0


def test_checkpoint_gate_rejects_concurrent_batch_open() -> None:
    manager, _store, _behavior_store, _ledger = _manager()
    errors: list[BaseException] = []

    def open_during_checkpoint() -> None:
        try:
            manager.open_batch(
                batch_id="checkpoint-race",
                trajectory_ids=("run/checkpoint-race/0/group",),
            )
        except BaseException as error:
            errors.append(error)

    with manager.checkpoint_barrier():
        contender = Thread(target=open_during_checkpoint)
        contender.start()
        contender.join(timeout=2.0)
        assert not contender.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "during checkpoint capture" in str(errors[0])

    batch = manager.open_batch(
        batch_id="after-checkpoint",
        trajectory_ids=("run/after-checkpoint/0/group",),
    )
    batch.abort()


def test_failed_zero_staleness_barrier_keeps_update_gate_until_retry() -> None:
    manager, store, behavior_store, ledger = _manager()
    batch = manager.open_batch(
        batch_id="zero-retry",
        trajectory_ids=("run/zero-retry/0/group",),
    )
    _populate_batch(
        batch,
        store=store,
        behavior_store=behavior_store,
        ledger=ledger,
        seed=8,
    )
    _complete_through_loss(batch)
    with batch.consume(PolicyBatchMilestone.OPTIMIZER_STEP):
        pass

    with pytest.raises(ValueError, match="sync failed"):
        with batch.consume(PolicyBatchMilestone.ZERO_STALENESS_BARRIER):
            raise ValueError("sync failed")
    with pytest.raises(RuntimeError, match="during a Policy update gate"):
        manager.open_batch(
            batch_id="blocked-by-update",
            trajectory_ids=("run/blocked-by-update/0/group",),
        )
    with pytest.raises(RuntimeError, match="cannot abort after.*optimizer"):
        batch.abort()

    with batch.consume(PolicyBatchMilestone.ZERO_STALENESS_BARRIER):
        pass
    batch.close()
    with manager.checkpoint_barrier():
        pass


def test_optimizer_exception_is_terminal_commit_unknown() -> None:
    manager, store, behavior_store, ledger = _manager()
    batch = manager.open_batch(
        batch_id="optimizer-unknown",
        trajectory_ids=("run/optimizer-unknown/0/group",),
    )
    _populate_batch(
        batch,
        store=store,
        behavior_store=behavior_store,
        ledger=ledger,
        seed=11,
    )
    _complete_through_loss(batch)

    with pytest.raises(ValueError, match="optimizer failed after mutation"):
        with batch.consume(PolicyBatchMilestone.OPTIMIZER_STEP):
            raise ValueError("optimizer failed after mutation")

    assert batch.state is PolicyBatchState.COMMIT_UNKNOWN
    assert PolicyBatchMilestone.OPTIMIZER_STEP not in batch.completed_milestones
    assert batch.active_consumer_count == 0
    with pytest.raises(RuntimeError, match="terminal commit_unknown"):
        batch.abort()
    with pytest.raises(RuntimeError, match="terminal commit_unknown"):
        batch.close()
    with pytest.raises(RuntimeError, match="during a Policy update gate"):
        manager.open_batch(
            batch_id="after-optimizer-unknown",
            trajectory_ids=("run/after-optimizer-unknown/0/group",),
        )
    with pytest.raises(RuntimeError, match="outstanding batches"):
        with manager.checkpoint_barrier():
            pytest.fail("checkpoint admitted commit-unknown optimizer state")


def test_checkpoint_rejects_manager_owned_orphan_evidence() -> None:
    manager, store, behavior_store, ledger = _manager()
    store.put_tensor("orphan", torch.tensor([1.0]))
    _record_behavior(
        behavior_store,
        trajectory_id="run/orphan/0/group",
        seed=10,
    )
    ledger.execute_once(
        key=("run/orphan/0/group", 0),
        fingerprint="orphan",
        operation=lambda: ObservationHandle("orphan", SHA),
    )

    with pytest.raises(RuntimeError, match="manager-owned orphan evidence") as error:
        with manager.checkpoint_barrier():
            pytest.fail("checkpoint admitted orphan evidence")
    message = str(error.value)
    assert "focus_ledger_entries" in message
    assert "observation_tensors" in message
    assert "behavior_traces" in message


def test_cleanup_failure_is_terminal_and_blocks_checkpoint(monkeypatch) -> None:
    manager, store, behavior_store, ledger = _manager()
    batch = manager.open_batch(
        batch_id="cleanup-failure",
        trajectory_ids=("run/cleanup-failure/0/group",),
    )
    _populate_batch(
        batch,
        store=store,
        behavior_store=behavior_store,
        ledger=ledger,
        seed=9,
    )

    def fail_release(_trajectory_ids):
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(behavior_store, "release_trajectories", fail_release)
    with pytest.raises(RuntimeError, match="exhaustive release attempts"):
        batch.abort()
    assert batch.state is PolicyBatchState.CLEANUP_FAILED
    assert batch.cleanup_errors
    with pytest.raises(RuntimeError, match="terminal cleanup_failed"):
        batch.abort()
    with pytest.raises(RuntimeError, match="outstanding batches"):
        with manager.checkpoint_barrier():
            pytest.fail("checkpoint admitted cleanup_failed batch")
    with pytest.raises(RuntimeError, match="terminal batch cleanup failure"):
        manager.open_batch(
            batch_id="after-cleanup-failure",
            trajectory_ids=("run/after-cleanup-failure/0/group",),
        )
