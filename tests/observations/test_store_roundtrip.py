from __future__ import annotations

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.observations.store import (
    ObservationStore,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
)
from tests.support import (
    populated_observation_store,
    policy_version,
    trajectory_source_visual,
)


def test_store_is_bit_preserving_and_returns_clones() -> None:
    store, handle = populated_observation_store()
    record = store.resolve_record(handle)
    first = store.resolve_verified(record.payload.main_d)
    assert first.dtype is torch.bfloat16
    first.zero_()
    second = store.resolve_verified(record.payload.main_d)
    assert torch.count_nonzero(second).item() > 0


def test_store_checkpoint_round_trip_keeps_record_and_tensors() -> None:
    store, handle = populated_observation_store()
    restored = ObservationStore.from_checkpoint_state(store.checkpoint_state())
    record = restored.resolve_record(handle)
    torch.testing.assert_close(
        restored.resolve_verified(record.payload.main_d),
        store.resolve_verified(record.payload.main_d),
        rtol=0,
        atol=0,
    )


def test_unknown_observation_fails_closed() -> None:
    store, handle = populated_observation_store()
    bad = type(handle)("missing", handle.record_sha256)
    with pytest.raises(ReplayMismatchError, match="unknown observation"):
        store.resolve_record(bad)


def _put_replay(store: ObservationStore, observation_handle):
    record = store.resolve_record(observation_handle)
    input_ids = store.put_tensor(
        "replay.input_ids", torch.arange(12, dtype=torch.int64).view(1, 12)
    )
    replay = TrajectoryReplayRecord(
        schema_version="trajectory-replay-v1",
        replay_id="replay-0",
        trajectory_id="smoke/sample/0/group",
        model=record.model,
        behavior_policy=policy_version(),
        source_visual=trajectory_source_visual(record),
        observation_handles=(observation_handle,),
        tensors=TrajectoryReplayTensorRefs(
            input_ids=input_ids,
            position_ids=record.payload.position_ids,
            attention_mask=record.payload.attention_mask,
            policy_attention_mask=record.masks.policy_visible,
            reference_attention_mask=record.masks.reference_visible,
            teacher_attention_mask=record.masks.teacher_visible,
        ),
    )
    return store.put_replay(replay)


def test_replay_bundle_is_self_contained_across_worker_store_boundary() -> None:
    store, observation_handle = populated_observation_store()
    replay_handle = _put_replay(store, observation_handle)
    bundle = store.export_replay_bundle(replay_handle)

    restored, restored_handle = ObservationStore.from_replay_bundle(bundle)
    assert restored_handle == replay_handle
    restored_replay = restored.resolve_replay(restored_handle)
    restored_record = restored.resolve_record(restored_replay.observation_handles[0])
    source_record = store.resolve_record(observation_handle)
    torch.testing.assert_close(
        restored.resolve_verified(restored_record.payload.main_d),
        store.resolve_verified(source_record.payload.main_d),
        rtol=0,
        atol=0,
    )


def test_replay_bundle_rejects_tensor_mutation_after_transport() -> None:
    store, observation_handle = populated_observation_store()
    bundle = store.export_replay_bundle(_put_replay(store, observation_handle))
    bundle.tensor_payloads[0].tensor.zero_()

    with pytest.raises(ReplayMismatchError, match="replay tensor payload"):
        ObservationStore.from_replay_bundle(bundle)
