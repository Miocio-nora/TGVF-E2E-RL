from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.observations.schema import (
    FOCUSED_OBSERVATION_SCHEMA_V2,
    FocusedObservationRecordV2,
    TrajectorySourceVisualV2,
)
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


def test_store_disambiguates_equal_bytes_with_different_tensor_semantics() -> None:
    store = ObservationStore()
    int_ref = store.put_tensor("int32-zero", torch.tensor([0], dtype=torch.int32))
    assert (
        int_ref.address.digest
        == store.put_tensor(
            "second-int32-zero", torch.tensor([0], dtype=torch.int32)
        ).address.digest
    )

    float_ref = store.put_tensor(
        "float32-zero", torch.tensor([0.0], dtype=torch.float32)
    )
    flat_ref = store.put_tensor("flat-zeros", torch.zeros(4, dtype=torch.uint8))
    matrix_ref = store.put_tensor(
        "matrix-zeros", torch.zeros((2, 2), dtype=torch.uint8)
    )

    assert int_ref.address.digest == float_ref.address.digest
    assert flat_ref.address.digest == matrix_ref.address.digest
    assert store.resolve_verified(int_ref).dtype is torch.int32
    assert store.resolve_verified(float_ref).dtype is torch.float32
    assert store.resolve_verified(flat_ref).shape == (4,)
    assert store.resolve_verified(matrix_ref).shape == (2, 2)


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


def test_legacy_v1_record_and_replay_checksums_remain_stable() -> None:
    store, observation_handle = populated_observation_store()
    replay_handle = _put_replay(store, observation_handle)

    assert observation_handle.record_sha256 == (
        "b3c99536c3629fc1304801572216d7f7e1e1896a046b34a9eacc20bf604e4bea"
    )
    assert replay_handle.record_sha256 == (
        "8b0afbaabe19eced858e82e8f9f4b8e2c926ac9c26a5343f212c5e52df02d6e7"
    )


def test_v2_replay_bundle_round_trip_carries_pixel_values_and_condition_hq() -> None:
    store, legacy_handle = populated_observation_store()
    legacy = store.resolve_record(legacy_handle)
    source_state = replace(legacy.source_visual, image_grid_thw=(1, 2, 2))
    hq = store.put_tensor(
        "observation-v2.condition_hq",
        torch.arange(8, dtype=torch.float32).view(2, 4),
        trajectory_id="smoke/sample/0/group",
    )
    record = FocusedObservationRecordV2(
        schema_version=FOCUSED_OBSERVATION_SCHEMA_V2,
        observation_id="observation-v2",
        call_index=legacy.call_index,
        model=legacy.model,
        representation=legacy.representation,
        condition=legacy.condition,
        source_visual=source_state,
        payload=legacy.payload,
        branches=legacy.branches,
        layout=legacy.layout,
        masks=legacy.masks,
        cache=legacy.cache,
        condition_hq=hq,
    )
    observation_handle = store.put(record)
    pixel_values = store.put_tensor(
        "replay-v2.pixel_values",
        torch.arange(12, dtype=torch.float32).view(4, 3),
        trajectory_id="smoke/sample/0/group",
    )
    source_visual = TrajectorySourceVisualV2(
        state=source_state,
        positions=record.layout.original_image_positions,
        deepstack_branch_layers=record.layout.deepstack_branch_layers,
        deepstack_injection_positions=tuple(
            record.layout.original_image_positions
            for _ in source_state.merged_deepstack
        ),
        preprocessed_pixel_values=pixel_values,
    )
    input_ids = store.put_tensor(
        "replay-v2.input_ids", torch.arange(12, dtype=torch.int64).view(1, 12)
    )
    replay_handle = store.put_replay(
        TrajectoryReplayRecord(
            schema_version="trajectory-replay-v1",
            replay_id="replay-v2",
            trajectory_id="smoke/sample/0/group",
            model=record.model,
            behavior_policy=policy_version(),
            source_visual=source_visual,
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
    )

    restored, restored_handle = ObservationStore.from_replay_bundle(
        store.export_replay_bundle(replay_handle)
    )
    restored_replay = restored.resolve_replay(restored_handle)
    assert isinstance(restored_replay.source_visual, TrajectorySourceVisualV2)
    restored_pixels = restored_replay.source_visual.preprocessed_pixel_values
    assert restored_pixels is not None
    torch.testing.assert_close(
        restored.resolve_verified(restored_pixels),
        torch.arange(12, dtype=torch.float32).view(4, 3),
        rtol=0,
        atol=0,
    )
    restored_record = restored.resolve_record(
        restored_replay.observation_handles[0]
    )
    assert isinstance(restored_record, FocusedObservationRecordV2)
    assert restored_record.condition_hq is not None
    torch.testing.assert_close(
        restored.resolve_verified(restored_record.condition_hq),
        torch.arange(8, dtype=torch.float32).view(2, 4),
        rtol=0,
        atol=0,
    )


def test_replay_bundle_rejects_tensor_mutation_after_transport() -> None:
    store, observation_handle = populated_observation_store()
    bundle = store.export_replay_bundle(_put_replay(store, observation_handle))
    bundle.tensor_payloads[0].tensor.zero_()

    with pytest.raises(ReplayMismatchError, match="replay tensor payload"):
        ObservationStore.from_replay_bundle(bundle)
