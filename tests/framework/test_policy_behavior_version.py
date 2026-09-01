from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.framework.verl.full_qwen_checkpoint_manager import (
    FullQwenBehaviorCheckpointEngineManager,
)
from tgvf_rl.framework.verl.policy_behavior_version import (
    FULL_QWEN_IDENTITY_KIND,
    FullQwenSyncReceipt,
    PolicyBehaviorPayload,
    adapter_acknowledgement_sha256,
    load_latest_policy_behavior_snapshot,
    publish_policy_behavior_snapshot,
)
from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyWeightSyncState,
    publish_policy_weight_sync_request,
)
from tgvf_rl.framework.verl.trainable_tgvf_weight_sync import (
    load_latest_trainable_rp66_snapshot,
    split_trainable_rp66_parameter_stream_for_snapshot,
)
from tgvf_rl.framework.verl.vllm_tool_runtime import adapter_owned_state_sha256


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "TGVF_POLICY_STATE_DIR": str(tmp_path.resolve()),
        "TGVF_POLICY_RUN_ID": "method-behavior-test",
        "TGVF_POLICY_RUN_IDENTITY_SHA256": "a" * 64,
    }


def _receipt(
    state: PolicyWeightSyncState,
    step: int,
) -> FullQwenSyncReceipt:
    request = publish_policy_weight_sync_request(
        state, step, nonce=f"behavior-step-{step}"
    )
    return FullQwenSyncReceipt.from_acknowledged_request(request)


def test_full_qwen_behavior_needs_no_lora_pointer_and_advances_by_step(
    tmp_path: Path,
) -> None:
    state = PolicyWeightSyncState.from_environment(_environment(tmp_path))
    first = publish_policy_behavior_snapshot(
        state,
        full_qwen=_receipt(state, 0),
        payload=PolicyBehaviorPayload.FULL_QWEN,
    )

    assert not state.latest_path.exists()
    assert first.policy_version.optimizer_step == 0
    assert first.full_qwen_identity_kind == FULL_QWEN_IDENTITY_KIND
    assert first.adapter_state_sha256 is None
    assert load_latest_policy_behavior_snapshot(state) == first

    second = publish_policy_behavior_snapshot(
        state,
        full_qwen=_receipt(state, 1),
        payload=PolicyBehaviorPayload.FULL_QWEN,
    )
    assert second.policy_version.optimizer_step == 1
    assert second.policy_version.weights_sha256 != first.policy_version.weights_sha256


def test_same_step_resync_keeps_policy_version_but_records_new_request(
    tmp_path: Path,
) -> None:
    state = PolicyWeightSyncState.from_environment(_environment(tmp_path))
    first_request = publish_policy_weight_sync_request(
        state, 2, nonce="first-step-two-sync"
    )
    first = publish_policy_behavior_snapshot(
        state,
        full_qwen=FullQwenSyncReceipt.from_acknowledged_request(first_request),
        payload=PolicyBehaviorPayload.FULL_QWEN,
    )
    second_request = publish_policy_weight_sync_request(
        state, 2, nonce="second-step-two-sync"
    )
    second = publish_policy_behavior_snapshot(
        state,
        full_qwen=FullQwenSyncReceipt.from_acknowledged_request(second_request),
        payload=PolicyBehaviorPayload.FULL_QWEN,
    )

    assert first.request_sha256 != second.request_sha256
    assert first.full_qwen_sync_ack_sha256 != second.full_qwen_sync_ack_sha256
    assert first.full_qwen_sync_identity_sha256 == second.full_qwen_sync_identity_sha256
    assert first.policy_version == second.policy_version


def test_composite_behavior_binds_full_qwen_and_rp67_snapshot_identities(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    state = PolicyWeightSyncState.from_environment(environment)
    receipt = _receipt(state, 3)
    adapter_tensors = {"query.weight": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)}
    forwarded = tuple(
        split_trainable_rp66_parameter_stream_for_snapshot(
            (
                (
                    "model.language_model.weight",
                    torch.tensor([3.0], dtype=torch.bfloat16),
                ),
                ("tgvf_adapter.query.weight", adapter_tensors["query.weight"]),
            ),
            base_sync_done=False,
            rank=0,
            world_size=1,
            global_steps=3,
            environment=environment,
        )
    )
    assert len(forwarded) == 1
    stored = load_latest_trainable_rp66_snapshot(
        state,
        expected_optimizer_step=3,
        expected_request_sha256=receipt.request_sha256,
    )
    adapter_state = adapter_owned_state_sha256(adapter_tensors)
    adapter_storage = stored.storage_sha256
    adapter_ack = adapter_acknowledgement_sha256(
        optimizer_step=3,
        state_sha256=adapter_state,
        snapshot_storage_sha256=adapter_storage,
        request_sha256=receipt.request_sha256,
        acknowledgement_count=2,
        applied_count=2,
    )
    snapshot = publish_policy_behavior_snapshot(
        state,
        full_qwen=receipt,
        payload=PolicyBehaviorPayload.FULL_QWEN_PLUS_RP66,
        adapter_state_sha256=adapter_state,
        adapter_snapshot_storage_sha256=adapter_storage,
        adapter_acknowledgement_sha256=adapter_ack,
        adapter_acknowledgement_count=2,
        adapter_applied_count=2,
    )

    assert snapshot.adapter_state_sha256 == adapter_state
    assert snapshot.adapter_snapshot_storage_sha256 == adapter_storage
    assert snapshot.adapter_acknowledgement_sha256 == adapter_ack
    # The behavior policy is composite; it must never collapse to the Adapter
    # tensor/storage digest alone.
    assert snapshot.policy_version.weights_sha256 not in {
        adapter_state,
        adapter_storage,
        adapter_ack,
        snapshot.full_qwen_sync_ack_sha256,
    }

    tampered = bytearray(snapshot.pointer_bytes)
    tampered[tampered.index(b"full_qwen_plus_rp66")] = ord("x")
    snapshot.pointer_file.write_bytes(tampered)
    with pytest.raises(ReplayMismatchError):
        load_latest_policy_behavior_snapshot(state)


class _Upstream:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def update_weights(self, *, global_steps: int) -> dict[str, int]:
        self.calls.append(global_steps)
        return {"accepted_step": global_steps}

    def sleep_replicas(self) -> str:
        return "slept"


class _CheckpointReplica:
    def __init__(self) -> None:
        self.calls = 0

    async def sleep_for_checkpoint(self) -> str:
        self.calls += 1
        return "retained"


def test_full_qwen_manager_reuses_upstream_transport_and_publishes_after_ack(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    upstream = _Upstream()
    replica = _CheckpointReplica()
    manager = FullQwenBehaviorCheckpointEngineManager(
        config=object(),
        actor_wg=object(),
        replicas=[replica],
        upstream_manager_factory=lambda **_kwargs: upstream,
        environment=environment,
    )

    assert manager.update_weights(0) == {"accepted_step": 0}
    assert upstream.calls == [0]
    assert manager.sleep_replicas() == "slept"
    assert manager.last_behavior_snapshot is not None
    assert manager.last_behavior_snapshot.payload is PolicyBehaviorPayload.FULL_QWEN
    state = PolicyWeightSyncState.from_environment(environment)
    assert not state.latest_path.exists()
    assert load_latest_policy_behavior_snapshot(state) == (
        manager.last_behavior_snapshot
    )
    assert manager.checkpoint_sleep_preserves_weights is True
    assert manager.sleep_replicas_for_checkpoint() == ("retained",)
    assert replica.calls == 1
