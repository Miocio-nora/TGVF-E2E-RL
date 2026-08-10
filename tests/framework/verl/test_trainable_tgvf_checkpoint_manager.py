from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.framework.verl.trainable_tgvf_checkpoint_manager import (
    TGVF_ADAPTER_UPDATE_MODE_FROZEN,
    TGVF_CHECKPOINT_ENGINE_CONTROL_KEY,
    TrainableTGVFCheckpointEngineManager,
)
from tgvf_rl.framework.verl.trainable_tgvf_weight_sync import (
    split_trainable_rp66_parameter_stream_for_snapshot,
)
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVF_ADAPTER_UPDATE_ACK_SCHEMA,
    adapter_owned_state_sha256,
)


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "TGVF_POLICY_STATE_DIR": str(tmp_path.resolve()),
        "TGVF_POLICY_RUN_ID": "PRL15-manager-test",
        "TGVF_POLICY_RUN_IDENTITY_SHA256": "1" * 64,
    }


def _replicas() -> list[object]:
    return [
        SimpleNamespace(servers=(object(),)),
        SimpleNamespace(servers=(object(),)),
    ]


def _config(*, adapter_update_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        engine_kwargs={
            TGVF_CHECKPOINT_ENGINE_CONTROL_KEY: {
                "adapter_update_mode": adapter_update_mode,
            }
        },
    )


def _ack(
    optimizer_step: int,
    state_sha256: str,
    tensor_count: int,
    *,
    applied: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": TGVF_ADAPTER_UPDATE_ACK_SCHEMA,
        "optimizer_step": optimizer_step,
        "state_sha256": state_sha256,
        "tensor_count": tensor_count,
        "applied": applied,
        "cleared_source_count": 0,
        "cleared_trace_count": 0,
    }


class _PublishingUpstream:
    def __init__(
        self,
        environment: dict[str, str],
        events: list[str],
        *,
        adapter_changes: bool = True,
    ) -> None:
        self.environment = environment
        self.events = events
        self.adapter_changes = adapter_changes
        self.calls: list[int] = []

    def update_weights(self, global_steps: int) -> dict[str, int]:
        self.calls.append(global_steps)
        self.events.append(f"qwen:{global_steps}")
        qwen = torch.tensor([100.0 + global_steps], dtype=torch.bfloat16)
        adapter_step = global_steps if self.adapter_changes else 0
        adapter = torch.tensor([float(adapter_step), 2.0], dtype=torch.bfloat16)
        forwarded = tuple(
            split_trainable_rp66_parameter_stream_for_snapshot(
                (
                    ("model.language_model.weight", qwen),
                    ("tgvf_adapter.query.weight", adapter),
                ),
                base_sync_done=False,
                rank=0,
                world_size=8,
                global_steps=global_steps,
                environment=self.environment,
            )
        )
        assert forwarded == (("model.language_model.weight", qwen),)
        return {"upstream_step": global_steps}

    def sleep_replicas(self) -> str:
        return "slept"


class _AcknowledgingRolloutManager:
    def __init__(self, events: list[str], *, ack_count: int = 2) -> None:
        self.events = events
        self.ack_count = ack_count
        self.calls: list[tuple[int, str, dict[str, torch.Tensor]]] = []

    async def update_adapter_owned_state(
        self,
        *,
        optimizer_step: int,
        state_sha256: str,
        state,
    ) -> tuple[dict[str, object], ...]:
        copied = {name: tensor.clone() for name, tensor in state.items()}
        assert adapter_owned_state_sha256(copied) == state_sha256
        self.calls.append((optimizer_step, state_sha256, copied))
        self.events.append(f"rp66:{optimizer_step}")
        return tuple(
            _ack(optimizer_step, state_sha256, len(copied))
            for _ in range(self.ack_count)
        )


def test_manager_syncs_step_zero_qwen_then_rp66_and_waits_for_all_acks(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    events: list[str] = []
    upstream = _PublishingUpstream(environment, events)
    rollout = _AcknowledgingRolloutManager(events)
    manager = TrainableTGVFCheckpointEngineManager(
        config="config",
        actor_wg="actor",
        replicas=_replicas(),
        upstream_manager_factory=lambda **_kwargs: upstream,
        rollout_manager_factory=lambda _replicas: rollout,
        environment=environment,
    )

    assert manager.sleep_replicas() == "slept"
    assert manager.update_weights(0) == {"upstream_step": 0}

    assert events == ["qwen:0", "rp66:0"]
    assert upstream.calls == [0]
    assert len(rollout.calls) == 1
    step, state_sha256, state = rollout.calls[0]
    assert step == 0
    assert set(state) == {"query.weight"}
    assert adapter_owned_state_sha256(state) == state_sha256
    publication = manager.last_publication
    assert publication is not None
    assert publication.optimizer_step == 0
    assert publication.adapter_state_sha256 == state_sha256
    assert publication.acknowledgement_count == 2
    assert publication.applied_count == 2


def test_manager_preserves_async_surface_for_later_optimizer_steps(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    events: list[str] = []
    upstream = _PublishingUpstream(environment, events)
    rollout = _AcknowledgingRolloutManager(events)
    manager = TrainableTGVFCheckpointEngineManager(
        config=object(),
        actor_wg=object(),
        replicas=_replicas(),
        upstream_manager_factory=lambda **_kwargs: upstream,
        rollout_manager_factory=lambda _replicas: rollout,
        environment=environment,
    )

    async def update_inside_loop() -> object:
        pending = manager.update_weights(1)
        assert inspect.isawaitable(pending)
        return await pending

    assert asyncio.run(update_inside_loop()) == {"upstream_step": 1}
    assert events == ["qwen:1", "rp66:1"]
    assert manager.last_publication is not None
    assert manager.last_publication.optimizer_step == 1


def test_historical_config_defaults_to_joint_and_allows_adapter_updates(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    events: list[str] = []
    upstream = _PublishingUpstream(environment, events)
    rollout = _AcknowledgingRolloutManager(events)
    manager = TrainableTGVFCheckpointEngineManager(
        config=object(),
        actor_wg=object(),
        replicas=_replicas(),
        upstream_manager_factory=lambda **_kwargs: upstream,
        rollout_manager_factory=lambda _replicas: rollout,
        environment=environment,
    )

    manager.update_weights(0)
    first = manager.last_publication
    assert first is not None
    manager.update_weights(1)

    assert [call[0] for call in rollout.calls] == [0, 1]
    assert rollout.calls[1][1] != first.adapter_state_sha256
    assert manager.last_publication is not None
    assert manager.last_publication.optimizer_step == 1


def test_frozen_adapter_publishes_a_step_pointer_with_constant_tensor_state(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    events: list[str] = []
    upstream = _PublishingUpstream(environment, events, adapter_changes=False)
    rollout = _AcknowledgingRolloutManager(events)
    manager = TrainableTGVFCheckpointEngineManager(
        config=_config(adapter_update_mode=TGVF_ADAPTER_UPDATE_MODE_FROZEN),
        actor_wg=object(),
        replicas=_replicas(),
        upstream_manager_factory=lambda **_kwargs: upstream,
        rollout_manager_factory=lambda _replicas: rollout,
        environment=environment,
    )

    manager.update_weights(0)
    first = manager.last_publication
    assert first is not None
    manager.update_weights(1)
    second = manager.last_publication

    assert second is not None
    assert [call[0] for call in rollout.calls] == [0, 1]
    assert rollout.calls[0][1] == rollout.calls[1][1]
    assert first.adapter_state_sha256 == second.adapter_state_sha256
    assert first.snapshot_storage_sha256 == second.snapshot_storage_sha256
    assert first.request_sha256 != second.request_sha256
    latest = json.loads((tmp_path / "latest-lora-snapshot.json").read_text())
    assert latest["optimizer_step"] == 1
    assert latest["request_sha256"] == second.request_sha256


def test_frozen_adapter_rejects_tensor_drift_before_rollout_publication(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    events: list[str] = []
    upstream = _PublishingUpstream(environment, events)
    rollout = _AcknowledgingRolloutManager(events)
    manager = TrainableTGVFCheckpointEngineManager(
        config=_config(adapter_update_mode=TGVF_ADAPTER_UPDATE_MODE_FROZEN),
        actor_wg=object(),
        replicas=_replicas(),
        upstream_manager_factory=lambda **_kwargs: upstream,
        rollout_manager_factory=lambda _replicas: rollout,
        environment=environment,
    )

    manager.update_weights(0)
    first = manager.last_publication
    assert first is not None
    with pytest.raises(
        IdentityMismatchError,
        match="frozen RP66 Adapter state changed",
    ):
        manager.update_weights(1)

    assert upstream.calls == [0, 1]
    assert [call[0] for call in rollout.calls] == [0]
    assert manager.last_publication == first


def test_manager_rejects_unknown_adapter_update_mode(tmp_path: Path) -> None:
    environment = _environment(tmp_path)

    with pytest.raises(ValueError, match="adapter_update_mode"):
        TrainableTGVFCheckpointEngineManager(
            config=_config(adapter_update_mode="frozen-ish"),
            actor_wg=object(),
            replicas=_replicas(),
            upstream_manager_factory=lambda **_kwargs: object(),
            rollout_manager_factory=lambda _replicas: object(),
            environment=environment,
        )


def test_manager_does_not_complete_when_any_rollout_server_ack_is_missing(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    events: list[str] = []
    upstream = _PublishingUpstream(environment, events)
    rollout = _AcknowledgingRolloutManager(events, ack_count=1)
    manager = TrainableTGVFCheckpointEngineManager(
        config=object(),
        actor_wg=object(),
        replicas=_replicas(),
        upstream_manager_factory=lambda **_kwargs: upstream,
        rollout_manager_factory=lambda _replicas: rollout,
        environment=environment,
    )

    with pytest.raises(RuntimeError, match="ACK count differs"):
        manager.update_weights(0)

    assert events == ["qwen:0", "rp66:0"]
    assert manager.last_publication is None


def test_manager_never_calls_rollout_when_rank_zero_snapshot_is_missing(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    rollout = _AcknowledgingRolloutManager([])

    class MissingSnapshotUpstream:
        def update_weights(self, global_steps: int) -> dict[str, int]:
            return {"upstream_step": global_steps}

    manager = TrainableTGVFCheckpointEngineManager(
        config=object(),
        actor_wg=object(),
        replicas=_replicas(),
        upstream_manager_factory=lambda **_kwargs: MissingSnapshotUpstream(),
        rollout_manager_factory=lambda _replicas: rollout,
        environment=environment,
    )

    with pytest.raises(ReplayMismatchError, match="latest trainable RP66 snapshot"):
        manager.update_weights(0)

    assert rollout.calls == []
    assert manager.last_publication is None
