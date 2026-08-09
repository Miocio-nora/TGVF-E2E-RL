from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.framework.verl.trainable_tgvf_checkpoint_manager import (
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
    def __init__(self, environment: dict[str, str], events: list[str]) -> None:
        self.environment = environment
        self.events = events
        self.calls: list[int] = []

    def update_weights(self, global_steps: int) -> dict[str, int]:
        self.calls.append(global_steps)
        self.events.append(f"qwen:{global_steps}")
        qwen = torch.tensor([100.0 + global_steps], dtype=torch.bfloat16)
        adapter = torch.tensor([float(global_steps), 2.0], dtype=torch.bfloat16)
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
