from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

import pytest
import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.framework.verl.policy_weight_sync import (
    POLICY_LORA_LATEST_FILENAME,
    PolicyWeightSyncState,
    TGVFPolicyCheckpointEngineManager,
    load_latest_lora_snapshot,
    load_latest_policy_version,
    load_policy_weight_sync_request,
    lora_parameter_mapping_sha256,
    publish_policy_weight_sync_request,
    wrap_lora_parameter_stream_for_snapshot,
)


RUN_IDENTITY = "7" * 64


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "TGVF_POLICY_STATE_DIR": str((tmp_path / "policy-state").resolve()),
        "TGVF_POLICY_RUN_ID": "policy-pilot-test",
        "TGVF_POLICY_RUN_IDENTITY_SHA256": RUN_IDENTITY,
        "RANK": "0",
        "WORLD_SIZE": "4",
    }


def _stream(step: int = 0) -> list[tuple[str, torch.Tensor]]:
    return [
        (
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            torch.tensor([[1.0 + step, 2.0]], dtype=torch.bfloat16),
        ),
        (
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
            torch.tensor([[3.0], [4.0 + step]], dtype=torch.bfloat16),
        ),
    ]


def _publish(
    tmp_path: Path,
    *,
    step: int,
    rank: int = 0,
) -> tuple[PolicyWeightSyncState, list[tuple[str, torch.Tensor]]]:
    environment = _environment(tmp_path)
    state = PolicyWeightSyncState.from_environment(environment)
    publish_policy_weight_sync_request(state, step, nonce=f"request-{step}")
    source = _stream(step)
    observed = list(
        wrap_lora_parameter_stream_for_snapshot(
            iter(source),
            base_sync_done=True,
            rank=rank,
            world_size=4,
            global_steps=step,
            environment=environment,
        )
    )
    return state, observed


def test_rank_zero_publishes_exact_snapshot_without_changing_stream(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    state = PolicyWeightSyncState.from_environment(environment)
    request = publish_policy_weight_sync_request(state, 5, nonce="fixed-request")
    source = _stream(5)
    wrapped = wrap_lora_parameter_stream_for_snapshot(
        iter(source),
        base_sync_done=True,
        rank=0,
        world_size=4,
        global_steps=5,
        environment=environment,
    )

    assert not state.latest_path.exists()
    observed = list(wrapped)

    assert len(observed) == len(source)
    assert all(actual is expected for actual, expected in zip(observed, source, strict=True))
    assert all(
        actual[1] is expected[1]
        for actual, expected in zip(observed, source, strict=True)
    )
    snapshot = load_latest_lora_snapshot(
        state,
        expected_optimizer_step=5,
        expected_request_sha256=request.request_sha256,
    )
    expected_mapping = {name: tensor for name, tensor in source}
    assert snapshot.policy_version == PolicyVersion(
        state.run_id,
        5,
        lora_parameter_mapping_sha256(expected_mapping),
    )
    assert tuple(sorted(snapshot.tensors)) == tuple(sorted(expected_mapping))
    for name, tensor in expected_mapping.items():
        torch.testing.assert_close(snapshot.tensors[name], tensor.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_nonzero_ranks_leave_lora_stream_and_state_unwritten(
    tmp_path: Path, rank: int
) -> None:
    environment = _environment(tmp_path)
    state = PolicyWeightSyncState.from_environment(environment)
    publish_policy_weight_sync_request(state, 2, nonce="non-writer")
    source = _stream(2)

    observed = list(
        wrap_lora_parameter_stream_for_snapshot(
            iter(source),
            base_sync_done=True,
            rank=rank,
            world_size=4,
            global_steps=2,
            environment=environment,
        )
    )

    assert all(actual is expected for actual, expected in zip(observed, source, strict=True))
    assert not state.latest_path.exists()


def test_base_model_stream_passes_without_snapshot_state(tmp_path: Path) -> None:
    source = _stream()
    observed = list(
        wrap_lora_parameter_stream_for_snapshot(
            iter(source),
            base_sync_done=False,
            environment={},
        )
    )

    assert all(actual is expected for actual, expected in zip(observed, source, strict=True))
    assert not (tmp_path / POLICY_LORA_LATEST_FILENAME).exists()


def test_strict_latest_load_rejects_safetensors_tampering(tmp_path: Path) -> None:
    state, _ = _publish(tmp_path, step=3)
    snapshot = load_latest_lora_snapshot(state, expected_optimizer_step=3)
    value = bytearray(snapshot.tensor_file.read_bytes())
    value[-1] ^= 1
    snapshot.tensor_file.write_bytes(value)

    with pytest.raises(ReplayMismatchError, match="safetensors file digest"):
        load_latest_policy_version(state, expected_optimizer_step=3)


def test_strict_latest_load_rejects_pointer_tampering(tmp_path: Path) -> None:
    state, _ = _publish(tmp_path, step=4)
    payload = json.loads(state.latest_path.read_text(encoding="utf-8"))
    payload["optimizer_step"] = 9
    state.latest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReplayMismatchError, match="integrity mismatch"):
        load_latest_policy_version(state)


def test_step_mismatch_fails_before_lora_stream_is_consumed(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    state = PolicyWeightSyncState.from_environment(environment)
    publish_policy_weight_sync_request(state, 6, nonce="step-six")
    consumed = False

    def source():
        nonlocal consumed
        consumed = True
        yield from _stream(6)

    with pytest.raises(IdentityMismatchError, match="optimizer step"):
        wrap_lora_parameter_stream_for_snapshot(
            source(),
            base_sync_done=True,
            rank=0,
            world_size=4,
            global_steps=7,
            environment=environment,
        )
    assert consumed is False
    assert not state.latest_path.exists()


class _PublishingUpstreamManager:
    def __init__(self, *, environment: dict[str, str], **kwargs: object) -> None:
        self.environment = environment
        self.constructor_kwargs = kwargs
        self.calls: list[int] = []

    async def update_weights(self, global_steps: int) -> dict[str, int]:
        await asyncio.sleep(0)
        state = PolicyWeightSyncState.from_environment(self.environment)
        request = load_policy_weight_sync_request(state)
        assert request.optimizer_step == global_steps
        list(
            wrap_lora_parameter_stream_for_snapshot(
                iter(_stream(global_steps)),
                base_sync_done=True,
                rank=0,
                world_size=4,
                global_steps=global_steps,
                environment=self.environment,
            )
        )
        self.calls.append(global_steps)
        return {"upstream_step": global_steps}

    def sleep_replicas(self) -> str:
        return "slept"


def test_manager_preserves_pinned_sync_and_async_surface(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    created: list[_PublishingUpstreamManager] = []

    def factory(**kwargs: object) -> _PublishingUpstreamManager:
        manager = _PublishingUpstreamManager(environment=environment, **kwargs)
        created.append(manager)
        return manager

    wrapper = TGVFPolicyCheckpointEngineManager(
        config="config",
        actor_wg="actor",
        replicas=["replica"],
        upstream_manager_factory=factory,
        environment=environment,
    )

    assert wrapper.sleep_replicas() == "slept"
    assert wrapper.update_weights(1) == {"upstream_step": 1}
    assert wrapper.last_policy_version == load_latest_policy_version(
        PolicyWeightSyncState.from_environment(environment),
        expected_optimizer_step=1,
    )

    async def update_inside_loop() -> object:
        pending = wrapper.update_weights(2)
        assert inspect.isawaitable(pending)
        return await pending

    assert asyncio.run(update_inside_loop()) == {"upstream_step": 2}
    assert created[0].calls == [1, 2]
    assert wrapper.last_policy_version is not None
    assert wrapper.last_policy_version.optimizer_step == 2


def test_manager_rejects_upstream_sync_without_exact_snapshot(tmp_path: Path) -> None:
    environment = _environment(tmp_path)

    class MissingSnapshotManager:
        async def update_weights(self, global_steps: int) -> dict[str, int]:
            return {"upstream_step": global_steps}

    wrapper = TGVFPolicyCheckpointEngineManager(
        config=object(),
        actor_wg=object(),
        replicas=[],
        upstream_manager_factory=lambda **kwargs: MissingSnapshotManager(),
        environment=environment,
    )

    with pytest.raises(ReplayMismatchError, match="latest LoRA pointer"):
        wrapper.update_weights(0)
    assert wrapper.last_policy_version is None
