from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.environment.focus_runtime import FocusExecutionLedger
from tgvf_rl.framework.verl import (
    POLICY_PILOT_CHECKPOINT_PAIR_FILENAME,
    POLICY_PILOT_PROJECT_STATE_FILENAME,
    FSDP2BridgeConfig,
    PairedPolicyPilotVerlCheckpoint,
)
from tgvf_rl.observations.store import ObservationStore, tensor_checksum
from tgvf_rl.policy.checkpoint import (
    DATA_CURSOR_OWNER,
    ROLLOUT_RNG_OWNER,
    ROLLOUT_SAMPLER_OWNER,
    OpaqueProjectState,
    PilotOptimizerDataCursor,
    PilotRunIdentityHashes,
)
from tgvf_rl.policy.lifecycle import PolicyBatchLifecycleManager
from tgvf_rl.policy.metrics import (
    PilotMetricsAccumulator,
    PilotMetricsCheckpointState,
)
from tgvf_rl.trajectories.behavior import BehaviorTraceStore


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64


def _run_identity() -> PilotRunIdentityHashes:
    return PilotRunIdentityHashes.from_hashes(
        "pilot-run-a",
        {
            "data_manifest": SHA0,
            "policy_config": SHA1,
            "prompt": SHA2,
        },
    )


def _metrics(step: int) -> PilotMetricsAccumulator:
    value = PilotMetricsAccumulator()
    value.restore_checkpoint_state(
        PilotMetricsCheckpointState(
            optimizer_steps=step,
            prompts=step,
            trajectories=8 * step,
            generated_policy_tokens=16 * step,
            reasoning_tokens=8 * step,
            original_visual_tokens=32 * step,
            total_visual_tokens=32 * step,
            step_time_seconds_total=0.25 * step,
        )
    )
    return value


def _lifecycle_manager() -> PolicyBatchLifecycleManager:
    return PolicyBatchLifecycleManager(
        observation_store=ObservationStore(),
        behavior_store=BehaviorTraceStore(),
        focus_execution_ledger=FocusExecutionLedger(),
    )


class _UpstreamCheckpoint:
    """CPU stand-in for veRL's synchronous model/optimizer/extra checkpoint."""

    def __init__(self, *, initial_value: float, fail_save: bool = False) -> None:
        self.lora = torch.nn.Parameter(torch.tensor([initial_value]))
        self.optimizer = torch.optim.AdamW([self.lora], lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
        )
        self.global_step = 0
        self.fail_save = fail_save
        self.save_calls = 0
        self.load_calls = 0
        self.on_save = None

    def update_once(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.lora.square().sum().backward()
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
    ):
        del hdfs_path, max_ckpt_to_keep
        self.save_calls += 1
        destination = Path(local_path)
        destination.mkdir(parents=True, exist_ok=True)
        if self.on_save is not None:
            self.on_save()
        if self.fail_save:
            raise RuntimeError("injected upstream save failure")
        if global_step != self.global_step:
            raise RuntimeError("fake upstream global step mismatch")
        torch.save(
            {
                "lora": self.lora.detach().clone(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            destination / "upstream_model_optimizer_extra.pt",
        )

    def load_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        del_local_after_load=False,
    ):
        del hdfs_path, del_local_after_load
        self.load_calls += 1
        state = torch.load(
            Path(local_path) / "upstream_model_optimizer_extra.pt",
            map_location="cpu",
            weights_only=False,
        )
        with torch.no_grad():
            self.lora.copy_(state["lora"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]


class _ProjectStatePort:
    def __init__(
        self,
        upstream: _UpstreamCheckpoint,
        *,
        progress: PilotOptimizerDataCursor,
        sampler: OpaqueProjectState,
        rng: OpaqueProjectState,
    ) -> None:
        self.upstream = upstream
        self._run_identity = _run_identity()
        self._progress = progress
        self._sampler = sampler
        self._rng = rng
        self._reference = PolicyVersion("frozen-qwen3-reference", 0, SHA2)

    def run_identity(self):
        return self._run_identity

    def progress(self):
        return self._progress

    def rollout_sampler_state(self):
        return self._sampler

    def rollout_rng_state(self):
        return self._rng

    def current_policy_version(self):
        return PolicyVersion(
            self._run_identity.run_id,
            self.upstream.global_step,
            tensor_checksum(self.upstream.lora.detach()),
        )

    def reference_policy_version(self):
        return self._reference

    def restore_progress(self, value):
        self._progress = value

    def restore_rollout_sampler_state(self, value):
        self._sampler = value

    def restore_rollout_rng_state(self, value):
        self._rng = value


def _project_values(step: int):
    return (
        PilotOptimizerDataCursor(
            step,
            OpaqueProjectState(
                DATA_CURSOR_OWNER,
                "json-v1",
                json.dumps({"next_prompt": 4 * step}).encode(),
            ),
        ),
        OpaqueProjectState(
            ROLLOUT_SAMPLER_OWNER, "sampler-v1", f"sampler-{step}".encode()
        ),
        OpaqueProjectState(
            ROLLOUT_RNG_OWNER, "vllm-rng-v1", f"rng-{step}".encode()
        ),
    )


def _bridge(upstream, manager, port, metrics):
    return PairedPolicyPilotVerlCheckpoint(
        upstream=upstream,
        fsdp2=FSDP2BridgeConfig(),
        lifecycle_manager=manager,
        state_port=port,
        metrics_accumulator=metrics,
    )


def test_clean_process_resume_pairs_upstream_and_project_owned_state(tmp_path) -> None:
    source_upstream = _UpstreamCheckpoint(initial_value=2.0)
    source_upstream.update_once()
    source_manager = _lifecycle_manager()
    source_values = _project_values(1)
    source_port = _ProjectStatePort(
        source_upstream,
        progress=source_values[0],
        sampler=source_values[1],
        rng=source_values[2],
    )
    source_metrics = _metrics(1)
    source = _bridge(source_upstream, source_manager, source_port, source_metrics)
    checkpoint_dir = (tmp_path / "global_step_1" / "actor").resolve()

    gate_errors = []

    def try_open_during_save() -> None:
        try:
            source_manager.open_batch(
                batch_id="during-save",
                trajectory_ids=("pilot/during-save/0/group",),
            )
        except RuntimeError as error:
            gate_errors.append(str(error))

    source_upstream.on_save = try_open_during_save
    saved = source.save_checkpoint(checkpoint_dir, global_step=1)
    assert gate_errors and "checkpoint capture" in gate_errors[0]
    assert (checkpoint_dir / POLICY_PILOT_PROJECT_STATE_FILENAME).is_file()
    assert (checkpoint_dir / POLICY_PILOT_CHECKPOINT_PAIR_FILENAME).is_file()

    target_upstream = _UpstreamCheckpoint(initial_value=-9.0)
    target_manager = _lifecycle_manager()
    target_values = _project_values(0)
    target_port = _ProjectStatePort(
        target_upstream,
        progress=target_values[0],
        sampler=target_values[1],
        rng=target_values[2],
    )
    target_metrics = _metrics(0)
    target = _bridge(target_upstream, target_manager, target_port, target_metrics)
    result = target.load_checkpoint(checkpoint_dir)

    assert result.optimizer_step == 1
    assert result.project_state == saved
    torch.testing.assert_close(target_upstream.lora, source_upstream.lora)
    source_optim = source_upstream.optimizer.state_dict()
    target_optim = target_upstream.optimizer.state_dict()
    assert source_optim["param_groups"] == target_optim["param_groups"]
    for name, expected in source_optim["state"][0].items():
        torch.testing.assert_close(target_optim["state"][0][name], expected)
    assert target_upstream.scheduler.state_dict() == source_upstream.scheduler.state_dict()
    assert target_port.progress() == saved.progress
    assert target_port.rollout_sampler_state() == saved.rollout_sampler_state
    assert target_port.rollout_rng_state() == saved.rollout_rng_state
    assert target_metrics.state == saved.metrics_state
    assert target_port.current_policy_version() == saved.policy_version

    after = target_manager.open_batch(
        batch_id="after-resume",
        trajectory_ids=("pilot/after-resume/0/group",),
    )
    after.abort()


def test_resume_records_operational_policy_after_upstream_load(tmp_path) -> None:
    source_upstream = _UpstreamCheckpoint(initial_value=2.0)
    source_upstream.update_once()
    source_values = _project_values(1)
    checkpoint_dir = (tmp_path / "global_step_1" / "actor").resolve()
    saved = _bridge(
        source_upstream,
        _lifecycle_manager(),
        _ProjectStatePort(
            source_upstream,
            progress=source_values[0],
            sampler=source_values[1],
            rng=source_values[2],
        ),
        _metrics(1),
    ).save_checkpoint(checkpoint_dir, global_step=1)

    class OperationalReceiptPort(_ProjectStatePort):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.receipt = PolicyVersion(self._run_identity.run_id, 0, SHA0)

        def record_loaded_policy_version(self, value) -> None:
            assert self.upstream.global_step == value.optimizer_step
            self.receipt = value

        def current_policy_version(self):
            return self.receipt

    target_upstream = _UpstreamCheckpoint(initial_value=-3.0)
    target_values = _project_values(0)
    target_port = OperationalReceiptPort(
        target_upstream,
        progress=target_values[0],
        sampler=target_values[1],
        rng=target_values[2],
    )
    result = _bridge(
        target_upstream,
        _lifecycle_manager(),
        target_port,
        _metrics(0),
    ).load_checkpoint(checkpoint_dir)

    assert result.optimizer_step == 1
    assert target_port.receipt == saved.policy_version


def test_incomplete_pair_is_rejected_before_upstream_or_project_restore(tmp_path) -> None:
    source_upstream = _UpstreamCheckpoint(initial_value=2.0)
    source_upstream.update_once()
    source_values = _project_values(1)
    source = _bridge(
        source_upstream,
        _lifecycle_manager(),
        _ProjectStatePort(
            source_upstream,
            progress=source_values[0],
            sampler=source_values[1],
            rng=source_values[2],
        ),
        _metrics(1),
    )
    checkpoint_dir = (tmp_path / "global_step_1" / "actor").resolve()
    source.save_checkpoint(checkpoint_dir, global_step=1)
    (checkpoint_dir / POLICY_PILOT_CHECKPOINT_PAIR_FILENAME).unlink()

    target_upstream = _UpstreamCheckpoint(initial_value=-3.0)
    target_values = _project_values(0)
    target_port = _ProjectStatePort(
        target_upstream,
        progress=target_values[0],
        sampler=target_values[1],
        rng=target_values[2],
    )
    target_metrics = _metrics(0)
    target = _bridge(
        target_upstream, _lifecycle_manager(), target_port, target_metrics
    )
    before = (target_port.progress(), target_metrics.state)

    with pytest.raises(ReplayMismatchError, match="incomplete"):
        target.load_checkpoint(checkpoint_dir)
    assert target_upstream.load_calls == 0
    assert (target_port.progress(), target_metrics.state) == before


def test_upstream_save_failure_never_commits_pair_and_releases_gate(tmp_path) -> None:
    upstream = _UpstreamCheckpoint(initial_value=2.0, fail_save=True)
    upstream.update_once()
    values = _project_values(1)
    manager = _lifecycle_manager()
    bridge = _bridge(
        upstream,
        manager,
        _ProjectStatePort(
            upstream, progress=values[0], sampler=values[1], rng=values[2]
        ),
        _metrics(1),
    )
    checkpoint_dir = (tmp_path / "failed" / "actor").resolve()

    with pytest.raises(RuntimeError, match="upstream save failure"):
        bridge.save_checkpoint(checkpoint_dir, global_step=1)
    assert not (checkpoint_dir / POLICY_PILOT_PROJECT_STATE_FILENAME).exists()
    assert not (checkpoint_dir / POLICY_PILOT_CHECKPOINT_PAIR_FILENAME).exists()
    batch = manager.open_batch(
        batch_id="after-failed-save",
        trajectory_ids=("pilot/after-failed-save/0/group",),
    )
    batch.abort()


def test_loaded_policy_identity_mismatch_does_not_restore_project_state(
    tmp_path, monkeypatch
) -> None:
    source_upstream = _UpstreamCheckpoint(initial_value=2.0)
    source_upstream.update_once()
    source_values = _project_values(1)
    checkpoint_dir = (tmp_path / "global_step_1" / "actor").resolve()
    _bridge(
        source_upstream,
        _lifecycle_manager(),
        _ProjectStatePort(
            source_upstream,
            progress=source_values[0],
            sampler=source_values[1],
            rng=source_values[2],
        ),
        _metrics(1),
    ).save_checkpoint(checkpoint_dir, global_step=1)

    target_upstream = _UpstreamCheckpoint(initial_value=-3.0)
    original_load = target_upstream.load_checkpoint

    def load_then_change_weight(*args, **kwargs):
        original_load(*args, **kwargs)
        with torch.no_grad():
            target_upstream.lora.add_(1.0)

    monkeypatch.setattr(target_upstream, "load_checkpoint", load_then_change_weight)
    target_values = _project_values(0)
    target_port = _ProjectStatePort(
        target_upstream,
        progress=target_values[0],
        sampler=target_values[1],
        rng=target_values[2],
    )
    target_metrics = _metrics(0)
    target = _bridge(
        target_upstream, _lifecycle_manager(), target_port, target_metrics
    )
    before = (target_port.progress(), target_metrics.state)

    with pytest.raises(IdentityMismatchError, match="policy_version"):
        target.load_checkpoint(checkpoint_dir)
    assert (target_port.progress(), target_metrics.state) == before
