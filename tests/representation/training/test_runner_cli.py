from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl import cli
from tgvf_rl.representation.training import runner as runner_module
from tgvf_rl.representation.training.performance import (
    RepresentationRankTrainStepResources,
    RepresentationTrainStepPerformance,
)
from tgvf_rl.representation.training.trainer import RepresentationStepMetrics


@pytest.fixture(autouse=True)
def _forbid_real_accelerator_or_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CPU runner contract test attempted accelerator startup")

    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)


def _launch_config(
    physical_gpu_ids: tuple[int, ...] = (2, 3),
) -> SimpleNamespace:
    return SimpleNamespace(
        fsdp2=SimpleNamespace(
            world_size=len(physical_gpu_ids),
            physical_gpu_ids=physical_gpu_ids,
        )
    )


def _set_valid_launch_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "CUDA_VISIBLE_DEVICES": "2,3",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "WORLD_SIZE": "2",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_launch_environment_is_exact_and_rank_values_are_ascii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _launch_config()
    _set_valid_launch_environment(monkeypatch)

    runner_module._require_launch_environment(config)
    assert runner_module._environment_integer("RANK") == 0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2")
    with pytest.raises(ValueError, match="launch environment mismatch"):
        runner_module._require_launch_environment(config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    monkeypatch.setenv("RANK", "١")
    with pytest.raises(ValueError, match="ASCII|torchrun integer|non-negative"):
        runner_module._require_launch_environment(config)
    monkeypatch.setenv("RANK", "-1")
    with pytest.raises(ValueError, match="non-negative torchrun integer"):
        runner_module._require_launch_environment(config)


def test_launch_environment_is_bound_to_configured_physical_gpu_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _launch_config((0, 3))
    _set_valid_launch_environment(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3")

    runner_module._require_launch_environment(config)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,0")
    with pytest.raises(ValueError, match="launch environment mismatch"):
        runner_module._require_launch_environment(config)


def test_launch_environment_accepts_exact_world4_gpu_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _launch_config((0, 1, 2, 3))
    _set_valid_launch_environment(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("WORLD_SIZE", "4")

    runner_module._require_launch_environment(config)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,3,2")
    with pytest.raises(ValueError, match="launch environment mismatch"):
        runner_module._require_launch_environment(config)


def test_public_runner_checks_launch_contract_before_distributed_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _launch_config()
    _set_valid_launch_environment(monkeypatch)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    verified: list[object] = []
    monkeypatch.setattr(
        runner_module,
        "load_representation_training_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        runner_module,
        "_verify_live_code_identity",
        lambda value: verified.append(value),
    )

    with pytest.raises(ValueError, match="launch environment mismatch"):
        runner_module.run_representation_training("/unused/config.toml")

    assert verified == []


def test_candidate_runtime_lock_is_part_of_representation_code_identity() -> None:
    assert {
        "requirements/compatibility.lock",
        "requirements/compatibility-torch211-cu129.lock",
    }.issubset(runner_module._CODE_IDENTITY_PATHS)


@pytest.mark.parametrize(
    ("world_size", "accumulation_steps", "direct_groups", "expected"),
    ((2, 4, 1, 8), (2, 1, 4, 8), (2, 2, 4, 8), (4, 2, 1, 8)),
)
def test_global_matrix_count_uses_optimizer_step_group_identity(
    world_size: int,
    accumulation_steps: int,
    direct_groups: int,
    expected: int,
) -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(
            gradient_accumulation_steps=accumulation_steps,
            groups_per_rank_per_optimizer_step=direct_groups,
        )
    )

    assert (
        runner_module._optimizer_step_global_matrix_count(config, world_size=world_size)
        == expected
    )


def test_public_runner_lifecycle_can_be_wired_without_starting_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = object()
    calls: list[object] = []
    state = {"initialized": False}
    monkeypatch.setattr(
        runner_module,
        "load_representation_training_config",
        lambda path: calls.append(("load", Path(path))) or config,
    )
    monkeypatch.setattr(
        runner_module,
        "_require_launch_environment",
        lambda value: calls.append(("environment", value)),
    )
    monkeypatch.setattr(
        runner_module,
        "_verify_live_code_identity",
        lambda value: calls.append(("code", value)),
    )

    def init_process_group(*, backend: str, timeout: object) -> None:
        calls.append(("init", backend, timeout))
        state["initialized"] = True

    def destroy_process_group() -> None:
        calls.append("destroy")
        state["initialized"] = False

    monkeypatch.setattr(torch.distributed, "init_process_group", init_process_group)
    monkeypatch.setattr(
        torch.distributed, "is_initialized", lambda: state["initialized"]
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", destroy_process_group
    )
    monkeypatch.setattr(
        runner_module,
        "_run_initialized",
        lambda value, *, rank, stop_after_global_step: (
            calls.append(("run", value, rank, stop_after_global_step))
            or {"status": "synthetic", "rank": rank}
        ),
    )

    result = runner_module.run_representation_training("/unused/config.toml")

    assert result == {"status": "synthetic", "rank": 0}
    assert [call[0] for call in calls if isinstance(call, tuple)] == [
        "load",
        "environment",
        "code",
        "init",
        "run",
    ]
    assert calls[-1] == "destroy"
    assert state["initialized"] is False


def test_public_runner_aborts_process_group_on_rank_local_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = object()
    calls: list[str] = []
    state = {"initialized": False}
    monkeypatch.setattr(
        runner_module, "load_representation_training_config", lambda _path: config
    )
    monkeypatch.setattr(runner_module, "_validate_invocation_stop", lambda *_args: None)
    monkeypatch.setattr(
        runner_module, "_require_launch_environment", lambda _value: None
    )
    monkeypatch.setattr(
        runner_module, "_verify_live_code_identity", lambda _value: None
    )

    def init_process_group(*, backend: str, timeout: object) -> None:
        assert backend == "nccl"
        state["initialized"] = True

    monkeypatch.setattr(torch.distributed, "init_process_group", init_process_group)
    monkeypatch.setattr(
        torch.distributed, "is_initialized", lambda: state["initialized"]
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda: calls.append("destroy"),
    )
    monkeypatch.setattr(
        runner_module,
        "_abort_distributed_process_group",
        lambda: calls.append("abort"),
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise ValueError("rank-local contract failure")

    monkeypatch.setattr(runner_module, "_run_initialized", fail)

    with pytest.raises(ValueError, match="rank-local contract failure"):
        runner_module.run_representation_training("/unused/config.toml")

    assert calls == ["abort"]


def test_validation_boundary_always_reshards_before_optimizer_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    binding = SimpleNamespace(
        reshard_owned_parameters=lambda: calls.append("reshard"),
        assert_optimizer_ownership=lambda _optimizer: calls.append("audit"),
    )
    optimizer = object()
    monkeypatch.setattr(
        runner_module,
        "_evaluate_validation",
        lambda **_kwargs: calls.append("validation") or "metrics",
    )

    result = runner_module._evaluate_validation_at_sharded_optimizer_boundary(
        binding=binding
    )
    binding.assert_optimizer_ownership(optimizer)

    assert result == "metrics"
    assert calls == ["validation", "reshard", "audit"]


def test_validation_boundary_reshards_when_validation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    binding = SimpleNamespace(reshard_owned_parameters=lambda: calls.append("reshard"))

    def fail(**_kwargs: object) -> None:
        calls.append("validation")
        raise ValueError("validation failed")

    monkeypatch.setattr(runner_module, "_evaluate_validation", fail)

    with pytest.raises(ValueError, match="validation failed"):
        runner_module._evaluate_validation_at_sharded_optimizer_boundary(
            binding=binding
        )

    assert calls == ["validation", "reshard"]


def test_invocation_stop_preserves_planned_scheduler_horizon() -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(
            target_optimizer_steps=2000,
            log_every_optimizer_steps=1,
        )
    )

    runner_module._validate_invocation_stop(config, 1)
    assert (
        runner_module._invocation_target_step(
            config,
            initial_global_step=0,
            stop_after_global_step=1,
        )
        == 1
    )
    assert (
        runner_module._invocation_target_step(
            config,
            initial_global_step=1,
            stop_after_global_step=None,
        )
        == 2000
    )


@pytest.mark.parametrize("value", (True, 0, -1, 2001, 1.5))
def test_invocation_stop_rejects_invalid_or_out_of_plan_steps(value: object) -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(
            target_optimizer_steps=2000,
            log_every_optimizer_steps=1,
        )
    )

    with pytest.raises(ValueError, match="stop_after_global_step"):
        runner_module._validate_invocation_stop(config, value)  # type: ignore[arg-type]


def test_invocation_target_must_advance_restored_state() -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(
            target_optimizer_steps=2,
            log_every_optimizer_steps=1,
        )
    )

    with pytest.raises(ValueError, match="beyond the restored global step"):
        runner_module._invocation_target_step(
            config,
            initial_global_step=1,
            stop_after_global_step=1,
        )


def test_terminal_checkpoint_can_run_zero_step_closeout() -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(
            target_optimizer_steps=2000,
            log_every_optimizer_steps=10,
        ),
        resume=SimpleNamespace(enabled=True),
    )

    assert (
        runner_module._invocation_target_step(
            config,
            initial_global_step=2000,
            stop_after_global_step=None,
        )
        == 2000
    )


def test_terminal_checkpoint_cursor_requires_exactly_one_pending_validation() -> None:
    assert (
        runner_module._pending_terminal_validation_count(
            global_step=2000,
            validation_every_optimizer_steps=2000,
            next_validation_event_index=0,
        )
        == 1
    )
    assert (
        runner_module._pending_terminal_validation_count(
            global_step=2000,
            validation_every_optimizer_steps=2000,
            next_validation_event_index=1,
        )
        == 0
    )
    with pytest.raises(ValueError, match="validation cursor"):
        runner_module._pending_terminal_validation_count(
            global_step=2000,
            validation_every_optimizer_steps=500,
            next_validation_event_index=0,
        )


def test_train_step_telemetry_follows_existing_log_cadence_without_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Trainer:
        def __init__(self) -> None:
            self.global_step = 0
            self.sample_order: list[str] = []

        def train_step(self) -> SimpleNamespace:
            self.global_step += 1
            sample_id = f"sample-{self.global_step}"
            self.sample_order.append(sample_id)
            return SimpleNamespace(global_step=self.global_step, sample_id=sample_id)

    trainer = _Trainer()
    measured_at_steps: list[int] = []
    performance = object()

    def measure(
        step: object,
        *,
        device: torch.device,
        global_matrix_count: int,
    ) -> tuple[object, object]:
        assert callable(step)
        assert device == torch.device("cpu")
        assert global_matrix_count == 8
        measured_at_steps.append(trainer.global_step)
        return step(), performance

    monkeypatch.setattr(runner_module, "measure_distributed_train_step", measure)

    results = tuple(
        runner_module._run_train_step_with_periodic_telemetry(
            trainer,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            global_matrix_count=8,
            log_every_optimizer_steps=10,
        )
        for _ in range(10)
    )

    assert measured_at_steps == [9]
    assert [result.global_step for result, _telemetry in results] == list(range(1, 11))
    assert trainer.sample_order == [f"sample-{step}" for step in range(1, 11)]
    assert all(telemetry is None for _result, telemetry in results[:9])
    assert results[9][1] is performance


def test_log_every_one_preserves_exact_per_step_performance_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = SimpleNamespace(global_step=0)
    train_calls: list[int] = []
    measure_calls: list[int] = []

    def train_step() -> SimpleNamespace:
        trainer.global_step += 1
        train_calls.append(trainer.global_step)
        return SimpleNamespace(global_step=trainer.global_step)

    trainer.train_step = train_step

    def measure(
        step: object,
        *,
        device: torch.device,
        global_matrix_count: int,
    ) -> tuple[object, object]:
        assert callable(step)
        assert device == torch.device("cpu")
        assert global_matrix_count == 4
        measure_calls.append(trainer.global_step)
        return step(), object()

    monkeypatch.setattr(runner_module, "measure_distributed_train_step", measure)

    first = runner_module._run_train_step_with_periodic_telemetry(
        trainer,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        global_matrix_count=4,
        log_every_optimizer_steps=1,
    )
    second = runner_module._run_train_step_with_periodic_telemetry(
        trainer,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        global_matrix_count=4,
        log_every_optimizer_steps=1,
    )

    assert train_calls == [1, 2]
    assert measure_calls == [0, 1]
    assert first[1] is not None
    assert second[1] is not None


def test_checkpoint_path_and_step_have_one_strict_ascii_round_trip(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        checkpoint=SimpleNamespace(
            directory=tmp_path,
            filename_prefix="representation",
        )
    )

    path = runner_module._checkpoint_path(config, 7)
    assert path == tmp_path / "representation-step-00000007"
    assert runner_module._checkpoint_step(path, "representation") == 7
    maximum = runner_module._checkpoint_path(config, 99_999_999)
    assert runner_module._checkpoint_step(maximum, "representation") == 99_999_999

    for invalid_step in (True, 0, -1, 100_000_000, 1.5):
        with pytest.raises(ValueError, match=r"\[1, 99999999\]"):
            runner_module._checkpoint_path(config, invalid_step)  # type: ignore[arg-type]
    invalid_names = (
        "other-step-00000001",
        "representation-step-00000000",
        "representation-step-0000001",
        "representation-step-000000001",
        "representation-step-٠٠٠٠٠٠٠١",
        "representation-step-0000000x",
    )
    for name in invalid_names:
        with pytest.raises(ValueError):
            runner_module._checkpoint_step(tmp_path / name, "representation")


def test_metric_jsonl_append_is_strict_incremental_and_rejects_nan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    runner_module._append_metric(
        path,
        {
            "z": (1, 2),
            "path": Path("/tmp/artifact"),
            "text": "目标证据",
            "loss": 1.25,
        },
    )
    runner_module._append_metric(path, {"event": "second", "step": 2})

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2
    assert json.loads(raw_lines[0]) == {
        "loss": 1.25,
        "path": "/tmp/artifact",
        "text": "目标证据",
        "z": [1, 2],
    }
    assert json.loads(raw_lines[1]) == {"event": "second", "step": 2}
    before = path.read_bytes()

    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="JSON compliant"):
            runner_module._append_metric(path, {"loss": invalid})
        assert path.read_bytes() == before
    with pytest.raises(TypeError, match="not JSON serializable"):
        runner_module._append_metric(path, {"invalid": object()})
    assert path.read_bytes() == before


def test_collective_metric_append_propagates_rank_zero_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, *, src: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_append_metric",
        lambda _path, _payload: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(RuntimeError, match="disk full"):
        runner_module._append_metric_rank_zero_collective(
            tmp_path / "metrics.jsonl",
            {"event": "train"},
        )


def test_training_metric_gathers_and_logs_versioned_qwen_physical_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(gathered: list[object], local: object) -> None:
        gathered[:] = [local, (32, 32)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    all_qwen_forward_batch_sizes = runner_module._gather_positive_int_tuples((32, 32))
    metrics = RepresentationStepMetrics(
        global_step=1,
        global_matrix_ce_loss=1.0,
        global_l_gen_loss=2.0,
        global_norm_loss=3.0,
        global_weighted_norm_loss=0.3,
        global_total_loss=3.3,
        global_row_count=32,
        global_sample_count=32,
        gradient_norm_before_clip=0.5,
        learning_rate=1e-4,
        local_sample_ids=tuple(f"sample-{index}" for index in range(16)),
        local_qwen_forward_batch_sizes=(32, 32),
    )
    rank_resources = tuple(
        RepresentationRankTrainStepResources(
            rank=rank,
            elapsed_ns=1_000_000_000 + rank,
            starting_allocated_bytes=100,
            starting_reserved_bytes=200,
            peak_allocated_bytes=150,
            peak_reserved_bytes=250,
            ending_allocated_bytes=110,
            ending_reserved_bytes=210,
        )
        for rank in range(2)
    )
    performance = RepresentationTrainStepPerformance(
        global_step=1,
        global_row_count=32,
        global_matrix_count=8,
        ranks=rank_resources,
    )
    captured: dict[str, object] = {}

    def capture(path: Path, payload: Mapping[str, object]) -> None:
        captured["path"] = path
        captured["payload"] = dict(payload)

    monkeypatch.setattr(
        runner_module,
        "_append_metric_rank_zero_collective",
        capture,
    )
    output_path = tmp_path / "metrics.jsonl"

    runner_module._log_training_metric(
        SimpleNamespace(output=SimpleNamespace(metrics_jsonl_path=output_path)),
        metrics=metrics,
        all_sample_ids=(("rank-0-sample",), ("rank-1-sample",)),
        all_qwen_forward_batch_sizes=all_qwen_forward_batch_sizes,
        run_identity=SimpleNamespace(identity_sha256="a" * 64),
        performance=performance,
    )

    assert captured["path"] == output_path
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert "local_sample_ids" not in payload
    assert "local_qwen_forward_batch_sizes" not in payload
    performance_payload = payload["performance"]
    assert isinstance(performance_payload, dict)
    assert performance_payload["qwen_physical_execution"] == {
        "schema_version": "representation-qwen-physical-execution-v1",
        "forward_batch_sizes_by_rank": [[32, 32], [32, 32]],
        "forward_call_count_by_rank": [2, 2],
        "cell_evaluation_count_by_rank": [64, 64],
        "max_forward_batch_size_by_rank": [32, 32],
        "global_forward_call_count": 4,
        "global_cell_evaluation_count": 128,
    }


def _write_metrics(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _resume_metrics_records() -> list[dict[str, object]]:
    identity = "a" * 64
    return [
        {
            "event": "start",
            "schema_version": runner_module.REPRESENTATION_RUNNER_SCHEMA_VERSION,
            "run_id": "representation-smoke",
            "run_identity_sha256": identity,
            "initial_global_step": 0,
        },
        {
            "event": "train",
            "run_identity_sha256": identity,
            "global_step": 1,
        },
        {
            "event": "validation",
            "run_identity_sha256": identity,
            "global_step": 1,
        },
        {
            "event": "train",
            "run_identity_sha256": identity,
            "global_step": 2,
        },
    ]


def test_resume_metrics_history_requires_exact_unfinished_checkpoint_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    records = _resume_metrics_records()
    _write_metrics(path, records)

    loaded = runner_module._validate_resume_metrics_history(
        path,
        run_id="representation-smoke",
        run_identity_sha256="a" * 64,
        checkpoint_global_step=2,
    )

    assert list(loaded) == records


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("identity", "changes run identity"),
        ("complete", "cannot be resumed"),
        ("advanced", "advanced beyond"),
        ("missing_checkpoint_train", "exactly one train event"),
        ("decreasing", "non-decreasing"),
    ),
)
def test_resume_metrics_history_rejects_drift_and_branching(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path = tmp_path / "metrics.jsonl"
    records = _resume_metrics_records()
    if mutation == "identity":
        records[2]["run_identity_sha256"] = "b" * 64
    elif mutation == "complete":
        records.append(
            {
                "event": "complete",
                "run_identity_sha256": "a" * 64,
                "global_step": 2,
            }
        )
    elif mutation == "advanced":
        records[-1]["global_step"] = 3
    elif mutation == "missing_checkpoint_train":
        records.pop()
    elif mutation == "decreasing":
        records[1]["global_step"] = 2
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    _write_metrics(path, records)

    with pytest.raises((TypeError, ValueError), match=message):
        runner_module._validate_resume_metrics_history(
            path,
            run_id="representation-smoke",
            run_identity_sha256="a" * 64,
            checkpoint_global_step=2,
        )


def test_resume_metrics_history_rejects_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    records = _resume_metrics_records()
    _write_metrics(path, records)
    path.write_bytes(path.read_bytes().removesuffix(b"\n"))

    with pytest.raises(ValueError, match="end with a newline"):
        runner_module._validate_resume_metrics_history(
            path,
            run_id="representation-smoke",
            run_identity_sha256="a" * 64,
            checkpoint_global_step=2,
        )


def test_checkpoint_retention_removes_only_current_invocation_paths(
    tmp_path: Path,
) -> None:
    preexisting = tmp_path / "representation-step-00000001"
    retired = tmp_path / "representation-step-00000002"
    current = tmp_path / "representation-step-00000003"
    for path in (preexisting, retired, current):
        path.mkdir()
    config = SimpleNamespace(
        checkpoint=SimpleNamespace(
            directory=tmp_path,
            filename_prefix="representation",
        )
    )

    runner_module._remove_created_checkpoints_rank_zero(
        config,
        paths=(retired,),
        current=current,
    )

    assert preexisting.is_dir()
    assert not retired.exists()
    assert current.is_dir()


def test_final_export_can_strictly_reuse_resume_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "adapter.pt"
    path.touch()
    run_identity = object()
    manifest = SimpleNamespace(run_identity=run_identity, digest="same")
    export = SimpleNamespace(is_writer=True, manifest=manifest)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, *, src: None,
    )
    monkeypatch.setattr(
        runner_module,
        "load_rank_zero_adapter_owned_state_export",
        lambda candidate, *, expected_run_identity: (
            SimpleNamespace(manifest=manifest)
            if candidate == path and expected_run_identity is run_identity
            else (_ for _ in ()).throw(AssertionError("wrong export identity"))
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "save_rank_zero_adapter_owned_state_export_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identical resume artifact must not be overwritten")
        ),
    )

    assert (
        runner_module._save_rank_zero_export_collective(
            path,
            export,
            allow_existing_identical=True,
        )
        == "reused"
    )


def test_current_process_seed_never_calls_cuda_manual_seed_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = torch.random.get_rng_state()
    previous_python = random.getstate()
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: calls.append(seed))
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda _seed: (_ for _ in ()).throw(
            AssertionError("manual_seed_all touches peer visible devices")
        ),
    )
    try:
        runner_module._seed_current_process(731)
        assert torch.initial_seed() == 731
        assert random.random() == random.Random(731).random()
        assert calls == [731]
    finally:
        torch.random.set_rng_state(previous)
        random.setstate(previous_python)


def test_qwen_serial_loader_broadcasts_active_rank_failure_without_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_transformers = ModuleType("transformers")

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            return object()

    class ModelLoader:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> nn.Module:
            raise RuntimeError("synthetic loader failure")

    fake_transformers.AutoProcessor = ProcessorLoader  # type: ignore[attr-defined]
    fake_transformers.AutoModelForImageTextToText = ModelLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(values: list[object], value: object) -> None:
        values[:] = [value, value]

    sources: list[int] = []
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda _values, *, src: sources.append(src),
    )
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda: (_ for _ in ()).throw(AssertionError("barrier can deadlock")),
    )
    config = SimpleNamespace(
        model=SimpleNamespace(
            local_path="/synthetic/qwen",
            local_files_only=True,
            trust_remote_code=False,
            dtype="float32",
            attention_backend="sdpa",
        ),
        fsdp2=SimpleNamespace(world_size=2),
    )

    with pytest.raises(RuntimeError, match="rank 0.*synthetic loader failure"):
        runner_module._load_qwen3(config, device=torch.device("cpu"), rank=0)
    assert sources == [0]


def test_representation_dtype_mapping_has_no_implicit_fallback() -> None:
    assert runner_module._torch_dtype("bfloat16") is torch.bfloat16
    assert runner_module._torch_dtype("float32") is torch.float32
    for name in ("float16", "bf16", "auto", ""):
        with pytest.raises(ValueError, match="unsupported representation dtype"):
            runner_module._torch_dtype(name)


def test_importing_cli_does_not_import_representation_runner() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")
    script = (
        "import sys\n"
        "import tgvf_rl.cli\n"
        "name = 'tgvf_rl.representation.training.runner'\n"
        "raise SystemExit(1 if name in sys.modules else 0)\n"
    )

    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=repository_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_cli_run_command_lazily_dispatches_and_prints_rank_zero_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[Path, int | None]] = []
    events: list[str] = []
    fake_runner = ModuleType("tgvf_rl.representation.training.runner")

    def fake_run(
        path: Path,
        *,
        stop_after_global_step: int | None,
    ) -> dict[str, object]:
        calls.append((path, stop_after_global_step))
        return {"status": "synthetic", "gpu_work_launched": False}

    fake_runner.run_representation_training = fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "tgvf_rl.representation.training.runner",
        fake_runner,
    )
    config_path = Path("/unused/representation.toml")
    binding = SimpleNamespace(
        source_path=config_path,
        authorization_parameters=lambda: {"canonical_config_sha256": "b" * 64},
    )
    python_identity = SimpleNamespace(
        authorization_parameters=lambda: {"python_executable_sha256": "d" * 64}
    )
    config = SimpleNamespace(
        run_id="REPRESENTATION-CLI-TEST",
        canonical_config_sha256="a" * 64,
        source_toml_sha256="b" * 64,
        source_path=config_path,
        fsdp2=SimpleNamespace(world_size=2),
    )
    identity = cli.CLIExecutionAuthorizationIdentity.create(
        run_id=config.run_id,
        phase=cli.REPRESENTATION_TRAINING_PHASE,
        command_id=cli._REPRESENTATION_COMMAND_ID,
        run_identity_sha256=config.canonical_config_sha256,
    )
    monkeypatch.setenv("TGVF_CLI_GATE_DIRECTORY", "/gate")
    monkeypatch.setenv(
        "TGVF_CLI_CONSUMPTION_RECEIPT_PATH",
        "/gate/consumptions/token.json",
    )
    monkeypatch.setenv("TGVF_CLI_CONSUMPTION_RECEIPT_SHA256", "c" * 64)
    monkeypatch.setenv(
        "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH",
        "/gate/cli-launches/token/launcher-liveness.json",
    )
    monkeypatch.setattr(
        cli,
        "verify_cli_worker_authorization_from_environment",
        lambda **_kwargs: events.append("authorize") or identity,
    )
    monkeypatch.setattr(
        cli,
        "assert_canonical_runtime_launch_enabled",
        lambda: events.append("runtime-closure"),
    )
    monkeypatch.setattr(
        cli,
        "bind_canonical_config_path",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        cli,
        "load_representation_training_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        cli,
        "assert_loaded_config_matches_binding",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli,
        "bind_current_python_executable",
        lambda _path: python_identity,
    )
    monkeypatch.setattr(
        cli,
        "_assert_worker_identity_parameters",
        lambda observed, _expected: (
            events.append("identity") if observed is identity else None
        ),
    )

    assert (
        cli.main(
            [
                "run-representation",
                "/unused/representation.toml",
                "--stop-after-global-step",
                "7",
                "--launcher-python-executable",
                "/audited/python",
                "--gate-directory",
                "/gate",
                "--launch-consumption-receipt",
                "/gate/consumptions/token.json",
                "--launch-consumption-sha256",
                "c" * 64,
                "--launcher-liveness-receipt",
                "/gate/cli-launches/token/launcher-liveness.json",
            ]
        )
        == 0
    )
    assert calls == [(Path("/unused/representation.toml"), 7)]
    assert events == ["authorize", "runtime-closure", "identity"]
    assert json.loads(capsys.readouterr().out) == {
        "gpu_work_launched": False,
        "status": "synthetic",
    }


def test_cli_resume_comparator_lazily_dispatches_all_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Path]] = []
    fake_module = ModuleType("tgvf_rl.representation.training.resume_parity")

    # ``asdict`` requires a real dataclass instance.
    @dataclass(frozen=True)
    class Result:
        exact: bool = True

    fake_module.compare_representation_resume_lanes = (  # type: ignore[attr-defined]
        lambda **kwargs: calls.append(kwargs) or Result()
    )
    monkeypatch.setitem(
        sys.modules,
        "tgvf_rl.representation.training.resume_parity",
        fake_module,
    )
    arguments = [
        "compare-representation-resume",
        "--continuous-artifact",
        "/a.pt",
        "--resumed-artifact",
        "/b.pt",
        "--continuous-checkpoint",
        "/a-dcp",
        "--resumed-checkpoint",
        "/b-dcp",
        "--continuous-metrics",
        "/a.jsonl",
        "--resumed-metrics",
        "/b.jsonl",
    ]

    assert cli.main(arguments) == 0
    assert calls == [
        {
            "continuous_artifact_path": Path("/a.pt"),
            "resumed_artifact_path": Path("/b.pt"),
            "continuous_checkpoint_path": Path("/a-dcp"),
            "resumed_checkpoint_path": Path("/b-dcp"),
            "continuous_metrics_path": Path("/a.jsonl"),
            "resumed_metrics_path": Path("/b.jsonl"),
        }
    ]
    assert json.loads(capsys.readouterr().out) == {"exact": True}
