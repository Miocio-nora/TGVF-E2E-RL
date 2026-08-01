from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest


_TOOL_PATH = (
    Path(__file__).resolve().parents[4]
    / "tools"
    / "launch_representation_answer_utility_evaluation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "answer_utility_multi_worker_launcher", _TOOL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launcher)


def _preflight(
    *, shard_index: int, shard_count: int, ordinals: list[int], samples: int
) -> dict[str, object]:
    return {
        "status": "validated",
        "arms": ["image_correct_D", "image_same_target_wrong_image_D"],
        "selected_group_ordinals": ordinals,
        "selected_sample_count": samples,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }


def _plan(tmp_path: Path) -> dict[str, object]:
    output = tmp_path / "evaluation"
    assignments = []
    for index, sample_id in enumerate(("sample-a", "sample-b")):
        name = f"shard-{index:04d}-of-0002"
        assignments.append(
            {
                "shard_index": index,
                "shard_count": 2,
                "physical_gpu_id": index,
                "output_root": str(output / name),
                "log_path": str(output / "logs" / f"{name}.log"),
                "preflight": _preflight(
                    shard_index=index,
                    shard_count=2,
                    ordinals=[index],
                    samples=1,
                ),
                "command": [sys.executable, "-c", "pass"],
                "test_sample_id": sample_id,
            }
        )
    plan: dict[str, object] = {
        "schema_version": launcher.PLAN_SCHEMA_VERSION,
        "output_root": str(output),
        "physical_gpu_ids": [0, 1],
        "workers_per_gpu": 1,
        "thread_environment": launcher.THREAD_ENVIRONMENT,
        "whole_preflight": _preflight(
            shard_index=0, shard_count=1, ordinals=[0, 1], samples=2
        ),
        "assignments": assignments,
    }
    plan["plan_sha256"] = launcher._canonical_sha256(plan)
    return plan


def _write_complete_shard(assignment: dict[str, object]) -> None:
    root = Path(assignment["output_root"])
    root.mkdir(parents=True)
    arms = assignment["preflight"]["arms"]
    sample_id = assignment["test_sample_id"]
    identity = {
        "arms": arms,
        "ordered_selected_samples": [{"sample_id": sample_id}],
        "shard_index": assignment["shard_index"],
        "shard_count": assignment["shard_count"],
    }
    identity_sha = launcher._canonical_sha256(identity)
    (root / "identity.json").write_text(
        json.dumps({"identity": identity, "identity_sha256": identity_sha}),
        encoding="utf-8",
    )
    records = [
        {
            "sample_id": sample_id,
            "arm": arm,
            "run_identity_sha256": identity_sha,
        }
        for arm in arms
    ]
    records_payload = b"".join(
        json.dumps(record, sort_keys=True).encode() + b"\n" for record in records
    )
    (root / "records.jsonl").write_bytes(records_payload)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "run_identity_sha256": identity_sha,
                "sample_count": 1,
                "record_count": len(records),
                "records_jsonl_sha256": sha256(records_payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_parser_and_worker_commands_assign_two_workers_per_gpu(tmp_path: Path) -> None:
    source = tmp_path / "source.toml"
    parsed = launcher._parser().parse_args(
        [
            "--production-source",
            "--source-evaluation-config",
            str(source),
            "--output-root",
            str(tmp_path / "output"),
            "--physical-gpu-id",
            "2",
            "--physical-gpu-id",
            "5",
            "--workers-per-gpu",
            "2",
            "--arm",
            "image_correct_D",
        ]
    )
    resolved = launcher._resolve_inputs(parsed)
    common = launcher._common_evaluator_args(parsed, resolved)

    commands = [
        launcher._worker_command(
            common,
            output_root=resolved["output_root"] / f"shard-{index}",
            gpu_id=(2, 5)[index % 2],
            shard_index=index,
            shard_count=4,
        )
        for index in range(4)
    ]

    assert [
        command[command.index("--physical-gpu-id") + 1] for command in commands
    ] == [
        "2",
        "5",
        "2",
        "5",
    ]
    assert [command[command.index("--shard-index") + 1] for command in commands] == [
        "0",
        "1",
        "2",
        "3",
    ]
    assert all(
        command[command.index("--shard-count") + 1] == "4" for command in commands
    )
    assert launcher._worker_environment(
        2
    ) | launcher.THREAD_ENVIRONMENT == launcher._worker_environment(2)
    assert launcher._worker_environment(2)["CUDA_VISIBLE_DEVICES"] == "2"


def test_preflight_partition_rejects_duplicate_or_missing_groups() -> None:
    whole = _preflight(shard_index=0, shard_count=1, ordinals=[4, 7], samples=3)
    valid = [
        _preflight(shard_index=0, shard_count=2, ordinals=[4], samples=1),
        _preflight(shard_index=1, shard_count=2, ordinals=[7], samples=2),
    ]
    launcher._validate_preflight_partition(whole, valid)

    duplicate = [valid[0], {**valid[1], "selected_group_ordinals": [4]}]
    with pytest.raises(RuntimeError, match="repeated group ordinals"):
        launcher._validate_preflight_partition(whole, duplicate)


def test_output_plan_is_resumable_but_rejects_conflicts(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    output = Path(plan["output_root"])
    output.mkdir(parents=True)
    (output / ".launch.lock").touch()

    launcher._prepare_output(plan)
    launcher._prepare_output(plan)

    (output / "foreign.txt").write_text("not owned", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unowned entries"):
        launcher._prepare_output(plan)


def test_completed_shards_require_exact_record_union(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    for assignment in plan["assignments"]:
        _write_complete_shard(assignment)

    result = launcher._validate_completed_shards(plan)

    assert result["status"] == "complete"
    assert result["sample_count"] == 2
    assert result["record_count"] == 4

    first_root = Path(plan["assignments"][0]["output_root"])
    first_records = (
        (first_root / "records.jsonl").read_text(encoding="utf-8").splitlines()
    )
    shortened = (first_records[0] + "\n").encode()
    (first_root / "records.jsonl").write_bytes(shortened)
    summary = json.loads((first_root / "summary.json").read_text(encoding="utf-8"))
    summary["record_count"] = 1
    summary["records_jsonl_sha256"] = sha256(shortened).hexdigest()
    (first_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="record coverage mismatch"):
        launcher._validate_completed_shards(plan)


def test_completed_shards_reject_cross_shard_duplicate_samples(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["assignments"][1]["test_sample_id"] = "sample-a"
    for assignment in plan["assignments"]:
        _write_complete_shard(assignment)

    with pytest.raises(RuntimeError, match="duplicate expected keys"):
        launcher._validate_completed_shards(plan)


def test_worker_failure_terminates_and_reaps_peers(tmp_path: Path) -> None:
    failed_log = tmp_path / "failed.log"
    peer_log = tmp_path / "peer.log"
    plan = {
        "assignments": [
            {
                "command": [sys.executable, "-c", "raise SystemExit(7)"],
                "physical_gpu_id": 0,
                "log_path": str(failed_log),
            },
            {
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                "physical_gpu_id": 1,
                "log_path": str(peer_log),
            },
        ]
    }
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="returncode=7"):
        launcher._launch_and_wait(plan, poll_interval_seconds=0.01)
    assert time.monotonic() - started < 5


def test_gpu_ids_must_be_unique(tmp_path: Path) -> None:
    parsed = launcher._parser().parse_args(
        [
            "--production-source",
            "--source-evaluation-config",
            str(tmp_path / "source.toml"),
            "--output-root",
            str(tmp_path / "output"),
            "--physical-gpu-id",
            "3",
            "--physical-gpu-id",
            "3",
        ]
    )
    with pytest.raises(ValueError, match="must not be repeated"):
        launcher._resolve_inputs(parsed)


def test_group_limit_is_rejected_with_multiple_workers(tmp_path: Path) -> None:
    parsed = launcher._parser().parse_args(
        [
            "--production-source",
            "--source-evaluation-config",
            str(tmp_path / "source.toml"),
            "--output-root",
            str(tmp_path / "output"),
            "--physical-gpu-id",
            "0",
            "--physical-gpu-id",
            "1",
            "--group-limit",
            "4",
        ]
    )
    with pytest.raises(ValueError, match="applies the limit independently"):
        launcher._resolve_inputs(parsed)


def test_output_root_symlink_is_rejected_before_launch(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "output-link"
    link.symlink_to(target, target_is_directory=True)
    parsed = launcher._parser().parse_args(
        [
            "--production-source",
            "--source-evaluation-config",
            str(tmp_path / "source.toml"),
            "--output-root",
            str(link),
            "--physical-gpu-id",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        launcher._resolve_inputs(parsed)
    assert not (target / ".launch.lock").exists()
