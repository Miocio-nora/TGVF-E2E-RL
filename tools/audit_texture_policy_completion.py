#!/usr/bin/env python3
"""Fail-closed completion audit for a config-bound texture policy suite."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, BinaryIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_benchmark_scoring import (  # noqa: E402
    load_policy_evaluation_identity,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    POLICY_BENCHMARK_SCHEMA,
    PolicyCoreDevConfig,
    load_bound_policy_benchmark_tasks,
    load_policy_benchmark_results,
    load_policy_coredev_config,
    policy_benchmark_task_path,
)
from tgvf_rl.evaluation.texture_bench.io import (  # noqa: E402
    write_json_idempotent,
)
from tgvf_rl.evaluation.texture_bench.schema import (  # noqa: E402
    canonical_json_sha256,
    file_sha256,
)


TEXTURE_POLICY_COMPLETION_AUDIT_SCHEMA = "tgvf-texture-policy-completion-audit-v1"
TEXTURE_SUITE_TASK_COUNT = 42_870
TEXTURE_POLICY_WORLD_SIZE = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact completion of one config-bound texture policy benchmark. "
            "This command is read-only unless --output is supplied."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=TEXTURE_POLICY_WORLD_SIZE)
    parser.add_argument(
        "--expected-task-count",
        type=int,
        help="Expected count; defaults to the immutable config value.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _regular_file(path: Path, *, owner: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{owner} is not a regular file: {path}")
    return resolved


def _require_mapping(value: object, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} is malformed")
    return value


def _require_equal(observed: object, expected: object, *, owner: str) -> None:
    if observed != expected:
        raise RuntimeError(f"{owner} differs")


def _validate_config_identity_binding(
    *,
    config: PolicyCoreDevConfig,
    config_path: Path,
    identity: Mapping[str, Any],
    tasks_path: Path,
    world_size: int,
    expected_task_count: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Reject config/task/evaluation drift before reading inference rows."""

    if config.schema_version != POLICY_BENCHMARK_SCHEMA:
        raise ValueError("texture completion audit requires generic benchmark schema")
    if world_size != TEXTURE_POLICY_WORLD_SIZE:
        raise ValueError("texture completion audit requires world-size=4")
    if len(config.gpu_ids) != world_size:
        raise ValueError("config GPU count differs from requested world size")
    if config.expected_task_count != expected_task_count:
        raise ValueError(
            f"texture completion audit requires exactly {expected_task_count} tasks"
        )
    if config.expected_single_image_count != expected_task_count:
        raise ValueError(
            "texture completion audit requires every task to be single-image"
        )

    _require_equal(
        identity.get("evaluation_id"), config.evaluation_id, owner="evaluation ID"
    )
    _require_equal(
        identity.get("evaluation_schema_version"),
        config.schema_version,
        owner="evaluation schema binding",
    )
    policy_config_path = _regular_file(
        config.policy_config_path, owner="bound policy run config"
    )
    _require_equal(
        identity.get("policy_config_path"),
        str(policy_config_path),
        owner="policy config path binding",
    )
    _require_equal(
        identity.get("policy_config_file_sha256"),
        file_sha256(policy_config_path),
        owner="policy config file binding",
    )

    task_binding = _require_mapping(
        identity.get("task_manifest"), owner="evaluation task binding"
    )
    expected_task_binding = {
        "path": str(tasks_path),
        "sha256": config.task_manifest_sha256,
        "task_count": expected_task_count,
        "single_image_count": expected_task_count,
    }
    _require_equal(
        dict(task_binding), expected_task_binding, owner="evaluation task binding"
    )
    _require_equal(
        file_sha256(tasks_path),
        config.task_manifest_sha256,
        owner="bound task manifest bytes",
    )

    execution = _require_mapping(
        identity.get("execution"), owner="evaluation execution binding"
    )
    expected_execution = {
        "world_size": world_size,
        "gpu_ids": list(config.gpu_ids),
        "max_model_len": config.max_model_len,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "enable_chunked_prefill": config.enable_chunked_prefill,
        "inference_concurrency_per_gpu": config.inference_concurrency_per_gpu,
    }
    for field, expected in expected_execution.items():
        _require_equal(
            execution.get(field), expected, owner=f"execution {field} binding"
        )
    image_max_pixels = execution.get("image_max_pixels")
    if type(image_max_pixels) is not int or image_max_pixels <= 0:
        raise ValueError("evaluation image_max_pixels binding is malformed")
    if config.image_max_pixels is not None:
        _require_equal(
            image_max_pixels,
            config.image_max_pixels,
            owner="execution image_max_pixels binding",
        )

    policy_snapshot = _require_mapping(
        identity.get("policy_snapshot"), owner="evaluation policy snapshot"
    )
    expected_snapshot_fields = {
        "run_id": config.expected_policy_run_id,
        "run_identity_sha256": config.expected_policy_run_identity_sha256,
        "optimizer_step": config.expected_optimizer_step,
        "weights_sha256": config.expected_policy_weights_sha256,
    }
    for field, expected in expected_snapshot_fields.items():
        _require_equal(
            policy_snapshot.get(field), expected, owner=f"policy snapshot {field}"
        )
    _require_equal(
        identity.get("policy_run_config_identity_sha256"),
        config.expected_policy_run_identity_sha256,
        owner="policy run config identity",
    )

    # Resolve and hash the caller-selected config only after all semantic checks.
    _regular_file(config_path, owner="policy benchmark config")
    return task_binding, execution, policy_snapshot


def _locked_rank_files(
    inference_root: Path, *, world_size: int, stack: ExitStack
) -> tuple[tuple[Path, BinaryIO], ...]:
    if not inference_root.is_dir():
        raise FileNotFoundError(
            f"missing policy benchmark inference root: {inference_root}"
        )
    expected = {f"rank-{rank}.jsonl" for rank in range(world_size)}
    observed = {
        path.name for path in inference_root.glob("rank-*.jsonl") if path.exists()
    }
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            "policy benchmark rank file set differs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    result: list[tuple[Path, BinaryIO]] = []
    for rank in range(world_size):
        path = inference_root / f"rank-{rank}.jsonl"
        _regular_file(path, owner=f"rank {rank} result")
        handle = stack.enter_context(path.open("rb"))
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        stack.callback(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
        result.append((path.resolve(), handle))
    return tuple(result)


def _rank_artifacts(
    locked_files: Sequence[tuple[Path, BinaryIO]],
    *,
    records: Mapping[int, Mapping[str, object]],
    expected_ordinals_by_rank: Sequence[set[int]],
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for rank, (path, handle) in enumerate(locked_files):
        handle.seek(0)
        payload = handle.read()
        observed_ordinals = {
            ordinal for ordinal, row in records.items() if row.get("rank") == rank
        }
        _require_equal(
            observed_ordinals,
            expected_ordinals_by_rank[rank],
            owner=f"rank {rank} ordinal coverage",
        )
        artifacts.append(
            {
                "rank": rank,
                "path": str(path),
                "row_count": len(observed_ordinals),
                "expected_row_count": len(expected_ordinals_by_rank[rank]),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return artifacts


def _nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _summarize_result_rows(
    rows: Sequence[tuple[int, Mapping[str, object]]],
) -> dict[str, object]:
    """Summarize one non-empty row tranche without adding nested tranches."""

    if not rows:
        raise ValueError("policy result summary tranche must be non-empty")
    stop_counts: Counter[str] = Counter()
    result_kind_counts: Counter[str] = Counter()
    tool_call_counts: Counter[str] = Counter()
    tool_call_count_histogram: Counter[int] = Counter()
    tool_error_counts: Counter[str] = Counter()
    assistant_turn_count_histogram: Counter[int] = Counter()
    wall_seconds: list[float] = []
    rows_with_tool_calls = 0
    rows_with_tool_errors = 0
    recoverable_tool_errors = 0
    assistant_turn_total = 0

    for ordinal, row in rows:
        stop = row.get("stop")
        if not isinstance(stop, str) or not stop:
            raise RuntimeError(f"policy result stop is malformed at ordinal {ordinal}")
        stop_counts[stop] += 1
        result_kind = row.get("result_kind", "trajectory")
        if not isinstance(result_kind, str) or not result_kind:
            raise RuntimeError(f"policy result kind is malformed at ordinal {ordinal}")
        result_kind_counts[result_kind] += 1

        calls = row.get("tool_calls")
        if not isinstance(calls, list):
            raise RuntimeError(f"policy tool calls are malformed at ordinal {ordinal}")
        tool_call_count_histogram[len(calls)] += 1
        rows_with_tool_calls += bool(calls)
        for call in calls:
            if not isinstance(call, Mapping):
                raise RuntimeError(
                    f"policy tool call is malformed at ordinal {ordinal}"
                )
            function_name = call.get("function_name")
            if not isinstance(function_name, str) or not function_name:
                raise RuntimeError(
                    f"policy tool call function is malformed at ordinal {ordinal}"
                )
            tool_call_counts[function_name] += 1

        errors = row.get("tool_errors")
        if not isinstance(errors, list):
            raise RuntimeError(f"policy tool errors are malformed at ordinal {ordinal}")
        rows_with_tool_errors += bool(errors)
        for error in errors:
            if not isinstance(error, Mapping):
                raise RuntimeError(
                    f"policy tool error is malformed at ordinal {ordinal}"
                )
            code = error.get("code")
            if not isinstance(code, str) or not code:
                raise RuntimeError(
                    f"policy tool error code is malformed at ordinal {ordinal}"
                )
            recoverable = error.get("recoverable")
            if type(recoverable) is not bool:
                raise RuntimeError(
                    f"policy tool error recoverability is malformed at ordinal {ordinal}"
                )
            tool_error_counts[code] += 1
            recoverable_tool_errors += recoverable

        assistant_turns = row.get("assistant_turns")
        if not isinstance(assistant_turns, list):
            raise RuntimeError(
                f"policy assistant turns are malformed at ordinal {ordinal}"
            )
        assistant_turn_count = len(assistant_turns)
        assistant_turn_total += assistant_turn_count
        assistant_turn_count_histogram[assistant_turn_count] += 1

        elapsed = row.get("wall_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise RuntimeError(
                f"policy result wall_seconds is malformed at ordinal {ordinal}"
            )
        wall_seconds.append(float(elapsed))

    return {
        "row_count": len(rows),
        "result_kind": dict(sorted(result_kind_counts.items())),
        "stop": dict(sorted(stop_counts.items())),
        "assistant_turns": {
            "total": assistant_turn_total,
            "mean": assistant_turn_total / len(rows),
            "per_row_count_histogram": {
                str(count): row_count
                for count, row_count in sorted(assistant_turn_count_histogram.items())
            },
        },
        "tool_calls": {
            "total": sum(tool_call_counts.values()),
            "rows_with_calls": rows_with_tool_calls,
            "by_function_name": dict(sorted(tool_call_counts.items())),
            "per_row_count_histogram": {
                str(count): rows
                for count, rows in sorted(tool_call_count_histogram.items())
            },
        },
        "tool_errors": {
            "total": sum(tool_error_counts.values()),
            "rows_with_errors": rows_with_tool_errors,
            "recoverable": recoverable_tool_errors,
            "non_recoverable": sum(tool_error_counts.values())
            - recoverable_tool_errors,
            "by_code": dict(sorted(tool_error_counts.items())),
        },
        "runtime": {
            "row_count": len(wall_seconds),
            "wall_seconds_sum": math.fsum(wall_seconds),
            "wall_seconds_mean": math.fsum(wall_seconds) / len(wall_seconds),
            "wall_seconds_min": min(wall_seconds),
            "wall_seconds_max": max(wall_seconds),
            "wall_seconds_p50_nearest_rank": _nearest_rank_percentile(
                wall_seconds, 0.50
            ),
            "wall_seconds_p95_nearest_rank": _nearest_rank_percentile(
                wall_seconds, 0.95
            ),
        },
    }


def _result_summaries(
    records: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    ordered_rows = [(ordinal, records[ordinal]) for ordinal in sorted(records)]
    summary = _summarize_result_rows(ordered_rows)
    rows_by_dataset: dict[str, list[tuple[int, Mapping[str, object]]]] = {}
    for ordinal, row in ordered_rows:
        dataset = row.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise RuntimeError(
                f"policy result dataset is malformed at ordinal {ordinal}"
            )
        rows_by_dataset.setdefault(dataset, []).append((ordinal, row))
    summary["by_dataset"] = {
        dataset: _summarize_result_rows(rows_by_dataset[dataset])
        for dataset in sorted(rows_by_dataset)
    }
    return summary


def audit_texture_policy_completion(
    config_path: str | Path,
    *,
    world_size: int = TEXTURE_POLICY_WORLD_SIZE,
    expected_task_count: int = TEXTURE_SUITE_TASK_COUNT,
) -> dict[str, Any]:
    """Return a deterministic, provenance-bound report or fail closed."""

    requested_config_path = Path(config_path)
    resolved_config_path = _regular_file(
        requested_config_path, owner="policy benchmark config"
    )
    config = load_policy_coredev_config(resolved_config_path)
    tasks_path = _regular_file(
        policy_benchmark_task_path(config), owner="bound task manifest"
    )
    identity_path = _regular_file(
        config.output_root / "runtime/evaluation-identity.json",
        owner="evaluation identity",
    )
    identity_file_sha256 = file_sha256(identity_path)
    identity, observed_identity_file_sha256 = load_policy_evaluation_identity(
        identity_path, expected_file_sha256=identity_file_sha256
    )
    task_binding, execution, policy_snapshot = _validate_config_identity_binding(
        config=config,
        config_path=resolved_config_path,
        identity=identity,
        tasks_path=tasks_path,
        world_size=world_size,
        expected_task_count=expected_task_count,
    )
    tasks = load_bound_policy_benchmark_tasks(config)
    if len(tasks) != expected_task_count or any(
        not task.single_image for task in tasks
    ):
        raise RuntimeError("bound texture task population differs")

    expected_ordinals = {task.ordinal for task in tasks}
    expected_ordinals_by_rank = [
        {ordinal for ordinal in expected_ordinals if ordinal % world_size == rank}
        for rank in range(world_size)
    ]
    inference_root = (config.output_root / "inference").resolve()
    with ExitStack() as stack:
        locked_files = _locked_rank_files(
            inference_root, world_size=world_size, stack=stack
        )
        records = load_policy_benchmark_results(
            inference_root,
            tasks=tasks,
            evaluation_identity=identity,
            require_complete=True,
        )
        _require_equal(
            set(records), expected_ordinals, owner="complete task ordinal coverage"
        )
        rank_artifacts = _rank_artifacts(
            locked_files,
            records=records,
            expected_ordinals_by_rank=expected_ordinals_by_rank,
        )

    content: dict[str, Any] = {
        "schema_version": TEXTURE_POLICY_COMPLETION_AUDIT_SCHEMA,
        "complete": True,
        "assignment": "ordinal_mod_world_size",
        "evaluation": {
            "evaluation_id": identity["evaluation_id"],
            "evaluation_identity_sha256": identity["identity_sha256"],
            "evaluation_identity_path": str(identity_path),
            "evaluation_identity_file_sha256": observed_identity_file_sha256,
            "evaluation_schema_version": identity["evaluation_schema_version"],
            "policy_snapshot": dict(policy_snapshot),
            "execution": dict(execution),
        },
        "task_manifest": {
            **dict(task_binding),
            "identity_sha256": canonical_json_sha256(dict(task_binding)),
        },
        "coverage": {
            "expected_task_count": expected_task_count,
            "observed_task_count": len(records),
            "missing_count": 0,
            "duplicate_count": 0,
            "first_ordinal": min(expected_ordinals),
            "last_ordinal": max(expected_ordinals),
        },
        "inputs": {
            "policy_benchmark_config_path": str(resolved_config_path),
            "policy_benchmark_config_file_sha256": file_sha256(resolved_config_path),
            "rank_results": rank_artifacts,
        },
        "summary": _result_summaries(records),
    }
    return {
        **content,
        "audit_identity_sha256": canonical_json_sha256(content),
    }


def materialize_texture_policy_completion_audit(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
    world_size: int = TEXTURE_POLICY_WORLD_SIZE,
    expected_task_count: int = TEXTURE_SUITE_TASK_COUNT,
) -> dict[str, Any]:
    """Audit and optionally create one immutable, idempotent JSON report."""

    report = audit_texture_policy_completion(
        config_path,
        world_size=world_size,
        expected_task_count=expected_task_count,
    )
    if output is not None:
        write_json_idempotent(Path(output).expanduser().resolve(), report)
    return report


def main() -> int:
    args = _parser().parse_args()
    config = load_policy_coredev_config(args.config.expanduser().resolve(strict=True))
    expected_task_count = (
        config.expected_task_count
        if args.expected_task_count is None
        else args.expected_task_count
    )
    report = materialize_texture_policy_completion_audit(
        args.config,
        output=args.output,
        world_size=args.world_size,
        expected_task_count=expected_task_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
