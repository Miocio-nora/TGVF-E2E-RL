#!/usr/bin/env python3
"""Build the fail-closed PRL-06 BS32 versus PRL-05 BS16 result report.

This command intentionally does not wait for training or evaluation artifacts.
The companion ``wait-and-generate-result-report.sh`` owns that lifecycle.  Once
invoked, every PRL-06 input is required and any incomplete or malformed input is
an error rather than an implicit partial report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any
from zoneinfo import ZoneInfo


DATASETS = (
    "VStarBench",
    "HRBench4K",
    "BLINK",
    "OCRBench_v2",
    "MMMU_Pro_10c",
    "MathVista_MINI",
    "MathVerse_MINI",
)

# The ValKit outputs mix fraction-valued and percent-valued metrics.  Keep the
# conversion explicit so that an upstream display-name change cannot silently
# alter the macro.
PRIMARY_METRICS: dict[str, tuple[str, float]] = {
    "VStarBench": ("split=none|Overall", 100.0),
    "HRBench4K": ("type=all|accuracy", 100.0),
    "BLINK": ("split=none|Overall", 100.0),
    "OCRBench_v2": ("Chinese Overall Score", 100.0),
    "MMMU_Pro_10c": ("split=test|Overall", 100.0),
    "MathVista_MINI": ("Task&Skill=Overall|acc", 1.0),
    "MathVerse_MINI": ("split=Text Dominant|Overall", 1.0),
}

AUDITED_STEP0_PERCENT = {
    "VStarBench": 56.02094240837696,
    "HRBench4K": 52.0,
    "BLINK": 27.857142857142858,
    "OCRBench_v2": 41.976733736748606,
    "MMMU_Pro_10c": 42.66666666666667,
    "MathVista_MINI": 70.33333333333334,
    "MathVerse_MINI": 68.0,
}
AUDITED_STEP0_MACRO_PERCENT = 51.2649741431812

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_LINE = re.compile(r"\bstep:(\d+) - ")
NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
TRAIN_LOG_FIELDS = (
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/pg_clipfrac",
    "actor/behavior_current_log_ratio_abs_mean",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"required non-empty input is unavailable: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def finite_number(value: Any, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{owner} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{owner} is not finite: {result}")
    return result


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def macro_percent(slices: dict[str, float]) -> float:
    if set(slices) != set(DATASETS):
        raise RuntimeError(f"seven-slice identity differs: {sorted(slices)}")
    return statistics.fmean(slices[name] for name in DATASETS)


def load_step0_scores(root: Path) -> dict[str, Any]:
    """Rebuild the audited common step-0 result from final slice statuses."""

    slices: dict[str, float] = {}
    sources: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for dataset in DATASETS:
        metric_name, scale = PRIMARY_METRICS[dataset]
        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        dataset_root = root / "scoring" / "crop-failclosed-v2" / dataset
        for status_path in dataset_root.glob("**/status.json"):
            try:
                status = read_json(status_path)
                record = status["datasets"][dataset]
                value = finite_number(
                    record["metrics"][metric_name],
                    f"{status_path}:{dataset}:{metric_name}",
                )
                if record.get("status") != "done":
                    continue
                if status.get("model_name") != "Qwen3-VL-8B-Instruct":
                    continue
                candidates.append((status_path.stat().st_mtime, status_path, {"value": value}))
            except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
                continue
        if not candidates:
            raise RuntimeError(f"no completed step-0 status contains {dataset}:{metric_name}")
        _, status_path, record = max(candidates, key=lambda item: item[0])
        slices[dataset] = record["value"] * scale
        sources[dataset] = str(status_path)
        hashes[dataset] = sha256_file(status_path)

    macro = macro_percent(slices)
    for dataset, expected in AUDITED_STEP0_PERCENT.items():
        if not math.isclose(slices[dataset], expected, abs_tol=1e-10):
            raise RuntimeError(
                f"derived step-0 {dataset} differs from audited value: "
                f"{slices[dataset]} != {expected}"
            )
    if not math.isclose(macro, AUDITED_STEP0_MACRO_PERCENT, abs_tol=1e-10):
        raise RuntimeError(
            f"derived step-0 macro differs from audit: {macro} != "
            f"{AUDITED_STEP0_MACRO_PERCENT}"
        )
    return {
        "label": "common_step0",
        "macro_percent": macro,
        "slices_percent": slices,
        "source_status_paths": sources,
        "source_status_sha256": hashes,
    }


def load_strict_summary(path: Path, label: str) -> dict[str, Any]:
    payload = read_json(path)
    expected_header = {
        "schema_version": 1,
        "model": "Qwen3-VL-8B-Instruct",
        "phase": "eval",
        "sample_count": 2511,
        "slice_count": 7,
    }
    for key, expected in expected_header.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"{label} strict summary {key} differs: {payload.get(key)!r} != {expected!r}"
            )
    rows = payload.get("slices")
    if not isinstance(rows, list) or len(rows) != len(DATASETS):
        raise RuntimeError(f"{label} strict summary does not contain exactly seven slices")
    by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("dataset") not in DATASETS:
            raise RuntimeError(f"{label} strict summary contains an unknown slice")
        dataset = str(row["dataset"])
        if dataset in by_dataset:
            raise RuntimeError(f"{label} strict summary repeats {dataset}")
        by_dataset[dataset] = row

    slices: dict[str, float] = {}
    for dataset in DATASETS:
        metric_name, scale = PRIMARY_METRICS[dataset]
        try:
            raw_value = by_dataset[dataset]["metrics"][metric_name]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"{label} lacks {dataset}:{metric_name} in strict summary"
            ) from error
        slices[dataset] = finite_number(
            raw_value, f"{label}:{dataset}:{metric_name}"
        ) * scale

    return {
        "label": label,
        "macro_percent": macro_percent(slices),
        "slices_percent": slices,
        "summary_path": str(path),
        "summary_sha256": sha256_file(path),
        "sample_count": 2511,
        "slice_count": 7,
    }


def load_inference(root: Path, label: str) -> dict[str, Any]:
    rank_paths = [root / "inference" / f"rank-{rank}.jsonl" for rank in range(4)]
    rows: list[dict[str, Any]] = []
    rank_counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for rank, path in enumerate(rank_paths):
        require_file(path)
        rank_rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            rank_rows.append(payload)
        rows.extend(rank_rows)
        rank_counts[str(rank)] = len(rank_rows)
        hashes[str(rank)] = sha256_file(path)

    if len(rows) != 2240:
        raise RuntimeError(f"{label} inference row count differs: {len(rows)} != 2240")
    identities = [str(row.get("trajectory_id")) for row in rows]
    if any(identity == "None" for identity in identities) or len(set(identities)) != len(rows):
        raise RuntimeError(f"{label} inference trajectory identities are incomplete or duplicated")
    observed_datasets = Counter(str(row.get("dataset")) for row in rows)
    if set(observed_datasets) != set(DATASETS):
        raise RuntimeError(f"{label} inference dataset identities differ")

    tool_users = 0
    tool_calls = 0
    successful_observations = 0
    rows_with_errors = 0
    error_codes: Counter[str] = Counter()
    per_slice: dict[str, dict[str, int]] = {
        dataset: {"rows": 0, "tool_users": 0, "tool_calls": 0}
        for dataset in DATASETS
    }
    for row in rows:
        dataset = str(row["dataset"])
        calls = row.get("tool_calls", [])
        errors = row.get("tool_errors", [])
        if not isinstance(calls, list) or not isinstance(errors, list):
            raise RuntimeError(f"{label} tool calls/errors are not lists")
        call_count = len(calls)
        if call_count > 4:
            raise RuntimeError(f"{label} exceeds the four-call Crop cap")
        success_count = row.get("successful_observation_count", 0)
        if isinstance(success_count, bool) or not isinstance(success_count, int):
            raise RuntimeError(f"{label} successful observation count is malformed")
        tool_users += int(call_count > 0)
        tool_calls += call_count
        successful_observations += success_count
        rows_with_errors += int(bool(errors))
        per_slice[dataset]["rows"] += 1
        per_slice[dataset]["tool_users"] += int(call_count > 0)
        per_slice[dataset]["tool_calls"] += call_count
        for error in errors:
            if isinstance(error, dict):
                error_codes[str(error.get("code", "missing_code"))] += 1
            else:
                error_codes["malformed_error"] += 1

    row_count = len(rows)
    return {
        "label": label,
        "row_count": row_count,
        "tool_user_count": tool_users,
        "tool_use_rate": tool_users / row_count,
        "tool_use_percent": 100.0 * tool_users / row_count,
        "tool_call_count": tool_calls,
        "mean_tool_calls": tool_calls / row_count,
        "successful_observation_count": successful_observations,
        "mean_successful_observations": successful_observations / row_count,
        "rows_with_tool_errors": rows_with_errors,
        "tool_error_row_rate": rows_with_errors / row_count,
        "tool_error_codes": dict(sorted(error_codes.items())),
        "rank_row_counts": rank_counts,
        "rank_jsonl_sha256": hashes,
        "per_slice": per_slice,
    }


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise RuntimeError("cannot summarize an empty numeric series")
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "population_stddev": statistics.pstdev(values),
        "last": values[-1],
    }


def load_metrics(path: Path, label: str, expected_steps: int) -> dict[str, Any]:
    require_file(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    observed_steps = [row.get("optimizer_step") for row in rows]
    if observed_steps != list(range(1, expected_steps + 1)):
        raise RuntimeError(
            f"{label} metrics are not the exact 1..{expected_steps} sequence: {observed_steps}"
        )

    fields = (
        "mean_answer_reward",
        "mean_conditional_tool_reward",
        "tool_call_attempt_rate",
        "mean_tool_call_attempts",
        "format_error_rate",
    )
    step_statistics: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [
            finite_number(row["step"][field], f"{label}:step{row['optimizer_step']}:{field}")
            for row in rows
        ]
        step_statistics[field] = stats(values)

    checkpoints: dict[str, dict[str, Any]] = {}
    for step in sorted({10, expected_steps}):
        if step > expected_steps:
            continue
        row = rows[step - 1]
        cumulative = row.get("cumulative")
        if not isinstance(cumulative, dict):
            raise RuntimeError(f"{label}:step{step} lacks cumulative metrics")
        checkpoints[str(step)] = {
            "prompts": int(cumulative["prompts"]),
            "trajectories": int(cumulative["trajectories"]),
            "mean_answer_reward": finite_number(
                cumulative["mean_answer_reward"], f"{label}:step{step}:answer"
            ),
            "mean_conditional_tool_reward": finite_number(
                cumulative["mean_conditional_tool_reward"], f"{label}:step{step}:tool_reward"
            ),
            "tool_call_attempt_rate": finite_number(
                cumulative["tool_call_attempt_rate"], f"{label}:step{step}:tool_rate"
            ),
            "mean_tool_call_attempts": finite_number(
                cumulative["mean_tool_call_attempts"], f"{label}:step{step}:calls"
            ),
            "format_error_rate": finite_number(
                cumulative["format_error_rate"], f"{label}:step{step}:format_error"
            ),
        }

    elapsed_values = [
        finite_number(
            row["timing"]["end_to_end_step_seconds"],
            f"{label}:step{row['optimizer_step']}:elapsed",
        )
        for row in rows
    ]
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256_file(path),
        "optimizer_steps": expected_steps,
        "checkpoints": checkpoints,
        "step_statistics": step_statistics,
        "timing_seconds": {
            **stats(elapsed_values),
            "sum": sum(elapsed_values),
        },
    }


def load_optimizer_health(path: Path, label: str, expected_steps: int) -> dict[str, Any]:
    require_file(path)
    records: dict[int, dict[str, float]] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        step_match = STEP_LINE.search(line)
        if not step_match:
            continue
        step = int(step_match.group(1))
        values: dict[str, float] = {}
        for field in TRAIN_LOG_FIELDS:
            match = re.search(
                re.escape(field) + r":(?:np\.float(?:32|64)\()?" + NUMBER,
                line,
            )
            if match:
                values[field] = finite_number(float(match.group(1)), f"{label}:step{step}:{field}")
        if set(values) == set(TRAIN_LOG_FIELDS):
            records[step] = values
    expected = set(range(1, expected_steps + 1))
    if set(records) != expected:
        missing = sorted(expected - set(records))
        raise RuntimeError(f"{label} optimizer-health log lacks steps: {missing}")

    field_stats = {
        field: stats([records[step][field] for step in range(1, expected_steps + 1)])
        for field in TRAIN_LOG_FIELDS
    }
    # Broad fail-closed bounds detect collapse/explosion, not ordinary noise.
    checks = {
        "all_20_steps_present_and_finite": True,
        "grad_norm_not_exploded": field_stats["actor/grad_norm"]["maximum"] <= 1.5,
        "clip_fraction_not_pathological": (
            field_stats["actor/pg_clipfrac"]["mean"] <= 0.25
            and field_stats["actor/pg_clipfrac"]["maximum"] <= 0.75
        ),
        "behavior_ratio_not_pathological": (
            field_stats["actor/behavior_current_log_ratio_abs_mean"]["mean"] <= 0.30
            and field_stats["actor/behavior_current_log_ratio_abs_mean"]["maximum"] <= 0.75
        ),
    }
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256_file(path),
        "steps": expected_steps,
        "fields": field_stats,
        "health_checks": checks,
        "health_pass": all(checks.values()),
    }


def tool_health(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    use_delta = candidate["tool_use_rate"] - baseline["tool_use_rate"]
    call_delta = candidate["mean_tool_calls"] - baseline["mean_tool_calls"]
    error_delta = candidate["tool_error_row_rate"] - baseline["tool_error_row_rate"]
    execution_failures = candidate["tool_error_codes"].get("tool_execution_failed", 0)
    checks = {
        "exact_supported_row_count": candidate["row_count"] == baseline["row_count"] == 2240,
        "tool_use_no_collapse_or_saturation": abs(use_delta) <= 0.10,
        "mean_calls_no_collapse_or_inflation": abs(call_delta) <= 0.75,
        "tool_error_rows_no_large_increase": error_delta <= 0.10,
        "execution_failure_rate_below_2pct": execution_failures / candidate["row_count"] <= 0.02,
    }
    return {
        "baseline_label": baseline["label"],
        "candidate_label": candidate["label"],
        "tool_use_delta_fraction": use_delta,
        "tool_use_delta_percentage_points": 100.0 * use_delta,
        "mean_calls_delta": call_delta,
        "tool_error_row_rate_delta": error_delta,
        "checks": checks,
        "pass": all(checks.values()),
        "interpretation": (
            "Tool health is a guardrail, not an optimization target: more calls are not "
            "automatically better. Bounds are relative to PRL-05 held-out behavior."
        ),
    }


def signed_direction(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "improved"
    if value < -tolerance:
        return "regressed"
    return "tied"


def build_comparison(
    step0: dict[str, Any],
    prl05: dict[str, Any],
    step10: dict[str, Any],
    step20: dict[str, Any],
    step10_tool_health: dict[str, Any],
    prl06_training_health: dict[str, Any],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    directions = Counter()
    for dataset in DATASETS:
        s0 = step0["slices_percent"][dataset]
        b16 = prl05["slices_percent"][dataset]
        b32s10 = step10["slices_percent"][dataset]
        b32s20 = step20["slices_percent"][dataset]
        delta = b32s10 - b16
        direction = signed_direction(delta)
        directions[direction] += 1
        rows[dataset] = {
            "step0_percent": s0,
            "prl05_bs16_step20_percent": b16,
            "prl06_bs32_step10_percent": b32s10,
            "prl06_bs32_step20_percent": b32s20,
            "sample_matched_delta_pp": delta,
            "step10_delta_vs_step0_pp": b32s10 - s0,
            "update_matched_delta_pp": b32s20 - b16,
            "step20_delta_vs_step0_pp": b32s20 - s0,
            "sample_matched_direction": direction,
        }

    prl05_gap = step0["macro_percent"] - prl05["macro_percent"]
    sample_delta = step10["macro_percent"] - prl05["macro_percent"]
    update_delta = step20["macro_percent"] - prl05["macro_percent"]
    recovery_fraction = sample_delta / prl05_gap if prl05_gap > 0 else None
    material_recovery = sample_delta >= 1.0
    majority_direction = directions["improved"] >= 4
    healthy = bool(step10_tool_health["pass"] and prl06_training_health["health_pass"])

    support = material_recovery and majority_direction and healthy
    materially_worse = sample_delta <= -1.0 and directions["regressed"] >= 4 and healthy
    if not healthy:
        decision = "inconclusive_due_to_health_guardrail"
        chinese = "训练或工具行为未通过健康门槛，不能据此判断大 BS 有效。"
    elif support:
        decision = "supports_larger_batch"
        chinese = (
            "支持大 BS：唯一因果可比的 BS32-step10 至少恢复 1 pp，且至少 4/7 "
            "切片同向改善，训练与工具行为健康。"
        )
    elif materially_worse:
        decision = "evidence_against_larger_batch"
        chinese = (
            "不支持大 BS：sample-budget 对齐的 BS32-step10 至少下降 1 pp，且至少 "
            "4/7 切片同向退化。"
        )
    else:
        decision = "no_clear_gain"
        chinese = (
            "没有明确的大 BS 增益：sample-budget 对齐改善不足约 1 pp，或七个切片方向混合。"
        )

    return {
        "sample_budget_matched": {
            "comparison": "PRL-06 BS32 step10 vs PRL-05 BS16 step20",
            "both_prompts": 320,
            "both_trajectories": 2560,
            "macro_delta_percentage_points": sample_delta,
            "prl05_loss_to_step0_percentage_points": prl05_gap,
            "recovery_fraction_of_prl05_loss": recovery_fraction,
            "slice_directions": dict(directions),
        },
        "update_count_matched_with_exposure_confound": {
            "comparison": "PRL-06 BS32 step20 vs PRL-05 BS16 step20",
            "both_optimizer_updates": 20,
            "prl05_prompts": 320,
            "prl06_prompts": 640,
            "prl05_trajectories": 2560,
            "prl06_trajectories": 5120,
            "macro_delta_percentage_points": update_delta,
            "caveat": (
                "This comparison mixes batch size with 2x sample exposure and cannot "
                "by itself establish a batch-size effect."
            ),
        },
        "slices": rows,
        "decision_rule": {
            "material_macro_recovery_threshold_pp": 1.0,
            "minimum_improved_slices_out_of_7": 4,
            "requires_tool_health": True,
            "requires_training_health": True,
            "sub_1pp_or_mixed_direction": "no_clear_gain",
        },
        "decision": decision,
        "conclusion_zh": chinese,
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    scores = report["scores"]
    heldout = report["heldout_tool_behavior"]
    comparison = report["comparison"]
    sample = comparison["sample_budget_matched"]
    update = comparison["update_count_matched_with_exposure_confound"]
    prl05_train = report["training"]["prl05_bs16"]
    prl06_train = report["training"]["prl06_bs32"]
    health05 = report["optimizer_health"]["prl05_bs16"]
    health06 = report["optimizer_health"]["prl06_bs32"]

    lines = [
        "# PRL-06 BS32 夜间实验结论",
        "",
        f"生成时间：{report['generated_at_jst']}",
        "",
        "## 结论",
        "",
        comparison["conclusion_zh"],
        "",
        (
            f"唯一能隔离 batch-size 的主对照是 **BS32-step10 vs BS16-step20**："
            f"两者均看过 320 prompts / 2,560 trajectories。macro 差为 "
            f"**{sample['macro_delta_percentage_points']:+.3f} pp**，七切片方向为 "
            f"{sample['slice_directions'].get('improved', 0)} 升 / "
            f"{sample['slice_directions'].get('regressed', 0)} 降 / "
            f"{sample['slice_directions'].get('tied', 0)} 平。"
        ),
        "",
        "## ACC-VAL 与工具行为",
        "",
        "| 检查点 | prompts / trajectories | 7-slice macro | held-out tool use | calls / sample |",
        "|---|---:|---:|---:|---:|",
        (
            f"| common step0 | — | {scores['common_step0']['macro_percent']:.3f}% | "
            f"{heldout['common_step0']['tool_use_percent']:.3f}% | "
            f"{heldout['common_step0']['mean_tool_calls']:.3f} |"
        ),
        (
            f"| PRL-05 BS16 step20 | 320 / 2,560 | "
            f"{scores['prl05_bs16_step20']['macro_percent']:.3f}% | "
            f"{heldout['prl05_bs16_step20']['tool_use_percent']:.3f}% | "
            f"{heldout['prl05_bs16_step20']['mean_tool_calls']:.3f} |"
        ),
        (
            f"| PRL-06 BS32 step10 | 320 / 2,560 | "
            f"{scores['prl06_bs32_step10']['macro_percent']:.3f}% | "
            f"{heldout['prl06_bs32_step10']['tool_use_percent']:.3f}% | "
            f"{heldout['prl06_bs32_step10']['mean_tool_calls']:.3f} |"
        ),
        (
            f"| PRL-06 BS32 step20 | 640 / 5,120 | "
            f"{scores['prl06_bs32_step20']['macro_percent']:.3f}% | "
            f"{heldout['prl06_bs32_step20']['tool_use_percent']:.3f}% | "
            f"{heldout['prl06_bs32_step20']['mean_tool_calls']:.3f} |"
        ),
        "",
        "| slice | step0 | BS16-s20 | BS32-s10 | Δ sample-matched | BS32-s20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        row = comparison["slices"][dataset]
        lines.append(
            f"| {dataset} | {row['step0_percent']:.3f}% | "
            f"{row['prl05_bs16_step20_percent']:.3f}% | "
            f"{row['prl06_bs32_step10_percent']:.3f}% | "
            f"{row['sample_matched_delta_pp']:+.3f} pp | "
            f"{row['prl06_bs32_step20_percent']:.3f}% |"
        )

    lines.extend(
        [
            "",
            "## 训练健康度",
            "",
            "| 指标（20 updates） | PRL-05 BS16 | PRL-06 BS32 |",
            "|---|---:|---:|",
            (
                "| pg_loss 标准差 | "
                f"{health05['fields']['actor/pg_loss']['population_stddev']:.5f} | "
                f"{health06['fields']['actor/pg_loss']['population_stddev']:.5f} |"
            ),
            (
                "| grad_norm 均值 / 最大值 | "
                f"{health05['fields']['actor/grad_norm']['mean']:.5f} / "
                f"{health05['fields']['actor/grad_norm']['maximum']:.5f} | "
                f"{health06['fields']['actor/grad_norm']['mean']:.5f} / "
                f"{health06['fields']['actor/grad_norm']['maximum']:.5f} |"
            ),
            (
                "| clipfrac 均值 / 最大值 | "
                f"{health05['fields']['actor/pg_clipfrac']['mean']:.5f} / "
                f"{health05['fields']['actor/pg_clipfrac']['maximum']:.5f} | "
                f"{health06['fields']['actor/pg_clipfrac']['mean']:.5f} / "
                f"{health06['fields']['actor/pg_clipfrac']['maximum']:.5f} |"
            ),
            (
                "| behavior/current abs(log-ratio) 均值 / 最大值 | "
                f"{health05['fields']['actor/behavior_current_log_ratio_abs_mean']['mean']:.5f} / "
                f"{health05['fields']['actor/behavior_current_log_ratio_abs_mean']['maximum']:.5f} | "
                f"{health06['fields']['actor/behavior_current_log_ratio_abs_mean']['mean']:.5f} / "
                f"{health06['fields']['actor/behavior_current_log_ratio_abs_mean']['maximum']:.5f} |"
            ),
            (
                "| cumulative answer reward | "
                f"{prl05_train['checkpoints']['20']['mean_answer_reward']:.4f} | "
                f"{prl06_train['checkpoints']['20']['mean_answer_reward']:.4f} |"
            ),
            (
                "| cumulative train tool rate / calls | "
                f"{100 * prl05_train['checkpoints']['20']['tool_call_attempt_rate']:.2f}% / "
                f"{prl05_train['checkpoints']['20']['mean_tool_call_attempts']:.3f} | "
                f"{100 * prl06_train['checkpoints']['20']['tool_call_attempt_rate']:.2f}% / "
                f"{prl06_train['checkpoints']['20']['mean_tool_call_attempts']:.3f} |"
            ),
            "",
            (
                f"训练健康门：PRL-06 **{'PASS' if health06['health_pass'] else 'FAIL'}**；"
                f"sample-matched held-out 工具健康门："
                f"**{'PASS' if report['tool_health']['step10_vs_prl05']['pass'] else 'FAIL'}**。"
            ),
            "",
            "## 解释边界",
            "",
            (
                f"- BS32-step20 相对 BS16-step20 为 {update['macro_delta_percentage_points']:+.3f} pp，"
                "但前者看过 640 prompts / 5,120 trajectories，后者只有 320 / 2,560；"
                "这个结果混合了 batch size 与 2× 数据暴露，不能单独证明大 BS 有效。"
            ),
            "- tool use 只是健康护栏；调用更多不自动等于答题更好。",
            (
                "- 保守判据：BS32-step10 至少恢复 1.0 pp、至少 4/7 slice 同向改善，"
                "并同时通过训练/工具健康门，才判为支持；不足约 1 pp 或方向混合均为 no clear gain。"
            ),
            "- 这里的 7-slice macro 是七个 slice 的等权平均，不是按 2,511 条样本加权。",
            "",
            "## 机器结果",
            "",
            f"同目录 JSON：`{Path(report['output_json']).name}`。其中保存全部输入 SHA-256、"
            "逐 slice 数值、轨迹工具统计与判定门槛。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    control = repository / "artifacts/policy-control/PRL-06-R0-bs32-20step"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repository)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=control / "prl06-bs32-result.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=control / "prl06-bs32-result.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    policy05 = repo / (
        "artifacts/policy/PRL-05-R0-qwen3-instruct-grpo-bs16-crop-t1full-"
        "toolw0p2-20step-gpu0123"
    )
    policy06 = repo / (
        "artifacts/policy/PRL-06-R0-qwen3-instruct-grpo-bs32-crop-t1full-"
        "toolw0p2-20step-gpu0123"
    )
    eval0 = repo / "artifacts/evaluation/PRL-04-R2-crop-step0-coredev2511-gpu0123"
    eval05 = repo / "artifacts/evaluation/PRL-05-R0-crop-step20-coredev2511-gpu0123"
    eval06s10 = repo / "artifacts/evaluation/PRL-06-R0-bs32-crop-step10-coredev2511-gpu0123"
    eval06s20 = repo / "artifacts/evaluation/PRL-06-R0-bs32-crop-step20-coredev2511-gpu0123"

    score0 = load_step0_scores(eval0)
    score05 = load_strict_summary(
        eval05 / "scoring/crop-auto-v0/coredev-2511-eval-summary.json",
        "prl05_bs16_step20",
    )
    score06s10 = load_strict_summary(
        eval06s10 / "scoring/crop-auto-v0/coredev-2511-eval-summary.json",
        "prl06_bs32_step10",
    )
    score06s20 = load_strict_summary(
        eval06s20 / "scoring/crop-auto-v0/coredev-2511-eval-summary.json",
        "prl06_bs32_step20",
    )

    tool0 = load_inference(eval0, "common_step0")
    tool05 = load_inference(eval05, "prl05_bs16_step20")
    tool06s10 = load_inference(eval06s10, "prl06_bs32_step10")
    tool06s20 = load_inference(eval06s20, "prl06_bs32_step20")

    metrics05 = load_metrics(policy05 / "metrics.jsonl", "prl05_bs16", 20)
    metrics06 = load_metrics(policy06 / "metrics.jsonl", "prl06_bs32", 20)
    expected_exposure = {
        "PRL-05 step20": (metrics05["checkpoints"]["20"], 320, 2560),
        "PRL-06 step10": (metrics06["checkpoints"]["10"], 320, 2560),
        "PRL-06 step20": (metrics06["checkpoints"]["20"], 640, 5120),
    }
    for owner, (checkpoint, expected_prompts, expected_trajectories) in expected_exposure.items():
        if (
            checkpoint["prompts"] != expected_prompts
            or checkpoint["trajectories"] != expected_trajectories
        ):
            raise RuntimeError(
                f"{owner} exposure differs: prompts={checkpoint['prompts']}, "
                f"trajectories={checkpoint['trajectories']}"
            )
    health05 = load_optimizer_health(
        repo / "artifacts/policy-control/PRL-05-R0-toolw0p2-20step/launch.log",
        "prl05_bs16",
        20,
    )
    health06 = load_optimizer_health(
        repo / "artifacts/policy-control/PRL-06-R0-bs32-20step/launch.log",
        "prl06_bs32",
        20,
    )

    step10_tool_health = tool_health(tool06s10, tool05)
    step20_tool_health = tool_health(tool06s20, tool05)
    comparison = build_comparison(
        score0,
        score05,
        score06s10,
        score06s20,
        step10_tool_health,
        health06,
    )

    now = datetime.now(timezone.utc)
    output_json = args.output_json.resolve()
    output_markdown = args.output_markdown.resolve()
    report: dict[str, Any] = {
        "schema_version": "prl06-bs32-result-report-v1",
        "status": "pass",
        "generated_at_utc": now.isoformat(),
        "generated_at_jst": now.astimezone(ZoneInfo("Asia/Tokyo")).isoformat(),
        "experiment": {
            "single_training_variable": "global prompt batch 16 -> 32 (GA4 -> GA8 mechanically)",
            "fixed": {
                "rollout_n": 8,
                "world_size": 4,
                "learning_rate": 1e-6,
                "answer_reward_weight": 0.8,
                "format_reward_weight": 0.2,
                "conditional_tool_reward_weight": 0.2,
                "scheduler_horizon_steps": 80,
            },
        },
        "scores": {
            "common_step0": score0,
            "prl05_bs16_step20": score05,
            "prl06_bs32_step10": score06s10,
            "prl06_bs32_step20": score06s20,
        },
        "heldout_tool_behavior": {
            "common_step0": tool0,
            "prl05_bs16_step20": tool05,
            "prl06_bs32_step10": tool06s10,
            "prl06_bs32_step20": tool06s20,
        },
        "training": {
            "prl05_bs16": metrics05,
            "prl06_bs32": metrics06,
        },
        "optimizer_health": {
            "prl05_bs16": health05,
            "prl06_bs32": health06,
        },
        "tool_health": {
            "step10_vs_prl05": step10_tool_health,
            "step20_vs_prl05_noncausal": step20_tool_health,
        },
        "comparison": comparison,
        "output_json": str(output_json),
        "output_markdown": str(output_markdown),
    }
    atomic_write(output_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write(output_markdown, render_markdown(report))
    print(
        json.dumps(
            {
                "status": "pass",
                "decision": comparison["decision"],
                "sample_matched_delta_pp": comparison["sample_budget_matched"][
                    "macro_delta_percentage_points"
                ],
                "output_json": str(output_json),
                "output_markdown": str(output_markdown),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
