#!/usr/bin/env python3
"""Publish the PRL-26 C/D headline table and per-subset tool-use audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.coredev_results import (  # noqa: E402
    extract_coredev_macro_star,
    write_json_atomic,
)


EVALUATION_ID = (
    "PRL26-CD-TRAIN512-S32-TGVF-TARGET-PROMPT-PAIR-PIXEL512-"
    "COREDEV2511-SEED42-V1"
)
DATASET_ROWS = {
    "VStarBench": 191,
    "HRBench4K": 200,
    "BLINK": 180,
    "OCRBench_v2": 600,
    "MMMU_Pro_10c": 269,
    "MathVista_MINI": 300,
    "MathVerse_MINI": 500,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"JSON artifact is unavailable or malformed: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return payload


def _read_rows(inference_root: Path) -> list[dict[str, Any]]:
    rank_paths = sorted(inference_root.glob("rank-*.jsonl"))
    if len(rank_paths) != 4:
        raise RuntimeError(f"inference rank coverage differs: {inference_root}")
    rows: list[dict[str, Any]] = []
    for path in rank_paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"malformed inference row: {path}:{line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"inference row must be an object: {path}:{line_number}"
                    )
                rows.append(row)
    if len(rows) != 2240:
        raise RuntimeError(f"inference row count differs: {len(rows)}")
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in sample_ids):
        raise RuntimeError("inference sample identity is malformed")
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("inference sample identity is duplicated")
    observed = Counter(row.get("dataset") for row in rows)
    if observed != Counter(DATASET_ROWS):
        raise RuntimeError(f"inference dataset coverage differs: {dict(observed)}")
    return rows


def _percentile(values: Iterable[int], quantile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _usage(rows: list[dict[str, Any]]) -> dict[str, object]:
    total_calls = 0
    used = 0
    successful_observations = 0
    tool_errors = 0
    token_counts: list[int] = []
    stop_counts: Counter[str] = Counter()
    function_counts: Counter[str] = Counter()
    for row in rows:
        calls = row.get("tool_calls")
        turns = row.get("assistant_turns")
        errors = row.get("tool_errors")
        observations = row.get("successful_observation_count")
        stop = row.get("stop")
        if (
            not isinstance(calls, list)
            or not isinstance(turns, list)
            or not isinstance(errors, list)
            or type(observations) is not int
            or observations < 0
            or not isinstance(stop, str)
        ):
            raise RuntimeError("trajectory audit structure differs")
        call_count = len(calls)
        total_calls += call_count
        used += int(call_count > 0)
        successful_observations += observations
        tool_errors += len(errors)
        stop_counts[stop] += 1
        generated_tokens = 0
        for turn in turns:
            if not isinstance(turn, dict) or type(turn.get("sampled_token_count")) is not int:
                raise RuntimeError("assistant-turn token audit differs")
            generated_tokens += turn["sampled_token_count"]
        token_counts.append(generated_tokens)
        for call in calls:
            function_name = call.get("function_name") if isinstance(call, dict) else None
            if not isinstance(function_name, str) or not function_name:
                raise RuntimeError("tool-call function identity differs")
            function_counts[function_name] += 1
    count = len(rows)
    return {
        "trajectory_count": count,
        "trajectories_using_tool": used,
        "tool_use_rate": used / count,
        "total_tool_calls": total_calls,
        "mean_tool_calls_per_trajectory": total_calls / count,
        "mean_tool_calls_when_used": total_calls / used if used else 0.0,
        "successful_observation_count": successful_observations,
        "mean_successful_observations_per_trajectory": (
            successful_observations / count
        ),
        "tool_error_count": tool_errors,
        "generated_token_mean": sum(token_counts) / count,
        "generated_token_p50": _percentile(token_counts, 0.50),
        "generated_token_p95": _percentile(token_counts, 0.95),
        "generated_token_p99": _percentile(token_counts, 0.99),
        "stop_counts": dict(sorted(stop_counts.items())),
        "function_call_counts": dict(sorted(function_counts.items())),
    }


def _arm_record(root: Path, arm: str) -> tuple[dict[str, object], dict[str, str]]:
    summary_path = (
        root
        / arm
        / "scoring/coredev-official-v1/coredev-2511-eval-summary.json"
    )
    summary = _read_json(summary_path)
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "pass"
        or summary.get("phase") != "eval"
        or summary.get("sample_count") != 2511
        or summary.get("slice_count") != 7
        or not isinstance(summary.get("slices"), list)
        or len(summary["slices"]) != 7
    ):
        raise RuntimeError(f"{arm} official CoreDev summary is incomplete")
    rows = _read_rows(root / arm / "inference")
    stream_ids: dict[str, str] = {}
    for row in rows:
        sample_id = row["sample_id"]
        stream_id = row.get("paired_rng_stream_identity_sha256")
        if (
            not isinstance(stream_id, str)
            or len(stream_id) != 64
            or any(character not in "0123456789abcdef" for character in stream_id)
        ):
            raise RuntimeError(f"{arm} paired RNG stream identity is malformed")
        stream_ids[sample_id] = stream_id
    by_dataset = {
        dataset: _usage([row for row in rows if row["dataset"] == dataset])
        for dataset in DATASET_ROWS
    }
    headline = extract_coredev_macro_star(summary)
    return (
        {
            "method": (
                "TGVF Short Train@512 S32"
                if arm == "short"
                else "TGVF Target-guide-v2 Train@512 S32"
            ),
            "optimizer_step": 32,
            "train_image_max_pixels": 262144,
            "evaluation_image_max_pixels": 262144,
            "macro_star_percent": headline["macro_star_percent"],
            "headline": headline,
            "seven_subset_statistics": summary["slices"],
            "tool_usage_overall": _usage(rows),
            "tool_usage_by_subset": by_dataset,
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
        },
        stream_ids,
    )


def summarize(root: Path, output: Path) -> dict[str, object]:
    paired_summary_path = root / "paired-summary.json"
    paired = _read_json(paired_summary_path)
    if (
        paired.get("schema_version") != "tgvf.paired-coredev-summary.v2"
        or paired.get("evaluation_id") != EVALUATION_ID
        or set(paired.get("arms", {})) != {"short", "full"}
        or paired.get("target_prompt_pair", {}).get("kind")
        != "target_prompt_pair_v1"
    ):
        raise RuntimeError("paired target-prompt summary identity differs")
    short, short_streams = _arm_record(root, "short")
    full, full_streams = _arm_record(root, "full")
    if short_streams != full_streams:
        raise RuntimeError("Short/Full per-task RNG stream identities differ")
    payload: dict[str, object] = {
        "schema_version": "tgvf.prl26-tgvf-target-prompt-s32-results.v1",
        "status": "pass",
        "evaluation_id": EVALUATION_ID,
        "contract": (
            "independent fresh-S0 Train@512 S32; matched Eval@512; prompt-axis "
            "common random numbers"
        ),
        "coverage": {
            "official_manifest_rows": 2511,
            "evaluated_single_image_rows": 2240,
            "held_multi_image_rows": 271,
            "subset_count": 7,
        },
        "paired_rng_streams_equal": True,
        "paired_rng_stream_count": len(short_streams),
        "paired_summary_path": str(paired_summary_path.resolve()),
        "paired_summary_sha256": _sha256(paired_summary_path),
        "arms": {"short": short, "full": full},
    }
    write_json_atomic(output.resolve(), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.evaluation_root.resolve(), args.output.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
