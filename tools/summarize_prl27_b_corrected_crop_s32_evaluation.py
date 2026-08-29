#!/usr/bin/env python3
"""Publish the PRL-27-B seven-subset score, tool-use, and length audit."""

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
    "PRL27-B-CROP-REPLAY-BYTE-PARITY-TRAIN512-S32-TRAINING-RUN-COREDEV2511-PIXEL512-V1"
)
TRAINING_RUN_ID = (
    "PRL-27-B-TRAIN512-S32-CROP-REPLAY-BYTE-PARITY-QWEN3-INSTRUCT-"
    "BS16-N16-TEACHER25-WS8"
)
PAIRED_SEED_NAMESPACE = (
    "coredev2511/prl27-b/crop-replay-byte-parity/training-run/"
    "train512-eval512/s32/temp1/seed42/v1"
)
PIXEL512 = 262_144
STEP = 32
ENVIRONMENT_SHA256 = "f745fa6cfcc3ba9eb27125a49581fd823fb5930b7b0a51b28e51982999fa2d0a"
INFERENCE_ROWS = {
    "VStarBench": 191,
    "HRBench4K": 200,
    "BLINK": 180,
    "OCRBench_v2": 600,
    "MMMU_Pro_10c": 269,
    "MathVista_MINI": 300,
    "MathVerse_MINI": 500,
}
OFFICIAL_ROWS = {
    "VStarBench": 191,
    "HRBench4K": 200,
    "BLINK": 420,
    "OCRBench_v2": 600,
    "MMMU_Pro_10c": 300,
    "MathVista_MINI": 300,
    "MathVerse_MINI": 500,
}
EXPECTED_COVERAGE = {
    "official_manifest_rows": 2511,
    "evaluated_single_image_rows": 2240,
    "held_multi_image_rows": 271,
    "multi_image_policy": "unsupported_explicit_hold",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON boundary differs: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"JSON artifact is malformed: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return payload


def _require_self_identity(payload: dict[str, Any], *, name: str) -> str:
    identity = payload.get("identity_sha256")
    content = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if identity != _canonical_sha256(content):
        raise RuntimeError(f"{name} self identity differs")
    return identity


def _read_rows(inference_root: Path, evaluation_identity: str) -> list[dict[str, Any]]:
    rank_paths = sorted(inference_root.glob("rank-*.jsonl"))
    if len(rank_paths) != 4 or any(path.is_symlink() for path in rank_paths):
        raise RuntimeError("PRL-27-B inference rank coverage differs")
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
        raise RuntimeError("PRL-27-B inference row count differs")
    if Counter(row.get("dataset") for row in rows) != Counter(INFERENCE_ROWS):
        raise RuntimeError("PRL-27-B inference subset coverage differs")
    sample_ids = [row.get("sample_id") for row in rows]
    if (
        any(not isinstance(value, str) or not value for value in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
        or any(row.get("evaluation_id") != EVALUATION_ID for row in rows)
        or any(row.get("optimizer_step") != STEP for row in rows)
        or any(row.get("world_size") != 4 for row in rows)
        or any(
            row.get("evaluation_identity_sha256") != evaluation_identity for row in rows
        )
    ):
        raise RuntimeError("PRL-27-B inference identity closure differs")
    return rows


def _percentile(values: Iterable[int], quantile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _usage(rows: list[dict[str, Any]]) -> dict[str, object]:
    attempts = 0
    successful_calls = 0
    errors = 0
    attempting_trajectories = 0
    successful_trajectories = 0
    repeated_trajectories = 0
    observations = 0
    token_counts: list[int] = []
    stop_counts: Counter[str] = Counter()
    function_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    for row in rows:
        calls = row.get("tool_calls")
        tool_errors = row.get("tool_errors")
        turns = row.get("assistant_turns")
        observed = row.get("successful_observation_count")
        stop = row.get("stop")
        if (
            not isinstance(calls, list)
            or not isinstance(tool_errors, list)
            or not isinstance(turns, list)
            or type(observed) is not int
            or observed < 0
            or not isinstance(stop, str)
            or not stop
        ):
            raise RuntimeError("PRL-27-B trajectory audit differs")
        call_count = len(calls)
        error_count = len(tool_errors)
        attempt_count = call_count + error_count
        if observed != call_count or call_count > 6 or attempt_count > 7:
            raise RuntimeError("PRL-27-B tool attempt counts differ")
        attempts += attempt_count
        successful_calls += call_count
        errors += error_count
        attempting_trajectories += int(attempt_count > 0)
        successful_trajectories += int(call_count > 0)
        repeated_trajectories += int(call_count > 1)
        observations += observed
        stop_counts[stop] += 1
        generated = 0
        for turn_index, turn in enumerate(turns):
            if (
                not isinstance(turn, dict)
                or turn.get("turn_index") != turn_index
                or type(turn.get("sampled_token_count")) is not int
                or turn["sampled_token_count"] < 0
            ):
                raise RuntimeError("PRL-27-B assistant-turn audit differs")
            generated += turn["sampled_token_count"]
        token_counts.append(generated)
        for call in calls:
            function_name = (
                call.get("function_name") if isinstance(call, dict) else None
            )
            if function_name != "image_zoom_in_tool":
                raise RuntimeError("PRL-27-B successful tool name differs")
            function_counts[function_name] += 1
        for error in tool_errors:
            code = error.get("code") if isinstance(error, dict) else None
            if not isinstance(code, str) or not code:
                raise RuntimeError("PRL-27-B tool error identity differs")
            error_counts[code] += 1
    count = len(rows)
    if count == 0:
        raise RuntimeError("PRL-27-B usage audit cannot be empty")
    return {
        "trajectory_count": count,
        "no_tool_trajectory_count": count - attempting_trajectories,
        "trajectories_attempting_tool": attempting_trajectories,
        "tool_attempt_trajectory_rate": attempting_trajectories / count,
        "total_tool_attempts": attempts,
        "mean_tool_attempts_per_trajectory": attempts / count,
        "trajectories_with_successful_tool_call": successful_trajectories,
        "successful_tool_call_trajectory_rate": successful_trajectories / count,
        "successful_tool_call_count": successful_calls,
        "trajectories_with_repeat_successful_tool_call": repeated_trajectories,
        "successful_observation_count": observations,
        "tool_error_count": errors,
        "tool_error_code_counts": dict(sorted(error_counts.items())),
        "function_call_counts": dict(sorted(function_counts.items())),
        "generated_token_mean": sum(token_counts) / count,
        "generated_token_p50": _percentile(token_counts, 0.50),
        "generated_token_p95": _percentile(token_counts, 0.95),
        "generated_token_p99": _percentile(token_counts, 0.99),
        "stop_counts": dict(sorted(stop_counts.items())),
    }


def summarize(evaluation_root: Path, output: Path) -> dict[str, object]:
    plan_path = evaluation_root / "runtime/bound-crop-plan.json"
    handoff_path = evaluation_root / "runtime/bound-handoff.json"
    proof_path = evaluation_root / "step32/runtime/pixel512-processor-proof.json"
    paired_path = evaluation_root / "paired-summary.json"
    summary_path = evaluation_root / (
        "step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json"
    )
    inference_root = evaluation_root / "step32/inference"

    plan = _read_json(plan_path)
    handoff = _read_json(handoff_path)
    proof = _read_json(proof_path)
    paired = _read_json(paired_path)
    summary = _read_json(summary_path)

    expected_arm = {
        "name": "step32",
        "optimizer_step": STEP,
        "evaluation_id": EVALUATION_ID,
        "source": {
            "kind": "owner_checkpoint",
            "relative_path": "permanent-checkpoints/global_step_32",
        },
    }
    owner = plan.get("checkpoint_owner")
    protocol = plan.get("protocol")
    identity = (
        protocol.get("training_run_identity") if isinstance(protocol, dict) else None
    )
    if (
        plan.get("schema_version") != "tgvf.paired-policy-benchmark-plan.v3"
        or plan.get("evaluation_id") != EVALUATION_ID
        or plan.get("evaluation_image_max_pixels") != PIXEL512
        or plan.get("expected_task_count") != 2511
        or plan.get("expected_single_image_count") != 2240
        or plan.get("unsupported_multi_image_count") != 271
        or plan.get("arms") != [expected_arm]
        or not isinstance(owner, dict)
        or owner.get("run_id") != TRAINING_RUN_ID
        or not isinstance(protocol, dict)
        or protocol.get("evaluation_protocol") != "training_run"
        or protocol.get("action_boundary")
        != {
            "stop_strings": ["</tool_call>"],
            "stop_token_ids": [151645],
            "include_stop_str_in_output": True,
            "ignore_eos": False,
        }
        or not isinstance(identity, dict)
        or identity.get("tool_profile") != "crop_only"
        or identity.get("success_environment_renderer")
        != "render_qwen_native_matched_crop_success_environment_text"
        or identity.get("success_environment_text_sha256") != ENVIRONMENT_SHA256
        or identity.get("response_budget_scope") != "total_response_tokens"
        or identity.get("single_response_max_tokens") != 10_240
        or plan.get("paired_rng", {}).get("seed_namespace") != PAIRED_SEED_NAMESPACE
        or plan.get("paired_rng", {}).get("protocol_sha256")
        != _canonical_sha256(identity)
    ):
        raise RuntimeError("PRL-27-B training-run plan identity differs")

    handoff_identity = _require_self_identity(handoff, name="handoff")
    crop = handoff.get("crop")
    if (
        handoff.get("schema_version")
        != "tgvf.prl27-b-crop-replay-byte-parity-training-run-evaluation-handoff.v1"
        or handoff.get("status") != "ready"
        or handoff.get("evaluation_id") != EVALUATION_ID
        or handoff.get("train_image_max_pixels") != PIXEL512
        or handoff.get("evaluation_image_max_pixels") != PIXEL512
        or handoff.get("optimizer_step") != STEP
        or not isinstance(crop, dict)
        or crop.get("run_id") != TRAINING_RUN_ID
        or crop.get("evaluation_protocol") != "training_run"
        or crop.get("paired_seed_namespace") != PAIRED_SEED_NAMESPACE
        or crop.get("protocol_sha256") != plan["paired_rng"]["protocol_sha256"]
        or crop.get("bound_plan_file_sha256") != _sha256(plan_path)
    ):
        raise RuntimeError("PRL-27-B handoff identity differs")

    proof_identity = _require_self_identity(proof, name="processor proof")
    proof_protocol = proof.get("protocol")
    dynamic_proof = proof.get("proof")
    if (
        proof.get("schema_version") != "tgvf.prl26-train512-processor-proof.v1"
        or proof.get("arm") != "crop"
        or proof.get("evaluation_id") != EVALUATION_ID
        or proof.get("optimizer_step") != STEP
        or proof.get("train_image_max_pixels") != PIXEL512
        or proof.get("evaluation_image_max_pixels") != PIXEL512
        or not isinstance(proof_protocol, dict)
        or proof_protocol.get("continuation_parity") is not True
        or proof_protocol.get("success_environment_text_sha256") != ENVIRONMENT_SHA256
        or not isinstance(dynamic_proof, dict)
        or dynamic_proof.get("continuation_environment_token_count") != 60
        or dynamic_proof.get("success_environment_renderer")
        != "render_qwen_native_matched_crop_success_environment_text"
    ):
        raise RuntimeError("PRL-27-B processor proof differs")

    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "pass"
        or summary.get("phase") != "eval"
        or summary.get("sample_count") != 2511
        or summary.get("slice_count") != 7
        or not isinstance(summary.get("slices"), list)
        or len(summary["slices"]) != 7
        or {
            item.get("dataset"): item.get("sample_count")
            for item in summary["slices"]
            if isinstance(item, dict)
        }
        != OFFICIAL_ROWS
    ):
        raise RuntimeError("PRL-27-B official seven-subset summary differs")

    arms = paired.get("arms")
    contracts = paired.get("identity_contracts")
    if (
        paired.get("schema_version") != "tgvf.paired-coredev-summary.v2"
        or paired.get("evaluation_id") != EVALUATION_ID
        or paired.get("coverage") != EXPECTED_COVERAGE
        or not isinstance(arms, dict)
        or set(arms) != {"step32"}
        or paired.get("step32") != summary
        or arms["step32"].get("optimizer_step") != STEP
        or arms["step32"].get("official_summary") != summary
        or not isinstance(contracts, dict)
        or contracts.get("backend") != "full_model"
        or contracts.get("evaluation_protocol_source")
        != "checkpoint_owner_policy_config"
        or contracts.get("training_run_protocol") != protocol
        or arms["step32"].get("evaluation_identity_sha256")
        != proof.get("evaluation_identity_sha256")
    ):
        raise RuntimeError("PRL-27-B paired summary identity differs")

    rows = _read_rows(inference_root, str(proof.get("evaluation_identity_sha256", "")))
    headline = extract_coredev_macro_star(summary)
    payload: dict[str, object] = {
        "schema_version": "tgvf.prl27-b-corrected-crop-s32-results.v1",
        "status": "pass",
        "evaluation_id": EVALUATION_ID,
        "contract": (
            "independent fresh-S0 Crop replay-byte-parity Train@512 S32; "
            "exact training_run Eval@512"
        ),
        "coverage": {
            "official_manifest_rows": 2511,
            "evaluated_single_image_rows": 2240,
            "held_multi_image_rows": 271,
            "subset_count": 7,
        },
        "handoff_identity_sha256": handoff_identity,
        "paired_rng_protocol_sha256": plan["paired_rng"]["protocol_sha256"],
        "processor_proof_identity_sha256": proof_identity,
        "paired_summary_path": str(paired_path),
        "paired_summary_sha256": _sha256(paired_path),
        "length_unit": "sampled model tokens across all assistant turns",
        "tool_usage_definitions": {
            "tool_attempt": "one successful tool call or one recorded tool error",
            "successful_tool_call": (
                "one executed crop action with one visual observation"
            ),
            "no_tool_trajectory": (
                "a trajectory with zero successful or failed tool attempts"
            ),
        },
        "arm": {
            "method": "Crop replay-byte-parity Train@512 S32",
            "optimizer_step": STEP,
            "train_image_max_pixels": PIXEL512,
            "evaluation_image_max_pixels": PIXEL512,
            "macro_star_percent": headline["macro_star_percent"],
            "headline": headline,
            "seven_subset_statistics": summary["slices"],
            "tool_usage_overall": _usage(rows),
            "tool_usage_by_subset": {
                dataset: _usage([row for row in rows if row["dataset"] == dataset])
                for dataset in INFERENCE_ROWS
            },
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
        },
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
