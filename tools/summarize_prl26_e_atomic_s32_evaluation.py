#!/usr/bin/env python3
"""Publish the PRL-26-E Atomic headline, subsets, and identity-bound tool audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.coredev_results import (  # noqa: E402
    extract_coredev_macro_star,
    write_json_atomic,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    load_benchmark_tasks,
)


EVALUATION_ID = "PRL26-E-ATOMIC-CROP-TGVF-TRAIN512-S32-PIXEL512-COREDEV2511-SEED42-V1"
ARM_NAME = "step32"
ARM_EVALUATION_ID = (
    "PRL26-E-ATOMIC-CROP-TGVF-TRAIN512-S32-MATCHED-COREDEV2511-PIXEL512-V1"
)
OPTIMIZER_STEP = 32
WORLD_SIZE = 4
MAXIMUM_TOOL_CALLS = 6
IMAGE_MAX_PIXELS = 262_144
POLICY_SNAPSHOT_BACKEND = "full_model_trainable_rp66"
PLAN_PATH = (
    REPOSITORY_ROOT
    / "configs/evaluation/"
    "prl26_e_atomic_crop_tgvf_train512_s32_pixel512_coredev2511_plan.json"
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
OFFICIAL_DATASET_ROWS = {
    "VStarBench": 191,
    "HRBench4K": 200,
    "BLINK": 420,
    "OCRBench_v2": 600,
    "MMMU_Pro_10c": 300,
    "MathVista_MINI": 300,
    "MathVerse_MINI": 500,
}
POLICY_SHA256_FIELDS = (
    "evaluation_identity_sha256",
    "task_manifest_sha256",
    "policy_config_identity_sha256",
    "policy_run_identity_sha256",
    "policy_weights_sha256",
    "policy_paired_snapshot_identity_sha256",
    "policy_qwen_tree_sha256",
    "policy_rp66_state_sha256",
    "policy_rp66_storage_sha256",
)
TOOL_USAGE_DEFINITIONS = {
    "tool_attempt": (
        "one successful tool_calls record or one tool_errors record; equivalent "
        "to len(tool_calls) + len(tool_errors)"
    ),
    "successful_tool_call": (
        "one tool_calls record that executed and produced one visual observation"
    ),
    "tool_error": (
        "one attempted action rejected by parsing, dispatch, execution, or the "
        "six-call cap; it remains a tool attempt"
    ),
    "no_tool_trajectory": "a trajectory with zero tool attempts, including errors",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} must be a lowercase SHA256")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"JSON artifact is unavailable or malformed: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return payload


def _resolve_repository_file(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} path is malformed")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{name} path must be repository-relative")
    candidate = REPOSITORY_ROOT / relative
    if candidate.is_symlink():
        raise RuntimeError(f"{name} file cannot be a symlink")
    path = candidate.resolve()
    if (
        not path.is_relative_to(REPOSITORY_ROOT.resolve())
        or not path.is_file()
    ):
        raise RuntimeError(f"{name} file is unavailable")
    return path


def _load_plan_contract() -> dict[str, Any]:
    if PLAN_PATH.is_symlink() or not PLAN_PATH.is_file():
        raise RuntimeError("Atomic evaluation plan file is unavailable")
    plan = _read_json(PLAN_PATH)
    arms = plan.get("arms")
    protocol = plan.get("protocol")
    paired_rng = plan.get("paired_rng")
    if (
        plan.get("schema_version")
        != "tgvf.prl15-paired-policy-benchmark-plan.v2"
        or plan.get("status") != "ready"
        or plan.get("evaluation_id") != EVALUATION_ID
        or plan.get("evaluation_image_max_pixels") != IMAGE_MAX_PIXELS
        or plan.get("expected_task_count") != 2511
        or plan.get("expected_single_image_count") != 2240
        or plan.get("unsupported_multi_image_count") != 271
        or not isinstance(arms, list)
        or arms
        != [
            {
                "name": ARM_NAME,
                "optimizer_step": OPTIMIZER_STEP,
                "qwen_source": "output.root/permanent-checkpoints/global_step_32",
                "rp66_source": (
                    "output.root/runtime-policy-state/lora-manifests/"
                    "step-00000032-*.json"
                ),
                "evaluation_id": ARM_EVALUATION_ID,
            }
        ]
        or not isinstance(protocol, dict)
        or protocol.get("evaluation_protocol") != "training_run"
        or protocol.get("tool_profile") != "crop_tgvf"
        or protocol.get("maximum_tool_calls") != MAXIMUM_TOOL_CALLS
        or not isinstance(paired_rng, dict)
        or paired_rng.get("schema_version")
        != "tgvf.policy-paired-evaluation-rng-plan.v1"
        or paired_rng.get("mode") != "common_random_numbers_per_task_turn"
        or paired_rng.get("master_seed") != 42
        or paired_rng.get("temperature") != 1.0
        or paired_rng.get("do_sample") is not True
        or paired_rng.get("task_manifest_sha256")
        != plan.get("task_manifest_sha256")
    ):
        raise RuntimeError("Atomic evaluation plan contract differs")
    _require_sha256(
        plan.get("task_manifest_sha256"), name="Atomic task manifest identity"
    )
    _require_sha256(
        paired_rng.get("protocol_sha256"), name="Atomic paired RNG protocol"
    )
    config_path = _resolve_repository_file(
        plan.get("policy_config"), name="Atomic policy config"
    )
    planned_config_sha256 = _require_sha256(
        plan.get("policy_config_sha256"), name="Atomic planned policy config"
    )
    if _sha256(config_path) != planned_config_sha256:
        raise RuntimeError("Atomic planned/current policy config identity differs")
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError("Atomic policy config is unavailable or malformed") from error
    sampling = config.get("sampling")
    model = config.get("model")
    config_protocol = config.get("protocol")
    if (
        not isinstance(sampling, dict)
        or not isinstance(model, dict)
        or not isinstance(config_protocol, dict)
        or model.get("image_max_pixels") != IMAGE_MAX_PIXELS
        or config_protocol.get("tool_profile") != "crop_tgvf"
        or config_protocol.get("maximum_tool_calls") != MAXIMUM_TOOL_CALLS
        or sampling.get("temperature") != 1.0
        or sampling.get("do_sample") is not True
        or sampling.get("rollout_master_seed") != 42
    ):
        raise RuntimeError("Atomic policy config evaluation semantics differ")
    expected_sampling = {
        "source": "bound_policy_run_config",
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
        "top_k": sampling.get("top_k"),
        "min_p": sampling.get("min_p"),
        "do_sample": sampling.get("do_sample"),
        "paired_rng": paired_rng,
    }
    return {
        "plan": plan,
        "plan_path": PLAN_PATH.resolve(),
        "plan_sha256": _sha256(PLAN_PATH),
        "policy_config_path": config_path,
        "policy_config_sha256": planned_config_sha256,
        "expected_sampling": expected_sampling,
    }


def _load_expected_single_image_tasks(
    plan: dict[str, Any],
) -> tuple[dict[int, dict[str, object]], str]:
    manifest_value = plan.get("task_manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise RuntimeError("Atomic task manifest path is malformed")
    manifest_path = Path(manifest_value)
    if (
        not manifest_path.is_absolute()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or _sha256(manifest_path) != plan["task_manifest_sha256"]
    ):
        raise RuntimeError("Atomic task manifest file identity differs")
    tasks = load_benchmark_tasks(
        manifest_path,
        expected_task_count=2511,
        expected_single_image_count=2240,
        expected_sha256=plan["task_manifest_sha256"],
        verify_image_paths=False,
        verify_image_contents=False,
    )
    expected = {
        task.ordinal: {
            "sample_id": task.bound_sample_id,
            "dataset": task.dataset,
            "row_number": task.row_number,
            "index": task.index,
            "question": task.question,
            "image_paths": list(task.image_paths),
            "image_sha256s": list(task.image_sha256s),
            "image_dimensions": [list(value) for value in task.image_dimensions],
        }
        for task in tasks
        if task.single_image
    }
    if len(expected) != 2240:
        raise RuntimeError("Atomic task manifest single-image selection differs")
    sequence = [
        {"ordinal": ordinal, **task}
        for ordinal, task in sorted(expected.items())
    ]
    return expected, _canonical_sha256(sequence)


def _validate_runtime_sampling_rng(
    value: object, *, planned: dict[str, Any]
) -> dict[str, Any]:
    expected_seed_components = [
        "master_seed",
        "seed_namespace",
        "task_manifest_sha256",
        "protocol_sha256",
        "sample_id",
        "rollout_index",
        "assistant_turn_index",
    ]
    if (
        not isinstance(value, dict)
        or value.get("schema_version")
        != "tgvf-policy-paired-evaluation-rng-v1"
        or value.get("mode") != planned.get("mode")
        or value.get("seed_namespace") != planned.get("seed_namespace")
        or value.get("master_seed") != planned.get("master_seed")
        or value.get("task_manifest_sha256")
        != planned.get("task_manifest_sha256")
        or value.get("protocol_sha256") != planned.get("protocol_sha256")
        or value.get("excluded_arm_components")
        != planned.get("excluded_arm_components")
        or value.get("seed_components") != expected_seed_components
    ):
        raise RuntimeError("Atomic runtime paired RNG contract differs")
    return value


def _identity_closure(
    rows: list[dict[str, Any]],
    *,
    expected_evaluation_id: str,
    expected_task_manifest_sha256: str,
    expected_optimizer_step: int = OPTIMIZER_STEP,
    expected_world_size: int = WORLD_SIZE,
    planned_paired_rng: dict[str, Any] | None = None,
    expected_tasks: dict[int, dict[str, object]] | None = None,
    expected_task_sequence_sha256: str | None = None,
) -> dict[str, object]:
    if not rows:
        raise RuntimeError("Atomic identity closure cannot be empty")
    ordered: list[tuple[int, dict[str, Any], str, str, str]] = []
    seen_ordinals: set[int] = set()
    seen_sample_ids: set[str] = set()
    seen_result_identities: set[str] = set()
    seen_stream_identities: set[str] = set()
    common_values: dict[str, object] = {}
    for row in rows:
        ordinal = row.get("ordinal")
        sample_id = row.get("sample_id")
        rank = row.get("rank")
        if type(ordinal) is not int or ordinal < 0 or ordinal >= 2511:
            raise RuntimeError("Atomic result ordinal is malformed")
        if ordinal in seen_ordinals:
            raise RuntimeError("Atomic result ordinal is duplicated")
        if not isinstance(sample_id, str) or not sample_id:
            raise RuntimeError("Atomic result sample identity is malformed")
        if sample_id in seen_sample_ids:
            raise RuntimeError("Atomic result sample identity is duplicated")
        if (
            row.get("evaluation_id") != expected_evaluation_id
            or row.get("optimizer_step") != expected_optimizer_step
            or row.get("world_size") != expected_world_size
            or type(rank) is not int
            or rank != ordinal % expected_world_size
            or row.get("rollout_index") != 0
            or row.get("policy_snapshot_backend") != POLICY_SNAPSHOT_BACKEND
        ):
            raise RuntimeError("Atomic result arm/rank identity differs")
        if expected_tasks is not None:
            expected_task = expected_tasks.get(ordinal)
            observed_task = {
                "sample_id": sample_id,
                "dataset": row.get("dataset"),
                "row_number": row.get("row_number"),
                "index": row.get("index"),
                "question": row.get("question"),
                "image_paths": row.get("image_paths"),
                "image_sha256s": row.get("image_sha256s"),
                "image_dimensions": row.get("image_dimensions"),
            }
            if expected_task is None or observed_task != expected_task:
                raise RuntimeError("Atomic result task differs from bound manifest")
        expected_result_identity = _require_sha256(
            row.get("result_identity_sha256"), name="Atomic result identity"
        )
        hash_payload = dict(row)
        hash_payload.pop("result_identity_sha256", None)
        hash_payload.pop("wall_seconds", None)
        if _canonical_sha256(hash_payload) != expected_result_identity:
            raise RuntimeError("Atomic result identity digest differs")
        stream_identity = _require_sha256(
            row.get("paired_rng_stream_identity_sha256"),
            name="Atomic paired RNG stream identity",
        )
        if expected_result_identity in seen_result_identities:
            raise RuntimeError("Atomic result identity is duplicated")
        if stream_identity in seen_stream_identities:
            raise RuntimeError("Atomic paired RNG stream identity is duplicated")
        for field in POLICY_SHA256_FIELDS:
            value = _require_sha256(row.get(field), name=f"Atomic {field}")
            if field in common_values and common_values[field] != value:
                raise RuntimeError(f"Atomic result {field} differs across rows")
            common_values[field] = value
        if row["task_manifest_sha256"] != expected_task_manifest_sha256:
            raise RuntimeError("Atomic result task manifest differs from plan")
        for field in ("policy_run_id", "model_identity", "sampling_rng"):
            value = row.get(field)
            if field == "policy_run_id":
                if not isinstance(value, str) or not value:
                    raise RuntimeError("Atomic policy run ID is malformed")
            elif not isinstance(value, dict):
                raise RuntimeError(f"Atomic result {field} is malformed")
            canonical_value = _canonical_sha256(value)
            if field in common_values and common_values[field] != canonical_value:
                raise RuntimeError(f"Atomic result {field} differs across rows")
            common_values[field] = canonical_value
        if planned_paired_rng is not None:
            _validate_runtime_sampling_rng(
                row["sampling_rng"], planned=planned_paired_rng
            )
        seen_ordinals.add(ordinal)
        seen_sample_ids.add(sample_id)
        seen_result_identities.add(expected_result_identity)
        seen_stream_identities.add(stream_identity)
        ordered.append(
            (
                ordinal,
                row,
                sample_id,
                expected_result_identity,
                stream_identity,
            )
        )
    if (
        common_values["policy_config_identity_sha256"]
        != common_values["policy_run_identity_sha256"]
    ):
        raise RuntimeError("Atomic checkpoint owner/protocol run identities differ")
    if expected_tasks is not None and seen_ordinals != set(expected_tasks):
        raise RuntimeError("Atomic result task selection differs from bound manifest")
    ordered.sort(key=lambda item: item[0])
    ordinal_sequence = [item[0] for item in ordered]
    sample_sequence = [
        {"ordinal": ordinal, "sample_id": sample_id}
        for ordinal, _row, sample_id, _result, _stream in ordered
    ]
    result_sequence = [
        {"ordinal": ordinal, "result_identity_sha256": result_identity}
        for ordinal, _row, _sample, result_identity, _stream in ordered
    ]
    stream_sequence = [
        {
            "ordinal": ordinal,
            "sample_id": sample_id,
            "paired_rng_stream_identity_sha256": stream_identity,
        }
        for ordinal, _row, sample_id, _result, stream_identity in ordered
    ]
    return {
        "result_row_count": len(ordered),
        "ordinal_count": len(seen_ordinals),
        "ordinal_sequence_sha256": _canonical_sha256(ordinal_sequence),
        "sample_id_count": len(seen_sample_ids),
        "sample_id_sequence_sha256": _canonical_sha256(sample_sequence),
        "manifest_single_image_task_sequence_sha256": (
            _require_sha256(
                expected_task_sequence_sha256,
                name="Atomic manifest single-image task sequence",
            )
            if expected_tasks is not None
            else None
        ),
        "result_identity_count": len(seen_result_identities),
        "result_identity_sequence_sha256": _canonical_sha256(result_sequence),
        "paired_rng_stream_count": len(seen_stream_identities),
        "paired_rng_stream_sequence_sha256": _canonical_sha256(stream_sequence),
        "evaluation_id": expected_evaluation_id,
        "evaluation_identity_sha256": common_values[
            "evaluation_identity_sha256"
        ],
        "task_manifest_sha256": common_values["task_manifest_sha256"],
        "policy_run_id": rows[0]["policy_run_id"],
        "policy_run_identity_sha256": common_values[
            "policy_run_identity_sha256"
        ],
        "policy_config_identity_sha256": common_values[
            "policy_config_identity_sha256"
        ],
        "policy_weights_sha256": common_values["policy_weights_sha256"],
        "policy_paired_snapshot_identity_sha256": common_values[
            "policy_paired_snapshot_identity_sha256"
        ],
        "policy_qwen_tree_sha256": common_values["policy_qwen_tree_sha256"],
        "policy_rp66_state_sha256": common_values["policy_rp66_state_sha256"],
        "policy_rp66_storage_sha256": common_values[
            "policy_rp66_storage_sha256"
        ],
        "model_identity_sha256": common_values["model_identity"],
        "sampling_rng_sha256": common_values["sampling_rng"],
        "optimizer_step": expected_optimizer_step,
        "rank_count": expected_world_size,
    }


def _read_rows(
    inference_root: Path, *, plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    expected_tasks, expected_task_sequence_sha256 = (
        _load_expected_single_image_tasks(plan)
    )
    expected_paths = [inference_root / f"rank-{rank}.jsonl" for rank in range(4)]
    observed_paths = sorted(inference_root.glob("rank-*.jsonl"))
    if observed_paths != expected_paths:
        raise RuntimeError("Atomic inference rank coverage differs")
    rows: list[dict[str, Any]] = []
    for rank, path in enumerate(expected_paths):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Atomic inference rank artifact differs")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"malformed Atomic inference row: {path}:{line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise RuntimeError("Atomic inference row must be an object")
                if row.get("rank") != rank:
                    raise RuntimeError("Atomic inference row is stored under wrong rank")
                rows.append(row)
    if len(rows) != 2240:
        raise RuntimeError(f"Atomic inference row count differs: {len(rows)}")
    observed = Counter(row.get("dataset") for row in rows)
    if observed != Counter(DATASET_ROWS):
        raise RuntimeError(
            f"Atomic inference dataset coverage differs: {dict(observed)}"
        )
    closure = _identity_closure(
        rows,
        expected_evaluation_id=ARM_EVALUATION_ID,
        expected_task_manifest_sha256=plan["task_manifest_sha256"],
        planned_paired_rng=plan["paired_rng"],
        expected_tasks=expected_tasks,
        expected_task_sequence_sha256=expected_task_sequence_sha256,
    )
    if any(closure[field] != 2240 for field in (
        "result_row_count",
        "ordinal_count",
        "sample_id_count",
        "result_identity_count",
        "paired_rng_stream_count",
    )):
        raise RuntimeError("Atomic inference identity closure is incomplete")
    return rows, closure


def _percentile(values: Iterable[int], quantile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _usage(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        raise RuntimeError("Atomic trajectory usage audit cannot be empty")
    total_attempts = 0
    total_successful_calls = 0
    trajectories_attempting = 0
    trajectories_with_success = 0
    trajectories_with_repeat_success = 0
    trajectories_with_error = 0
    repeat_successful_calls = 0
    successful_observations = 0
    tool_errors = 0
    unnamed_error_functions = 0
    token_counts: list[int] = []
    stop_counts: Counter[str] = Counter()
    result_kind_counts: Counter[str] = Counter()
    successful_function_counts: Counter[str] = Counter()
    attempted_function_counts: Counter[str] = Counter()
    error_function_counts: Counter[str] = Counter()
    error_code_counts: Counter[str] = Counter()
    for row in rows:
        calls = row.get("tool_calls")
        turns = row.get("assistant_turns")
        errors = row.get("tool_errors")
        observations = row.get("successful_observation_count")
        stop = row.get("stop")
        result_kind = row.get("result_kind", "trajectory")
        if (
            not isinstance(calls, list)
            or not isinstance(turns, list)
            or not isinstance(errors, list)
            or type(observations) is not int
            or observations < 0
            or not isinstance(stop, str)
            or not stop
            or result_kind not in {"trajectory", "sample_local_failure"}
        ):
            raise RuntimeError("Atomic trajectory audit structure differs")
        call_count = len(calls)
        error_count = len(errors)
        attempt_count = call_count + error_count
        if (
            observations != call_count
            or call_count > MAXIMUM_TOOL_CALLS
            or attempt_count > MAXIMUM_TOOL_CALLS + 1
        ):
            raise RuntimeError("Atomic tool success/attempt counts differ")
        total_attempts += attempt_count
        total_successful_calls += call_count
        trajectories_attempting += int(attempt_count > 0)
        trajectories_with_success += int(call_count > 0)
        trajectories_with_repeat_success += int(call_count > 1)
        trajectories_with_error += int(error_count > 0)
        repeat_successful_calls += max(0, call_count - 1)
        successful_observations += observations
        tool_errors += error_count
        stop_counts[stop] += 1
        result_kind_counts[result_kind] += 1
        generated_tokens = 0
        for turn_index, turn in enumerate(turns):
            if (
                not isinstance(turn, dict)
                or turn.get("turn_index") != turn_index
                or type(turn.get("sampled_token_count")) is not int
                or turn["sampled_token_count"] < 0
            ):
                raise RuntimeError("Atomic assistant-turn token audit differs")
            generated_tokens += turn["sampled_token_count"]
        token_counts.append(generated_tokens)
        event_turn_indices: list[int] = []
        for call_index, call in enumerate(calls):
            function_name = (
                call.get("function_name") if isinstance(call, dict) else None
            )
            assistant_turn_index = (
                call.get("assistant_turn_index") if isinstance(call, dict) else None
            )
            if (
                not isinstance(function_name, str)
                or not function_name
                or call.get("call_index") != call_index
                or type(assistant_turn_index) is not int
                or not 0 <= assistant_turn_index < len(turns)
            ):
                raise RuntimeError("Atomic tool-call audit identity differs")
            successful_function_counts[function_name] += 1
            attempted_function_counts[function_name] += 1
            event_turn_indices.append(assistant_turn_index)
        error_attempt_indices: list[int] = []
        for error in errors:
            if not isinstance(error, dict):
                raise RuntimeError("Atomic tool-error audit must be an object")
            attempt_index = error.get("attempt_index")
            assistant_turn_index = error.get("assistant_turn_index")
            code = error.get("code")
            function_name = error.get("function_name")
            if (
                type(attempt_index) is not int
                or not 0 <= attempt_index < attempt_count
                or type(assistant_turn_index) is not int
                or not 0 <= assistant_turn_index < len(turns)
                or not isinstance(code, str)
                or not code
                or not isinstance(error.get("payload_json"), str)
                or not error["payload_json"]
                or type(error.get("recoverable")) is not bool
                or (
                    function_name is not None
                    and (not isinstance(function_name, str) or not function_name)
                )
            ):
                raise RuntimeError("Atomic tool-error audit identity differs")
            error_attempt_indices.append(attempt_index)
            event_turn_indices.append(assistant_turn_index)
            error_code_counts[code] += 1
            if function_name is None:
                unnamed_error_functions += 1
            else:
                error_function_counts[function_name] += 1
                attempted_function_counts[function_name] += 1
        if (
            error_attempt_indices != sorted(error_attempt_indices)
            or len(set(error_attempt_indices)) != len(error_attempt_indices)
            or len(set(event_turn_indices)) != len(event_turn_indices)
        ):
            raise RuntimeError("Atomic tool-attempt ordering differs")
    count = len(rows)
    return {
        "trajectory_count": count,
        "no_tool_trajectory_count": count - trajectories_attempting,
        "trajectories_attempting_tool": trajectories_attempting,
        "tool_attempt_trajectory_rate": trajectories_attempting / count,
        "total_tool_attempts": total_attempts,
        "mean_tool_attempts_per_trajectory": total_attempts / count,
        "mean_tool_attempts_when_attempted": (
            total_attempts / trajectories_attempting
            if trajectories_attempting
            else 0.0
        ),
        "trajectories_with_successful_tool_call": trajectories_with_success,
        "successful_tool_call_trajectory_rate": trajectories_with_success / count,
        "successful_tool_call_count": total_successful_calls,
        "trajectories_with_repeat_successful_tool_call": (
            trajectories_with_repeat_success
        ),
        "repeat_successful_tool_call_trajectory_rate": (
            trajectories_with_repeat_success / count
        ),
        "repeat_successful_tool_call_count": repeat_successful_calls,
        "mean_successful_tool_calls_per_trajectory": (
            total_successful_calls / count
        ),
        "mean_successful_tool_calls_when_present": (
            total_successful_calls / trajectories_with_success
            if trajectories_with_success
            else 0.0
        ),
        "successful_tool_attempt_rate": (
            total_successful_calls / total_attempts if total_attempts else 0.0
        ),
        "successful_observation_count": successful_observations,
        "mean_successful_observations_per_trajectory": (
            successful_observations / count
        ),
        "trajectories_with_tool_error": trajectories_with_error,
        "tool_error_trajectory_rate": trajectories_with_error / count,
        "tool_error_count": tool_errors,
        "mean_tool_errors_per_trajectory": tool_errors / count,
        "tool_error_code_counts": dict(sorted(error_code_counts.items())),
        "tool_error_function_counts": dict(sorted(error_function_counts.items())),
        "tool_errors_without_function_name": unnamed_error_functions,
        "successful_function_call_counts": dict(
            sorted(successful_function_counts.items())
        ),
        "attempted_named_function_counts": dict(
            sorted(attempted_function_counts.items())
        ),
        "generated_token_mean": sum(token_counts) / count,
        "generated_token_p50": _percentile(token_counts, 0.50),
        "generated_token_p95": _percentile(token_counts, 0.95),
        "generated_token_p99": _percentile(token_counts, 0.99),
        "stop_counts": dict(sorted(stop_counts.items())),
        "result_kind_counts": dict(sorted(result_kind_counts.items())),
    }


def _validate_official_summary(summary: dict[str, Any]) -> None:
    slices = summary.get("slices")
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "pass"
        or summary.get("phase") != "eval"
        or summary.get("sample_count") != 2511
        or summary.get("slice_count") != 7
        or not isinstance(slices, list)
        or len(slices) != 7
    ):
        raise RuntimeError("Atomic official CoreDev summary is incomplete")
    observed: dict[str, int] = {}
    for item in slices:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("dataset"), str)
            or type(item.get("sample_count")) is not int
            or item["dataset"] in observed
        ):
            raise RuntimeError("Atomic official CoreDev slice identity differs")
        observed[item["dataset"]] = item["sample_count"]
    if observed != OFFICIAL_DATASET_ROWS:
        raise RuntimeError("Atomic official CoreDev subset coverage differs")


def _validate_paired_summary(
    paired: dict[str, Any],
    *,
    summary: dict[str, Any],
    closure: dict[str, object],
    plan_contract: dict[str, Any],
) -> None:
    expected_coverage = {
        "official_manifest_rows": 2511,
        "evaluated_single_image_rows": 2240,
        "held_multi_image_rows": 271,
        "multi_image_policy": "unsupported_explicit_hold",
    }
    expected_top_level_fields = {
        "schema_version",
        "evaluation_id",
        "coverage",
        "materialization",
        "sampling",
        "arms",
        ARM_NAME,
    }
    arms = paired.get("arms")
    materialization = paired.get("materialization")
    if (
        set(paired) != expected_top_level_fields
        or paired.get("schema_version")
        != "tgvf.prl15-paired-coredev-summary.v1"
        or paired.get("evaluation_id") != EVALUATION_ID
        or paired.get("coverage") != expected_coverage
        or paired.get("sampling") != plan_contract["expected_sampling"]
        or not isinstance(arms, dict)
        or set(arms) != {ARM_NAME}
        or not isinstance(materialization, dict)
        or set(materialization) != {ARM_NAME}
        or paired.get(ARM_NAME) != summary
    ):
        raise RuntimeError("Atomic paired summary identity differs")
    arm = arms[ARM_NAME]
    if (
        not isinstance(arm, dict)
        or set(arm)
        != {
            "optimizer_step",
            "evaluation_identity_sha256",
            "official_summary",
        }
        or arm.get("optimizer_step") != OPTIMIZER_STEP
        or arm.get("evaluation_identity_sha256")
        != closure["evaluation_identity_sha256"]
        or arm.get("official_summary") != summary
    ):
        raise RuntimeError("Atomic paired arm identity differs")


def summarize(root: Path, output: Path) -> dict[str, object]:
    plan_contract = _load_plan_contract()
    plan = plan_contract["plan"]
    paired_summary_path = root / "paired-summary.json"
    if paired_summary_path.is_symlink() or not paired_summary_path.is_file():
        raise RuntimeError("Atomic paired summary file boundary differs")
    paired = _read_json(paired_summary_path)
    summary_path = (
        root / "step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json"
    )
    if summary_path.is_symlink() or not summary_path.is_file():
        raise RuntimeError("Atomic official summary file boundary differs")
    summary = _read_json(summary_path)
    _validate_official_summary(summary)
    rows, closure = _read_rows(root / "step32/inference", plan=plan)
    _validate_paired_summary(
        paired,
        summary=summary,
        closure=closure,
        plan_contract=plan_contract,
    )
    by_dataset = {
        dataset: _usage([row for row in rows if row["dataset"] == dataset])
        for dataset in DATASET_ROWS
    }
    headline = extract_coredev_macro_star(summary)
    payload: dict[str, object] = {
        "schema_version": "tgvf.prl26-e-atomic-s32-results.v2",
        "status": "pass",
        "evaluation_id": EVALUATION_ID,
        "contract": "independent fresh-S0 Train@512 S32; matched Eval@512",
        "coverage": {
            "official_manifest_rows": 2511,
            "evaluated_single_image_rows": 2240,
            "held_multi_image_rows": 271,
            "subset_count": 7,
        },
        "plan_identity": {
            "path": str(plan_contract["plan_path"]),
            "sha256": plan_contract["plan_sha256"],
            "policy_config_path": str(plan_contract["policy_config_path"]),
            "policy_config_sha256": plan_contract["policy_config_sha256"],
            "task_manifest_sha256": plan["task_manifest_sha256"],
            "paired_rng_protocol_sha256": plan["paired_rng"]["protocol_sha256"],
        },
        "tool_usage_definitions": TOOL_USAGE_DEFINITIONS,
        "arm": {
            "method": "Atomic Crop+TGVF Train@512 S32",
            "optimizer_step": OPTIMIZER_STEP,
            "train_image_max_pixels": IMAGE_MAX_PIXELS,
            "evaluation_image_max_pixels": IMAGE_MAX_PIXELS,
            "macro_star_percent": headline["macro_star_percent"],
            "headline": headline,
            "seven_subset_statistics": summary["slices"],
            "tool_usage_overall": _usage(rows),
            "tool_usage_by_subset": by_dataset,
            "identity_closure": closure,
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
        },
        "paired_summary_path": str(paired_summary_path.resolve()),
        "paired_summary_sha256": _sha256(paired_summary_path),
        "paired_materialization_sha256": _canonical_sha256(
            paired["materialization"]
        ),
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
