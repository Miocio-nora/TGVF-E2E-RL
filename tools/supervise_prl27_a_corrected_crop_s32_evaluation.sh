#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
training_root="$main_root/artifacts/policy/PRL-27-A-train512-s32-crop-exact-continuation-qwen3-instruct-bs16-n16-teacher25-ws8"
receipt="$training_root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"
training_control_root="$main_root/artifacts/control/PRL-27-A-crop-train512-s32-20260829"
training_complete="$training_control_root/state/s32-accepted"
training_failed="$training_control_root/state/failed"
evaluation_id=PRL27-A-CROP-EXACT-CONTINUATION-TRAIN512-S32-MATCHED-COREDEV2511-PIXEL512-V1
eval_root="$training_root/evaluation/$evaluation_id"
plan="$eval_root/runtime/bound-crop-plan.json"
handoff="$eval_root/runtime/bound-handoff.json"
config="$eval_root/step32/benchmark-config.json"
validation="$eval_root/step32/logs/prl27-a-pixel512-static-validation.json"
proof="$eval_root/step32/runtime/pixel512-processor-proof.json"
summary="$eval_root/step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json"
paired_summary="$eval_root/paired-summary.json"
runner_complete="$eval_root/evaluation-complete"
result="$eval_root/corrected-crop-s32-pixel512-results.json"

# Control state deliberately lives outside the not-yet-authorized evaluation
# root.  The evaluation tree is created only after the permanent S32 receipt
# exists and the trainer has released every GPU and Ray process twice.
control_root="$main_root/artifacts/control/PRL-27-A-corrected-crop-s32-eval512-20260829"
runtime_root="$control_root/runtime"
log_root="$control_root/logs"
state_root="$control_root/state"
supervisor_complete="$state_root/evaluation-complete"

binder="$repo_root/tools/bind_prl27_a_corrected_crop_training_run_evaluation.py"
runner="$repo_root/tools/run_prl15_paired_evaluation.py"
benchmark_runner="$repo_root/tools/run_policy_benchmark.py"
proof_validator="$repo_root/tools/validate_prl26_train512_processor_proof.py"
resource_validator="$repo_root/tools/validate_prl26_train512_training_handoff.py"

for path in "$python_bin" "$binder" "$runner" "$benchmark_runner" \
  "$proof_validator" "$resource_validator"; do
  if [[ ! -f "$path" ]]; then
    echo "required PRL-27-A evaluation file is absent: $path" >&2
    exit 1
  fi
done

poll_seconds=${PRL27_A_POLL_SECONDS:-30}
release_stable_polls=${PRL27_A_RELEASE_STABLE_POLLS:-2}
release_maximum_polls=${PRL27_A_RELEASE_MAXIMUM_POLLS:-240}
gpu_memory_threshold_mib=${PRL27_A_GPU_IDLE_MEMORY_THRESHOLD_MIB:-32}
admitted_head=${PRL27_A_ADMITTED_HEAD:-}
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$release_stable_polls" =~ ^[1-9][0-9]*$ \
      || ! "$release_maximum_polls" =~ ^[1-9][0-9]*$ \
      || ! "$gpu_memory_threshold_mib" =~ ^[0-9]+$ ]]; then
  echo "PRL-27-A evaluation polling setting is malformed" >&2
  exit 1
fi
if (( release_stable_polls < 2 )); then
  echo "PRL-27-A evaluation requires two consecutive clean resource probes" >&2
  exit 1
fi
if [[ ! "$admitted_head" =~ ^[0-9a-f]{40}$ ]]; then
  echo "PRL27_A_ADMITTED_HEAD is required for evaluation" >&2
  exit 1
fi

validate_worktree() {
  local observed_root observed_head dirty
  observed_root=$(git -C "$repo_root" rev-parse --show-toplevel)
  observed_head=$(git -C "$repo_root" rev-parse HEAD)
  dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
  if [[ "$observed_root" != "$repo_root" \
        || "$observed_head" != "$admitted_head" \
        || -n "$dirty" ]]; then
    echo "clean PRL-27-A evaluation worktree identity differs" >&2
    exit 1
  fi
}

validate_worktree

mkdir -p "$runtime_root" "$log_root" "$state_root"
exec 9>"$runtime_root/supervisor.lock"
flock -n 9 || {
  echo "PRL-27-A corrected Crop evaluator is already active" >&2
  exit 1
}
if [[ -L "$control_root/admitted-head.txt" ]]; then
  echo "PRL-27-A evaluation admitted HEAD cannot be a symlink" >&2
  exit 1
fi
if [[ -e "$control_root/admitted-head.txt" ]]; then
  if [[ ! -s "$control_root/admitted-head.txt" \
        || "$(tr -d '\r\n' <"$control_root/admitted-head.txt")" != "$admitted_head" ]]; then
    echo "PRL-27-A evaluation control state belongs to another HEAD" >&2
    exit 1
  fi
else
  printf '%s\n' "$admitted_head" >"$control_root/admitted-head.txt"
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo_root/src:$main_root/.deps/verl${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_REPOSITORY_ROOT="$repo_root"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

phase=waiting_for_s32_training_acceptance
active_pid=

timestamp() {
  date '+%F %T %Z'
}

record_phase() {
  printf 'phase=%s\ntime=%s\n' "$phase" "$(timestamp)" \
    >"$state_root/current-phase"
}

stop_process_group() {
  local pid=${1:-}
  [[ -n "$pid" ]] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$pid" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local status=$?
  set +e
  stop_process_group "$active_pid"
  if (( status == 0 )); then
    rm -f "$state_root/failed"
  else
    rm -f "$supervisor_complete"
    printf 'status=failed\nphase=%s\ntime=%s\nexit_status=%s\n' \
      "$phase" "$(timestamp)" "$status" >"$state_root/failed"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'phase=signal; record_phase; exit 130' INT TERM
rm -f "$state_root/failed" "$supervisor_complete"
record_phase

run_group() {
  local log_path=$1
  shift
  setsid "$@" >"$log_path" 2>&1 9>&- &
  active_pid=$!
  set +e
  wait "$active_pid"
  local status=$?
  set -e
  active_pid=
  return "$status"
}

while [[ ! -f "$training_complete" || ! -s "$receipt" ]]; do
  if [[ -L "$training_complete" || -L "$receipt" ]]; then
    echo "PRL-27-A training completion boundary cannot be a symlink" >&2
    exit 1
  fi
  if [[ -e "$training_failed" || -L "$training_failed" ]]; then
    echo "PRL-27-A training supervisor failed before accepted S32" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done
if [[ -L "$training_complete" || ! -f "$training_complete" \
      || -L "$receipt" || ! -s "$receipt" \
      || -e "$training_failed" || -L "$training_failed" ]]; then
  echo "PRL-27-A accepted S32 boundary changed during evaluation admission" >&2
  exit 1
fi

# A permanent checkpoint receipt can precede the trainer's final process exit.
# Preserve every probe and require two consecutive clean all-GPU/Ray samples.
phase=waiting_for_training_resource_release
record_phase
quiet=0
total=0
while (( quiet < release_stable_polls )); do
  if [[ -e "$training_failed" || -L "$training_failed" ]]; then
    echo "PRL-27-A training failure appeared during resource admission" >&2
    exit 1
  fi
  total=$((total + 1))
  probe="$runtime_root/resource-probe-${total}.json"
  if "$python_bin" "$resource_validator" resources-free \
      --memory-threshold-mib "$gpu_memory_threshold_mib" >"$probe"; then
    quiet=$((quiet + 1))
  else
    quiet=0
  fi
  if (( quiet < release_stable_polls )); then
    if (( total >= release_maximum_polls )); then
      echo "GPUs or Ray did not become clean before PRL-27-A evaluation" >&2
      exit 1
    fi
    sleep "$poll_seconds"
  fi
done
cp "$probe" "$runtime_root/resources-released.json"
validate_worktree

# The binder is the first operation allowed to create the PRL-27-A evaluation
# root.  It revalidates the complete S32 receipt/metrics and binds the exact
# training-run continuation, action boundary, response budget, and RNG identity.
phase=binding_corrected_crop_s32
record_phase
mkdir -p "$eval_root/logs"
env CUDA_VISIBLE_DEVICES= "$python_bin" "$binder" \
  --crop-plan-output "$plan" --handoff-output "$handoff" \
  >"$log_root/bind-handoff.json" 2>"$log_root/bind-handoff.stderr.log"

phase=preparing_corrected_crop_full_model
record_phase
validate_worktree
run_group "$log_root/prepare.log" \
  "$python_bin" "$runner" --plan "$plan" --mode prepare \
  --output-root "$eval_root" --gpu-ids 0 1 2 3

phase=proving_pixel512_and_exact_continuation
record_phase
validate_worktree
mkdir -p "$(dirname "$validation")" "$(dirname "$proof")"
env CUDA_VISIBLE_DEVICES= "$python_bin" "$benchmark_runner" \
  --config "$config" --mode validate --world-size 4 \
  >"$validation" 2>"$eval_root/logs/prl27-a-pixel512-static-validation.stderr.log"
env CUDA_VISIBLE_DEVICES= "$python_bin" "$proof_validator" \
  --arm crop --config "$config" --validation-json "$validation" \
  --output "$proof" >"$log_root/pixel512-exact-continuation-proof.json"
touch "$state_root/processor-proof-complete"

phase=running_corrected_crop_four_gpu_inference
record_phase
validate_worktree
run_group "$log_root/inference.log" \
  "$python_bin" "$runner" --plan "$plan" --mode infer \
  --output-root "$eval_root" --gpu-ids 0 1 2 3
touch "$state_root/inference-complete"

phase=scoring_corrected_crop_seven_subsets
record_phase
validate_worktree
run_group "$log_root/scoring.log" \
  "$python_bin" "$runner" --plan "$plan" --mode score \
  --output-root "$eval_root" --gpu-ids 0 1 2 3

phase=publishing_corrected_crop_result_and_tool_usage
record_phase
validate_worktree
env CUDA_VISIBLE_DEVICES= PYTHONPATH="$repo_root/src" "$python_bin" - \
  "$evaluation_id" "$plan" "$handoff" "$proof" "$paired_summary" \
  "$summary" "$eval_root/step32/inference" "$result" \
  >"$log_root/result-table.json" <<'PY'
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

from tgvf_rl.evaluation.coredev_results import (
    extract_coredev_macro_star,
    write_json_atomic,
)


(
    evaluation_id,
    plan_path,
    handoff_path,
    proof_path,
    paired_path,
    summary_path,
    inference_root,
    output_path,
) = (sys.argv[1], *(Path(value).resolve() for value in sys.argv[2:]))

PIXEL512 = 262_144
STEP = 32
EXPECTED_INFERENCE_ROWS = {
    "VStarBench": 191,
    "HRBench4K": 200,
    "BLINK": 180,
    "OCRBench_v2": 600,
    "MMMU_Pro_10c": 269,
    "MathVista_MINI": 300,
    "MathVerse_MINI": 500,
}
EXPECTED_OFFICIAL_ROWS = {
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
ENVIRONMENT_SHA256 = (
    "f745fa6cfcc3ba9eb27125a49581fd823fb5930b7b0a51b28e51982999fa2d0a"
)


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON boundary differs: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def require_self_identity(payload: dict[str, Any], *, name: str) -> str:
    identity = payload.get("identity_sha256")
    content = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if identity != canonical_sha256(content):
        raise RuntimeError(f"{name} self identity differs")
    return identity


plan = read_json(plan_path)
handoff = read_json(handoff_path)
proof = read_json(proof_path)
paired = read_json(paired_path)
summary = read_json(summary_path)

if (
    plan.get("schema_version") != "tgvf.paired-policy-benchmark-plan.v3"
    or plan.get("evaluation_id") != evaluation_id
    or plan.get("evaluation_image_max_pixels") != PIXEL512
    or plan.get("expected_task_count") != 2511
    or plan.get("expected_single_image_count") != 2240
    or plan.get("unsupported_multi_image_count") != 271
    or plan.get("arms")
    != [
        {
            "name": "step32",
            "optimizer_step": STEP,
            "evaluation_id": evaluation_id,
            "source": {
                "kind": "owner_checkpoint",
                "relative_path": "permanent-checkpoints/global_step_32",
            },
        }
    ]
):
    raise RuntimeError("corrected Crop bound plan identity differs")
protocol = plan.get("protocol")
training_identity = protocol.get("training_run_identity") if isinstance(protocol, dict) else None
if (
    not isinstance(protocol, dict)
    or protocol.get("evaluation_protocol") != "training_run"
    or protocol.get("action_boundary")
    != {
        "stop_strings": ["</tool_call>"],
        "stop_token_ids": [151645],
        "include_stop_str_in_output": True,
        "ignore_eos": False,
    }
    or not isinstance(training_identity, dict)
    or training_identity.get("success_environment_renderer")
    != "render_qwen_native_matched_crop_success_environment_text"
    or training_identity.get("success_environment_text_sha256")
    != ENVIRONMENT_SHA256
    or training_identity.get("response_budget_scope") != "total_response_tokens"
    or training_identity.get("single_response_max_tokens") != 10240
    or training_identity.get("cap_error_behavior") != "one_final_answer_turn"
    or plan.get("paired_rng", {}).get("protocol_sha256")
    != canonical_sha256(training_identity)
):
    raise RuntimeError("corrected Crop training-run protocol differs")

handoff_identity = require_self_identity(handoff, name="handoff")
handoff_crop = handoff.get("crop")
if (
    handoff.get("status") != "ready"
    or handoff.get("evaluation_id") != evaluation_id
    or handoff.get("train_image_max_pixels") != PIXEL512
    or handoff.get("evaluation_image_max_pixels") != PIXEL512
    or handoff.get("optimizer_step") != STEP
    or not isinstance(handoff_crop, dict)
    or handoff_crop.get("evaluation_protocol") != "training_run"
    or handoff_crop.get("protocol_sha256")
    != plan["paired_rng"]["protocol_sha256"]
    or handoff_crop.get("bound_plan_file_sha256") != sha256(plan_path)
):
    raise RuntimeError("corrected Crop handoff identity differs")

proof_identity = require_self_identity(proof, name="processor proof")
proof_protocol = proof.get("protocol")
dynamic_proof = proof.get("proof")
if (
    proof.get("schema_version") != "tgvf.prl26-train512-processor-proof.v1"
    or proof.get("arm") != "crop"
    or proof.get("evaluation_id") != evaluation_id
    or proof.get("optimizer_step") != STEP
    or proof.get("train_image_max_pixels") != PIXEL512
    or proof.get("evaluation_image_max_pixels") != PIXEL512
    or not isinstance(proof_protocol, dict)
    or proof_protocol.get("continuation_parity") is not True
    or proof_protocol.get("success_environment_text_sha256")
    != ENVIRONMENT_SHA256
    or not isinstance(dynamic_proof, dict)
    or dynamic_proof.get("continuation_environment_token_count") != 60
    or dynamic_proof.get("success_environment_renderer")
    != "render_qwen_native_matched_crop_success_environment_text"
):
    raise RuntimeError("corrected Crop real-processor proof differs")

if (
    summary.get("schema_version") != 1
    or summary.get("status") != "pass"
    or summary.get("phase") != "eval"
    or summary.get("sample_count") != 2511
    or summary.get("slice_count") != 7
    or not isinstance(summary.get("slices"), list)
    or len(summary["slices"]) != 7
):
    raise RuntimeError("corrected Crop official CoreDev summary is incomplete")
official_counts = {
    item.get("dataset"): item.get("sample_count")
    for item in summary["slices"]
    if isinstance(item, dict)
}
if official_counts != EXPECTED_OFFICIAL_ROWS:
    raise RuntimeError("corrected Crop official seven-subset coverage differs")

arms = paired.get("arms")
identity_contracts = paired.get("identity_contracts")
if (
    paired.get("schema_version") != "tgvf.paired-coredev-summary.v2"
    or paired.get("evaluation_id") != evaluation_id
    or paired.get("coverage") != EXPECTED_COVERAGE
    or not isinstance(arms, dict)
    or set(arms) != {"step32"}
    or paired.get("step32") != summary
    or arms["step32"].get("optimizer_step") != STEP
    or arms["step32"].get("evaluation_id") != evaluation_id
    or arms["step32"].get("official_summary") != summary
    or not isinstance(identity_contracts, dict)
    or identity_contracts.get("backend") != "full_model"
    or identity_contracts.get("evaluation_protocol_source")
    != "checkpoint_owner_policy_config"
    or identity_contracts.get("protocol_contract_role")
    != "full_model_materialization_only"
    or identity_contracts.get("training_run_protocol") != protocol
    or arms["step32"].get("evaluation_identity_sha256")
    != proof.get("evaluation_identity_sha256")
):
    raise RuntimeError("corrected Crop paired summary identity differs")

rank_paths = sorted(inference_root.glob("rank-*.jsonl"))
if len(rank_paths) != 4 or any(path.is_symlink() for path in rank_paths):
    raise RuntimeError("corrected Crop inference rank coverage differs")
rows: list[dict[str, Any]] = []
for rank_path in rank_paths:
    with rank_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"inference row must be an object: {rank_path}:{line_number}"
                )
            rows.append(row)
if len(rows) != 2240:
    raise RuntimeError("corrected Crop inference row count differs")
if Counter(row.get("dataset") for row in rows) != Counter(EXPECTED_INFERENCE_ROWS):
    raise RuntimeError("corrected Crop inference subset coverage differs")
sample_ids = [row.get("sample_id") for row in rows]
if (
    any(not isinstance(value, str) or not value for value in sample_ids)
    or len(set(sample_ids)) != len(sample_ids)
    or any(row.get("evaluation_id") != evaluation_id for row in rows)
    or any(row.get("optimizer_step") != STEP for row in rows)
    or any(row.get("world_size") != 4 for row in rows)
    or any(
        row.get("evaluation_identity_sha256")
        != proof.get("evaluation_identity_sha256")
        for row in rows
    )
):
    raise RuntimeError("corrected Crop inference identity closure differs")


def percentile(values: Iterable[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def usage(selected: list[dict[str, Any]]) -> dict[str, object]:
    total_attempts = 0
    total_calls = 0
    total_errors = 0
    attempting = 0
    using = 0
    repeat_using = 0
    observations = 0
    token_counts: list[int] = []
    stop_counts: Counter[str] = Counter()
    function_counts: Counter[str] = Counter()
    error_code_counts: Counter[str] = Counter()
    for row in selected:
        calls = row.get("tool_calls")
        errors = row.get("tool_errors")
        turns = row.get("assistant_turns")
        observed = row.get("successful_observation_count")
        stop = row.get("stop")
        if (
            not isinstance(calls, list)
            or not isinstance(errors, list)
            or not isinstance(turns, list)
            or type(observed) is not int
            or observed < 0
            or not isinstance(stop, str)
            or not stop
        ):
            raise RuntimeError("corrected Crop trajectory audit differs")
        call_count = len(calls)
        error_count = len(errors)
        attempt_count = call_count + error_count
        if observed != call_count or call_count > 6 or attempt_count > 7:
            raise RuntimeError("corrected Crop tool attempt counts differ")
        total_attempts += attempt_count
        total_calls += call_count
        total_errors += error_count
        attempting += int(attempt_count > 0)
        using += int(call_count > 0)
        repeat_using += int(call_count > 1)
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
                raise RuntimeError("corrected Crop assistant-turn audit differs")
            generated += turn["sampled_token_count"]
        token_counts.append(generated)
        for call in calls:
            function_name = call.get("function_name") if isinstance(call, dict) else None
            if function_name != "image_zoom_in_tool":
                raise RuntimeError("corrected Crop successful tool name differs")
            function_counts[function_name] += 1
        for error in errors:
            code = error.get("code") if isinstance(error, dict) else None
            if not isinstance(code, str) or not code:
                raise RuntimeError("corrected Crop tool-error audit differs")
            error_code_counts[code] += 1
    count = len(selected)
    if count == 0:
        raise RuntimeError("corrected Crop usage audit cannot be empty")
    return {
        "trajectory_count": count,
        "no_tool_trajectory_count": count - attempting,
        "trajectories_attempting_tool": attempting,
        "tool_attempt_trajectory_rate": attempting / count,
        "total_tool_attempts": total_attempts,
        "mean_tool_attempts_per_trajectory": total_attempts / count,
        "trajectories_with_successful_tool_call": using,
        "successful_tool_call_trajectory_rate": using / count,
        "successful_tool_call_count": total_calls,
        "trajectories_with_repeat_successful_tool_call": repeat_using,
        "successful_observation_count": observations,
        "tool_error_count": total_errors,
        "tool_error_code_counts": dict(sorted(error_code_counts.items())),
        "function_call_counts": dict(sorted(function_counts.items())),
        "generated_token_mean": sum(token_counts) / count,
        "generated_token_p50": percentile(token_counts, 0.50),
        "generated_token_p95": percentile(token_counts, 0.95),
        "generated_token_p99": percentile(token_counts, 0.99),
        "stop_counts": dict(sorted(stop_counts.items())),
    }


headline = extract_coredev_macro_star(summary)
payload: dict[str, object] = {
    "schema_version": "tgvf.prl27-a-corrected-crop-s32-results.v1",
    "status": "pass",
    "evaluation_id": evaluation_id,
    "contract": (
        "independent fresh-S0 exact-continuation Train@512 S32; exact "
        "training_run Eval@512"
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
    "paired_summary_sha256": sha256(paired_path),
    "tool_usage_definitions": {
        "tool_attempt": "one successful tool call or one recorded tool error",
        "successful_tool_call": "one executed crop action with one visual observation",
        "no_tool_trajectory": "a trajectory with zero successful or failed tool attempts",
    },
    "arm": {
        "method": "Crop exact-continuation Train@512 S32",
        "optimizer_step": STEP,
        "train_image_max_pixels": PIXEL512,
        "evaluation_image_max_pixels": PIXEL512,
        "macro_star_percent": headline["macro_star_percent"],
        "headline": headline,
        "seven_subset_statistics": summary["slices"],
        "tool_usage_overall": usage(rows),
        "tool_usage_by_subset": {
            dataset: usage([row for row in rows if row["dataset"] == dataset])
            for dataset in EXPECTED_INFERENCE_ROWS
        },
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
    },
}
write_json_atomic(output_path, payload)
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY

for path in "$plan" "$handoff" "$proof" "$summary" "$paired_summary" \
  "$runner_complete" "$result"; do
  if [[ ! -s "$path" ]]; then
    echo "PRL-27-A evaluator omitted a completion artifact: $path" >&2
    exit 1
  fi
done

phase=complete
record_phase
touch "$supervisor_complete"
printf '[%s] PRL-27-A corrected Crop S32 Eval@512 complete\n' "$(timestamp)"
