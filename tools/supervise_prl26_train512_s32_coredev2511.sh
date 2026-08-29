#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
resource_validator="$repo_root/tools/validate_prl26_train512_training_handoff.py"
control_root="$main_root/artifacts/evaluation/PRL26-TRAIN512-S32-PIXEL512-COREDEV2511-V1"
runtime_root="$control_root/runtime"
log_root="$control_root/logs"
handoff="$runtime_root/bound-handoff.json"

notool_training_root="$main_root/artifacts/policy/PRL-26-A-train512-s32-parity-notool-qwen3-instruct-bs16-n16-teacher25-ws8"
crop_training_root="$main_root/artifacts/policy/PRL-26-B-train512-s32-parity-crop-qwen3-instruct-bs16-n16-teacher25-ws8"
notool_completion="$notool_training_root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"
crop_completion="$crop_training_root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"

notool_evaluation_id=PRL26-A-TRAIN512-S32-NOTOOL-MATCHED-COREDEV2511-S32-PIXEL512-V1
crop_evaluation_id=PRL26-B-TRAIN512-S32-CROP-MATCHED-COREDEV2511-PIXEL512-BOUNDARYFIX-V1
notool_eval_root="$notool_training_root/evaluation/$notool_evaluation_id"
crop_eval_root="$crop_training_root/evaluation/$crop_evaluation_id"
crop_plan="$crop_eval_root/runtime/bound-crop-plan.json"
crop_config="$crop_eval_root/step32/benchmark-config.json"
crop_validation="$crop_eval_root/logs/prl26-pixel512-static-validation.json"
crop_proof="$crop_eval_root/step32/runtime/pixel512-processor-proof.json"

poll_seconds=${PRL26_AB_POLL_SECONDS:-30}
release_stable_polls=${PRL26_AB_RELEASE_STABLE_POLLS:-2}
release_maximum_polls=${PRL26_AB_RELEASE_MAXIMUM_POLLS:-240}
gpu_memory_threshold_mib=${PRL26_AB_GPU_IDLE_MEMORY_THRESHOLD_MIB:-32}

if [[ ! -f "$resource_validator" ]]; then
  echo "PRL-26 A/B resource validator is absent: $resource_validator" >&2
  exit 1
fi
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$release_stable_polls" =~ ^[1-9][0-9]*$ \
      || ! "$release_maximum_polls" =~ ^[1-9][0-9]*$ \
      || ! "$gpu_memory_threshold_mib" =~ ^[0-9]+$ ]]; then
  echo "PRL-26 A/B resource polling setting is malformed" >&2
  exit 1
fi
if (( release_stable_polls < 2 )); then
  echo "PRL-26 A/B requires at least two consecutive clean resource probes" >&2
  exit 1
fi

# Keep both canonical training roots absent until their trainers have closed
# S32.  Evaluation control belongs under the separate evaluation namespace.
mkdir -p "$runtime_root" "$log_root"
exec 9>"$runtime_root/supervisor.lock"
flock -n 9 || {
  echo "PRL-26 Train@512 S32 evaluator is already active" >&2
  exit 1
}

export PYTHONPATH="$repo_root/src:$main_root/.deps/verl${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_REPOSITORY_ROOT="$repo_root"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

phase=waiting_for_s32_checkpoints
active_pids=()

timestamp() {
  date '+%F %T %Z'
}

wait_for_resources() {
  local label=$1
  local quiet=0
  local total=0
  local probe
  phase="waiting_for_${label}_resource_release"
  while (( quiet < release_stable_polls )); do
    total=$((total + 1))
    probe="$runtime_root/${label}-resource-probe-${total}.json"
    if "$python_bin" "$resource_validator" resources-free \
        --memory-threshold-mib "$gpu_memory_threshold_mib" >"$probe"; then
      quiet=$((quiet + 1))
    else
      quiet=0
    fi
    if (( quiet < release_stable_polls )); then
      if (( total >= release_maximum_polls )); then
        echo "GPUs or Ray did not become clean after $label" >&2
        return 1
      fi
      sleep "$poll_seconds"
    fi
  done
  cp "$probe" "$runtime_root/${label}-resources-released.json"
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
  local pid
  for pid in "${active_pids[@]:-}"; do
    stop_process_group "$pid"
  done
  if (( status == 0 )); then
    rm -f "$runtime_root/failed"
  else
    rm -f "$runtime_root/evaluation-complete"
    printf 'status=failed\nphase=%s\ntime=%s\nexit_status=%s\n' \
      "$phase" "$(timestamp)" "$status" >"$runtime_root/failed"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'phase=signal; exit 130' INT TERM

rm -f "$runtime_root/failed" "$runtime_root/evaluation-complete"
while [[ ! -s "$notool_completion" || ! -s "$crop_completion" ]]; do
  sleep "$poll_seconds"
done

# The permanent S32 receipt is published before the trainer has necessarily
# torn down every vLLM/Ray worker.  Require two consecutive all-GPU and Ray
# clean observations before either arm is allowed to materialize or infer.
wait_for_resources training

phase=binding_completed_s32_checkpoints
# Only a closed Crop S32 receipt authorizes materializing its evaluation tree.
mkdir -p "$crop_eval_root/logs"
env CUDA_VISIBLE_DEVICES= "$python_bin" \
  "$repo_root/tools/bind_prl26_train512_s32_evaluation.py" \
  --crop-plan-output "$crop_plan" --handoff-output "$handoff" \
  >"$log_root/bind-handoff.json" 2>"$log_root/bind-handoff.stderr.log"

# Materialize and statically prove the Crop evaluator before any GPU worker is
# launched.  The generic evaluator then revalidates the same immutable config
# when infer mode resumes it.
phase=preparing_crop_full_model
env CUDA_VISIBLE_DEVICES= "$python_bin" \
  "$repo_root/tools/run_prl15_paired_evaluation.py" \
  --plan "$crop_plan" --mode prepare --output-root "$crop_eval_root" \
  --gpu-ids 4 5 6 7 \
  >"$log_root/crop-prepare.log" 2>&1
phase=validating_crop_pixel512_processor
env CUDA_VISIBLE_DEVICES= "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
  --config "$crop_config" --mode validate --world-size 4 \
  >"$crop_validation" 2>"$crop_eval_root/logs/prl26-pixel512-static-validation.stderr.log"
env CUDA_VISIBLE_DEVICES= "$python_bin" \
  "$repo_root/tools/validate_prl26_train512_processor_proof.py" \
  --arm crop --config "$crop_config" --validation-json "$crop_validation" \
  --output "$crop_proof" >"$log_root/crop-pixel512-proof.json"

phase=running_parallel_four_plus_four_inference
setsid "$repo_root/tools/supervise_prl26_a_no_tool_train512_s32_inference.sh" \
  >"$log_root/notool-inference.log" 2>&1 9>&- &
notool_pid=$!
active_pids+=("$notool_pid")
setsid "$python_bin" "$repo_root/tools/run_prl15_paired_evaluation.py" \
  --plan "$crop_plan" --mode infer --output-root "$crop_eval_root" \
  --gpu-ids 4 5 6 7 \
  >"$log_root/crop-inference.log" 2>&1 9>&- &
crop_pid=$!
active_pids+=("$crop_pid")

while (( ${#active_pids[@]} > 0 )); do
  completed_pid=
  set +e
  wait -n -p completed_pid "${active_pids[@]}"
  inference_status=$?
  set -e
  remaining_pids=()
  for pid in "${active_pids[@]}"; do
    [[ "$pid" == "$completed_pid" ]] || remaining_pids+=("$pid")
  done
  active_pids=("${remaining_pids[@]}")
  if (( inference_status != 0 )); then
    phase=parallel_inference_failed
    exit "$inference_status"
  fi
done
touch "$runtime_root/parallel-inference-complete"

# One local Qwen2.5-72B judge is used at a time.  This keeps the judge identity
# and port fail-closed while both arms retain all seven official slice reports.
phase=scoring_crop_seven_subsets
"$python_bin" "$repo_root/tools/run_prl15_paired_evaluation.py" \
  --plan "$crop_plan" --mode score --output-root "$crop_eval_root" \
  --gpu-ids 0 1 2 3 >"$log_root/crop-scoring.log" 2>&1

phase=scoring_notool_seven_subsets
env \
  PRL25_F_MATCHED_EVAL_ROOT="$notool_eval_root" \
  PRL25_F_MATCHED_EVALUATION_ID_PREFIX=PRL26-A-TRAIN512-S32-NOTOOL-MATCHED-COREDEV2511-S \
  PRL25_F_MATCHED_EVALUATION_ID_SUFFIX=-PIXEL512-V1 \
  PRL25_F_MATCHED_REQUIRED_IMAGE_MAX_PIXELS=262144 \
  PRL25_F_MATCHED_PROCESSOR_PROOF_FILENAME=pixel512-processor-proof.json \
  PRL25_F_MATCHED_STEPS=32 \
  PRL25_F_MATCHED_RUN_ID_S32=T20260829_G26a32 \
  PRL25_F_MATCHED_INFERENCE_FAILURE_MARKER="$notool_eval_root/runtime/supervisor/failed" \
  "$repo_root/tools/supervise_prl25_f_no_tool_matched_scoring.sh" \
  >"$log_root/notool-scoring.log" 2>&1

phase=building_train512_s32_result_table
crop_summary="$crop_eval_root/step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json"
notool_summary="$notool_eval_root/matched/step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json"
PYTHONPATH="$repo_root/src" "$python_bin" - \
  "$notool_summary" "$crop_summary" "$handoff" \
  "$control_root/train512-s32-pixel512-results.json" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from tgvf_rl.evaluation.coredev_results import (
    extract_coredev_macro_star,
    write_json_atomic,
)

notool_path, crop_path, handoff_path, output_path = map(Path, sys.argv[1:])
handoff = json.loads(handoff_path.read_text(encoding="utf-8"))

def arm_record(name: str, path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("schema_version") != 1
        or summary.get("phase") != "eval"
        or summary.get("sample_count") != 2511
        or summary.get("slice_count") != 7
        or len(summary.get("slices", [])) != 7
    ):
        raise RuntimeError(f"{name} CoreDev summary is incomplete")
    headline = extract_coredev_macro_star(summary)
    return {
        "method": name,
        "train_image_max_pixels": 262144,
        "evaluation_image_max_pixels": 262144,
        "optimizer_step": 32,
        "macro_star_percent": headline["macro_star_percent"],
        "headline": headline,
        "seven_subset_statistics": summary["slices"],
        "summary_path": str(path.resolve()),
    }

payload = {
    "schema_version": "tgvf.prl26-train512-s32-pixel512-results.v1",
    "status": "pass",
    "contract": "fresh-S0 Train@512 S32; matched Eval@512",
    "coverage": {
        "official_manifest_rows": 2511,
        "evaluated_single_image_rows": 2240,
        "held_multi_image_rows": 271,
        "subset_count": 7,
    },
    "handoff_identity_sha256": handoff["identity_sha256"],
    "arms": {
        "no_tool": arm_record("NoTool Train@512 S32", notool_path),
        "crop": arm_record("Crop Train@512 S32", crop_path),
    },
}
write_json_atomic(output_path.resolve(), payload)
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY

phase=complete
touch "$runtime_root/evaluation-complete"
printf '[%s] PRL-26 Train@512 S32 NoTool/Crop evaluation complete\n' "$(timestamp)"
