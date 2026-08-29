#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
plan="$repo_root/configs/evaluation/prl26_e_atomic_crop_tgvf_train512_s32_pixel512_coredev2511_plan.json"
training_root="$main_root/artifacts/policy/PRL-26-E-train512-s32-parity-atomic-crop-tgvf-qwen3-instruct-bs16-n16-teacher25-ws8"
receipt="$training_root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"
eval_id=PRL26-E-ATOMIC-CROP-TGVF-TRAIN512-S32-PIXEL512-COREDEV2511-SEED42-V1
eval_root="$training_root/evaluation/$eval_id"
runtime_root="$eval_root/runtime"
log_root="$eval_root/logs"
result="$eval_root/atomic-s32-pixel512-results.json"
runner="$repo_root/tools/run_prl15_paired_evaluation.py"
benchmark_runner="$repo_root/tools/run_policy_benchmark.py"
proof_validator="$repo_root/tools/validate_prl26_train512_processor_proof.py"
summarizer="$repo_root/tools/summarize_prl26_e_atomic_s32_evaluation.py"
resource_validator="$repo_root/tools/validate_prl26_train512_training_handoff.py"
control_root="$main_root/artifacts/control/PRL-26-E-atomic-train512-s32-20260829"
state_root="$control_root/state"
admitted_head_file="$control_root/admitted-head.txt"
config="$eval_root/step32/benchmark-config.json"
validation="$eval_root/step32/logs/prl26-e-pixel512-static-validation.json"
proof="$eval_root/step32/runtime/pixel512-processor-proof.json"

admitted_head=${1:-}
if [[ ! "$admitted_head" =~ ^[0-9a-f]{40}$ ]]; then
  echo "exact admitted Atomic implementation HEAD is required" >&2
  exit 1
fi
for path in "$python_bin" "$plan" "$receipt" "$runner" "$benchmark_runner" \
  "$proof_validator" "$summarizer" "$resource_validator" \
  "$admitted_head_file" "$state_root/atomic-s32-accepted" \
  "$state_root/formal-training-complete"; do
  if [[ ! -f "$path" ]]; then
    echo "required PRL-26-E evaluation file is absent: $path" >&2
    exit 1
  fi
done

if [[ -L "$admitted_head_file" \
      || -L "$state_root/atomic-s32-accepted" \
      || -L "$state_root/formal-training-complete" ]]; then
  echo "PRL-26-E evaluation control state cannot be a symlink" >&2
  exit 1
fi
recorded_admitted_head=$(tr -d '\r\n' <"$admitted_head_file")
if [[ "$recorded_admitted_head" != "$admitted_head" ]]; then
  echo "PRL-26-E evaluation admitted HEAD differs from training control" >&2
  exit 1
fi

validate_worktree() {
  local observed_root observed_head dirty recorded
  observed_root=$(git -C "$repo_root" rev-parse --show-toplevel)
  observed_head=$(git -C "$repo_root" rev-parse HEAD)
  dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
  recorded=$(tr -d '\r\n' <"$admitted_head_file")
  if [[ "$observed_root" != "$repo_root" \
        || "$observed_head" != "$admitted_head" \
        || "$recorded" != "$admitted_head" \
        || -n "$dirty" ]]; then
    echo "clean PRL-26-E evaluation worktree identity differs" >&2
    exit 1
  fi
}

validate_worktree

# This script may be resumed directly, so it independently requires two
# consecutive all-GPU/Ray-clean probes before creating evaluation state.
release_stable_polls=${PRL26_E_RELEASE_STABLE_POLLS:-2}
release_maximum_polls=${PRL26_E_RELEASE_MAXIMUM_POLLS:-240}
poll_seconds=${PRL26_E_POLL_SECONDS:-30}
gpu_memory_threshold_mib=${PRL26_E_GPU_IDLE_MEMORY_THRESHOLD_MIB:-32}
if [[ ! "$release_stable_polls" =~ ^[1-9][0-9]*$ \
      || ! "$release_maximum_polls" =~ ^[1-9][0-9]*$ \
      || ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$gpu_memory_threshold_mib" =~ ^[0-9]+$ ]]; then
  echo "PRL-26-E evaluation polling setting is malformed" >&2
  exit 1
fi
if (( release_stable_polls < 2 )); then
  echo "PRL-26-E evaluation requires two consecutive clean resource probes" >&2
  exit 1
fi
quiet=0
total=0
while (( quiet < release_stable_polls )); do
  total=$((total + 1))
  if "$python_bin" "$resource_validator" resources-free \
      --memory-threshold-mib "$gpu_memory_threshold_mib" >/dev/null; then
    quiet=$((quiet + 1))
  else
    quiet=0
  fi
  if (( quiet < release_stable_polls )); then
    if (( total >= release_maximum_polls )); then
      echo "GPUs or Ray did not become clean before Atomic evaluation" >&2
      exit 1
    fi
    sleep "$poll_seconds"
  fi
done
validate_worktree

mkdir -p "$runtime_root" "$log_root"
exec 9>"$runtime_root/supervisor.lock"
flock -n 9 || {
  echo "PRL-26-E Atomic evaluator is already active" >&2
  exit 1
}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo_root/src:$main_root/.deps/verl${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_REPOSITORY_ROOT="$repo_root"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

phase=initializing
active_pid=

timestamp() {
  date '+%F %T %Z'
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

phase=validating_static_atomic_plan
validate_worktree
env CUDA_VISIBLE_DEVICES= "$python_bin" - "$runner" "$plan" \
  >"$log_root/static-plan-validation.json" <<'PY'
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

runner_path, plan_path = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("prl26_e_atomic_runner", runner_path)
if spec is None or spec.loader is None:
    raise RuntimeError("Atomic paired evaluator module cannot be loaded")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
plan = module._load_plan(plan_path.resolve())
print(json.dumps({
    "schema_version": plan["schema_version"],
    "evaluation_id": plan["evaluation_id"],
    "protocol_sha256": plan["paired_rng"]["protocol_sha256"],
}, indent=2, sort_keys=True))
PY
validate_worktree

phase=preparing_atomic_s32_snapshot
validate_worktree
run_group "$log_root/prepare.log" \
  "$python_bin" "$runner" --plan "$plan" --mode prepare \
  --output-root "$eval_root" --gpu-ids 0 1 2 3
validate_worktree

phase=validating_atomic_pixel512_processor_and_boundary
env CUDA_VISIBLE_DEVICES= "$python_bin" "$benchmark_runner" \
  --config "$config" --mode validate --world-size 4 \
  >"$validation" 2>"$eval_root/step32/logs/prl26-e-pixel512-static-validation.stderr.log"
env CUDA_VISIBLE_DEVICES= "$python_bin" "$proof_validator" \
  --arm atomic --config "$config" --validation-json "$validation" \
  --output "$proof" >"$log_root/atomic-pixel512-proof.json"
validate_worktree
touch "$runtime_root/processor-proof-complete"

phase=running_atomic_four_gpu_inference
validate_worktree
run_group "$log_root/inference.log" \
  "$python_bin" "$runner" --plan "$plan" --mode infer \
  --output-root "$eval_root" --gpu-ids 0 1 2 3
validate_worktree
touch "$runtime_root/inference-complete"

phase=scoring_atomic_seven_subsets
validate_worktree
run_group "$log_root/scoring.log" \
  "$python_bin" "$runner" --plan "$plan" --mode score \
  --output-root "$eval_root" --gpu-ids 0 1 2 3
validate_worktree

phase=publishing_atomic_result_and_tool_usage
env CUDA_VISIBLE_DEVICES= "$python_bin" "$summarizer" \
  --evaluation-root "$eval_root" --output "$result" \
  >"$log_root/result-table.json"
validate_worktree

if [[ ! -s "$eval_root/paired-summary.json" \
      || ! -s "$eval_root/evaluation-complete" \
      || ! -s "$result" ]]; then
  echo "PRL-26-E evaluator omitted a completion artifact" >&2
  exit 1
fi

phase=complete
validate_worktree
touch "$runtime_root/evaluation-complete"
printf '[%s] PRL-26-E Atomic S32 Eval@512 complete\n' "$(timestamp)"
