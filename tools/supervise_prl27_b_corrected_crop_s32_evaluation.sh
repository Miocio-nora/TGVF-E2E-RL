#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"

training_root="$main_root/artifacts/policy/PRL-27-B-train512-s32-crop-replay-byte-parity-qwen3-instruct-bs16-n16-teacher25-ws8"
receipt="$training_root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"
training_control_root="$main_root/artifacts/control/PRL-27-B-crop-train512-s32-20260830"
training_complete="$training_control_root/state/s32-accepted"
training_failed="$training_control_root/state/failed"
training_admitted_head="$training_control_root/admitted-head.txt"
expected_training_admitted_head=f50fe3c66c719dd10f5dc5522e5142594831038b

evaluation_id=PRL27-B-CROP-REPLAY-BYTE-PARITY-TRAIN512-S32-TRAINING-RUN-COREDEV2511-PIXEL512-V1
eval_root="$training_root/evaluation/$evaluation_id"
plan="$eval_root/runtime/bound-crop-plan.json"
handoff="$eval_root/runtime/bound-handoff.json"
config="$eval_root/step32/benchmark-config.json"
validation="$eval_root/step32/logs/prl27-b-pixel512-static-validation.json"
proof="$eval_root/step32/runtime/pixel512-processor-proof.json"
summary="$eval_root/step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json"
paired_summary="$eval_root/paired-summary.json"
runner_complete="$eval_root/evaluation-complete"
result="$eval_root/corrected-crop-s32-pixel512-results.json"

# Wait state is outside the not-yet-authorized evaluation root.  The binder is
# the first operation that may create that root after S32 and resource release.
control_root="$main_root/artifacts/control/PRL-27-B-corrected-crop-s32-eval512-20260830"
runtime_root="$control_root/runtime"
log_root="$control_root/logs"
state_root="$control_root/state"
supervisor_complete="$state_root/evaluation-complete"

binder="$repo_root/tools/bind_prl27_b_corrected_crop_training_run_evaluation.py"
runner="$repo_root/tools/run_prl15_paired_evaluation.py"
benchmark_runner="$repo_root/tools/run_policy_benchmark.py"
proof_validator="$repo_root/tools/validate_prl26_train512_processor_proof.py"
resource_validator="$repo_root/tools/validate_prl26_train512_training_handoff.py"
summarizer="$repo_root/tools/summarize_prl27_b_corrected_crop_s32_evaluation.py"

for path in "$python_bin" "$binder" "$runner" "$benchmark_runner" \
  "$proof_validator" "$resource_validator" "$summarizer"; do
  if [[ ! -f "$path" ]]; then
    echo "required PRL-27-B evaluation file is absent: $path" >&2
    exit 1
  fi
done

poll_seconds=${PRL27_B_EVAL_POLL_SECONDS:-30}
release_stable_polls=${PRL27_B_EVAL_RELEASE_STABLE_POLLS:-3}
release_maximum_polls=${PRL27_B_EVAL_RELEASE_MAXIMUM_POLLS:-240}
gpu_memory_threshold_mib=${PRL27_B_EVAL_GPU_IDLE_MEMORY_THRESHOLD_MIB:-32}
admitted_head=${PRL27_B_EVAL_ADMITTED_HEAD:-}
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$release_stable_polls" =~ ^[1-9][0-9]*$ \
      || ! "$release_maximum_polls" =~ ^[1-9][0-9]*$ \
      || ! "$gpu_memory_threshold_mib" =~ ^[0-9]+$ ]]; then
  echo "PRL-27-B evaluation polling setting is malformed" >&2
  exit 1
fi
if (( release_stable_polls < 3 )); then
  echo "PRL-27-B evaluation requires at least three clean resource probes" >&2
  exit 1
fi
if [[ ! "$admitted_head" =~ ^[0-9a-f]{40}$ ]]; then
  echo "PRL27_B_EVAL_ADMITTED_HEAD is required for evaluation" >&2
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
    echo "clean PRL-27-B evaluation worktree identity differs" >&2
    exit 1
  fi
}

validate_worktree
if [[ -L "$control_root" || ( -e "$control_root" && ! -d "$control_root" ) ]]; then
  echo "PRL-27-B evaluation control root is unsafe" >&2
  exit 1
fi
mkdir -p "$runtime_root" "$log_root" "$state_root"
if [[ -L "$runtime_root/supervisor.lock" ]]; then
  echo "PRL-27-B evaluation supervisor lock cannot be a symlink" >&2
  exit 1
fi
exec 9>"$runtime_root/supervisor.lock"
flock -n 9 || {
  echo "PRL-27-B corrected Crop evaluator is already active" >&2
  exit 1
}
if [[ -L "$control_root/admitted-head.txt" ]]; then
  echo "PRL-27-B evaluation admitted HEAD cannot be a symlink" >&2
  exit 1
fi
if [[ -e "$control_root/admitted-head.txt" ]]; then
  if [[ ! -s "$control_root/admitted-head.txt" \
        || "$(tr -d '\r\n' <"$control_root/admitted-head.txt")" != "$admitted_head" ]]; then
    echo "PRL-27-B evaluation control state belongs to another HEAD" >&2
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

phase=waiting_for_training_admission
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

# Bind this waiter to the exact clean HEAD admitted by the training supervisor.
while [[ ! -s "$training_admitted_head" ]]; do
  if [[ -e "$training_failed" || -L "$training_failed" ]]; then
    echo "PRL-27-B training failed before admission" >&2
    exit 1
  fi
  if [[ -L "$training_admitted_head" ]]; then
    echo "PRL-27-B training admitted HEAD cannot be a symlink" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done
if [[ -L "$training_admitted_head" \
      || "$(tr -d '\r\n' <"$training_admitted_head")" \
          != "$expected_training_admitted_head" ]]; then
  echo "PRL-27-B training admitted HEAD differs from the frozen owner" >&2
  exit 1
fi
printf 'training_admitted_head=%s\nevaluation_admitted_head=%s\n' \
  "$expected_training_admitted_head" "$admitted_head" \
  >"$runtime_root/training-evaluation-head-binding.txt"

phase=waiting_for_s32_training_acceptance
record_phase
while [[ ! -f "$training_complete" || ! -s "$receipt" ]]; do
  if [[ -L "$training_complete" || -L "$receipt" ]]; then
    echo "PRL-27-B training completion boundary cannot be a symlink" >&2
    exit 1
  fi
  if [[ -e "$training_failed" || -L "$training_failed" ]]; then
    echo "PRL-27-B training supervisor failed before accepted S32" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done
if [[ -L "$training_complete" || ! -f "$training_complete" \
      || -L "$receipt" || ! -s "$receipt" \
      || -e "$training_failed" || -L "$training_failed" ]]; then
  echo "PRL-27-B accepted S32 boundary changed during evaluation admission" >&2
  exit 1
fi

# The permanent receipt can precede trainer teardown.  Preserve every probe and
# require consecutive all-GPU/Ray clean samples before materialization.
phase=waiting_for_training_resource_release
record_phase
quiet=0
total=0
while (( quiet < release_stable_polls )); do
  if [[ -e "$training_failed" || -L "$training_failed" ]]; then
    echo "PRL-27-B training failure appeared during resource admission" >&2
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
      echo "GPUs or Ray did not become clean before PRL-27-B evaluation" >&2
      exit 1
    fi
    sleep "$poll_seconds"
  fi
done
cp "$probe" "$runtime_root/resources-released.json"
validate_worktree

# The binder revalidates the permanent receipt, contiguous S1--S32 metrics, the
# checkpoint owner, exact continuation, action boundary, and independent RNG.
phase=binding_corrected_crop_s32
record_phase
env CUDA_VISIBLE_DEVICES= "$python_bin" "$binder" \
  --crop-plan-output "$plan" --handoff-output "$handoff" \
  >"$log_root/bind-handoff.json" 2>"$log_root/bind-handoff.stderr.log"
mkdir -p "$eval_root/logs"

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
  >"$validation" 2>"$eval_root/logs/prl27-b-pixel512-static-validation.stderr.log"
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

phase=publishing_macro_tool_use_and_length
record_phase
validate_worktree
env CUDA_VISIBLE_DEVICES= "$python_bin" "$summarizer" \
  --evaluation-root "$eval_root" --output "$result" \
  >"$log_root/result-table.json" 2>"$log_root/result-table.stderr.log"

for path in "$plan" "$handoff" "$proof" "$summary" "$paired_summary" "$result"; do
  if [[ ! -s "$path" ]]; then
    echo "PRL-27-B evaluator omitted a completion artifact: $path" >&2
    exit 1
  fi
done
if [[ -L "$runner_complete" || ! -e "$runner_complete" ]]; then
  echo "PRL-27-B paired evaluator did not publish completion" >&2
  exit 1
fi

phase=complete
record_phase
touch "$supervisor_complete"
printf '[%s] PRL-27-B corrected Crop S32 Eval@512 complete\n' "$(timestamp)"
