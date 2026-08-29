#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"

training_root="$main_root/artifacts/policy/PRL-26-B-train512-s32-parity-crop-qwen3-instruct-bs16-n16-teacher25-ws8"
receipt="$training_root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"
evaluation_id=PRL26-B-S32-OWNER-GENERIC86-TRAINING-RUN-COREDEV2511-PIXEL512-V1
eval_root="$training_root/evaluation/$evaluation_id"
plan="$eval_root/runtime/bound-crop-plan.json"
handoff="$eval_root/runtime/bound-handoff.json"
reuse_proof="$eval_root/runtime/full-model-materialization-reuse.json"
config="$eval_root/step32/benchmark-config.json"
validation="$eval_root/step32/logs/generic86-pixel512-static-validation.json"
proof="$eval_root/step32/runtime/pixel512-processor-proof.json"
summary="$eval_root/step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json"
paired_summary="$eval_root/paired-summary.json"
runner_complete="$eval_root/evaluation-complete"
result="$eval_root/generic86-crop-s32-pixel512-results.json"

attempt=${PRL26_B_GENERIC86_EVAL_ATTEMPT:-0}
if [[ ! "$attempt" =~ ^[0-9]+$ ]]; then
  echo "PRL-26-B generic86 evaluation attempt is malformed" >&2
  exit 1
fi
control_name=PRL-26-B-generic86-s32-eval512-20260830
if (( attempt > 0 )); then
  control_name="${control_name}-recovery${attempt}"
fi
control_root="$main_root/artifacts/control/$control_name"
runtime_root="$control_root/runtime"
log_root="$control_root/logs"
state_root="$control_root/state"
supervisor_complete="$state_root/evaluation-complete"

binder="$repo_root/tools/bind_prl26_b_generic86_training_run_evaluation.py"
reuser="$repo_root/tools/reuse_prl26_b_s32_full_model_for_generic86_eval.py"
runner="$repo_root/tools/run_prl15_paired_evaluation.py"
benchmark_runner="$repo_root/tools/run_policy_benchmark.py"
proof_validator="$repo_root/tools/validate_prl26_train512_processor_proof.py"
resource_validator="$repo_root/tools/validate_prl26_train512_training_handoff.py"
summarizer="$repo_root/tools/summarize_prl26_b_generic86_s32_evaluation.py"

for path in "$python_bin" "$receipt" "$binder" "$reuser" "$runner" \
  "$benchmark_runner" "$proof_validator" "$resource_validator" "$summarizer"; do
  if [[ ! -f "$path" ]]; then
    echo "required PRL-26-B generic86 evaluation file is absent: $path" >&2
    exit 1
  fi
done

poll_seconds=${PRL26_B_GENERIC86_EVAL_POLL_SECONDS:-15}
release_stable_polls=${PRL26_B_GENERIC86_EVAL_RELEASE_STABLE_POLLS:-3}
release_maximum_polls=${PRL26_B_GENERIC86_EVAL_RELEASE_MAXIMUM_POLLS:-480}
gpu_memory_threshold_mib=${PRL26_B_GENERIC86_EVAL_GPU_IDLE_MEMORY_THRESHOLD_MIB:-32}
admitted_head=${PRL26_B_GENERIC86_EVAL_ADMITTED_HEAD:-}
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$release_stable_polls" =~ ^[1-9][0-9]*$ \
      || ! "$release_maximum_polls" =~ ^[1-9][0-9]*$ \
      || ! "$gpu_memory_threshold_mib" =~ ^[0-9]+$ \
      || ! "$admitted_head" =~ ^[0-9a-f]{40}$ ]]; then
  echo "PRL-26-B generic86 evaluation admission setting is malformed" >&2
  exit 1
fi
if (( release_stable_polls < 3 )); then
  echo "PRL-26-B generic86 evaluation requires three clean resource probes" >&2
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
    echo "clean PRL-26-B generic86 evaluation worktree identity differs" >&2
    exit 1
  fi
}

validate_worktree
if [[ -L "$control_root" || ( -e "$control_root" && ! -d "$control_root" ) ]]; then
  echo "PRL-26-B generic86 control root is unsafe" >&2
  exit 1
fi
mkdir -p "$runtime_root" "$log_root" "$state_root"
exec 9>"$runtime_root/supervisor.lock"
flock -n 9 || {
  echo "PRL-26-B generic86 evaluator is already active" >&2
  exit 1
}
if [[ -e "$control_root/admitted-head.txt" ]]; then
  if [[ -L "$control_root/admitted-head.txt" \
        || "$(tr -d '\r\n' <"$control_root/admitted-head.txt")" != "$admitted_head" ]]; then
    echo "PRL-26-B generic86 control state belongs to another HEAD" >&2
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
# The generic86 arm evaluates the exact training runtime with precomputed image
# embeddings.  Its full-model engine therefore selects the repo-owned
# TGVFQwen3VLForConditionalGeneration architecture and worker extension.  Keep
# the complete audited rollout environment enabled in every spawned rank.
export VLLM_PLUGINS=tgvf_qwen3_precomputed
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VERL_FULL_DETERMINISM=0
export VLLM_BATCH_INVARIANT=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

phase=admission
active_pid=
timestamp() { date '+%F %T %Z'; }
record_phase() {
  printf 'phase=%s\ntime=%s\n' "$phase" "$(timestamp)" >"$state_root/current-phase"
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

phase=waiting_for_all_gpu_and_ray_release
record_phase
quiet=0
total=0
while (( quiet < release_stable_polls )); do
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
      echo "GPUs or Ray did not release before generic86 evaluation" >&2
      exit 1
    fi
    sleep "$poll_seconds"
  fi
done
cp "$probe" "$runtime_root/resources-released.json"
validate_worktree

phase=binding_owner_native_generic86_s32
record_phase
env CUDA_VISIBLE_DEVICES= "$python_bin" "$binder" \
  --plan-output "$plan" --handoff-output "$handoff" \
  >"$log_root/bind-handoff.json" 2>"$log_root/bind-handoff.stderr.log"

phase=rebinding_existing_full_model_read_only
record_phase
env CUDA_VISIBLE_DEVICES= "$python_bin" "$reuser" \
  >"$log_root/full-model-reuse.json" 2>"$log_root/full-model-reuse.stderr.log"

phase=preparing_generic86_full_model
record_phase
if (( attempt == 0 )); then
  validate_worktree
  run_group "$log_root/prepare.log" \
    "$python_bin" "$runner" --plan "$plan" --mode prepare \
    --output-root "$eval_root" --gpu-ids 0 1 2 3

  phase=proving_pixel512_and_generic86_continuation
  record_phase
  validate_worktree
  mkdir -p "$eval_root/logs" "$(dirname "$validation")" "$(dirname "$proof")"
  env CUDA_VISIBLE_DEVICES= "$python_bin" "$benchmark_runner" \
    --config "$config" --mode validate --world-size 4 \
    >"$validation" 2>"$eval_root/logs/generic86-static-validation.stderr.log"
else
  phase=validating_frozen_generic86_recovery_artifacts
  record_phase
  for path in "$plan" "$handoff" "$reuse_proof" "$config" "$validation" "$proof"; do
    if [[ -L "$path" || ! -f "$path" || ! -s "$path" ]]; then
      echo "PRL-26-B generic86 recovery artifact is unsafe: $path" >&2
      exit 1
    fi
  done
fi
env CUDA_VISIBLE_DEVICES= "$python_bin" "$proof_validator" \
  --arm crop --config "$config" --validation-json "$validation" \
  --output "$proof" >"$log_root/pixel512-generic86-proof.json"
touch "$state_root/processor-proof-complete"

phase=running_generic86_four_gpu_inference
record_phase
validate_worktree
run_group "$log_root/inference.log" \
  "$python_bin" "$runner" --plan "$plan" --mode infer \
  --output-root "$eval_root" --gpu-ids 0 1 2 3
touch "$state_root/inference-complete"

phase=scoring_generic86_seven_subsets
record_phase
validate_worktree
run_group "$log_root/scoring.log" \
  "$python_bin" "$runner" --plan "$plan" --mode score \
  --output-root "$eval_root" --gpu-ids 0 1 2 3

phase=publishing_generic86_macro_tool_use_and_length
record_phase
validate_worktree
env CUDA_VISIBLE_DEVICES= "$python_bin" "$summarizer" \
  --evaluation-root "$eval_root" --output "$result" \
  >"$log_root/result-table.json" 2>"$log_root/result-table.stderr.log"

for path in "$plan" "$handoff" "$reuse_proof" "$proof" "$summary" \
  "$paired_summary" "$result"; do
  if [[ ! -s "$path" ]]; then
    echo "PRL-26-B generic86 evaluator omitted: $path" >&2
    exit 1
  fi
done
if [[ -L "$runner_complete" || ! -e "$runner_complete" ]]; then
  echo "PRL-26-B generic86 paired evaluator omitted completion" >&2
  exit 1
fi

phase=complete
record_phase
touch "$supervisor_complete"
printf '[%s] PRL-26-B generic86 S32 Eval@512 complete\n' "$(timestamp)"
