#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
verl_dependency="$main_root/.deps/verl"
control_root="$main_root/artifacts/control/PRL-26-E-atomic-train512-s32-20260829"
state_root="$control_root/state"
log_root="$control_root/logs"
validator="$repo_root/tools/validate_prl26_e_atomic_handoff.py"
resource_validator="$repo_root/tools/validate_prl26_train512_training_handoff.py"
completion_validator="$repo_root/tools/validate_prl26_train512_training_handoff.py"
post_train_eval="$repo_root/tools/supervise_prl26_e_atomic_s32_evaluation.sh"

canary="$repo_root/configs/policy/runs/prl_26_e_c0_qwen3_instruct_full_atomic_crop_tgvf_train512_parity_bs4_n2_teacher25_1step_ws4.toml"
formal="$repo_root/configs/policy/runs/prl_26_e_qwen3_instruct_full_atomic_crop_tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
reference_canary="$repo_root/configs/policy/runs/prl_26_c_c0_qwen3_instruct_short_tgvf_train512_parity_bs4_n2_teacher25_1step_ws4.toml"
reference_formal="$repo_root/configs/policy/runs/prl_26_c_qwen3_instruct_short_tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
plan="$repo_root/configs/evaluation/prl26_e_atomic_crop_tgvf_train512_s32_pixel512_coredev2511_plan.json"

canary_root="$main_root/artifacts/policy/PRL-26-E-C0-train512-parity-atomic-crop-tgvf-qwen3-instruct-bs4-n2-teacher25-ws4"
formal_root="$main_root/artifacts/policy/PRL-26-E-train512-s32-parity-atomic-crop-tgvf-qwen3-instruct-bs16-n16-teacher25-ws8"

prerequisite_root="$main_root/artifacts/evaluation/PRL26-CD-TGVF-PROMPT-PAIR-S32-PIXEL512-COREDEV2511-V1"
prerequisite_result="$prerequisite_root/tgvf-target-prompt-s32-pixel512-results.json"
prerequisite_complete="$prerequisite_root/runtime/evaluation-complete"
prerequisite_failed="$prerequisite_root/runtime/failed"

poll_seconds=${PRL26_E_POLL_SECONDS:-30}
release_stable_polls=${PRL26_E_RELEASE_STABLE_POLLS:-2}
release_maximum_polls=${PRL26_E_RELEASE_MAXIMUM_POLLS:-240}
gpu_memory_threshold_mib=${PRL26_E_GPU_IDLE_MEMORY_THRESHOLD_MIB:-32}
maximum_restarts=${PRL26_E_TRAIN_MAXIMUM_RESTARTS:-12}
cooldown_seconds=${PRL26_E_TRAIN_RESTART_COOLDOWN_SECONDS:-15}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for PRL-26-E Atomic training" >&2
  exit 1
fi
for path in "$validator" "$resource_validator" "$completion_validator" \
  "$post_train_eval" "$canary" "$formal" "$reference_canary" \
  "$reference_formal" "$plan"; do
  if [[ ! -f "$path" ]]; then
    echo "required PRL-26-E file is absent: $path" >&2
    exit 1
  fi
done
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$release_stable_polls" =~ ^[1-9][0-9]*$ \
      || ! "$release_maximum_polls" =~ ^[1-9][0-9]*$ \
      || ! "$gpu_memory_threshold_mib" =~ ^[0-9]+$ \
      || ! "$maximum_restarts" =~ ^[0-9]+$ \
      || ! "$cooldown_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "PRL-26-E polling or retry setting is malformed" >&2
  exit 1
fi
if (( release_stable_polls < 2 )); then
  echo "PRL-26-E requires at least two consecutive clean resource probes" >&2
  exit 1
fi

mkdir -p "$state_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
flock -n 9 || {
  echo "another PRL-26-E Atomic supervisor is active" >&2
  exit 1
}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo_root/src:$verl_dependency${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_REPOSITORY_ROOT="$repo_root"
export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=mio_nora
export WANDB_PROJECT=tgvf-policy-rl
export WANDB_MODE=offline
unset RAY_ADDRESS

phase=initializing
active_pid=

timestamp() {
  date '+%F %T %Z'
}

validate_worktree() {
  local observed_root observed_head dirty
  observed_root=$(git -C "$repo_root" rev-parse --show-toplevel)
  observed_head=$(git -C "$repo_root" rev-parse HEAD)
  dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
  if [[ "$observed_root" != "$repo_root" \
        || "$observed_head" != "$admitted_head" \
        || -n "$dirty" ]]; then
    echo "clean PRL-26-E worktree identity differs" >&2
    exit 1
  fi
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
    printf 'status=failed\nphase=%s\ntime=%s\nexit_status=%s\n' \
      "$phase" "$(timestamp)" "$status" >"$state_root/failed"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'phase=signal; exit 130' INT TERM

wait_for_resources() {
  local label=$1
  local quiet=0
  local total=0
  local probe
  phase="waiting_for_${label}_resource_release"
  while (( quiet < release_stable_polls )); do
    total=$((total + 1))
    probe="$control_root/${label}-resource-probe-${total}.json"
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
  cp "$probe" "$control_root/${label}-resources-released.json"
}

admitted_head=$(git -C "$repo_root" rev-parse HEAD)
validate_worktree
resume_admission=false
if [[ -L "$control_root/admitted-head.txt" ]]; then
  echo "PRL-26-E admitted HEAD state cannot be a symlink" >&2
  exit 1
fi
if [[ -e "$control_root/admitted-head.txt" ]]; then
  if [[ ! -s "$control_root/admitted-head.txt" ]]; then
    echo "PRL-26-E admitted HEAD state is empty or non-regular" >&2
    exit 1
  fi
  previous_admitted_head=$(tr -d '\r\n' <"$control_root/admitted-head.txt")
  if [[ "$previous_admitted_head" != "$admitted_head" ]]; then
    echo "PRL-26-E control state belongs to another admitted HEAD" >&2
    exit 1
  fi
  resume_admission=true
fi
contract_arguments=(
  contracts
  --repository "$repo_root"
  --canary "$canary"
  --formal "$formal"
  --reference-canary "$reference_canary"
  --reference-formal "$reference_formal"
  --plan "$plan"
  --require-clean
)
if [[ "$resume_admission" != true ]]; then
  contract_arguments+=(--require-output-roots-absent)
fi
"$python_bin" "$validator" "${contract_arguments[@]}" \
  >"$control_root/static-contract-audit.json"
printf '%s\n' "$admitted_head" >"$control_root/admitted-head.txt"
rm -f "$state_root/failed"

for marker in "$state_root/canary-accepted" "$state_root/atomic-s32-accepted"; do
  if [[ -L "$marker" || ( -e "$marker" && ! -f "$marker" ) ]]; then
    echo "PRL-26-E accepted state marker is not a regular file: $marker" >&2
    exit 1
  fi
done

# PRL-26-E is downstream of the complete C/D paired evaluation, not merely of
# their S32 checkpoint receipts. A stale or failed result cannot authorize it.
phase=waiting_for_tgvf_paired_evaluation
while [[ ! -f "$prerequisite_complete" ]]; do
  if [[ -e "$prerequisite_failed" || -L "$prerequisite_failed" ]]; then
    echo "TGVF paired evaluation failed; Atomic remains untouched" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done
"$python_bin" "$validator" prerequisite \
  --result "$prerequisite_result" --complete-marker "$prerequisite_complete" \
  --failed-marker "$prerequisite_failed" \
  >"$control_root/prerequisite-evaluation-audit.json"
wait_for_resources prerequisite
validate_worktree
touch "$state_root/prerequisite-accepted"

phase=running_atomic_c0
if [[ ! -f "$state_root/canary-accepted" ]]; then
  if [[ ! -e "$canary_root" ]]; then
    mkdir -p "$log_root/c0"
    setsid env \
      TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=4 \
      TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8 \
      TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2 \
      TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30 \
      TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0 \
      "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$canary" --mode canary --target-step 1 \
      >"$log_root/c0/run.log" 2>&1 9>&- &
    active_pid=$!
    wait "$active_pid"
    active_pid=
  fi
  "$python_bin" "$validator" canary-complete \
    --config "$canary" --repository "$repo_root" \
    --expected-head "$admitted_head" >"$control_root/canary-audit.json"
  touch "$state_root/canary-accepted"
fi
wait_for_resources canary
validate_worktree

phase=training_atomic_to_s32
if [[ ! -f "$state_root/atomic-s32-accepted" ]]; then
  if [[ -e "$formal_root" \
        && ! -s "$formal_root/checkpoints/latest_checkpointed_iteration.txt" \
        && ! -s "$formal_root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json" ]]; then
    echo "Atomic formal root exists without canonical recovery ownership" >&2
    exit 1
  fi
  mkdir -p "$log_root/formal"
  validate_worktree
  env \
    TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=8 \
    TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8 \
    TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2 \
    TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30 \
    TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0 \
    WANDB_RUN_ID=prl26e_train512_atomic_s32 WANDB_RESUME=allow \
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_supervisor \
    --run-config "$formal" --target-step 32 \
    --log-directory "$log_root/formal" \
    --maximum-restarts "$maximum_restarts" \
    --cooldown-seconds "$cooldown_seconds"
  "$python_bin" "$completion_validator" source-complete \
    --config "$formal" --events "$log_root/formal/supervisor-events.jsonl" \
    --target-step 32 >"$control_root/atomic-s32-completion-audit.json"
  if [[ ! -s "$formal_root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json" ]]; then
    echo "Atomic S32 receipt disappeared after validation" >&2
    exit 1
  fi
  touch "$state_root/atomic-s32-accepted"
fi
wait_for_resources atomic
validate_worktree

phase=starting_atomic_evaluation
touch "$state_root/formal-training-complete"
exec "$post_train_eval" "$admitted_head"
