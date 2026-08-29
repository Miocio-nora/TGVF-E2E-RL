#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
verl_dependency="$main_root/.deps/verl"
control_root="$main_root/artifacts/control/PRL-26-tgvf-prompt-parity-20260829"
state_root="$control_root/state"
log_root="$control_root/logs"
validator="$repo_root/tools/validate_prl26_tgvf_prompt_handoff.py"
resource_validator="$repo_root/tools/validate_prl26_train512_training_handoff.py"
completion_validator="$repo_root/tools/validate_prl26_train512_training_handoff.py"
post_train_eval="$repo_root/tools/supervise_prl26_tgvf_prompt_s32_evaluation.sh"

short_c0="$repo_root/configs/policy/runs/prl_26_c_c0_qwen3_instruct_short_tgvf_train512_parity_bs4_n2_teacher25_1step_ws4.toml"
full_c0="$repo_root/configs/policy/runs/prl_26_d_c0_qwen3_instruct_target_guide_v2_tgvf_train512_parity_bs4_n2_teacher25_1step_ws4.toml"
short_formal="$repo_root/configs/policy/runs/prl_26_c_qwen3_instruct_short_tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
full_formal="$repo_root/configs/policy/runs/prl_26_d_qwen3_instruct_target_guide_v2_tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml"

short_c0_root="$main_root/artifacts/policy/PRL-26-C-C0-train512-parity-tgvf-short-qwen3-instruct-bs4-n2-teacher25-ws4"
full_c0_root="$main_root/artifacts/policy/PRL-26-D-C0-train512-parity-tgvf-target-guide-v2-qwen3-instruct-bs4-n2-teacher25-ws4"
short_root="$main_root/artifacts/policy/PRL-26-C-train512-s32-parity-tgvf-short-qwen3-instruct-bs16-n16-teacher25-ws8"
full_root="$main_root/artifacts/policy/PRL-26-D-train512-s32-parity-tgvf-target-guide-v2-qwen3-instruct-bs16-n16-teacher25-ws8"

prerequisite_root="$main_root/artifacts/evaluation/PRL26-TRAIN512-S32-PIXEL512-COREDEV2511-V1"
prerequisite_result="$prerequisite_root/train512-s32-pixel512-results.json"
prerequisite_complete="$prerequisite_root/runtime/evaluation-complete"
prerequisite_failed="$prerequisite_root/runtime/failed"
prerequisite_session=${PRL26_AB_EVAL_TMUX_SESSION:-prl26-train512-s32-eval}

poll_seconds=${PRL26_CD_POLL_SECONDS:-30}
release_stable_polls=${PRL26_CD_RELEASE_STABLE_POLLS:-2}
release_maximum_polls=${PRL26_CD_RELEASE_MAXIMUM_POLLS:-240}
gpu_memory_threshold_mib=${PRL26_CD_GPU_IDLE_MEMORY_THRESHOLD_MIB:-32}
maximum_restarts=${PRL26_CD_TRAIN_MAXIMUM_RESTARTS:-12}
cooldown_seconds=${PRL26_CD_TRAIN_RESTART_COOLDOWN_SECONDS:-15}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for PRL-26 C/D" >&2
  exit 1
fi
for path in "$validator" "$resource_validator" "$completion_validator" \
  "$post_train_eval" "$short_c0" "$full_c0" "$short_formal" "$full_formal"; do
  if [[ ! -f "$path" ]]; then
    echo "required PRL-26 C/D file is absent: $path" >&2
    exit 1
  fi
done
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$release_stable_polls" =~ ^[1-9][0-9]*$ \
      || ! "$release_maximum_polls" =~ ^[1-9][0-9]*$ \
      || ! "$gpu_memory_threshold_mib" =~ ^[0-9]+$ \
      || ! "$maximum_restarts" =~ ^[0-9]+$ \
      || ! "$cooldown_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "PRL-26 C/D polling or retry setting is malformed" >&2
  exit 1
fi

mkdir -p "$state_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
flock -n 9 || {
  echo "another PRL-26 C/D prompt supervisor is active" >&2
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
active_pids=()

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
    echo "clean PRL-26 C/D worktree identity differs" >&2
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
  local pid
  for pid in "${active_pids[@]:-}"; do
    stop_process_group "$pid"
  done
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
"$python_bin" "$validator" contracts \
  --repository "$repo_root" \
  --short-canary "$short_c0" --full-canary "$full_c0" \
  --short-formal "$short_formal" --full-formal "$full_formal" \
  >"$control_root/static-contract-audit.json"
printf '%s\n' "$admitted_head" >"$control_root/admitted-head.txt"
rm -f "$state_root/failed"

phase=waiting_for_prl26_ab_evaluation
mapfile -t prerequisite_panes < <(
  tmux list-panes -t "$prerequisite_session" -F '#{pane_id}' 2>/dev/null
)
if [[ ${#prerequisite_panes[@]} -ne 1 ]]; then
  echo "PRL-26 A/B evaluation pane is absent or ambiguous" >&2
  exit 1
fi
prerequisite_pane=${prerequisite_panes[0]}
prerequisite_command=$(tmux display-message -p -t "$prerequisite_pane" '#{pane_start_command}')
if [[ "$prerequisite_command" != *"supervise_prl26_train512_s32_coredev2511.sh"* ]]; then
  echo "PRL-26 A/B evaluation pane identity differs" >&2
  exit 1
fi
prerequisite_remain=$(
  tmux show-options -t "$prerequisite_session" -v remain-on-exit 2>/dev/null
)
if [[ "$prerequisite_remain" != on ]]; then
  echo "PRL-26 A/B evaluation pane is not retained for exit-status audit" >&2
  exit 1
fi
while [[ ! -f "$prerequisite_complete" ]]; do
  if [[ -s "$prerequisite_failed" ]]; then
    echo "PRL-26 A/B evaluation failed; C/D remains untouched" >&2
    exit 1
  fi
  if ! prerequisite_dead=$(
    tmux display-message -p -t "$prerequisite_pane" '#{pane_dead}' 2>/dev/null
  ); then
    echo "PRL-26 A/B evaluator pane disappeared before completion" >&2
    exit 1
  fi
  if [[ "$prerequisite_dead" == 1 ]]; then
    echo "PRL-26 A/B evaluator exited before publishing completion" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done
while true; do
  if ! prerequisite_dead=$(
    tmux display-message -p -t "$prerequisite_pane" '#{pane_dead}' 2>/dev/null
  ); then
    echo "PRL-26 A/B evaluator pane disappeared after completion" >&2
    exit 1
  fi
  [[ "$prerequisite_dead" == 1 ]] && break
  sleep 1
done
if ! prerequisite_status=$(
  tmux display-message -p -t "$prerequisite_pane" '#{pane_dead_status}' 2>/dev/null
); then
  echo "PRL-26 A/B evaluator exit status is unavailable" >&2
  exit 1
fi
if [[ "$prerequisite_status" != 0 ]]; then
  echo "PRL-26 A/B evaluator exited nonzero" >&2
  exit 1
fi
"$python_bin" "$validator" prerequisite \
  --result "$prerequisite_result" \
  --complete-marker "$prerequisite_complete" \
  --failed-marker "$prerequisite_failed" \
  >"$control_root/prerequisite-evaluation-audit.json"
wait_for_resources prerequisite
validate_worktree
touch "$state_root/prerequisite-accepted"

phase=running_parallel_c0
if [[ ! -s "$state_root/canaries-accepted" ]]; then
  if [[ ! -e "$short_c0_root" && ! -e "$full_c0_root" ]]; then
    mkdir -p "$log_root/c0-short" "$log_root/c0-full"
    setsid env \
      TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=4 \
      TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8 \
      TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2 \
      TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30 \
      TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0 \
      "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$short_c0" --mode canary --target-step 1 \
      >"$log_root/c0-short/run.log" 2>&1 9>&- &
    short_c0_pid=$!
    active_pids+=("$short_c0_pid")
    setsid env \
      TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=4 \
      TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8 \
      TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2 \
      TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30 \
      TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0 \
      "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$full_c0" --mode canary --target-step 1 \
      >"$log_root/c0-full/run.log" 2>&1 9>&- &
    full_c0_pid=$!
    active_pids+=("$full_c0_pid")
    wait "$short_c0_pid"
    wait "$full_c0_pid"
    active_pids=()
  elif [[ ! -e "$short_c0_root" || ! -e "$full_c0_root" ]]; then
    echo "only one C/D canary root exists; refusing an asymmetric rerun" >&2
    exit 1
  fi
  "$python_bin" "$validator" canary-complete \
    --config "$short_c0" --repository "$repo_root" \
    --expected-head "$admitted_head" >"$control_root/short-canary-audit.json"
  "$python_bin" "$validator" canary-complete \
    --config "$full_c0" --repository "$repo_root" \
    --expected-head "$admitted_head" >"$control_root/full-canary-audit.json"
  touch "$state_root/canaries-accepted"
fi
wait_for_resources canaries
validate_worktree

run_formal() {
  local label=$1
  local config=$2
  local root=$3
  local wandb_id=$4
  local marker="$state_root/${label}-s32-accepted"
  local arm_log="$log_root/$label"
  if [[ -s "$marker" ]]; then
    return 0
  fi
  mkdir -p "$arm_log"
  phase="training_${label}_to_s32"
  validate_worktree
  env \
    TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=8 \
    TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8 \
    TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2 \
    TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30 \
    TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0 \
    WANDB_RUN_ID="$wandb_id" WANDB_RESUME=allow \
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_supervisor \
    --run-config "$config" --target-step 32 \
    --log-directory "$arm_log" \
    --maximum-restarts "$maximum_restarts" \
    --cooldown-seconds "$cooldown_seconds"
  "$python_bin" "$completion_validator" source-complete \
    --config "$config" --events "$arm_log/supervisor-events.jsonl" \
    --target-step 32 >"$control_root/${label}-s32-completion-audit.json"
  if [[ ! -s "$root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json" ]]; then
    echo "$label S32 receipt disappeared after validation" >&2
    return 1
  fi
  touch "$marker"
}

run_formal short "$short_formal" "$short_root" prl26c_train512_tgvf_short_s32
wait_for_resources short
run_formal full "$full_formal" "$full_root" prl26d_train512_tgvf_target_v2_s32
wait_for_resources full
validate_worktree

phase=starting_tgvf_prompt_evaluation
touch "$state_root/formal-training-complete"
exec "$post_train_eval" "$admitted_head"
