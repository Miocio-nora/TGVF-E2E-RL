#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
verl_dependency="$main_root/.deps/verl"

source_config="$repo_root/configs/policy/runs/prl_26_c_qwen3_instruct_short_tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
target_config="$repo_root/configs/policy/runs/prl_27_a_qwen3_instruct_full_crop_train512_exact_continuation_s32_bs16_n16_teacher25_ws8.toml"
validator="$repo_root/tools/validate_prl27_a_crop_training_handoff.py"

prl26_control_root="$main_root/artifacts/control/PRL-26-tgvf-prompt-parity-20260829"
source_events="$prl26_control_root/logs/short/supervisor-events.jsonl"
source_failed="$prl26_control_root/state/failed"
source_supervisor_lock="$prl26_control_root/supervisor.lock"
source_session=${PRL27_A_SOURCE_TMUX_SESSION:-prl26-cd-tgvf-prompt-s32}

control_root="$main_root/artifacts/control/PRL-27-A-crop-train512-s32-20260829"
state_root="$control_root/state"
log_root="$control_root/logs/formal"
authorization="$control_root/crop-fresh-s0-authorization.json"
target_events="$log_root/supervisor-events.jsonl"

poll_seconds=${PRL27_A_POLL_SECONDS:-15}
release_stable_polls=${PRL27_A_RELEASE_STABLE_POLLS:-3}
release_maximum_polls=${PRL27_A_RELEASE_MAXIMUM_POLLS:-240}
gpu_memory_threshold_mib=${PRL27_A_GPU_IDLE_MEMORY_THRESHOLD_MIB:-32}
source_lock_timeout_seconds=${PRL27_A_SOURCE_LOCK_TIMEOUT_SECONDS:-3600}
maximum_restarts=${PRL27_A_TRAIN_MAXIMUM_RESTARTS:-12}
cooldown_seconds=${PRL27_A_TRAIN_RESTART_COOLDOWN_SECONDS:-15}
admitted_head=${PRL27_A_ADMITTED_HEAD:-}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required before arming PRL-27-A" >&2
  exit 1
fi
if [[ ! "$admitted_head" =~ ^[0-9a-f]{40}$ ]]; then
  echo "PRL27_A_ADMITTED_HEAD is required before arming PRL-27-A" >&2
  exit 1
fi
for path in "$python_bin" "$validator" "$source_config" "$target_config"; do
  if [[ ! -f "$path" ]]; then
    echo "required PRL-27-A handoff file is absent: $path" >&2
    exit 1
  fi
done
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$release_stable_polls" =~ ^[1-9][0-9]*$ \
      || ! "$release_maximum_polls" =~ ^[1-9][0-9]*$ \
      || ! "$gpu_memory_threshold_mib" =~ ^[0-9]+$ \
      || ! "$source_lock_timeout_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$maximum_restarts" =~ ^[0-9]+$ \
      || ! "$cooldown_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "PRL-27-A polling, lock, or retry setting is malformed" >&2
  exit 1
fi
if (( release_stable_polls < 2 )); then
  echo "PRL-27-A requires at least two consecutive clean resource probes" >&2
  exit 1
fi

# Control state is external to the immutable worktree and target output. The
# target root remains absent until the formal launcher owns its first write.
mkdir -p "$state_root" "$log_root"
exec 9>"$control_root/handoff.lock"
flock -n 9 || {
  echo "another PRL-27-A handoff supervisor is active" >&2
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
    echo "clean PRL-27-A worktree identity differs" >&2
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

validate_worktree
if [[ -L "$control_root/admitted-head.txt" ]]; then
  echo "PRL-27-A admitted HEAD state cannot be a symlink" >&2
  exit 1
fi
if [[ -e "$control_root/admitted-head.txt" ]]; then
  if [[ ! -s "$control_root/admitted-head.txt" \
        || "$(tr -d '\r\n' <"$control_root/admitted-head.txt")" != "$admitted_head" ]]; then
    echo "PRL-27-A control state belongs to another admitted HEAD" >&2
    exit 1
  fi
else
  printf '%s\n' "$admitted_head" >"$control_root/admitted-head.txt"
fi
"$python_bin" "$validator" contracts \
  --repository "$repo_root" \
  --source-config "$source_config" \
  --target-config "$target_config" \
  --admitted-head "$admitted_head" \
  --require-clean >"$control_root/static-contract-audit.json"
rm -f "$state_root/failed"

# Compose the exact formal launch without allocating a GPU or creating the
# policy output root. This catches upstream veRL/config drift before waiting.
phase=static_compose
env CUDA_VISIBLE_DEVICES= "$python_bin" \
  -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config "$target_config" --mode formal --target-step 32 --compose-only \
  >"$control_root/formal-compose.json"
validate_worktree

# Accept the source only after the recoverable supervisor has recorded a
# successful terminal attempt and S32 tracker, metrics, rolling generation,
# permanent hard-link generation, and receipt all agree. A live source pane is
# required while that proof is incomplete; a stale receipt alone is rejected.
phase=waiting_for_prl26_c_s32
source_audit_candidate="$control_root/.prl26-c-source-completion-audit.candidate"
source_rejection="$control_root/prl26-c-source-completion-last-rejection.log"
while ! "$python_bin" "$validator" source-complete \
    --config "$source_config" --events "$source_events" \
    >"$source_audit_candidate" 2>"$source_rejection"; do
  if [[ -e "$source_failed" || -L "$source_failed" ]]; then
    echo "PRL-26-C failed before exact S32 completion; PRL-27-A remains untouched" >&2
    exit 1
  fi
  mapfile -t source_panes < <(
    tmux list-panes -t "$source_session" -F '#{pane_id}' 2>/dev/null
  )
  if [[ ${#source_panes[@]} -ne 1 ]]; then
    echo "PRL-26-C source pane is absent or ambiguous before S32" >&2
    exit 1
  fi
  source_pane=${source_panes[0]}
  source_start_command=$(
    tmux display-message -p -t "$source_pane" '#{pane_start_command}'
  )
  source_dead=$(tmux display-message -p -t "$source_pane" '#{pane_dead}')
  source_remain=$(
    tmux show-options -w -v -t "$source_pane" remain-on-exit 2>/dev/null
  )
  if [[ "$source_start_command" != *"supervise_prl26_tgvf_prompt_train_and_eval.sh"* \
        || "$source_dead" != 0 \
        || "$source_remain" != on ]]; then
    echo "PRL-26-C source pane identity/lifecycle differs before S32" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done
mv "$source_audit_candidate" "$control_root/prl26-c-source-completion-audit.json"
rm -f "$source_rejection"

# The current C/D supervisor owns this lock across C, D, and their evaluation.
# Taking the same lock is the fail-closed scheduling boundary: an operator must
# release the old chain after C closes; this handoff never kills it and cannot
# race the already-configured D arm for GPUs. For the authorized 2026-08-29
# reorder, the outer bash alone received SIGTERM while its foreground Short
# child remained healthy. Bash defers that trap until the child closes, then
# the retained source pane must exit 130 and release this lock. Exit 130 is
# therefore positive scheduling evidence here, not a PRL-26-C training failure.
phase=claiming_prl26_cd_schedule
if [[ -L "$source_supervisor_lock" || ! -f "$source_supervisor_lock" ]]; then
  echo "PRL-26 C/D supervisor lock is absent or unsafe" >&2
  exit 1
fi
exec 8>>"$source_supervisor_lock"
if ! flock -w "$source_lock_timeout_seconds" 8; then
  echo "PRL-26 C/D schedule was not released; PRL-27-A was not started" >&2
  exit 1
fi

mapfile -t released_source_panes < <(
  tmux list-panes -t "$source_session" -F '#{pane_id}' 2>/dev/null
)
if [[ ${#released_source_panes[@]} -ne 1 ]]; then
  echo "retained PRL-26 C/D source pane disappeared after schedule release" >&2
  exit 1
fi
released_source_pane=${released_source_panes[0]}
released_source_dead=$(
  tmux display-message -p -t "$released_source_pane" '#{pane_dead}'
)
released_source_status=$(
  tmux display-message -p -t "$released_source_pane" '#{pane_dead_status}'
)
if [[ "$released_source_dead" != 1 || "$released_source_status" != 130 ]]; then
  echo "PRL-26 C/D outer pane did not close at the authorized exit-130 reorder" >&2
  exit 1
fi
if [[ -L "$source_failed" || ! -s "$source_failed" ]] \
    || ! rg -Fxq 'phase=signal' "$source_failed" \
    || ! rg -Fxq 'exit_status=130' "$source_failed"; then
  echo "PRL-26 C/D exit-130 scheduling receipt differs" >&2
  exit 1
fi
printf 'source_session=%s\npane_id=%s\npane_dead=1\npane_dead_status=130\n' \
  "$source_session" "$released_source_pane" \
  >"$control_root/prl26-cd-authorized-reorder-audit.txt"
"$python_bin" "$validator" source-complete \
  --config "$source_config" --events "$source_events" \
  >"$control_root/prl26-c-source-completion-after-lock-audit.json"

# Require consecutive all-clear snapshots. Do not kill Ray, evict a CUDA
# owner, or infer that low utilization means ownership is safe.
phase=waiting_for_gpu_ray_release
quiet_polls=0
total_polls=0
while (( quiet_polls < release_stable_polls )); do
  total_polls=$((total_polls + 1))
  resource_probe="$control_root/resource-probe-${total_polls}.json"
  if "$python_bin" "$validator" resources-free \
      --memory-threshold-mib "$gpu_memory_threshold_mib" >"$resource_probe"; then
    quiet_polls=$((quiet_polls + 1))
  else
    quiet_polls=0
  fi
  if (( quiet_polls < release_stable_polls )); then
    if (( total_polls >= release_maximum_polls )); then
      echo "GPUs or Ray did not reach a stable clean boundary; PRL-27-A was not started" >&2
      exit 1
    fi
    sleep "$poll_seconds"
  fi
done
cp "$resource_probe" "$control_root/resources-released.json"
validate_worktree

# First admission requires an absent canonical root. Recovery requires this
# external authorization plus a committed tracker/generation/metrics chain
# owned by the identical run config; empty or foreign roots fail closed.
phase=authorizing_fresh_s0_or_exact_resume
"$python_bin" "$validator" target-ready \
  --config "$target_config" \
  --authorization "$authorization" \
  --admitted-head "$admitted_head" >"$control_root/target-launch-readiness.json"
validate_worktree

export TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=8
export TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8
export TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2
export TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30
export TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0
export WANDB_RUN_ID=prl27a_train512_crop_exact_continuation_s32
export WANDB_RESUME=allow

phase=training_prl27_a_to_s32
setsid "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_supervisor \
  --run-config "$target_config" \
  --target-step 32 \
  --log-directory "$log_root" \
  --maximum-restarts "$maximum_restarts" \
  --cooldown-seconds "$cooldown_seconds" \
  >>"$control_root/supervisor-console.log" 2>&1 8>&- 9>&- &
active_pid=$!
wait "$active_pid"
active_pid=

phase=validating_prl27_a_s32
"$python_bin" "$validator" target-complete \
  --config "$target_config" --events "$target_events" \
  >"$control_root/prl27-a-s32-completion-audit.json"
touch "$state_root/s32-accepted"
phase=complete
printf '[%s] PRL-27-A corrected Crop training accepted through S32\n' "$(timestamp)"
