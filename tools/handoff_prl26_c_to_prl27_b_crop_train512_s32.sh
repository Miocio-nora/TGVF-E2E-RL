#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
verl_dependency="$main_root/.deps/verl"

source_config="$repo_root/configs/policy/runs/prl_26_c_qwen3_instruct_short_tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
source_events="$main_root/artifacts/control/PRL-26-tgvf-prompt-parity-20260829/logs/short/supervisor-events.jsonl"
target_config="$repo_root/configs/policy/runs/prl_27_b_qwen3_instruct_full_crop_train512_replay_byte_parity_s32_bs16_n16_teacher25_ws8.toml"
validator="$repo_root/tools/validate_prl27_b_crop_training_handoff.py"
canary_driver="$repo_root/tools/validate_prl27_real_processor_crop_replay.py"

# PRL-27-A's failed root/control tree is never read as recoverable state and is
# never overwritten.  PRL-27-B owns a new output root through target_config and
# this disjoint external control root.
control_root="$main_root/artifacts/control/PRL-27-B-crop-train512-s32-20260830"
state_root="$control_root/state"
log_root="$control_root/logs/formal"
authorization="$control_root/crop-fresh-s0-authorization.json"
canary_receipt="$control_root/real-processor-double-crop-final-replay-canary.json"
target_events="$log_root/supervisor-events.jsonl"

poll_seconds=${PRL27_B_POLL_SECONDS:-5}
release_stable_polls=${PRL27_B_RELEASE_STABLE_POLLS:-3}
release_maximum_polls=${PRL27_B_RELEASE_MAXIMUM_POLLS:-240}
gpu_memory_threshold_mib=${PRL27_B_GPU_IDLE_MEMORY_THRESHOLD_MIB:-32}
maximum_restarts=${PRL27_B_TRAIN_MAXIMUM_RESTARTS:-12}
cooldown_seconds=${PRL27_B_TRAIN_RESTART_COOLDOWN_SECONDS:-15}
admitted_head=${PRL27_B_ADMITTED_HEAD:-}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required before arming PRL-27-B" >&2
  exit 1
fi
if [[ ! "$admitted_head" =~ ^[0-9a-f]{40}$ ]]; then
  echo "PRL27_B_ADMITTED_HEAD is required before arming PRL-27-B" >&2
  exit 1
fi
for path in "$python_bin" "$source_config" "$source_events" "$target_config" \
  "$validator" "$canary_driver"; do
  if [[ ! -f "$path" ]]; then
    echo "required PRL-27-B launch file is absent: $path" >&2
    exit 1
  fi
done
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$release_stable_polls" =~ ^[1-9][0-9]*$ \
      || ! "$release_maximum_polls" =~ ^[1-9][0-9]*$ \
      || ! "$gpu_memory_threshold_mib" =~ ^[0-9]+$ \
      || ! "$maximum_restarts" =~ ^[0-9]+$ \
      || ! "$cooldown_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "PRL-27-B polling or retry setting is malformed" >&2
  exit 1
fi
if (( release_stable_polls < 3 )); then
  echo "PRL-27-B requires at least three consecutive clean resource probes" >&2
  exit 1
fi

if [[ -L "$control_root" \
      || ( -e "$control_root" && ! -d "$control_root" ) ]]; then
  echo "PRL-27-B control root is unsafe" >&2
  exit 1
fi
mkdir -p "$state_root" "$log_root"
if [[ -L "$control_root/handoff.lock" ]]; then
  echo "PRL-27-B handoff lock cannot be a symlink" >&2
  exit 1
fi
exec 9>"$control_root/handoff.lock"
flock -n 9 || {
  echo "another PRL-27-B training supervisor is active" >&2
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
    echo "clean PRL-27-B worktree identity differs" >&2
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
  echo "PRL-27-B admitted HEAD state cannot be a symlink" >&2
  exit 1
fi
if [[ -e "$control_root/admitted-head.txt" ]]; then
  if [[ ! -s "$control_root/admitted-head.txt" \
        || "$(tr -d '\r\n' <"$control_root/admitted-head.txt")" != "$admitted_head" ]]; then
    echo "PRL-27-B control state belongs to another admitted HEAD" >&2
    exit 1
  fi
else
  printf '%s\n' "$admitted_head" >"$control_root/admitted-head.txt"
fi
rm -f "$state_root/failed"

phase=validating_static_contracts
"$python_bin" "$validator" contracts \
  --repository "$repo_root" \
  --source-config "$source_config" \
  --target-config "$target_config" \
  --admitted-head "$admitted_head" \
  --require-clean >"$control_root/static-contract-audit.json"

# Compose without a GPU or target-root write.  This catches config/Hydra drift
# before accepting any launch authorization.
phase=static_compose
env CUDA_VISIBLE_DEVICES= "$python_bin" \
  -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config "$target_config" --mode formal --target-step 32 --compose-only \
  >"$control_root/formal-compose.json"
validate_worktree

# PRL-26-C is already complete.  Its immutable S32 tracker, metrics,
# generation, permanent receipt and successful supervisor terminal event are
# sufficient; no live or retained tmux pane participates in this handoff.
phase=validating_finalized_prl26_c_s32
"$python_bin" "$validator" source-complete \
  --config "$source_config" --events "$source_events" \
  >"$control_root/prl26-c-finalized-source-completion-audit.json"
printf 'source_mode=finalized-receipt\nsource_live_pane_required=false\n' \
  >"$control_root/prl26-c-finalized-source-mode.txt"
validate_worktree

# Preserve each probe and require at least three consecutive all-GPU/Ray-free
# samples.  This supervisor never kills Ray or evicts another CUDA owner.
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
      echo "GPUs or Ray did not reach a stable clean PRL-27-B boundary" >&2
      exit 1
    fi
    sleep "$poll_seconds"
  fi
done
cp "$resource_probe" "$control_root/resources-released.json"
validate_worktree

# The real local Qwen processor must prove two matched Crop turns, exact
# layout/appender bytes, successful final replay, and negative drift rejection.
# The immutable receipt is bound to the same admitted clean HEAD.
phase=proving_real_processor_double_crop_final_replay
if [[ -L "$canary_receipt" ]]; then
  echo "PRL-27-B real-processor canary receipt cannot be a symlink" >&2
  exit 1
fi
if [[ ! -e "$canary_receipt" ]]; then
  env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none \
    "$python_bin" "$canary_driver" \
    --expected-head "$admitted_head" --output "$canary_receipt" \
    >"$control_root/real-processor-canary-driver.stdout.log" \
    2>"$control_root/real-processor-canary-driver.stderr.log"
fi
"$python_bin" "$validator" canary-complete \
  --receipt "$canary_receipt" "$admitted_head" \
  >"$control_root/real-processor-double-crop-final-replay-audit.json"
validate_worktree

# Close the CPU-canary scheduling window with one more fail-closed all-resource
# sample before authorizing the still-absent PRL-27-B policy root.
phase=final_resource_probe
"$python_bin" "$validator" resources-free \
  --memory-threshold-mib "$gpu_memory_threshold_mib" \
  >"$control_root/resource-probe-after-canary.json"
validate_worktree

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
export WANDB_RUN_ID=prl27b_train512_crop_replay_byte_parity_s32
export WANDB_RESUME=allow

phase=training_prl27_b_to_s32
setsid "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_supervisor \
  --run-config "$target_config" \
  --target-step 32 \
  --log-directory "$log_root" \
  --maximum-restarts "$maximum_restarts" \
  --cooldown-seconds "$cooldown_seconds" \
  >>"$control_root/supervisor-console.log" 2>&1 9>&- &
active_pid=$!
wait "$active_pid"
active_pid=

phase=validating_prl27_b_s32
"$python_bin" "$validator" target-complete \
  --config "$target_config" --events "$target_events" \
  >"$control_root/prl27-b-s32-completion-audit.json"
touch "$state_root/s32-accepted"
phase=complete
printf '[%s] PRL-27-B corrected Crop training accepted through S32\n' "$(timestamp)"
