#!/usr/bin/env bash
set -euo pipefail

handoff_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
training_repo=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-notool-s32
expected_training_head=40f1728a69e0a3f868117776c80c45ad6de70b8c
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
verl_dependency=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/verl
notool_config="$training_repo/configs/policy/runs/prl_26_a_qwen3_instruct_full_no_tool_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
crop_config="$training_repo/configs/policy/runs/prl_26_b_qwen3_instruct_full_crop_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
control_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/control/PRL-26-train512-formal-20260829/crop-handoff
crop_log_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/control/PRL-26-train512-formal-20260829/crop
notool_event_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/control/PRL-26-train512-formal-20260829/notool
notool_events="$notool_event_root/supervisor-events.jsonl"
authorization="$control_root/crop-fresh-root-authorization.json"
validator="$handoff_repo/tools/validate_prl26_train512_training_handoff.py"
source_session=${PRL26_A_SOURCE_TMUX_SESSION:-prl26a_train512_notool_s32}
poll_seconds=${PRL26_HANDOFF_POLL_SECONDS:-15}
release_stable_polls=${PRL26_RELEASE_STABLE_POLLS:-3}
release_maximum_polls=${PRL26_RELEASE_MAXIMUM_POLLS:-120}
gpu_memory_threshold_mib=${PRL26_GPU_IDLE_MEMORY_THRESHOLD_MIB:-32}
maximum_restarts=${PRL26_B_TRAIN_MAXIMUM_RESTARTS:-8}
cooldown_seconds=${PRL26_B_TRAIN_RESTART_COOLDOWN_SECONDS:-5}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required before arming the PRL-26 Crop handoff" >&2
  exit 1
fi
if [[ ! -f "$validator" || ! -f "$notool_config" || ! -f "$crop_config" ]]; then
  echo "PRL-26 handoff code or formal config is absent" >&2
  exit 1
fi
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
      || ! "$release_stable_polls" =~ ^[1-9][0-9]*$ \
      || ! "$release_maximum_polls" =~ ^[1-9][0-9]*$ \
      || ! "$gpu_memory_threshold_mib" =~ ^[0-9]+$ \
      || ! "$maximum_restarts" =~ ^[0-9]+$ \
      || ! "$cooldown_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "PRL-26 handoff polling or retry setting is malformed" >&2
  exit 1
fi

# Control state and attempt logs live outside both immutable code and Crop output.
# In particular, nothing below creates the Crop output root before the trainer.
mkdir -p "$control_root" "$crop_log_root"
exec 9>"$control_root/handoff.lock"
if ! flock -n 9; then
  echo "another PRL-26 NoTool-to-Crop handoff is active" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$training_repo/src:$verl_dependency${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_REPOSITORY_ROOT="$training_repo"

validate_training_worktree() {
  local observed_root observed_head dirty
  observed_root=$(git -C "$training_repo" rev-parse --show-toplevel)
  observed_head=$(git -C "$training_repo" rev-parse HEAD)
  dirty=$(git -C "$training_repo" status --porcelain=v1 --untracked-files=all)
  if [[ "$observed_root" != "$training_repo" \
        || "$observed_head" != "$expected_training_head" \
        || -n "$dirty" ]]; then
    echo "clean PRL-26 training worktree identity differs" >&2
    exit 1
  fi
}

validate_training_worktree
"$python_bin" "$validator" contracts \
  --training-repository "$training_repo" \
  --notool-config "$notool_config" \
  --crop-config "$crop_config" > "$control_root/static-contract-audit.json"

# Bind to the already-running, remain-on-exit source pane. Session existence by
# itself is insufficient because tmux deliberately retains a dead pane.
mapfile -t source_panes < <(tmux list-panes -t "$source_session" -F '#{pane_id}' 2>/dev/null)
if [[ ${#source_panes[@]} -ne 1 ]]; then
  echo "NoTool source tmux session is absent or ambiguous" >&2
  exit 1
fi
source_pane=${source_panes[0]}
if [[ "$(tmux show-options -w -v -t "$source_pane" remain-on-exit 2>/dev/null)" != "on" ]]; then
  echo "NoTool source pane does not retain an auditable exit status" >&2
  exit 1
fi
source_start_command=$(tmux display-message -p -t "$source_pane" '#{pane_start_command}')
if [[ "$source_start_command" != *"trainable_tgvf_supervisor"* \
      || "$source_start_command" != *"--run-config $notool_config"* \
      || "$source_start_command" != *"--target-step 32"* ]]; then
  echo "NoTool source pane command identity differs" >&2
  exit 1
fi

while true; do
  source_dead=$(tmux display-message -p -t "$source_pane" '#{pane_dead}' 2>/dev/null) || {
    echo "NoTool source pane disappeared before its exit could be audited" >&2
    exit 1
  }
  if [[ "$source_dead" == 1 ]]; then
    break
  fi
  if [[ "$source_dead" != 0 ]]; then
    echo "NoTool source pane lifecycle state is malformed" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done
source_status=$(tmux display-message -p -t "$source_pane" '#{pane_dead_status}')
if [[ "$source_status" != 0 ]]; then
  echo "NoTool source supervisor exited nonzero: $source_status" >&2
  exit 1
fi

# Exit zero is necessary but not sufficient: accept only an exact S32 tracker,
# closed permanent receipt, exact project/pair identity, and 32 finite metrics.
"$python_bin" "$validator" source-complete \
  --config "$notool_config" \
  --events "$notool_events" \
  --target-step 32 > "$control_root/notool-source-completion-audit.json"

# Require three consecutive all-clear snapshots. We neither kill Ray nor evict
# another CUDA owner: unexplained residue times out and leaves Crop untouched.
quiet_polls=0
total_polls=0
while (( quiet_polls < release_stable_polls )); do
  total_polls=$((total_polls + 1))
  resource_probe="$control_root/resource-probe-${total_polls}.json"
  if "$python_bin" "$validator" resources-free \
      --memory-threshold-mib "$gpu_memory_threshold_mib" > "$resource_probe"; then
    quiet_polls=$((quiet_polls + 1))
  else
    quiet_polls=0
  fi
  if (( quiet_polls < release_stable_polls )); then
    if (( total_polls >= release_maximum_polls )); then
      echo "GPUs or Ray did not reach a stable clean boundary; Crop was not started" >&2
      exit 1
    fi
    sleep "$poll_seconds"
  fi
done
cp "$resource_probe" "$control_root/resources-released.json"

# The first launch is authorized only while the canonical root does not exist.
# A later recovery requires this external authorization plus a tracker whose
# paired project state belongs to the exact Crop config; an empty/foreign root
# is rejected. The validator never creates crop_root.
"$python_bin" "$validator" target-ready \
  --config "$crop_config" \
  --authorization "$authorization" \
  --training-head "$expected_training_head" > "$control_root/crop-launch-readiness.json"
validate_training_worktree

export TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=8
export TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8
export TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2
export TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30
export TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0
export WANDB_ENTITY=mio_nora
export WANDB_PROJECT=tgvf-policy-rl
export WANDB_RUN_ID=prl26btrain512crops32pixel512
export WANDB_RESUME=allow
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
unset RAY_ADDRESS

exec "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_supervisor \
  --run-config "$crop_config" \
  --target-step 32 \
  --log-directory "$crop_log_root" \
  --maximum-restarts "$maximum_restarts" \
  --cooldown-seconds "$cooldown_seconds"
