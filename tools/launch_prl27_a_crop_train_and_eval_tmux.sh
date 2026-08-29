#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
training_handoff="$repo_root/tools/handoff_prl26_c_to_prl27_a_crop_train512_s32.sh"
evaluation_waiter="$repo_root/tools/supervise_prl27_a_corrected_crop_s32_evaluation.sh"
training_session=${PRL27_A_TRAIN_TMUX_SESSION:-prl27-a-crop-train512-s32}
evaluation_session=${PRL27_A_EVAL_TMUX_SESSION:-prl27-a-crop-s32-eval512}
training_log="$main_root/artifacts/control/PRL-27-A-crop-train512-s32-20260829/tmux-launch.log"
evaluation_log="$main_root/artifacts/control/PRL-27-A-corrected-crop-s32-eval512-20260829/tmux-launch.log"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required before arming PRL-27-A" >&2
  exit 1
fi
for path in "$training_handoff" "$evaluation_waiter"; do
  if [[ ! -x "$path" ]]; then
    echo "required PRL-27-A supervisor is absent or not executable: $path" >&2
    exit 1
  fi
done
for name in "$training_session" "$evaluation_session"; do
  if [[ ! "$name" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "PRL-27-A tmux session name is unsafe: $name" >&2
    exit 1
  fi
  if tmux has-session -t "=$name" 2>/dev/null; then
    echo "PRL-27-A tmux session already exists: $name" >&2
    exit 1
  fi
done

observed_root=$(git -C "$repo_root" rev-parse --show-toplevel)
admitted_head=$(git -C "$repo_root" rev-parse HEAD)
dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
if [[ "$observed_root" != "$repo_root" \
      || ! "$admitted_head" =~ ^[0-9a-f]{40}$ \
      || -n "$dirty" ]]; then
  echo "PRL-27-A requires one clean committed worktree" >&2
  exit 1
fi

mkdir -p "$(dirname "$training_log")" "$(dirname "$evaluation_log")"
launch_gate="prl27-a-launch-${admitted_head}"
training_command=$(printf \
  'tmux wait-for %q; exec %q >>%q 2>&1' \
  "$launch_gate" "$training_handoff" "$training_log")
evaluation_command=$(printf \
  'tmux wait-for %q; exec %q >>%q 2>&1' \
  "$launch_gate" "$evaluation_waiter" "$evaluation_log")

# Secrets are session-scoped tmux environment entries. They are never embedded
# in pane_start_command, shell source, log paths, or printed launch receipts.
training_environment=(
  -e "OPENROUTER_API_KEY=$OPENROUTER_API_KEY"
  -e "PRL27_A_ADMITTED_HEAD=$admitted_head"
)
evaluation_environment=(
  -e "OPENROUTER_API_KEY=$OPENROUTER_API_KEY"
  -e "PRL27_A_ADMITTED_HEAD=$admitted_head"
)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  training_environment+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
fi

created_sessions=()
cleanup_unarmed_sessions() {
  local status=$?
  if (( status != 0 )); then
    local session
    for session in "${created_sessions[@]:-}"; do
      tmux kill-session -t "=$session" 2>/dev/null || true
    done
  fi
  exit "$status"
}
trap cleanup_unarmed_sessions EXIT

# Both panes first block on a tmux gate. Only after both retained sessions and
# their scoped environments exist is the gate signalled, so partial arming
# cannot silently start one half of the pipeline.
tmux new-session -d -E \
  -s "$training_session" -c "$repo_root" \
  "${training_environment[@]}" "$training_command"
created_sessions+=("$training_session")
tmux set-option -w -t "=$training_session" remain-on-exit on

tmux new-session -d -E \
  -s "$evaluation_session" -c "$repo_root" \
  "${evaluation_environment[@]}" "$evaluation_command"
created_sessions+=("$evaluation_session")
tmux set-option -w -t "=$evaluation_session" remain-on-exit on

for session in "$training_session" "$evaluation_session"; do
  pane_count=$(tmux list-panes -t "=$session" -F '#{pane_id}' | wc -l)
  remain=$(tmux show-options -w -v -t "=$session" remain-on-exit)
  if [[ "$pane_count" != 1 || "$remain" != on ]]; then
    echo "PRL-27-A tmux session admission differs: $session" >&2
    exit 1
  fi
done

tmux wait-for -S "$launch_gate"
trap - EXIT
printf 'PRL-27-A queued: training=%s evaluation=%s admitted_head=%s\n' \
  "$training_session" "$evaluation_session" "$admitted_head"
