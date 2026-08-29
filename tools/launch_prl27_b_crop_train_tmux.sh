#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
training_handoff="$repo_root/tools/handoff_prl26_c_to_prl27_b_crop_train512_s32.sh"
training_session=prl27-b-crop-train512-s32
training_log="$main_root/artifacts/control/PRL-27-B-crop-train512-s32-20260830/tmux-launch.log"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required before arming PRL-27-B" >&2
  exit 1
fi
if [[ ! -x "$training_handoff" ]]; then
  echo "required PRL-27-B training supervisor is absent or not executable" >&2
  exit 1
fi
if tmux has-session -t "=$training_session" 2>/dev/null; then
  echo "PRL-27-B tmux session already exists: $training_session" >&2
  exit 1
fi

observed_root=$(git -C "$repo_root" rev-parse --show-toplevel)
admitted_head=$(git -C "$repo_root" rev-parse HEAD)
dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
if [[ "$observed_root" != "$repo_root" \
      || ! "$admitted_head" =~ ^[0-9a-f]{40}$ \
      || -n "$dirty" ]]; then
  echo "PRL-27-B requires one clean committed worktree" >&2
  exit 1
fi

mkdir -p "$(dirname "$training_log")"
launch_gate="prl27-b-launch-${admitted_head}"
training_command=$(printf \
  'tmux wait-for %q; exec %q >>%q 2>&1' \
  "$launch_gate" "$training_handoff" "$training_log")

# Secrets and the immutable admitted HEAD are scoped to the tmux session; none
# is embedded in pane_start_command or printed by the launcher.
training_environment=(
  -e "OPENROUTER_API_KEY=$OPENROUTER_API_KEY"
  -e "PRL27_B_ADMITTED_HEAD=$admitted_head"
)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  training_environment+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
fi

cleanup_unarmed_session() {
  local status=$?
  if (( status != 0 )); then
    tmux kill-session -t "=$training_session" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup_unarmed_session EXIT

tmux new-session -d -E \
  -s "$training_session" -c "$repo_root" \
  "${training_environment[@]}" "$training_command"
tmux set-option -w -t "${training_session}:" remain-on-exit on

pane_count=$(tmux list-panes -t "=$training_session" -F '#{pane_id}' | wc -l)
remain=$(tmux show-options -w -v -t "${training_session}:" remain-on-exit)
pane_dead=$(tmux display-message -p -t "${training_session}:" '#{pane_dead}')
pane_start_command=$(
  tmux display-message -p -t "${training_session}:" '#{pane_start_command}'
)
if [[ "$pane_count" != 1 || "$remain" != on || "$pane_dead" != 0 \
      || "$pane_start_command" == *"OPENROUTER_API_KEY"* ]]; then
  echo "PRL-27-B tmux session admission differs" >&2
  exit 1
fi

# The pane is still blocked on the launch gate.  Abort it if HEAD or worktree
# state changed between admission and signal.
observed_head=$(git -C "$repo_root" rev-parse HEAD)
dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
if [[ "$observed_head" != "$admitted_head" || -n "$dirty" ]]; then
  echo "PRL-27-B worktree changed while tmux was being admitted" >&2
  exit 1
fi

tmux wait-for -S "$launch_gate"
trap - EXIT
printf 'PRL-27-B queued: training=%s admitted_head=%s\n' \
  "$training_session" "$admitted_head"
