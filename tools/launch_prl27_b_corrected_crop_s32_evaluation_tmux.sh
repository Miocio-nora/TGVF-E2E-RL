#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
supervisor="$repo_root/tools/supervise_prl27_b_corrected_crop_s32_evaluation.sh"
session=prl27-b-crop-s32-eval512
control_root="$main_root/artifacts/control/PRL-27-B-corrected-crop-s32-eval512-20260830"
log_path="$control_root/tmux-launch.log"

if [[ ! -x "$supervisor" ]]; then
  echo "required PRL-27-B evaluation supervisor is absent or not executable" >&2
  exit 1
fi
if [[ -L "$control_root" || ( -e "$control_root" && ! -d "$control_root" ) ]]; then
  echo "PRL-27-B evaluation control root is unsafe" >&2
  exit 1
fi
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "PRL-27-B evaluation tmux session already exists: $session" >&2
  exit 1
fi

observed_root=$(git -C "$repo_root" rev-parse --show-toplevel)
admitted_head=$(git -C "$repo_root" rev-parse HEAD)
dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
if [[ "$observed_root" != "$repo_root" \
      || ! "$admitted_head" =~ ^[0-9a-f]{40}$ \
      || -n "$dirty" ]]; then
  echo "PRL-27-B evaluation requires one clean committed worktree" >&2
  exit 1
fi

mkdir -p "$control_root"
gate="prl27-b-eval-launch-${admitted_head}"
command=$(printf \
  'tmux wait-for %q; exec %q >>%q 2>&1' \
  "$gate" "$supervisor" "$log_path")

cleanup_unarmed_session() {
  local status=$?
  if (( status != 0 )); then
    tmux kill-session -t "=$session" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup_unarmed_session EXIT

tmux new-session -d -E \
  -s "$session" -c "$repo_root" \
  -e "PRL27_B_EVAL_ADMITTED_HEAD=$admitted_head" "$command"
tmux set-option -w -t "${session}:" remain-on-exit on

pane_count=$(tmux list-panes -t "=$session" -F '#{pane_id}' | wc -l)
remain=$(tmux show-options -w -v -t "${session}:" remain-on-exit)
pane_dead=$(tmux display-message -p -t "${session}:" '#{pane_dead}')
if [[ "$pane_count" != 1 || "$remain" != on || "$pane_dead" != 0 ]]; then
  echo "PRL-27-B evaluation tmux admission differs" >&2
  exit 1
fi

observed_head=$(git -C "$repo_root" rev-parse HEAD)
dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
if [[ "$observed_head" != "$admitted_head" || -n "$dirty" ]]; then
  echo "PRL-27-B evaluation worktree changed during tmux admission" >&2
  exit 1
fi

tmux wait-for -S "$gate"
trap - EXIT
printf 'PRL-27-B evaluation queued: session=%s admitted_head=%s\n' \
  "$session" "$admitted_head"
