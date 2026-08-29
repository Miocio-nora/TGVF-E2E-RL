#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
supervisor="$repo_root/tools/supervise_prl26_b_generic86_s32_evaluation.sh"
attempt=${PRL26_B_GENERIC86_EVAL_ATTEMPT:-0}
if [[ ! "$attempt" =~ ^[0-9]+$ ]]; then
  echo "PRL-26-B generic86 evaluation attempt is malformed" >&2
  exit 1
fi
session=prl26-b-generic86-s32-eval512
control_name=PRL-26-B-generic86-s32-eval512-20260830
if (( attempt > 0 )); then
  session="${session}-r${attempt}"
  control_name="${control_name}-recovery${attempt}"
fi
control_root="$main_root/artifacts/control/$control_name"
log_path="$control_root/tmux-launch.log"

if [[ ! -x "$supervisor" ]]; then
  echo "generic86 evaluation supervisor is absent or not executable" >&2
  exit 1
fi
if [[ -L "$control_root" || ( -e "$control_root" && ! -d "$control_root" ) ]]; then
  echo "generic86 evaluation control root is unsafe" >&2
  exit 1
fi
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "generic86 evaluation tmux session already exists: $session" >&2
  exit 1
fi

observed_root=$(git -C "$repo_root" rev-parse --show-toplevel)
admitted_head=$(git -C "$repo_root" rev-parse HEAD)
dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
if [[ "$observed_root" != "$repo_root" \
      || ! "$admitted_head" =~ ^[0-9a-f]{40}$ \
      || -n "$dirty" ]]; then
  echo "generic86 evaluation requires one clean committed worktree" >&2
  exit 1
fi

mkdir -p "$control_root"
gate="prl26-b-generic86-eval-launch-${admitted_head}"
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
  -e "PRL26_B_GENERIC86_EVAL_ADMITTED_HEAD=$admitted_head" \
  -e "PRL26_B_GENERIC86_EVAL_ATTEMPT=$attempt" "$command"
tmux set-option -w -t "${session}:" remain-on-exit on
pane_count=$(tmux list-panes -t "=$session" -F '#{pane_id}' | wc -l)
remain=$(tmux show-options -w -v -t "${session}:" remain-on-exit)
pane_dead=$(tmux display-message -p -t "${session}:" '#{pane_dead}')
if [[ "$pane_count" != 1 || "$remain" != on || "$pane_dead" != 0 ]]; then
  echo "generic86 evaluation tmux admission differs" >&2
  exit 1
fi
observed_head=$(git -C "$repo_root" rev-parse HEAD)
dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
if [[ "$observed_head" != "$admitted_head" || -n "$dirty" ]]; then
  echo "generic86 evaluation worktree changed during tmux admission" >&2
  exit 1
fi
tmux wait-for -S "$gate"
trap - EXIT
printf 'PRL-26-B generic86 evaluation queued: session=%s admitted_head=%s\n' \
  "$session" "$admitted_head"
