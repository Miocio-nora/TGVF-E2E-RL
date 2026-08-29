#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
supervisor="$repo_root/tools/supervise_prl26_e_atomic_train_and_eval.sh"
session=${PRL26_E_TMUX_SESSION:-prl26-e-atomic-train512-s32}
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
control_root="$main_root/artifacts/control/PRL-26-E-atomic-train512-s32-20260829"
launch_log="$control_root/tmux-launch.log"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for PRL-26-E Atomic training" >&2
  exit 1
fi
if [[ ! -x "$supervisor" ]]; then
  echo "PRL-26-E Atomic supervisor is absent or not executable" >&2
  exit 1
fi
observed_root=$(git -C "$repo_root" rev-parse --show-toplevel)
dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
if [[ "$observed_root" != "$repo_root" || -n "$dirty" ]]; then
  echo "PRL-26-E launcher requires its exact clean repository" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

mkdir -p "$control_root"
tmux new-session -d -s "$session" \
  "exec '$supervisor' >>'$launch_log' 2>&1"
tmux set-option -t "$session" remain-on-exit on
printf 'started tmux session %s at HEAD %s\n' \
  "$session" "$(git -C "$repo_root" rev-parse HEAD)"
