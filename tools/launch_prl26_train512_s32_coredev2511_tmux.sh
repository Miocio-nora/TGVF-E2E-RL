#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
session=${PRL26_EVAL_TMUX_SESSION:-prl26-train512-s32-eval}

if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

tmux new-session -d -s "$session" \
  "cd '$repo_root' && exec '$repo_root/tools/supervise_prl26_train512_s32_coredev2511.sh'"
if ! tmux set-option -t "$session" remain-on-exit on; then
  tmux kill-session -t "$session" 2>/dev/null || true
  echo "failed to retain the PRL-26 A/B evaluator pane" >&2
  exit 1
fi
if [[ "$(tmux show-options -t "$session" -v remain-on-exit)" != on ]]; then
  tmux kill-session -t "$session" 2>/dev/null || true
  echo "PRL-26 A/B evaluator pane retention was not admitted" >&2
  exit 1
fi
echo "$session"
