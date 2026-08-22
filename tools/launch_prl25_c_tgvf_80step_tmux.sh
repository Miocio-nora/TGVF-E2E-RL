#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
session=prl25_c_tgvf_80step
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8
top_log="$training_root/logs/tmux-train-and-eval.log"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

mkdir -p "$(dirname "$top_log")"
# Transfer credentials through tmux's environment, never through command text
# or repository files. The supervisor owns training and automatic evaluation.
tmux set-environment -g OPENROUTER_API_KEY "$OPENROUTER_API_KEY"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  tmux set-environment -g WANDB_API_KEY "$WANDB_API_KEY"
fi
tmux new-session -d -s "$session" \
  "cd '$repo_root' && exec ./tools/supervise_prl25_c_tgvf_80step_and_eval.sh 2>&1 | tee -a '$top_log'"

sleep 2
if [[ "$(tmux display-message -p -t "$session" '#{pane_dead}')" == "1" ]]; then
  echo "PRL25-C supervisor exited during startup; inspect $top_log" >&2
  exit 1
fi

echo "session=$session"
echo "log=$top_log"
echo "automatic_evaluation=S8,S16,S32,S48,S64,S80"
