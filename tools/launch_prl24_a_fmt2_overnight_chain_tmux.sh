#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
session=prl24_a_fmt2_overnight
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-24-A-FMT2-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-8step-ws8
top_log="$training_root/logs/overnight-step8-to16-eval-joint.log"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

mkdir -p "$(dirname "$top_log")"
tmux set-environment -g OPENROUTER_API_KEY "$OPENROUTER_API_KEY"
tmux new-session -d -s "$session" \
  "cd '$repo_root' && exec ./tools/supervise_prl24_a_fmt2_step8_to16_eval_and_joint.sh 2>&1 | tee -a '$top_log'"

echo "session=$session"
echo "log=$top_log"

