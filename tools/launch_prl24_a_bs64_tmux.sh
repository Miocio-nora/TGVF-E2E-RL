#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
session=prl24_a_bs64
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-24-A-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-16step-ws8
top_log="$training_root/logs/tmux-supervisor.log"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

mkdir -p "$(dirname "$top_log")"
# tmux servers can predate the calling shell.  Transfer only the already
# configured task credential; never serialize it into a repository file.
tmux set-environment -g OPENROUTER_API_KEY "$OPENROUTER_API_KEY"
tmux new-session -d -s "$session" \
  "cd '$repo_root' && exec ./tools/supervise_prl24_a_bs64_16step_and_eval.sh 2>&1 | tee -a '$top_log'"

echo "session=$session"
echo "log=$top_log"
