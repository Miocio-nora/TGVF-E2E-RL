#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
session=prl25_d_after_c_eval
c_evaluation_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-C-FROZEN-RP67-TFREE-TEACHER25-COREDEV2511-S8-S16-S32-S48-S64-S80-PAIRED-SEED-V1
c_complete="$c_evaluation_root/runtime/supervisor/evaluation-complete"
d_training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-25-D-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-80step-ws8
top_log="$d_training_root/logs/tmux-after-c-train-and-eval.log"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

mkdir -p "$(dirname "$top_log")"
tmux new-session -d -s "$session" \
  "cd '$repo_root' && while [[ ! -f '$c_complete' ]]; do sleep 15; done; sleep 5; exec ./tools/supervise_prl25_d_atomic_80step_and_eval.sh 2>&1 | tee -a '$top_log'"

sleep 2
if [[ "$(tmux display-message -p -t "$session" '#{pane_dead}')" == "1" ]]; then
  echo "PRL25-D handoff supervisor exited during startup; inspect $top_log" >&2
  exit 1
fi

echo "session=$session"
echo "waiting_for=$c_complete"
echo "next_training=PRL25-D Atomic Crop+TGVF S0->S80"
echo "automatic_evaluation=S8,S16,S32,S48,S64,S80"
echo "log=$top_log"
