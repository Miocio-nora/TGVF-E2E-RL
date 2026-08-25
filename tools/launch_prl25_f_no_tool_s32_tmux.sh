#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
session=prl25_f_no_tool_s32
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8
top_log="$training_root/logs/tmux-train.log"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

# Formal PRL25-F owns all eight accelerators. Reject a launch that would share
# them with an unclosed canary or an unrelated process.
mapfile -t gpu_memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ "${#gpu_memory[@]}" != 8 ]]; then
  echo "PRL25-F requires exactly eight visible GPUs" >&2
  exit 1
fi
for index in "${!gpu_memory[@]}"; do
  if (( gpu_memory[index] > 128 )); then
    echo "GPU $index is not idle (${gpu_memory[index]} MiB used)" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$top_log")"
# Transfer credentials through tmux's environment, never through command text
# or repository files.
tmux set-environment -g OPENROUTER_API_KEY "$OPENROUTER_API_KEY"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  tmux set-environment -g WANDB_API_KEY "$WANDB_API_KEY"
fi
tmux new-session -d -s "$session" \
  "cd '$repo_root' && exec ./tools/supervise_prl25_f_no_tool_s32.sh 2>&1 | tee -a '$top_log'"

sleep 2
if [[ "$(tmux display-message -p -t "$session" '#{pane_dead}')" == "1" ]]; then
  echo "PRL25-F supervisor exited during startup; inspect $top_log" >&2
  exit 1
fi

echo "session=$session"
echo "log=$top_log"
echo "target=S32"
echo "permanent_checkpoints=S8,S16,S32"
