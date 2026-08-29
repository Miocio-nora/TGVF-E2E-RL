#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
session=${PRL26_CD_TMUX_SESSION:-prl26-cd-tgvf-prompt-s32}
control_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/control/PRL-26-tgvf-prompt-parity-20260829
top_log="$control_root/supervisor.log"
supervisor="$repo_root/tools/supervise_prl26_tgvf_prompt_train_and_eval.sh"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
if [[ ! -x "$supervisor" ]]; then
  echo "PRL-26 C/D supervisor is absent or not executable" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi
if [[ -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "PRL-26 C/D launch requires a clean worktree" >&2
  exit 1
fi

mkdir -p "$control_root"
# Credentials move through the tmux server environment, never through command
# text, logs, configs or repository files.
tmux set-environment -g OPENROUTER_API_KEY "$OPENROUTER_API_KEY"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  tmux set-environment -g WANDB_API_KEY "$WANDB_API_KEY"
fi
tmux new-session -d -s "$session" \
  "cd '$repo_root' && exec '$supervisor' >> '$top_log' 2>&1"
tmux set-option -t "$session" remain-on-exit on

sleep 2
if [[ "$(tmux display-message -p -t "$session" '#{pane_dead}')" == 1 ]]; then
  echo "PRL-26 C/D supervisor exited during startup; inspect $top_log" >&2
  exit 1
fi

echo "session=$session"
echo "log=$top_log"
echo "phase=wait A/B Eval@512 -> C0 Short/Full -> Short S32 -> Full S32 -> paired eval"
