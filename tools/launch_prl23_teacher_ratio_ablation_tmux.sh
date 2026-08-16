#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_session=${PRL23_A_SOURCE_TMUX_SESSION:-prl23_a_tgvf_teacher50}
relay_session=${PRL23_B_RELAY_TMUX_SESSION:-prl23_b_after_a}
a_supervisor="$repo_root/tools/supervise_prl23_a_tgvf_teacher50_step16_and_eval.sh"
relay="$repo_root/tools/relay_prl23_a_complete_to_prl23_b_tgvf_teacher100.sh"
a_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-23-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher50-8step-ws8
b_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-23-B-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher100-8step-ws8
config_a="$repo_root/configs/policy/runs/prl_23_a_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_teacher50_8step_ws8.toml"
config_b="$repo_root/configs/policy/runs/prl_23_b_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_teacher100_8step_ws8.toml"
plan_a="$repo_root/configs/evaluation/prl23_a_frozen_rp67_tfree_teacher50_step8_step16_paired_seed_coredev2511_plan.json"
plan_b="$repo_root/configs/evaluation/prl23_b_frozen_rp67_tfree_teacher100_step8_step16_paired_seed_coredev2511_plan.json"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi
for path in "$a_supervisor" "$relay"; do
  if [[ ! -x "$path" ]]; then
    echo "required launcher is not executable: $path" >&2
    exit 1
  fi
done
for path in "$config_a" "$config_b" "$plan_a" "$plan_b"; do
  if rg -q '"0{40}"|"0{64}"' "$path"; then
    echo "PRL23 launcher binding placeholders remain in $path" >&2
    exit 1
  fi
done
for session in "$source_session" "$relay_session"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

mkdir -p "$a_root/logs" "$b_root/runtime/relay-after-prl23-a"

# tmux's server may predate this shell. Publish only the required credentials
# into its global environment; values are not interpolated into pane commands
# or written to experiment logs.
tmux set-environment -g OPENROUTER_API_KEY "$OPENROUTER_API_KEY"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  tmux set-environment -g WANDB_API_KEY "$WANDB_API_KEY"
fi

a_console="$a_root/logs/tmux-supervisor-console.log"
relay_console="$b_root/runtime/relay-after-prl23-a/relay.log"
tmux new-session -d -s "$source_session" \
  "cd '$repo_root' && exec '$a_supervisor' >> '$a_console' 2>&1"
sleep 2
if [[ "$(tmux display-message -p -t "$source_session" '#{pane_dead}')" == "1" ]]; then
  echo "PRL23-A supervisor exited during startup; inspect $a_console" >&2
  exit 1
fi

tmux new-session -d -s "$relay_session" \
  "cd '$repo_root' && exec '$relay' >> '$relay_console' 2>&1"
sleep 2
if [[ "$(tmux display-message -p -t "$relay_session" '#{pane_dead}')" == "1" ]]; then
  echo "PRL23-B relay exited during startup; PRL23-A remains active" >&2
  echo "inspect $relay_console and re-arm the relay" >&2
  exit 1
fi

echo "PRL23 Teacher50 active in tmux session: $source_session"
echo "PRL23 Teacher100 relay armed in tmux session: $relay_session"
echo "Teacher50 training+evaluation will hand off only through its canonical receipt."
