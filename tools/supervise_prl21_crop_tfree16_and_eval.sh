#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-21-R0-qwen3-instruct-full-crop-bs16-n16-tfree-16step-ws8
control_root="$training_root/runtime/supervisor"
log_root="$training_root/logs/supervisor"
post_train_eval="$repo_root/tools/supervise_prl21_crop_tfree16_eval.sh"

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/formal.lock"
if ! flock -n 9; then
  echo "another PRL21 Crop T-free formal supervisor is active" >&2
  exit 1
fi

if [[ ! -s "$training_root/smoke-integration/completion.json" ]]; then
  echo "PRL21 integration smoke has not completed" >&2
  exit 1
fi
if [[ -z ${OPENROUTER_API_KEY:-} ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_ENTITY=mio_nora
export WANDB_PROJECT=tgvf-policy-rl
export WANDB_RUN_ID=prl21r0croptfree
export WANDB_RESUME=allow
export WANDB_MODE=online

attempt=$(( ${PRL21_FORMAL_ATTEMPT:-0} + 1 ))
attempt_log="$log_root/attempt-$(printf '%02d' "$attempt").log"
"$python_bin" "$repo_root/tools/launch_prl21_crop_tfree16.py" \
  --mode formal --launch 2>&1 | tee -a "$attempt_log"

if [[ ! -s "$training_root/completion.json" ]]; then
  echo "PRL21 training exited without completion.json" >&2
  exit 1
fi
if [[ ! -x "$post_train_eval" ]]; then
  echo "PRL21 post-training evaluator is absent or not executable" >&2
  exit 1
fi
exec "$post_train_eval"
