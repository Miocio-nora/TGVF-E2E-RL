#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint only.  All checkpoint binding, inference, scoring,
# judge startup, resume, and completion semantics live in the unified paired
# evaluator; PRL21 carries no experiment-specific evaluator implementation.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
plan="$repo_root/configs/evaluation/prl21_r0_crop_tfree_step8_step16_full_model_coredev2511_plan.json"
evaluator="$repo_root/tools/run_paired_policy_evaluation.py"
lock=/tmp/tgvf-prl21-crop-tfree16-evaluation.lock

if (($# != 0)); then
  echo "usage: $0" >&2
  exit 2
fi

exec 9>"$lock"
if ! flock -n 9; then
  echo "PRL21 evaluation supervisor is already running" >&2
  exit 1
fi
exec "$python_bin" "$evaluator" \
  --plan "$plan" \
  --mode run \
  --gpu-ids 0 1 2 3 4 5 6 7 \
  --wait-for-final-arm \
  --wait-for-gpus
