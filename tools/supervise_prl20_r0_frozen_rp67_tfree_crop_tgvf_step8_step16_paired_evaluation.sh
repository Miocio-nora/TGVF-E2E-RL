#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
plan="$repo_root/configs/evaluation/prl20_r0_frozen_rp67_tfree_crop_tgvf_step8_step16_paired_seed_coredev2511_plan.json"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-20-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-8step-ws8
evaluation_id=PRL20-R0-FROZEN-RP67-TFREE-CROP-TGVF-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1
evaluation_root="$training_root/evaluation/$evaluation_id"
control_root="$evaluation_root/runtime/supervisor"
max_restarts=${PRL20_R0_PAIRED_EVAL_MAX_RESTARTS:-4}
cooldown_seconds=${PRL20_R0_PAIRED_EVAL_RESTART_COOLDOWN_SECONDS:-30}

mkdir -p "$control_root" "$evaluation_root/logs"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL20 paired evaluator is active" >&2
  exit 1
fi

deterministic_error_pattern='ValueError:|SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 differs|schema differs|immutable .*collision|adapter update mode differs|frozen RP67|plan .* differs|policy config .* differs|task manifest .* differs|paired seed namespace differs|paired RNG .* differs'
attempt=0
while true; do
  attempt=$((attempt + 1))
  attempt_log="$evaluation_root/logs/supervisor-attempt-$(printf '%02d' "$attempt").log"
  set +e
  "$python_bin" "$repo_root/tools/run_prl15_paired_evaluation.py" \
    --plan "$plan" \
    --mode run \
    --output-root "$evaluation_root" \
    --gpu-ids 0 1 2 3 4 5 6 7 \
    --wait-for-final-arm \
    --wait-for-gpus \
    --wait-timeout-seconds 86400 \
    --poll-seconds 30 \
    "$@" 2>&1 | tee -a "$attempt_log"
  code=${PIPESTATUS[0]}
  set -e
  if [[ "$code" == 0 ]]; then
    exit 0
  fi
  if rg -q "$deterministic_error_pattern" "$attempt_log"; then
    echo "deterministic PRL20 evaluation contract failure; refusing retry" >&2
    exit "$code"
  fi
  if (( attempt > max_restarts )); then
    echo "PRL20 evaluation retry budget exhausted after $attempt attempts" >&2
    exit "$code"
  fi
  echo "recoverable PRL20 evaluation failure; retrying in ${cooldown_seconds}s" >&2
  sleep "$cooldown_seconds"
done
