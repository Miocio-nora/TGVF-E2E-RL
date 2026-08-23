#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
plan="$repo_root/configs/evaluation/prl25_d_atomic_crop_tgvf_tfree_teacher25_s8_s16_s32_s48_s64_s80_paired_seed_coredev2511_plan.json"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-25-D-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-80step-ws8
evaluation_id=PRL25-D-ATOMIC-CROP-TGVF-RP67-TFREE-TEACHER25-COREDEV2511-S8-S16-S32-S48-S64-S80-PAIRED-SEED-V1
evaluation_root="$training_root/evaluation/$evaluation_id"
control_root="$evaluation_root/runtime/supervisor"
max_restarts=${PRL25_D_PAIRED_EVAL_MAX_RESTARTS:-10}
cooldown_seconds=${PRL25_D_PAIRED_EVAL_RESTART_COOLDOWN_SECONDS:-30}

mkdir -p "$control_root" "$evaluation_root/logs"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL25-D six-point evaluator is active" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
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
    touch "$control_root/evaluation-complete"
    exit 0
  fi
  if rg -q "$deterministic_error_pattern" "$attempt_log"; then
    echo "deterministic evaluation contract failure; refusing retry" >&2
    exit "$code"
  fi
  if (( attempt > max_restarts )); then
    echo "evaluation retry budget exhausted after $attempt attempts" >&2
    exit "$code"
  fi
  echo "recoverable evaluation failure; retrying in ${cooldown_seconds}s" >&2
  sleep "$cooldown_seconds"
done
