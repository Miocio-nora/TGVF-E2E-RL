#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
artifact_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts
c_training_root="$artifact_root/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8"
d_training_root="$artifact_root/policy/PRL-25-D-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-80step-ws8"
c_evaluation_id=PRL25-C-FROZEN-RP67-TFREE-TEACHER25-COREDEV2511-S72-PAIRED-SEED-V1
d_evaluation_id=PRL25-D-ATOMIC-CROP-TGVF-RP67-TFREE-TEACHER25-COREDEV2511-S72-PAIRED-SEED-V1
c_plan="$repo_root/configs/evaluation/prl25_c_frozen_rp67_tfree_teacher25_s72_paired_seed_coredev2511_plan.json"
d_plan="$repo_root/configs/evaluation/prl25_d_atomic_crop_tgvf_tfree_teacher25_s72_paired_seed_coredev2511_plan.json"
c_evaluation_root="$c_training_root/evaluation/$c_evaluation_id"
d_evaluation_root="$d_training_root/evaluation/$d_evaluation_id"
control_root="$artifact_root/evaluation/PRL25-CD-S72-PAIRED-SEED-V1/runtime/supervisor"
max_restarts=${PRL25_CD_S72_EVAL_MAX_RESTARTS:-10}
cooldown_seconds=${PRL25_CD_S72_EVAL_RESTART_COOLDOWN_SECONDS:-30}

mkdir -p "$control_root" "$c_evaluation_root/logs" "$d_evaluation_root/logs"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL25 C/D S72 evaluator is active" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
deterministic_error_pattern='ValueError:|SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 differs|schema differs|immutable .*collision|adapter update mode differs|frozen RP67|plan .* differs|policy config .* differs|task manifest .* differs|paired seed namespace differs|paired RNG .* differs'

run_with_retries() {
  local label=$1
  local plan=$2
  local evaluation_root=$3
  shift 3
  local -a gpu_ids=("$@")
  local attempt=0
  local code=1
  local attempt_log

  while true; do
    attempt=$((attempt + 1))
    attempt_log="$evaluation_root/logs/supervisor-attempt-$(printf '%02d' "$attempt").log"
    set +e
    "$python_bin" "$repo_root/tools/run_prl15_paired_evaluation.py" \
      --plan "$plan" \
      --mode run \
      --output-root "$evaluation_root" \
      --gpu-ids "${gpu_ids[@]}" \
      --wait-for-gpus \
      --wait-timeout-seconds 86400 \
      --poll-seconds 30 \
      2>&1 | tee -a "$attempt_log"
    code=${PIPESTATUS[0]}
    set -e

    if [[ "$code" == 0 ]]; then
      touch "$control_root/${label}-evaluation-complete"
      return 0
    fi
    if rg -q "$deterministic_error_pattern" "$attempt_log"; then
      echo "$label: deterministic evaluation contract failure; refusing retry" >&2
      return "$code"
    fi
    if (( attempt > max_restarts )); then
      echo "$label: evaluation retry budget exhausted after $attempt attempts" >&2
      return "$code"
    fi
    echo "$label: recoverable evaluation failure; retrying in ${cooldown_seconds}s" >&2
    sleep "$cooldown_seconds"
  done
}

run_with_retries c "$c_plan" "$c_evaluation_root" 0 1 2 3 &
c_pid=$!
run_with_retries d "$d_plan" "$d_evaluation_root" 4 5 6 7 &
d_pid=$!
trap 'kill "$c_pid" "$d_pid" 2>/dev/null || true' INT TERM

set +e
wait "$c_pid"
c_code=$?
wait "$d_pid"
d_code=$?
set -e

if [[ "$c_code" != 0 || "$d_code" != 0 ]]; then
  echo "PRL25 C/D S72 evaluation failed: C=$c_code D=$d_code" >&2
  exit 1
fi

touch "$control_root/evaluation-complete"
