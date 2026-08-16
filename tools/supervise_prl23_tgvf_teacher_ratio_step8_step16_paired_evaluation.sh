#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {teacher50|teacher100}" >&2
  exit 2
fi
arm=$1
shift
case "$arm" in
  teacher50)
    arm_id=PRL23-A
    plan="$repo_root/configs/evaluation/prl23_a_frozen_rp67_tfree_teacher50_step8_step16_paired_seed_coredev2511_plan.json"
    training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-23-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher50-8step-ws8
    evaluation_id=PRL23-A-FROZEN-RP67-TFREE-TEACHER50-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1
    ;;
  teacher100)
    arm_id=PRL23-B
    plan="$repo_root/configs/evaluation/prl23_b_frozen_rp67_tfree_teacher100_step8_step16_paired_seed_coredev2511_plan.json"
    training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-23-B-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher100-8step-ws8
    evaluation_id=PRL23-B-FROZEN-RP67-TFREE-TEACHER100-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1
    ;;
  *)
    echo "unsupported PRL23 teacher-ratio arm: $arm" >&2
    exit 2
    ;;
esac

evaluation_root="$training_root/evaluation/$evaluation_id"
control_root="$evaluation_root/runtime/supervisor"
max_restarts=${PRL23_PAIRED_EVAL_MAX_RESTARTS:-6}
cooldown_seconds=${PRL23_PAIRED_EVAL_RESTART_COOLDOWN_SECONDS:-30}

mkdir -p "$control_root" "$evaluation_root/logs"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another $arm_id paired evaluator is active" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
deterministic_error_pattern='SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 differs|schema differs|immutable .*collision|adapter update mode differs|frozen RP67|plan .* differs|policy config .* differs|task manifest .* differs|paired seed namespace differs|paired RNG .* differs'
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
    if [[ ! -s "$evaluation_root/evaluation-complete" ]]; then
      echo "paired evaluator exited successfully without canonical completion receipt" >&2
      exit 1
    fi
    touch "$control_root/evaluation-complete"
    exit 0
  fi
  if rg -q "$deterministic_error_pattern" "$attempt_log"; then
    echo "deterministic $arm_id evaluation contract failure; refusing retry" >&2
    exit "$code"
  fi
  if (( attempt > max_restarts )); then
    echo "$arm_id evaluation retry budget exhausted after $attempt attempts" >&2
    exit "$code"
  fi
  echo "recoverable $arm_id evaluation failure; retrying in ${cooldown_seconds}s" >&2
  sleep "$cooldown_seconds"
done
