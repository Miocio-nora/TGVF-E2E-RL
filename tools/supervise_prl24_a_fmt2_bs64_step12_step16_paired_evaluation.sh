#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
plan="$repo_root/configs/evaluation/prl24_a_fmt2_frozen_rp67_tfree_teacher25_bs64_step12_step16_paired_seed_coredev2511_plan.json"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-24-A-FMT2-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-8step-ws8
evaluation_id=PRL24-A-FMT2-FROZEN-RP67-TFREE-TEACHER25-BS64-COREDEV2511-STEP12-STEP16-PAIRED-SEED-V1
evaluation_root="$training_root/evaluation/$evaluation_id"
control_root="$evaluation_root/runtime/supervisor"
max_restarts=${PRL24_A_FMT2_EVAL_MAX_RESTARTS:-8}
cooldown_seconds=${PRL24_A_FMT2_EVAL_RESTART_COOLDOWN_SECONDS:-60}

mkdir -p "$control_root" "$evaluation_root/logs"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL24-A FMT2 Step12/16 paired evaluator is active" >&2
  exit 1
fi
if [[ -s "$evaluation_root/evaluation-complete" ]]; then
  exit 0
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
fatal_pattern='SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 mismatch|schema differs|adapter update mode differs|frozen RP67|plan .* differs|policy config .* differs|task manifest .* differs|paired seed namespace differs|paired RNG .* differs|CUDA out of memory|OutOfMemoryError'
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
      echo "paired evaluator exited without its completion receipt" >&2
      exit 1
    fi
    touch "$control_root/evaluation-complete"
    exit 0
  fi
  if rg -q "$fatal_pattern" "$attempt_log"; then
    echo "PRL24-A FMT2 evaluation hit a deterministic contract failure" >&2
    exit "$code"
  fi
  if (( attempt >= max_restarts )); then
    echo "PRL24-A FMT2 evaluation retry budget exhausted after $attempt attempts" >&2
    exit "$code"
  fi
  echo "recoverable PRL24-A FMT2 evaluation failure; retrying in ${cooldown_seconds}s" >&2
  sleep "$cooldown_seconds"
done

