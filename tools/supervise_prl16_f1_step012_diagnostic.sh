#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
plan="$repo_root/configs/evaluation/prl16_f1_frozen_rp66_step0_step1_step2_coredev2511_plan.json"
run_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-16-F1-qwen3-instruct-full-frozen-rp66-bs16-n16-t1-crop16-exact-matched-8step-ws8
evaluation_id=PRL16-F1-FROZEN-RP66-COREDEV2511-STEP0-STEP1-STEP2-DIAGNOSTIC-V1
evaluation_root="$run_root/evaluation/$evaluation_id"
log_root="$evaluation_root/logs"

mkdir -p "$log_root"
exec 9>"$evaluation_root/diagnostic-supervisor.lock"
flock -n 9 || { echo "another diagnostic supervisor is active" >&2; exit 1; }

# The first two arms are deliberately launched immediately by the operator.
# Wait for them, then replay each idempotently so an interrupted worker resumes
# from its durable JSONL records rather than leaving the chain stopped.
while tmux has-session -t prl16_f1_diag_step0 2>/dev/null \
  || tmux has-session -t prl16_f1_diag_step1 2>/dev/null; do
  sleep 15
done

run_arm() {
  local arm=$1
  shift
  local attempt code
  for attempt in 1 2 3; do
    set +e
    "$python_bin" "$repo_root/tools/run_prl16_f1_diagnostic_arm.py" \
      --plan "$plan" --arm "$arm" --gpu-ids "$@" \
      2>&1 | tee -a "$log_root/${arm}-supervisor-attempt-${attempt}.log"
    code=${PIPESTATUS[0]}
    set -e
    if [[ $code == 0 ]]; then
      return 0
    fi
    sleep 30
  done
  return "$code"
}

run_arm step0 0 1 2 3 &
step0_pid=$!
run_arm step1 4 5 6 7 &
step1_pid=$!
wait "$step0_pid"
wait "$step1_pid"

run_arm step2 0 1 2 3

for attempt in 1 2 3; do
  set +e
  "$python_bin" "$repo_root/tools/run_prl15_paired_evaluation.py" \
    --plan "$plan" --mode score --gpu-ids 0 1 2 3 4 5 6 7 \
    2>&1 | tee -a "$log_root/official-score-attempt-${attempt}.log"
  code=${PIPESTATUS[0]}
  set -e
  if [[ $code == 0 ]]; then
    exit 0
  fi
  sleep 30
done
exit "$code"
