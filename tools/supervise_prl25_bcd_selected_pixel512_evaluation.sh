#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
s32_root="$main_root/artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-RAW-DIRECT-512-S32-V1"
control_root="$main_root/artifacts/evaluation/PRL25-BCD-SELECTED-PIXEL512-COREDEV2511-V1/runtime/supervisor"
log_root="$main_root/artifacts/evaluation/PRL25-BCD-SELECTED-PIXEL512-COREDEV2511-V1/logs"

plans=(
  "$repo_root/configs/evaluation/prl25_b_crop_exact_step80_pixel512_coredev2511_plan.json"
  "$repo_root/configs/evaluation/prl25_c_frozen_rp67_tfree_teacher25_s64_matched_pixel512_coredev2511_plan.json"
  "$repo_root/configs/evaluation/prl25_d_atomic_crop_tgvf_s16_matched_pixel512_coredev2511_plan.json"
)
labels=(crop-s80 tgvf-s64 atomic-s16)
output_roots=(
  "$main_root/artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-STEP80-PIXEL512-TEMP1-SEED42-V1"
  "$main_root/artifacts/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-C-FROZEN-RP67-TFREE-TEACHER25-S64-MATCHED-PIXEL512-COREDEV2511-SEED42-V1"
  "$main_root/artifacts/policy/PRL-25-D-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-80step-ws8/evaluation/PRL25-D-ATOMIC-CROP-TGVF-RP67-TFREE-TEACHER25-S16-MATCHED-PIXEL512-COREDEV2511-SEED42-V1"
)

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
flock -n 9 || { echo "selected pixel512 evaluator is already active" >&2; exit 1; }

phase=waiting_for_s32_raw_direct_512
active_label=""
active_pids=()
launched_pid=""
attached_tgvf_infer_pid=${PRL25_TGVF_INFER_PID:-}

timestamp() {
  date '+%F %T %Z'
}

cleanup() {
  local status=$?
  set +e
  local pid
  for pid in "${active_pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "${active_pids[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  if (( status == 0 )); then
    rm -f "$control_root/failed"
  else
    printf 'status=failed\nphase=%s\narm=%s\ntime=%s\nexit_status=%s\n' \
      "$phase" "$active_label" "$(timestamp)" "$status" >"$control_root/failed"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'phase=signal; exit 130' INT TERM

launch_eval() {
  local index=$1
  local mode=$2
  shift 2
  local label=${labels[$index]}
  local plan=${plans[$index]}
  local output_root=${output_roots[$index]}
  local command=(
    "$python_bin" "$repo_root/tools/run_prl15_paired_evaluation.py"
    --plan "$plan"
    --mode "$mode"
    --output-root "$output_root"
    --gpu-ids "$@"
  )
  if [[ "$mode" != score ]]; then
    command+=(
      --wait-for-gpus
      --wait-timeout-seconds 86400
      --poll-seconds 30
    )
  fi
  active_label=$label
  phase="running_${label}_${mode}"
  mkdir -p "$output_root/logs"
  # Do not leak the supervisor flock into long-lived evaluation children.  A
  # leaked descriptor prevents a replacement supervisor from taking over.
  setsid "${command[@]}" >"$log_root/${label}-${mode}.log" 2>&1 9>&- &
  launched_pid=$!
  active_pids+=("$launched_pid")
}

wait_eval() {
  local pid=$1
  local label=$2
  local mode=$3
  active_label=$label
  phase="waiting_${label}_${mode}"
  wait "$pid"
}

wait_attached_eval() {
  local pid=$1
  local label=$2
  local mode=$3
  active_label=$label
  phase="waiting_attached_${label}_${mode}"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
  done
}

validate_inference_rows() {
  local output_root=$1
  local step=$2
  local expected_rows=$3
  local inference_root="$output_root/$step/inference"
  local actual_rows
  actual_rows=$(find "$inference_root" -maxdepth 1 -type f -name 'rank-*.jsonl' -print0 \
    | xargs -0 -r cat | wc -l)
  if [[ "$actual_rows" -ne "$expected_rows" ]]; then
    echo "incomplete inference rows: root=$inference_root expected=$expected_rows actual=$actual_rows" >&2
    return 1
  fi
}

while [[ ! -f "$s32_root/runtime/scoring-supervisor/raw-direct-512-s32-scoring-complete" ]]; do
  sleep 15
done

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
rm -f "$control_root/failed" "$control_root/pixel512-selected-evaluation-complete"

# Fill all eight GPUs during the first inference phase.  As soon as Crop has
# released GPU4-7, start Atomic there instead of waiting at a cross-arm barrier.
if [[ ! -f "$control_root/crop-s80-inference-complete" ]]; then
  launch_eval 0 infer 4 5 6 7
  crop_infer_pid=$launched_pid
fi
if [[ ! -f "$control_root/tgvf-s64-inference-complete" && -z "$attached_tgvf_infer_pid" ]]; then
  launch_eval 1 infer 0 1 2 3
  tgvf_infer_pid=$launched_pid
fi
if [[ -n "${crop_infer_pid:-}" ]]; then
  wait_eval "$crop_infer_pid" crop-s80 infer
  validate_inference_rows "${output_roots[0]}" step80 2240
  touch "$control_root/crop-s80-inference-complete"
fi

launch_eval 2 infer 4 5 6 7
atomic_infer_pid=$launched_pid

if [[ ! -f "$control_root/tgvf-s64-inference-complete" ]]; then
  if [[ -n "$attached_tgvf_infer_pid" ]]; then
    wait_attached_eval "$attached_tgvf_infer_pid" tgvf-s64 infer
  else
    wait_eval "$tgvf_infer_pid" tgvf-s64 infer
  fi
  validate_inference_rows "${output_roots[1]}" step64 2240
  touch "$control_root/tgvf-s64-inference-complete"
fi

# GPU0/1 are now free for the TP=2 judge while Atomic continues on GPU4-7.
launch_eval 0 score 0 1 2 3
crop_score_pid=$launched_pid
wait_eval "$crop_score_pid" crop-s80 score
touch "$control_root/crop-s80-complete"
launch_eval 1 score 0 1 2 3
tgvf_score_pid=$launched_pid
wait_eval "$tgvf_score_pid" tgvf-s64 score
touch "$control_root/tgvf-s64-complete"
wait_eval "$atomic_infer_pid" atomic-s16 infer
validate_inference_rows "${output_roots[2]}" step16 2240
touch "$control_root/atomic-s16-inference-complete"
launch_eval 2 score 0 1 2 3
atomic_score_pid=$launched_pid
wait_eval "$atomic_score_pid" atomic-s16 score
touch "$control_root/atomic-s16-complete"

phase=complete
touch "$control_root/pixel512-selected-evaluation-complete"
