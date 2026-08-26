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

timestamp() {
  date '+%F %T %Z'
}

cleanup() {
  local status=$?
  set +e
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

while [[ ! -f "$s32_root/runtime/scoring-supervisor/raw-direct-512-s32-scoring-complete" ]]; do
  sleep 15
done

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
for index in "${!plans[@]}"; do
  active_label=${labels[$index]}
  plan=${plans[$index]}
  output_root=${output_roots[$index]}
  phase="running_${active_label}"
  mkdir -p "$output_root/logs"
  "$python_bin" "$repo_root/tools/run_prl15_paired_evaluation.py" \
    --plan "$plan" \
    --mode run \
    --output-root "$output_root" \
    --gpu-ids 4 5 6 7 \
    --wait-for-gpus \
    --wait-timeout-seconds 86400 \
    --poll-seconds 30 \
    2>&1 | tee -a "$log_root/${active_label}.log"
  touch "$control_root/${active_label}-complete"
done

phase=complete
touch "$control_root/pixel512-selected-evaluation-complete"
