#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
evaluation_id=PRL26-A-TRAIN512-S32-NOTOOL-MATCHED-COREDEV2511-S32-PIXEL512-V1
eval_root="$main_root/artifacts/policy/PRL-26-A-train512-s32-parity-notool-qwen3-instruct-bs16-n16-teacher25-ws8/evaluation/$evaluation_id"
arm_root="$eval_root/matched/step32"
config="$arm_root/config.json"
control_root="$eval_root/runtime/supervisor"
log_root="$eval_root/logs"
proof_path="$arm_root/runtime/pixel512-processor-proof.json"

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
flock -n 9 || {
  echo "PRL-26-A Train@512 S32 inference supervisor is already active" >&2
  exit 1
}

export PYTHONPATH="$repo_root/src:$main_root/.deps/verl${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_REPOSITORY_ROOT="$repo_root"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

phase=initializing
worker_pids=()

timestamp() {
  date '+%F %T %Z'
}

stop_process_group() {
  local pid=${1:-}
  [[ -n "$pid" ]] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "-$pid" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local status=$?
  set +e
  local pid
  for pid in "${worker_pids[@]:-}"; do
    stop_process_group "$pid"
  done
  if (( status == 0 )); then
    rm -f "$control_root/failed"
  else
    rm -f "$control_root/matched-inference-complete"
    printf 'status=failed\nphase=%s\ntime=%s\nexit_status=%s\n' \
      "$phase" "$(timestamp)" "$status" >"$control_root/failed"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'phase=signal; exit 130' INT TERM

rm -f "$control_root/failed" "$control_root/matched-inference-complete"
phase=materializing_full_model
env CUDA_VISIBLE_DEVICES= "$python_bin" \
  "$repo_root/tools/prepare_prl26_a_no_tool_train512_s32_eval.py" \
  --gpu-ids 0 1 2 3 >"$log_root/prepare-s32.log" 2>&1

phase=preparing_tasks
env CUDA_VISIBLE_DEVICES= "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
  --config "$config" --mode prepare >"$log_root/prepare-tasks-s32.log" 2>&1

phase=validating_pixel512_processor
env CUDA_VISIBLE_DEVICES= "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
  --config "$config" --mode validate --world-size 4 \
  >"$log_root/validate-s32.json" 2>"$log_root/validate-s32.stderr.log"
env CUDA_VISIBLE_DEVICES= "$python_bin" \
  "$repo_root/tools/validate_prl26_train512_processor_proof.py" \
  --arm no-tool --config "$config" \
  --validation-json "$log_root/validate-s32.json" --output "$proof_path" \
  >"$log_root/pixel512-proof-s32.json"

phase=running_four_rank_inference
for rank in 0 1 2 3; do
  cache_root="$arm_root/runtime/cache/rank-$rank"
  mkdir -p "$cache_root/triton" "$cache_root/torchinductor" "$cache_root/vllm"
  setsid env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$rank" \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    CPATH="$main_root/.deps/python312-dev/root/usr/include:$main_root/.deps/python312-dev/root/usr/include/python3.12" \
    LIBRARY_PATH="$main_root/.venv312/lib" \
    TRITON_CACHE_DIR="$cache_root/triton" \
    TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor" \
    VLLM_CACHE_ROOT="$cache_root/vllm" \
    "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
      --config "$config" --mode worker --rank "$rank" --world-size 4 \
      >"$log_root/s32-rank${rank}.log" 2>&1 9>&- &
  worker_pids+=("$!")
done

while (( ${#worker_pids[@]} > 0 )); do
  completed_pid=
  set +e
  wait -n -p completed_pid "${worker_pids[@]}"
  worker_status=$?
  set -e
  remaining_pids=()
  for pid in "${worker_pids[@]}"; do
    [[ "$pid" == "$completed_pid" ]] || remaining_pids+=("$pid")
  done
  worker_pids=("${remaining_pids[@]}")
  if (( worker_status != 0 )); then
    phase=inference_worker_failed
    exit "$worker_status"
  fi
done
worker_pids=()

phase=validating_complete_coverage
"$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
  --config "$config" --mode status --world-size 4 \
  >"$log_root/status-s32.json"
"$python_bin" - "$log_root/status-s32.json" "$evaluation_id" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    payload.get("evaluation_id") != sys.argv[2]
    or payload.get("completed_single_image") != 2240
    or payload.get("total_single_image") != 2240
    or payload.get("remaining_single_image") != 0
    or payload.get("multi_image_pending_protocol_decision") != 271
):
    raise RuntimeError("PRL-26-A inference coverage differs")
PY

phase=complete
touch "$control_root/matched-inference-complete"
printf '[%s] PRL-26-A Train@512 S32 NoTool inference complete\n' "$(timestamp)"
