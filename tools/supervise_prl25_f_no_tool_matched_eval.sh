#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8
eval_root="$training_root/evaluation/PRL25-F-NO-TOOL-RL-COREDEV2511-S0-S8-S16-S32-DUAL-V1"
control_root="$eval_root/runtime/supervisor"
log_root="$eval_root/logs"
mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
flock -n 9 || { echo "PRL25-F matched evaluator supervisor already active" >&2; exit 1; }

export PYTHONPATH="$repo_root/src:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/verl${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_REPOSITORY_ROOT="$repo_root"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

prepare_arm() {
  local step=$1
  shift
  "$python_bin" "$repo_root/tools/prepare_prl25_f_no_tool_matched_arm.py" \
    --step "$step" --gpu-ids "$@" \
    > "$log_root/prepare-artifacts-s${step}.log" 2>&1
  local config="$eval_root/matched/step${step}/config.json"
  "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
    --config "$config" --mode prepare \
    > "$log_root/prepare-eval-s${step}.log" 2>&1
  "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
    --config "$config" --mode validate --world-size 4 \
    > "$log_root/validate-s${step}.log" 2>&1
  touch "$control_root/s${step}-prepared"
}

smoke_arm() {
  local step=$1
  local gpu=$2
  local config="$eval_root/matched/step${step}/config.json"
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
    "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
    --config "$config" --mode worker --rank 0 --world-size 4 --max-tasks 1 \
    > "$log_root/smoke-s${step}.log" 2>&1
  touch "$control_root/s${step}-smoke-accepted"
}

run_arm() {
  local step=$1
  local gpu_base=$2
  local config="$eval_root/matched/step${step}/config.json"
  local pids=()
  for rank in 0 1 2 3; do
    local gpu=$((gpu_base + rank))
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
      --config "$config" --mode worker --rank "$rank" --world-size 4 \
      > "$log_root/s${step}-rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" == 0 ]]
  "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
    --config "$config" --mode status --world-size 4 \
    > "$log_root/status-s${step}.json"
  touch "$control_root/s${step}-inference-complete"
}

prepare_arm 0 0 1 2 3
smoke_arm 0 0
run_arm 0 0 &
pid0=$!

prepare_arm 8 4 5 6 7
smoke_arm 8 4
run_arm 8 4 &
pid8=$!

prepare_arm 16 0 1 2 3
wait "$pid0"
smoke_arm 16 0
run_arm 16 0 &
pid16=$!

prepare_arm 32 4 5 6 7
wait "$pid8"
smoke_arm 32 4
run_arm 32 4 &
pid32=$!

wait "$pid16"
wait "$pid32"
touch "$control_root/matched-inference-complete"
