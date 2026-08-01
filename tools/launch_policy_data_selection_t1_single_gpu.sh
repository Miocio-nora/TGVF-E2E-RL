#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 REPO CONFIG OUTPUT_ROOT CUDA_VISIBLE_DEVICE" >&2
  exit 2
fi

repo=$1
config=$2
output_root=$3
cuda_visible_device=$4
python_env="$repo/.venv312"
python_header_root="$repo/.deps/python312-dev/root/usr/include"

if [[ ! -f "$python_header_root/python3.12/Python.h" ]] || \
   [[ ! -f "$python_header_root/python3.12/pyconfig.h" ]]; then
  echo "extracted Python 3.12 development headers are missing" >&2
  exit 1
fi

mkdir -p \
  "$output_root/logs" \
  "$output_root/runtime/cache/single-gpu/triton" \
  "$output_root/runtime/cache/single-gpu/torchinductor"

printf '%s\n' "$$" > "$output_root/runtime/single-gpu.pgid"

for rank in 0 1 2 3; do
  env -u VLLM_ATTENTION_BACKEND \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$cuda_visible_device" \
    VLLM_USE_V1=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    TOKENIZERS_PARALLELISM=false \
    PYTHONHASHSEED=42 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    CPATH="$python_header_root:$python_header_root/python3.12" \
    LIBRARY_PATH="$python_env/lib" \
    TRITON_CACHE_DIR="$output_root/runtime/cache/single-gpu/triton" \
    TORCHINDUCTOR_CACHE_DIR="$output_root/runtime/cache/single-gpu/torchinductor" \
    "$python_env/bin/python" -u \
    "$repo/tools/run_policy_data_selection_t1.py" worker \
    --config "$config" \
    --rank "$rank" \
    --cuda-visible-device "$cuda_visible_device" \
    --budget-revision 0
done
