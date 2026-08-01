#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 REPO CONFIG OUTPUT_ROOT LOGICAL_RANK CUDA_VISIBLE_DEVICE SUBSHARD_COUNT SUBSHARD_INDEX" >&2
  exit 2
fi

repo=$1
config=$2
output_root=$3
logical_rank=$4
cuda_visible_device=$5
subshard_count=$6
subshard_index=$7
python_env="$repo/.venv312"
python_header_root="$repo/.deps/python312-dev/root/usr/include"

if [[ ! -f "$python_header_root/python3.12/Python.h" ]] || \
   [[ ! -f "$python_header_root/python3.12/pyconfig.h" ]]; then
  echo "extracted Python 3.12 development headers are missing" >&2
  exit 1
fi

if [[ ! "$logical_rank" =~ ^[0-3]$ ]]; then
  echo "logical rank must be in [0, 3]" >&2
  exit 2
fi
if [[ ! "$cuda_visible_device" =~ ^[0-9]+$ ]]; then
  echo "CUDA visible device must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "$subshard_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "subshard count must be a positive integer" >&2
  exit 2
fi
if [[ ! "$subshard_index" =~ ^(0|[1-9][0-9]*)$ ]] || (( subshard_index >= subshard_count )); then
  echo "subshard index must be inside subshard count" >&2
  exit 2
fi

configured_output_root=$(
  "$python_env/bin/python" -c \
    'import json, pathlib, sys; print(pathlib.Path(json.load(open(sys.argv[1], encoding="utf-8"))["output_root"]).resolve())' \
    "$config"
)
resolved_output_root=$(realpath -m "$output_root")
if [[ "$resolved_output_root" != "$configured_output_root" ]]; then
  echo "OUTPUT_ROOT differs from the configured output_root" >&2
  exit 2
fi
output_root=$resolved_output_root

runtime_root="$output_root/runtime"
worker_tag="rank-${logical_rank}-subshard-${subshard_index}-of-${subshard_count}"
cache_root="$runtime_root/cache/$worker_tag"
pgid_path="$runtime_root/$worker_tag.pgid"
log_path="$output_root/logs/revision0-${worker_tag}-local.log"
mkdir -p "$output_root/logs" "$cache_root/triton" "$cache_root/torchinductor"

launch_lock="$runtime_root/rank-${logical_rank}-subshard-launch.lock"
exec 9>"$launch_lock"
flock -x 9

broad_pgid_path="$runtime_root/rank-$logical_rank.pgid"
if [[ -f "$broad_pgid_path" ]]; then
  broad_pgid=$(<"$broad_pgid_path")
  if [[ ! "$broad_pgid" =~ ^[0-9]+$ ]]; then
    echo "invalid broad process-group record: $broad_pgid_path" >&2
    exit 1
  fi
  if kill -0 -- "-$broad_pgid" 2>/dev/null; then
    echo "logical rank $logical_rank still has live broad process group $broad_pgid" >&2
    exit 1
  fi
fi

shopt -s nullglob
for other_pgid_path in "$runtime_root"/rank-"$logical_rank"-subshard-*-of-*.pgid; do
  other_name=${other_pgid_path##*/}
  if [[ ! "$other_name" =~ ^rank-${logical_rank}-subshard-([0-9]+)-of-([0-9]+)\.pgid$ ]]; then
    echo "unrecognized subshard process record: $other_pgid_path" >&2
    exit 1
  fi
  other_index=${BASH_REMATCH[1]}
  other_count=${BASH_REMATCH[2]}
  other_pgid=$(<"$other_pgid_path")
  if [[ ! "$other_pgid" =~ ^[0-9]+$ ]]; then
    echo "invalid subshard process-group record: $other_pgid_path" >&2
    exit 1
  fi
  if kill -0 -- "-$other_pgid" 2>/dev/null; then
    if (( other_count != subshard_count )); then
      echo "logical rank $logical_rank already has live topology count $other_count" >&2
      exit 1
    fi
    if (( other_index == subshard_index )); then
      echo "$worker_tag already has live process group $other_pgid" >&2
      exit 1
    fi
  fi
done
shopt -u nullglob

if [[ -f "$pgid_path" ]]; then
  recorded_pgid=$(<"$pgid_path")
  if [[ ! "$recorded_pgid" =~ ^[0-9]+$ ]]; then
    echo "invalid process-group record: $pgid_path" >&2
    exit 1
  fi
  if kill -0 -- "-$recorded_pgid" 2>/dev/null; then
    echo "$worker_tag already has live process group $recorded_pgid" >&2
    exit 1
  fi
fi

setsid env -u VLLM_ATTENTION_BACKEND \
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
  TRITON_CACHE_DIR="$cache_root/triton" \
  TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor" \
  "$python_env/bin/python" -u \
  "$repo/tools/run_policy_data_selection_t1.py" worker \
  --config "$config" \
  --rank "$logical_rank" \
  --cuda-visible-device "$cuda_visible_device" \
  --budget-revision 0 \
  --chunk-subshard-count "$subshard_count" \
  --chunk-subshard-index "$subshard_index" >>"$log_path" 2>&1 9>&- &

worker_pgid=$!
printf '%s\n' "$worker_pgid" >"$pgid_path"
sleep 1
if ! kill -0 -- "-$worker_pgid" 2>/dev/null; then
  echo "$worker_tag failed during launch; inspect $log_path" >&2
  exit 1
fi
printf 'rank=%s physical_gpu=%s subshard=%s/%s pgid=%s log=%s\n' \
  "$logical_rank" "$cuda_visible_device" "$subshard_index" "$subshard_count" \
  "$worker_pgid" "$log_path"
