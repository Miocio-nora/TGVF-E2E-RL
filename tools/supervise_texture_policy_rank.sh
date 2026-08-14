#!/usr/bin/env bash
set -Eeuo pipefail

# Durable, restartable supervision for one four-way-sharded texture policy
# worker. Setup/materialization is deliberately outside this script; this owns
# only one rank and never signals a GPU process that it did not start.

if (( $# != 5 )); then
  echo "usage: $0 LABEL CONFIG RANK GPU CONTROL_ROOT" >&2
  exit 2
fi

label=$1
config=$2
rank=$3
gpu=$4
control_root=$5

default_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=${TEXTURE_POLICY_REPO_ROOT:-$default_repo_root}
repo_root=$("/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python" \
  -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' \
  "$repo_root")
dependency_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$dependency_root/.venv312/bin/python"
cuda_home=/home/dredvpn009/Flash_Storage/anaconda3/envs/revisit-vlm
nvidia_smi=/usr/bin/nvidia-smi
world_size=4
max_restarts=${TEXTURE_POLICY_MAX_RESTARTS:-4}
restart_cooldown=${TEXTURE_POLICY_RESTART_COOLDOWN:-30}

if [[ ! "$label" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "label must be filename-safe" >&2
  exit 2
fi
for item in "$rank" "$gpu" "$max_restarts" "$restart_cooldown"; do
  if [[ ! "$item" =~ ^[0-9]+$ ]]; then
    echo "rank, GPU, restart count, and cooldown must be non-negative integers" >&2
    exit 2
  fi
done
if (( rank >= world_size )); then
  echo "rank must be in [0,3]" >&2
  exit 2
fi
config=$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$config")
control_root=$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$control_root")

"$python_bin" - "$config" "$rank" "$gpu" <<'PY'
import json
from pathlib import Path
import sys

config, rank, gpu = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
payload = json.loads(config.read_text(encoding="utf-8"))
gpu_ids = payload.get("gpu_ids")
if gpu_ids is None or len(gpu_ids) != 4 or gpu_ids[rank] != gpu:
    raise SystemExit(
        f"rank/GPU differs from immutable config: rank={rank} gpu={gpu} gpu_ids={gpu_ids}"
    )
if payload.get("expected_task_count") != 42870:
    raise SystemExit("texture policy config does not bind the full 42,870-task suite")
if payload.get("image_max_pixels") != 262144:
    raise SystemExit("texture policy config does not bind max_pixels=262144")
if payload.get("evaluation_protocol") != "training_run":
    raise SystemExit("texture TGVF policy config must use training_run")
PY

rank_root="$control_root/workers/$label/rank-$rank"
cache_root="$control_root/cache/$label/rank-$rank"
mkdir -p "$rank_root" "$cache_root/triton" "$cache_root/torchinductor" \
  "$cache_root/torch-extensions" "$cache_root/flashinfer"
exec 9>"$rank_root/supervisor.lock"
if ! /usr/bin/flock -n 9; then
  echo "another supervisor owns $label rank $rank" >&2
  exit 1
fi

check_gpu_idle() {
  "$python_bin" - "$nvidia_smi" "$gpu" <<'PY'
import csv
import io
import json
import subprocess
import sys

nvidia_smi, target_text = sys.argv[1:]
target = int(target_text)
inventory = subprocess.run(
    [
        nvidia_smi,
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
observed = None
for row in csv.reader(io.StringIO(inventory.stdout), skipinitialspace=True):
    if row and int(row[0]) == target:
        if len(row) != 5:
            raise SystemExit(f"unexpected nvidia-smi GPU row: {row!r}")
        observed = {
            "index": target,
            "uuid": row[1],
            "name": row[2],
            "memory_used_mib": int(row[3]),
            "utilization_percent": int(row[4]),
        }
        break
if observed is None:
    raise SystemExit(f"target GPU {target} is absent from nvidia-smi inventory")

problems = []
if observed["name"] != "NVIDIA B200":
    problems.append(
        f"GPU {target} model is {observed['name']!r}, expected NVIDIA B200"
    )
if observed["memory_used_mib"] > 16:
    problems.append(
        f"GPU {target} uses {observed['memory_used_mib']} MiB (>16 MiB idle allowance)"
    )
if observed["utilization_percent"] != 0:
    problems.append(
        f"GPU {target} utilization is {observed['utilization_percent']}%"
    )

processes = subprocess.run(
    [
        nvidia_smi,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
active = [
    row
    for row in csv.reader(io.StringIO(processes.stdout), skipinitialspace=True)
    if row and row[0] == observed["uuid"]
]
if active:
    problems.append(f"GPU {target} has compute processes: {active!r}")
if problems:
    raise SystemExit(
        "GPU idle gate failed; no process was signalled:\n" + "\n".join(problems)
    )
print(json.dumps({"idle": True, "gpu": observed}, indent=2, sort_keys=True))
PY
}

python_include_root="$dependency_root/.deps/python312-dev/root/usr/include"
python_include="$python_include_root/python3.12"
site_packages="$dependency_root/.venv312/lib/python3.12/site-packages"
cublas_include="$site_packages/nvidia/cublas/include"
nvrtc_include="$site_packages/nvidia/cuda_nvrtc/include"
cuda_runtime_include="$site_packages/nvidia/cuda_runtime/include"
cuda_runtime_library="$site_packages/nvidia/cuda_runtime/lib"
runtime_link_root="$dependency_root/.eval-runtime-python312-dev/lib"

unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH CUDA_PATH LD_LIBRARY_PATH \
  VLLM_ATTENTION_BACKEND VLLM_PLUGINS
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME="$cuda_home"
export PATH="$cuda_home/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CPATH="$python_include_root:$python_include:$cublas_include:$nvrtc_include:$cuda_runtime_include"
export LIBRARY_PATH="$runtime_link_root:$dependency_root/.venv312/lib:$cuda_runtime_library:$site_packages/nvidia/cublas/lib:$site_packages/nvidia/cuda_nvrtc/lib"
export PYTHONPATH="$repo_root/src:$dependency_root/.deps/verl"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

socket_tmp=$(mktemp -d "/tmp/tp-${label:0:1}${rank}.XXXXXX")
if (( ${#socket_tmp} + 37 > 107 )); then
  echo "short TMPDIR still exceeds the ZeroMQ Unix-socket path budget" >&2
  exit 1
fi

next_attempt=1
while compgen -G "$rank_root/attempt-$(printf '%03d' "$next_attempt").*" >/dev/null; do
  next_attempt=$((next_attempt + 1))
done

child_pid=''
terminate_child() {
  [[ -n "$child_pid" ]] || return 0
  kill -0 "$child_pid" 2>/dev/null || return 0
  local pgid
  pgid=$(ps -o pgid= -p "$child_pid" 2>/dev/null | tr -d ' ')
  if [[ -n "$pgid" ]] && [[ "$pgid" == "$child_pid" ]]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
  local count
  for count in $(seq 1 20); do
    kill -0 "$child_pid" 2>/dev/null || return 0
    sleep 1
  done
  if [[ -n "$pgid" ]] && [[ "$pgid" == "$child_pid" ]]; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  else
    kill -KILL "$child_pid" 2>/dev/null || true
  fi
}
trap terminate_child EXIT
trap 'trap - EXIT; terminate_child; exit 130' INT
trap 'trap - EXIT; terminate_child; exit 143' TERM HUP

retries=0
while true; do
  attempt=$next_attempt
  next_attempt=$((next_attempt + 1))
  prefix="$rank_root/attempt-$(printf '%03d' "$attempt")"
  if ! check_gpu_idle >"$prefix.gpu-idle.json" 2>"$prefix.gpu-idle.stderr"; then
    echo "$label rank=$rank GPU=$gpu failed the idle gate; no process was signalled" >&2
    exit 125
  fi
  printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=$gpu rank=$rank world_size=$world_size config=$config" \
    >"$prefix.command"

  /usr/bin/setsid --wait env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$gpu" \
    TRITON_CACHE_DIR="$cache_root/triton" \
    TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor" \
    TORCH_EXTENSIONS_DIR="$cache_root/torch-extensions" \
    FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer" \
    TMPDIR="$socket_tmp" \
    "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
      --config "$config" --mode worker --rank "$rank" --world-size "$world_size" \
      >"$prefix.log" 2>&1 &
  child_pid=$!
  printf '%s\n' "$child_pid" >"$prefix.pid"
  printf '%s started label=%s rank=%s gpu=%s pid=%s attempt=%s\n' \
    "$(date --iso-8601=seconds)" "$label" "$rank" "$gpu" "$child_pid" "$attempt" \
    | tee -a "$control_root/supervisor.log"

  if wait "$child_pid"; then
    exit_code=0
  else
    exit_code=$?
  fi
  printf '{"label":"%s","rank":%d,"gpu":%d,"pid":%d,"attempt":%d,"exit_code":%d}\n' \
    "$label" "$rank" "$gpu" "$child_pid" "$attempt" "$exit_code" \
    >"$prefix.exit.json"
  child_pid=''
  if (( exit_code == 0 )); then
    printf '%s completed label=%s rank=%s gpu=%s attempt=%s\n' \
      "$(date --iso-8601=seconds)" "$label" "$rank" "$gpu" "$attempt" \
      | tee -a "$control_root/supervisor.log"
    trap - EXIT INT TERM HUP
    exit 0
  fi
  if (( retries >= max_restarts )); then
    echo "$label rank=$rank exhausted $max_restarts restart(s); exit=$exit_code" >&2
    trap - EXIT INT TERM HUP
    exit "$exit_code"
  fi
  retries=$((retries + 1))
  printf '%s retry label=%s rank=%s exit=%s retry=%s/%s\n' \
    "$(date --iso-8601=seconds)" "$label" "$rank" "$exit_code" "$retries" "$max_restarts" \
    | tee -a "$control_root/supervisor.log"
  sleep "$restart_cooldown"
done
