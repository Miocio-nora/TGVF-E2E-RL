#!/usr/bin/env bash
set -Eeuo pipefail

# Supervise the full texture benchmark comparison between stock Qwen and the
# PRL14 Crop-only step-16 policy.  The script is intentionally host-specific:
# every compiler, development header, CUDA toolkit, GPU, and Python path below
# is part of the accepted B200 runtime closure.

usage() {
  cat <<'EOF'
Usage:
  tools/supervise_texture_original_crop_step16.sh [--matrix PATH]

The default matrix is the full LAS&T/MMAD original-versus-Crop step-16 matrix.
Crop ranks use physical GPUs 0-3 and original ranks use physical GPUs 4-7.

Environment overrides:
  TEXTURE_TWO_ARM_MATRIX                 alternate matrix path
  TEXTURE_TWO_ARM_CONTROL_ROOT           logs/PIDs/cache root
  TEXTURE_TWO_ARM_SETUP_MODE             strict (default) or resume
  TEXTURE_TWO_ARM_RESUME_VALIDATE_EVIDENCE
                                             prior static-validate JSON for resume
  TEXTURE_TWO_ARM_MAX_RESTARTS           retries per failed rank (default: 4)
  TEXTURE_TWO_ARM_RESTART_COOLDOWN       seconds before retry (default: 30)
  TEXTURE_ORIGINAL_BATCH_SIZE            original batch size (default: 8)
  TEXTURE_ORIGINAL_MAX_TOKENS            original output cap (default: 2048)
  TEXTURE_ORIGINAL_ENGINE_KWARGS_JSON     exact vLLM kwargs JSON

This supervisor never kills processes discovered through nvidia-smi.  It
refuses to launch on a busy target GPU and only signals process groups that it
started itself when the supervisor receives a termination signal.
EOF
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dependency_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$dependency_root/.venv312/bin/python"
cuda_home=/home/dredvpn009/Flash_Storage/anaconda3/envs/revisit-vlm
nvidia_smi=/usr/bin/nvidia-smi
default_matrix="$repo_root/configs/evaluation/texture_last_mmad_original_crop_prl14_step16_512_v1.json"
matrix=${TEXTURE_TWO_ARM_MATRIX:-$default_matrix}
setup_mode=${TEXTURE_TWO_ARM_SETUP_MODE:-strict}
resume_validate_evidence=${TEXTURE_TWO_ARM_RESUME_VALIDATE_EVIDENCE:-}
max_restarts=${TEXTURE_TWO_ARM_MAX_RESTARTS:-4}
restart_cooldown=${TEXTURE_TWO_ARM_RESTART_COOLDOWN:-30}
original_batch_size=${TEXTURE_ORIGINAL_BATCH_SIZE:-8}
original_max_tokens=${TEXTURE_ORIGINAL_MAX_TOKENS:-2048}
original_engine_kwargs=${TEXTURE_ORIGINAL_ENGINE_KWARGS_JSON:-'{"gpu_memory_utilization":0.8,"max_model_len":32768,"max_num_batched_tokens":32768,"max_num_seqs":8}'}
readonly world_size=4
readonly -a crop_gpu_ids=(0 1 2 3)
readonly -a original_gpu_ids=(4 5 6 7)

while (( $# > 0 )); do
  case "$1" in
    --matrix)
      if (( $# < 2 )); then
        echo "--matrix requires a path" >&2
        exit 2
      fi
      matrix=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_nonnegative_integer() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer, observed: $value" >&2
    exit 2
  fi
}

require_positive_integer() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, observed: $value" >&2
    exit 2
  fi
}

require_nonnegative_integer TEXTURE_TWO_ARM_MAX_RESTARTS "$max_restarts"
require_nonnegative_integer TEXTURE_TWO_ARM_RESTART_COOLDOWN "$restart_cooldown"
require_positive_integer TEXTURE_ORIGINAL_BATCH_SIZE "$original_batch_size"
require_positive_integer TEXTURE_ORIGINAL_MAX_TOKENS "$original_max_tokens"
if [[ "$setup_mode" != strict ]] && [[ "$setup_mode" != resume ]]; then
  echo "TEXTURE_TWO_ARM_SETUP_MODE must be strict or resume, observed: $setup_mode" >&2
  exit 2
fi
if [[ "$setup_mode" == strict ]] && [[ -n "$resume_validate_evidence" ]]; then
  echo "TEXTURE_TWO_ARM_RESUME_VALIDATE_EVIDENCE requires setup mode resume" >&2
  exit 2
fi

for executable in "$python_bin" /usr/bin/gcc /usr/bin/g++ "$cuda_home/bin/nvcc" \
  "$nvidia_smi" /usr/bin/flock /usr/bin/setsid; do
  if [[ ! -x "$executable" ]]; then
    echo "required executable is absent: $executable" >&2
    exit 1
  fi
done

matrix=$(
  "$python_bin" - "$matrix" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve(strict=True))
PY
)

matrix_metadata=$(
  "$python_bin" - "$matrix" <<'PY'
import json
from pathlib import Path
import sys

matrix_path = Path(sys.argv[1])
payload = json.loads(matrix_path.read_text(encoding="utf-8"))
arms = payload.get("arms")
if not isinstance(arms, list):
    raise SystemExit("texture matrix arms must be a list")
original = [arm for arm in arms if arm.get("kind") == "original"]
crop = [arm for arm in arms if arm.get("kind") == "crop"]
if len(original) != 1 or len(crop) != 1:
    raise SystemExit("supervisor requires exactly one original arm and one crop arm")
if original[0].get("backend") != "stock_qwen_vllm":
    raise SystemExit("original arm must use stock_qwen_vllm")
if crop[0].get("backend") != "policy_benchmark":
    raise SystemExit("crop arm must use policy_benchmark")
if payload.get("gpu_ids") != [0, 1, 2, 3]:
    raise SystemExit("Crop matrix gpu_ids must be exactly [0,1,2,3]")

selected = {
    "output_root": payload.get("output_root"),
    "task_manifest_path": payload.get("task_manifest_path"),
    "task_count": payload.get("task_count"),
    "matrix_id": payload.get("matrix_id"),
    "original_arm_id": original[0].get("arm_id"),
    "crop_arm_id": crop[0].get("arm_id"),
}
for name, value in selected.items():
    if name == "task_count":
        if type(value) is not int or value <= 0:
            raise SystemExit("matrix task_count must be a positive integer")
    elif not isinstance(value, str) or not value or any(c in value for c in "\x00\r\n"):
        raise SystemExit(f"matrix {name} must be a non-empty single-line string")
print(json.dumps(selected, sort_keys=True, separators=(",", ":")))
PY
)

json_field() {
  local source=$1
  local field=$2
  "$python_bin" - "$source" "$field" <<'PY'
import json
from pathlib import Path
import sys

source, field = sys.argv[1:]
if source.startswith("{"):
    payload = json.loads(source)
else:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
value = payload
for component in field.split("."):
    if not isinstance(value, dict) or component not in value:
        raise SystemExit(f"JSON field is absent: {field}")
    value = value[component]
if isinstance(value, (dict, list)):
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
elif isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

output_root=$(json_field "$matrix_metadata" output_root)
task_manifest=$(json_field "$matrix_metadata" task_manifest_path)
task_count=$(json_field "$matrix_metadata" task_count)
matrix_id=$(json_field "$matrix_metadata" matrix_id)
original_arm_id=$(json_field "$matrix_metadata" original_arm_id)
crop_arm_id=$(json_field "$matrix_metadata" crop_arm_id)
control_root=${TEXTURE_TWO_ARM_CONTROL_ROOT:-$output_root/runtime/original-crop-step16-supervisor}

mkdir -p "$control_root" "$control_root/setup" "$control_root/final" \
  "$control_root/workers" "$control_root/cache"
exec 9>"$control_root/supervisor.lock"
if ! /usr/bin/flock -n 9; then
  echo "another texture original/Crop supervisor holds $control_root/supervisor.lock" >&2
  exit 1
fi

supervisor_log="$control_root/supervisor.log"
log() {
  local timestamp
  timestamp=$(date --iso-8601=seconds)
  printf '%s %s\n' "$timestamp" "$*" | tee -a "$supervisor_log"
}

python_include_root="$dependency_root/.deps/python312-dev/root/usr/include"
python_include="$python_include_root/python3.12"
site_packages="$dependency_root/.venv312/lib/python3.12/site-packages"
cublas_include="$site_packages/nvidia/cublas/include"
nvrtc_include="$site_packages/nvidia/cuda_nvrtc/include"
cuda_runtime_include="$site_packages/nvidia/cuda_runtime/include"
cuda_runtime_library="$site_packages/nvidia/cuda_runtime/lib"
runtime_link_root="$dependency_root/.eval-runtime-python312-dev/lib"
runtime_link="$runtime_link_root/libcudart.so"
runtime_target="$cuda_runtime_library/libcudart.so.12"

for required_file in \
  "$python_include/Python.h" \
  "$python_include/pyconfig.h" \
  "$python_include_root/x86_64-linux-gnu/python3.12/pyconfig.h" \
  "$cublas_include/cublasLt.h" \
  "$nvrtc_include/nvrtc.h" \
  "$cuda_runtime_include/cuda_runtime.h" \
  "$runtime_target"; do
  if [[ ! -f "$required_file" ]]; then
    log "required development file is absent: $required_file"
    exit 1
  fi
done
if [[ ! -L "$runtime_link" ]] || [[ "$(readlink -f "$runtime_link")" != "$(readlink -f "$runtime_target")" ]]; then
  log "accepted unversioned libcudart link is absent or points elsewhere: $runtime_link"
  exit 1
fi
if ! "$cuda_home/bin/nvcc" --version | grep -Fq 'release 12.8'; then
  log "CUDA compiler is not the accepted 12.8 toolchain: $cuda_home/bin/nvcc"
  exit 1
fi
if ! "$python_bin" - "$original_engine_kwargs" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
if not isinstance(payload, dict):
    raise SystemExit("original engine kwargs must be a JSON object")
owned = {"model", "trust_remote_code", "limit_mm_per_prompt", "tensor_parallel_size", "mm_encoder_attn_backend"}
if collision := owned.intersection(payload):
    raise SystemExit(f"original engine kwargs override runner-owned fields: {sorted(collision)}")
PY
then
  log "original vLLM engine kwargs failed the runner-owned-field contract"
  exit 1
fi
if ! "$python_bin" - "$repo_root/src" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from tgvf_rl.evaluation.texture_bench.stock_qwen import (  # noqa: E402
    STOCK_QWEN_MM_ENCODER_ATTN_BACKEND,
)

if STOCK_QWEN_MM_ENCODER_ATTN_BACKEND != "TORCH_SDPA":
    raise SystemExit("stock-Qwen runner no longer owns vision TORCH_SDPA")
PY
then
  log "stock-Qwen runner failed the B200-safe vision-backend contract"
  exit 1
fi

# Remove ambient compiler/backend selection before installing the accepted
# runtime closure.  In particular, a global TRITON_ATTN selection must not be
# inherited by stock Qwen's multimodal encoder.
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

run_logged() {
  local destination=$1
  shift
  local temporary="$destination.tmp.$$"
  local code=0
  log "run: $(printf '%q ' "$@")"
  "$@" >"$temporary" 2>&1 || code=$?
  if (( code == 0 )); then
    mv "$temporary" "$destination"
    cat "$destination"
    return 0
  fi
  mv "$temporary" "$destination"
  cat "$destination" >&2
  log "command failed with exit $code; log=$destination"
  return "$code"
}

run_json() {
  local destination=$1
  shift
  local temporary="$destination.tmp.$$"
  local error_log="$destination.stderr"
  local code=0
  log "run JSON: $(printf '%q ' "$@")"
  "$@" >"$temporary" 2>"$error_log" || code=$?
  if (( code == 0 )); then
    if ! "$python_bin" - "$temporary" <<'PY'
import json
from pathlib import Path
import sys

json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
PY
    then
      mv "$temporary" "$destination"
      log "command returned malformed JSON; stdout=$destination stderr=$error_log"
      return 1
    fi
    mv "$temporary" "$destination"
    cat "$destination"
    return 0
  fi
  mv "$temporary" "$destination"
  cat "$destination" >&2
  cat "$error_log" >&2
  log "JSON command failed with exit $code; stdout=$destination stderr=$error_log"
  return "$code"
}

execute_crop_plan_step() {
  local step=$1
  "$python_bin" - "$crop_plan" "$step" <<'PY'
import json
import os
from pathlib import Path
import sys

plan_path, step = sys.argv[1:]
plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
argv = plan.get(step)
if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
    raise SystemExit(f"Crop plan {step!r} is not a non-empty argv array")
os.execvpe(argv[0], argv, os.environ)
PY
}

if [[ "$setup_mode" == strict ]]; then
  matrix_validation_path="$control_root/setup/matrix-validation.json"
  log "strict CPU-side matrix validation: matrix=$matrix task_count=$task_count"
  run_json "$matrix_validation_path" \
    "$python_bin" "$repo_root/tools/run_texture_benchmark.py" validate \
    --matrix "$matrix" --verify-images
else
  matrix_validation_path="$control_root/setup/matrix-resume-validation.json"
  log "resume CPU-side matrix validation without repeated image reads: matrix=$matrix"
  run_json "$matrix_validation_path" \
    "$python_bin" "$repo_root/tools/run_texture_benchmark.py" validate \
    --matrix "$matrix" --no-verify-images
fi

crop_plan="$control_root/setup/crop-policy-command.json"
run_json "$crop_plan" \
  "$python_bin" "$repo_root/tools/run_texture_benchmark.py" policy-command \
  --matrix "$matrix" --arm "$crop_arm_id"

if [[ "$setup_mode" == strict ]]; then
  run_logged "$control_root/setup/crop-materialize.log" execute_crop_plan_step materialize
else
  log "resume setup: skipping Crop materialize; existing config closure is required"
fi
crop_config=$(json_field "$crop_plan" config_path)
crop_metadata=$(
  "$python_bin" - "$crop_plan" "$crop_config" "$matrix" \
    "$python_bin" "$repo_root/tools/run_policy_benchmark.py" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config_source = Path(sys.argv[2])
if config_source.is_symlink() or not config_source.is_file():
    raise SystemExit("materialized Crop config must be a regular file")
config_path = config_source.resolve(strict=True)
config = json.loads(config_path.read_text(encoding="utf-8"))
matrix = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
expected_python = sys.argv[4]
expected_runner = sys.argv[5]
crop_arms = [arm for arm in matrix.get("arms", ()) if arm.get("kind") == "crop"]
if len(crop_arms) != 1:
    raise SystemExit("matrix must contain exactly one Crop arm")
crop_arm = crop_arms[0]
expected_config_path = Path(
    matrix.get("output_root"), crop_arm.get("arm_id"), "policy-benchmark-config.json"
).resolve()
if config_path != expected_config_path:
    raise SystemExit("materialized Crop config path differs from matrix arm")
if config.get("gpu_ids") != [0, 1, 2, 3]:
    raise SystemExit("materialized Crop config must bind physical GPUs [0,1,2,3]")
expected_config = {
    "enable_chunked_prefill": False,
    "evaluation_id": f"{matrix.get('matrix_id')}-{crop_arm.get('arm_id')}",
    "evaluation_protocol": crop_arm.get("evaluation_protocol"),
    "expected_optimizer_step": crop_arm.get("expected_optimizer_step"),
    "expected_task_count": matrix.get("task_count"),
    "expected_single_image_count": matrix.get("task_count"),
    "full_model_snapshot_manifest_path": crop_arm.get(
        "full_model_snapshot_manifest_path"
    ),
    "full_model_materialization_receipt_path": crop_arm.get(
        "full_model_materialization_receipt_path"
    ),
    "gpu_memory_utilization": 0.9,
    "image_max_pixels": matrix.get("vision", {}).get("max_pixels"),
    "inference_concurrency_per_gpu": 8,
    "max_model_len": 32768,
    "max_num_batched_tokens": 32768,
    "output_root": str(
        Path(matrix.get("output_root"), crop_arm.get("arm_id"), "evaluation")
    ),
    "snapshot_backend": "full_model",
    "task_manifest_path": matrix.get("task_manifest_path"),
    "task_manifest_sha256": matrix.get("task_manifest_sha256"),
}
for name, value in expected_config.items():
    if config.get(name) != value:
        raise SystemExit(f"materialized Crop config {name} differs from matrix arm")


def sha256_file(value: object, *, owner: str) -> str:
    if not isinstance(value, str):
        raise SystemExit(f"{owner} path is invalid")
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{owner} is absent or not regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


if config.get("full_model_snapshot_manifest_sha256") != sha256_file(
    crop_arm.get("full_model_snapshot_manifest_path"), owner="source snapshot manifest"
):
    raise SystemExit("materialized Crop config source snapshot hash differs")
if config.get("full_model_materialization_receipt_sha256") != sha256_file(
    crop_arm.get("full_model_materialization_receipt_path"),
    owner="source materialization receipt",
):
    raise SystemExit("materialized Crop config source receipt hash differs")
if sha256_file(
    config.get("policy_config_path"), owner="frozen Crop policy config"
) != sha256_file(crop_arm.get("policy_config_path"), owner="source Crop policy config"):
    raise SystemExit("materialized Crop frozen policy config differs from matrix arm")
workers = plan.get("workers_run_concurrently")
if not isinstance(workers, list) or len(workers) != 4:
    raise SystemExit("Crop plan must contain exactly four workers")
plan_arm = plan.get("arm")
for name in ("arm_id", "kind", "backend"):
    if not isinstance(plan_arm, dict) or plan_arm.get(name) != crop_arm.get(name):
        raise SystemExit(f"Crop plan {name} differs from matrix arm")
for rank, worker in enumerate(workers):
    expected = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(rank),
    }
    if worker.get("environment") != expected:
        raise SystemExit(f"Crop rank {rank} CUDA environment differs")
    argv = worker.get("argv")
    expected_tail = [
        "--config",
        str(config_path),
        "--mode",
        "worker",
        "--rank",
        str(rank),
        "--world-size",
        "4",
    ]
    if not isinstance(argv, list) or len(argv) != 10 or argv[2:] != expected_tail:
        raise SystemExit(f"Crop rank {rank} argv differs")
    if argv[:2] != [expected_python, expected_runner]:
        raise SystemExit(f"Crop rank {rank} executable prefix differs")
selected = {
    "config_path": str(config_path),
    "evaluation_root": config.get("output_root"),
}
if not isinstance(selected["evaluation_root"], str):
    raise SystemExit("Crop config output_root is invalid")
print(json.dumps(selected, sort_keys=True, separators=(",", ":")))
PY
)
crop_config=$(json_field "$crop_metadata" config_path)
crop_evaluation_root=$(json_field "$crop_metadata" evaluation_root)

if [[ "$setup_mode" == strict ]]; then
  run_logged "$control_root/setup/crop-prepare.log" execute_crop_plan_step prepare
  run_json "$control_root/setup/crop-validation.json" execute_crop_plan_step validate
fi

stage_resume_validation_evidence() {
  local source=$1
  local destination=$2
  "$python_bin" - "$source" "$destination" <<'PY'
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

source = Path(sys.argv[1]).expanduser().resolve(strict=True)
destination = Path(sys.argv[2]).resolve()
if source.is_symlink() or not source.is_file():
    raise SystemExit(f"resume static-validation evidence is not regular: {source}")
data = source.read_bytes()
payload = json.loads(data)
if not isinstance(payload, dict):
    raise SystemExit("resume static-validation evidence must be one JSON object")
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists() or destination.is_symlink():
    if destination.is_symlink() or not destination.is_file():
        raise SystemExit(f"stored resume evidence is not regular: {destination}")
    if destination.read_bytes() != data:
        raise SystemExit(f"stored resume evidence differs: {destination}")
else:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
PY
}

write_resume_setup_evidence() {
  local destination=$1
  local validation_provenance=$2
  local validation_path=$3
  "$python_bin" - "$destination" "$validation_provenance" "$validation_path" \
    "$matrix_validation_path" "$crop_plan" "$crop_config" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from uuid import uuid4

(
    destination_value,
    validation_provenance,
    validation_value,
    matrix_validation_value,
    crop_plan_value,
    config_value,
) = sys.argv[1:]


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def regular_file(value: str | Path, *, owner: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{owner} is absent or not regular: {path}")
    return path.resolve(strict=True)


def bytes_and_sha256(path: Path) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
            chunks.append(block)
    return b"".join(chunks), digest.hexdigest()


def artifact(value: str | Path, *, owner: str) -> dict[str, object]:
    path = regular_file(value, owner=owner)
    data, digest = bytes_and_sha256(path)
    return {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": digest,
    }


def load_json(value: str | Path, *, owner: str) -> tuple[Path, dict[str, object]]:
    path = regular_file(value, owner=owner)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{owner} must contain one JSON object")
    return path, payload


def require_internal_identity(payload: dict[str, object], *, owner: str) -> str:
    observed = payload.get("identity_sha256")
    content = {name: value for name, value in payload.items() if name != "identity_sha256"}
    expected = canonical_sha256(content)
    if observed != expected:
        raise SystemExit(f"{owner} canonical identity differs")
    return expected


def require_relative_path(value: object, *, owner: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise SystemExit(f"{owner} relative path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SystemExit(f"{owner} relative path is unsafe: {value!r}")
    return relative


def verify_file_records(
    root_value: object,
    records_value: object,
    *,
    owner: str,
    exact_tree: bool,
) -> dict[str, object]:
    if not isinstance(root_value, str):
        raise SystemExit(f"{owner} root is invalid")
    root = Path(root_value).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise SystemExit(f"{owner} root is not a regular directory: {root}")
    if not isinstance(records_value, list) or not records_value:
        raise SystemExit(f"{owner} file records are absent")
    expected_names: set[str] = set()
    small_files_hashed = 0
    logical_bytes = 0
    for index, record in enumerate(records_value):
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise SystemExit(f"{owner} file record {index} fields differ")
        relative = require_relative_path(record["relative_path"], owner=owner)
        relative_text = relative.as_posix()
        if relative_text in expected_names:
            raise SystemExit(f"{owner} contains duplicate file record: {relative_text}")
        expected_names.add(relative_text)
        size = record["size_bytes"]
        digest = record["sha256"]
        if type(size) is not int or size < 0 or not isinstance(digest, str):
            raise SystemExit(f"{owner} file record {relative_text} is invalid")
        path = root.joinpath(*relative.parts)
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise SystemExit(f"{owner} file is absent or not regular: {path}")
        if metadata.st_size != size:
            raise SystemExit(f"{owner} file size changed: {path}")
        logical_bytes += size
        if size <= 64 * 1024 * 1024:
            _, observed = bytes_and_sha256(path)
            if observed != digest:
                raise SystemExit(f"{owner} small-file bytes changed: {path}")
            small_files_hashed += 1
    if exact_tree:
        actual_names: set[str] = set()
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] in {".cache", ".git"}:
                continue
            if path.is_symlink():
                raise SystemExit(f"{owner} contains a symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise SystemExit(f"{owner} contains a special file: {path}")
            actual_names.add(relative.as_posix())
        if actual_names != expected_names:
            raise SystemExit(f"{owner} exact file set changed")
    return {
        "root": str(root),
        "file_count": len(expected_names),
        "logical_bytes": logical_bytes,
        "small_files_hashed": small_files_hashed,
        "large_files_size_checked": len(expected_names) - small_files_hashed,
    }


config_path, config = load_json(config_value, owner="Crop policy config")
if config.get("gpu_ids") != [0, 1, 2, 3]:
    raise SystemExit("resume Crop config GPU mapping differs")
if config.get("snapshot_backend") != "full_model":
    raise SystemExit("resume Crop config must use the full-model snapshot backend")
task_count = config.get("expected_task_count")
single_image_count = config.get("expected_single_image_count")
if type(task_count) is not int or task_count <= 0 or single_image_count != task_count:
    raise SystemExit("resume Crop task counts are invalid")
output_root_value = config.get("output_root")
if not isinstance(output_root_value, str):
    raise SystemExit("resume Crop output root is invalid")
output_root = Path(output_root_value).resolve(strict=True)
runtime_root = output_root / "runtime"

bound_tasks = regular_file(
    runtime_root / "policy-benchmark-tasks.jsonl",
    owner="bound policy benchmark manifest",
)
bound_digest = hashlib.sha256()
bound_size = 0
bound_lines = 0
last_byte = b""
with bound_tasks.open("rb") as handle:
    while block := handle.read(8 * 1024 * 1024):
        bound_digest.update(block)
        bound_size += len(block)
        bound_lines += block.count(b"\n")
        last_byte = block[-1:]
if bound_digest.hexdigest() != config.get("task_manifest_sha256"):
    raise SystemExit("bound policy benchmark manifest SHA256 differs")
if bound_lines != task_count or last_byte != b"\n":
    raise SystemExit("bound policy benchmark manifest line closure differs")

frozen_config = regular_file(config.get("policy_config_path", ""), owner="frozen policy config")
try:
    frozen_config.relative_to(runtime_root)
except ValueError as error:
    raise SystemExit("frozen policy config escapes the evaluation runtime root") from error

frozen_snapshot_path, frozen_snapshot = load_json(
    runtime_root / "frozen-full-model-state/snapshot-manifest.json",
    owner="frozen full-model snapshot manifest",
)
frozen_receipt_path, frozen_receipt = load_json(
    runtime_root / "frozen-full-model-state/materialization-receipt.json",
    owner="frozen full-model materialization receipt",
)
if artifact(frozen_snapshot_path, owner="frozen snapshot manifest")["sha256"] != config.get(
    "full_model_snapshot_manifest_sha256"
):
    raise SystemExit("frozen full-model snapshot manifest SHA256 differs")
if artifact(frozen_receipt_path, owner="frozen materialization receipt")["sha256"] != config.get(
    "full_model_materialization_receipt_sha256"
):
    raise SystemExit("frozen full-model materialization receipt SHA256 differs")
snapshot_identity = require_internal_identity(frozen_snapshot, owner="frozen snapshot manifest")
receipt_identity = require_internal_identity(frozen_receipt, owner="frozen materialization receipt")
if snapshot_identity != config.get("required_snapshot_identity_sha256"):
    raise SystemExit("frozen snapshot identity differs from Crop config")
if frozen_receipt.get("snapshot_identity_sha256") != snapshot_identity:
    raise SystemExit("frozen materialization receipt belongs to another snapshot")

source_closure = verify_file_records(
    frozen_snapshot.get("source_path"),
    frozen_snapshot.get("source_files"),
    owner="full-model source closure",
    exact_tree=False,
)
model_closure = verify_file_records(
    frozen_receipt.get("model_path"),
    frozen_receipt.get("model_files"),
    owner="materialized full-model closure",
    exact_tree=True,
)

identity_path, evaluation_identity = load_json(
    runtime_root / "evaluation-identity.json",
    owner="policy evaluation identity",
)
evaluation_identity_sha256 = require_internal_identity(
    evaluation_identity, owner="policy evaluation identity"
)
if evaluation_identity.get("evaluation_id") != config.get("evaluation_id"):
    raise SystemExit("policy evaluation identity evaluation_id differs")
identity_tasks = evaluation_identity.get("task_manifest")
if not isinstance(identity_tasks, dict) or identity_tasks != {
    "path": str(bound_tasks),
    "sha256": config.get("task_manifest_sha256"),
    "task_count": task_count,
    "single_image_count": single_image_count,
}:
    raise SystemExit("policy evaluation identity task closure differs")
execution = evaluation_identity.get("execution")
expected_execution = {
    "world_size": 4,
    "gpu_ids": [0, 1, 2, 3],
    "max_model_len": config.get("max_model_len"),
    "max_num_batched_tokens": config.get("max_num_batched_tokens"),
    "enable_chunked_prefill": config.get("enable_chunked_prefill"),
    "inference_concurrency_per_gpu": config.get("inference_concurrency_per_gpu"),
    "image_max_pixels": config.get("image_max_pixels"),
}
if not isinstance(execution, dict):
    raise SystemExit("policy evaluation identity execution binding is absent")
for name, value in expected_execution.items():
    if execution.get(name) != value:
        raise SystemExit(f"policy evaluation identity execution {name} differs")
policy_snapshot = evaluation_identity.get("policy_snapshot")
if not isinstance(policy_snapshot, dict):
    raise SystemExit("policy evaluation identity snapshot binding is absent")
expected_snapshot_fields = {
    "snapshot_backend": "full_model",
    "optimizer_step": config.get("expected_optimizer_step"),
    "weights_sha256": config.get("expected_policy_weights_sha256"),
    "snapshot_identity_sha256": snapshot_identity,
    "materialization_identity_sha256": receipt_identity,
}
for name, value in expected_snapshot_fields.items():
    if policy_snapshot.get(name) != value:
        raise SystemExit(f"policy evaluation identity {name} differs")

validation_path, validation = load_json(
    validation_value, owner="Crop static-validation evidence"
)
expected_validation = {
    "evaluation_id": config.get("evaluation_id"),
    "evaluation_identity_sha256": evaluation_identity_sha256,
    "task_count": task_count,
    "single_image_count": single_image_count,
    "optimizer_step": config.get("expected_optimizer_step"),
    "policy_weights_sha256": config.get("expected_policy_weights_sha256"),
    "evaluation_protocol": config.get("evaluation_protocol"),
    "gpu_or_api_used": False,
    "vllm_engine_constructed": False,
}
for name, value in expected_validation.items():
    if validation.get(name) != value:
        raise SystemExit(f"Crop static-validation evidence {name} differs")
processor_proof = validation.get("official_visible_processor_proof")
if not isinstance(processor_proof, dict) or not processor_proof:
    raise SystemExit("Crop static-validation evidence lacks processor proof")

skipped_steps = [
    "full_matrix_verify_images",
    "crop_materialize",
    "crop_prepare",
]
if validation_provenance != "executed_by_supervisor":
    skipped_steps.append("crop_static_validate")
content = {
    "schema_version": "tgvf-texture-original-crop-resume-setup-evidence-v1",
    "setup_mode": "resume",
    "validation_provenance": validation_provenance,
    "skipped_steps": skipped_steps,
    "executed_checks": [
        "matrix_validate_without_image_reads",
        "crop_policy_command_regeneration",
        "crop_config_gpu_and_worker_argv_validation",
        "bound_manifest_sha256_and_line_count",
        "frozen_snapshot_metadata_and_file_size_closure",
        "evaluation_identity_canonical_binding",
        "static_validation_evidence_binding",
    ],
    "matrix_validation": artifact(matrix_validation_value, owner="resume matrix validation"),
    "crop_plan": artifact(crop_plan_value, owner="regenerated Crop command plan"),
    "crop_config": artifact(config_path, owner="existing Crop policy config"),
    "bound_task_manifest": {
        "path": str(bound_tasks),
        "size_bytes": bound_size,
        "sha256": bound_digest.hexdigest(),
        "line_count": bound_lines,
    },
    "frozen_policy_config": artifact(frozen_config, owner="frozen policy config"),
    "frozen_snapshot_manifest": artifact(
        frozen_snapshot_path, owner="frozen snapshot manifest"
    ),
    "frozen_materialization_receipt": artifact(
        frozen_receipt_path, owner="frozen materialization receipt"
    ),
    "full_model_source_closure": source_closure,
    "materialized_model_closure": model_closure,
    "evaluation_identity": artifact(identity_path, owner="policy evaluation identity"),
    "evaluation_identity_sha256": evaluation_identity_sha256,
    "static_validation_evidence": artifact(
        validation_path, owner="Crop static-validation evidence"
    ),
}
payload = {**content, "identity_sha256": canonical_sha256(content)}
encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
destination = Path(destination_value)
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists() or destination.is_symlink():
    raise SystemExit(f"resume setup evidence destination already exists: {destination}")
temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
try:
    with temporary.open("xb") as handle:
        handle.write(encoded.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
print(encoded, end="")
PY
}

if [[ "$setup_mode" == resume ]]; then
  resume_validation_path="$control_root/setup/resume-static-validation.json"
  if [[ -n "$resume_validate_evidence" ]]; then
    stage_resume_validation_evidence "$resume_validate_evidence" "$resume_validation_path"
    validation_provenance=external_preexisting_evidence
    log "resume setup: reusing explicit Crop static-validation evidence"
  elif [[ -f "$resume_validation_path" ]] && [[ ! -L "$resume_validation_path" ]]; then
    validation_provenance=control_root_preexisting_evidence
    log "resume setup: reusing control-root Crop static-validation evidence"
  else
    validation_provenance=executed_by_supervisor
    log "resume setup: no prior static-validation evidence; running one lightweight validation"
    run_json "$resume_validation_path" execute_crop_plan_step validate
  fi
  mkdir -p "$control_root/setup/resume-records"
  resume_record_index=1
  while [[ -e "$control_root/setup/resume-records/attempt-$(printf '%03d' "$resume_record_index").json" ]]; do
    resume_record_index=$((resume_record_index + 1))
  done
  resume_record="$control_root/setup/resume-records/attempt-$(printf '%03d' "$resume_record_index").json"
  write_resume_setup_evidence \
    "$resume_record" "$validation_provenance" "$resume_validation_path"
  log "resume setup closure validated; evidence=$resume_record"
fi

compute_source_closure() {
  "$python_bin" - \
    "$repo_root/src/tgvf_rl" \
    "$dependency_root/.deps/verl/verl" \
    "$repo_root/tools/run_texture_benchmark.py" \
    "$repo_root/tools/run_policy_benchmark.py" \
    "$repo_root/tools/score_texture_benchmark.py" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


records: list[dict[str, object]] = []
for label, value in zip(("tgvf_rl", "verl"), sys.argv[1:3], strict=True):
    root = Path(value).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise SystemExit(f"source closure root is not a regular directory: {root}")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise SystemExit(f"source closure contains a special file: {path}")
        records.append(
            {
                "path": f"{label}/{relative.as_posix()}",
                "size_bytes": metadata.st_size,
                "sha256": sha256_file(path),
            }
        )
for value in sys.argv[3:]:
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"source closure tool is not regular: {path}")
    path = path.resolve(strict=True)
    records.append(
        {
            "path": f"tools/{path.name}",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )
if len({record["path"] for record in records}) != len(records):
    raise SystemExit("source closure contains duplicate paths")
encoded = json.dumps(
    records,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
print(
    json.dumps(
        {
            "schema_version": "tgvf-texture-worker-source-closure-v1",
            "file_count": len(records),
            "logical_bytes": sum(int(record["size_bytes"]) for record in records),
            "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
}

source_closure=$(compute_source_closure)
assert_source_closure() {
  local observed
  observed=$(compute_source_closure) || return 1
  [[ "$observed" == "$source_closure" ]]
}

write_launch_plan() {
  local destination=$1
  local repository_revision
  repository_revision=$(git -C "$repo_root" rev-parse HEAD)
  "$python_bin" - "$destination" "$matrix" "$crop_config" \
    "$repo_root/tools/run_texture_benchmark.py" \
    "$repo_root/src/tgvf_rl/evaluation/texture_bench/stock_qwen.py" \
    "$repo_root/tools/run_policy_benchmark.py" "${BASH_SOURCE[0]}" \
    "$repository_revision" "$original_engine_kwargs" "$source_closure" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

(
    destination,
    matrix,
    crop_config,
    original_runner,
    stock_runner,
    policy_runner,
    supervisor,
    revision,
    engine_kwargs,
    source_closure_json,
) = sys.argv[1:]

def identity(path: str) -> dict[str, object]:
    source = Path(path).resolve(strict=True)
    data = source.read_bytes()
    return {"path": str(source), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

environment_names = (
    "CC", "CXX", "CUDA_HOME", "PATH", "CPATH", "LIBRARY_PATH", "PYTHONPATH",
    "VLLM_USE_V1", "VLLM_WORKER_MULTIPROC_METHOD", "TOKENIZERS_PARALLELISM",
    "PYTHONHASHSEED", "TORCH_DEVICE_BACKEND_AUTOLOAD", "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONNOUSERSITE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
)
payload = {
    "schema_version": "tgvf-texture-original-crop-supervisor-plan-v1",
    "repository_revision": revision,
    "worker_source_closure": json.loads(source_closure_json),
    "sources": [identity(path) for path in (matrix, crop_config, original_runner, stock_runner, policy_runner, supervisor)],
    "python": identity(sys.executable),
    "environment": {name: os.environ[name] for name in environment_names},
    "original": {
        "gpu_ids": [4, 5, 6, 7],
        "world_size": 4,
        "batch_size": int(os.environ["TEXTURE_RECORDED_ORIGINAL_BATCH_SIZE"]),
        "max_tokens": int(os.environ["TEXTURE_RECORDED_ORIGINAL_MAX_TOKENS"]),
        "mm_encoder_attn_backend": "TORCH_SDPA",
        "engine_kwargs": json.loads(engine_kwargs),
    },
    "crop": {"gpu_ids": [0, 1, 2, 3], "world_size": 4},
    "assignment": "ordinal_mod_world_size",
}
encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
path = Path(destination)
if path.exists() and path.read_text(encoding="utf-8") != encoded:
    raise SystemExit(f"existing supervisor launch plan differs: {path}")
if not path.exists():
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != encoded:
                raise SystemExit(f"existing supervisor launch plan differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)
PY
}

export TEXTURE_RECORDED_ORIGINAL_BATCH_SIZE="$original_batch_size"
export TEXTURE_RECORDED_ORIGINAL_MAX_TOKENS="$original_max_tokens"
write_launch_plan "$control_root/launch-plan.json"
unset TEXTURE_RECORDED_ORIGINAL_BATCH_SIZE TEXTURE_RECORDED_ORIGINAL_MAX_TOKENS
if ! assert_source_closure; then
  log "repository worker source closure changed during setup; launch is forbidden"
  exit 1
fi

check_gpus_idle() {
  "$python_bin" - "$nvidia_smi" "$@" <<'PY'
import csv
import io
import json
import subprocess
import sys

nvidia_smi = sys.argv[1]
targets = [int(value) for value in sys.argv[2:]]
if not targets or len(targets) != len(set(targets)):
    raise SystemExit("GPU idle check requires distinct target indices")

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
observed = {}
for row in csv.reader(io.StringIO(inventory.stdout), skipinitialspace=True):
    if not row:
        continue
    if len(row) != 5:
        raise SystemExit(f"unexpected nvidia-smi GPU row: {row!r}")
    index = int(row[0])
    if index in targets:
        observed[index] = {
            "uuid": row[1],
            "name": row[2],
            "memory_used_mib": int(row[3]),
            "utilization_percent": int(row[4]),
        }
if set(observed) != set(targets):
    raise SystemExit(f"target GPU inventory differs: expected={targets}, observed={sorted(observed)}")

problems = []
target_uuids = {record["uuid"] for record in observed.values()}
for index, record in observed.items():
    if record["name"] != "NVIDIA B200":
        problems.append(f"GPU {index} model is {record['name']!r}, expected NVIDIA B200")
    if record["memory_used_mib"] > 16:
        problems.append(f"GPU {index} uses {record['memory_used_mib']} MiB (>16 MiB idle allowance)")
    if record["utilization_percent"] != 0:
        problems.append(f"GPU {index} utilization is {record['utilization_percent']}%")

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
active = []
for row in csv.reader(io.StringIO(processes.stdout), skipinitialspace=True):
    if row and row[0] in target_uuids:
        active.append(row)
if active:
    problems.append(f"target GPUs have compute processes: {active!r}")
if problems:
    raise SystemExit("GPU idle gate failed; no process was signalled:\n" + "\n".join(problems))
print(json.dumps({"idle": True, "gpus": observed}, indent=2, sort_keys=True))
PY
}

idle_snapshot=$(check_gpus_idle "${crop_gpu_ids[@]}" "${original_gpu_ids[@]}")
printf '%s\n' "$idle_snapshot" >"$control_root/gpu-idle-preflight.json"
log "all eight target B200 GPUs passed the fail-closed idle gate"

manager_jobs_active=0
cleanup_managers() {
  local code=$?
  trap - EXIT INT TERM HUP
  if (( manager_jobs_active )); then
    local pid
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      kill -TERM "$pid" 2>/dev/null || true
    done < <(jobs -pr)
    wait 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup_managers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

supervise_rank() (
  set -Eeuo pipefail
  local arm=$1
  local rank=$2
  local gpu=$3
  shift 3
  local -a argv=("$@")
  local rank_root="$control_root/workers/$arm/rank-$rank"
  local cache_root="$control_root/cache/$arm/rank-$rank"
  # vLLM creates a UUID-named ZeroMQ IPC socket below TMPDIR. Unix-domain
  # socket paths are limited to 107 bytes on this host, so the durable control
  # tree is deliberately not suitable as TMPDIR. mktemp provides a short,
  # private rank-local directory while durable caches stay below control_root.
  local socket_tmp
  socket_tmp=$(mktemp -d "/tmp/t2a-${arm:0:1}${rank}.XXXXXX")
  mkdir -p "$rank_root" "$cache_root/triton" "$cache_root/torchinductor" \
    "$cache_root/torch-extensions" "$cache_root/flashinfer"
  if (( ${#socket_tmp} + 37 > 107 )); then
    log "$arm rank=$rank short TMPDIR still exceeds the ZeroMQ IPC path budget"
    return 123
  fi

  local next_attempt=1
  while compgen -G "$rank_root/attempt-$(printf '%03d' "$next_attempt").*" >/dev/null; do
    next_attempt=$((next_attempt + 1))
  done
  local retries=0
  local worker_pid=''

  terminate_owned_worker() {
    [[ -n "$worker_pid" ]] || return 0
    if ! kill -0 "$worker_pid" 2>/dev/null; then
      return 0
    fi
    local pgid
    pgid=$(ps -o pgid= -p "$worker_pid" 2>/dev/null | tr -d ' ')
    if [[ -n "$pgid" ]] && [[ "$pgid" == "$worker_pid" ]]; then
      kill -TERM -- "-$pgid" 2>/dev/null || true
    else
      kill -TERM "$worker_pid" 2>/dev/null || true
    fi
    local count
    for count in $(seq 1 20); do
      kill -0 "$worker_pid" 2>/dev/null || return 0
      sleep 1
    done
    if [[ -n "$pgid" ]] && [[ "$pgid" == "$worker_pid" ]]; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    else
      kill -KILL "$worker_pid" 2>/dev/null || true
    fi
  }
  trap terminate_owned_worker EXIT
  trap 'trap - EXIT; terminate_owned_worker; exit 130' INT
  trap 'trap - EXIT; terminate_owned_worker; exit 143' TERM HUP

  while true; do
    local attempt=$next_attempt
    next_attempt=$((next_attempt + 1))
    local prefix="$rank_root/attempt-$(printf '%03d' "$attempt")"
    if (( retries > 0 )) && ! assert_source_closure; then
      log "$arm rank=$rank source closure changed before retry; mixed-code resume is forbidden"
      return 124
    fi
    if ! check_gpus_idle "$gpu" >"$prefix.gpu-idle.json" 2>"$prefix.gpu-idle.stderr"; then
      log "$arm rank=$rank GPU=$gpu failed the idle gate; refusing to kill the occupant"
      return 125
    fi
    {
      printf 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%q ' "$gpu"
      printf 'TRITON_CACHE_DIR=%q TORCHINDUCTOR_CACHE_DIR=%q ' \
        "$cache_root/triton" "$cache_root/torchinductor"
      printf 'TORCH_EXTENSIONS_DIR=%q FLASHINFER_WORKSPACE_BASE=%q TMPDIR=%q ' \
        "$cache_root/torch-extensions" "$cache_root/flashinfer" "$socket_tmp"
      printf '%q ' "${argv[@]}"
      printf '\n'
    } >"$prefix.command"

    /usr/bin/setsid --wait env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="$gpu" \
      TRITON_CACHE_DIR="$cache_root/triton" \
      TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor" \
      TORCH_EXTENSIONS_DIR="$cache_root/torch-extensions" \
      FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer" \
      TMPDIR="$socket_tmp" \
      "${argv[@]}" >"$prefix.log" 2>&1 &
    worker_pid=$!
    printf '%s\n' "$worker_pid" >"$prefix.pid"
    log "started $arm rank=$rank GPU=$gpu pid=$worker_pid attempt=$attempt log=$prefix.log"

    local exit_code
    if wait "$worker_pid"; then
      exit_code=0
    else
      exit_code=$?
    fi
    printf '{"arm":"%s","rank":%d,"gpu":%d,"pid":%d,"attempt":%d,"exit_code":%d}\n' \
      "$arm" "$rank" "$gpu" "$worker_pid" "$attempt" "$exit_code" >"$prefix.exit.json"
    worker_pid=''
    if (( exit_code == 0 )); then
      log "completed $arm rank=$rank GPU=$gpu attempt=$attempt"
      trap - EXIT INT TERM HUP
      return 0
    fi
    if (( retries >= max_restarts )); then
      log "$arm rank=$rank exhausted $max_restarts restart(s); last_exit=$exit_code"
      trap - EXIT INT TERM HUP
      return "$exit_code"
    fi
    retries=$((retries + 1))
    log "$arm rank=$rank exited $exit_code; retry=$retries/$max_restarts after ${restart_cooldown}s"
    sleep "$restart_cooldown"
  done
)

declare -a manager_pids=()
declare -a manager_labels=()
manager_jobs_active=1
for rank in $(seq 0 $((world_size - 1))); do
  original_argv=(
    "$python_bin" "$repo_root/tools/run_texture_benchmark.py" original
    --matrix "$matrix"
    --batch-size "$original_batch_size"
    --max-tokens "$original_max_tokens"
    --engine-kwargs-json "$original_engine_kwargs"
    --rank "$rank"
    --world-size "$world_size"
    --gpu-ids "${original_gpu_ids[@]}"
    --no-verify-images
  )
  supervise_rank original "$rank" "${original_gpu_ids[$rank]}" "${original_argv[@]}" &
  manager_pids+=("$!")
  manager_labels+=("original-rank-$rank")

  crop_argv=(
    "$python_bin" "$repo_root/tools/run_policy_benchmark.py"
    --config "$crop_config"
    --mode worker
    --rank "$rank"
    --world-size "$world_size"
  )
  supervise_rank crop "$rank" "${crop_gpu_ids[$rank]}" "${crop_argv[@]}" &
  manager_pids+=("$!")
  manager_labels+=("crop-rank-$rank")
done

worker_failure=0
for index in "${!manager_pids[@]}"; do
  pid=${manager_pids[$index]}
  label=${manager_labels[$index]}
  if wait "$pid"; then
    printf '0\n' >"$control_root/workers/$label.manager-exit"
  else
    code=$?
    printf '%s\n' "$code" >"$control_root/workers/$label.manager-exit"
    log "rank supervisor failed: label=$label pid=$pid exit=$code"
    worker_failure=1
  fi
done
manager_jobs_active=0
if (( worker_failure )); then
  log "one or more rank supervisors failed; finalization is forbidden"
  exit 1
fi
if ! assert_source_closure; then
  log "repository worker source closure changed during execution; finalization is forbidden"
  exit 1
fi

original_common=(
  --matrix "$matrix"
  --world-size "$world_size"
  --gpu-ids "${original_gpu_ids[@]}"
  --batch-size "$original_batch_size"
  --max-tokens "$original_max_tokens"
  --engine-kwargs-json "$original_engine_kwargs"
  --no-verify-images
)
original_status_path="$control_root/final/original-status.json"
crop_status_path="$control_root/final/crop-status.json"
run_json "$original_status_path" \
  "$python_bin" "$repo_root/tools/run_texture_benchmark.py" original-status \
  "${original_common[@]}"
run_json "$crop_status_path" \
  "$python_bin" "$repo_root/tools/run_policy_benchmark.py" \
  --config "$crop_config" --mode status --world-size "$world_size"

original_remaining=$(json_field "$original_status_path" remaining)
crop_remaining=$(json_field "$crop_status_path" remaining_single_image)
if [[ "$original_remaining" != 0 ]] || [[ "$crop_remaining" != 0 ]]; then
  log "workers exited zero but coverage is incomplete: original=$original_remaining crop=$crop_remaining"
  exit 1
fi

run_json "$control_root/final/original-finalize.json" \
  "$python_bin" "$repo_root/tools/run_texture_benchmark.py" original-finalize \
  "${original_common[@]}"

original_results="$output_root/$original_arm_id/results.jsonl"
original_score="$output_root/$original_arm_id/score.json"
crop_score="$output_root/$crop_arm_id/score.json"
declare -a crop_results=()
for rank in $(seq 0 $((world_size - 1))); do
  crop_results+=("$crop_evaluation_root/inference/rank-$rank.jsonl")
done

run_json "$control_root/final/original-score.stdout.json" \
  "$python_bin" "$repo_root/tools/score_texture_benchmark.py" \
  --tasks "$task_manifest" --results "$original_results" \
  --output "$original_score" --no-verify-images
run_json "$control_root/final/crop-score.stdout.json" \
  "$python_bin" "$repo_root/tools/score_texture_benchmark.py" \
  --tasks "$task_manifest" --results "${crop_results[@]}" \
  --output "$crop_score" --no-verify-images

summary_path="$control_root/final/supervisor-summary.json"
"$python_bin" - "$summary_path" "$matrix" "$matrix_id" "$task_count" \
  "$original_status_path" "$crop_status_path" "$original_score" "$crop_score" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

(
    destination,
    matrix,
    matrix_id,
    task_count,
    original_status,
    crop_status,
    original_score,
    crop_score,
) = sys.argv[1:]

def artifact(path: str) -> dict[str, object]:
    source = Path(path).resolve(strict=True)
    data = source.read_bytes()
    return {"path": str(source), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

payload = {
    "schema_version": "tgvf-texture-original-crop-supervisor-summary-v1",
    "complete": True,
    "matrix": artifact(matrix),
    "matrix_id": matrix_id,
    "task_count": int(task_count),
    "original_status": artifact(original_status),
    "crop_status": artifact(crop_status),
    "original_score": artifact(original_score),
    "crop_score": artifact(crop_score),
}
encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
path = Path(destination)
if path.exists() and path.read_text(encoding="utf-8") != encoded:
    raise SystemExit(f"existing completion summary differs: {path}")
if not path.exists():
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != encoded:
                raise SystemExit(f"existing completion summary differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)
print(encoded, end="")
PY

trap - EXIT INT TERM HUP
log "two-arm texture benchmark completed and scored: summary=$summary_path"
