#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
run_config="$repo_root/configs/policy/runs/prl_24_d_fmt2_qwen3_instruct_full_crop_bs64_n16_tfree_teacher25_16step_ws8.toml"
launcher="$repo_root/tools/launch_prl21_crop_tfree16.py"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-24-D-FMT2-qwen3-instruct-full-crop-bs64-n16-tfree-teacher25-16step-ws8-sp1
smoke_root="$training_root/smoke-integration"
control_root="$training_root/runtime/supervisor"
log_root="$training_root/logs"
max_restarts=${PRL24_D_TRAIN_MAX_RESTARTS:-20}
cooldown_seconds=${PRL24_D_TRAIN_RESTART_COOLDOWN_SECONDS:-60}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for the matched answer judge" >&2
  exit 1
fi

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL24-D supervisor is active" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/src:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/verl${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=8
export TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8
export TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2
export TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30
export TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0
export WANDB_ENTITY=mio_nora
export WANDB_PROJECT=tgvf-policy-rl
export WANDB_RUN_ID=prl24dfmt2cropt25bs64s16sp1
export WANDB_RESUME=allow

completion_is_valid() {
  local root=$1
  local expected_status=$2
  local expected_step=$3
  local expected_world_size=$4
  "$python_bin" - "$root" "$expected_status" "$expected_step" "$expected_world_size" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected_status = sys.argv[2]
step = int(sys.argv[3])
world_size = int(sys.argv[4])
try:
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if completion.get("status") != expected_status:
    raise SystemExit(1)
checkpoint = root / "checkpoints" / f"global_step_{step}"
actor = checkpoint / "actor"
required = (checkpoint / "data.pt", actor / "huggingface" / "config.json")
if any(not path.is_file() or path.stat().st_size == 0 for path in required):
    raise SystemExit(1)
shards = tuple(actor.glob(f"model_world_size_{world_size}_rank_*.pt"))
if len(shards) != world_size or any(path.stat().st_size == 0 for path in shards):
    raise SystemExit(1)
PY
}

run_with_resume() {
  local stage=$1
  shift
  local attempt=0
  local fatal_pattern='SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 mismatch|schema differs|Crop T-free fixed field differs|CUDA out of memory|OutOfMemoryError|non-finite|NaN|401 Unauthorized|403 Forbidden|invalid_api_key|model_not_found'
  while true; do
    attempt=$((attempt + 1))
    local attempt_log="$log_root/${stage}-attempt-$(printf '%02d' "$attempt").log"
    set +e
    "$@" 2>&1 | tee -a "$attempt_log"
    local code=${PIPESTATUS[0]}
    set -e
    if [[ "$code" == 0 ]]; then
      return 0
    fi
    if rg -q "$fatal_pattern" "$attempt_log"; then
      echo "$stage hit a deterministic failure; preserving recovery state" >&2
      return "$code"
    fi
    if (( attempt >= max_restarts )); then
      echo "$stage retry budget exhausted after $attempt attempts" >&2
      return "$code"
    fi
    echo "$stage was interrupted or hit a transient failure; resuming in ${cooldown_seconds}s" >&2
    sleep "$cooldown_seconds"
  done
}

"$python_bin" "$launcher" --run-config "$run_config" --mode preflight \
  >"$control_root/preflight.json"
touch "$control_root/preflight-accepted"

if ! completion_is_valid "$smoke_root" smoke_checkpoint_complete 1 4; then
  run_with_resume smoke \
    "$python_bin" "$launcher" --run-config "$run_config" --mode smoke --launch
fi
completion_is_valid "$smoke_root" smoke_checkpoint_complete 1 4
touch "$control_root/smoke-accepted"

if ! completion_is_valid "$training_root" target_checkpoint_complete 16 8; then
  run_with_resume step0-to16 \
    "$python_bin" "$launcher" --run-config "$run_config" --mode formal --launch
fi
completion_is_valid "$training_root" target_checkpoint_complete 16 8
touch "$control_root/step16-accepted"

echo "PRL24-D FMT2 native-Crop training completed through optimizer step 16"
