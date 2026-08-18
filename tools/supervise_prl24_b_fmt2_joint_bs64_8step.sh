#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
config="$repo_root/configs/policy/runs/prl_24_b_fmt2_joint_qwen3_instruct_full_rp67_bs64_n16_tfree_teacher25_8step_ws8.toml"
canary_config="$repo_root/configs/policy/runs/prl_24_b_fmt2_joint_c0_qwen3_instruct_full_rp67_bs4_n2_tfree_teacher25_1step_ws4.toml"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-24-B-FMT2-JOINT-qwen3-instruct-full-joint-rp67-bs64-n16-tfree-teacher25-8step-ws8
canary_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-24-B-FMT2-JOINT-C0-qwen3-instruct-full-joint-rp67-bs4-n2-tfree-teacher25-1step-ws4
control_root="$training_root/runtime/supervisor"
log_root="$training_root/logs"
initial_adapter_sha256=3f60f36589a3c0f3549c12b949eaabb140f6edfac849aa2b25a623bbcde53a14
max_restarts=${PRL24_B_FMT2_TRAIN_MAX_RESTARTS:-12}
cooldown_seconds=${PRL24_B_FMT2_TRAIN_RESTART_COOLDOWN_SECONDS:-60}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for the matched answer judge" >&2
  exit 1
fi

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL24-B FMT2 Joint supervisor is active" >&2
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
export WANDB_RUN_ID=prl24bfmt2jointt25bs64s8
export WANDB_RESUME=allow

checkpoint_is_complete() {
  local root=$1
  local step=$2
  local receipt="$root/permanent-checkpoints/global_step_${step}/tgvf_permanent_checkpoint_receipt.json"
  local tracker="$root/checkpoints/latest_checkpointed_iteration.txt"
  [[ -s "$receipt" && -s "$tracker" ]] || return 1
  "$python_bin" - "$receipt" "$tracker" "$step" <<'PY'
import json
from pathlib import Path
import sys

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
observed = int(Path(sys.argv[2]).read_text(encoding="utf-8").strip())
step = int(sys.argv[3])
if receipt.get("schema_version") != "tgvf.prl15-permanent-checkpoint-receipt.v1":
    raise SystemExit(1)
if receipt.get("optimizer_step") != step or observed < step:
    raise SystemExit(1)
PY
}

canary_is_complete() {
  local pointer="$canary_root/runtime-policy-state/latest-lora-snapshot.json"
  local tracker="$canary_root/checkpoints/latest_checkpointed_iteration.txt"
  [[ -s "$pointer" && -s "$tracker" ]] || return 1
  "$python_bin" - "$pointer" "$tracker" "$initial_adapter_sha256" <<'PY'
import json
from pathlib import Path
import sys

pointer = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if Path(sys.argv[2]).read_text(encoding="utf-8").strip() != "1":
    raise SystemExit(1)
if pointer.get("optimizer_step") != 1 or pointer.get("weights_sha256") == sys.argv[3]:
    raise SystemExit(1)
PY
}

run_with_resume() {
  local stage=$1
  shift
  local attempt=0
  local fatal_pattern='SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 mismatch|schema differs|adapter update mode differs|CUDA out of memory|OutOfMemoryError|non-finite|NaN|401 Unauthorized|403 Forbidden|invalid_api_key|model_not_found'
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
    echo "$stage was interrupted or hit a transient service failure; resuming in ${cooldown_seconds}s" >&2
    sleep "$cooldown_seconds"
  done
}

"$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config "$canary_config" --mode canary --compose-only
"$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config "$config" --mode formal --target-step 8 --compose-only
touch "$control_root/compose-accepted"

if ! canary_is_complete; then
  run_with_resume canary \
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$canary_config" --mode canary
fi
if ! canary_is_complete; then
  echo "Joint RP67 canary did not publish a changed Adapter" >&2
  exit 1
fi
touch "$control_root/canary-accepted"

if ! checkpoint_is_complete "$training_root" 8; then
  run_with_resume step0-to8 \
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$config" --mode formal --target-step 8
fi
if ! checkpoint_is_complete "$training_root" 8; then
  echo "PRL24-B FMT2 Joint did not close its Step-8 checkpoint" >&2
  exit 1
fi
touch "$control_root/step8-accepted"

