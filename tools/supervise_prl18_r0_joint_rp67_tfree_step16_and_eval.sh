#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
config="$repo_root/configs/policy/runs/prl_18_r0_qwen3_instruct_full_joint_rp67_bs16_n16_tfree_novisual_8step_ws8.toml"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-18-R0-qwen3-instruct-full-joint-rp67-bs16-n16-tfree-novisual-8step-ws8
control_root="$training_root/runtime/supervisor"
log_root="$training_root/logs"
extension="$control_root/prl18-r0-step8-to16.json"
post_train_eval="$repo_root/tools/supervise_prl18_r0_joint_rp67_tfree_step8_step16_paired_evaluation.sh"
smoke_id=joint-rp67-fullstep-v1
smoke_root="$training_root/smoke/$smoke_id"
initial_adapter_sha256=3f60f36589a3c0f3549c12b949eaabb140f6edfac849aa2b25a623bbcde53a14
max_restarts=${PRL18_R0_TRAIN_MAX_RESTARTS:-6}
cooldown_seconds=${PRL18_R0_TRAIN_RESTART_COOLDOWN_SECONDS:-30}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for the matched answer judge" >&2
  exit 1
fi

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL18-R0 training supervisor is active" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=8
export TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8
export TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2
export TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30
export TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0

checkpoint_is_complete() {
  local root=$1
  local step=$2
  local receipt="$root/permanent-checkpoints/global_step_${step}/tgvf_permanent_checkpoint_receipt.json"
  local tracker="$root/checkpoints/latest_checkpointed_iteration.txt"
  [[ -f "$receipt" && -f "$tracker" ]] || return 1
  "$python_bin" - "$receipt" "$tracker" "$step" <<'PY'
import json
from pathlib import Path
import sys

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tracker = Path(sys.argv[2]).read_text(encoding="utf-8").strip()
step = int(sys.argv[3])
if receipt.get("schema_version") != "tgvf.prl15-permanent-checkpoint-receipt.v1":
    raise SystemExit(1)
if receipt.get("optimizer_step") != step or tracker != str(step):
    raise SystemExit(1)
PY
}

smoke_is_complete() {
  local pointer="$smoke_root/runtime-policy-state/latest-lora-snapshot.json"
  local tracker="$smoke_root/checkpoints/latest_checkpointed_iteration.txt"
  local metrics="$smoke_root/metrics.jsonl"
  [[ -f "$pointer" && -f "$tracker" && -f "$metrics" ]] || return 1
  "$python_bin" - "$pointer" "$tracker" "$metrics" "$initial_adapter_sha256" <<'PY'
import json
import math
from pathlib import Path
import sys

pointer = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tracker = Path(sys.argv[2]).read_text(encoding="utf-8").strip()
rows = [json.loads(line) for line in Path(sys.argv[3]).read_text(encoding="utf-8").splitlines() if line]
initial = sys.argv[4]
if tracker != "1" or len(rows) != 1 or rows[0].get("optimizer_step") != 1:
    raise SystemExit(1)
if pointer.get("optimizer_step") != 1 or pointer.get("weights_sha256") == initial:
    raise SystemExit(1)
for value in rows[0].get("rewards", {}).values():
    if isinstance(value, (int, float)) and not math.isfinite(value):
        raise SystemExit(1)
PY
}

run_with_resume() {
  local stage=$1
  shift
  local attempt=0
  local deterministic_error_pattern='ValueError:|AssertionError:|SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 differs|schema differs|immutable .*collision|adapter update mode differs|CUDA out of memory|OutOfMemoryError|non-finite|NaN'
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
    if rg -q "$deterministic_error_pattern" "$attempt_log"; then
      echo "$stage hit a deterministic failure; refusing blind retry" >&2
      return "$code"
    fi
    if (( attempt > max_restarts )); then
      echo "$stage retry budget exhausted after $attempt attempts" >&2
      return "$code"
    fi
    echo "$stage was interrupted; resuming in ${cooldown_seconds}s" >&2
    sleep "$cooldown_seconds"
  done
}

# Engineering gate: one exact world8/BS16/n16 optimizer step, kept out of W&B.
if ! smoke_is_complete; then
  run_with_resume smoke \
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$config" \
      --mode smoke \
      --smoke-id "$smoke_id"
fi
if ! smoke_is_complete; then
  echo "joint-RP67 smoke did not publish a changed Adapter at step 1" >&2
  exit 1
fi
touch "$control_root/smoke-accepted"

# Formal Step 0 -> 8.  This is a fresh run and uses the same W&B identity on resume.
export WANDB_ENTITY=mio_nora
export WANDB_PROJECT=tgvf-policy-rl
export WANDB_RUN_ID=prl18r0u
export WANDB_RESUME=allow
if ! checkpoint_is_complete "$training_root" 8; then
  run_with_resume step0-to8 \
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$config" \
      --mode formal \
      --target-step 8
fi
if ! checkpoint_is_complete "$training_root" 8; then
  echo "formal run did not close the permanent step-8 checkpoint" >&2
  exit 1
fi
touch "$control_root/step8-accepted"

# Bind the exact Step-8 optimizer/data/RNG/Adapter boundary, then continue in place.
"$python_bin" "$repo_root/tools/materialize_policy_horizon_extension.py" \
  --run-config "$config" \
  --output "$extension" \
  --extension-id PRL-18-R0-JOINT-RP67-TFREE-STEP8-TO16 \
  --target-step 16 \
  --repository "$repo_root"
export TGVF_POLICY_HORIZON_EXTENSION_PATH="$extension"
export TGVF_POLICY_HORIZON_EXTENSION_SHA256
TGVF_POLICY_HORIZON_EXTENSION_SHA256=$(sha256sum "$extension" | awk '{print $1}')
if ! checkpoint_is_complete "$training_root" 16; then
  run_with_resume step8-to16 \
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$config" \
      --mode formal \
      --target-step 16
fi
if ! checkpoint_is_complete "$training_root" 16; then
  echo "continuation did not close the permanent step-16 checkpoint" >&2
  exit 1
fi
touch "$control_root/step16-accepted"

unset TGVF_POLICY_HORIZON_EXTENSION_PATH TGVF_POLICY_HORIZON_EXTENSION_SHA256
exec "$post_train_eval"
