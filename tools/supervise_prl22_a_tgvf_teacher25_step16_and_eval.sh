#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
config="$repo_root/configs/policy/runs/prl_22_a_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_teacher25_8step_ws8.toml"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-22-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-8step-ws8
control_root="$training_root/runtime/supervisor"
log_root="$training_root/logs"
extension_root="$repo_root/artifacts/policy-horizon-extensions/PRL22-A"
extension="$extension_root/prl22-a-step8-to16.json"
post_train_eval="$repo_root/tools/supervise_prl22_a_tgvf_teacher25_step8_step16_paired_evaluation.sh"
max_restarts=${PRL22_A_TRAIN_MAX_RESTARTS:-8}
cooldown_seconds=${PRL22_A_TRAIN_RESTART_COOLDOWN_SECONDS:-30}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for the matched answer judge" >&2
  exit 1
fi

mkdir -p "$control_root" "$log_root" "$extension_root"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL22-A training supervisor is active" >&2
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
export WANDB_RUN_ID=prl22at25

checkpoint_is_complete() {
  local step=$1
  "$python_bin" - "$config" "$step" <<'PY'
import json
from pathlib import Path
import sys

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

config = load_policy_e2e_smoke_run_config(
    Path(sys.argv[1]), allow_external_agent_loop_config=True
)
step = int(sys.argv[2])
tracker = config.output.checkpoint_directory / "latest_checkpointed_iteration.txt"
receipt_path = (
    config.output.root
    / "permanent-checkpoints"
    / f"global_step_{step}"
    / "tgvf_permanent_checkpoint_receipt.json"
)
try:
    observed = int(tracker.read_text(encoding="utf-8").strip())
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
if observed < step:
    raise SystemExit(1)
if (
    receipt.get("schema_version")
    != "tgvf.prl15-permanent-checkpoint-receipt.v1"
    or receipt.get("optimizer_step") != step
):
    raise SystemExit(1)
PY
}

run_with_resume() {
  local stage=$1
  shift
  local attempt=0
  local deterministic_error_pattern='ValueError:|AssertionError:|SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 differs|schema differs|immutable .*collision|adapter update mode differs|frozen .*changed|CUDA out of memory|OutOfMemoryError|non-finite|NaN'
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
    echo "$stage was operationally interrupted; resuming in ${cooldown_seconds}s" >&2
    sleep "$cooldown_seconds"
  done
}

# CPU-only preflight. Formal W&B is initialized only by the launched trainer.
"$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config "$config" \
  --mode formal \
  --target-step 8 \
  --compose-only
touch "$control_root/compose-accepted"

# Fresh/auto-resumable Step 0 -> 8 using one stable W&B identity.
export WANDB_RESUME=allow
if ! checkpoint_is_complete 8; then
  run_with_resume step0-to8 \
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$config" \
      --mode formal \
      --target-step 8
fi
if ! checkpoint_is_complete 8; then
  echo "formal run did not close the permanent Step-8 checkpoint" >&2
  exit 1
fi
touch "$control_root/step8-accepted"

# Bind the exact Step-8 state, then change only the absolute stopping horizon.
if [[ ! -f "$extension" ]]; then
  "$python_bin" "$repo_root/tools/materialize_policy_horizon_extension.py" \
    --run-config "$config" \
    --output "$extension" \
    --extension-id PRL-22-A-TGVF-TEACHER25-STEP8-TO16 \
    --target-step 16 \
    --repository "$repo_root"
fi
export TGVF_POLICY_HORIZON_EXTENSION_PATH="$extension"
export TGVF_POLICY_HORIZON_EXTENSION_SHA256
TGVF_POLICY_HORIZON_EXTENSION_SHA256=$(sha256sum "$extension" | awk '{print $1}')
export WANDB_RESUME=must

"$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config "$config" \
  --mode formal \
  --target-step 16 \
  --compose-only
if ! checkpoint_is_complete 16; then
  run_with_resume step8-to16 \
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$config" \
      --mode formal \
      --target-step 16
fi
if ! checkpoint_is_complete 16; then
  echo "continuation did not close the permanent Step-16 checkpoint" >&2
  exit 1
fi
touch "$control_root/step16-accepted"

unset TGVF_POLICY_HORIZON_EXTENSION_PATH TGVF_POLICY_HORIZON_EXTENSION_SHA256
exec "$post_train_eval"
