#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
config="$repo_root/configs/policy/runs/prl_17_r2_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_novisual_8step_ws8.toml"
extension="$repo_root/configs/policy/continuations/prl_17_r2_tfree_step8_to16.json"
extension_sha256=c4519efdf51667fe5a603ccfcb501b2f21f35c53b100b55cef891d25b850da30
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-17-R2-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-novisual-8step-ws8
control_root="$training_root/runtime/supervisor"
log_root="$training_root/logs"
post_train_eval="$repo_root/tools/supervise_prl17_r2_tfree_step0_step8_step16_paired_evaluation.sh"
max_restarts=${PRL17_R2_TFREE_STEP16_MAX_RESTARTS:-4}
cooldown_seconds=${PRL17_R2_TFREE_STEP16_RESTART_COOLDOWN_SECONDS:-30}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for the matched answer judge" >&2
  exit 1
fi

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL17-R2 training supervisor is active" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_POLICY_HORIZON_EXTENSION_PATH="$extension"
export TGVF_POLICY_HORIZON_EXTENSION_SHA256="$extension_sha256"
export TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=8
export TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8
export TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2
export TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30
export TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0
export WANDB_ENTITY=mio_nora
export WANDB_PROJECT=tgvf-policy-rl
export WANDB_RUN_ID=4ksf993e
export WANDB_RESUME=must

step16_is_complete() {
  "$python_bin" - "$config" "$extension" <<'PY'
import json
from pathlib import Path
import sys

from tgvf_rl.policy.horizon_extension import load_policy_horizon_extension
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

config = load_policy_e2e_smoke_run_config(Path(sys.argv[1]))
extension = load_policy_horizon_extension(Path(sys.argv[2]), config)
if extension.target_optimizer_step != 16:
    raise SystemExit(1)
permanent = config.output.root / "permanent-checkpoints/global_step_16"
receipt_path = permanent / "tgvf_permanent_checkpoint_receipt.json"
try:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if (
    receipt.get("schema_version")
    != "tgvf.prl15-permanent-checkpoint-receipt.v1"
    or receipt.get("optimizer_step") != 16
):
    raise SystemExit(1)
PY
}

"$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config "$config" \
  --mode formal \
  --target-step 16 \
  --compose-only

deterministic_error_pattern='ValueError:|AssertionError:|SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 differs|schema differs|immutable .*collision|adapter update mode differs|frozen .*changed|CUDA out of memory|OutOfMemoryError|non-finite|NaN'
attempt=0
while ! step16_is_complete; do
  attempt=$((attempt + 1))
  attempt_log="$log_root/step8-to16-supervisor-attempt-$(printf '%02d' "$attempt").log"
  set +e
  "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
    --run-config "$config" \
    --mode formal \
    --target-step 16 2>&1 | tee -a "$attempt_log"
  code=${PIPESTATUS[0]}
  set -e

  if [[ "$code" == 0 ]]; then
    if step16_is_complete; then
      break
    fi
    echo "formal launcher exited successfully without complete step 16" >&2
    exit 1
  fi
  if rg -q "$deterministic_error_pattern" "$attempt_log"; then
    echo "deterministic continuation failure; refusing blind retry" >&2
    exit "$code"
  fi
  if (( attempt > max_restarts )); then
    echo "continuation retry budget exhausted after $attempt attempts" >&2
    exit "$code"
  fi
  echo "recoverable interruption; resuming latest complete checkpoint in ${cooldown_seconds}s" >&2
  sleep "$cooldown_seconds"
done

if [[ -x "$post_train_eval" ]]; then
  exec "$post_train_eval"
fi
echo "step 16 is complete; paired evaluator is not installed in this runtime yet"
