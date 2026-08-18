#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
config="$repo_root/configs/policy/runs/prl_24_a_fmt2_qwen3_instruct_full_frozen_rp67_bs64_n16_tfree_teacher25_8step_ws8.toml"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-24-A-FMT2-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-8step-ws8
control_root="$training_root/runtime/step8-to16-supervisor"
log_root="$training_root/logs"
extension="$control_root/prl24-a-fmt2-step8-to16.json"
post_train_eval="$repo_root/tools/supervise_prl24_a_fmt2_bs64_step12_step16_paired_evaluation.sh"
post_eval_joint="$repo_root/tools/supervise_prl24_b_fmt2_joint_bs64_8step.sh"
max_restarts=${PRL24_A_FMT2_CONTINUATION_MAX_RESTARTS:-12}
cooldown_seconds=${PRL24_A_FMT2_CONTINUATION_RESTART_COOLDOWN_SECONDS:-60}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for the matched answer judge" >&2
  exit 1
fi
for executable in "$post_train_eval" "$post_eval_joint"; do
  [[ -x "$executable" ]] || { echo "required handoff is not executable: $executable" >&2; exit 1; }
done

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL24-A FMT2 continuation supervisor is active" >&2
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
export WANDB_RUN_ID=prl24afmt2t25bs64s8
export WANDB_RESUME=must

checkpoint_is_complete() {
  local step=$1
  local receipt="$training_root/permanent-checkpoints/global_step_${step}/tgvf_permanent_checkpoint_receipt.json"
  local tracker="$training_root/checkpoints/latest_checkpointed_iteration.txt"
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

run_with_resume() {
  local attempt=0
  local fatal_pattern='SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 mismatch|schema differs|adapter update mode differs|frozen .*changed|CUDA out of memory|OutOfMemoryError|non-finite|NaN|401 Unauthorized|403 Forbidden|invalid_api_key|model_not_found'
  while true; do
    attempt=$((attempt + 1))
    local attempt_log="$log_root/step8-to16-attempt-$(printf '%02d' "$attempt").log"
    set +e
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$config" --mode formal --target-step 16 2>&1 | tee -a "$attempt_log"
    local code=${PIPESTATUS[0]}
    set -e
    if [[ "$code" == 0 ]]; then
      return 0
    fi
    if rg -q "$fatal_pattern" "$attempt_log"; then
      echo "continuation hit a deterministic failure; preserving recovery state" >&2
      return "$code"
    fi
    if (( attempt >= max_restarts )); then
      echo "continuation retry budget exhausted after $attempt attempts" >&2
      return "$code"
    fi
    echo "continuation was interrupted or hit a transient service failure; resuming in ${cooldown_seconds}s" >&2
    sleep "$cooldown_seconds"
  done
}

if [[ ! -f "$extension" ]]; then
  "$python_bin" "$repo_root/tools/materialize_policy_horizon_extension.py" \
    --run-config "$config" \
    --output "$extension" \
    --extension-id PRL-24-A-FMT2-STEP8-TO16-PERMANENT12 \
    --target-step 16 \
    --permanent-step 8 \
    --permanent-step 12 \
    --permanent-step 16 \
    --repository "$repo_root"
fi
export TGVF_POLICY_HORIZON_EXTENSION_PATH="$extension"
export TGVF_POLICY_HORIZON_EXTENSION_SHA256
TGVF_POLICY_HORIZON_EXTENSION_SHA256=$(sha256sum "$extension" | awk '{print $1}')

"$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config "$config" --mode formal --target-step 16 --compose-only
touch "$control_root/compose-accepted"

if ! checkpoint_is_complete 16; then
  run_with_resume
fi
if ! checkpoint_is_complete 12 || ! checkpoint_is_complete 16; then
  echo "continuation did not close both permanent Step-12 and Step-16 checkpoints" >&2
  exit 1
fi
touch "$control_root/step16-accepted"
unset TGVF_POLICY_HORIZON_EXTENSION_PATH TGVF_POLICY_HORIZON_EXTENSION_SHA256

"$post_train_eval"
touch "$control_root/evaluation-accepted"
exec "$post_eval_joint"

