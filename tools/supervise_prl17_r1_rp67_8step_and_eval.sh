#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
config="$repo_root/configs/policy/runs/prl_17_r1_qwen3_instruct_full_frozen_rp67_bs16_n16_t1_shaped_novisual_8step_ws8.toml"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-17-R1-qwen3-instruct-full-frozen-rp67-bs16-n16-t1-shaped-novisual-8step-ws8
control_root="$training_root/runtime/supervisor"
log_root="$training_root/logs"
receipt="$training_root/permanent-checkpoints/global_step_8/tgvf_permanent_checkpoint_receipt.json"
tracker="$training_root/checkpoints/latest_checkpointed_iteration.txt"
max_restarts=${PRL17_R1_TRAIN_MAX_RESTARTS:-4}
cooldown_seconds=${PRL17_R1_TRAIN_RESTART_COOLDOWN_SECONDS:-30}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for the matched answer judge" >&2
  exit 1
fi

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL17-R1 training supervisor is active" >&2
  exit 1
fi

export TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=8
export TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8
export TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2
export TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30
export TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

step8_is_complete() {
  [[ -f "$receipt" && -f "$tracker" ]] || return 1
  "$python_bin" - "$receipt" "$tracker" <<'PY'
import json
from pathlib import Path
import sys

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tracker = Path(sys.argv[2]).read_text(encoding="utf-8").strip()
if receipt.get("schema_version") != "tgvf.prl15-permanent-checkpoint-receipt.v1":
    raise SystemExit(1)
if receipt.get("optimizer_step") != 8 or tracker != "8":
    raise SystemExit(1)
PY
}

"$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config "$config" \
  --mode formal \
  --target-step 8 \
  --compose-only

deterministic_error_pattern='ValueError:|AssertionError:|SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 differs|schema differs|immutable .*collision|adapter update mode differs|frozen .*changed|CUDA out of memory|OutOfMemoryError|non-finite|NaN'
attempt=0
while ! step8_is_complete; do
  attempt=$((attempt + 1))
  attempt_log="$log_root/formal-supervisor-attempt-$(printf '%02d' "$attempt").log"
  set +e
  "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
    --run-config "$config" \
    --mode formal \
    --target-step 8 2>&1 | tee -a "$attempt_log"
  code=${PIPESTATUS[0]}
  set -e

  if [[ "$code" == 0 ]]; then
    if step8_is_complete; then
      break
    fi
    echo "formal launcher exited successfully without a complete step-8 pair" >&2
    exit 1
  fi
  if rg -q "$deterministic_error_pattern" "$attempt_log"; then
    echo "deterministic training failure; refusing blind retry" >&2
    exit "$code"
  fi
  if (( attempt > max_restarts )); then
    echo "training retry budget exhausted after $attempt attempts" >&2
    exit "$code"
  fi
  echo "recoverable training interruption; resuming from the latest complete pair in ${cooldown_seconds}s" >&2
  sleep "$cooldown_seconds"
done

"$repo_root/tools/supervise_prl17_r1_step0_step8_evaluation.sh"
