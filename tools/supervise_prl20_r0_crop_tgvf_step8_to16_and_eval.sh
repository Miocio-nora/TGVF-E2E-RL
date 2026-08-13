#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
config="$repo_root/configs/policy/runs/prl_20_r0_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_crop_tgvf_8step_ws8.toml"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-20-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-8step-ws8
control_root="$training_root/runtime/step16-handoff"
log_root="$training_root/logs/step16-handoff"
extension="$control_root/prl20-r0-step8-to16.json"
source_session=${PRL20_R0_SOURCE_TMUX_SESSION:-prl20_r0_formal}
post_train_eval=${PRL20_R0_POST_TRAIN_EVAL:-$repo_root/tools/supervise_prl20_r0_frozen_rp67_tfree_crop_tgvf_step8_step16_paired_evaluation.sh}
poll_seconds=${PRL20_R0_HANDOFF_POLL_SECONDS:-15}
maximum_restarts=${PRL20_R0_STEP16_MAX_RESTARTS:-8}
cooldown_seconds=${PRL20_R0_STEP16_RESTART_COOLDOWN_SECONDS:-10}

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/handoff.lock"
if ! flock -n 9; then
  echo "another PRL20 Step-16 handoff supervisor is active" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=8
export TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8
export TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS=2
export TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS=30
export TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION=0
export WANDB_ENTITY=mio_nora
export WANDB_PROJECT=tgvf-policy-rl
export WANDB_RUN_ID=prl20r0
export WANDB_RESUME=must

checkpoint_is_closed() {
  local step=$1
  local tracker="$training_root/checkpoints/latest_checkpointed_iteration.txt"
  local receipt="$training_root/permanent-checkpoints/global_step_${step}/tgvf_permanent_checkpoint_receipt.json"
  [[ -f "$tracker" && -f "$receipt" ]] || return 1
  "$python_bin" - "$config" "$tracker" "$receipt" "$step" <<'PY'
import json
from pathlib import Path
import sys

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

config = load_policy_e2e_smoke_run_config(
    Path(sys.argv[1]), allow_external_agent_loop_config=True
)
tracker = int(Path(sys.argv[2]).read_text(encoding="utf-8").strip())
receipt = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
step = int(sys.argv[4])
rows = [
    json.loads(line)
    for line in config.output.metrics_path.read_text(encoding="utf-8").splitlines()
    if line
]
if tracker < step or [row.get("optimizer_step") for row in rows[:step]] != list(range(1, step + 1)):
    raise SystemExit(1)
if (
    receipt.get("schema_version")
    != "tgvf.prl15-permanent-checkpoint-receipt.v1"
    or receipt.get("optimizer_step") != step
):
    raise SystemExit(1)
actor = config.output.checkpoint_directory / f"global_step_{step}" / "actor"
for stem in ("model", "optim", "extra_state"):
    shards = tuple(actor.glob(f"{stem}_world_size_{config.distributed.world_size}_rank_*.pt"))
    if len(shards) != config.distributed.world_size or any(path.stat().st_size == 0 for path in shards):
        raise SystemExit(1)
PY
}

source_training_is_running() {
  if ! tmux has-session -t "$source_session" 2>/dev/null; then
    return 1
  fi
  # The formal session intentionally uses remain-on-exit=on, so session
  # existence alone remains true after training has exited.  pane_dead is the
  # exact process-lifecycle boundary needed before another CUDA owner starts.
  [[ "$(tmux display-message -p -t "$source_session" '#{pane_dead}')" != "1" ]]
}

# The active Step-0-to-8 process owns all GPUs. Wait until both its durable
# Step-8 boundary and process exit are visible; never race it for CUDA state.
while source_training_is_running || ! checkpoint_is_closed 8; do
  sleep "$poll_seconds"
done
touch "$control_root/step8-accepted"

if [[ ! -f "$extension" ]]; then
  "$python_bin" "$repo_root/tools/materialize_policy_horizon_extension.py" \
    --run-config "$config" \
    --output "$extension" \
    --extension-id PRL-20-R0-CROP-TGVF-STEP8-TO16 \
    --target-step 16 \
    --repository "$repo_root"
fi
export TGVF_POLICY_HORIZON_EXTENSION_PATH="$extension"
export TGVF_POLICY_HORIZON_EXTENSION_SHA256
TGVF_POLICY_HORIZON_EXTENSION_SHA256=$(sha256sum "$extension" | awk '{print $1}')

attempt=0
while ! checkpoint_is_closed 16; do
  attempt=$((attempt + 1))
  attempt_log="$log_root/step8-to16-attempt-$(printf '%02d' "$attempt").log"
  set +e
  "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
    --run-config "$config" \
    --mode formal \
    --target-step 16 2>&1 | tee -a "$attempt_log"
  code=${PIPESTATUS[0]}
  set -e
  if checkpoint_is_closed 16; then
    break
  fi
  if rg -q 'ValueError:|AssertionError:|SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 differs|schema differs|immutable .*collision|adapter update mode differs|frozen .*changed|CUDA out of memory|OutOfMemoryError|non-finite|NaN' "$attempt_log"; then
    echo "PRL20 Step-16 continuation hit a deterministic failure" >&2
    exit "$code"
  fi
  if (( attempt > maximum_restarts )); then
    echo "PRL20 Step-16 continuation exhausted its retry budget" >&2
    exit "$code"
  fi
  echo "recoverable PRL20 continuation interruption; resuming in ${cooldown_seconds}s" >&2
  sleep "$cooldown_seconds"
done
touch "$control_root/step16-accepted"
unset TGVF_POLICY_HORIZON_EXTENSION_PATH TGVF_POLICY_HORIZON_EXTENSION_SHA256

if [[ ! -x "$post_train_eval" ]]; then
  echo "PRL20 post-training evaluator is absent or not executable: $post_train_eval" >&2
  exit 1
fi
exec "$post_train_eval"
