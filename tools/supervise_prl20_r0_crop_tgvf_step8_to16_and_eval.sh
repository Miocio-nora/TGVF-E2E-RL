#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
config="$repo_root/configs/policy/runs/prl_20_r0_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_crop_tgvf_8step_ws8.toml"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-20-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-8step-ws8
control_root="$training_root/runtime/step16-handoff"
log_root="$training_root/logs/step16-handoff"
extension_root="$repo_root/artifacts/policy-horizon-extensions/PRL-20-R0"
extension="$extension_root/prl20-r0-step8-to16.json"
source_session=${PRL20_R0_SOURCE_TMUX_SESSION:-prl20_r0_formal}
post_train_eval=${PRL20_R0_POST_TRAIN_EVAL:-$repo_root/tools/supervise_prl20_r0_frozen_rp67_tfree_crop_tgvf_step8_step16_paired_evaluation.sh}
poll_seconds=${PRL20_R0_HANDOFF_POLL_SECONDS:-15}
maximum_restarts=${PRL20_R0_STEP16_MAX_RESTARTS:-8}
cooldown_seconds=${PRL20_R0_STEP16_RESTART_COOLDOWN_SECONDS:-10}

mkdir -p "$control_root" "$log_root" "$extension_root"
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

permanent_checkpoint_is_closed() {
  local step=$1
  "$python_bin" - "$config" "$step" <<'PY'
import json
import hashlib
from pathlib import Path
import sys

from tgvf_rl.framework.verl.checkpoint_bridge import (
    read_committed_policy_checkpoint_pair,
)
from tgvf_rl.framework.verl.compatibility import FSDP2BridgeConfig
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

config = load_policy_e2e_smoke_run_config(
    Path(sys.argv[1]), allow_external_agent_loop_config=True
)
step = int(sys.argv[2])
permanent = config.output.root / "permanent-checkpoints" / f"global_step_{step}"
receipt_path = permanent / "tgvf_permanent_checkpoint_receipt.json"
try:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
state, pair = read_committed_policy_checkpoint_pair(
    permanent / "actor",
    fsdp2=FSDP2BridgeConfig(
        world_size=config.distributed.world_size,
        fsdp_size=config.distributed.world_size,
    ),
)
expected_hashes = {
    "run_config": config.identity_sha256,
    "run_config_file": config.source_sha256,
    "dataset_content": config.dataset.runtime_binding.content_sha256,
    "dataset_samples": config.dataset.samples_sha256,
    "dataset_iteration": config.dataset.iteration_identity_sha256,
    "prompt": config.protocol.prompt_sha256,
    "tool_schema": config.protocol.tool_schema_sha256,
    "representation_artifact_file": config.representation.artifact_file_sha256,
}
observed_hashes = {
    item.name: item.sha256 for item in state.run_identity.hashes
}
identity_sha256 = hashlib.sha256(
    json.dumps(
        state.run_identity.to_checkpoint_mapping(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()
if state.run_identity.run_id != config.run_id:
    raise SystemExit(1)
if any(observed_hashes.get(name) != value for name, value in expected_hashes.items()):
    raise SystemExit(1)
if state.progress.optimizer_step != step or pair.optimizer_step != step:
    raise SystemExit(1)
rows = [
    json.loads(line)
    for line in config.output.metrics_path.read_text(encoding="utf-8").splitlines()
    if line
]
if [row.get("optimizer_step") for row in rows[:step]] != list(range(1, step + 1)):
    raise SystemExit(1)
if (
    receipt.get("schema_version")
    != "tgvf.prl15-permanent-checkpoint-receipt.v1"
    or receipt.get("optimizer_step") != step
    or receipt.get("run_identity_sha256") != identity_sha256
    or receipt.get("project_state_sha256") != state.integrity_sha256
    or receipt.get("pair_integrity_sha256") != pair.integrity_sha256
):
    raise SystemExit(1)
for path in (permanent / "data.pt", permanent / "actor/fsdp_config.json"):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(1)
actor = permanent / "actor"
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
while source_training_is_running; do sleep "$poll_seconds"; done
if ! permanent_checkpoint_is_closed 8; then
  echo "PRL20 source training exited without a closed permanent Step-8 checkpoint" >&2
  exit 1
fi
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

"$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_supervisor \
  --run-config "$config" \
  --target-step 16 \
  --log-directory "$log_root/supervisor" \
  --maximum-restarts "$maximum_restarts" \
  --cooldown-seconds "$cooldown_seconds"
if ! permanent_checkpoint_is_closed 16; then
  echo "PRL20 continuation exited without a closed permanent Step-16 checkpoint" >&2
  exit 1
fi
touch "$control_root/step16-accepted"
unset TGVF_POLICY_HORIZON_EXTENSION_PATH TGVF_POLICY_HORIZON_EXTENSION_SHA256

if [[ ! -x "$post_train_eval" ]]; then
  echo "PRL20 post-training evaluator is absent or not executable: $post_train_eval" >&2
  exit 1
fi
exec "$post_train_eval"
