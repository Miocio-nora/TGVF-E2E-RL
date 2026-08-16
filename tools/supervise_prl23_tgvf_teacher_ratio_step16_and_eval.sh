#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {teacher50|teacher100}" >&2
  exit 2
fi
arm=$1
shift
case "$arm" in
  teacher50)
    arm_id=PRL23-A
    config="$repo_root/configs/policy/runs/prl_23_a_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_teacher50_8step_ws8.toml"
    training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-23-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher50-8step-ws8
    post_train_eval="$repo_root/tools/supervise_prl23_a_tgvf_teacher50_step8_step16_paired_evaluation.sh"
    extension_id=PRL-23-A-TGVF-TEACHER50-STEP8-TO16
    wandb_run_id=prl23at50
    ;;
  teacher100)
    arm_id=PRL23-B
    config="$repo_root/configs/policy/runs/prl_23_b_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_teacher100_8step_ws8.toml"
    training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-23-B-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher100-8step-ws8
    post_train_eval="$repo_root/tools/supervise_prl23_b_tgvf_teacher100_step8_step16_paired_evaluation.sh"
    extension_id=PRL-23-B-TGVF-TEACHER100-STEP8-TO16
    wandb_run_id=prl23bt100
    ;;
  *)
    echo "unsupported PRL23 teacher-ratio arm: $arm" >&2
    exit 2
    ;;
esac

control_root="$training_root/runtime/supervisor"
log_root="$training_root/logs"
extension_root="$repo_root/artifacts/policy-horizon-extensions/$arm_id"
extension="$extension_root/${arm_id,,}-step8-to16.json"
max_restarts=${PRL23_TRAIN_MAX_RESTARTS:-8}
cooldown_seconds=${PRL23_TRAIN_RESTART_COOLDOWN_SECONDS:-30}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for the matched answer judge" >&2
  exit 1
fi
if [[ ! -x "$post_train_eval" ]]; then
  echo "$arm_id post-training evaluator is absent or not executable" >&2
  exit 1
fi

mkdir -p "$control_root" "$log_root" "$extension_root"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another $arm_id training supervisor is active" >&2
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
export WANDB_RUN_ID="$wandb_run_id"

checkpoint_is_complete() {
  local step=$1
  "$python_bin" - "$config" "$step" <<'PY'
import hashlib
import json
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
tracker = config.output.checkpoint_directory / "latest_checkpointed_iteration.txt"
permanent = config.output.root / "permanent-checkpoints" / f"global_step_{step}"
receipt_path = permanent / "tgvf_permanent_checkpoint_receipt.json"
try:
    observed_step = int(tracker.read_text(encoding="utf-8").strip())
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
if observed_step < step:
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
observed_hashes = {item.name: item.sha256 for item in state.run_identity.hashes}
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

try:
    rows = [
        json.loads(line)
        for line in config.output.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
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
    shards = tuple(
        actor.glob(f"{stem}_world_size_{config.distributed.world_size}_rank_*.pt")
    )
    if len(shards) != config.distributed.world_size or any(
        path.stat().st_size == 0 for path in shards
    ):
        raise SystemExit(1)
PY
}

run_with_resume() {
  local stage=$1
  shift
  local attempt=0
  local deterministic_error_pattern='ValueError:|AssertionError:|SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 differs|schema differs|immutable .*collision|adapter update mode differs|frozen .*changed|CUDA out of memory|OutOfMemoryError|non-finite|NaN|401 Unauthorized|402 Payment Required|403 Forbidden|invalid_api_key|model_not_found'
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

if [[ ! -f "$extension" ]]; then
  "$python_bin" "$repo_root/tools/materialize_policy_horizon_extension.py" \
    --run-config "$config" \
    --output "$extension" \
    --extension-id "$extension_id" \
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
