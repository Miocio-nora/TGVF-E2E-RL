#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
config="$repo_root/configs/policy/runs/prl_25_c_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_teacher25_80step_ws8.toml"
training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8
control_root="$training_root/runtime/supervisor"
log_root="$training_root/logs"
post_train_eval="$repo_root/tools/supervise_prl25_c_tgvf_six_point_evaluation.sh"
max_restarts=${PRL25_C_TRAIN_MAX_RESTARTS:-20}
cooldown_seconds=${PRL25_C_TRAIN_RESTART_COOLDOWN_SECONDS:-60}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for the matched answer judge" >&2
  exit 1
fi
if [[ ! -x "$post_train_eval" ]]; then
  echo "PRL25-C post-training evaluator is absent or not executable" >&2
  exit 1
fi

# A first launch must begin at S0. Existing canonical checkpoints are accepted
# only as recovery state for this same run identity.
tracker="$training_root/checkpoints/latest_checkpointed_iteration.txt"
if [[ ! -f "$tracker" ]] && { [[ -f "$training_root/metrics.jsonl" ]] || [[ -d "$training_root/permanent-checkpoints" ]]; }; then
  echo "PRL25-C has ambiguous pre-existing artifacts without a checkpoint tracker" >&2
  exit 1
fi

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
if ! flock -n 9; then
  echo "another PRL25-C training supervisor is active" >&2
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
export WANDB_RUN_ID=prl25ct25s80
export WANDB_RESUME=allow

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
permanent = config.output.root / "permanent-checkpoints" / f"global_step_{step}"
receipt_path = permanent / "tgvf_permanent_checkpoint_receipt.json"
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
actor = permanent / "actor"
required = (permanent / "data.pt", actor / "fsdp_config.json")
if any(not path.is_file() or path.stat().st_size == 0 for path in required):
    raise SystemExit(1)
for stem in ("model", "optim", "extra_state"):
    shards = tuple(actor.glob(f"{stem}_world_size_8_rank_*.pt"))
    if len(shards) != 8 or any(path.stat().st_size == 0 for path in shards):
        raise SystemExit(1)
PY
}

run_with_resume() {
  local stage=$1
  shift
  local attempt=0
  local fatal_pattern='SyntaxError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:|identity differs|SHA256 mismatch|SHA256 differs|schema differs|adapter update mode differs|frozen .*changed|CUDA out of memory|OutOfMemoryError|non-finite|NaN|401 Unauthorized|403 Forbidden|invalid_api_key|model_not_found'
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
      echo "$stage hit a deterministic failure; stopping with its recovery state intact" >&2
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

# CPU-only contract checks. The formal trainer is the first process that
# allocates GPUs or initializes W&B; no smoke samples enter this run.
"$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
  --run-config "$config" --mode formal --target-step 80 --compose-only
"$python_bin" - "$config" <<'PY'
import json
from pathlib import Path
import sys

from tgvf_rl.framework.verl.trainable_tgvf_launcher import (
    build_trainable_tgvf_verl_launch_plan,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

config = load_policy_e2e_smoke_run_config(
    Path(sys.argv[1]), allow_external_agent_loop_config=True
)
plan = build_trainable_tgvf_verl_launch_plan(config, mode="formal", target_step=80)
contract = plan.overrides["actor_rollout_ref.rollout.custom"]["actor_batch_contract"]
expected = {
    "global_prompt_batch_size": 16,
    "rollouts_per_prompt": 16,
    "configured_gradient_accumulation_steps": 1,
    "derived_actor_forward_backward_microbatches": 1,
    "optimizer_steps_per_trainer_step": 1,
}
if any(contract.get(key) != value for key, value in expected.items()):
    raise SystemExit(f"PRL25-C actor batch contract differs: {contract!r}")
if config.reward.protocol_error_penalty != 2.0:
    raise SystemExit("PRL25-C must use FMT2 protocol_error_penalty=2.0")

samples_path = config.dataset.root / "samples.jsonl"
rows = []
with samples_path.open(encoding="utf-8") as handle:
    for _ in range(80 * 16):
        rows.append(json.loads(next(handle)))
for start in range(0, len(rows), 16):
    teacher = sum(row.get("data_source") == "teacher" for row in rows[start:start + 16])
    if teacher != 4:
        raise SystemExit(f"Teacher25 BS16 slice {start // 16} has {teacher} teacher rows")
PY
touch "$control_root/preflight-accepted"

if ! checkpoint_is_complete 80; then
  run_with_resume step0-to80 \
    "$python_bin" -m tgvf_rl.framework.verl.trainable_tgvf_launcher \
      --run-config "$config" --mode formal --target-step 80
fi
if ! checkpoint_is_complete 80; then
  echo "PRL25-C did not close the permanent Step-80 checkpoint" >&2
  exit 1
fi
touch "$control_root/step80-accepted"

exec "$post_train_eval"
