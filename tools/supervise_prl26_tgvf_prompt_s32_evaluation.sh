#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
plan="$repo_root/configs/evaluation/prl26_cd_tgvf_target_prompt_pair_s32_pixel512_coredev2511_plan.json"
eval_root="$main_root/artifacts/evaluation/PRL26-CD-TGVF-PROMPT-PAIR-S32-PIXEL512-COREDEV2511-V1"
runtime_root="$eval_root/runtime"
log_root="$eval_root/logs"
result="$eval_root/tgvf-target-prompt-s32-pixel512-results.json"
runner="$repo_root/tools/run_prl15_paired_evaluation.py"
benchmark_runner="$repo_root/tools/run_policy_benchmark.py"
proof_validator="$repo_root/tools/validate_prl26_train512_processor_proof.py"
summarizer="$repo_root/tools/summarize_prl26_tgvf_prompt_s32_evaluation.py"

short_training_root="$main_root/artifacts/policy/PRL-26-C-train512-s32-parity-tgvf-short-qwen3-instruct-bs16-n16-teacher25-ws8"
full_training_root="$main_root/artifacts/policy/PRL-26-D-train512-s32-parity-tgvf-target-guide-v2-qwen3-instruct-bs16-n16-teacher25-ws8"
short_receipt="$short_training_root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"
full_receipt="$full_training_root/permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"

admitted_head=${1:-}
if [[ ! "$admitted_head" =~ ^[0-9a-f]{40}$ ]]; then
  echo "exact admitted implementation HEAD is required" >&2
  exit 1
fi
for path in "$python_bin" "$plan" "$runner" "$benchmark_runner" \
  "$proof_validator" "$summarizer"; do
  if [[ ! -f "$path" ]]; then
    echo "required PRL-26 C/D evaluation file is absent: $path" >&2
    exit 1
  fi
done
for receipt in "$short_receipt" "$full_receipt"; do
  if [[ ! -s "$receipt" ]]; then
    echo "PRL-26 C/D S32 completion receipt is absent: $receipt" >&2
    exit 1
  fi
done

observed_root=$(git -C "$repo_root" rev-parse --show-toplevel)
observed_head=$(git -C "$repo_root" rev-parse HEAD)
dirty=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
if [[ "$observed_root" != "$repo_root" \
      || "$observed_head" != "$admitted_head" \
      || -n "$dirty" ]]; then
  echo "clean PRL-26 C/D evaluation worktree identity differs" >&2
  exit 1
fi

# Evaluation state is created only after both immutable S32 receipts exist.
mkdir -p "$runtime_root" "$log_root"
exec 9>"$runtime_root/supervisor.lock"
flock -n 9 || {
  echo "PRL-26 C/D target-prompt evaluator is already active" >&2
  exit 1
}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo_root/src:$main_root/.deps/verl${PYTHONPATH:+:$PYTHONPATH}"
export TGVF_REPOSITORY_ROOT="$repo_root"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

phase=initializing
active_pid=

timestamp() {
  date '+%F %T %Z'
}

stop_process_group() {
  local pid=${1:-}
  [[ -n "$pid" ]] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$pid" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local status=$?
  set +e
  stop_process_group "$active_pid"
  if (( status == 0 )); then
    rm -f "$runtime_root/failed"
  else
    rm -f "$runtime_root/evaluation-complete"
    printf 'status=failed\nphase=%s\ntime=%s\nexit_status=%s\n' \
      "$phase" "$(timestamp)" "$status" >"$runtime_root/failed"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'phase=signal; exit 130' INT TERM
rm -f "$runtime_root/failed" "$runtime_root/evaluation-complete"

run_group() {
  local log_path=$1
  shift
  setsid "$@" >"$log_path" 2>&1 9>&- &
  active_pid=$!
  set +e
  wait "$active_pid"
  local status=$?
  set -e
  active_pid=
  return "$status"
}

phase=validating_static_target_prompt_pair_plan
env CUDA_VISIBLE_DEVICES= "$python_bin" - "$runner" "$plan" \
  >"$log_root/static-plan-validation.json" <<'PY'
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

runner_path, plan_path = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("prl26_target_pair_runner", runner_path)
if spec is None or spec.loader is None:
    raise RuntimeError("paired evaluator module cannot be loaded")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
plan = module._load_plan(plan_path.resolve())
print(json.dumps({
    "schema_version": plan["schema_version"],
    "evaluation_id": plan["evaluation_id"],
    "seed_protocol_sha256": plan["paired_rng"]["seed_protocol_sha256"],
}, indent=2, sort_keys=True))
PY

phase=preparing_short_full_snapshots
run_group "$log_root/prepare.log" \
  "$python_bin" "$runner" --plan "$plan" --mode prepare \
  --output-root "$eval_root" --gpu-ids 0 1 2 3 4 5 6 7

for arm in short full; do
  config="$eval_root/$arm/benchmark-config.json"
  validation="$eval_root/$arm/logs/prl26-pixel512-static-validation.json"
  proof="$eval_root/$arm/runtime/pixel512-processor-proof.json"
  phase="validating_${arm}_pixel512_prompt_processor"
  env CUDA_VISIBLE_DEVICES= "$python_bin" "$benchmark_runner" \
    --config "$config" --mode validate --world-size 4 \
    >"$validation" 2>"$eval_root/$arm/logs/prl26-pixel512-static-validation.stderr.log"
  env CUDA_VISIBLE_DEVICES= "$python_bin" "$proof_validator" \
    --arm "$arm" --config "$config" --validation-json "$validation" \
    --output "$proof" >"$log_root/${arm}-pixel512-proof.json"
done
touch "$runtime_root/processor-proofs-complete"

phase=running_short_full_parallel_four_plus_four_inference
run_group "$log_root/inference.log" \
  "$python_bin" "$runner" --plan "$plan" --mode infer \
  --output-root "$eval_root" --gpu-ids 0 1 2 3 4 5 6 7
touch "$runtime_root/parallel-inference-complete"

# Two disjoint TP=2 Qwen2.5-72B judges use GPU0-1 and GPU2-3. Both arms retain
# all seven official scorer reports; GPT fallback remains disabled.
phase=scoring_short_full_seven_subsets
run_group "$log_root/scoring.log" \
  "$python_bin" "$runner" --plan "$plan" --mode score \
  --output-root "$eval_root" --gpu-ids 0 1 2 3

phase=publishing_headline_and_tool_usage_table
env CUDA_VISIBLE_DEVICES= "$python_bin" "$summarizer" \
  --evaluation-root "$eval_root" --output "$result" \
  >"$log_root/result-table.json"

if [[ ! -s "$eval_root/paired-summary.json" \
      || ! -s "$eval_root/evaluation-complete" \
      || ! -s "$result" ]]; then
  echo "PRL-26 C/D evaluator omitted a completion artifact" >&2
  exit 1
fi

phase=complete
touch "$runtime_root/evaluation-complete"
printf '[%s] PRL-26 C/D target-prompt S32 evaluation complete\n' "$(timestamp)"
