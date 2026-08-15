#!/usr/bin/env bash
set -euo pipefail

resume_scoring_only=false
case "${1:-}" in
  "") ;;
  --resume-scoring) resume_scoring_only=true ;;
  *)
    echo "usage: $0 [--resume-scoring]" >&2
    exit 2
    ;;
esac

if [[ "$resume_scoring_only" == true ]]; then
  scoring_view=coredev-official-v1-recovery1
else
  scoring_view=coredev-official-v1
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
eval_repo=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl13-integration
python_bin="$main_root/.venv312/bin/python"
vllm_bin="$main_root/.venv312/bin/vllm"
train_root="$main_root/artifacts/policy/PRL-21-R0-qwen3-instruct-full-crop-bs16-n16-tfree-16step-ws8"
evaluation_id=PRL21-R0-CROP-TFREE-COREDEV2511-STEP8-STEP16-TEMP1-SEED42-V1
eval_root="$train_root/evaluation/$evaluation_id"
task_manifest="$main_root/artifacts/evaluation/CoreDev2511-official-visible-v1/tasks.jsonl"
source_root=/nvmesv/dredvpn009/datasets/benchmarks/coredev_2511_vlmevalkit_7055d301_v1
mathverse_json=/nvmesv/dredvpn009/datasets/benchmarks/mathverse/snapshot/testmini.json
contract="$repo_root/configs/policy/runs/prl_13_a_qwen3_instruct_grpo_bs256_n16_native_crop_t1_stratified_80step_gpu0123.toml"
overlay="$repo_root/configs/policy/runs/prl_21_r0_qwen3_instruct_full_crop_bs16_n16_tfree_16step_ws8.toml"
# The suffix is sha256(evaluation identity + scoring-view identity).  VLMEvalKit
# only discovers timestamp or legacy TYYYYMMDD_G<hex> run directories for reuse.
run_id=T20260815_Gc2f1cd6d5d93579e517ad6a7dd97fd43782ad9a2a67645bbc73fb7967b5daf8a
judge_url=http://127.0.0.1:8012/v1
judge_pid=

mkdir -p "$eval_root/logs" "$eval_root/runtime"
exec 9>"$eval_root/runtime/evaluation.lock"
if ! flock -n 9; then
  echo "another PRL21 evaluator is active" >&2
  exit 1
fi
exec > >(tee -a "$eval_root/logs/supervisor.log") 2>&1

cleanup() {
  if [[ -n "$judge_pid" ]] && kill -0 "$judge_pid" 2>/dev/null; then
    kill "$judge_pid" 2>/dev/null || true
    wait "$judge_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ ! -s "$train_root/completion.json" ]]; then
  echo "PRL21 training completion is absent" >&2
  exit 1
fi

validate_scoring_inputs() {
  local step=$1
  local status="$eval_root/logs/status-step${step}.json"
  local summary="$eval_root/logs/scoring-materialize-${scoring_view}-step${step}.json"

  [[ -s "$status" ]] || {
    echo "step${step} inference status is absent" >&2
    return 1
  }
  "$python_bin" - "$status" "$summary" "$run_id" <<'PY'
import hashlib
import json
import pathlib
import sys

status_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
expected_run_id = sys.argv[3]
status = json.loads(status_path.read_text())
if status.get("completed_single_image") != 2240 or status.get("remaining_single_image") != 0:
    raise SystemExit(f"incomplete inference status: {status_path}")
if status.get("multi_image_pending_protocol_decision") != 271:
    raise SystemExit(f"unexpected multi-image count: {status_path}")

if not summary_path.is_file():
    raise SystemExit(f"scoring materialization summary is absent: {summary_path}")
summary = json.loads(summary_path.read_text())
if summary.get("official_row_count") != 2511:
    raise SystemExit(f"unexpected official row count: {summary_path}")
if summary.get("observed_single_image_count") != 2240:
    raise SystemExit(f"unexpected single-image count: {summary_path}")
if summary.get("unsupported_multi_image_count") != 271:
    raise SystemExit(f"unexpected unsupported multi-image count: {summary_path}")
if summary.get("run_id") != expected_run_id:
    raise SystemExit(f"unexpected scoring run identity: {summary_path}")
slices = summary.get("slices", [])
expected_counts = {
    "VStarBench": (191, 191, 0),
    "HRBench4K": (200, 200, 0),
    "BLINK": (420, 180, 240),
    "OCRBench_v2": (600, 600, 0),
    "MMMU_Pro_10c": (300, 269, 31),
    "MathVista_MINI": (300, 300, 0),
    "MathVerse_MINI": (500, 500, 0),
}
if {item.get("dataset") for item in slices} != set(expected_counts):
    raise SystemExit(f"unexpected scoring slices: {summary_path}")
for item in slices:
    counts = (
        item.get("official_row_count"),
        item.get("observed_single_image_count"),
        item.get("unsupported_multi_image_count"),
    )
    if counts != expected_counts[item["dataset"]]:
        raise SystemExit(f"unexpected slice counts for {item['dataset']}: {counts}")
    prediction = pathlib.Path(item["prediction_file"])
    if not prediction.is_file() or prediction.stat().st_size == 0:
        raise SystemExit(f"prediction file is absent or empty: {prediction}")
    manifest_path = pathlib.Path(item["manifest"])
    manifest = json.loads(manifest_path.read_text())
    derived = manifest.get("derived", {})
    verification = manifest.get("verification", {})
    if pathlib.Path(derived.get("path", "")) != prediction:
        raise SystemExit(f"manifest prediction path mismatch: {manifest_path}")
    if derived.get("row_count") != item["official_row_count"]:
        raise SystemExit(f"manifest prediction row count mismatch: {manifest_path}")
    digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
    if derived.get("sha256") != digest:
        raise SystemExit(f"manifest prediction hash mismatch: {manifest_path}")
    required_verification = (
        "index_order_and_values_identical",
        "non_prediction_source_fields_verified",
        "unchanged_non_prediction_source_fields_identical",
    )
    if not all(verification.get(key) is True for key in required_verification):
        raise SystemExit(f"non-prediction field verification failed: {manifest_path}")
PY
}

materialize_scoring_views() {
  local step scoring_root summary
  for step in 8 16; do
    scoring_root="$eval_root/step${step}/scoring/$scoring_view"
    summary="$eval_root/logs/scoring-materialize-${scoring_view}-step${step}.json"
    if [[ -s "$summary" ]] && validate_scoring_inputs "$step"; then
      echo "step${step} immutable scoring view already validated; reusing it"
      continue
    fi
    if [[ -e "$scoring_root" ]] && [[ -n "$(find "$scoring_root" -mindepth 1 -print -quit)" ]]; then
      echo "partial immutable step${step} scoring view requires a new recovery identity: $scoring_root" >&2
      return 1
    fi
    "$python_bin" "$eval_repo/tools/materialize_policy_coredev_scoring.py" \
      --inference-root "$eval_root/step${step}/inference" \
      --tasks "$task_manifest" \
      --source-root "$source_root" \
      --output-root "$scoring_root" \
      --evaluation-id "PRL21-R0-CROP-TFREE-COREDEV2511-STEP${step}-TEMP1-SEED42-V1" \
      --run-id "$run_id" \
      --mathverse-source-json "$mathverse_json" \
      > "$summary"
  done
}

if [[ "$resume_scoring_only" == false ]]; then
  cp "$overlay" "$eval_root/runtime/training-overlay.toml"
  cat > "$eval_root/runtime/step0-reuse.json" <<'JSON'
{
  "schema_version": "tgvf.policy-step0-reuse.v1",
  "source": "PRL14 clean-final Crop Step0 / original Qwen3-VL-8B-Instruct",
  "reason": "PRL21 keeps the identical base model, clean-final Crop prompt, tool schema, inference sampling and seed; only the training reward changes",
  "rerun": false
}
JSON

step_source() {
  case "$1" in
    8) printf '%s\n' "$train_root/permanent-checkpoints/global_step_8" ;;
    16) printf '%s\n' "$train_root/permanent-checkpoints/global_step_16" ;;
    *) return 1 ;;
  esac
}

for step in 8 16; do
  source_path=$(step_source "$step")
  "$python_bin" "$eval_repo/tools/materialize_prl13_full_model.py" materialize \
    --run-config "$contract" \
    --optimizer-step "$step" \
    --source "$source_path" \
    --runtime-fsdp-world-size 8 \
    --snapshot-manifest "$eval_root/runtime/step${step}-snapshot.json" \
    --receipt "$eval_root/runtime/step${step}-receipt.json" \
    > "$eval_root/logs/materialize-step${step}.json"
done

for step in 8 16; do
  if [[ "$step" == 8 ]]; then
    gpu_ids=(0 1 2 3)
  else
    gpu_ids=(4 5 6 7)
  fi
  arm_root="$eval_root/step${step}"
  mkdir -p "$arm_root"
  "$python_bin" "$eval_repo/tools/materialize_full_model_policy_benchmark_config.py" \
    --evaluation-id "PRL21-R0-CROP-TFREE-COREDEV2511-STEP${step}-TEMP1-SEED42-V1" \
    --policy-config "$contract" \
    --snapshot-manifest "$eval_root/runtime/step${step}-snapshot.json" \
    --materialization-receipt "$eval_root/runtime/step${step}-receipt.json" \
    --expected-optimizer-step "$step" \
    --tasks "$task_manifest" \
    --expected-task-count 2511 \
    --expected-single-image-count 2240 \
    --output-root "$arm_root" \
    --config-output "$arm_root/config.json" \
    --inference-concurrency-per-gpu 8 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --no-enable-chunked-prefill \
    --gpu-memory-utilization 0.8 \
    --gpu-ids "${gpu_ids[@]}" \
    > "$eval_root/logs/config-step${step}.json"
  "$python_bin" "$eval_repo/tools/run_policy_benchmark.py" \
    --config "$arm_root/config.json" --mode prepare \
    > "$eval_root/logs/prepare-step${step}.json"
  "$python_bin" "$eval_repo/tools/run_policy_benchmark.py" \
    --config "$arm_root/config.json" --mode validate --world-size 4 \
    > "$eval_root/logs/validate-step${step}.json"
done

run_inference_arm() {
  local step=$1
  local gpu_base=$2
  local arm_root="$eval_root/step${step}"
  local pids=()
  for rank in 0 1 2 3; do
    local gpu=$((gpu_base + rank))
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      "$python_bin" "$eval_repo/tools/run_policy_benchmark.py" \
      --config "$arm_root/config.json" --mode worker \
      --rank "$rank" --world-size 4 \
      > "$eval_root/logs/step${step}-rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" != 0 ]]; then
    echo "step${step} inference worker failed"
    return 1
  fi
  "$python_bin" "$eval_repo/tools/run_policy_benchmark.py" \
    --config "$arm_root/config.json" --mode status --world-size 4 \
    > "$eval_root/logs/status-step${step}.json"
}

run_inference_arm 8 0 &
step8_pid=$!
run_inference_arm 16 4 &
step16_pid=$!
wait "$step8_pid"
wait "$step16_pid"

fi

materialize_scoring_views
for step in 8 16; do
  validate_scoring_inputs "$step"
done

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
  VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  "$vllm_bin" serve /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct \
  --served-model-name Qwen2.5-72B-Instruct \
  --host 127.0.0.1 --port 8012 \
  --dtype bfloat16 --tensor-parallel-size 2 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 --generation-config vllm --enable-prefix-caching \
  > "$eval_root/logs/judge-qwen25-72b.log" 2>&1 &
judge_pid=$!

for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "$judge_url/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$judge_pid" 2>/dev/null; then
    echo "Qwen2.5-72B judge exited during startup"
    exit 1
  fi
  sleep 5
done
curl -fsS --max-time 5 "$judge_url/models" >/dev/null

datasets=(VStarBench HRBench4K BLINK OCRBench_v2 MMMU_Pro_10c MathVista_MINI MathVerse_MINI)
for step in 8 16; do
  score_pids=()
  for dataset in "${datasets[@]}"; do
    work_dir="$eval_root/step${step}/scoring/$scoring_view/$dataset"
    env OPENAI_API_KEY=EMPTY \
      "$python_bin" "$eval_repo/tools/run_coredev_2511_vlmevalkit.py" \
      --data "$dataset" --model Qwen3-VL-8B-Instruct \
      --work-dir "$work_dir" --mode eval --reuse --reuse-aux infer \
      --judge Qwen2.5-72B-Instruct --judge-base-url "$judge_url" \
      --judge-key EMPTY --judge-api-nproc 4 --judge-retry 6 \
      --judge-timeout 600 \
      > "$eval_root/logs/score-step${step}-${dataset}.log" 2>&1 &
    score_pids+=("$!")
  done
  failed=0
  for pid in "${score_pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" != 0 ]]; then
    echo "step${step} scoring failed"
    exit 1
  fi
done

touch "$eval_root/evaluation-complete"
echo "PRL21 Step8/16 unified CoreDev inference and scoring complete; Step0 reuses the protocol-identical clean Crop baseline."
