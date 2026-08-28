#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
eval_root=${PRL25_F_MATCHED_EVAL_ROOT:-"$main_root/artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-COREDEV2511-S0-S8-S16-S32-DUAL-V1"}
evaluation_id_prefix=${PRL25_F_MATCHED_EVALUATION_ID_PREFIX:-PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S}
evaluation_id_suffix=${PRL25_F_MATCHED_EVALUATION_ID_SUFFIX:--V1}
required_image_max_pixels=${PRL25_F_MATCHED_REQUIRED_IMAGE_MAX_PIXELS:-}
processor_proof_filename=${PRL25_F_MATCHED_PROCESSOR_PROOF_FILENAME:-true1m-processor-proof.json}
steps_text=${PRL25_F_MATCHED_STEPS:-"0 8 16 32"}
read -r -a steps <<<"$steps_text"
if (( ${#steps[@]} == 0 )); then
  echo "matched scoring requires at least one optimizer step" >&2
  exit 1
fi
for step in "${steps[@]}"; do
  [[ "$step" =~ ^[0-9]+$ ]] || {
    echo "invalid matched scoring optimizer step: $step" >&2
    exit 1
  }
done
inference_control="$eval_root/runtime/supervisor"
inference_failure_marker=${PRL25_F_MATCHED_INFERENCE_FAILURE_MARKER:-"$inference_control/failed"}
control_root="$eval_root/runtime/scoring-supervisor"
log_root="$eval_root/logs"
tasks="$main_root/artifacts/evaluation/CoreDev2511-official-visible-v1/tasks.jsonl"
source_root=/nvmesv/dredvpn009/datasets/benchmarks/coredev_2511_vlmevalkit_7055d301_v1
mathverse_source=/nvmesv/dredvpn009/datasets/benchmarks/mathverse/snapshot/testmini.json
judge_port=8012
judge_base_url="http://127.0.0.1:${judge_port}/v1"
complete_marker="$control_root/matched-scoring-complete"
complete_receipt="$control_root/matched-scoring-complete.json"
failure_marker="$control_root/failed"
formal_true1m_v2=0
if [[ "$evaluation_id_suffix" == "-TRUE1M-V2" && "$required_image_max_pixels" == "1003520" ]]; then
  formal_true1m_v2=1
fi

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
flock -n 9 || { echo "PRL25-F matched scoring supervisor already active" >&2; exit 1; }

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false

score_pids=()
judge_pid=""
phase=initializing

timestamp() {
  date '+%F %T %Z'
}

stop_process_group() {
  local pid=${1:-}
  [[ -n "$pid" ]] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "-$pid" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local status=$?
  set +e
  for pid in "${score_pids[@]:-}"; do
    stop_process_group "$pid"
  done
  stop_process_group "$judge_pid"
  if (( status == 0 )); then
    rm -f "$failure_marker"
  else
    rm -f "$complete_marker" "$complete_receipt"
    printf 'status=failed\nphase=%s\ntime=%s\nexit_status=%s\n' \
      "$phase" "$(timestamp)" "$status" >"$failure_marker"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'phase=signal; exit 130' INT TERM

wait_for_inference() {
  phase=waiting_for_matched_inference
  while [[ ! -f "$inference_control/matched-inference-complete" ]]; do
    if [[ -f "$inference_failure_marker" ]]; then
      echo "matched inference supervisor reported failure"
      return 1
    fi
    sleep 5
  done
}

validate_step_inference() {
  local step=$1
  "$python_bin" - "$eval_root" "$step" "$evaluation_id_prefix" \
    "$evaluation_id_suffix" "$required_image_max_pixels" \
    "$processor_proof_filename" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
step = int(sys.argv[2])
evaluation_id = f"{sys.argv[3]}{step}{sys.argv[4]}"
required_image_max_pixels = int(sys.argv[5]) if sys.argv[5] else None
processor_proof_filename = sys.argv[6]
arm = root / f"matched/step{step}"
config = json.loads((arm / "config.json").read_text(encoding="utf-8"))
if config.get("evaluation_id") != evaluation_id:
    raise RuntimeError(f"S{step} scoring source evaluation identity differs")
if required_image_max_pixels is not None:
    proof = json.loads((arm / "runtime" / processor_proof_filename).read_text(encoding="utf-8"))
    processor = proof.get("proof", {})
    represented = processor.get("synthetic_native_represented_pixel_area")
    visual_tokens = processor.get("synthetic_native_visual_token_count")
    if (
        config.get("evaluation_image_max_pixels") != required_image_max_pixels
        or processor.get("configured_image_max_pixels") != required_image_max_pixels
        or processor.get("synthetic_native_source_pixel_area") != 3_145_728
        or type(represented) is not int
        or represented <= 0
        or represented > required_image_max_pixels
        or type(visual_tokens) is not int
        or visual_tokens <= 0
        or processor.get("runtime_mm_processor_kwargs")
        != {"size": {"shortest_edge": 65_536, "longest_edge": required_image_max_pixels}}
        or processor.get("runtime_override_path")
        != "mm_processor_kwargs.size.longest_edge"
        or processor.get("vllm_012_shallow_hashable") is not True
        or processor.get("nested_images_kwargs_present") is not False
        or processor.get("max_pixels_kwarg_present") is not False
    ):
        raise RuntimeError(f"S{step} processor cap proof differs")
    if required_image_max_pixels == 1_003_520 and (
        represented != 995_328 or visual_tokens != 972
    ):
        raise RuntimeError(f"S{step} corrected true1M processor geometry differs")
    if required_image_max_pixels not in {262_144, 1_003_520}:
        raise RuntimeError(f"S{step} unsupported formal processor cap")
tasks = [json.loads(line) for line in (arm / "runtime/policy-benchmark-tasks.jsonl").read_text(encoding="utf-8").splitlines()]
single = {row["ordinal"] for row in tasks if len(row["image_paths"]) == 1}
if len(tasks) != 2511 or len(single) != 2240 or len(tasks) - len(single) != 271:
    raise RuntimeError(f"S{step} task coverage differs")
observed: set[int] = set()
for rank in range(4):
    rows = [json.loads(line) for line in (arm / f"inference/rank-{rank}.jsonl").read_text(encoding="utf-8").splitlines()]
    if any(row.get("evaluation_id") != evaluation_id for row in rows):
        raise RuntimeError(f"S{step} rank{rank} evaluation identity differs")
    ordinals = [row.get("ordinal") for row in rows]
    expected = {ordinal for ordinal in single if ordinal % 4 == rank}
    if len(ordinals) != len(expected) or set(ordinals) != expected:
        raise RuntimeError(f"S{step} rank{rank} coverage differs")
    observed.update(ordinals)
if observed != single:
    raise RuntimeError(f"S{step} single-image coverage differs")
print(json.dumps({"step": step, "supported": len(observed), "unsupported": 271}, sort_keys=True))
PY
}

run_id_for_step() {
  local step=$1
  local override_name="PRL25_F_MATCHED_RUN_ID_S${step}"
  local override=${!override_name:-}
  if [[ -n "$override" ]]; then
    echo "$override"
    return 0
  fi
  case "$step" in
    0) echo T20260826_G0 ;;
    8) echo T20260826_G8 ;;
    16) echo T20260826_G10 ;;
    32) echo T20260826_G20 ;;
    *) return 1 ;;
  esac
}

materialize_step() {
  local step=$1
  local score_root="$eval_root/matched/step${step}/scoring/coredev-official-v1"
  local summary="$score_root/materialization-summary.json"
  local run_id
  run_id=$(run_id_for_step "$step")
  if [[ ! -s "$summary" ]]; then
    if [[ -e "$score_root/raw" ]]; then
      echo "partial S${step} materialization exists without summary"
      return 1
    fi
    "$python_bin" "$repo_root/tools/materialize_policy_coredev_scoring.py" \
      --inference-root "$eval_root/matched/step${step}/inference" \
      --tasks "$tasks" \
      --source-root "$source_root" \
      --output-root "$score_root" \
      --evaluation-id "${evaluation_id_prefix}${step}${evaluation_id_suffix}" \
      --run-id "$run_id" \
      --mathverse-source-json "$mathverse_source" \
      >"$log_root/materialization-s${step}.log"
  fi
  "$python_bin" - "$summary" "$step" "$run_id" \
    "$evaluation_id_prefix" "$evaluation_id_suffix" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = int(sys.argv[2])
expected_run_id = sys.argv[3]
expected_evaluation_id = f"{sys.argv[4]}{step}{sys.argv[5]}"
expected = {
    "evaluation_id": expected_evaluation_id,
    "run_id": expected_run_id,
    "observed_single_image_count": 2240,
    "unsupported_multi_image_count": 271,
    "official_row_count": 2511,
}
for field, value in expected.items():
    if payload.get(field) != value:
        raise RuntimeError(f"S{step} materialization {field} differs")
if len(payload.get("slices", [])) != 7:
    raise RuntimeError(f"S{step} materialization lacks seven slices")
print(json.dumps({"step": step, "materialization": "pass"}, sort_keys=True))
PY
  touch "$control_root/s${step}-materialized"
}

gpu01_are_idle() {
  local gpu active
  for gpu in 0 1; do
    active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null) || return 1
    active=$(printf '%s\n' "$active" | tr -d '[:space:]')
    [[ -z "$active" ]] || return 1
  done
}

wait_for_idle_gpu01() {
  phase=waiting_for_idle_gpu01
  local consecutive=0
  while (( consecutive < 3 )); do
    if gpu01_are_idle; then
      ((consecutive += 1))
    else
      consecutive=0
    fi
    (( consecutive >= 3 )) || sleep 10
  done
}

start_judge() {
  phase=starting_qwen25_72b_judge
  "$python_bin" - "$judge_port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
PY
  (
    cd "$main_root"
    exec setsid "$python_bin" \
      "$repo_root/tools/exec_with_controlled_toolchain.py" \
      --python-environment-root "$main_root/.venv312" \
      --python-header-root "$main_root/.deps/python312-dev/root/usr/include" \
      --environment CUDA_DEVICE_ORDER=PCI_BUS_ID \
      --environment CUDA_VISIBLE_DEVICES=0,1 \
      --environment VLLM_USE_V1=1 \
      --environment VLLM_WORKER_MULTIPROC_METHOD=spawn \
      --environment VLLM_PLUGINS= \
      --environment VLLM_ATTENTION_BACKEND=TRITON_ATTN \
      --environment TOKENIZERS_PARALLELISM=false \
      --contract-out "$control_root/judge-toolchain-contract.json" \
      -- "$python_bin" -m vllm.entrypoints.openai.api_server \
        --model /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct \
        --served-model-name Qwen2.5-72B-Instruct \
        --host 127.0.0.1 --port "$judge_port" \
        --tensor-parallel-size 2 --dtype bfloat16 \
        --max-model-len 32768 --gpu-memory-utilization 0.85 \
        --max-num-seqs 64 --seed 42 --generation-config vllm \
        --enable-prefix-caching
  ) >"$log_root/qwen25-72b-judge-matched.log" 2>&1 &
  judge_pid=$!
  local ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${judge_port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 -- "-$judge_pid" 2>/dev/null; then
      echo "Qwen2.5-72B judge exited during startup"
      return 1
    fi
    sleep 5
  done
  [[ "$ready" == 1 ]] || { echo "Qwen2.5-72B judge readiness timeout"; return 1; }
  curl -fsS "$judge_base_url/models" | "$python_bin" -c \
    'import json,sys; p=json.load(sys.stdin); assert any(x["id"]=="Qwen2.5-72B-Instruct" for x in p["data"])'
  touch "$control_root/judge-ready"
}

score_all_steps() {
  phase=scoring_all_steps
  local datasets=(VStarBench HRBench4K BLINK OCRBench_v2 MMMU_Pro_10c MathVista_MINI MathVerse_MINI)
  local step dataset score_root source_run_id source_manifest receipt
  score_pids=()
  for step in "${steps[@]}"; do
    score_root="$eval_root/matched/step${step}/scoring/coredev-official-v1"
    source_run_id=$(run_id_for_step "$step")
    for dataset in "${datasets[@]}"; do
      source_manifest="$score_root/$dataset/Qwen3-VL-8B-Instruct/$source_run_id/final-answer-view-manifest.json"
      receipt="$score_root/$dataset/pinned-reuse-receipt.json"
      if [[ -s "$receipt" ]]; then
        printf '[%s] reuse completed S%s/%s receipt\n' "$(timestamp)" "$step" "$dataset"
        continue
      fi
      (
        # OCRBench creates evaluator-local temporary ground-truth files under
        # the current directory.  A per-step/per-dataset cwd prevents the four
        # matched arms from racing over one shared .vlmeval tree.
        cd "$score_root/$dataset"
        exec setsid env OPENAI_API_KEY=EMPTY CUDA_VISIBLE_DEVICES= VLLM_PLUGINS= \
          PYTHONPATH="$repo_root/src" PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false \
          "$python_bin" "$repo_root/tools/run_coredev_2511_vlmevalkit.py" \
            --data "$dataset" \
            --model Qwen3-VL-8B-Instruct \
            --work-dir "$score_root/$dataset" \
            --mode eval --reuse --reuse-aux infer \
            --tgvf-reuse-source-run-id "$source_run_id" \
            --tgvf-reuse-manifest "$source_manifest" \
            --judge Qwen2.5-72B-Instruct \
            --judge-base-url "$judge_base_url" \
            --judge-key EMPTY --judge-api-nproc 4 --judge-retry 6 --judge-timeout 600
      ) >"$log_root/score-s${step}-${dataset}.log" 2>&1 &
      score_pids+=("$!")
    done
  done
  local failed=0
  local pid
  for pid in "${score_pids[@]}"; do
    wait "$pid" || failed=1
  done
  score_pids=()
  (( failed == 0 )) || { echo "one or more CoreDev scoring slices failed"; return 1; }
}

all_scoring_receipts_present() {
  local datasets=(VStarBench HRBench4K BLINK OCRBench_v2 MMMU_Pro_10c MathVista_MINI MathVerse_MINI)
  local step dataset score_root
  for step in "${steps[@]}"; do
    score_root="$eval_root/matched/step${step}/scoring/coredev-official-v1"
    for dataset in "${datasets[@]}"; do
      [[ -s "$score_root/$dataset/pinned-reuse-receipt.json" ]] || return 1
    done
  done
}

summarize_step() {
  local step=$1
  local score_root="$eval_root/matched/step${step}/scoring/coredev-official-v1"
  local summary="$score_root/coredev-2511-eval-summary.json"
  PYTHONPATH="$repo_root/src" "$python_bin" - \
    "$score_root" "$summary" "$judge_base_url" "$step" \
    "$evaluation_id_prefix" "$evaluation_id_suffix" \
    >"$log_root/summary-s${step}.log" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from tgvf_rl.evaluation.coredev_results import (
    summarize_coredev_results,
    write_json_atomic,
)
from tgvf_rl.evaluation.policy_coredev_scoring import DATASETS

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
judge_base_url = sys.argv[3]
step = int(sys.argv[4])
evaluation_id_prefix = sys.argv[5]
evaluation_id_suffix = sys.argv[6]
expected_eval_ids: dict[str, str] = {}
for dataset in DATASETS:
    receipt_path = root / dataset / "pinned-reuse-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_source = f"{evaluation_id_prefix}{step}{evaluation_id_suffix}"
    if (
        receipt.get("schema_version")
        != "tgvf.vlmevalkit-pinned-reuse-receipt.v1"
        or receipt.get("dataset") != dataset
        or receipt.get("model") != "Qwen3-VL-8B-Instruct"
        or receipt.get("source_evaluation_id") != expected_source
    ):
        raise RuntimeError(f"S{step}/{dataset} pinned receipt identity differs")
    destination = receipt.get("destination_run_id")
    if not isinstance(destination, str) or not destination:
        raise RuntimeError(f"S{step}/{dataset} receipt has no destination run")
    expected_eval_ids[dataset] = destination

payload = summarize_coredev_results(
    work_dir=root,
    repository_root=Path.cwd(),
    phase="eval",
    expected_judge_base_url=judge_base_url,
    expected_model="Qwen3-VL-8B-Instruct",
    expected_eval_ids=expected_eval_ids,
)
write_json_atomic(output, payload)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
  if (( formal_true1m_v2 == 1 )); then
    PYTHONPATH="$repo_root/src" "$python_bin" \
      "$repo_root/tools/finalize_prl25_f_no_tool_true1m_v2_scoring.py" \
      --eval-root "$eval_root" --step "$step" \
      >"$log_root/headline-s${step}.log"
  else
  "$python_bin" - "$summary" "$step" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = int(sys.argv[2])
if payload.get("schema_version") != 1 or payload.get("phase") != "eval":
    raise RuntimeError(f"S{step} summary contract differs")
if payload.get("model") != "Qwen3-VL-8B-Instruct":
    raise RuntimeError(f"S{step} summary model differs")
if payload.get("sample_count") != 2511 or payload.get("slice_count") != 7:
    raise RuntimeError(f"S{step} summary completion differs")
if len(payload.get("slices", [])) != 7:
    raise RuntimeError(f"S{step} summary lacks seven slices")
print(json.dumps({"step": step, "summary": "pass"}, sort_keys=True))
PY
  fi
  touch "$control_root/s${step}-scoring-complete"
}

rm -f "$complete_marker" "$complete_receipt" "$failure_marker"
printf '[%s] PRL25-F matched scoring supervisor started\n' "$(timestamp)"
wait_for_inference
if (( formal_true1m_v2 == 1 )); then
  "$python_bin" \
    "$repo_root/tools/supervise_prl25_f_no_tool_true1m_v2_inference.py" \
    --verify-completion-only \
    >"$log_root/formal-inference-completion-validation.json"
fi
for step in "${steps[@]}"; do
  validate_step_inference "$step" >"$log_root/scoring-inference-validation-s${step}.json"
  materialize_step "$step"
done
if all_scoring_receipts_present; then
  printf '[%s] all 28 pinned scoring receipts already exist; skip judge restart\n' "$(timestamp)"
else
  wait_for_idle_gpu01
  start_judge
  score_all_steps
fi
phase=summarizing
for step in "${steps[@]}"; do
  summarize_step "$step"
done
if (( formal_true1m_v2 == 1 )); then
  PYTHONPATH="$repo_root/src" "$python_bin" \
    "$repo_root/tools/finalize_prl25_f_no_tool_true1m_v2_scoring.py" \
    --eval-root "$eval_root" --finalize-all \
    >"$log_root/headline-all-steps.log"
fi
phase=complete
steps_csv=$(IFS=,; echo "${steps[*]}")
printf 'status=pass\ntime=%s\nsteps=%s\nslice_count_per_step=7\nsample_count_per_step=2511\n' \
  "$(timestamp)" "$steps_csv" >"$complete_marker"
printf '[%s] PRL25-F matched scoring complete\n' "$(timestamp)"
