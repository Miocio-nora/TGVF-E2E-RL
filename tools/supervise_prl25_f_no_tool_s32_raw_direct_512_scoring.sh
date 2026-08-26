#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
eval_root="$main_root/artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-RAW-DIRECT-512-S32-V1"
inference_control="$eval_root/runtime/inference-supervisor"
control_root="$eval_root/runtime/scoring-supervisor"
log_root="$eval_root/logs"
judge_port=8012
judge_base_url="http://127.0.0.1:${judge_port}/v1"

datasets=(
  VStarBench
  HRBench4K
  BLINK
  OCRBench_v2
  MMMU_Pro_10c
  MathVista_MINI
  MathVerse_MINI
)

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
flock -n 9 || { echo "raw-direct@512 scoring supervisor already active" >&2; exit 1; }

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
    rm -f "$control_root/failed"
  else
    printf 'status=failed\nphase=%s\ntime=%s\nexit_status=%s\n' \
      "$phase" "$(timestamp)" "$status" >"$control_root/failed"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'phase=signal; exit 130' INT TERM

wait_for_inference() {
  phase=waiting_for_inference
  while [[ ! -f "$inference_control/raw-direct-512-s32-inference-complete" ]]; do
    if ! tmux list-sessions -F '#S' 2>/dev/null \
      | grep -Fxq prl25-raw512-s32; then
      echo "inference supervisor exited without a completion marker" >&2
      return 1
    fi
    sleep 5
  done
}

capture_source_runs() {
  phase=capturing_source_runs
  : >"$control_root/source-runs.tsv"
  local dataset status source_run
  for dataset in "${datasets[@]}"; do
    status=$(find "$eval_root/inference/$dataset/work/Qwen3-VL-8B-Instruct" \
      -mindepth 2 -maxdepth 2 -type f -name status.json -printf '%T@ %p\n' \
      | sort -nr | head -1 | cut -d' ' -f2-)
    [[ -n "$status" ]] || { echo "$dataset has no inference status" >&2; return 1; }
    source_run=$("$python_bin" - "$status" "$dataset" <<'PY'
import json
from pathlib import Path
import sys

status_path = Path(sys.argv[1])
dataset = sys.argv[2]
payload = json.loads(status_path.read_text(encoding="utf-8"))
entry = payload.get("datasets", {}).get(dataset, {})
if payload.get("mode") != "infer" or entry.get("status") != "done":
    raise RuntimeError(f"{dataset} source inference is not complete")
print(payload["eval_id"])
PY
    )
    printf '%s\t%s\t%s\n' "$dataset" "$source_run" "$status" \
      >>"$control_root/source-runs.tsv"
  done
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
    exec setsid env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
      CC=/usr/bin/gcc CXX=/usr/bin/g++ \
      CPATH="$main_root/.deps/python312-dev/root/usr/include:$main_root/.deps/python312-dev/root/usr/include/python3.12" \
      VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_PLUGINS= \
      VLLM_ATTENTION_BACKEND=TRITON_ATTN TOKENIZERS_PARALLELISM=false \
      "$python_bin" -m vllm.entrypoints.openai.api_server \
        --model /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct \
        --served-model-name Qwen2.5-72B-Instruct \
        --host 127.0.0.1 --port "$judge_port" \
        --tensor-parallel-size 2 --dtype bfloat16 \
        --max-model-len 32768 --gpu-memory-utilization 0.85 \
        --max-num-seqs 64 --seed 42 --generation-config vllm \
        --enable-prefix-caching
  ) >"$log_root/qwen25-72b-judge-raw-direct-512.log" 2>&1 &
  judge_pid=$!
  local ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${judge_port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 -- "-$judge_pid" 2>/dev/null; then
      echo "Qwen2.5-72B judge exited during startup" >&2
      return 1
    fi
    sleep 5
  done
  [[ "$ready" == 1 ]] || { echo "Qwen2.5-72B judge readiness timeout" >&2; return 1; }
  curl -fsS "$judge_base_url/models" | "$python_bin" -c \
    'import json,sys; p=json.load(sys.stdin); assert any(x["id"]=="Qwen2.5-72B-Instruct" for x in p["data"])'
  touch "$control_root/judge-ready"
}

score_all_datasets() {
  phase=scoring_all_datasets
  local dataset work_dir cwd
  score_pids=()
  for dataset in "${datasets[@]}"; do
    work_dir="$eval_root/inference/$dataset/work"
    cwd="$eval_root/inference/$dataset/cwd"
    (
      cd "$cwd"
      exec setsid env \
        OPENAI_API_KEY=EMPTY CUDA_VISIBLE_DEVICES= VLLM_PLUGINS= \
        PYTHONPATH="$repo_root/src" PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_root/tools/run_coredev_2511_vlmevalkit.py" \
          --data "$dataset" \
          --model Qwen3-VL-8B-Instruct \
          --work-dir "$work_dir" \
          --mode eval --reuse --reuse-aux infer \
          --judge Qwen2.5-72B-Instruct \
          --judge-base-url "$judge_base_url" \
          --judge-key EMPTY --judge-api-nproc 4 --judge-retry 6 --judge-timeout 600
    ) >"$log_root/score-$dataset.log" 2>&1 &
    score_pids+=("$!")
  done
  local failed=0
  local pid
  for pid in "${score_pids[@]}"; do
    wait "$pid" || failed=1
  done
  score_pids=()
  (( failed == 0 )) || { echo "one or more raw-direct scorers failed" >&2; return 1; }
}

aggregate_summary() {
  phase=aggregating_summary
  PYTHONPATH="$repo_root/src" "$python_bin" - \
    "$eval_root" "$control_root/source-runs.tsv" "$judge_base_url" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from tgvf_rl.evaluation.coredev_results import (
    extract_coredev_macro_star,
    write_json_atomic,
)

root = Path(sys.argv[1]).resolve()
source_map_path = Path(sys.argv[2]).resolve()
judge_base_url = sys.argv[3]
datasets = (
    "VStarBench",
    "HRBench4K",
    "BLINK",
    "OCRBench_v2",
    "MMMU_Pro_10c",
    "MathVista_MINI",
    "MathVerse_MINI",
)
source_runs = {}
for line in source_map_path.read_text(encoding="utf-8").splitlines():
    dataset, run_id, _status_path = line.split("\t", maxsplit=2)
    source_runs[dataset] = run_id
if tuple(source_runs) != datasets:
    raise RuntimeError("raw-direct source-run map differs")

slices = []
for dataset in datasets:
    partial_path = root / f"inference/{dataset}/work/coredev-2511-eval-summary.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    if (
        partial.get("schema_version") != 1
        or partial.get("phase") != "eval"
        or partial.get("status") != "pass"
        or partial.get("model") != "Qwen3-VL-8B-Instruct"
        or partial.get("slice_count") != 1
        or len(partial.get("slices", [])) != 1
        or partial["slices"][0].get("dataset") != dataset
    ):
        raise RuntimeError(f"{dataset} partial summary differs")
    item = partial["slices"][0]
    status_path = Path(item["status_path"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    entry = status.get("datasets", {}).get(dataset, {})
    if entry.get("source_run") != source_runs[dataset]:
        raise RuntimeError(f"{dataset} scorer did not reuse the frozen inference run")
    slices.append(item)

summary = {
    "schema_version": 1,
    "suite": "coredev-2511-vlmevalkit-7055d301-v1",
    "phase": "eval",
    "status": "pass",
    "model": "Qwen3-VL-8B-Instruct",
    "evaluation_contract": "raw-direct@512-s32-v1",
    "max_pixels": 262144,
    "judge_base_url": judge_base_url,
    "vlmevalkit_commit": slices[0] and json.loads(
        (root / "inference/VStarBench/work/coredev-2511-eval-summary.json").read_text(
            encoding="utf-8"
        )
    )["vlmevalkit_commit"],
    "sample_count": sum(item["sample_count"] for item in slices),
    "slice_count": len(slices),
    "judge_parse_failure_policy": "deterministic_incorrect",
    "judge_parse_failure_rate_limit": 0.02,
    "judge_parse_failure_count": sum(
        item["judge_parse_failure_count"] for item in slices
    ),
    "slices": slices,
}
if summary["sample_count"] != 2511 or summary["slice_count"] != 7:
    raise RuntimeError("raw-direct aggregate coverage differs")
summary["headline"] = extract_coredev_macro_star(summary)
output = root / "scoring/coredev-2511-eval-summary.json"
write_json_atomic(output, summary)
print(json.dumps(summary["headline"], indent=2, ensure_ascii=False))
PY
}

rm -f "$control_root/raw-direct-512-s32-scoring-complete" "$control_root/failed"
printf '[%s] raw-direct@512 scoring supervisor started\n' "$(timestamp)"
wait_for_inference
capture_source_runs
wait_for_idle_gpu01
start_judge
score_all_datasets
aggregate_summary >"$log_root/raw-direct-512-headline.json"
phase=complete
printf 'status=pass\ntime=%s\nsample_count=2511\nslice_count=7\n' \
  "$(timestamp)" >"$control_root/raw-direct-512-s32-scoring-complete"
printf '[%s] raw-direct@512 scoring complete\n' "$(timestamp)"
