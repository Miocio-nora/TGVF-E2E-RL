#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
eval_root="$main_root/artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/evaluation/PRL25-F-NO-TOOL-RL-RAW-DIRECT-512-S32-V1"
control_root="$eval_root/runtime/inference-supervisor"
log_root="$eval_root/logs"

datasets=(
  VStarBench
  HRBench4K
  BLINK
  OCRBench_v2
  MMMU_Pro_10c
  MathVista_MINI
  MathVerse_MINI
)
gpus=(0 1 2 3 4 5 6)
expected_rows=(191 200 420 600 300 300 500)
configs=(
  "$eval_root/inference/VStarBench/resolved-configs/coredev-subset-50c51df9d603df97.json"
  "$eval_root/inference/HRBench4K/resolved-configs/coredev-subset-1524f78f29187235.json"
  "$eval_root/inference/BLINK/resolved-configs/coredev-subset-03c98c5f2a041a2e.json"
  "$eval_root/inference/OCRBench_v2/resolved-configs/coredev-subset-5fbb6f2a6a9a9ae8.json"
  "$eval_root/inference/MMMU_Pro_10c/resolved-configs/coredev-subset-a079f7ce16405e18.json"
  "$eval_root/inference/MathVista_MINI/resolved-configs/coredev-subset-ba4b2b9f2dc391a4.json"
  "$eval_root/inference/MathVerse_MINI/resolved-configs/coredev-subset-d57c6f29481b8b03.json"
)

mkdir -p "$control_root" "$log_root"
exec 9>"$control_root/supervisor.lock"
flock -n 9 || { echo "raw-direct@512 inference supervisor already active" >&2; exit 1; }

pids=()
phase=initializing

timestamp() {
  date '+%F %T %Z'
}

stop_process_group() {
  local pid=${1:-}
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  set +e
  if (( status != 0 )); then
    for pid in "${pids[@]:-}"; do
      stop_process_group "$pid"
    done
    printf 'status=failed\nphase=%s\ntime=%s\nexit_status=%s\n' \
      "$phase" "$(timestamp)" "$status" >"$control_root/failed"
  else
    rm -f "$control_root/failed"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'phase=signal; exit 130' INT TERM

phase=validating_contract
"$python_bin" - "${configs[@]}" <<'PY'
import json
from pathlib import Path
import sys

expected_model = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8/"
    "evaluation/PRL25-F-NO-TOOL-RL-COREDEV2511-S0-S8-S16-S32-DUAL-V1/"
    "shared/step32/model"
).resolve()
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload["model"]["Qwen3-VL-8B-Instruct"]
    if Path(model["model_path"]).resolve() != expected_model:
        raise RuntimeError(f"unexpected model path in {path}")
    if model["max_pixels"] != 512 * 512:
        raise RuntimeError(f"raw-direct pixel cap differs in {path}")
    if model["request_seed_namespace"] != (
        "coredev-2511-qwen3-instruct-direct-prl04-comparable-v1"
    ):
        raise RuntimeError(f"raw-direct seed namespace differs in {path}")
    if len(payload["data"]) != 1:
        raise RuntimeError(f"expected one isolated dataset in {path}")
print("raw-direct@512 contract: pass")
PY

phase=checking_gpus
for gpu in "${gpus[@]}"; do
  active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
  active=$(printf '%s\n' "$active" | tr -d '[:space:]')
  [[ -z "$active" ]] || { echo "GPU $gpu is busy" >&2; exit 1; }
done

phase=launching
: >"$control_root/launch.tsv"
: >"$control_root/receipts.tsv"
for index in "${!datasets[@]}"; do
  dataset=${datasets[$index]}
  gpu=${gpus[$index]}
  config=${configs[$index]}
  work_dir="$eval_root/inference/$dataset/work"
  cwd="$eval_root/inference/$dataset/cwd"
  mkdir -p "$work_dir" "$cwd"
  (
    cd "$cwd"
    exec setsid env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
      PYTHONHASHSEED=42 \
      TOKENIZERS_PARALLELISM=false \
      VLLM_USE_V1=1 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
      "$python_bin" "$repo_root/tools/run_coredev_2511_vlmevalkit.py" \
        --config "$config" \
        --work-dir "$work_dir" \
        --mode infer
  ) >"$log_root/infer-$dataset.log" 2>&1 &
  pids+=("$!")
  printf '%s\t%s\t%s\n' "$dataset" "$gpu" "$!" >>"$control_root/launch.tsv"
done

phase=waiting
for index in "${!datasets[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "${datasets[$index]} inference failed" >&2
    exit 1
  fi
done
pids=()

phase=validating_outputs
for index in "${!datasets[@]}"; do
  dataset=${datasets[$index]}
  work_dir="$eval_root/inference/$dataset/work"
  status=$(find "$work_dir/Qwen3-VL-8B-Instruct" -mindepth 2 -maxdepth 2 -type f -name status.json -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  [[ -n "$status" ]] || { echo "$dataset has no status.json" >&2; exit 1; }
  prediction=$("$python_bin" - "$status" "$dataset" <<'PY'
import json
from pathlib import Path
import sys

status_path = Path(sys.argv[1])
dataset = sys.argv[2]
payload = json.loads(status_path.read_text(encoding="utf-8"))
record = payload.get("datasets", {}).get(dataset, {})
if payload.get("mode") != "infer" or record.get("status") != "done":
    raise RuntimeError(f"{dataset} inference status is not done")
prediction = Path(record["prediction_file"])
if not prediction.is_absolute():
    candidate = status_path.parent / prediction.name
    prediction = candidate if candidate.exists() else prediction.resolve()
print(prediction)
PY
  )
  [[ -f "$prediction" ]] || { echo "$dataset prediction is missing: $prediction" >&2; exit 1; }
  rows=$(( $(wc -l <"$prediction") - 1 ))
  [[ "$rows" -eq "${expected_rows[$index]}" ]] || {
    echo "$dataset row count differs: $rows" >&2
    exit 1
  }
  printf '%s\t%s\t%s\t%s\n' "$dataset" "$rows" "$status" "$prediction" >>"$control_root/receipts.tsv"
done

phase=complete
printf 'status=complete\ntime=%s\n' "$(timestamp)" >"$control_root/raw-direct-512-s32-inference-complete"
