#!/usr/bin/env bash
set -euo pipefail

main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$main_root/.venv312/bin/python"
vllm_bin="$main_root/.venv312/bin/vllm"
eval_root="$main_root/artifacts/evaluation/PRL14-A-CoreDev2511-cleanfinal-step0-step8-step16-v1"
judge_url=http://127.0.0.1:8012/v1
judge_pid=

cleanup() {
  if [[ -n "$judge_pid" ]] && kill -0 "$judge_pid" 2>/dev/null; then
    kill "$judge_pid" 2>/dev/null || true
    wait "$judge_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
  VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  "$vllm_bin" serve /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct \
  --served-model-name Qwen2.5-72B-Instruct \
  --host 127.0.0.1 --port 8012 \
  --dtype bfloat16 --tensor-parallel-size 2 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 --generation-config vllm --enable-prefix-caching \
  > "$eval_root/logs/judge-qwen25-72b-resume.log" 2>&1 &
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
for step in 16; do
  score_pids=()
  for dataset in "${datasets[@]}"; do
    work_dir="$eval_root/step${step}/scoring/coredev-official-v2/$dataset"
    env OPENAI_API_KEY=EMPTY \
      "$python_bin" "$main_root/tools/run_coredev_2511_vlmevalkit.py" \
      --data "$dataset" --model Qwen3-VL-8B-Instruct \
      --config "$main_root/configs/evaluation/coredev_2511_qwen3_instruct_direct_prl04_v1.json" \
      --work-dir "$work_dir" --mode eval --reuse-aux infer \
      --judge Qwen2.5-72B-Instruct --judge-base-url "$judge_url" \
      --judge-key EMPTY --judge-api-nproc 4 --judge-retry 6 \
      --judge-timeout 600 \
      > "$eval_root/logs/score-resume-step${step}-${dataset}.log" 2>&1 &
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
echo "PRL14 step16 scoring complete; step8 is already scored and step0 uses the prior clean paired baseline."
