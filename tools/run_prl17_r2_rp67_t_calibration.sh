#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
schedule_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_rl/PRL17-RP66-MATCHED-BS16-8STEP-UTILITY-SCHEDULE-v1
forced_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_rl/PRL17-RP67-CALIBRATED-BS16-8STEP-FORCED-TGVF-v1
sidecar_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_rl/PRL17-RP67-CALIBRATED-BS16-8STEP-UTILITY-SIDECAR-v1
run_id=PRL17-RP67-CALIBRATED-BS16-8STEP-FORCED-TGVF-v1
source_config="$repo_root/configs/representation/experiments/image_axis_grounding/evaluation/rp67_step2000_full867_gpu0.toml"
judge_config="$repo_root/configs/policy/judges/openrouter_qwen25_72b_formal_pilot_judge_v4.json"
log_root="$forced_root/logs"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for semantic fallback scoring" >&2
  exit 1
fi

mkdir -p "$log_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

pids=()
for shard_index in 0 1 2 3; do
  "$python_bin" -u "$repo_root/tools/run_forced_tgvf_counterfactual.py" run-shard \
    --schedule-root "$schedule_root" \
    --output-root "$forced_root" \
    --run-id "$run_id" \
    --sample-count 128 \
    --attempts-per-sample 8 \
    --shard-count 4 \
    --source-evaluation-config "$source_config" \
    --judge-config "$judge_config" \
    --master-seed 42 \
    --max-new-tokens 40960 \
    --eos-token-id 151645 \
    --eos-token-id 151643 \
    --shard-index "$shard_index" \
    --physical-gpu-id "$shard_index" \
    >"$log_root/shard-$(printf '%02d' "$shard_index").log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "one or more RP67 calibration shards failed; ledgers remain resumable" >&2
  exit 1
fi

"$python_bin" "$repo_root/tools/run_forced_tgvf_counterfactual.py" finalize \
  --schedule-root "$schedule_root" \
  --output-root "$forced_root" \
  --run-id "$run_id" \
  --sample-count 128 \
  --attempts-per-sample 8 \
  --shard-count 4 \
  >"$log_root/finalize.log" 2>&1

run_identity_sha256=$(jq -er '.run_identity_sha256' "$forced_root/run-identity.json")
"$python_bin" "$repo_root/tools/materialize_tgvf_tool_utility_sidecar.py" aggregate \
  --schedule-root "$schedule_root" \
  --attempts "$forced_root/attempts.jsonl" \
  --output-root "$sidecar_root" \
  --run-id "$run_id" \
  --run-identity-sha256 "$run_identity_sha256" \
  --sample-count 128 \
  --attempts-per-sample 8 \
  --needed-threshold 0.25 \
  --unnecessary-threshold -0.25 \
  --confidence 0.5 \
  >"$log_root/aggregate.log" 2>&1

echo "RP67 calibration sidecar complete: $sidecar_root"
