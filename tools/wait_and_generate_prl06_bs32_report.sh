#!/usr/bin/env bash
set -euo pipefail

repo_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
python_bin="$repo_root/.venv312/bin/python"
control_root="$repo_root/artifacts/policy-control/PRL-06-R0-bs32-20step"
complete_marker="$control_root/coredev-step10-step20.complete"
failure_marker="$control_root/coredev-step10-step20.failed"
wait_log="$control_root/result-report-wait.log"
wait_seconds=${TGVF_PRL06_REPORT_WAIT_SECONDS:-43200}
started=$SECONDS

mkdir -p "$control_root"
exec > >(tee -a "$wait_log") 2>&1
date '+report_wait_start=%F %T %Z timeout_seconds='"$wait_seconds"

while [[ ! -s "$complete_marker" ]]; do
  if [[ -s "$failure_marker" ]]; then
    echo "PRL-06 evaluation supervisor failed:"
    cat "$failure_marker"
    exit 1
  fi
  if (( SECONDS - started >= wait_seconds )); then
    echo "timed out waiting for $complete_marker"
    exit 1
  fi
  sleep 15
done

if [[ -s "$failure_marker" ]]; then
  echo "PRL-06 has both completion and failure markers; refusing ambiguous state"
  cat "$failure_marker"
  exit 1
fi
grep -Fxq 'status=pass' "$complete_marker"
date '+strict_evaluations_complete=%F %T %Z'

"$python_bin" "$repo_root/tools/generate_prl06_bs32_report.py" \
  --repo-root "$repo_root" \
  --output-json "$control_root/prl06-bs32-result.json" \
  --output-markdown "$control_root/prl06-bs32-result.md"

date '+result_report_complete=%F %T %Z'
