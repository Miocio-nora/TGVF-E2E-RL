#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python
source_session=${PRL22_A_SOURCE_TMUX_SESSION:-prl22_a_tgvf_teacher25}
source_training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-22-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-8step-ws8
source_evaluation_id=PRL22-A-FROZEN-RP67-TFREE-TEACHER25-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1
source_evaluation_root="$source_training_root/evaluation/$source_evaluation_id"
source_receipt="$source_evaluation_root/evaluation-complete"
target_training_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/PRL-22-B-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-8step-ws8
control_root="$target_training_root/runtime/relay-after-prl22-a"
target_supervisor="$repo_root/tools/supervise_prl22_b_crop_tgvf_teacher25_step16_and_eval.sh"
poll_seconds=${PRL22_B_RELAY_POLL_SECONDS:-30}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required before arming the PRL22-B relay" >&2
  exit 1
fi
if [[ ! -x "$target_supervisor" ]]; then
  echo "PRL22-B target supervisor is absent or not executable: $target_supervisor" >&2
  exit 1
fi

mkdir -p "$control_root"
exec 9>"$control_root/relay.lock"
if ! flock -n 9; then
  echo "another PRL22-A to PRL22-B relay is active" >&2
  exit 1
fi
touch "$control_root/relay-armed"

source_is_running() {
  tmux has-session -t "$source_session" 2>/dev/null \
    && [[ "$(tmux display-message -p -t "$source_session" '#{pane_dead}')" != "1" ]]
}

validate_source_receipt() {
  "$python_bin" - "$source_receipt" "$source_evaluation_root" "$source_evaluation_id" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

receipt_path = Path(sys.argv[1]).resolve()
evaluation_root = Path(sys.argv[2]).resolve()
expected_evaluation_id = sys.argv[3]
try:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid source evaluation receipt: {exc}")
if receipt.get("schema_version") != "tgvf.paired-coredev-evaluation-complete.v1":
    raise SystemExit("source evaluation receipt schema differs")
if receipt.get("status") != "complete":
    raise SystemExit("source evaluation receipt status differs")
if receipt.get("evaluation_id") != expected_evaluation_id:
    raise SystemExit("source evaluation receipt identity differs")
try:
    paired_summary = Path(receipt["paired_summary_path"]).resolve(strict=True)
    expected_sha256 = receipt["paired_summary_sha256"]
except (KeyError, OSError) as exc:
    raise SystemExit(f"source paired summary binding is invalid: {exc}")
if paired_summary.parent != evaluation_root:
    raise SystemExit("source paired summary escaped the evaluation root")
observed_sha256 = hashlib.sha256(paired_summary.read_bytes()).hexdigest()
if observed_sha256 != expected_sha256:
    raise SystemExit("source paired summary SHA256 differs")
PY
}

# Wait for the canonical, content-addressed completion receipt. A dead source
# without that receipt is a failed experiment, not a handoff signal.
while [[ ! -s "$source_receipt" ]]; do
  if ! source_is_running; then
    echo "PRL22-A stopped without a canonical successful evaluation receipt" >&2
    touch "$control_root/source-failed"
    exit 1
  fi
  sleep "$poll_seconds"
done
validate_source_receipt
touch "$control_root/source-evaluation-accepted"

# The receipt is written atomically before the A supervisor exits. Wait for its
# process boundary as well so B cannot race live evaluation workers for CUDA.
while source_is_running; do sleep "$poll_seconds"; done

# Require two consecutive GPU-quiescent polls. If another legitimate task owns
# a GPU, wait instead of oversubscribing or killing it.
quiet_polls=0
while (( quiet_polls < 2 )); do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | rg -q '^[[:space:]]*[0-9]+'; then
    quiet_polls=0
  else
    quiet_polls=$((quiet_polls + 1))
  fi
  if (( quiet_polls < 2 )); then sleep "$poll_seconds"; fi
done
touch "$control_root/gpus-released"
touch "$control_root/target-starting"
exec "$target_supervisor"
