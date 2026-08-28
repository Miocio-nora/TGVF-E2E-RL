#!/usr/bin/env bash
set -euo pipefail

main_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
training_root="$main_root/artifacts/policy/PRL-25-F-qwen3-instruct-full-no-tool-rl-bs16-n16-tfree-teacher25-32step-ws8"
eval_root="$training_root/evaluation/PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S0-S8-S16-S32-TRUE1M-V2"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export PRL25_F_MATCHED_EVAL_ROOT="$eval_root"
export PRL25_F_MATCHED_EVALUATION_ID_PREFIX=PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S
export PRL25_F_MATCHED_EVALUATION_ID_SUFFIX=-TRUE1M-V2
export PRL25_F_MATCHED_REQUIRED_IMAGE_MAX_PIXELS=1003520
export PRL25_F_MATCHED_INFERENCE_FAILURE_MARKER="$eval_root/runtime/supervisor/failed.json"

exec "$repo_root/tools/supervise_prl25_f_no_tool_matched_scoring.sh"
