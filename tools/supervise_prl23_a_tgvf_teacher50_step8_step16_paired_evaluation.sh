#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$repo_root/tools/supervise_prl23_tgvf_teacher_ratio_step8_step16_paired_evaluation.sh" teacher50 "$@"
