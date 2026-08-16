#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$repo_root/tools/supervise_prl23_tgvf_teacher_ratio_step16_and_eval.sh" teacher100 "$@"
