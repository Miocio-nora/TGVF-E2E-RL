#!/bin/bash -p
set -euo pipefail

script_source=${BASH_SOURCE[0]}
readonly script_source
if [[ -L "$script_source" ]]; then
  builtin printf '%s\n' "refusing symlinked legacy shell controller: $script_source" >&2
  exit 3
fi
if [[ "$script_source" == /* ]]; then
  script_parent=${script_source%/*}
elif [[ "$script_source" == */* ]]; then
  script_parent=./${script_source%/*}
else
  script_parent=.
fi
readonly script_parent
builtin cd -P -- "$script_parent"
script_directory=$PWD
readonly script_directory
exec /usr/bin/python3 -I -- "$script_directory/check_launch_gate.py" quarantine-legacy \
  --tool-id "tools/launch_policy_data_selection_t1_rank.sh"
