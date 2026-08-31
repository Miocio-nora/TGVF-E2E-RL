#!/usr/bin/env bash
# Fresh 80-step TGVF-shaped training on GPU0-3, followed automatically by the
# strict CoreDev-2511 ACC-VAL controller.  This script deliberately does not
# start the local visual judge: the bound Qwen3-VL-32B service must already be
# running on physical GPU0 under the colocated judge controller.
set -euo pipefail

script_pgid=$(ps -o pgid= -p "$$" | tr -d '[:space:]')
if [[ "$script_pgid" != "$$" ]]; then
  exec setsid /bin/bash "$0" "$@"
fi

repo_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
runtime_root=${TGVF_SHAPED_RUNTIME_ROOT:-/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl09-tgvf-shaped}
python_bin="$repo_root/.venv312/bin/python"
ray_bin="$repo_root/.venv312/bin/ray"
policy_config="$runtime_root/configs/policy/runs/prl_09_r2_qwen3_instruct_grpo_bs16_tgvf_shaped_t1mixed_v2_80step_gpu0123.toml"
policy_root="$repo_root/artifacts/policy/PRL-09-R2-qwen3-instruct-grpo-bs16-tgvf-shaped-t1mixed-v2-80step-gpu0123"
control_root="$repo_root/artifacts/policy-control/PRL-09-TGVF-SHAPED/R2-fresh80"
controller_log="$control_root/controller.log"
training_log="$control_root/training.log"
health_log="$control_root/early-step-health.jsonl"
health_failure="$control_root/early-step-monitor.failed.json"
training_complete="$control_root/training.complete"
controller_complete="$control_root/complete"
controller_failed="$control_root/failed"
controller_pid_marker="$control_root/controller.pid"
ray_temp=/tmp/prl09-r2-tgvf-shaped-80
ray_address=192.168.100.15:51519
visual_judge_base_url=http://127.0.0.1:8013/v1
visual_judge_model=Qwen3-VL-32B-Thinking
visual_judge_control="$repo_root/artifacts/policy-control/PRL-09-TGVF-SHAPED/local-judge-formal-gpu0-colocated"
visual_judge_process_record="$visual_judge_control/server-process.json"
visual_judge_ready="$visual_judge_control/ready"
visual_judge_complete="$visual_judge_control/complete"
visual_judge_failed="$visual_judge_control/failed"

mkdir -p "$control_root"
exec > >(tee -a "$controller_log") 2>&1

monitor_pid=""
training_group_pid=""
phase=initializing

timestamp() {
  date '+%F %T %Z'
}

scoped_ray_pids() {
  ps -eo pid=,args= | awk -v marker="$ray_temp" \
    'index($0, marker) && index($0, "awk -v marker") == 0 {print $1}'
}

stop_scoped_ray() {
  local -a pids=()
  mapfile -t pids < <(scoped_ray_pids)
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      mapfile -t pids < <(scoped_ray_pids)
      ((${#pids[@]} == 0)) && return 0
      sleep 1
    done
    kill -KILL "${pids[@]}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      mapfile -t pids < <(scoped_ray_pids)
      ((${#pids[@]} == 0)) && return 0
      sleep 1
    done
    return 1
  fi
}

group_live_count() {
  local pgid=$1
  ps -eo pgid=,stat= | awk -v expected="$pgid" \
    '$1 == expected && $2 !~ /^Z/ {count += 1} END {print count + 0}'
}

stop_training_group() {
  local pid=${training_group_pid:-} live_count actual_pgid process_state
  [[ -n "$pid" ]] || return 0
  actual_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')
  if [[ -n "$actual_pgid" && "$actual_pgid" != "$pid" ]]; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do
      process_state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')
      [[ -z "$process_state" || "$process_state" == Z* ]] && break
      sleep 0.1
    done
    if [[ -n "$process_state" && "$process_state" != Z* ]]; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    training_group_pid=""
    return 0
  fi
  live_count=$(group_live_count "$pid") || return 1
  if (( live_count > 0 )); then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      live_count=$(group_live_count "$pid") || return 1
      (( live_count == 0 )) && break
      sleep 1
    done
    if (( live_count > 0 )); then
      kill -KILL -- "-$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        live_count=$(group_live_count "$pid") || return 1
        (( live_count == 0 )) && break
        sleep 1
      done
    fi
  fi
  (( live_count == 0 )) || return 1
  wait "$pid" 2>/dev/null || true
  training_group_pid=""
}

cleanup() {
  local status=$?
  set +e
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if ! stop_training_group; then
    status=1
  fi
  if ! stop_scoped_ray; then
    status=1
  fi
  rm -f "$controller_pid_marker"
  if (( status == 0 )); then
    rm -f "$controller_failed"
  else
    printf 'status=failed\nphase=%s\ntime=%s\nexit_status=%s\nlog=%s\n' \
      "$phase" "$(timestamp)" "$status" "$controller_log" >"$controller_failed"
  fi
  printf 'controller_exit=%s status=%d phase=%s\n' "$(timestamp)" "$status" "$phase"
  exit "$status"
}
on_signal() {
  phase=signal
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

gpu0123_have_only_bound_colocated_judge() {
  local gpu pid actual_pgid judge_pgid active_output
  local -a active_pids=()
  [[ -s "$visual_judge_ready" ]] || return 1
  grep -q '^status=ready$' "$visual_judge_ready" || return 1
  grep -q '^physical_gpu=0$' "$visual_judge_ready" || return 1
  grep -q '^gpu_memory_utilization=0.39$' "$visual_judge_ready" || return 1
  [[ -s "$visual_judge_process_record" ]] || return 1
  judge_pgid=$("$python_bin" - "$visual_judge_process_record" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("physical_gpu") != 0:
    raise RuntimeError("colocated visual judge physical GPU differs")
if payload.get("config_file_sha256") != "d05a0e91554eaef7f700732d78d392b9ebec04be1012104af3a6d27d9e3be331":
    raise RuntimeError("colocated visual judge config identity differs")
pid = payload.get("pid")
pgid = payload.get("process_group")
if not isinstance(pid, int) or not isinstance(pgid, int) or pid != pgid:
    raise RuntimeError("colocated visual judge process identity differs")
command = payload.get("command")
if not isinstance(command, list):
    raise RuntimeError("colocated visual judge command is unavailable")
expected = {
    "--gpu-memory-utilization": "0.39",
    "--max-num-batched-tokens": "16384",
    "--max-num-seqs": "32",
    "--port": "8013",
}
for flag, value in expected.items():
    try:
        observed = command[command.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"colocated visual judge lacks {flag}") from error
    if observed != value:
        raise RuntimeError(f"colocated visual judge {flag} differs: {observed!r}")
print(pgid)
PY
  ) || return 1
  kill -0 "$judge_pgid" 2>/dev/null || return 1
  actual_pgid=$(ps -o pgid= -p "$judge_pgid" | tr -d '[:space:]')
  [[ "$actual_pgid" == "$judge_pgid" ]] || return 1
  for gpu in 0 1 2 3; do
    active_pids=()
    if ! active_output=$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null); then
      return 1
    fi
    active_output=$(printf '%s\n' "$active_output" | sed '/^[[:space:]]*$/d')
    if [[ -n "$active_output" ]]; then
      mapfile -t active_pids <<<"$active_output"
    fi
    if (( gpu == 0 )); then
      ((${#active_pids[@]} > 0)) || return 1
      for pid in "${active_pids[@]}"; do
        actual_pgid=$(ps -o pgid= -p "$pid" | tr -d '[:space:]')
        [[ "$actual_pgid" == "$judge_pgid" ]] || return 1
      done
    elif ((${#active_pids[@]} > 0)); then
      return 1
    fi
  done
}

preflight_visual_judge() {
  phase=preflighting_visual_judge
  curl -fsS "${visual_judge_base_url%/v1}/health" >/dev/null
  curl -fsS "$visual_judge_base_url/models" | "$python_bin" -c '
import json, sys
expected = sys.argv[1]
models = {row.get("id") for row in json.load(sys.stdin).get("data", [])}
if expected not in models:
    raise RuntimeError(f"local visual judge model differs: {sorted(models)!r}")
print(f"visual_judge=ready model={expected}")
' "$visual_judge_model"
}

start_early_step_monitor() {
  local training_pgid=$1
  phase=starting_early_step_monitor
  (
    exec "$python_bin" - \
      "$policy_root/metrics.jsonl" "$health_log" "$health_failure" "$$" \
      "$training_pgid" <<'PY'
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import sys
import time

metrics_path = Path(sys.argv[1])
health_path = Path(sys.argv[2])
failure_path = Path(sys.argv[3])
controller_pid = int(sys.argv[4])
training_pgid = int(sys.argv[5])
seen = 0
watch = {1, 2, 3, 5, 10, 20, 40, 60, 80}


def fail_closed(message: str, *, optimizer_step: object = None) -> None:
    payload = {
        "status": "failed",
        "reason": message,
        "optimizer_step": optimizer_step,
    }
    temporary = failure_path.with_name(f".{failure_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(failure_path)
    try:
        os.killpg(training_pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if os.getpgid(controller_pid) != controller_pid:
        raise RuntimeError("formal controller is not its recorded process-group leader")
    os.killpg(controller_pid, signal.SIGTERM)
    raise SystemExit(1)


def finite(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite metric: {value!r}")


while True:
    if metrics_path.exists():
        try:
            raw = metrics_path.read_text(encoding="utf-8")
            if raw and not raw.endswith("\n"):
                boundary = raw.rfind("\n")
                raw = raw[: boundary + 1] if boundary >= 0 else ""
            lines = [line for line in raw.splitlines() if line]
            rows = [json.loads(line) for line in lines]
        except Exception as error:
            fail_closed(f"metrics.jsonl could not be parsed: {type(error).__name__}: {error}")
        if len(lines) < seen:
            fail_closed("metrics.jsonl shrank during a fresh run")
        steps = [row.get("optimizer_step") for row in rows]
        if steps != list(range(1, len(rows) + 1)):
            fail_closed(f"optimizer-step metric sequence differs: {steps!r}")
        for row in rows[seen:]:
            step = row["optimizer_step"]
            try:
                finite(row)
            except RuntimeError as error:
                fail_closed(str(error), optimizer_step=step)
            step_metrics = row.get("step", {})
            cumulative = row.get("cumulative", {})
            applicable = step_metrics.get("stage3_quality_judge_applicable")
            covered = step_metrics.get("stage3_quality_judge_covered")
            failures = step_metrics.get("stage3_quality_judge_failures")
            calls = step_metrics.get("stage3_visual_judge_calls")
            degraded = (
                not isinstance(applicable, int)
                or applicable <= 0
                or not isinstance(covered, int)
                or covered != applicable
                or not isinstance(failures, int)
                or failures != 0
                or not isinstance(calls, int)
                or calls != applicable
            )
            if step in watch or degraded:
                record = {
                    "status": "degraded" if degraded else "healthy",
                    "optimizer_step": step,
                    "step_time_seconds": step_metrics.get("pre_publication_elapsed_seconds"),
                    "step_tool_call_attempt_rate": step_metrics.get("tool_call_attempt_rate"),
                    "step_mean_answer_reward": step_metrics.get("mean_answer_reward"),
                    "step_mean_tgvf_answer_component": step_metrics.get("mean_stage3_answer_reward"),
                    "step_mean_tgvf_tool_component": step_metrics.get("mean_stage3_tool_reward"),
                    "step_mean_tgvf_focus_component": step_metrics.get("mean_stage3_focus_reward"),
                    "step_mean_tgvf_grounding_component": step_metrics.get("mean_stage3_grounding_reward"),
                    "step_mean_tgvf_protocol_component": step_metrics.get("mean_stage3_protocol_reward"),
                    "visual_quality_applicable": applicable,
                    "visual_quality_covered": covered,
                    "visual_quality_failures": failures,
                    "visual_quality_coverage": step_metrics.get("stage3_quality_judge_coverage"),
                    "visual_judge_calls": calls,
                    "tool_call_attempt_rate": cumulative.get("tool_call_attempt_rate"),
                    "mean_answer_reward": cumulative.get("mean_answer_reward"),
                    "mean_conditional_tool_reward": cumulative.get("mean_conditional_tool_reward"),
                    "judge_calls": cumulative.get("judge_calls"),
                }
                with health_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
            if degraded:
                fail_closed(
                    "visual-quality judge coverage degraded",
                    optimizer_step=step,
                )
            if step == 80:
                raise SystemExit(0)
        seen = len(rows)
    time.sleep(15)
PY
  ) &
  monitor_pid=$!
}

rm -f "$training_complete" "$controller_complete" "$controller_failed" \
  "$health_log" "$health_failure" "$controller_pid_marker"
printf '%s\n' "$$" >"$controller_pid_marker"
printf '[%s] PRL-09 R2 fresh80 controller started\n' "$(timestamp)"
[[ -x "$python_bin" ]] || { echo "project Python is unavailable: $python_bin"; exit 1; }
[[ -x "$ray_bin" ]] || { echo "project Ray is unavailable: $ray_bin"; exit 1; }
[[ -d "$runtime_root/.git" ]] || { echo "clean runtime checkout is unavailable: $runtime_root"; exit 1; }
[[ -z "$(git -C "$runtime_root" status --porcelain --untracked-files=normal)" ]] || {
  echo "TGVF-shaped runtime checkout is dirty: $runtime_root"
  exit 1
}
[[ -s "$policy_config" ]] || { echo "finalized fresh80 config is unavailable: $policy_config"; exit 1; }
if rg -n '__FILL_[A-Z0-9_]+__' "$policy_config"; then
  echo "fresh80 config still contains template placeholders"
  exit 1
fi
[[ ! -e "$policy_root" ]] || { echo "fresh output already exists: $policy_root"; exit 1; }
[[ ! -e "$ray_temp" ]] || { echo "scoped Ray temp path already exists: $ray_temp"; exit 1; }
gpu0123_have_only_bound_colocated_judge || {
  echo "GPU0-3 do not contain exactly the bound GPU0 colocated visual judge"
  exit 1
}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  IFS= read -r key_record < <(tmux show-environment -g OPENROUTER_API_KEY)
  export OPENROUTER_API_KEY="${key_record#OPENROUTER_API_KEY=}"
  unset key_record
fi
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is unavailable"; exit 1; }

export PYTHONPATH="$runtime_root/src"
mapfile -t run_identity_fields < <(
  "$python_bin" - "$policy_config" "$runtime_root" <<'PY'
from pathlib import Path
import sys

from tgvf_rl.policy.launch import assert_policy_execution_identity
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

config = load_policy_e2e_smoke_run_config(sys.argv[1])
runtime = Path(sys.argv[2]).resolve()
assert_policy_execution_identity(config, repository_root=runtime)
if config.training.maximum_optimizer_steps != 80:
    raise RuntimeError("fresh run is not exactly 80 optimizer steps")
if config.scheduler.total_steps != 80:
    raise RuntimeError("fresh run scheduler is not exactly 80 steps")
if config.training.validation_before_training is not False:
    raise RuntimeError("validation_before_training must be false")
if config.training.validation_frequency != -1:
    raise RuntimeError("in-training validation must be disabled")
if config.training.resume_mode != "disable" or config.training.resume_from_path is not None:
    raise RuntimeError("PRL-09 R2 must be a fresh, non-resuming run")
if config.distributed.physical_gpu_ids != (0, 1, 2, 3):
    raise RuntimeError("fresh80 physical GPU identity differs")
if config.reward.visual_quality_judge_config_path is None:
    raise RuntimeError("TGVF-shaped visual-quality judge is not bound")
print(config.run_id)
print(config.identity_sha256)
print(config.code.commit)
PY
)
(( ${#run_identity_fields[@]} == 3 )) || { echo "config identity preflight returned an incomplete record"; exit 1; }
run_id=${run_identity_fields[0]}
run_identity=${run_identity_fields[1]}
expected_commit=${run_identity_fields[2]}
printf 'configured_code_commit=%s observed_runtime_head=%s\n' \
  "$expected_commit" "$(git -C "$runtime_root" rev-parse HEAD)"

preflight_visual_judge

export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_ADDRESS="$ray_address"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export VERL_FULL_DETERMINISM=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_BATCH_INVARIANT=0
export VLLM_PLUGINS=tgvf_qwen3_precomputed
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CPATH="$runtime_root/.deps/python312-dev/root/usr/include:$runtime_root/.deps/python312-dev/root/usr/include/python3.12"
export WANDB_MODE=online
export WANDB_RUN_GROUP=PRL-09-TGVF-SHAPED
export WANDB_NAME="$run_id"
export TGVF_POLICY_RUN_CONFIG_PATH="$policy_config"
export TGVF_POLICY_RUN_ID="$run_id"
export TGVF_POLICY_RUN_IDENTITY="$run_identity"
export TGVF_POLICY_RUN_IDENTITY_SHA256="$run_identity"
export TGVF_POLICY_SERVER_TIMEOUT_SECONDS=2400
export TGVF_POLICY_STATE_DIR="$policy_root/runtime-policy-state"

phase=validating_and_planning_fresh80
"$python_bin" -m tgvf_rl.cli validate-policy-config "$policy_config" >"$control_root/config-validation.json"
"$python_bin" -m tgvf_rl.cli plan-policy "$policy_config" --python "$python_bin" >"$control_root/launch-plan.json"

phase=starting_ray
"$ray_bin" start --head --node-ip-address=192.168.100.15 --port=51519 \
  --dashboard-host=127.0.0.1 --dashboard-port=8379 \
  --min-worker-port=55002 --max-worker-port=59999 --num-cpus=32 --num-gpus=4 \
  --object-store-memory=200000000000 --temp-dir="$ray_temp" \
  --disable-usage-stats >"$control_root/ray-start.log" 2>&1

phase=training_fresh80
printf 'training_launch=%s run_id=%s source_step=0 target_step=80\n' "$(timestamp)" "$run_id"
setsid /bin/bash -c '
set -euo pipefail
cd "$1"
"$2" -m tgvf_rl.cli run-policy "$3" --python "$2" 2>&1 | tee "$4"
' prl09-r2-training "$runtime_root" "$python_bin" "$policy_config" "$training_log" &
training_group_pid=$!

training_pgid=""
for _ in $(seq 1 50); do
  training_pgid=$(ps -o pgid= -p "$training_group_pid" 2>/dev/null | tr -d '[:space:]')
  [[ "$training_pgid" == "$training_group_pid" ]] && break
  kill -0 "$training_group_pid" 2>/dev/null || break
  sleep 0.1
done
if [[ "$training_pgid" != "$training_group_pid" ]]; then
  set +e
  kill -TERM "$training_group_pid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    kill -0 "$training_group_pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -KILL "$training_group_pid" 2>/dev/null || true
  wait "$training_group_pid"
  training_status=$?
  set -e
  training_group_pid=""
  echo "training process group failed to establish; child status=$training_status"
  (( training_status != 0 )) && exit "$training_status"
  exit 1
fi

start_early_step_monitor "$training_group_pid"
set +e
wait "$training_group_pid"
training_status=$?
set -e
training_group_pid=""
printf 'training_exit=%s status=%d\n' "$(timestamp)" "$training_status"

if [[ -n "$monitor_pid" ]]; then
  if (( training_status == 0 )); then
    phase=waiting_for_early_step_monitor_step80
    monitor_state=""
    for _ in $(seq 1 45); do
      monitor_state=$(ps -o stat= -p "$monitor_pid" 2>/dev/null | tr -d '[:space:]')
      if [[ -z "$monitor_state" || "$monitor_state" == Z* ]]; then
        break
      fi
      sleep 1
    done
    if [[ -n "$monitor_state" && "$monitor_state" != Z* ]]; then
      echo "early-step monitor did not consume optimizer step 80"
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
      monitor_pid=""
      exit 1
    fi
    set +e
    wait "$monitor_pid"
    monitor_status=$?
    set -e
    monitor_pid=""
    (( monitor_status == 0 )) || {
      echo "early-step monitor exited non-zero: $monitor_status"
      exit 1
    }
  else
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
fi
stop_scoped_ray
(( training_status == 0 )) || exit "$training_status"

phase=validating_early_step_health_ledger
"$python_bin" - "$health_log" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
expected = [1, 2, 3, 5, 10, 20, 40, 60, 80]
observed = [row.get("optimizer_step") for row in rows]
if observed != expected:
    raise RuntimeError(f"early-step health sequence differs: {observed!r}")
if any(row.get("status") != "healthy" for row in rows):
    raise RuntimeError(f"early-step health contains a degraded record: {rows!r}")
for row in rows:
    if row.get("visual_quality_applicable", 0) <= 0:
        raise RuntimeError(f"visual judge was not applicable: {row!r}")
    if row.get("visual_quality_covered") != row["visual_quality_applicable"]:
        raise RuntimeError(f"visual judge coverage differs: {row!r}")
    if row.get("visual_quality_failures") != 0:
        raise RuntimeError(f"visual judge failure observed: {row!r}")
    if row.get("visual_judge_calls") != row["visual_quality_applicable"]:
        raise RuntimeError(f"visual judge call count differs: {row!r}")
print(json.dumps({"status": "pass", "watched_optimizer_steps": observed}, sort_keys=True))
PY

phase=recording_successful_training_exit
printf 'status=pass\ntime=%s\nrun_id=%s\nrun_identity_sha256=%s\noptimizer_step=80\n' \
  "$(timestamp)" "$run_id" "$run_identity" >"$training_complete"

phase=waiting_for_colocated_visual_judge_shutdown
judge_stopped=0
for _ in $(seq 1 180); do
  if [[ -s "$visual_judge_failed" ]]; then
    echo "colocated visual judge wrapper failed while shutting down"
    exit 1
  fi
  if [[ -s "$visual_judge_complete" ]] && \
    grep -q '^status=stopped$' "$visual_judge_complete" && \
    grep -q '^reason=training_complete$' "$visual_judge_complete"; then
    judge_stopped=1
    break
  fi
  sleep 1
done
(( judge_stopped == 1 )) || {
  echo "colocated visual judge did not stop within 180 seconds"
  exit 1
}
gpu0123_idle=0
for _ in $(seq 1 120); do
  if gpu0123_active=$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null) && \
    test -z "$(printf '%s\n' "$gpu0123_active" | sed '/^[[:space:]]*$/d')"; then
    gpu0123_idle=1
    break
  fi
  sleep 1
done
(( gpu0123_idle == 1 )) || {
  echo "GPU0-3 did not become idle after training and judge shutdown"
  exit 1
}

phase=running_automatic_coredev_acc_val
"$runtime_root/tools/supervise_prl09_r2_tgvf_shaped_coredev.sh"

phase=complete
printf 'status=pass\ntime=%s\nrun_id=%s\ncoredev_marker=%s\n' \
  "$(timestamp)" "$run_id" \
  "$repo_root/artifacts/policy-control/PRL-09-TGVF-SHAPED/R2-fresh80/post80-coredev/complete" \
  >"$controller_complete"
printf '[%s] PRL-09 R2 fresh80 training and strict CoreDev ACC-VAL passed\n' "$(timestamp)"
