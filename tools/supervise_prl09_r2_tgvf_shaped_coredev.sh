#!/usr/bin/env bash
# Strict post-training CoreDev-2511 ACC-VAL for PRL-09 R2.  It can be called by
# the fresh80 controller or started independently before training completes.
set -euo pipefail

repo_root=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl
runtime_root=${TGVF_SHAPED_RUNTIME_ROOT:-/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl09-tgvf-shaped}
python_bin="$repo_root/.venv312/bin/python"
policy_config="$runtime_root/configs/policy/runs/prl_09_r2_qwen3_instruct_grpo_bs16_tgvf_shaped_t1mixed_v2_80step_gpu0123.toml"
eval_config="$runtime_root/configs/evaluation/coredev_2511_tgvf_shaped_prl09_r2_step80_gpu0123.json"
policy_root="$repo_root/artifacts/policy/PRL-09-R2-qwen3-instruct-grpo-bs16-tgvf-shaped-t1mixed-v2-80step-gpu0123"
state_root="$policy_root/runtime-policy-state"
checkpoint_root="$policy_root/checkpoints"
fresh_control_root="$repo_root/artifacts/policy-control/PRL-09-TGVF-SHAPED/R2-fresh80"
training_controller_log="$fresh_control_root/controller.log"
training_complete="$fresh_control_root/training.complete"
control_root="$fresh_control_root/post80-coredev"
supervisor_log="$control_root/supervisor.log"
complete_marker="$control_root/complete"
failure_marker="$control_root/failed"
eval_root="$repo_root/artifacts/evaluation/PRL-09-R2-tgvf-shaped-t1mixed-v2-step80-coredev2511-gpu0123"
evaluation_id=PRL-09-R2-TGVF-SHAPED-T1MIXED-V2-STEP80-COREDEV2511-GPU0123
score_root="$eval_root/scoring/tgvf-shaped-auto-v1"
summary_path="$score_root/coredev-2511-eval-summary.json"
judge_port=8012
judge_base_url="http://127.0.0.1:$judge_port/v1"

mkdir -p "$control_root" "$eval_root/logs" "$eval_root/inference"
exec > >(tee -a "$supervisor_log") 2>&1

worker_pids=()
score_pids=()
judge_pid=""
phase=initializing

timestamp() {
  date '+%F %T %Z'
}

stop_process_group() {
  local pid=${1:-}
  [[ -n "$pid" ]] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "-$pid" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local status=$?
  set +e
  for pid in "${worker_pids[@]:-}"; do
    stop_process_group "$pid"
  done
  for pid in "${score_pids[@]:-}"; do
    stop_process_group "$pid"
  done
  stop_process_group "$judge_pid"
  if (( status == 0 )); then
    rm -f "$failure_marker"
  else
    printf 'status=failed\nphase=%s\ntime=%s\nexit_status=%s\nlog=%s\n' \
      "$phase" "$(timestamp)" "$status" "$supervisor_log" >"$failure_marker"
  fi
  exit "$status"
}
on_signal() {
  phase=signal
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

basic_step80_boundary_present() {
  local latest="$checkpoint_root/latest_checkpointed_iteration.txt"
  [[ -s "$training_complete" ]] || return 1
  grep -q '^status=pass$' "$training_complete" || return 1
  grep -q '^optimizer_step=80$' "$training_complete" || return 1
  [[ -s "$latest" ]] || return 1
  [[ "$(tr -d '[:space:]' <"$latest")" == 80 ]] || return 1
  [[ -s "$policy_root/metrics.jsonl" ]] || return 1
  [[ -s "$checkpoint_root/global_step_80/actor/tgvf_policy_checkpoint_pair.json" ]] || return 1
  [[ -s "$checkpoint_root/global_step_80/actor/tgvf_policy_project_state.json" ]] || return 1
  [[ -s "$state_root/latest-lora-snapshot.json" ]] || return 1
  compgen -G "$state_root/lora-manifests/step-00000080-*.json" >/dev/null || return 1
}

training_reported_failure() {
  grep -Eq '^training_exit=.* status=[1-9][0-9]*$' "$training_controller_log" 2>/dev/null
}

training_reported_success() {
  grep -Eq '^training_exit=.* status=0$' "$training_controller_log" 2>/dev/null
}

wait_for_training() {
  local wait_seconds=${TGVF_SHAPED_TRAIN_WAIT_SECONDS:-28800}
  local started=$SECONDS
  phase=waiting_for_training_exit0_and_durable_step80
  printf '[%s] waiting for training exit=0 plus durable fresh step80 (timeout=%ss)\n' \
    "$(timestamp)" "$wait_seconds"
  while ! { training_reported_success && basic_step80_boundary_present; }; do
    if training_reported_failure; then
      echo "fresh80 controller reported a non-zero training exit"
      return 1
    fi
    if (( SECONDS - started >= wait_seconds )); then
      echo "timed out before successful fresh step80 completion"
      return 1
    fi
    sleep 15
  done
  printf '[%s] training exit=0 and basic durable step80 markers are present\n' "$(timestamp)"
}

validate_training_and_pin_step80() {
  phase=validating_training_and_pinning_step80
  PYTHONPATH="$runtime_root/src" "$python_bin" - \
    "$policy_config" "$policy_root" "$runtime_root" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from tgvf_rl.framework.verl.checkpoint_bridge import PolicyPilotVerlCheckpointPair
from tgvf_rl.policy.checkpoint import PilotProjectCheckpointState
from tgvf_rl.policy.launch import assert_policy_execution_identity
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


config_path = Path(sys.argv[1]).resolve()
policy_root = Path(sys.argv[2]).resolve()
runtime_root = Path(sys.argv[3]).resolve()
state_root = policy_root / "runtime-policy-state"
checkpoint_root = policy_root / "checkpoints"
config = load_policy_e2e_smoke_run_config(config_path)


def canonical(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_integrity(payload: dict[str, object], owner: str) -> None:
    expected = payload.get("integrity_sha256")
    content = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    actual = hashlib.sha256(canonical(content)).hexdigest()
    if expected != actual:
        raise RuntimeError(f"{owner} integrity mismatch")


def finite(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite metric: {value!r}")


expected_run_id = "PRL-09-R2-QWEN3-INSTRUCT-GRPO-BS16-TGVF-SHAPED-T1MIXED-V2-80STEP-GPU0123"
if config.run_id != expected_run_id:
    raise RuntimeError("fresh80 run identity differs")
if subprocess.run(
    ("git", "status", "--porcelain", "--untracked-files=normal"),
    cwd=runtime_root,
    check=True,
    capture_output=True,
    text=True,
).stdout:
    raise RuntimeError("fresh80 runtime became dirty")
assert_policy_execution_identity(config, repository_root=runtime_root)
if config.reward.profile != "stage3-shaped-v1":
    raise RuntimeError("internal shaped reward compatibility profile differs")
if config.reward.tool_utility is None:
    raise RuntimeError("full counterfactual utility sidecar is not bound")
if config.reward.visual_quality_judge_config_path is None:
    raise RuntimeError("local visual-quality judge is not bound")
visual_config = json.loads(
    config.reward.visual_quality_judge_config_path.read_text(encoding="utf-8")
)
if visual_config["service"]["base_url"] != "http://127.0.0.1:8013/v1":
    raise RuntimeError("visual-quality judge endpoint differs")
if config.protocol.maximum_tool_calls != 1:
    raise RuntimeError("TGVF-shaped run does not use the one-call contract")
if config.accumulation.global_prompt_batch_size != 16:
    raise RuntimeError("fresh80 is not BS16")
if config.accumulation.gradient_accumulation_steps != 4:
    raise RuntimeError("fresh80 does not use GA4")
if config.training.maximum_optimizer_steps != 80 or config.scheduler.total_steps != 80:
    raise RuntimeError("fresh80 optimizer/scheduler horizon differs")
if config.training.validation_before_training is not False:
    raise RuntimeError("validation_before_training is enabled")
if config.training.validation_frequency != -1:
    raise RuntimeError("in-training validation is enabled")
if config.training.resume_mode != "disable" or config.training.resume_from_path is not None:
    raise RuntimeError("fresh80 resume contract differs")
if config.distributed.physical_gpu_ids != (0, 1, 2, 3):
    raise RuntimeError("fresh80 GPU identity differs")

rows = [
    json.loads(line)
    for line in (policy_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
steps = [row.get("optimizer_step") for row in rows]
if steps != list(range(1, 81)):
    raise RuntimeError(f"metrics are not the exact fresh 1..80 sequence: {steps!r}")
finite(rows)
required_shaped_step_fields = {
    "mean_stage3_answer_reward",
    "mean_stage3_tool_reward",
    "mean_stage3_focus_reward",
    "mean_stage3_grounding_reward",
    "mean_stage3_protocol_reward",
    "stage3_quality_judge_applicable",
    "stage3_quality_judge_covered",
    "stage3_quality_judge_failures",
    "stage3_quality_judge_coverage",
    "stage3_visual_judge_calls",
}
visual_applicable = 0
visual_covered = 0
visual_failures = 0
for row in rows:
    step_metrics = row.get("step")
    if not isinstance(step_metrics, dict):
        raise RuntimeError("optimizer-step metric lacks its step mapping")
    missing = required_shaped_step_fields.difference(step_metrics)
    if missing:
        raise RuntimeError(f"optimizer step {row['optimizer_step']} lacks shaped metrics: {sorted(missing)!r}")
    applicable = step_metrics["stage3_quality_judge_applicable"]
    covered = step_metrics["stage3_quality_judge_covered"]
    failures = step_metrics["stage3_quality_judge_failures"]
    calls = step_metrics["stage3_visual_judge_calls"]
    if calls != applicable or covered != applicable or failures != 0:
        raise RuntimeError(f"optimizer step {row['optimizer_step']} visual-judge accounting differs")
    visual_applicable += applicable
    visual_covered += covered
    visual_failures += failures
if (
    visual_applicable <= 0
    or visual_covered != visual_applicable
    or visual_failures != 0
):
    raise RuntimeError("fresh80 local visual-quality judge coverage is incomplete")

step = 80
actor_root = checkpoint_root / "global_step_80" / "actor"
pair_path = actor_root / "tgvf_policy_checkpoint_pair.json"
project_path = actor_root / "tgvf_policy_project_state.json"
pair = PolicyPilotVerlCheckpointPair.from_checkpoint_mapping(
    json.loads(pair_path.read_text(encoding="utf-8"))
)
project = PilotProjectCheckpointState.from_checkpoint_mapping(
    json.loads(project_path.read_text(encoding="utf-8"))
)
if pair.run_id != expected_run_id or pair.optimizer_step != step:
    raise RuntimeError("step80 checkpoint-pair identity differs")
if pair.project_state_sha256 != project.integrity_sha256:
    raise RuntimeError("step80 checkpoint pair/project digest differs")
if project.policy_version.run_id != expected_run_id:
    raise RuntimeError("step80 project policy run differs")
if project.policy_version.optimizer_step != step:
    raise RuntimeError("step80 project policy step differs")
for prefix in ("model", "optim", "extra_state"):
    files = sorted(actor_root.glob(f"{prefix}_world_size_4_rank_*.pt"))
    if len(files) != 4 or any(not path.is_file() or path.stat().st_size == 0 for path in files):
        raise RuntimeError(f"step80 upstream {prefix} rank set is incomplete")

manifest_paths = sorted((state_root / "lora-manifests").glob("step-00000080-*.json"))
if len(manifest_paths) != 1:
    raise RuntimeError(f"step80 has {len(manifest_paths)} LoRA manifests, expected one")
manifest_path = manifest_paths[0]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
verify_integrity(manifest, "step80 LoRA manifest")
if manifest.get("schema_version") != "tgvf-policy-lora-snapshot-v1":
    raise RuntimeError("step80 LoRA manifest schema differs")
if manifest.get("run_id") != expected_run_id or manifest.get("optimizer_step") != 80:
    raise RuntimeError("step80 LoRA manifest identity differs")
if manifest.get("run_identity_sha256") != config.identity_sha256:
    raise RuntimeError("step80 LoRA run identity differs")
if manifest.get("weights_sha256") != project.policy_version.weights_sha256:
    raise RuntimeError("step80 checkpoint/LoRA weights differ")
tensor_path = state_root / str(manifest["tensor_file"])
if not tensor_path.is_file():
    raise RuntimeError("step80 LoRA tensor is missing")
if file_sha256(tensor_path) != manifest.get("tensor_file_sha256"):
    raise RuntimeError("step80 LoRA tensor digest differs")

latest_path = state_root / "latest-lora-snapshot.json"
latest = json.loads(latest_path.read_text(encoding="utf-8"))
verify_integrity(latest, "latest LoRA pointer")
if latest.get("optimizer_step") != 80 or latest.get("run_id") != expected_run_id:
    raise RuntimeError("latest LoRA pointer is not fresh step80")
if latest.get("run_identity_sha256") != config.identity_sha256:
    raise RuntimeError("latest LoRA pointer run identity differs")

relative_manifest = manifest_path.relative_to(state_root).as_posix()
pointer_content = {
    "schema_version": "tgvf-policy-lora-latest-v1",
    "run_id": manifest["run_id"],
    "run_identity_sha256": manifest["run_identity_sha256"],
    "optimizer_step": manifest["optimizer_step"],
    "request_sha256": manifest["request_sha256"],
    "weights_sha256": manifest["weights_sha256"],
    "manifest_file": relative_manifest,
    "manifest_file_sha256": file_sha256(manifest_path),
}
pointer = dict(pointer_content)
pointer["integrity_sha256"] = hashlib.sha256(canonical(pointer_content)).hexdigest()
encoded = canonical(pointer) + b"\n"
target = state_root / "step80-lora-snapshot-pointer.json"
if target.exists():
    if target.read_bytes() != encoded:
        raise RuntimeError("step80 pointer identity collision")
else:
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
if target.read_bytes() != latest_path.read_bytes():
    raise RuntimeError("pinned step80 pointer differs from durable latest pointer")

print(json.dumps({
    "status": "pass",
    "run_id": expected_run_id,
    "run_identity_sha256": config.identity_sha256,
    "optimizer_steps": 80,
    "global_prompt_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "visual_quality_applicable": visual_applicable,
    "visual_quality_covered": visual_covered,
    "visual_quality_failures": visual_failures,
    "weights_sha256": manifest["weights_sha256"],
}, sort_keys=True))
PY
}

gpu0123_are_idle() {
  local gpu active
  for gpu in 0 1 2 3; do
    if ! active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null); then
      return 1
    fi
    active=$(printf '%s\n' "$active" | tr -d '[:space:]')
    [[ -z "$active" ]] || return 1
  done
}

wait_for_idle_gpus() {
  local wait_seconds=${TGVF_SHAPED_GPU_IDLE_WAIT_SECONDS:-1800}
  local started=$SECONDS
  local consecutive=0
  phase=waiting_for_idle_gpu0123
  while (( consecutive < 3 )); do
    if gpu0123_are_idle; then
      ((consecutive += 1))
    else
      consecutive=0
    fi
    if (( consecutive < 3 )); then
      if (( SECONDS - started >= wait_seconds )); then
        echo "GPU0-3 did not become idle before evaluation timeout"
        return 1
      fi
      sleep 10
    fi
  done
  printf '[%s] GPU0-3 were process-free for three consecutive checks\n' "$(timestamp)"
}

validate_inference_completion() {
  "$python_bin" - "$eval_config" "$eval_root" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
tasks = [
    json.loads(line)
    for line in (root / "runtime/coredev-official-tasks.jsonl").read_text(encoding="utf-8").splitlines()
]
single = {row["ordinal"] for row in tasks if len(row["image_paths"]) == 1}
if len(tasks) != 2511 or len(single) != 2240 or len(tasks) - len(single) != 271:
    raise RuntimeError("CoreDev task tranche is not 2240 supported / 271 unsupported")

observed: set[int] = set()
for rank in range(4):
    result_path = root / f"inference/rank-{rank}.jsonl"
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    expected_rank = {ordinal for ordinal in single if ordinal % 4 == rank}
    ordinals = [row.get("ordinal") for row in rows]
    if set(ordinals) != expected_rank or len(ordinals) != len(expected_rank):
        raise RuntimeError(f"rank{rank} result identity/count differs")
    if any(row.get("dataset") is None or row.get("index") is None for row in rows):
        raise RuntimeError(f"rank{rank} result identity is incomplete")
    observed.update(ordinals)

    completion = []
    for line in (root / f"logs/rank-{rank}.log").read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("rank") == rank and (
            type(payload.get("completed")) is int or payload.get("remaining") == 0
        ):
            completion.append(payload)
    if not completion:
        raise RuntimeError(f"rank{rank} lacks an explicit completion marker")

if observed != single:
    raise RuntimeError("supported CoreDev inference coverage differs")
print(json.dumps({
    "status": "pass",
    "supported": len(observed),
    "unsupported": 271,
    "evaluation_id": config["evaluation_id"],
}, sort_keys=True))
PY
}

start_scoring_judge() {
  local judge_log="$eval_root/logs/qwen25-72b-judge-gpu23.log"
  phase=starting_qwen25_72b_scoring_judge
  "$python_bin" - "$judge_port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
PY
  (
    cd "$repo_root"
    exec setsid env \
      CUDA_VISIBLE_DEVICES=2,3 \
      CC=/usr/bin/gcc CXX=/usr/bin/g++ \
      CPATH="$repo_root/.deps/python312-dev/root/usr/include:$repo_root/.deps/python312-dev/root/usr/include/python3.12" \
      PATH="$repo_root/.venv312/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
      VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
      VLLM_PLUGINS= \
      VLLM_ATTENTION_BACKEND=TRITON_ATTN \
      TOKENIZERS_PARALLELISM=false \
      "$python_bin" -m vllm.entrypoints.openai.api_server \
        --model /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct \
        --served-model-name Qwen2.5-72B-Instruct \
        --host 127.0.0.1 --port "$judge_port" \
        --tensor-parallel-size 2 --dtype bfloat16 \
        --max-model-len 32768 --gpu-memory-utilization 0.85 \
        --max-num-seqs 64 --seed 42 --generation-config vllm \
        --enable-prefix-caching
  ) >"$judge_log" 2>&1 &
  judge_pid=$!

  local ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$judge_port/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 -- "-$judge_pid" 2>/dev/null; then
      echo "Qwen2.5-72B scoring judge exited during startup"
      return 1
    fi
    sleep 5
  done
  [[ "$ready" == 1 ]] || { echo "Qwen2.5-72B scoring judge readiness timeout"; return 1; }
  curl -fsS "$judge_base_url/models" | "$python_bin" -c \
    'import json,sys; p=json.load(sys.stdin); assert any(x["id"]=="Qwen2.5-72B-Instruct" for x in p["data"])'
  printf '[%s] Qwen2.5-72B TP2 scoring judge ready\n' "$(timestamp)"
}

stop_scoring_judge() {
  stop_process_group "$judge_pid"
  if [[ -n "$judge_pid" ]]; then
    wait "$judge_pid" 2>/dev/null || true
  fi
  judge_pid=""
}

strict_collect_summary() {
  (
    cd "$runtime_root"
    PYTHONPATH="$runtime_root/src" "$python_bin" tools/summarize_coredev_2511.py \
      --work-dir "$score_root" --phase eval \
      --judge-base-url "$judge_base_url" \
      --expected-model Qwen3-VL-8B-Instruct \
      --output "$summary_path"
  )
}

validate_strict_summary() {
  "$python_bin" - "$summary_path" <<'PY'
import json
from pathlib import Path
import sys

p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if p.get("schema_version") != 1 or p.get("phase") != "eval":
    raise RuntimeError("strict CoreDev summary schema/phase differs")
if p.get("model") != "Qwen3-VL-8B-Instruct":
    raise RuntimeError("strict CoreDev summary model differs")
if p.get("sample_count") != 2511 or p.get("slice_count") != 7:
    raise RuntimeError("strict CoreDev summary completion marker differs")
if len(p.get("slices", [])) != 7:
    raise RuntimeError("strict CoreDev summary lacks seven slices")
print(json.dumps({"validation": "pass", "sample_count": 2511, "slice_count": 7}, sort_keys=True))
PY
}

score_evaluation() {
  phase=materializing_strict_seven_slice_scoring_views
  if [[ ! -s "$score_root/materialization-summary.json" ]]; then
    if [[ -e "$score_root/raw" ]]; then
      echo "partial scoring materialization exists without a committed summary: $score_root"
      return 1
    fi
    local score_run_id="T$(date '+%Y%m%d-%H%M%S')"
    (
      cd "$runtime_root"
      PYTHONPATH="$runtime_root/src" "$python_bin" tools/materialize_policy_coredev_scoring.py \
        --inference-root "$eval_root/inference" \
        --tasks "$eval_root/runtime/coredev-official-tasks.jsonl" \
        --source-root /nvmesv/dredvpn009/datasets/benchmarks/coredev_2511_vlmevalkit_7055d301_v1 \
        --output-root "$score_root" \
        --evaluation-id "$evaluation_id" \
        --run-id "$score_run_id" \
        --mathverse-source-json /nvmesv/dredvpn009/datasets/benchmarks/mathverse/snapshot/testmini.json
    ) | tee "$eval_root/logs/materialization.log"
  fi
  "$python_bin" - "$score_root/materialization-summary.json" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "official_row_count": 2511,
    "observed_single_image_count": 2240,
    "unsupported_multi_image_count": 271,
    "forced_invalid_count": 0,
}
for field, value in expected.items():
    if payload.get(field) != value:
        raise RuntimeError(f"materialization {field} differs: {payload.get(field)}")
if len(payload.get("slices", [])) != 7:
    raise RuntimeError("materialization does not contain exactly seven slices")
print(json.dumps({"status": "pass", **expected}, sort_keys=True))
PY

  phase=polygon3_preflight
  "$python_bin" - <<'PY'
import Polygon
from pathlib import Path

if not Path(Polygon.__file__).is_file():
    raise RuntimeError("Polygon3 import did not resolve to an installed module")
print(f"Polygon3 preflight pass: {Polygon.__file__}")
PY

  wait_for_idle_gpus
  start_scoring_judge

  if [[ -s "$summary_path" ]]; then
    phase=revalidating_existing_strict_summary
    if strict_collect_summary >"$eval_root/logs/resume-summary-strict.log" 2>&1; then
      validate_strict_summary
      stop_scoring_judge
      printf '[%s] existing strict seven-slice score artifacts revalidated\n' "$(timestamp)"
      return 0
    fi
  fi

  phase=scoring_strict_seven_slices
  local datasets=(VStarBench HRBench4K BLINK OCRBench_v2 MMMU_Pro_10c MathVista_MINI MathVerse_MINI)
  score_pids=()
  local dataset
  for dataset in "${datasets[@]}"; do
    (
      cd "$runtime_root"
      exec setsid env \
        CUDA_VISIBLE_DEVICES= \
        VLLM_PLUGINS= \
        PYTHONPATH="$runtime_root/src" \
        PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false \
        "$python_bin" tools/run_coredev_2511_vlmevalkit.py \
          --config configs/evaluation/coredev_2511_qwen3_instruct_direct_prl04_v1.json \
          --model Qwen3-VL-8B-Instruct \
          --data "$dataset" \
          --work-dir "$score_root/$dataset" \
          --mode eval --reuse-aux infer \
          --judge Qwen2.5-72B-Instruct \
          --judge-base-url "$judge_base_url" \
          --judge-api-nproc 8 --judge-timeout 600
    ) >"$eval_root/logs/score-$dataset.log" 2>&1 &
    score_pids+=("$!")
  done

  local scoring_status=0
  for pid in "${score_pids[@]}"; do
    if ! wait "$pid"; then
      scoring_status=1
    fi
  done
  score_pids=()
  if (( scoring_status != 0 )); then
    echo "one or more fail-closed CoreDev scoring slices failed"
    return 1
  fi

  phase=collecting_strict_seven_slice_summary
  strict_collect_summary | tee "$eval_root/logs/final-summary-strict.log"
  validate_strict_summary
  stop_scoring_judge
}

run_evaluation() {
  phase=preparing_step80_policy_inference
  (
    cd "$runtime_root"
    PYTHONPATH="$runtime_root/src" "$python_bin" tools/run_policy_coredev_2511.py \
      --config "$eval_config" --mode prepare
  ) | tee "$eval_root/logs/prepare.log"

  wait_for_idle_gpus
  phase=running_step80_policy_inference
  worker_pids=()
  local rank
  for rank in 0 1 2 3; do
    (
      cd "$runtime_root"
      exec setsid env \
        CUDA_VISIBLE_DEVICES="$rank" \
        PYTHONPATH="$runtime_root/src" \
        PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false \
        CC=/usr/bin/gcc CXX=/usr/bin/g++ \
        CPATH="$repo_root/.deps/python312-dev/root/usr/include:$repo_root/.deps/python312-dev/root/usr/include/python3.12" \
        VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
        "$python_bin" tools/run_policy_coredev_2511.py \
          --config "$eval_config" --mode worker --rank "$rank" --world-size 4
    ) >"$eval_root/logs/rank-$rank.log" 2>&1 &
    worker_pids+=("$!")
  done

  local inference_status=0
  for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
      inference_status=1
    fi
  done
  worker_pids=()
  if (( inference_status != 0 )); then
    echo "one or more step80 policy inference ranks failed"
    return 1
  fi

  (
    cd "$runtime_root"
    PYTHONPATH="$runtime_root/src" "$python_bin" tools/run_policy_coredev_2511.py \
      --config "$eval_config" --mode status
  ) | tee "$eval_root/logs/inference-status.json"
  validate_inference_completion
  score_evaluation
  printf '[%s] strict TGVF-shaped step80 CoreDev-2511 evaluation complete\n' "$(timestamp)"
}

rm -f "$complete_marker" "$failure_marker"
printf '[%s] PRL-09 R2 post-step80 strict CoreDev supervisor started\n' "$(timestamp)"
[[ -x "$python_bin" ]] || { echo "project Python is unavailable: $python_bin"; exit 1; }
[[ -d "$runtime_root/.git" ]] || { echo "clean runtime checkout is unavailable: $runtime_root"; exit 1; }
[[ -z "$(git -C "$runtime_root" status --porcelain --untracked-files=normal)" ]] || {
  echo "TGVF-shaped runtime checkout is dirty: $runtime_root"
  exit 1
}
[[ -s "$policy_config" ]] || { echo "finalized fresh80 config is unavailable: $policy_config"; exit 1; }
[[ -s "$eval_config" ]] || { echo "CoreDev config is unavailable: $eval_config"; exit 1; }
if rg -n '__FILL_[A-Z0-9_]+__' "$policy_config"; then
  echo "fresh80 config still contains template placeholders"
  exit 1
fi

wait_for_training
validate_training_and_pin_step80 | tee "$control_root/step80-checkpoint-validation.json"
wait_for_idle_gpus
run_evaluation

phase=writing_completion_marker
printf 'status=pass\ntime=%s\nrun_id=%s\nsummary=%s\nslice_count=7\nsample_count=2511\n' \
  "$(timestamp)" \
  PRL-09-R2-QWEN3-INSTRUCT-GRPO-BS16-TGVF-SHAPED-T1MIXED-V2-80STEP-GPU0123 \
  "$summary_path" >"$complete_marker"
phase=complete
printf '[%s] PRL-09 R2 strict CoreDev ACC-VAL passed\n' "$(timestamp)"
