#!/usr/bin/env python3
"""Real veRL V1 FSDP2-actor to vLLM generation/sync compatibility smoke.

The smoke runs two optimizer steps through the pinned upstream veRL V1 sync
trainer.  It uses ``AgentLoopManagerTQ`` and TransferQueue, a composable-FSDP2
actor over two ranks, a colocated vLLM TP=2 rollout, and veRL's naive
CUDA-IPC/ZMQ weight-transfer path.  Step 1 samples weight version 0, performs a
real non-RL actor update, and synchronizes version 1.  Step 2 must then generate
with version 1 and numerically agree with the pre-update FSDP2 replay.

Sleep/wake is intentionally disabled.  This is the viable NCCL 2.28.9 route;
it is not evidence for the separate upstream image's NCCL >=2.29.7 sleep path.
It also does not exercise the repository's TGVF latent/DeepStack plugin or
claim production policy/reference replay or objective mathematics.

Without ``--launch-gpu`` the script prints the complete plan and performs no
writes or CUDA work.  A GPU launch is permitted only after the exact command
and output identity have a complete PLANNED experiment-ledger entry.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.experiment_identity import validate_run_id  # noqa: E402

ACCEPTED_MODEL_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking")
REWARD_PATH = Path(__file__).with_name("verl_sync_fixed_reward.py").resolve()
LOCK_PATH = REPOSITORY_ROOT / "requirements/compatibility-torch211-cu129.lock"
OBJECTIVE_MODULE = "tgvf_rl.framework.verl.sync_gate_objective"
MANAGER_MODULE = "tgvf_rl.framework.verl.sync_gate_manager"
MANAGER_CLASS = f"{MANAGER_MODULE}.SyncGateAgentLoopManagerTQ"
ADVANTAGE_ESTIMATOR = "tgvf_sync_gate_zero"
POLICY_LOSS = "tgvf_sync_gate_nll"
RESULT_SCHEMA_VERSION = "tgvf-verl-fsdp2-vllm-sync-v1"
PLAN_SCHEMA_VERSION = "tgvf-verl-fsdp2-vllm-sync-plan-v1"
FIXTURE_SCHEMA_VERSION = "tgvf-verl-vllm-sync-fixture-v1"
EXPECTED_VERL_COMMIT = "638b8ff84f279e054982f1f4633a546f3c6ced68"
EXPECTED_DISTRIBUTIONS: Mapping[str, str] = {
    "torch": "2.11.0+cu129",
    "torchvision": "0.26.0+cu129",
    "torchaudio": "2.11.0+cu129",
    "transformers": "4.57.6",
    "vllm": "0.23.0+cu129",
    "verl": "0.9.0.dev0",
    "TransferQueue": "0.1.8",
    "nvidia-nccl-cu12": "2.28.9",
}
PHYSICAL_GPUS = "2,3"
PYTHON_HEADER_ROOT = REPOSITORY_ROOT / ".deps/python312-dev/root/usr/include"
REQUIRED_ENVIRONMENT: Mapping[str, str] = {
    "CUDA_VISIBLE_DEVICES": PHYSICAL_GPUS,
    "CC": "/usr/bin/gcc",
    "CXX": "/usr/bin/g++",
    "CPATH": f"{PYTHON_HEADER_ROOT}:{PYTHON_HEADER_ROOT / 'python3.12'}",
    "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
    "VLLM_PLUGINS": "__tgvf_native_sync_gate_no_plugins__",
    "VLLM_USE_V1": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "NCCL_DEBUG": "WARN",
}


@dataclass(frozen=True, slots=True)
class GatePaths:
    result: Path
    fixture: Path
    metrics: Path
    log: Path
    resolved_config: Path
    plan: Path


@dataclass(frozen=True, slots=True)
class LogprobTolerance:
    minimum_pearson: float = 0.90
    gross_max_probability_difference: float = 0.20
    gross_mean_probability_difference: float = 0.02
    step_two_relative_multiplier: float = 5.0
    step_two_max_floor: float = 0.05
    step_two_mean_floor: float = 0.01


TOLERANCE = LogprobTolerance()


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_result_path(raw: Path, *, require_new: bool) -> Path:
    """Resolve one result JSON below artifacts/compatibility and never overwrite."""

    path = raw if raw.is_absolute() else REPOSITORY_ROOT / raw
    path = path.resolve()
    allowed = (REPOSITORY_ROOT / "artifacts" / "compatibility").resolve()
    if path == allowed or allowed not in path.parents:
        raise ValueError(f"output must be a child of {allowed}")
    if path.suffix != ".json":
        raise ValueError("sync-gate result must have a .json suffix")
    if require_new and path.exists():
        raise FileExistsError(f"result already exists: {path}")
    return path


def absolute_executable(raw: Path) -> Path:
    """Make an executable path absolute without resolving a virtualenv symlink."""

    path = raw if raw.is_absolute() else Path.cwd() / raw
    path = Path(os.path.abspath(path))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(
            f"Python executable is absent or not executable: {path}"
        )
    return path


def derive_paths(result: Path) -> GatePaths:
    stem = result.with_suffix("")
    return GatePaths(
        result=result,
        fixture=stem.with_name(f"{stem.name}.fixture.parquet"),
        metrics=stem.with_name(f"{stem.name}.metrics.jsonl"),
        log=stem.with_name(f"{stem.name}.log"),
        resolved_config=stem.with_name(f"{stem.name}.resolved.yaml"),
        plan=stem.with_name(f"{stem.name}.plan.json"),
    )


def fixture_rows(*, run_id: str) -> tuple[dict[str, Any], ...]:
    """Return the two deterministic prompts repeated at both trainer steps."""

    run_id = validate_run_id(run_id)
    prompts = (
        "Reply with one short statement about a red square.",
        "Reply with one short statement about a blue circle.",
    )
    return tuple(
        {
            "run_id": run_id,
            "data_source": "tgvf_verl_vllm_sync_gate",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "compatibility",
            "reward_model": {"style": "rule", "ground_truth": 0.0},
            "extra_info": {
                "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
                "run_id": run_id,
                "index": index,
            },
            # Candidate V1 otherwise ignores rollout.do_sample for train rows.
            "__do_sample__": False,
        }
        for index, prompt in enumerate(prompts)
    )


def fixture_logical_sha256(*, run_id: str) -> str:
    payload = json.dumps(
        fixture_rows(run_id=run_id),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def write_fixture(path: Path, *, run_id: str) -> None:
    if path.exists():
        raise FileExistsError(f"fixture already exists: {path}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(list(fixture_rows(run_id=run_id)))
    pq.write_table(table, path, compression="zstd")


def hydra_overrides(*, model_path: Path, paths: GatePaths) -> tuple[str, ...]:
    """Return the complete fail-closed candidate V1 configuration."""

    return (
        "algorithm.adv_estimator=tgvf_sync_gate_zero",
        "algorithm.use_kl_in_reward=false",
        "algorithm.rollout_correction.rollout_is=null",
        f"data.train_files={paths.fixture}",
        f"data.val_files={paths.fixture}",
        "data.train_batch_size=2",
        "data.val_batch_size=2",
        "data.max_prompt_length=128",
        "data.max_response_length=16",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.shuffle=false",
        "data.dataloader_num_workers=0",
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.model.external_lib={OBJECTIVE_MODULE}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.model.use_remove_padding=false",
        "actor_rollout_ref.model.enable_gradient_checkpointing=false",
        "actor_rollout_ref.model.lora.dropout=0.0",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.optim.lr=1e-3",
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        "actor_rollout_ref.actor.ppo_mini_batch_size=2",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=false",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.shuffle=false",
        f"actor_rollout_ref.actor.policy_loss.loss_mode={POLICY_LOSS}",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.use_torch_compile=false",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=2",
        "actor_rollout_ref.actor.fsdp_config.strategy=fsdp2",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
        "actor_rollout_ref.actor.fsdp_config.full_determinism=true",
        "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true",
        "actor_rollout_ref.actor.fsdp_config.offload_policy=false",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=false",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.data_parallel_size=1",
        "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        "actor_rollout_ref.rollout.load_format=dummy",
        "actor_rollout_ref.rollout.checkpoint_engine.backend=naive",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256",
        "actor_rollout_ref.rollout.free_cache_engine=false",
        "+actor_rollout_ref.rollout.enable_sleep_mode=false",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.full_determinism=true",
        "actor_rollout_ref.rollout.seed=20260720",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.do_sample=false",
        "actor_rollout_ref.rollout.ignore_eos=true",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.max_model_len=192",
        "actor_rollout_ref.rollout.max_num_batched_tokens=256",
        "actor_rollout_ref.rollout.max_num_seqs=2",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.2",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.enable_chunked_prefill=false",
        "actor_rollout_ref.rollout.enable_prefix_caching=false",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={MANAGER_CLASS}",
        "critic.enable=false",
        "reward.num_workers=1",
        "reward.reward_model.enable=false",
        f"reward.custom_reward_function.path={REWARD_PATH}",
        "reward.custom_reward_function.name=compute_score",
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=1",
        "transfer_queue.backend.SimpleStorage.total_storage_size=64",
        "trainer.use_v1=true",
        "trainer.v1.trainer_mode=sync",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "trainer.total_epochs=2",
        "trainer.total_training_steps=2",
        'trainer.logger=["console","file"]',
        "trainer.project_name=tgvf_compatibility",
        "trainer.experiment_name=torch211_verl_vllm_fsdp2_no_sleep",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.resume_mode=disable",
        "trainer.balance_batch=false",
    )


def build_command(
    *, python: Path, model_path: Path, paths: GatePaths
) -> tuple[str, ...]:
    return (
        str(python),
        "-m",
        "verl.trainer.main_ppo",
        *hydra_overrides(model_path=model_path, paths=paths),
    )


def child_environment(paths: GatePaths) -> dict[str, str]:
    if not (PYTHON_HEADER_ROOT / "python3.12/Python.h").is_file():
        raise FileNotFoundError("pinned local Python 3.12 headers are missing")
    result = dict(os.environ)
    # Ray must remap each actor to a logical CUDA ordinal.  Keeping physical
    # host IDs visible makes veRL interpret accelerator IDs 2/3 as local
    # ordinals inside a two-device process and fail before worker creation.
    result.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
    result.update(REQUIRED_ENVIRONMENT)
    result["VERL_FILE_LOGGER_PATH"] = str(paths.metrics)
    return result


def _verl_commit_identity() -> str | None:
    distribution = metadata.distribution("verl")
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    direct = json.loads(raw)
    vcs = direct.get("vcs_info")
    if isinstance(vcs, dict):
        commit = vcs.get("commit_id")
        return commit if isinstance(commit, str) else None
    return None


def runtime_identity() -> dict[str, Any]:
    versions = {name: metadata.version(name) for name in EXPECTED_DISTRIBUTIONS}
    checks = {
        name: versions[name] == expected
        for name, expected in EXPECTED_DISTRIBUTIONS.items()
    }
    commit = _verl_commit_identity()
    checks["verl_commit"] = commit == EXPECTED_VERL_COMMIT
    checks["python_3_12"] = platform.python_version_tuple()[:2] == ("3", "12")
    result = {
        "python": platform.python_version(),
        "distributions": versions,
        "verl_commit": commit,
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"candidate runtime identity failed: {checks}")
    return result


def validate_model_path(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    config_path = resolved / "config.json"
    if resolved != ACCEPTED_MODEL_PATH.resolve():
        raise ValueError("sync gate requires the accepted stable local Qwen3 path")
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is absent: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checks = {
        "model_type": config.get("model_type") == "qwen3_vl",
        "architecture": config.get("architectures")
        == ["Qwen3VLForConditionalGeneration"],
    }
    if not all(checks.values()):
        raise ValueError(f"accepted Qwen3 model metadata changed: {checks}")
    return {
        "path": str(resolved),
        "config_sha256": _sha256_path(config_path),
        "checks": checks,
    }


def plan_payload(
    *,
    run_id: str,
    python: Path,
    model_path: Path,
    paths: GatePaths,
    timeout_seconds: int,
) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    command = build_command(python=python, model_path=model_path, paths=paths)
    env = {**REQUIRED_ENVIRONMENT, "VERL_FILE_LOGGER_PATH": str(paths.metrics)}
    source_paths = {
        "driver": Path(__file__).resolve(),
        "reward": REWARD_PATH,
        "objective": SOURCE_ROOT / "tgvf_rl/framework/verl/sync_gate_objective.py",
        "manager_import_hook": SOURCE_ROOT
        / "tgvf_rl/framework/verl/sync_gate_manager.py",
        "candidate_lock": LOCK_PATH,
        "python_header": PYTHON_HEADER_ROOT / "python3.12/Python.h",
    }
    source_hashes = {
        name: _sha256_path(path) if path.is_file() else None
        for name, path in source_paths.items()
    }
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "scope": {
            "trainer": "upstream veRL V1 sync",
            "manager": "verl.trainer.ppo.v1.AgentLoopManagerTQ",
            "manager_contract": (
                "create(config,llm_client,teacher_client,reward_loop_worker_handles); "
                "generate_sequences(TensorDict)->None via TransferQueue"
            ),
            "actor": "two-rank composable FSDP2",
            "rollout": "vLLM TP=2 generation and processed logprobs",
            "weight_transfer": "veRL naive full-tensor materialization plus ZMQ/CUDA IPC",
            "sleep_wake": "disabled",
            "objective": "zero-advantage generated-token NLL; infrastructure only",
            "excluded": [
                "production GRPO/PPO/SDPO mathematics",
                "reference-policy replay",
                "TGVF main-D/D-DeepStack plugin transport",
                "NCCL >=2.29.7 communicator sleep/suspend path",
            ],
        },
        "expected_runtime": {
            "distributions": dict(EXPECTED_DISTRIBUTIONS),
            "verl_commit": EXPECTED_VERL_COMMIT,
        },
        "physical_gpus": [2, 3],
        "timeout_seconds": timeout_seconds,
        "fixture_logical_sha256": fixture_logical_sha256(run_id=run_id),
        "source_sha256": source_hashes,
        "artifacts": {name: str(path) for name, path in asdict(paths).items()},
        "environment": env,
        "command": list(command),
        "logprob_tolerance": asdict(TOLERANCE),
    }


def read_metric_records(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("data"), dict):
                raise ValueError(f"invalid veRL file metric at line {line_number}")
            records.append(value)
    return tuple(records)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
    )


def evaluate_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    tolerance: LogprobTolerance = TOLERANCE,
) -> dict[str, Any]:
    """Evaluate version, update, generation, and replay evidence from one run."""

    steps = [record.get("step") for record in records]
    data = [record.get("data", {}) for record in records]
    checks: dict[str, bool] = {"exact_two_steps": steps == [1, 2]}
    required_logprob = (
        "training/rollout_probs_diff_max",
        "training/rollout_probs_diff_mean",
        "training/rollout_probs_diff_std",
        "training/rollout_actor_probs_pearson_corr",
    )
    checks["all_required_metrics_present"] = len(data) == 2 and all(
        all(
            name in row
            for name in (
                *required_logprob,
                "training/rollout_probs_diff_valid",
                "training/off_policy/trajectory_spans/mean",
                "training/off_policy/trajectory_staleness/mean",
                "training/off_policy/trajectory_staleness_worst/mean",
                "response_length/min",
                "response_length/max",
                "actor/grad_norm",
                "actor/lr",
                "actor/sync_gate_zero_advantage_valid",
                "timing_s/update_weights",
            )
        )
        for row in data
    )
    if not checks["all_required_metrics_present"] or len(data) != 2:
        return {"passed": False, "checks": checks, "steps": steps}

    checks["rollout_logprobs_valid"] = all(
        row["training/rollout_probs_diff_valid"] == 1 for row in data
    )
    checks["rollout_logprob_metrics_finite"] = all(
        _finite_number(row[name]) for row in data for name in required_logprob
    )
    checks["single_version_trajectories"] = all(
        row["training/off_policy/trajectory_spans/mean"] == 1 for row in data
    )
    checks["zero_staleness"] = all(
        row["training/off_policy/trajectory_staleness/mean"] == 0
        and row["training/off_policy/trajectory_staleness_worst/mean"] == 0
        for row in data
    )
    checks["sixteen_generated_tokens"] = all(
        row["response_length/min"] == 16 and row["response_length/max"] == 16
        for row in data
    )
    checks["real_actor_updates"] = all(
        _finite_number(row["actor/grad_norm"])
        and row["actor/grad_norm"] > 0
        and _finite_number(row["actor/lr"])
        and row["actor/lr"] > 0
        and row["actor/sync_gate_zero_advantage_valid"] == 1
        for row in data
    )
    checks["weight_sync_timed"] = all(
        _finite_number(row["timing_s/update_weights"])
        and row["timing_s/update_weights"] > 0
        for row in data
    )

    if checks["rollout_logprob_metrics_finite"]:
        first, second = data
        checks["gross_logprob_agreement"] = all(
            row["training/rollout_actor_probs_pearson_corr"]
            >= tolerance.minimum_pearson
            and row["training/rollout_probs_diff_max"]
            <= tolerance.gross_max_probability_difference
            and row["training/rollout_probs_diff_mean"]
            <= tolerance.gross_mean_probability_difference
            for row in data
        )
        checks["post_update_no_regression"] = second[
            "training/rollout_probs_diff_max"
        ] <= max(
            tolerance.step_two_max_floor,
            first["training/rollout_probs_diff_max"]
            * tolerance.step_two_relative_multiplier,
        ) and second["training/rollout_probs_diff_mean"] <= max(
            tolerance.step_two_mean_floor,
            first["training/rollout_probs_diff_mean"]
            * tolerance.step_two_relative_multiplier,
        )
    else:
        checks["gross_logprob_agreement"] = False
        checks["post_update_no_regression"] = False

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "steps": steps,
        # Zero staleness and span one imply versions global_step-1: [0, 1].
        "inferred_rollout_weight_versions": [0, 1]
        if checks["exact_two_steps"]
        and checks["single_version_trajectories"]
        and checks["zero_staleness"]
        else None,
        "step_metrics": [
            {
                "step": steps[index],
                "grad_norm": row["actor/grad_norm"],
                "learning_rate": row["actor/lr"],
                "weight_sync_seconds": row["timing_s/update_weights"],
                "rollout_probs_diff_max": row["training/rollout_probs_diff_max"],
                "rollout_probs_diff_mean": row["training/rollout_probs_diff_mean"],
                "rollout_actor_probs_pearson_corr": row[
                    "training/rollout_actor_probs_pearson_corr"
                ],
            }
            for index, row in enumerate(data)
        ],
    }


def _run_process(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> int:
    with log_path.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=dict(environment),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            return 124


def _pip_check(python: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(python), "-m", "pip", "check"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }


def launch(
    *,
    run_id: str,
    python: Path,
    model_path: Path,
    paths: GatePaths,
    timeout_seconds: int,
) -> int:
    run_id = validate_run_id(run_id)
    for path in asdict(paths).values():
        if path.exists():
            raise FileExistsError(f"gate artifact already exists: {path}")
    paths.result.parent.mkdir(parents=True, exist_ok=True)

    identity = runtime_identity()
    model_identity = validate_model_path(model_path)
    pip_check = _pip_check(python)
    if not pip_check["passed"]:
        raise RuntimeError(f"pip check failed: {pip_check}")

    plan = plan_payload(
        run_id=run_id,
        python=python,
        model_path=model_path,
        paths=paths,
        timeout_seconds=timeout_seconds,
    )
    paths.plan.write_text(
        json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    write_fixture(paths.fixture, run_id=run_id)

    command = build_command(python=python, model_path=model_path, paths=paths)
    environment = child_environment(paths)
    config_environment = dict(environment)
    config_environment["CUDA_VISIBLE_DEVICES"] = ""
    config = subprocess.run(
        [*command, "--cfg", "job", "--resolve"],
        cwd=REPOSITORY_ROOT,
        env=config_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    paths.resolved_config.write_text(config.stdout, encoding="utf-8")
    if config.returncode != 0:
        raise RuntimeError(f"Hydra config composition failed: {config.stderr}")

    returncode = _run_process(
        command,
        environment=environment,
        log_path=paths.log,
        timeout_seconds=timeout_seconds,
    )
    metric_result = (
        evaluate_metrics(read_metric_records(paths.metrics))
        if paths.metrics.is_file()
        else {"passed": False, "checks": {"metrics_file_exists": False}}
    )
    log_text = paths.log.read_text(encoding="utf-8", errors="replace")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "PASS"
        if returncode == 0 and metric_result.get("passed") is True
        else "FAIL",
        "runtime_identity": identity,
        "model_identity": model_identity,
        "pip_check": pip_check,
        "plan_sha256": _sha256_path(paths.plan),
        "resolved_config_sha256": _sha256_path(paths.resolved_config),
        "fixture_parquet_sha256": _sha256_path(paths.fixture),
        "fixture_logical_sha256": fixture_logical_sha256(run_id=run_id),
        "process_returncode": returncode,
        "metrics": metric_result,
        "log_diagnostics": {
            "update_weights_done_count": log_text.count("update_weights done"),
            "sleep_mode_false_observed": "enable_sleep_mode: False" in log_text,
        },
        "artifacts": {name: str(path) for name, path in asdict(paths).items()},
    }
    paths.result.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if result["status"] == "PASS" else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=_run_id_argument,
        required=True,
        help="explicit experiment identity; never inferred from --output",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=ACCEPTED_MODEL_PATH)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--launch-gpu",
        action="store_true",
        help="launch the recorded GPU gate; without this flag only print its plan",
    )
    return parser.parse_args(argv)


def _run_id_argument(value: str) -> str:
    try:
        return validate_run_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.timeout_seconds < 60 or args.timeout_seconds > 3600:
        raise ValueError("timeout must be between 60 and 3600 seconds")
    result = bounded_result_path(args.output, require_new=args.launch_gpu)
    paths = derive_paths(result)
    # ``Path.resolve`` would follow .venv/bin/python to /usr/bin/python and
    # silently discard the candidate environment.
    python = absolute_executable(args.python)
    model_path = args.model_path.resolve()
    if not args.launch_gpu:
        print(
            json.dumps(
                plan_payload(
                    run_id=args.run_id,
                    python=python,
                    model_path=model_path,
                    paths=paths,
                    timeout_seconds=args.timeout_seconds,
                ),
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    return launch(
        run_id=args.run_id,
        python=python,
        model_path=model_path,
        paths=paths,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
