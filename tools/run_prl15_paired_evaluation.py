#!/usr/bin/env python3
"""Wait for, run, resume, and officially score ordered policy checkpoints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_benchmark_config import (  # noqa: E402
    materialize_paired_tgvf_policy_benchmark_config,
)
from tgvf_rl.evaluation.coredev_results import (  # noqa: E402
    check_qwen25_72b_judge,
    summarize_coredev_results,
    write_json_atomic,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    PolicyEvaluationSnapshot,
    freeze_policy_evaluation_snapshot,
    load_benchmark_tasks,
    load_policy_coredev_config,
    load_policy_evaluation_snapshot,
    materialize_vllm_lora_adapter,
    policy_benchmark_task_path,
    write_policy_evaluation_identity,
)
from tgvf_rl.evaluation.policy_coredev_scoring import (  # noqa: E402
    DATASETS as COREDEV_DATASETS,
    MODEL_NAME as EVALUATED_MODEL,
    materialize_policy_coredev_scoring_views,
    validate_vlmevalkit_eval_id,
)
from tgvf_rl.evaluation.policy_paired_qwen_materialization import (  # noqa: E402
    materialize_qwen_only_policy_checkpoint,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    load_policy_e2e_smoke_run_config,
)


DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs/evaluation/prl15_r1_rp66_step0_step8_coredev2511_plan.json"
)
RUNNER = REPOSITORY_ROOT / "tools/run_policy_benchmark.py"
COREDEV_RUNNER = REPOSITORY_ROOT / "tools/run_coredev_2511_vlmevalkit.py"
PLAN_SCHEMA = "tgvf.prl15-paired-policy-benchmark-plan.v2"
PAIR_SUMMARY_SCHEMA = "tgvf.prl15-paired-coredev-summary.v1"
PAIRED_RNG_PLAN_SCHEMA = "tgvf.policy-paired-evaluation-rng-plan.v1"
_PAIRED_RNG_EXCLUSIONS = (
    "evaluation_id",
    "arm_name",
    "optimizer_step",
    "checkpoint_hash",
    "policy_weights_sha256",
    "prompt_token_ids_sha256",
)
_SCORING_RUN_PREFIX = re.compile(r"T(?P<date>\d{8})-[A-Za-z0-9][A-Za-z0-9._-]*")


def _vlmevalkit_scoring_run_id(
    *, run_id_prefix: object, arm_evaluation_id: object
) -> str:
    """Derive a stable legacy VLMEvalKit ID without losing semantic identity.

    The date remains visibly pinned by the plan.  The complete arm evaluation
    ID is retained separately as ``evaluation_id`` in materializer metadata;
    its SHA-256 supplies the legacy ``G<hex>`` component used only for reuse
    discovery and filesystem layout.
    """

    if (
        not isinstance(run_id_prefix, str)
        or (match := _SCORING_RUN_PREFIX.fullmatch(run_id_prefix)) is None
    ):
        raise ValueError(
            "PRL15 VLMEvalKit run prefix must match TYYYYMMDD-<semantic-name>"
        )
    try:
        datetime.strptime(match.group("date"), "%Y%m%d")
    except ValueError as error:
        raise ValueError("PRL15 VLMEvalKit run prefix date is invalid") from error
    if not isinstance(arm_evaluation_id, str) or not arm_evaluation_id:
        raise ValueError("PRL15 arm evaluation ID must be non-empty")
    identity_hex = hashlib.sha256(arm_evaluation_id.encode("utf-8")).hexdigest()
    return validate_vlmevalkit_eval_id(f"T{match.group('date')}_G{identity_hex}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _paired_seed_namespace(plan: dict[str, Any]) -> str | None:
    contract = plan.get("paired_rng")
    return None if contract is None else str(contract["seed_namespace"])


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("PRL15 paired evaluation plan schema differs")
    arms = payload.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("paired evaluation plan must contain at least one arm")
    steps = [arm.get("optimizer_step") for arm in arms if isinstance(arm, dict)]
    names = [arm.get("name") for arm in arms if isinstance(arm, dict)]
    if (
        len(steps) != len(arms)
        or any(type(step) is not int or step < 0 for step in steps)
        or steps != sorted(steps)
        or len(set(steps)) != len(steps)
        or names != [f"step{step}" for step in steps]
    ):
        raise ValueError("paired evaluation arms must be unique ordered stepN states")
    if (
        payload.get("expected_task_count") != 2511
        or payload.get("expected_single_image_count") != 2240
        or payload.get("unsupported_multi_image_count") != 271
    ):
        raise ValueError("PRL15 CoreDev coverage must remain 2,240 + 271")
    scoring = payload.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("PRL15 paired evaluation plan lacks official scoring")
    if tuple(scoring.get("datasets", ())) != COREDEV_DATASETS:
        raise ValueError("PRL15 official seven-suite order differs")
    if scoring.get("evaluated_model") != EVALUATED_MODEL:
        raise ValueError("PRL15 evaluated model name differs")
    if scoring.get("gpt_fallback") is not False:
        raise ValueError("PRL15 official scoring must forbid GPT fallback")
    evaluation_id = payload.get("evaluation_id")
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise ValueError("PRL15 paired evaluation ID must be non-empty")
    generated_run_ids = {
        _vlmevalkit_scoring_run_id(
            run_id_prefix=scoring.get("run_id_prefix"),
            arm_evaluation_id=f"{evaluation_id}-{arm['name'].upper()}",
        )
        for arm in payload["arms"]
    }
    if len(generated_run_ids) != len(payload["arms"]):
        raise ValueError("PRL15 VLMEvalKit arm run IDs must be distinct")
    paired_rng = payload.get("paired_rng")
    if paired_rng is not None:
        expected_fields = {
            "schema_version",
            "mode",
            "seed_namespace",
            "master_seed",
            "task_manifest_sha256",
            "protocol_sha256",
            "temperature",
            "do_sample",
            "excluded_arm_components",
        }
        if not isinstance(paired_rng, dict) or set(paired_rng) != expected_fields:
            raise ValueError("paired evaluation RNG plan fields differ")
        if (
            paired_rng["schema_version"] != PAIRED_RNG_PLAN_SCHEMA
            or paired_rng["mode"] != "common_random_numbers_per_task_turn"
            or not isinstance(paired_rng["seed_namespace"], str)
            or not paired_rng["seed_namespace"]
            or paired_rng["seed_namespace"].strip() != paired_rng["seed_namespace"]
            or any(character.isspace() for character in paired_rng["seed_namespace"])
        ):
            raise ValueError("paired evaluation RNG plan identity differs")
        if (
            type(paired_rng["master_seed"]) is not int
            or paired_rng["master_seed"] < 0
            or paired_rng["temperature"] != 1.0
            or paired_rng["do_sample"] is not True
        ):
            raise ValueError("paired evaluation must retain canonical temp=1 sampling")
        if paired_rng["task_manifest_sha256"] != payload["task_manifest_sha256"]:
            raise ValueError("paired RNG task manifest differs from evaluation plan")
        if tuple(paired_rng["excluded_arm_components"]) != _PAIRED_RNG_EXCLUSIONS:
            raise ValueError("paired RNG arm-invariant exclusions differ")
    for field in ("judge_api_nproc", "judge_retry", "judge_timeout_seconds"):
        if type(scoring.get(field)) is not int or scoring[field] <= 0:
            raise ValueError(f"PRL15 scoring {field} must be positive")
    source_root = Path(scoring["source_root"])
    if not source_root.is_absolute() or not source_root.is_dir():
        raise RuntimeError("PRL15 pinned CoreDev source root is unavailable")
    for path_field, digest_field in (
        ("policy_config", "policy_config_sha256"),
        ("task_manifest_path", "task_manifest_sha256"),
    ):
        resolved = _resolve_repo_path(payload[path_field])
        if not resolved.is_file() or _sha256_file(resolved) != payload[digest_field]:
            raise RuntimeError(f"PRL15 plan {path_field} identity differs")
    judge_path = _resolve_repo_path(scoring["judge_config_path"])
    if (
        not judge_path.is_file()
        or _sha256_file(judge_path) != scoring["judge_config_sha256"]
    ):
        raise RuntimeError("PRL15 benchmark judge config identity differs")
    for path_field, digest_field in (
        ("pinned_artifacts_config_path", "pinned_artifacts_config_sha256"),
        ("vlmevalkit_deployment_config_path", "vlmevalkit_deployment_config_sha256"),
        ("mathverse_source_json", "mathverse_source_sha256"),
    ):
        resolved = _resolve_repo_path(scoring[path_field])
        if not resolved.is_file() or _sha256_file(resolved) != scoring[digest_field]:
            raise RuntimeError(f"PRL15 scoring {path_field} identity differs")
    return payload


def _validate_plan_run(plan: dict[str, Any], run: Any) -> None:
    protocol = plan.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("PRL15 plan protocol is missing")
    expected = {
        "evaluation_protocol": "training_run",
        "prompt_sha256": run.protocol.prompt_sha256,
        "tool_profile": run.protocol.tool_profile.value,
        "tool_schema_sha256": run.protocol.tool_schema_sha256,
        "maximum_tool_calls": run.protocol.maximum_tool_calls,
        "sampling_source": "bound_policy_run_config",
        "same_tasks_and_rank_partition": True,
    }
    if protocol != expected:
        raise RuntimeError("PRL15 plan protocol differs from its policy run")
    paired_rng = plan.get("paired_rng")
    if paired_rng is not None:
        runtime_protocol = {
            "profile": "training_run",
            "prompt_sha256": run.protocol.prompt_sha256,
            "tool_schema_sha256": run.protocol.tool_schema_sha256,
            "tool_profile": run.protocol.tool_profile.value,
            "enabled_tool_names": list(run.protocol.enabled_tool_names),
            "maximum_tool_calls": run.protocol.maximum_tool_calls,
            "native_pixels": False,
        }
        if _canonical_json_sha256(runtime_protocol) != paired_rng["protocol_sha256"]:
            raise RuntimeError("paired RNG protocol identity differs from policy run")
        if run.rollout_rng.master_seed != paired_rng["master_seed"]:
            raise RuntimeError("paired RNG master seed differs from policy run")
        sampling = run.policy.sampling
        if sampling.temperature != 1.0 or sampling.do_sample is not True:
            raise RuntimeError("paired evaluation no longer uses canonical temp=1")
    required_pairing = plan.get("required_pairing")
    if not isinstance(required_pairing, dict):
        raise ValueError("paired evaluation plan required_pairing is missing")
    expected_update_mode = required_pairing.get("adapter_update_mode")
    if expected_update_mode is not None:
        observed_update_mode = getattr(
            run.representation.adapter_update_mode,
            "value",
            run.representation.adapter_update_mode,
        )
        if observed_update_mode != expected_update_mode:
            raise RuntimeError("paired evaluation adapter update mode differs")
    if required_pairing.get("rp66_state_must_remain_constant") is True:
        if expected_update_mode != "frozen_adapter":
            raise ValueError(
                "constant RP66 pairing requires adapter_update_mode=frozen_adapter"
            )
        expected_weights = required_pairing.get("expected_runtime_rp66_weights_sha256")
        if (
            not isinstance(expected_weights, str)
            or len(expected_weights) != 64
            or any(
                character not in "0123456789abcdef" for character in expected_weights
            )
        ):
            raise ValueError("constant RP66 pairing requires a lowercase SHA256")


def _validate_materialized_frozen_pairing(
    plan: dict[str, Any], configs: dict[str, Path]
) -> None:
    """Prove that a frozen-RP66 evaluation binds identical endpoint state."""

    required_pairing = plan.get("required_pairing", {})
    if required_pairing.get("rp66_state_must_remain_constant") is not True:
        return
    snapshots = []
    for arm in plan["arms"]:
        config_path = configs[arm["name"]]
        config = load_policy_coredev_config(config_path)
        if getattr(config, "paired_seed_namespace", None) != _paired_seed_namespace(
            plan
        ):
            raise RuntimeError(
                f"existing {arm['name']} paired seed namespace differs from plan"
            )
        snapshot = load_policy_evaluation_snapshot(config)
        receipt = getattr(snapshot, "receipt", None)
        if receipt is None:
            raise RuntimeError("constant RP66 pairing requires paired snapshots")
        snapshots.append(receipt)
    expected_runtime_storage = required_pairing["expected_runtime_rp66_weights_sha256"]
    expected_state: str | None = None
    for arm, receipt in zip(plan["arms"], snapshots, strict=True):
        step = arm["optimizer_step"]
        expected_kind = "stage1_artifact" if step == 0 else "runtime_snapshot"
        if receipt.rp66_kind != expected_kind:
            raise RuntimeError(f"frozen RP66 step{step} must use the {expected_kind}")
        if expected_state is None:
            expected_state = receipt.rp66_state_sha256
        elif receipt.rp66_state_sha256 != expected_state:
            raise RuntimeError("frozen RP66 state changed between evaluation arms")
        if step > 0 and receipt.rp66_storage_sha256 != expected_runtime_storage:
            raise RuntimeError("frozen RP66 runtime storage identity differs")


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return (
        (REPOSITORY_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _write_immutable_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable evaluation source differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _link_immutable_file(source: Path, destination: Path, *, sha256: str) -> None:
    if _sha256_file(source) != sha256:
        raise RuntimeError("runtime RP66 source file SHA256 differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or _sha256_file(destination) != sha256
        ):
            raise RuntimeError(f"immutable evaluation source differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    if _sha256_file(destination) != sha256:
        raise RuntimeError("materialized runtime RP66 file SHA256 differs")


def _materialize_step_pointer(run: Any, *, step: int, destination: Path) -> Path:
    """Freeze one exact historical RP66 manifest without changing training state."""

    if type(step) is not int or step <= 0:
        raise ValueError("historical RP66 pointer requires a positive step")
    source_root = run.output.root / "runtime-policy-state"
    manifests = tuple(
        sorted((source_root / "lora-manifests").glob(f"step-{step:08d}-*.json"))
    )
    if not manifests:
        raise RuntimeError(f"runtime RP66 manifest is missing for step {step}")

    # A checkpoint lifecycle may republish the same frozen RP66 state after
    # waking vLLM.  Such a corrective publication has a distinct request ID,
    # but it is not a distinct model state.  Prefer the manifest named by the
    # durable latest pointer when it refers to this step, and otherwise accept
    # multiple publications only when their model/tensor identities agree.
    parsed_manifests: dict[Path, dict[str, Any]] = {}
    state_identities: set[tuple[object, object, object]] = set()
    for candidate in manifests:
        payload = json.loads(candidate.read_bytes())
        if not isinstance(payload, dict):
            raise RuntimeError("runtime RP66 manifest payload is malformed")
        parsed_manifests[candidate] = payload
        state_identities.add(
            (
                payload.get("weights_sha256"),
                payload.get("tensor_file"),
                payload.get("tensor_file_sha256"),
            )
        )
    if len(state_identities) != 1:
        raise RuntimeError(
            f"runtime RP66 manifests disagree on model state for step {step}"
        )

    source_manifest: Path | None = None
    latest_pointer = source_root / "latest-lora-snapshot.json"
    if latest_pointer.is_file():
        latest = json.loads(latest_pointer.read_bytes())
        if isinstance(latest, dict) and latest.get("optimizer_step") == step:
            referenced = source_root / str(latest.get("manifest_file", ""))
            if referenced in parsed_manifests:
                expected_sha256 = latest.get("manifest_file_sha256")
                if expected_sha256 != _sha256_file(referenced):
                    raise RuntimeError("latest runtime RP66 manifest SHA256 differs")
                source_manifest = referenced
    if source_manifest is None:
        source_manifest = manifests[-1]
    manifest_bytes = source_manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = parsed_manifests[source_manifest]
    expected_fields = {
        "schema_version",
        "run_id",
        "run_identity_sha256",
        "optimizer_step",
        "request_sha256",
        "weights_sha256",
        "tensor_file",
        "tensor_file_sha256",
        "tensor_names",
        "tensor_metadata",
        "integrity_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise RuntimeError("runtime RP66 manifest fields differ")
    if (
        manifest["schema_version"] != "tgvf-policy-lora-snapshot-v1"
        or manifest["run_id"] != run.run_id
        or manifest["run_identity_sha256"] != run.identity_sha256
        or manifest["optimizer_step"] != step
    ):
        raise RuntimeError("runtime RP66 manifest identity differs")
    manifest_relative = Path("lora-manifests") / source_manifest.name
    tensor_relative = Path(str(manifest["tensor_file"]))
    if tensor_relative.is_absolute() or ".." in tensor_relative.parts:
        raise RuntimeError("runtime RP66 tensor path is unsafe")
    source_tensor = source_root / tensor_relative
    destination_root = destination.resolve()
    _link_immutable_file(
        source_manifest,
        destination_root / manifest_relative,
        sha256=manifest_sha256,
    )
    _link_immutable_file(
        source_tensor,
        destination_root / tensor_relative,
        sha256=str(manifest["tensor_file_sha256"]),
    )
    content = {
        "schema_version": "tgvf-policy-lora-latest-v1",
        "run_id": run.run_id,
        "run_identity_sha256": run.identity_sha256,
        "optimizer_step": step,
        "request_sha256": manifest["request_sha256"],
        "weights_sha256": manifest["weights_sha256"],
        "manifest_file": manifest_relative.as_posix(),
        "manifest_file_sha256": manifest_sha256,
    }
    pointer = {
        **content,
        "integrity_sha256": hashlib.sha256(_canonical_json_bytes(content)).hexdigest(),
    }
    pointer_path = destination_root / f"step-{step:08d}-pointer.json"
    _write_immutable_file(pointer_path, _canonical_json_bytes(pointer) + b"\n")
    return pointer_path


def _step_sources(
    run: Any, *, step: int, output_base: Path, arm: str
) -> tuple[Path, Path]:
    checkpoint = run.output.checkpoint_directory / f"global_step_{step}"
    if not checkpoint.is_dir():
        permanent = run.output.root / "permanent-checkpoints" / f"global_step_{step}"
        if permanent.is_dir():
            checkpoint = permanent
    manifests = tuple(
        (run.output.root / "runtime-policy-state/lora-manifests").glob(
            f"step-{step:08d}-*.json"
        )
    )
    if step == 8 and not manifests:
        return (
            checkpoint,
            run.output.root / "runtime-policy-state/latest-lora-snapshot.json",
        )
    pointer = _materialize_step_pointer(
        run,
        step=step,
        destination=output_base / arm / "runtime/source-rp66-state",
    )
    return checkpoint, pointer


def _wait_for_optimizer_step(
    run: Any,
    *,
    optimizer_step: int,
    timeout_seconds: int,
    poll_seconds: int,
) -> None:
    if type(optimizer_step) is not int or optimizer_step <= 0:
        raise ValueError("checkpoint wait requires a positive optimizer step")
    pointer = run.output.root / "runtime-policy-state/latest-lora-snapshot.json"
    deadline = time.monotonic() + timeout_seconds
    while True:
        latest = run.output.checkpoint_directory / "latest_checkpointed_iteration.txt"
        latest_step = None
        if latest.is_file():
            try:
                latest_step = int(latest.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                latest_step = None
        checkpoints = (
            run.output.checkpoint_directory / f"global_step_{optimizer_step}",
            run.output.root / "permanent-checkpoints" / f"global_step_{optimizer_step}",
        )
        complete_checkpoint = False
        for checkpoint in checkpoints:
            actor = checkpoint / "actor"
            required = (
                checkpoint / "data.pt",
                actor / "fsdp_config.json",
                actor / "huggingface/config.json",
                actor / "tgvf_policy_checkpoint_pair.json",
                actor / "tgvf_policy_project_state.json",
                *(
                    actor
                    / f"model_world_size_{run.distributed.world_size}_rank_{rank}.pt"
                    for rank in range(run.distributed.world_size)
                ),
            )
            if checkpoint.is_dir() and all(
                path.is_file() and path.stat().st_size > 0 for path in required
            ):
                complete_checkpoint = True
                break
        pointer_step = None
        if pointer.is_file():
            try:
                pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
                pointer_step = pointer_payload.get("optimizer_step")
            except (OSError, json.JSONDecodeError, AttributeError):
                pointer_step = None
        if (
            latest_step == optimizer_step
            and complete_checkpoint
            and pointer_step == optimizer_step
        ):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for complete optimizer-step "
                f"{optimizer_step} closure"
            )
        time.sleep(poll_seconds)


def _wait_for_step8(run: Any, *, timeout_seconds: int, poll_seconds: int) -> None:
    """Compatibility wrapper retained for existing two-arm supervisors."""

    _wait_for_optimizer_step(
        run,
        optimizer_step=8,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


def _gpu_memory_mib() -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError("nvidia-smi GPU-memory output is malformed")
        index, memory = map(int, fields)
        result[index] = memory
    return result


def _wait_for_gpus(
    gpu_ids: tuple[int, ...],
    *,
    timeout_seconds: int,
    poll_seconds: int,
    free_threshold_mib: int,
    stable_polls: int = 2,
) -> None:
    """Wait until training has released every evaluation GPU."""

    if free_threshold_mib < 0 or stable_polls <= 0:
        raise ValueError("GPU availability thresholds are invalid")
    deadline = time.monotonic() + timeout_seconds
    consecutive = 0
    while True:
        memory = _gpu_memory_mib()
        missing = set(gpu_ids).difference(memory)
        if missing:
            raise RuntimeError(f"nvidia-smi omitted requested GPUs: {sorted(missing)}")
        if all(memory[gpu_id] <= free_threshold_mib for gpu_id in gpu_ids):
            consecutive += 1
            if consecutive >= stable_polls:
                return
        else:
            consecutive = 0
        if time.monotonic() >= deadline:
            occupied = {gpu_id: memory[gpu_id] for gpu_id in gpu_ids}
            raise TimeoutError(f"timed out waiting for evaluation GPUs: {occupied}")
        time.sleep(poll_seconds)


def _arm_paths(base: Path, arm: str) -> dict[str, Path]:
    root = base / arm
    return {
        "root": root,
        "config": root / "benchmark-config.json",
        "receipt": root / "runtime/source-paired-snapshot.json",
    }


def _materialize_arm(
    *,
    plan: dict[str, Any],
    run: Any,
    arm: str,
    step: int,
    output_base: Path,
    gpu_ids: tuple[int, int, int, int],
) -> Path:
    paths = _arm_paths(output_base, arm)
    if paths["config"].is_file() and paths["receipt"].is_file():
        config = load_policy_coredev_config(paths["config"])
        if config.paired_seed_namespace != _paired_seed_namespace(plan):
            raise RuntimeError(
                f"existing {arm} paired seed namespace differs from plan"
            )
        load_policy_evaluation_snapshot(config)
        return paths["config"]
    if step == 0:
        qwen_model = Path(run.model.revision_or_path)
        rp66_pointer = None
    else:
        checkpoint, rp66_pointer = _step_sources(
            run,
            step=step,
            output_base=output_base,
            arm=arm,
        )
        qwen_model = materialize_qwen_only_policy_checkpoint(
            policy_config_path=_resolve_repo_path(plan["policy_config"]),
            optimizer_step=step,
            checkpoint_path=checkpoint,
            rp66_pointer_path=rp66_pointer,
            bundle_path=paths["root"] / "runtime/qwen-only-bundle",
        )
    materialize_paired_tgvf_policy_benchmark_config(
        evaluation_id=f"{plan['evaluation_id']}-{arm.upper()}",
        policy_config_path=_resolve_repo_path(plan["policy_config"]),
        optimizer_step=step,
        qwen_model_path=qwen_model,
        rp66_pointer_path=rp66_pointer,
        paired_snapshot_receipt_path=paths["receipt"],
        task_manifest_path=plan["task_manifest_path"],
        expected_task_count=plan["expected_task_count"],
        expected_single_image_count=plan["expected_single_image_count"],
        output_root=paths["root"],
        config_path=paths["config"],
        gpu_ids=gpu_ids,
        inference_concurrency_per_gpu=8,
        max_model_len=32768,
        max_num_batched_tokens=32768,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.9,
        paired_seed_namespace=_paired_seed_namespace(plan),
    )
    return paths["config"]


def _load_existing_arm(
    *,
    plan: dict[str, Any],
    output_base: Path,
    arm: str,
    step: int,
) -> Path:
    """Load one completed arm without preparing or mutating it."""

    paths = _arm_paths(output_base, arm)
    if not paths["config"].is_file() or not paths["receipt"].is_file():
        raise FileNotFoundError(
            f"score mode requires existing {arm} config and receipt"
        )
    config = load_policy_coredev_config(paths["config"])
    if config.paired_seed_namespace != _paired_seed_namespace(plan):
        raise RuntimeError(f"existing {arm} paired seed namespace differs from plan")
    snapshot = load_policy_evaluation_snapshot(config)
    if config.evaluation_id != f"{plan['evaluation_id']}-{arm.upper()}":
        raise RuntimeError(f"existing {arm} evaluation ID differs from plan")
    if config.output_root.resolve() != paths["root"].resolve():
        raise RuntimeError(f"existing {arm} output root differs from plan")
    if snapshot.policy_version.optimizer_step != step:
        raise RuntimeError(f"existing {arm} optimizer step differs from plan")
    for rank in range(4):
        inference = config.output_root / "inference" / f"rank-{rank}.jsonl"
        if not inference.is_file() or inference.stat().st_size == 0:
            raise RuntimeError(f"score mode requires complete {arm} inference ranks")
    return paths["config"]


def _run_checked(
    command: list[str], *, environment: dict[str, str] | None = None
) -> None:
    subprocess.run(command, check=True, env=environment)


def _prepare(config_path: Path) -> None:
    _run_checked(
        [sys.executable, str(RUNNER), "--config", str(config_path), "--mode", "prepare"]
    )


def _prepare_prevalidated_bound_manifest(config_path: Path) -> None:
    """Prepare one materialized arm without re-decoding all 2,511 images.

    Arm materialization immediately before this call already validates every
    image byte/dimension while binding the immutable config.  Repeating that
    full decode two or three more times adds minutes but no new identity proof;
    this path rechecks the manifest bytes and snapshot closure only.
    """

    config = load_policy_coredev_config(config_path)
    if config.task_manifest_path is None or config.task_manifest_sha256 is None:
        _prepare(config_path)
        return
    source = config.task_manifest_path
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != config.task_manifest_sha256:
        raise RuntimeError("prevalidated task manifest SHA256 changed")
    tasks = load_benchmark_tasks(
        source,
        expected_task_count=config.expected_task_count,
        expected_single_image_count=config.expected_single_image_count,
        expected_sha256=config.task_manifest_sha256,
        verify_image_contents=False,
    )
    target = policy_benchmark_task_path(config)
    _write_immutable_file(target, source_bytes)
    source_snapshot = load_policy_evaluation_snapshot(config)
    snapshot = freeze_policy_evaluation_snapshot(config, source_snapshot)
    if isinstance(snapshot, PolicyEvaluationSnapshot):
        materialize_vllm_lora_adapter(config, snapshot)
    write_policy_evaluation_identity(config, snapshot)
    print(
        json.dumps(
            {
                "total": len(tasks),
                "single_image": sum(task.single_image for task in tasks),
                "multi_image": sum(not task.single_image for task in tasks),
                "image_content_scan_reused_from_arm_materialization": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _validate(config_path: Path) -> None:
    _run_checked(
        [
            sys.executable,
            str(RUNNER),
            "--config",
            str(config_path),
            "--mode",
            "validate",
            "--world-size",
            "4",
        ]
    )


def _launch_workers(config_path: Path) -> list[subprocess.Popen[bytes]]:
    config = load_policy_coredev_config(config_path)
    processes: list[subprocess.Popen[bytes]] = []
    for rank, gpu_id in enumerate(config.gpu_ids):
        environment = dict(os.environ)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        dependency_root = Path(sys.prefix).parent
        header_root = dependency_root / ".deps/python312-dev/root/usr/include"
        python_headers = header_root / "python3.12"
        required_headers = (
            python_headers / "Python.h",
            python_headers / "pyconfig.h",
            header_root / "x86_64-linux-gnu/python3.12/pyconfig.h",
        )
        if any(not path.is_file() for path in required_headers):
            raise RuntimeError("policy evaluator Python development headers are absent")
        triton_cache = config.output_root / "runtime/cache/triton" / f"rank-{rank}"
        inductor_cache = (
            config.output_root / "runtime/cache/torchinductor" / f"rank-{rank}"
        )
        triton_cache.mkdir(parents=True, exist_ok=True)
        inductor_cache.mkdir(parents=True, exist_ok=True)
        environment.update(
            {
                "VLLM_USE_V1": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONHASHSEED": "42",
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "CC": "/usr/bin/gcc",
                "CXX": "/usr/bin/g++",
                "CPATH": os.pathsep.join((str(header_root), str(python_headers))),
                "LIBRARY_PATH": str(Path(sys.prefix) / "lib"),
                "TRITON_CACHE_DIR": str(triton_cache),
                "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache),
            }
        )
        command = [
            sys.executable,
            str(RUNNER),
            "--config",
            str(config_path),
            "--mode",
            "worker",
            "--rank",
            str(rank),
            "--world-size",
            "4",
        ]
        # vLLM starts EngineCore/resource-tracker descendants.  Give each rank
        # its own process group so a failed sibling cannot leave those children
        # holding the supervisor's stdout pipe and all remaining GPUs forever.
        processes.append(
            subprocess.Popen(command, env=environment, start_new_session=True)
        )
    return processes


def _worker_group_exists(process: subprocess.Popen[bytes]) -> bool:
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_worker_groups(
    processes: list[subprocess.Popen[bytes]], *, grace_seconds: float = 30.0
) -> None:
    """Drain every rank process group, including reparented vLLM children."""

    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        for process in processes:
            process.poll()
        if not any(_worker_group_exists(process) for process in processes):
            break
        time.sleep(0.2)
    for process in processes:
        if _worker_group_exists(process):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "policy evaluator process group did not drain"
            ) from error


def _wait_workers(
    processes: list[subprocess.Popen[bytes]], *, owner: str = "paired evaluation"
) -> None:
    failure: tuple[int, int] | None = None
    while any(process.poll() is None for process in processes):
        for index, process in enumerate(processes):
            code = process.poll()
            if code not in {None, 0}:
                failure = (index, code)
                break
        if failure is not None:
            break
        time.sleep(5)
    if failure is not None:
        _terminate_worker_groups(processes)
        raise RuntimeError(f"{owner} worker {failure[0]} exited with {failure[1]}")
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        _terminate_worker_groups(processes)
        raise RuntimeError(f"{owner} workers failed: {codes}")


def _load_judge_config(plan: dict[str, Any]) -> dict[str, Any]:
    scoring = plan["scoring"]
    path = _resolve_repo_path(scoring["judge_config_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("benchmark judge config is not an object")
    model = payload.get("model")
    server = payload.get("server")
    devices = payload.get("devices")
    scope = payload.get("scope")
    if not all(isinstance(value, dict) for value in (model, server, devices, scope)):
        raise RuntimeError("benchmark judge config sections are malformed")
    assert isinstance(model, dict) and isinstance(server, dict)
    assert isinstance(devices, dict) and isinstance(scope, dict)
    if model.get("served_name") != "Qwen2.5-72B-Instruct":
        raise RuntimeError("benchmark judge served model differs")
    if not Path(str(model.get("local_path"))).is_dir():
        raise RuntimeError("benchmark judge local model is unavailable")
    if devices.get("tensor_parallel_size") != 2:
        raise RuntimeError("benchmark judge must remain tensor-parallel two")
    if scope.get("allows_vlmevalkit_benchmark_judging") is not True:
        raise RuntimeError("benchmark judge is not authorized for VLMEvalKit")
    if scope.get("allows_gpt_fallback") is not False:
        raise RuntimeError("GPT fallback must remain disabled")
    return payload


def _judge_command(judge: dict[str, Any]) -> list[str]:
    model = judge["model"]
    server = judge["server"]
    devices = judge["devices"]
    vllm = Path(sys.executable).with_name("vllm")
    command = [
        str(vllm),
        "serve",
        str(model["local_path"]),
        "--served-model-name",
        str(model["served_name"]),
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--dtype",
        str(server["dtype"]),
        "--tensor-parallel-size",
        str(devices["tensor_parallel_size"]),
        "--max-model-len",
        str(server["max_model_len"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--seed",
        str(server["seed"]),
        "--generation-config",
        str(server["generation_config"]),
    ]
    if server.get("prefix_caching") is True:
        command.append("--enable-prefix-caching")
    return command


def _judge_environment(judge: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(value) for value in judge["devices"]["physical"]
    )
    environment["VLLM_ATTENTION_BACKEND"] = str(judge["server"]["attention_backend"])
    runtime = judge.get("runtime", {})
    for source, destination in (("cc", "CC"), ("cxx", "CXX"), ("cpath", "CPATH")):
        value = runtime.get(source)
        if isinstance(value, str) and value:
            environment[destination] = value
    return environment


def _wait_for_judge(
    process: subprocess.Popen[bytes],
    judge: dict[str, Any],
    *,
    timeout_seconds: int,
    poll_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    base_url = str(judge["server"]["base_url"])
    served_name = str(judge["model"]["served_name"])
    last_error: Exception | None = None
    while True:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"benchmark judge exited during startup with {code}")
        try:
            check_qwen25_72b_judge(
                base_url=base_url,
                expected_model=served_name,
                timeout=min(30, max(1, poll_seconds)),
            )
            return
        except Exception as error:  # service is expected to reject until ready
            last_error = error
        if time.monotonic() >= deadline:
            raise TimeoutError("benchmark judge readiness timed out") from last_error
        time.sleep(poll_seconds)


@contextmanager
def _local_judge_service(
    judge: dict[str, Any],
    *,
    log_path: Path,
    timeout_seconds: int,
    poll_seconds: int,
):
    try:
        check_qwen25_72b_judge(
            base_url=str(judge["server"]["base_url"]),
            expected_model=str(judge["model"]["served_name"]),
            timeout=2,
        )
    except Exception:
        pass
    else:
        raise RuntimeError("benchmark judge endpoint is already occupied")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            _judge_command(judge),
            env=_judge_environment(judge),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        _wait_for_judge(
            process,
            judge,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        yield
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def _scoring_root(config_path: Path, plan: dict[str, Any]) -> Path:
    config = load_policy_coredev_config(config_path)
    return config.output_root / "scoring" / str(plan["scoring"]["view_name"])


def _materialize_official_scoring_view(
    config_path: Path, plan: dict[str, Any], *, arm: str
) -> dict[str, Any]:
    config = load_policy_coredev_config(config_path)
    root = _scoring_root(config_path, plan)
    summary_path = root / "materialization-summary.json"
    run_id = _vlmevalkit_scoring_run_id(
        run_id_prefix=plan["scoring"]["run_id_prefix"],
        arm_evaluation_id=config.evaluation_id,
    )
    if summary_path.is_file():
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "evaluation_id": config.evaluation_id,
            "run_id": run_id,
            "observed_single_image_count": 2240,
            "unsupported_multi_image_count": 271,
            "official_row_count": 2511,
        }
        if not isinstance(result, dict) or any(
            result.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError(f"existing {arm} scoring view identity differs")
        return result
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"partial immutable {arm} scoring view exists without its summary"
        )
    return materialize_policy_coredev_scoring_views(
        inference_root=config.output_root / "inference",
        tasks_path=plan["task_manifest_path"],
        source_root=plan["scoring"]["source_root"],
        output_root=root,
        evaluation_id=config.evaluation_id,
        run_id=run_id,
        mathverse_source_json=plan["scoring"]["mathverse_source_json"],
    )


def _load_existing_official_scoring_view(
    config_path: Path, plan: dict[str, Any], *, arm: str
) -> dict[str, Any]:
    """Strictly accept an immutable scoring view without rematerializing it."""

    config = load_policy_coredev_config(config_path)
    root = _scoring_root(config_path, plan)
    summary_path = root / "materialization-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"score mode requires existing {arm} materialization")
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "evaluation_id": config.evaluation_id,
        "observed_single_image_count": 2240,
        "unsupported_multi_image_count": 271,
        "official_row_count": 2511,
    }
    if not isinstance(result, dict) or any(
        result.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(f"existing {arm} scoring view identity differs")
    slices = result.get("slices")
    if not isinstance(slices, list) or [item.get("dataset") for item in slices] != list(
        COREDEV_DATASETS
    ):
        raise RuntimeError(f"existing {arm} scoring slice order differs")
    observed_total = unsupported_total = official_total = 0
    for item in slices:
        dataset = item["dataset"]
        work_dir = (root / dataset).resolve()
        prediction = Path(str(item.get("prediction_file", "")))
        manifest = Path(str(item.get("manifest", "")))
        if (
            Path(str(item.get("work_dir", ""))).resolve() != work_dir
            or not prediction.is_file()
            or not manifest.is_file()
            or prediction.parent.parent.name != EVALUATED_MODEL
        ):
            raise RuntimeError(f"existing {arm} {dataset} materialization differs")
        run_dir = prediction.parent.resolve()
        discoverable = False
        for candidate in prediction.parent.parent.iterdir():
            try:
                validate_vlmevalkit_eval_id(candidate.name)
            except ValueError:
                continue
            if candidate.is_dir() and candidate.resolve() == run_dir:
                discoverable = (candidate / "status.json").is_file()
                if discoverable:
                    break
        if not discoverable:
            raise RuntimeError(
                f"existing {arm} {dataset} prediction is not discoverable by VLMEvalKit"
            )
        counts = tuple(
            item.get(field)
            for field in (
                "observed_single_image_count",
                "unsupported_multi_image_count",
                "official_row_count",
            )
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise RuntimeError(f"existing {arm} {dataset} counts are invalid")
        observed_total += counts[0]
        unsupported_total += counts[1]
        official_total += counts[2]
    if (observed_total, unsupported_total, official_total) != (2240, 271, 2511):
        raise RuntimeError(f"existing {arm} scoring coverage differs")
    return result


def _official_score_command(
    *, dataset: str, scoring_root: Path, judge: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    scoring = plan["scoring"]
    return [
        sys.executable,
        str(COREDEV_RUNNER),
        "--data",
        dataset,
        "--model",
        EVALUATED_MODEL,
        "--work-dir",
        str(scoring_root / dataset),
        "--mode",
        "eval",
        "--reuse",
        "--reuse-aux",
        "infer",
        "--judge",
        str(judge["model"]["served_name"]),
        "--judge-base-url",
        str(judge["server"]["base_url"]),
        "--judge-key",
        "EMPTY",
        "--judge-api-nproc",
        str(scoring["judge_api_nproc"]),
        "--judge-retry",
        str(scoring["judge_retry"]),
        "--judge-timeout",
        str(scoring["judge_timeout_seconds"]),
    ]


def _accepted_official_summary(
    scoring_root: Path, judge: dict[str, Any]
) -> dict[str, Any] | None:
    path = scoring_root / "coredev-2511-eval-summary.json"
    if not path.is_file():
        return None
    result = summarize_coredev_results(
        work_dir=scoring_root.resolve(),
        repository_root=REPOSITORY_ROOT,
        phase="eval",
        expected_judge_base_url=str(judge["server"]["base_url"]),
        expected_model=EVALUATED_MODEL,
    )
    if result.get("status") != "pass" or result.get("sample_count") != 2511:
        raise RuntimeError("existing official CoreDev summary is not accepted")
    return result


def _score_arm(
    config_path: Path,
    plan: dict[str, Any],
    judge: dict[str, Any],
    *,
    arm: str,
    log_root: Path,
) -> dict[str, Any]:
    scoring_root = _scoring_root(config_path, plan)
    accepted = _accepted_official_summary(scoring_root, judge)
    if accepted is not None:
        return accepted
    workers = _launch_score_arm(
        config_path,
        plan,
        judge,
        arm=arm,
        log_root=log_root,
    )
    failures = [dataset for dataset, process in workers if process.wait() != 0]
    if failures:
        raise RuntimeError(f"{arm} official scorers failed after drain: {failures}")
    return _summarize_scored_arm(config_path, plan, judge)


def _launch_score_arm(
    config_path: Path,
    plan: dict[str, Any],
    judge: dict[str, Any],
    *,
    arm: str,
    log_root: Path,
) -> list[tuple[str, subprocess.Popen[bytes]]]:
    scoring_root = _scoring_root(config_path, plan)
    workers: list[tuple[str, subprocess.Popen[bytes]]] = []
    for dataset in COREDEV_DATASETS:
        log_path = log_root / f"score-{arm}-{dataset}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log_handle:
            environment = dict(os.environ)
            environment["OPENAI_API_KEY"] = "EMPTY"
            workers.append(
                (
                    dataset,
                    subprocess.Popen(
                        _official_score_command(
                            dataset=dataset,
                            scoring_root=scoring_root,
                            judge=judge,
                            plan=plan,
                        ),
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    ),
                )
            )
    return workers


def _summarize_scored_arm(
    config_path: Path,
    plan: dict[str, Any],
    judge: dict[str, Any],
) -> dict[str, Any]:
    scoring_root = _scoring_root(config_path, plan)
    result = summarize_coredev_results(
        work_dir=scoring_root.resolve(),
        repository_root=REPOSITORY_ROOT,
        phase="eval",
        expected_judge_base_url=str(judge["server"]["base_url"]),
        expected_model=EVALUATED_MODEL,
    )
    write_json_atomic(scoring_root / "coredev-2511-eval-summary.json", result)
    return result


def _score_missing_arms(
    configs: dict[str, Path],
    existing: dict[str, dict[str, Any] | None],
    plan: dict[str, Any],
    judge: dict[str, Any],
    *,
    log_root: Path,
) -> dict[str, dict[str, Any] | None]:
    """Run every missing arm concurrently and drain all scorers before failure."""

    launched: list[tuple[str, str, subprocess.Popen[bytes]]] = []
    arm_names = [arm["name"] for arm in plan.get("arms", ())] or list(configs)
    pending = [arm for arm in arm_names if existing[arm] is None]
    for arm in pending:
        launched.extend(
            (arm, dataset, process)
            for dataset, process in _launch_score_arm(
                configs[arm], plan, judge, arm=arm, log_root=log_root
            )
        )
    failures: list[str] = []
    arm_failed: set[str] = set()
    for arm, dataset, process in launched:
        code = process.wait()
        if code != 0:
            arm_failed.add(arm)
            failures.append(f"{arm}/{dataset}={code}")
    for arm in pending:
        if arm in arm_failed:
            continue
        try:
            existing[arm] = _summarize_scored_arm(configs[arm], plan, judge)
        except Exception as error:
            failures.append(f"{arm}/summary={error}")
    if failures:
        raise RuntimeError(
            "official scorers failed after all workers drained: " + ", ".join(failures)
        )
    return existing


def _write_evaluation_complete(
    output_base: Path, report_path: Path, *, evaluation_id: str
) -> dict[str, str]:
    """Atomically publish completion only after the paired summary is durable."""

    record = {
        "schema_version": "tgvf.paired-coredev-evaluation-complete.v1",
        "status": "complete",
        "evaluation_id": evaluation_id,
        "paired_summary_path": str(report_path.resolve()),
        "paired_summary_sha256": _sha256_file(report_path),
    }
    write_json_atomic(output_base / "evaluation-complete", record)
    return record


def _arm_evaluation_identity_sha256(config_path: Path) -> str:
    config = load_policy_coredev_config(config_path)
    identity_path = config.output_root / "runtime/evaluation-identity.json"
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    identity_sha256 = payload.get("identity_sha256")
    if (
        not isinstance(identity_sha256, str)
        or len(identity_sha256) != 64
        or any(character not in "0123456789abcdef" for character in identity_sha256)
    ):
        raise RuntimeError("arm evaluation identity SHA256 is malformed")
    return identity_sha256


def _sampling_report(plan: dict[str, Any], run: Any) -> dict[str, object]:
    sampling = run.policy.sampling
    return {
        "source": "bound_policy_run_config",
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "min_p": sampling.min_p,
        "do_sample": sampling.do_sample,
        "paired_rng": plan.get("paired_rng"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--mode", choices=("prepare", "run", "score", "status"), default="run"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=tuple(range(8)))
    parser.add_argument("--wait-for-step8", action="store_true")
    parser.add_argument("--wait-for-final-arm", action="store_true")
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--gpu-free-threshold-mib", type=int, default=1024)
    parser.add_argument("--judge-startup-timeout-seconds", type=int, default=15 * 60)
    args = parser.parse_args()
    if (
        min(
            args.wait_timeout_seconds,
            args.poll_seconds,
            args.judge_startup_timeout_seconds,
        )
        <= 0
    ):
        raise ValueError("evaluation wait durations must be positive")
    if len(args.gpu_ids) not in {4, 8} or len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("paired evaluator requires four or eight distinct GPU IDs")
    if args.wait_for_step8 and args.wait_for_final_arm:
        raise ValueError("select only one checkpoint-wait mode")
    if args.mode == "score" and (args.wait_for_step8 or args.wait_for_final_arm):
        raise ValueError("score mode cannot wait on a training checkpoint")
    plan = _load_plan(args.plan.resolve())
    policy_config = _resolve_repo_path(plan["policy_config"])
    run = load_policy_e2e_smoke_run_config(
        policy_config, allow_external_agent_loop_config=True
    )
    _validate_plan_run(plan, run)
    judge = _load_judge_config(plan)
    judge_gpus = tuple(judge["devices"]["physical"])
    if (
        len(judge_gpus) != 2
        or len(set(judge_gpus)) != 2
        or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in judge_gpus)
    ):
        raise RuntimeError("pinned benchmark judge GPU binding is malformed")
    if any(gpu_id not in args.gpu_ids for gpu_id in judge_gpus):
        raise ValueError("evaluation GPU set must include pinned judge GPUs")
    if args.wait_for_step8:
        _wait_for_step8(
            run,
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    if args.wait_for_final_arm:
        final_step = max(arm["optimizer_step"] for arm in plan["arms"])
        if final_step <= 0:
            raise ValueError("final-arm wait requires a positive checkpoint arm")
        _wait_for_optimizer_step(
            run,
            optimizer_step=final_step,
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    if args.wait_for_gpus:
        _wait_for_gpus(
            tuple(args.gpu_ids),
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
            free_threshold_mib=args.gpu_free_threshold_mib,
        )
    output_base = (
        args.output_root.resolve()
        if args.output_root is not None
        else run.output.root / "evaluation" / plan["evaluation_id"]
    )
    gpu_groups = [tuple(args.gpu_ids[:4])]
    if len(args.gpu_ids) == 8:
        gpu_groups.append(tuple(args.gpu_ids[4:]))
    plan_arms = plan.get("arms") or (
        {"name": "step0", "optimizer_step": 0},
        {"name": "step8", "optimizer_step": 8},
    )
    arms = tuple((arm["name"], arm["optimizer_step"]) for arm in plan_arms)
    if args.mode == "score":
        configs = {
            arm: _load_existing_arm(
                plan=plan, output_base=output_base, arm=arm, step=step
            )
            for arm, step in arms
        }
    else:
        configs = {
            arm: _materialize_arm(
                plan=plan,
                run=run,
                arm=arm,
                step=step,
                output_base=output_base,
                gpu_ids=gpu_groups[index % len(gpu_groups)],
            )
            for index, (arm, step) in enumerate(arms)
        }
    _validate_materialized_frozen_pairing(plan, configs)
    if args.mode == "status":
        for config in configs.values():
            _run_checked(
                [
                    sys.executable,
                    str(RUNNER),
                    "--config",
                    str(config),
                    "--mode",
                    "status",
                ]
            )
        return 0
    if args.mode == "score":
        for arm, _step in arms:
            _materialize_official_scoring_view(configs[arm], plan, arm=arm)
        materialization = {
            arm: _load_existing_official_scoring_view(configs[arm], plan, arm=arm)
            for arm, _step in arms
        }
    else:
        for config in configs.values():
            _prepare(config)
        for config in configs.values():
            _validate(config)
        if args.mode == "prepare":
            return 0
        if len(args.gpu_ids) == 8:
            for offset in range(0, len(arms), 2):
                batch = arms[offset : offset + 2]
                processes = [
                    process
                    for arm, _step in batch
                    for process in _launch_workers(configs[arm])
                ]
                _wait_workers(
                    processes,
                    owner=" + ".join(arm for arm, _step in batch),
                )
        else:
            for arm, _step in arms:
                _wait_workers(_launch_workers(configs[arm]), owner=arm)
        for config in configs.values():
            _run_checked(
                [
                    sys.executable,
                    str(RUNNER),
                    "--config",
                    str(config),
                    "--mode",
                    "status",
                    "--world-size",
                    "4",
                ]
            )
        materialization = {
            arm: _materialize_official_scoring_view(configs[arm], plan, arm=arm)
            for arm, _step in arms
        }
    log_root = output_base / "logs"
    existing = {
        arm: _accepted_official_summary(_scoring_root(configs[arm], plan), judge)
        for arm, _step in arms
    }
    if any(value is None for value in existing.values()):
        with _local_judge_service(
            judge,
            log_path=log_root / "judge-qwen25-72b.log",
            timeout_seconds=args.judge_startup_timeout_seconds,
            poll_seconds=max(5, min(args.poll_seconds, 30)),
        ):
            existing = _score_missing_arms(
                configs,
                existing,
                plan,
                judge,
                log_root=log_root,
            )
    report = {
        "schema_version": PAIR_SUMMARY_SCHEMA,
        "evaluation_id": plan["evaluation_id"],
        "coverage": {
            "official_manifest_rows": 2511,
            "evaluated_single_image_rows": 2240,
            "held_multi_image_rows": 271,
            "multi_image_policy": "unsupported_explicit_hold",
        },
        "materialization": materialization,
        "sampling": _sampling_report(plan, run),
        "arms": {
            arm: {
                "optimizer_step": step,
                "evaluation_identity_sha256": _arm_evaluation_identity_sha256(
                    configs[arm]
                ),
                "official_summary": existing[arm],
            }
            for arm, step in arms
        },
    }
    for arm, _step in arms:
        report[arm] = existing[arm]
    report_path = output_base / "paired-summary.json"
    write_json_atomic(report_path, report)
    _write_evaluation_complete(
        output_base, report_path, evaluation_id=plan["evaluation_id"]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
