#!/usr/bin/env python3
"""Wait for, run, resume, and officially score ordered policy checkpoints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_benchmark_config import (  # noqa: E402
    materialize_full_model_policy_benchmark_config,
    materialize_paired_tgvf_policy_benchmark_config,
)
from tgvf_rl.evaluation.coredev_results import (  # noqa: E402
    check_qwen25_72b_judge,
    extract_coredev_macro_star,
    summarize_coredev_results,
    write_json_atomic,
)
from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    PolicyEvaluationSnapshot,
    freeze_policy_evaluation_snapshot,
    load_benchmark_tasks,
    load_policy_coredev_config,
    load_policy_evaluation_snapshot,
    materialize_vllm_lora_adapter,
    paired_evaluation_rng_contract,
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
from tgvf_rl.evaluation.policy_full_model_snapshot import (  # noqa: E402
    FULL_MODEL_EVALUATION_BACKEND,
    FULL_MODEL_EVALUATION_IMAGE_MAX_PIXELS,
    FullModelCheckpointOwner,
    FullModelEvaluationSnapshot,
    build_full_model_snapshot_manifest,
    load_full_model_evaluation_snapshot,
    load_full_model_snapshot_manifest,
    materialize_full_model_snapshot,
    write_full_model_materialization_receipt,
    write_full_model_snapshot_manifest,
)
from tgvf_rl.evaluation.policy_paired_tgvf_snapshot import (  # noqa: E402
    PAIRED_TGVF_EVALUATION_BACKEND,
)
from tgvf_rl.policy.crop_tfree_contract import (  # noqa: E402
    CropTFreeRunContract,
    load_crop_tfree_run_contract,
)
from tgvf_rl.policy.deepeyes_native_contract import (  # noqa: E402
    DeepEyesNativeRunContract,
    load_deepeyes_native_run_contract,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    POLICY_E2E_CROP_TFREE_EXACT_MATCHED_RUN_CONFIG_SCHEMA,
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)


DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs/evaluation/prl15_r1_rp66_step0_step8_coredev2511_plan.json"
)
RUNNER = REPOSITORY_ROOT / "tools/run_policy_benchmark.py"
COREDEV_RUNNER = REPOSITORY_ROOT / "tools/run_coredev_2511_vlmevalkit.py"
COREDEV_PINNED_ARTIFACTS = (
    REPOSITORY_ROOT / "configs/evaluation/coredev_2511_vlmevalkit_v1.json"
)
VLMEVALKIT_DEPLOYMENT = (
    REPOSITORY_ROOT / "configs/evaluation/vlmevalkit_deployment_v1.json"
)
JUDGE_SERVICE_CONFIG = (
    REPOSITORY_ROOT / "configs/evaluation/qwen25_72b_judge_service_v1.json"
)
PLAN_SCHEMA_V2 = "tgvf.prl15-paired-policy-benchmark-plan.v2"
PLAN_SCHEMA_V3 = "tgvf.paired-policy-benchmark-plan.v3"
# Historical tests and downstream imports use this name for the v2 ABI.
PLAN_SCHEMA = PLAN_SCHEMA_V2
PAIRED_TGVF_BACKEND = "paired_tgvf"
FULL_MODEL_BACKEND = "full_model"
PAIR_SUMMARY_SCHEMA = "tgvf.prl15-paired-coredev-summary.v1"
GENERIC_PAIR_SUMMARY_SCHEMA = "tgvf.paired-coredev-summary.v2"
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
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCORING_SLICE_COUNTS = {
    "VStarBench": (191, 0, 191),
    "HRBench4K": (200, 0, 200),
    "BLINK": (180, 240, 420),
    "OCRBench_v2": (600, 0, 600),
    "MMMU_Pro_10c": (269, 31, 300),
    "MathVista_MINI": (300, 0, 300),
    "MathVerse_MINI": (500, 0, 500),
}
_ACTIVE_PROCESS_GROUPS: dict[int, subprocess.Popen[bytes]] = {}
_EVALUATION_LOCK_HANDLES: list[Any] = []
_CHILD_SPAWN_SIGNALS = frozenset(
    candidate
    for name in ("SIGINT", "SIGTERM", "SIGHUP")
    if (candidate := getattr(signal, name, None)) is not None
)
_SPAWN_CRITICAL_SECTION_ACTIVE = False
_DEFERRED_INTERRUPT_SIGNAL: int | None = None


class _EvaluationInterrupted(BaseException):
    """Controlled process-lifecycle interruption that broad Exception cannot eat."""


def _controlled_interrupt(signum: int, _frame: object) -> None:
    """Defer termination only across the Popen-to-registration critical section."""

    global _DEFERRED_INTERRUPT_SIGNAL
    if _SPAWN_CRITICAL_SECTION_ACTIVE:
        if _DEFERRED_INTERRUPT_SIGNAL is None:
            _DEFERRED_INTERRUPT_SIGNAL = signum
        return
    raise _EvaluationInterrupted(f"paired evaluator received signal {signum}")


def _spawn_registered_process(
    command: list[str], **kwargs: Any
) -> subprocess.Popen[bytes]:
    """Create and register one process group without a signal race window.

    Blocking signals around ``Popen`` is unsafe here because the child inherits
    the parent's thread signal mask across fork/exec.  Instead the installed
    handler records an interrupt during this tiny critical section.  The child
    is registered first, then the deferred interrupt is raised so normal
    process-group cleanup can always find it.
    """

    global _DEFERRED_INTERRUPT_SIGNAL, _SPAWN_CRITICAL_SECTION_ACTIVE
    if "start_new_session" in kwargs:
        raise ValueError("registered process groups own start_new_session")
    if _SPAWN_CRITICAL_SECTION_ACTIVE:
        raise RuntimeError("nested registered-process spawn is unsupported")
    process: subprocess.Popen[bytes] | None = None
    spawn_error: BaseException | None = None
    _SPAWN_CRITICAL_SECTION_ACTIVE = True
    try:
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        _ACTIVE_PROCESS_GROUPS[process.pid] = process
    except BaseException as error:
        spawn_error = error
    finally:
        _SPAWN_CRITICAL_SECTION_ACTIVE = False
    deferred_signal = _DEFERRED_INTERRUPT_SIGNAL
    _DEFERRED_INTERRUPT_SIGNAL = None
    if deferred_signal is not None:
        raise _EvaluationInterrupted(
            f"paired evaluator received signal {deferred_signal} during child spawn"
        ) from spawn_error
    if spawn_error is not None:
        raise spawn_error
    assert process is not None
    return process


class _EvaluationRuntime:
    __slots__ = (
        "backend",
        "checkpoint_owner",
        "protocol_contract",
        "output_root",
        "checkpoint_world_size",
    )

    def __init__(
        self,
        *,
        backend: str,
        checkpoint_owner: object,
        protocol_contract: object,
        output_root: Path,
        checkpoint_world_size: int,
    ) -> None:
        self.backend = backend
        self.checkpoint_owner = checkpoint_owner
        self.protocol_contract = protocol_contract
        self.output_root = output_root
        self.checkpoint_world_size = checkpoint_world_size


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


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _arm_evaluation_id(plan: dict[str, Any], arm_name: str) -> str:
    arm = next(
        (item for item in plan.get("arms", ()) if item["name"] == arm_name), None
    )
    explicit = None if arm is None else arm.get("evaluation_id")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError("arm evaluation_id must be non-empty text")
        return explicit
    return f"{plan['evaluation_id']}-{arm_name.upper()}"


def _validate_v3_static_plan(payload: dict[str, Any]) -> None:
    required_top_level = {
        "schema_version",
        "evaluation_id",
        "status",
        "checkpoint_owner",
        "protocol_contract",
        "snapshot",
        "task_manifest_path",
        "task_manifest_sha256",
        "expected_task_count",
        "expected_single_image_count",
        "unsupported_multi_image_count",
        "paired_rng",
        "arms",
        "protocol",
        "scoring",
    }
    if set(payload) != required_top_level or payload.get("status") != "ready":
        raise ValueError("v3 paired evaluation plan fields/status differ")

    owner = payload.get("checkpoint_owner")
    owner_fields = {
        "contract_type",
        "config_path",
        "config_sha256",
        "run_id",
        "run_identity_sha256",
        "output_root",
        "checkpoint_world_size",
        "completion_path",
        "completion_sha256",
    }
    if not isinstance(owner, dict) or set(owner) != owner_fields:
        raise ValueError("v3 checkpoint owner fields differ")
    if owner["contract_type"] not in {
        "crop_tfree_run_contract_v1",
        "policy_e2e_crop_exact_run_config_v1",
    }:
        raise ValueError("v3 checkpoint owner contract type differs")
    _require_sha256(owner["config_sha256"], name="checkpoint owner config")
    _require_sha256(owner["run_identity_sha256"], name="checkpoint owner identity")
    _require_sha256(owner["completion_sha256"], name="checkpoint owner completion")
    if not isinstance(owner["run_id"], str) or not owner["run_id"]:
        raise ValueError("v3 checkpoint owner run ID must be non-empty")
    owner_root = Path(str(owner["output_root"]))
    if not owner_root.is_absolute():
        raise ValueError("v3 checkpoint owner output root must be absolute")
    completion_path = Path(str(owner["completion_path"]))
    if not completion_path.is_absolute() or not completion_path.is_relative_to(
        owner_root
    ):
        raise ValueError("v3 checkpoint owner completion path differs")
    if (
        type(owner["checkpoint_world_size"]) is not int
        or owner["checkpoint_world_size"] <= 0
    ):
        raise ValueError("v3 checkpoint owner world size must be positive")

    protocol = payload.get("protocol_contract")
    protocol_fields = {
        "contract_type",
        "config_path",
        "config_sha256",
        "run_id",
        "run_identity_sha256",
    }
    if not isinstance(protocol, dict) or set(protocol) != protocol_fields:
        raise ValueError("v3 protocol contract fields differ")
    if protocol["contract_type"] != "deepeyes_native_crop_v1":
        raise ValueError("v3 protocol contract type differs")
    _require_sha256(protocol["config_sha256"], name="protocol contract config")
    _require_sha256(protocol["run_identity_sha256"], name="protocol contract identity")
    if not isinstance(protocol["run_id"], str) or not protocol["run_id"]:
        raise ValueError("v3 protocol contract run ID must be non-empty")

    snapshot = payload.get("snapshot")
    snapshot_fields = {
        "backend",
        "inference_concurrency_per_gpu",
        "max_model_len",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "gpu_memory_utilization",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != snapshot_fields:
        raise ValueError("v3 full-model snapshot fields differ")
    if snapshot["backend"] != FULL_MODEL_BACKEND:
        raise ValueError("v3 snapshot backend must be full_model")
    paired_rng = payload["paired_rng"]
    if not isinstance(paired_rng, dict):
        raise ValueError("v3 full-model plan requires paired RNG")
    _require_sha256(
        paired_rng.get("protocol_sha256"),
        name="v3 paired RNG protocol identity",
    )
    for field in (
        "inference_concurrency_per_gpu",
        "max_model_len",
        "max_num_batched_tokens",
    ):
        if type(snapshot[field]) is not int or snapshot[field] <= 0:
            raise ValueError(f"v3 snapshot {field} must be positive")
    if type(snapshot["enable_chunked_prefill"]) is not bool:
        raise ValueError("v3 chunked-prefill flag must be boolean")
    utilization = snapshot["gpu_memory_utilization"]
    if (
        isinstance(utilization, bool)
        or not isinstance(utilization, (int, float))
        or not 0.0 < float(utilization) < 1.0
    ):
        raise ValueError("v3 GPU memory utilization must lie in (0,1)")

    for arm in payload["arms"]:
        if not isinstance(arm, dict) or set(arm) != {
            "name",
            "optimizer_step",
            "source",
            "evaluation_id",
        }:
            raise ValueError("v3 full-model arm fields differ")
        if not isinstance(arm["evaluation_id"], str) or not arm["evaluation_id"]:
            raise ValueError("v3 full-model arm evaluation ID must be non-empty")
        source = arm["source"]
        if not isinstance(source, dict) or source.get("kind") not in {
            "protocol_base_model",
            "owner_checkpoint",
        }:
            raise ValueError("v3 full-model arm source kind differs")
        if source["kind"] == "protocol_base_model":
            if set(source) != {"kind"} or arm["optimizer_step"] != 0:
                raise ValueError("v3 base-model source is valid only for step0")
        else:
            if set(source) != {"kind", "relative_path"}:
                raise ValueError("v3 owner-checkpoint source fields differ")
            relative = Path(str(source["relative_path"]))
            if (
                arm["optimizer_step"] <= 0
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.name != f"global_step_{arm['optimizer_step']}"
            ):
                raise ValueError("v3 owner-checkpoint source path is unsafe")

    scoring = payload["scoring"]
    execution = scoring.get("execution")
    if execution != {"mode": "eval", "reuse": True, "reuse_aux": "infer"}:
        raise ValueError(
            "v3 score-only contract must be mode=eval,reuse=true,reuse_aux=infer"
        )


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        PLAN_SCHEMA_V2,
        PLAN_SCHEMA_V3,
    }:
        raise ValueError("PRL15 paired evaluation plan schema differs")
    if payload["schema_version"] == PLAN_SCHEMA_V3:
        _validate_v3_static_plan(payload)
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
            arm_evaluation_id=_arm_evaluation_id(payload, arm["name"]),
        )
        for arm in payload["arms"]
    }
    if len(generated_run_ids) != len(payload["arms"]):
        raise ValueError("PRL15 VLMEvalKit arm run IDs must be distinct")
    arm_evaluation_ids = tuple(
        _arm_evaluation_id(payload, arm["name"]) for arm in payload["arms"]
    )
    if len(set(arm_evaluation_ids)) != len(arm_evaluation_ids):
        raise ValueError("paired evaluation arm IDs must be distinct")
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
    plan_paths = [("task_manifest_path", "task_manifest_sha256")]
    if payload["schema_version"] == PLAN_SCHEMA_V2:
        # The policy config is live repository metadata, not the evaluated
        # checkpoint identity.  Requiring byte-for-byte equality here made
        # harmless provenance-only edits (for example, advancing code.commit)
        # reject an otherwise valid post-training handoff.  Its evaluation
        # semantics are checked below by _validate_plan_run, while checkpoint
        # and frozen-adapter identities are validated from their own receipts.
        # Keep the recorded digest for provenance, but do not use it as a gate.
        policy_config = _resolve_repo_path(payload["policy_config"])
        if not policy_config.is_file():
            raise RuntimeError("PRL15 plan policy_config is unavailable")
    else:
        for section in ("checkpoint_owner", "protocol_contract"):
            resolved = _resolve_repo_path(payload[section]["config_path"])
            if (
                not resolved.is_file()
                or _sha256_file(resolved) != payload[section]["config_sha256"]
            ):
                raise RuntimeError(f"v3 {section} config identity differs")
        completion = Path(payload["checkpoint_owner"]["completion_path"])
        if (
            completion.is_symlink()
            or not completion.is_file()
            or _sha256_file(completion)
            != payload["checkpoint_owner"]["completion_sha256"]
        ):
            raise RuntimeError("v3 checkpoint owner completion identity differs")
    for path_field, digest_field in plan_paths:
        resolved = _resolve_repo_path(payload[path_field])
        if not resolved.is_file() or _sha256_file(resolved) != payload[digest_field]:
            raise RuntimeError(f"PRL15 plan {path_field} identity differs")
    judge_path = _resolve_repo_path(scoring["judge_config_path"])
    if (
        judge_path != JUDGE_SERVICE_CONFIG.resolve()
        or not judge_path.is_file()
        or _sha256_file(judge_path) != scoring["judge_config_sha256"]
    ):
        raise RuntimeError("PRL15 benchmark judge config identity differs")
    for path_field, digest_field, canonical in (
        (
            "pinned_artifacts_config_path",
            "pinned_artifacts_config_sha256",
            COREDEV_PINNED_ARTIFACTS,
        ),
        (
            "vlmevalkit_deployment_config_path",
            "vlmevalkit_deployment_config_sha256",
            VLMEVALKIT_DEPLOYMENT,
        ),
        ("mathverse_source_json", "mathverse_source_sha256", None),
    ):
        resolved = _resolve_repo_path(scoring[path_field])
        if (
            (canonical is not None and resolved != canonical.resolve())
            or not resolved.is_file()
            or _sha256_file(resolved) != scoring[digest_field]
        ):
            raise RuntimeError(f"PRL15 scoring {path_field} identity differs")
    pinned = json.loads(COREDEV_PINNED_ARTIFACTS.read_text(encoding="utf-8"))
    if source_root.resolve() != Path(str(pinned.get("artifact_root", ""))).resolve():
        raise RuntimeError("PRL15 CoreDev source root differs from pinned runner")
    return payload


def _validate_plan_run(plan: dict[str, Any], run: Any) -> None:
    if plan["schema_version"] != PLAN_SCHEMA_V2:
        raise ValueError("training-run validator accepts only v2 TGVF plans")
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


def _validate_v3_runtime(
    plan: dict[str, Any],
    owner: CropTFreeRunContract,
    protocol: DeepEyesNativeRunContract,
) -> None:
    owner_plan = plan["checkpoint_owner"]
    protocol_plan = plan["protocol_contract"]
    if (
        owner.source_sha256 != owner_plan["config_sha256"]
        or owner.run_id != owner_plan["run_id"]
        or owner.identity_sha256 != owner_plan["run_identity_sha256"]
        or owner.output_root.resolve() != Path(owner_plan["output_root"]).resolve()
    ):
        raise RuntimeError("v3 checkpoint owner identity differs")
    owner_world_size = owner.payload["matched_training"]["world_size"]
    if owner_world_size != owner_plan["checkpoint_world_size"]:
        raise RuntimeError("v3 checkpoint owner world size differs")
    completion_path = Path(owner_plan["completion_path"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if not isinstance(completion, dict):
        raise RuntimeError("v3 checkpoint owner completion is malformed")
    if (
        completion.get("schema_version") != "tgvf.prl21-crop-tfree-launch-provenance.v1"
        or completion.get("status") != "target_checkpoint_complete"
        or completion.get("run_id") != owner.run_id
        or completion.get("overlay_config_sha256") != owner.source_sha256
        or completion.get("overlay_identity_sha256") != owner.identity_sha256
        or completion.get("base_contract_sha256") != protocol.source_sha256
        or completion.get("contract_sha256") != protocol.identity_sha256
    ):
        raise RuntimeError("v3 checkpoint owner completion contract differs")
    if (
        protocol.source_sha256 != protocol_plan["config_sha256"]
        or protocol.run_id != protocol_plan["run_id"]
        or protocol.identity_sha256 != protocol_plan["run_identity_sha256"]
    ):
        raise RuntimeError("v3 protocol contract identity differs")
    if (
        owner.base_contract.source_sha256 != protocol.source_sha256
        or owner.base_contract.run_id != protocol.run_id
        or owner.base_contract.identity_sha256 != protocol.identity_sha256
    ):
        raise RuntimeError(
            "v3 checkpoint owner does not inherit the bound protocol contract"
        )

    protocol_payload = protocol.payload["protocol"]
    expected_protocol = {
        "evaluation_protocol": DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
        "visual_prompt_bundle_sha256": protocol_payload["visual_prompt_bundle_sha256"],
        "tool_name": protocol_payload["tool_name"],
        "tool_parser": protocol_payload["tool_parser"],
        "maximum_tool_calls": protocol_payload["max_active_perception"],
        "native_pixels": True,
        "sampling_source": "bound_protocol_contract",
        "same_tasks_and_rank_partition": True,
    }
    if plan.get("protocol") != expected_protocol:
        raise RuntimeError("v3 protocol contract differs from its native Crop run")
    if protocol.payload["model"]["native_pixels"] is not True:
        raise RuntimeError("v3 native Crop protocol no longer uses native pixels")
    if (
        protocol.payload["rollout"]["temperature"] != 1.0
        or protocol.payload["dataset"]["schedule_seed"] != 42
    ):
        raise RuntimeError("v3 native Crop sampling differs from temp1/seed42")
    paired_rng = plan["paired_rng"]
    if (
        paired_rng["master_seed"] != protocol.payload["dataset"]["schedule_seed"]
        or paired_rng["temperature"] != protocol.payload["rollout"]["temperature"]
        or paired_rng["do_sample"] is not True
        or paired_rng["task_manifest_sha256"] != plan["task_manifest_sha256"]
    ):
        raise RuntimeError("v3 paired RNG task/seed identity differs")
    protocol_probe_step = next(
        (arm["optimizer_step"] for arm in plan["arms"] if arm["optimizer_step"] > 0),
        None,
    )
    if protocol_probe_step is not None:
        protocol_probe = SimpleNamespace(
            run=SimpleNamespace(
                model=SimpleNamespace(model_name=protocol.payload["model"]["name"]),
                policy=SimpleNamespace(
                    image_max_pixels=FULL_MODEL_EVALUATION_IMAGE_MAX_PIXELS,
                    sampling=SimpleNamespace(
                        temperature=float(protocol.payload["rollout"]["temperature"]),
                        do_sample=True,
                    ),
                ),
                rollout_rng=SimpleNamespace(
                    master_seed=int(protocol.payload["dataset"]["schedule_seed"])
                ),
            ),
            policy_version=SimpleNamespace(optimizer_step=protocol_probe_step),
        )
        protocol_probe_config = SimpleNamespace(
            evaluation_protocol=DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
            paired_seed_namespace=paired_rng["seed_namespace"],
        )
        _validate_runtime_paired_rng(
            plan,
            protocol_probe_config,
            protocol_probe,
        )

    configured_steps = tuple(owner.payload["evaluation"]["checkpoint_steps"])
    requested_steps = tuple(
        arm["optimizer_step"]
        for arm in plan["arms"]
        if arm["source"]["kind"] == "owner_checkpoint"
    )
    if any(step not in configured_steps for step in requested_steps):
        raise RuntimeError("v3 evaluation arm is not retained by checkpoint owner")
    for arm in plan["arms"]:
        step = arm["optimizer_step"]
        source = arm["source"]
        if source["kind"] != "owner_checkpoint":
            continue
        expected = (owner.output_root / Path(source["relative_path"])).resolve()
        if (
            Path(str(completion.get(f"retained_step{step}_checkpoint", ""))).resolve()
            != expected
        ):
            raise RuntimeError(
                f"v3 checkpoint owner completion does not retain step {step}"
            )


def _policy_checkpoint_receipt(
    owner: PolicyE2ESmokeRunConfig,
    *,
    checkpoint: Path,
    optimizer_step: int,
) -> tuple[dict[str, Any], Path]:
    """Validate one permanent Policy-E2E checkpoint against its run config."""

    from tgvf_rl.framework.verl.checkpoint_bridge import (
        read_committed_policy_checkpoint_pair,
    )
    from tgvf_rl.framework.verl.compatibility import FSDP2BridgeConfig

    receipt_path = checkpoint / "tgvf_permanent_checkpoint_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RuntimeError(f"step{optimizer_step} permanent receipt is unavailable")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"step{optimizer_step} permanent receipt is malformed"
        ) from error
    if not isinstance(receipt, dict):
        raise RuntimeError(f"step{optimizer_step} permanent receipt is malformed")

    state, pair = read_committed_policy_checkpoint_pair(
        checkpoint / "actor",
        fsdp2=FSDP2BridgeConfig(
            world_size=owner.distributed.world_size,
            fsdp_size=owner.distributed.world_size,
        ),
    )
    expected_hashes = {
        "run_config": owner.identity_sha256,
        "run_config_file": owner.source_sha256,
        "dataset_content": owner.dataset.runtime_binding.content_sha256,
        "dataset_samples": owner.dataset.samples_sha256,
        "dataset_iteration": owner.dataset.iteration_identity_sha256,
        "prompt": owner.protocol.prompt_sha256,
        "tool_schema": owner.protocol.tool_schema_sha256,
        "representation_artifact_file": owner.representation.artifact_file_sha256,
    }
    observed_hashes = {item.name: item.sha256 for item in state.run_identity.hashes}
    checkpoint_run_identity = hashlib.sha256(
        _canonical_json_bytes(state.run_identity.to_checkpoint_mapping())
    ).hexdigest()
    if (
        state.run_identity.run_id != owner.run_id
        or any(
            observed_hashes.get(name) != expected
            for name, expected in expected_hashes.items()
        )
        or state.progress.optimizer_step != optimizer_step
        or pair.optimizer_step != optimizer_step
        or receipt.get("schema_version")
        != "tgvf.prl15-permanent-checkpoint-receipt.v1"
        or receipt.get("optimizer_step") != optimizer_step
        or receipt.get("run_identity_sha256") != checkpoint_run_identity
        or receipt.get("project_state_sha256") != state.integrity_sha256
        or receipt.get("pair_integrity_sha256") != pair.integrity_sha256
    ):
        raise RuntimeError(f"step{optimizer_step} permanent checkpoint identity differs")

    actor = checkpoint / "actor"
    fsdp_config = actor / "fsdp_config.json"
    try:
        fsdp_payload = json.loads(fsdp_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"step{optimizer_step} FSDP config is malformed") from error
    if fsdp_payload != {
        "FSDP_version": 2,
        "world_size": owner.distributed.world_size,
    }:
        raise RuntimeError(f"step{optimizer_step} FSDP world size differs")
    required = [checkpoint / "data.pt", fsdp_config]
    for prefix in ("model", "optim", "extra_state"):
        required.extend(
            actor
            / f"{prefix}_world_size_{owner.distributed.world_size}_rank_{rank}.pt"
            for rank in range(owner.distributed.world_size)
        )
    if any(
        path.is_symlink() or not path.is_file() or path.stat().st_size <= 0
        for path in required
    ):
        raise RuntimeError(f"step{optimizer_step} permanent checkpoint is incomplete")
    return receipt, receipt_path.resolve()


def _validate_v3_policy_run_runtime(
    plan: dict[str, Any],
    owner: PolicyE2ESmokeRunConfig,
    protocol: DeepEyesNativeRunContract,
) -> None:
    """Bind an exact-Crop Policy-E2E owner to the native Crop eval protocol."""

    owner_plan = plan["checkpoint_owner"]
    protocol_plan = plan["protocol_contract"]
    if (
        owner.schema_version
        != POLICY_E2E_CROP_TFREE_EXACT_MATCHED_RUN_CONFIG_SCHEMA
        or owner.source_sha256 != owner_plan["config_sha256"]
        or owner.run_id != owner_plan["run_id"]
        or owner.identity_sha256 != owner_plan["run_identity_sha256"]
        or owner.output.root.resolve() != Path(owner_plan["output_root"]).resolve()
        or owner.distributed.world_size != owner_plan["checkpoint_world_size"]
    ):
        raise RuntimeError("v3 Policy-E2E checkpoint owner identity differs")
    if (
        protocol.source_sha256 != protocol_plan["config_sha256"]
        or protocol.run_id != protocol_plan["run_id"]
        or protocol.identity_sha256 != protocol_plan["run_identity_sha256"]
    ):
        raise RuntimeError("v3 protocol contract identity differs")

    protocol_payload = protocol.payload["protocol"]
    expected_protocol = {
        "evaluation_protocol": DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
        "visual_prompt_bundle_sha256": protocol_payload["visual_prompt_bundle_sha256"],
        "tool_name": protocol_payload["tool_name"],
        "tool_parser": protocol_payload["tool_parser"],
        "maximum_tool_calls": protocol_payload["max_active_perception"],
        "native_pixels": True,
        "sampling_source": "bound_protocol_contract",
        "same_tasks_and_rank_partition": True,
    }
    sampling = owner.policy.sampling
    if (
        plan.get("protocol") != expected_protocol
        or protocol.payload["model"]["native_pixels"] is not True
        or owner.model.model_name != protocol.payload["model"]["name"]
        or Path(owner.model.revision_or_path).resolve()
        != Path(protocol.payload["model"]["path"]).resolve()
        or owner.protocol.prompt_sha256
        != protocol_payload["visual_prompt_bundle_sha256"]
        or owner.protocol.tool_profile.value != "crop_only"
        or owner.protocol.enabled_tool_names != (protocol_payload["tool_name"],)
        or owner.protocol.maximum_tool_calls != protocol_payload["max_active_perception"]
        or owner.policy.image_max_pixels != FULL_MODEL_EVALUATION_IMAGE_MAX_PIXELS
        or sampling.temperature != 1.0
        or sampling.do_sample is not True
        or owner.rollout_rng.master_seed != 42
    ):
        raise RuntimeError("v3 Policy-E2E owner differs from native Crop protocol")

    paired_rng = plan["paired_rng"]
    if (
        protocol.payload["rollout"]["temperature"] != 1.0
        or protocol.payload["dataset"]["schedule_seed"] != 42
        or paired_rng["master_seed"] != owner.rollout_rng.master_seed
        or paired_rng["temperature"] != sampling.temperature
        or paired_rng["do_sample"] is not sampling.do_sample
        or paired_rng["task_manifest_sha256"] != plan["task_manifest_sha256"]
    ):
        raise RuntimeError("v3 paired RNG task/seed identity differs")
    probe_step = next(
        (arm["optimizer_step"] for arm in plan["arms"] if arm["optimizer_step"] > 0),
        None,
    )
    if probe_step is not None:
        protocol_probe = SimpleNamespace(
            run=SimpleNamespace(
                model=owner.model,
                policy=owner.policy,
                rollout_rng=owner.rollout_rng,
            ),
            policy_version=SimpleNamespace(optimizer_step=probe_step),
        )
        _validate_runtime_paired_rng(
            plan,
            SimpleNamespace(
                evaluation_protocol=DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
                paired_seed_namespace=paired_rng["seed_namespace"],
            ),
            protocol_probe,
        )

    requested = [
        arm for arm in plan["arms"] if arm["source"]["kind"] == "owner_checkpoint"
    ]
    if any(
        arm["optimizer_step"] not in owner.training.permanent_checkpoint_steps
        for arm in requested
    ):
        raise RuntimeError("v3 evaluation arm is not retained by checkpoint owner")
    receipts: dict[int, Path] = {}
    for arm in requested:
        step = arm["optimizer_step"]
        checkpoint = (
            owner.output.root / Path(arm["source"]["relative_path"])
        ).resolve()
        _receipt, receipts[step] = _policy_checkpoint_receipt(
            owner, checkpoint=checkpoint, optimizer_step=step
        )
    if requested:
        final_step = max(receipts)
        if Path(owner_plan["completion_path"]).resolve() != receipts[final_step]:
            raise RuntimeError(
                "v3 Policy-E2E completion evidence is not the final arm receipt"
            )
    metrics = [
        json.loads(line)
        for line in owner.output.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    required_step = max(receipts, default=0)
    if [row.get("optimizer_step") for row in metrics[:required_step]] != list(
        range(1, required_step + 1)
    ):
        raise RuntimeError("v3 Policy-E2E metrics are incomplete")


def _load_evaluation_runtime(plan: dict[str, Any]) -> _EvaluationRuntime:
    if plan["schema_version"] == PLAN_SCHEMA_V2:
        policy_config = _resolve_repo_path(plan["policy_config"])
        run = load_policy_e2e_smoke_run_config(
            policy_config, allow_external_agent_loop_config=True
        )
        _validate_plan_run(plan, run)
        return _EvaluationRuntime(
            backend=PAIRED_TGVF_BACKEND,
            checkpoint_owner=run,
            protocol_contract=run,
            output_root=run.output.root.resolve(),
            checkpoint_world_size=run.distributed.world_size,
        )

    owner_path = _resolve_repo_path(plan["checkpoint_owner"]["config_path"])
    protocol_path = _resolve_repo_path(plan["protocol_contract"]["config_path"])
    protocol = load_deepeyes_native_run_contract(protocol_path)
    owner_type = plan["checkpoint_owner"]["contract_type"]
    if owner_type == "crop_tfree_run_contract_v1":
        owner = load_crop_tfree_run_contract(
            owner_path,
            repository_root=REPOSITORY_ROOT,
            allow_placeholder=False,
        )
        _validate_v3_runtime(plan, owner, protocol)
        output_root = owner.output_root.resolve()
    elif owner_type == "policy_e2e_crop_exact_run_config_v1":
        owner = load_policy_e2e_smoke_run_config(
            owner_path, allow_external_agent_loop_config=True
        )
        _validate_v3_policy_run_runtime(plan, owner, protocol)
        output_root = owner.output.root.resolve()
    else:  # pragma: no cover - static plan validation rejects this first
        raise ValueError("v3 checkpoint owner contract type differs")
    return _EvaluationRuntime(
        backend=FULL_MODEL_BACKEND,
        checkpoint_owner=owner,
        protocol_contract=protocol,
        output_root=output_root,
        checkpoint_world_size=plan["checkpoint_owner"]["checkpoint_world_size"],
    )


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


def _full_model_checkpoint_structurally_complete(
    source: Path, *, step: int, world_size: int
) -> bool:
    if step == 0:
        return source.is_dir() and (source / "config.json").is_file()
    actor = source / "actor"
    fsdp_config = actor / "fsdp_config.json"
    if (
        not source.is_dir()
        or not actor.is_dir()
        or not (source / "data.pt").is_file()
        or not fsdp_config.is_file()
    ):
        return False
    try:
        payload = json.loads(fsdp_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload != {"FSDP_version": 2, "world_size": world_size}:
        return False
    required = [source / "data.pt", fsdp_config]
    for prefix in ("model", "optim", "extra_state"):
        required.extend(
            actor / f"{prefix}_world_size_{world_size}_rank_{rank}.pt"
            for rank in range(world_size)
        )
    huggingface = actor / "huggingface"
    weights = tuple(huggingface.glob("*.safetensors")) + tuple(
        huggingface.glob("*.bin")
    )
    required.extend((huggingface / "config.json", *weights))
    return bool(weights) and all(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0
        for path in required
    )


def _wait_for_full_model_arm(
    plan: dict[str, Any],
    runtime: _EvaluationRuntime,
    *,
    arm: str,
    optimizer_step: int,
    timeout_seconds: int,
    poll_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    source = _full_model_source_path(plan, runtime, arm=arm, step=optimizer_step)
    while not _full_model_checkpoint_structurally_complete(
        source,
        step=optimizer_step,
        world_size=runtime.checkpoint_world_size,
    ):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for complete {arm} full-model checkpoint"
            )
        time.sleep(poll_seconds)


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
        "full_model_snapshot": root / "runtime/full-model-snapshot.json",
        "full_model_receipt": root / "runtime/full-model-materialization.json",
        "full_model_merge": root / "runtime/full-model-hf",
    }


def _full_model_source_path(
    plan: dict[str, Any],
    runtime: _EvaluationRuntime,
    *,
    arm: str,
    step: int,
) -> Path:
    arm_plan = next(item for item in plan["arms"] if item["name"] == arm)
    source = arm_plan["source"]
    if source["kind"] == "protocol_base_model":
        if step != 0 or not isinstance(
            runtime.protocol_contract, DeepEyesNativeRunContract
        ):
            raise RuntimeError("full-model base source identity differs")
        return Path(runtime.protocol_contract.payload["model"]["path"]).resolve()
    relative = Path(source["relative_path"])
    candidate = (runtime.output_root / relative).resolve()
    if not candidate.is_relative_to(runtime.output_root.resolve()):
        raise RuntimeError("full-model checkpoint source escapes its owner root")
    if candidate.name != f"global_step_{step}":
        raise RuntimeError("full-model checkpoint source step differs")
    return candidate


def _full_model_checkpoint_owner(plan: dict[str, Any]) -> FullModelCheckpointOwner:
    owner = plan["checkpoint_owner"]
    return FullModelCheckpointOwner(
        run_id=owner["run_id"],
        run_identity_sha256=owner["run_identity_sha256"],
        config_path=str(_resolve_repo_path(owner["config_path"])),
        config_file_sha256=owner["config_sha256"],
        completion_path=owner["completion_path"],
        completion_file_sha256=owner["completion_sha256"],
    )


def _expected_arm_runtime_settings(
    plan: dict[str, Any], runtime: _EvaluationRuntime | None
) -> dict[str, object]:
    if runtime is not None and runtime.backend == FULL_MODEL_BACKEND:
        snapshot = plan["snapshot"]
        return {
            "snapshot_backend": FULL_MODEL_EVALUATION_BACKEND,
            "inference_concurrency_per_gpu": snapshot["inference_concurrency_per_gpu"],
            "max_model_len": snapshot["max_model_len"],
            "max_num_batched_tokens": snapshot["max_num_batched_tokens"],
            "enable_chunked_prefill": snapshot["enable_chunked_prefill"],
            "gpu_memory_utilization": snapshot["gpu_memory_utilization"],
        }
    return {
        "snapshot_backend": PAIRED_TGVF_EVALUATION_BACKEND,
        "inference_concurrency_per_gpu": 8,
        "max_model_len": 32768,
        "max_num_batched_tokens": 32768,
        "enable_chunked_prefill": False,
        "gpu_memory_utilization": 0.9,
    }


def _validate_existing_arm_config(
    config: Any,
    *,
    plan: dict[str, Any],
    runtime: _EvaluationRuntime | None,
    paths: dict[str, Path],
    arm: str,
    gpu_ids: tuple[int, int, int, int] | None,
) -> None:
    expected: dict[str, object] = {
        **_expected_arm_runtime_settings(plan, runtime),
        "evaluation_id": _arm_evaluation_id(plan, arm),
        "evaluation_protocol": plan["protocol"]["evaluation_protocol"],
        "output_root": paths["root"].resolve(),
        "task_manifest_path": _resolve_repo_path(plan["task_manifest_path"]),
        "task_manifest_sha256": plan["task_manifest_sha256"],
        "expected_task_count": plan["expected_task_count"],
        "expected_single_image_count": plan["expected_single_image_count"],
    }
    if gpu_ids is not None:
        expected["gpu_ids"] = gpu_ids
    observed = {
        field: (value.resolve() if isinstance(value, Path) else value)
        for field in expected
        if (value := getattr(config, field, None)) is not None
    }
    differing = [
        field for field, value in expected.items() if observed.get(field) != value
    ]
    if differing:
        raise RuntimeError(
            f"existing {arm} evaluator capacity/identity differs: "
            + ", ".join(sorted(differing))
        )
    if config.paired_seed_namespace != _paired_seed_namespace(plan):
        raise RuntimeError(f"existing {arm} paired_seed_namespace differs from plan")


def _validate_runtime_paired_rng(
    plan: dict[str, Any], config: Any, snapshot: Any
) -> None:
    planned = plan.get("paired_rng")
    observed = paired_evaluation_rng_contract(
        config,
        snapshot,
        task_manifest_sha256=plan["task_manifest_sha256"],
    )
    if planned is None:
        if observed is not None:
            raise RuntimeError("unplanned paired RNG contract was materialized")
        return
    if observed is None:
        raise RuntimeError("planned paired RNG contract was not materialized")
    for field in (
        "mode",
        "seed_namespace",
        "master_seed",
        "task_manifest_sha256",
        "protocol_sha256",
        "excluded_arm_components",
    ):
        if observed.get(field) != planned.get(field):
            raise RuntimeError(f"paired RNG runtime {field} differs from plan")
    sampling = snapshot.run.policy.sampling
    temperature = float(sampling.temperature)
    do_sample = getattr(sampling, "do_sample", temperature > 0.0)
    if temperature != planned["temperature"] or do_sample is not planned["do_sample"]:
        raise RuntimeError("paired RNG runtime sampling differs from plan")


def _materialize_full_model_arm(
    *,
    plan: dict[str, Any],
    runtime: _EvaluationRuntime,
    arm: str,
    step: int,
    output_base: Path,
    gpu_ids: tuple[int, int, int, int],
) -> Path:
    if not isinstance(runtime.protocol_contract, DeepEyesNativeRunContract):
        raise TypeError("full-model backend requires a native Crop protocol contract")
    paths = _arm_paths(output_base, arm)
    source = _full_model_source_path(plan, runtime, arm=arm, step=step)
    checkpoint_owner = _full_model_checkpoint_owner(plan)
    complete = (
        paths["config"].is_file()
        and paths["full_model_snapshot"].is_file()
        and paths["full_model_receipt"].is_file()
    )
    if complete:
        config = load_policy_coredev_config(paths["config"])
        _validate_existing_arm_config(
            config,
            plan=plan,
            runtime=runtime,
            paths=paths,
            arm=arm,
            gpu_ids=gpu_ids,
        )
        snapshot = load_full_model_evaluation_snapshot(
            paths["full_model_snapshot"],
            paths["full_model_receipt"],
            runtime_lightweight=True,
        )
        _validate_runtime_paired_rng(plan, config, snapshot)
        if (
            config.evaluation_id != _arm_evaluation_id(plan, arm)
            or snapshot.policy_version.optimizer_step != step
            or snapshot.manifest.checkpoint_owner != checkpoint_owner
            or Path(snapshot.manifest.source_path).resolve() != source
            or snapshot.manifest.run_id != plan["protocol_contract"]["run_id"]
            or snapshot.manifest.run_identity_sha256
            != plan["protocol_contract"]["run_identity_sha256"]
        ):
            raise RuntimeError(f"existing {arm} full-model identity differs")
        return paths["config"]
    manifest = build_full_model_snapshot_manifest(
        runtime.protocol_contract,
        source_path=source,
        optimizer_step=step,
        runtime_fsdp_world_size=(runtime.checkpoint_world_size if step > 0 else None),
        checkpoint_owner=checkpoint_owner,
    )
    if paths["full_model_snapshot"].is_file():
        if load_full_model_snapshot_manifest(paths["full_model_snapshot"]) != manifest:
            raise RuntimeError(f"partial {arm} full-model snapshot differs")
    else:
        write_full_model_snapshot_manifest(paths["full_model_snapshot"], manifest)
    if paths["full_model_receipt"].is_file():
        snapshot = load_full_model_evaluation_snapshot(
            paths["full_model_snapshot"],
            paths["full_model_receipt"],
            runtime_lightweight=True,
        )
        receipt = snapshot.receipt
    else:
        receipt = materialize_full_model_snapshot(
            manifest,
            target_dir=paths["full_model_merge"],
        )
        write_full_model_materialization_receipt(paths["full_model_receipt"], receipt)
    snapshot_plan = plan["snapshot"]
    materialize_full_model_policy_benchmark_config(
        evaluation_id=_arm_evaluation_id(plan, arm),
        policy_config_path=_resolve_repo_path(plan["checkpoint_owner"]["config_path"]),
        snapshot_manifest_path=paths["full_model_snapshot"],
        materialization_receipt_path=paths["full_model_receipt"],
        expected_optimizer_step=step,
        task_manifest_path=plan["task_manifest_path"],
        expected_task_count=plan["expected_task_count"],
        expected_single_image_count=plan["expected_single_image_count"],
        output_root=paths["root"],
        config_path=paths["config"],
        gpu_ids=gpu_ids,
        inference_concurrency_per_gpu=snapshot_plan["inference_concurrency_per_gpu"],
        max_model_len=snapshot_plan["max_model_len"],
        max_num_batched_tokens=snapshot_plan["max_num_batched_tokens"],
        enable_chunked_prefill=snapshot_plan["enable_chunked_prefill"],
        gpu_memory_utilization=snapshot_plan["gpu_memory_utilization"],
        paired_seed_namespace=_paired_seed_namespace(plan),
    )
    config = load_policy_coredev_config(paths["config"])
    snapshot = load_full_model_evaluation_snapshot(
        paths["full_model_snapshot"],
        paths["full_model_receipt"],
        runtime_lightweight=True,
    )
    _validate_runtime_paired_rng(plan, config, snapshot)
    return paths["config"]


def _materialize_arm(
    *,
    plan: dict[str, Any],
    runtime: _EvaluationRuntime | None = None,
    run: Any,
    arm: str,
    step: int,
    output_base: Path,
    gpu_ids: tuple[int, int, int, int],
) -> Path:
    if runtime is not None and runtime.backend == FULL_MODEL_BACKEND:
        return _materialize_full_model_arm(
            plan=plan,
            runtime=runtime,
            arm=arm,
            step=step,
            output_base=output_base,
            gpu_ids=gpu_ids,
        )
    paths = _arm_paths(output_base, arm)
    if paths["config"].is_file() and paths["receipt"].is_file():
        config = load_policy_coredev_config(paths["config"])
        _validate_existing_arm_config(
            config,
            plan=plan,
            runtime=runtime,
            paths=paths,
            arm=arm,
            gpu_ids=gpu_ids,
        )
        snapshot = load_policy_evaluation_snapshot(config)
        _validate_runtime_paired_rng(plan, config, snapshot)
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
        evaluation_id=_arm_evaluation_id(plan, arm),
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
    runtime: _EvaluationRuntime | None = None,
    output_base: Path,
    arm: str,
    step: int,
    gpu_ids: tuple[int, int, int, int] | None = None,
) -> Path:
    """Load one completed arm without preparing or mutating it."""

    paths = _arm_paths(output_base, arm)
    required_receipt = (
        paths["full_model_receipt"]
        if runtime is not None and runtime.backend == FULL_MODEL_BACKEND
        else paths["receipt"]
    )
    if not paths["config"].is_file() or not required_receipt.is_file():
        raise FileNotFoundError(
            f"score mode requires existing {arm} config and receipt"
        )
    config = load_policy_coredev_config(paths["config"])
    _validate_existing_arm_config(
        config,
        plan=plan,
        runtime=runtime,
        paths=paths,
        arm=arm,
        gpu_ids=gpu_ids,
    )
    snapshot = load_policy_evaluation_snapshot(config)
    _validate_runtime_paired_rng(plan, config, snapshot)
    if snapshot.policy_version.optimizer_step != step:
        raise RuntimeError(f"existing {arm} optimizer step differs from plan")
    if runtime is not None and runtime.backend == FULL_MODEL_BACKEND:
        if not paths["full_model_snapshot"].is_file():
            raise RuntimeError(f"score mode requires complete {arm} snapshot manifest")
        if not isinstance(snapshot, FullModelEvaluationSnapshot):
            raise RuntimeError(f"existing {arm} snapshot backend differs")
        full_snapshot = snapshot
        source = _full_model_source_path(plan, runtime, arm=arm, step=step)
        manifest = full_snapshot.manifest
        if (
            manifest.checkpoint_owner != _full_model_checkpoint_owner(plan)
            or Path(manifest.source_path).resolve() != source
            or manifest.run_id != plan["protocol_contract"]["run_id"]
            or manifest.run_identity_sha256
            != plan["protocol_contract"]["run_identity_sha256"]
        ):
            raise RuntimeError(f"existing {arm} full-model snapshot binding differs")
    for rank in range(len(config.gpu_ids)):
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
    try:
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
                raise RuntimeError(
                    "policy evaluator Python development headers are absent"
                )
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
            # vLLM starts EngineCore/resource-tracker descendants. Give each rank
            # its own process group and register it before another launch.
            process = _spawn_registered_process(command, env=environment)
            processes.append(process)
    except BaseException:
        if processes:
            _terminate_worker_groups(processes)
        raise
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
        finally:
            _ACTIVE_PROCESS_GROUPS.pop(process.pid, None)


def _wait_workers(
    processes: list[subprocess.Popen[bytes]], *, owner: str = "paired evaluation"
) -> None:
    try:
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
            raise RuntimeError(f"{owner} worker {failure[0]} exited with {failure[1]}")
        codes = [process.wait() for process in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"{owner} workers failed: {codes}")
    except BaseException:
        _terminate_worker_groups(processes)
        raise
    if any(_worker_group_exists(process) for process in processes):
        _terminate_worker_groups(processes)
    else:
        for process in processes:
            _ACTIVE_PROCESS_GROUPS.pop(process.pid, None)


def _load_judge_config(
    plan: dict[str, Any], *, require_local_model: bool = True
) -> dict[str, Any]:
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
    if require_local_model and not Path(str(model.get("local_path"))).is_dir():
        raise RuntimeError("benchmark judge local model is unavailable")
    if devices.get("tensor_parallel_size") != 2:
        raise RuntimeError("benchmark judge must remain tensor-parallel two")
    if scope.get("allows_vlmevalkit_benchmark_judging") is not True:
        raise RuntimeError("benchmark judge is not authorized for VLMEvalKit")
    if scope.get("allows_gpt_fallback") is not False:
        raise RuntimeError("GPT fallback must remain disabled")
    return payload


def _require_local_judge_runtime(judge: dict[str, Any]) -> None:
    if not Path(str(judge["model"].get("local_path"))).is_dir():
        raise RuntimeError("benchmark judge local model is unavailable")


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
        process = _spawn_registered_process(
            _judge_command(judge),
            env=_judge_environment(judge),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
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
        if _worker_group_exists(process):
            _terminate_worker_groups([process], grace_seconds=60.0)
        else:
            process.wait()
            _ACTIVE_PROCESS_GROUPS.pop(process.pid, None)


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
        return _load_existing_official_scoring_view(config_path, plan, arm=arm)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"partial immutable {arm} scoring view exists without its summary"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    try:
        materialize_policy_coredev_scoring_views(
            inference_root=config.output_root / "inference",
            tasks_path=plan["task_manifest_path"],
            source_root=plan["scoring"]["source_root"],
            output_root=staging,
            logical_output_root=root,
            evaluation_id=config.evaluation_id,
            run_id=run_id,
            mathverse_source_json=plan["scoring"]["mathverse_source_json"],
        )
        staging.rename(root)
        directory = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _load_existing_official_scoring_view(config_path, plan, arm=arm)


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
    run_id = _vlmevalkit_scoring_run_id(
        run_id_prefix=plan["scoring"]["run_id_prefix"],
        arm_evaluation_id=config.evaluation_id,
    )
    expected = {
        "schema_version": "tgvf-policy-coredev-scoring-view-v1",
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
    slices = result.get("slices")
    if not isinstance(slices, list) or [item.get("dataset") for item in slices] != list(
        COREDEV_DATASETS
    ):
        raise RuntimeError(f"existing {arm} scoring slice order differs")
    observed_total = unsupported_total = official_total = 0
    for item in slices:
        dataset = item["dataset"]
        work_dir = (root / dataset).resolve()
        run_dir = work_dir / EVALUATED_MODEL / run_id
        prediction = run_dir / f"{EVALUATED_MODEL}_{dataset}.tsv"
        manifest = run_dir / "final-answer-view-manifest.json"
        expected_slice_fields = {
            "dataset",
            "official_row_count",
            "observed_single_image_count",
            "sample_local_failure_count",
            "unsupported_multi_image_count",
            "work_dir",
            "prediction_file",
            "manifest",
        }
        expected_counts = _SCORING_SLICE_COUNTS[dataset]
        if (
            set(item) != expected_slice_fields
            or Path(str(item.get("work_dir", ""))).resolve() != work_dir
            or Path(str(item.get("prediction_file", ""))).resolve()
            != prediction.resolve()
            or Path(str(item.get("manifest", ""))).resolve() != manifest.resolve()
            or prediction.is_symlink()
            or not prediction.is_file()
            or manifest.is_symlink()
            or not manifest.is_file()
            or not prediction.resolve().is_relative_to(work_dir)
            or not manifest.resolve().is_relative_to(work_dir)
        ):
            raise RuntimeError(f"existing {arm} {dataset} materialization differs")
        counts = (
            item.get("observed_single_image_count"),
            item.get("unsupported_multi_image_count"),
            item.get("official_row_count"),
        )
        if counts != expected_counts:
            raise RuntimeError(f"existing {arm} {dataset} fixed counts differ")
        sample_failures = item.get("sample_local_failure_count")
        if (
            type(sample_failures) is not int
            or not 0 <= sample_failures <= expected_counts[0]
        ):
            raise RuntimeError(f"existing {arm} {dataset} failure count differs")

        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        source_record = manifest_payload.get("source")
        derived_record = manifest_payload.get("derived")
        verification = manifest_payload.get("verification")
        raw_path = (root / "raw" / f"{dataset}.tsv").resolve()
        if not all(
            isinstance(value, dict)
            for value in (source_record, derived_record, verification)
        ):
            raise RuntimeError(f"existing {arm} {dataset} view manifest is malformed")
        assert isinstance(source_record, dict)
        assert isinstance(derived_record, dict)
        assert isinstance(verification, dict)
        if (
            Path(str(source_record.get("path", ""))).resolve() != raw_path
            or Path(str(derived_record.get("path", ""))).resolve()
            != prediction.resolve()
            or not raw_path.is_file()
            or source_record.get("sha256") != _sha256_file(raw_path)
            or derived_record.get("sha256") != _sha256_file(prediction)
            or source_record.get("row_count") != expected_counts[2]
            or derived_record.get("row_count") != expected_counts[2]
            or manifest_payload.get("counts", {}).get("row_count") != expected_counts[2]
            or any(
                verification.get(field) is not True
                for field in (
                    "index_order_and_values_identical",
                    "non_prediction_source_fields_verified",
                    "unchanged_non_prediction_source_fields_identical",
                )
            )
        ):
            raise RuntimeError(f"existing {arm} {dataset} view proof differs")
        materializer_output = run_dir / "materializer-output.json"
        status_path = run_dir / "status.json"
        if (
            materializer_output.is_symlink()
            or not materializer_output.is_file()
            or json.loads(materializer_output.read_text(encoding="utf-8"))
            != manifest_payload
            or status_path.is_symlink()
            or not status_path.is_file()
        ):
            raise RuntimeError(f"existing {arm} {dataset} view records differ")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status_entry = status.get("datasets", {}).get(dataset, {})
        if (
            status.get("eval_id") != run_id
            or status.get("mode") != "infer"
            or status.get("reuse_aux") != "infer"
            or Path(str(status_entry.get("prediction_file", ""))).resolve()
            != prediction.resolve()
        ):
            raise RuntimeError(f"existing {arm} {dataset} status identity differs")

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
        observed_total += counts[0]
        unsupported_total += counts[1]
        official_total += counts[2]
    if (observed_total, unsupported_total, official_total) != (2240, 271, 2511):
        raise RuntimeError(f"existing {arm} scoring coverage differs")
    return result


def _official_score_command(
    *,
    dataset: str,
    scoring_root: Path,
    source_run_id: str,
    judge: dict[str, Any],
    plan: dict[str, Any],
) -> list[str]:
    scoring = plan["scoring"]
    source_manifest = (
        scoring_root
        / dataset
        / EVALUATED_MODEL
        / source_run_id
        / "final-answer-view-manifest.json"
    )
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
        "--tgvf-reuse-source-run-id",
        source_run_id,
        "--tgvf-reuse-manifest",
        str(source_manifest),
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


def _status_prediction_path(value: object, *, run_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("pinned scorer status has no prediction_file")
    path = Path(value)
    return (path if path.is_absolute() else run_dir / path).resolve()


def _load_pinned_scoring_receipts(
    scoring_root: Path,
    *,
    source_run_id: str,
    source_evaluation_id: str,
) -> dict[str, str]:
    """Revalidate exact scorer destinations rather than selecting latest T*."""

    validate_vlmevalkit_eval_id(source_run_id)
    expected_fields = {
        "schema_version",
        "dataset",
        "model",
        "source_evaluation_id",
        "source_run_id",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_prediction_path",
        "source_prediction_sha256",
        "destination_run_id",
        "destination_status_path",
        "destination_status_sha256",
        "destination_prediction_path",
        "destination_prediction_sha256",
    }
    eval_ids: dict[str, str] = {}
    for dataset in COREDEV_DATASETS:
        dataset_root = (scoring_root / dataset).resolve()
        source_run_dir = dataset_root / EVALUATED_MODEL / source_run_id
        source_manifest = source_run_dir / "final-answer-view-manifest.json"
        source_prediction = source_run_dir / f"{EVALUATED_MODEL}_{dataset}.tsv"
        source_status_path = source_run_dir / "status.json"
        receipt_path = dataset_root / "pinned-reuse-receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise RuntimeError(f"missing pinned scorer receipt for {dataset}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not isinstance(receipt, dict)
            or set(receipt) != expected_fields
            or receipt.get("schema_version")
            != "tgvf.vlmevalkit-pinned-reuse-receipt.v1"
            or receipt.get("dataset") != dataset
            or receipt.get("model") != EVALUATED_MODEL
            or receipt.get("source_evaluation_id") != source_evaluation_id
            or receipt.get("source_run_id") != source_run_id
            or Path(str(receipt.get("source_manifest_path", ""))).resolve()
            != source_manifest.resolve()
            or Path(str(receipt.get("source_prediction_path", ""))).resolve()
            != source_prediction.resolve()
        ):
            raise RuntimeError(f"pinned scorer receipt identity differs for {dataset}")
        source_manifest_sha256 = receipt.get("source_manifest_sha256")
        source_prediction_sha256 = receipt.get("source_prediction_sha256")
        if (
            source_manifest.is_symlink()
            or not source_manifest.is_file()
            or source_prediction.is_symlink()
            or not source_prediction.is_file()
            or source_status_path.is_symlink()
            or not source_status_path.is_file()
            or not isinstance(source_manifest_sha256, str)
            or _SHA256.fullmatch(source_manifest_sha256) is None
            or not isinstance(source_prediction_sha256, str)
            or _SHA256.fullmatch(source_prediction_sha256) is None
            or _sha256_file(source_manifest) != source_manifest_sha256
            or _sha256_file(source_prediction) != source_prediction_sha256
        ):
            raise RuntimeError(f"pinned scorer source bytes differ for {dataset}")
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        derived = manifest.get("derived") if isinstance(manifest, dict) else None
        source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
        source_entry = (
            source_status.get("datasets", {}).get(dataset)
            if isinstance(source_status, dict)
            else None
        )
        if (
            not isinstance(derived, dict)
            or derived.get("sha256") != source_prediction_sha256
            or Path(str(derived.get("path", ""))).resolve()
            != source_prediction.resolve()
            or not isinstance(source_entry, dict)
            or source_status.get("eval_id") != source_run_id
            or source_status.get("mode") != "infer"
            or source_status.get("reuse_aux") != "infer"
            or source_entry.get("status") != "done"
            or source_entry.get("source_run") != source_evaluation_id
            or _status_prediction_path(
                source_entry.get("prediction_file"), run_dir=source_run_dir
            )
            != source_prediction.resolve()
        ):
            raise RuntimeError(f"pinned scorer source proof differs for {dataset}")

        destination_run_id = receipt.get("destination_run_id")
        validate_vlmevalkit_eval_id(destination_run_id)
        destination_run_dir = dataset_root / EVALUATED_MODEL / destination_run_id
        destination_status = destination_run_dir / "status.json"
        destination_prediction = (
            destination_run_dir / f"{EVALUATED_MODEL}_{dataset}.tsv"
        )
        if (
            Path(str(receipt.get("destination_status_path", ""))).resolve()
            != destination_status.resolve()
            or Path(str(receipt.get("destination_prediction_path", ""))).resolve()
            != destination_prediction.resolve()
            or destination_run_dir.is_symlink()
            or not destination_run_dir.is_dir()
            or destination_status.is_symlink()
            or not destination_status.is_file()
            or destination_prediction.is_symlink()
            or not destination_prediction.is_file()
            or receipt.get("destination_status_sha256")
            != _sha256_file(destination_status)
            or receipt.get("destination_prediction_sha256") != source_prediction_sha256
            or _sha256_file(destination_prediction) != source_prediction_sha256
        ):
            raise RuntimeError(f"pinned scorer destination bytes differ for {dataset}")
        status = json.loads(destination_status.read_text(encoding="utf-8"))
        entry = (
            status.get("datasets", {}).get(dataset)
            if isinstance(status, dict)
            else None
        )
        if (
            not isinstance(entry, dict)
            or status.get("eval_id") != destination_run_id
            or status.get("mode") != "eval"
            or status.get("reuse") is not True
            or status.get("reuse_aux") != "infer"
            or entry.get("status") != "done"
            or entry.get("source_run") != source_run_id
            or _status_prediction_path(
                entry.get("prediction_file"), run_dir=destination_run_dir
            )
            != destination_prediction.resolve()
        ):
            raise RuntimeError(f"pinned scorer destination proof differs for {dataset}")
        eval_ids[dataset] = destination_run_id
    return eval_ids


def _accepted_official_summary(
    scoring_root: Path,
    judge: dict[str, Any],
    *,
    include_headline: bool = False,
    source_run_id: str | None = None,
    source_evaluation_id: str | None = None,
) -> dict[str, Any] | None:
    path = scoring_root / "coredev-2511-eval-summary.json"
    if not path.is_file():
        return None
    expected_eval_ids = None
    if (source_run_id is None) != (source_evaluation_id is None):
        raise ValueError("pinned scorer summary requires both source identities")
    if source_run_id is not None:
        assert source_evaluation_id is not None
        expected_eval_ids = _load_pinned_scoring_receipts(
            scoring_root,
            source_run_id=source_run_id,
            source_evaluation_id=source_evaluation_id,
        )
    result = summarize_coredev_results(
        work_dir=scoring_root.resolve(),
        repository_root=REPOSITORY_ROOT,
        phase="eval",
        expected_judge_base_url=str(judge["server"]["base_url"]),
        expected_model=EVALUATED_MODEL,
        expected_eval_ids=expected_eval_ids,
    )
    if result.get("status") != "pass" or result.get("sample_count") != 2511:
        raise RuntimeError("existing official CoreDev summary is not accepted")
    if include_headline:
        result["headline"] = extract_coredev_macro_star(result)
    return result


def _score_arm(
    config_path: Path,
    plan: dict[str, Any],
    judge: dict[str, Any],
    *,
    arm: str,
    log_root: Path,
) -> dict[str, Any]:
    accepted = _accepted_scored_arm(config_path, plan, judge)
    if accepted is not None:
        return accepted
    workers = _launch_score_arm(
        config_path,
        plan,
        judge,
        arm=arm,
        log_root=log_root,
    )
    try:
        failures = [dataset for dataset, process in workers if process.wait() != 0]
    except BaseException:
        _terminate_worker_groups([process for _dataset, process in workers])
        raise
    residual = [
        process for _dataset, process in workers if _worker_group_exists(process)
    ]
    if residual:
        _terminate_worker_groups(residual)
    for _dataset, process in workers:
        _ACTIVE_PROCESS_GROUPS.pop(process.pid, None)
    if failures:
        raise RuntimeError(f"{arm} official scorers failed after drain: {failures}")
    return _summarize_scored_arm(config_path, plan, judge)


def _accepted_scored_arm(
    config_path: Path,
    plan: dict[str, Any],
    judge: dict[str, Any],
) -> dict[str, Any] | None:
    config = load_policy_coredev_config(config_path)
    scoring_root = _scoring_root(config_path, plan)
    source_run_id = _vlmevalkit_scoring_run_id(
        run_id_prefix=plan["scoring"]["run_id_prefix"],
        arm_evaluation_id=config.evaluation_id,
    )
    require_pinned_receipts = plan.get("schema_version") == PLAN_SCHEMA_V3
    return _accepted_official_summary(
        scoring_root,
        judge,
        include_headline=require_pinned_receipts,
        source_run_id=source_run_id if require_pinned_receipts else None,
        source_evaluation_id=config.evaluation_id if require_pinned_receipts else None,
    )


def _launch_score_arm(
    config_path: Path,
    plan: dict[str, Any],
    judge: dict[str, Any],
    *,
    arm: str,
    log_root: Path,
) -> list[tuple[str, subprocess.Popen[bytes]]]:
    config = load_policy_coredev_config(config_path)
    scoring_root = _scoring_root(config_path, plan)
    source_run_id = _vlmevalkit_scoring_run_id(
        run_id_prefix=plan["scoring"]["run_id_prefix"],
        arm_evaluation_id=config.evaluation_id,
    )
    workers: list[tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for dataset in COREDEV_DATASETS:
            log_path = log_root / f"score-{arm}-{dataset}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab", buffering=0) as log_handle:
                environment = dict(os.environ)
                environment["OPENAI_API_KEY"] = "EMPTY"
                process = _spawn_registered_process(
                    _official_score_command(
                        dataset=dataset,
                        scoring_root=scoring_root,
                        source_run_id=source_run_id,
                        judge=judge,
                        plan=plan,
                    ),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                workers.append((dataset, process))
    except BaseException:
        if workers:
            _terminate_worker_groups([process for _dataset, process in workers])
        raise
    return workers


def _summarize_scored_arm(
    config_path: Path,
    plan: dict[str, Any],
    judge: dict[str, Any],
) -> dict[str, Any]:
    config = load_policy_coredev_config(config_path)
    scoring_root = _scoring_root(config_path, plan)
    source_run_id = _vlmevalkit_scoring_run_id(
        run_id_prefix=plan["scoring"]["run_id_prefix"],
        arm_evaluation_id=config.evaluation_id,
    )
    expected_eval_ids = _load_pinned_scoring_receipts(
        scoring_root,
        source_run_id=source_run_id,
        source_evaluation_id=config.evaluation_id,
    )
    result = summarize_coredev_results(
        work_dir=scoring_root.resolve(),
        repository_root=REPOSITORY_ROOT,
        phase="eval",
        expected_judge_base_url=str(judge["server"]["base_url"]),
        expected_model=EVALUATED_MODEL,
        expected_eval_ids=expected_eval_ids,
    )
    if plan.get("schema_version") == PLAN_SCHEMA_V3:
        result["headline"] = extract_coredev_macro_star(result)
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
    try:
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
    except BaseException:
        if launched:
            _terminate_worker_groups([process for _arm, _dataset, process in launched])
        raise
    residual = [
        process for _arm, _dataset, process in launched if _worker_group_exists(process)
    ]
    if residual:
        _terminate_worker_groups(residual)
    for _arm, _dataset, process in launched:
        _ACTIVE_PROCESS_GROUPS.pop(process.pid, None)
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


def _write_inference_complete(
    output_base: Path,
    *,
    evaluation_id: str,
    configs: dict[str, Path],
    arms: tuple[tuple[str, int], ...],
    materialization: dict[str, Any],
) -> dict[str, Any]:
    """Publish a durable handoff from GPU inference to deferred scoring."""

    record = {
        "schema_version": "tgvf.paired-coredev-inference-complete.v1",
        "status": "inference_complete",
        "evaluation_id": evaluation_id,
        "arms": {
            arm: {
                "optimizer_step": step,
                "config_path": str(configs[arm].resolve()),
                "config_sha256": _sha256_file(configs[arm]),
                "materialization": materialization[arm],
            }
            for arm, step in arms
        },
    }
    write_json_atomic(output_base / "inference-complete", record)
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


def _sampling_report(
    plan: dict[str, Any], runtime: _EvaluationRuntime
) -> dict[str, object]:
    if runtime.backend == PAIRED_TGVF_BACKEND:
        sampling = runtime.protocol_contract.policy.sampling
        return {
            "source": "bound_policy_run_config",
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "do_sample": sampling.do_sample,
            "paired_rng": plan.get("paired_rng"),
        }
    if not isinstance(runtime.protocol_contract, DeepEyesNativeRunContract):
        raise TypeError("full-model sampling requires a native Crop contract")
    rollout = runtime.protocol_contract.payload["rollout"]
    return {
        "source": "bound_protocol_contract",
        "temperature": rollout["temperature"],
        "top_p": rollout["top_p"],
        "top_k": -1,
        "do_sample": rollout["temperature"] > 0,
        "master_seed": runtime.protocol_contract.payload["dataset"]["schedule_seed"],
        "paired_rng": plan.get("paired_rng"),
    }


def _identity_contract_report(
    plan: dict[str, Any], runtime: _EvaluationRuntime
) -> dict[str, object]:
    if runtime.backend == PAIRED_TGVF_BACKEND:
        return {
            "backend": PAIRED_TGVF_BACKEND,
            "checkpoint_owner_and_protocol_contract_are_same": True,
            "policy_config_path": str(_resolve_repo_path(plan["policy_config"])),
            "policy_config_sha256": plan["policy_config_sha256"],
            "run_id": runtime.checkpoint_owner.run_id,
            "run_identity_sha256": runtime.checkpoint_owner.identity_sha256,
        }
    return {
        "backend": FULL_MODEL_BACKEND,
        "checkpoint_owner_and_protocol_contract_are_same": False,
        "checkpoint_owner": dict(plan["checkpoint_owner"]),
        "protocol_contract": dict(plan["protocol_contract"]),
    }


def _build_paired_report(
    *,
    plan: dict[str, Any],
    runtime: _EvaluationRuntime,
    configs: dict[str, Path],
    materialization: dict[str, Any],
    official_summaries: dict[str, dict[str, Any] | None],
    arms: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    """Build v2 byte-compatible or v3 owner-aware paired reports."""

    is_v3 = plan.get("schema_version") == PLAN_SCHEMA_V3
    arm_reports: dict[str, dict[str, Any]] = {}
    for arm, step in arms:
        arm_report: dict[str, Any] = {
            "optimizer_step": step,
            "evaluation_identity_sha256": _arm_evaluation_identity_sha256(configs[arm]),
            "official_summary": official_summaries[arm],
        }
        if is_v3:
            arm_report["evaluation_id"] = _arm_evaluation_id(plan, arm)
        arm_reports[arm] = arm_report

    report: dict[str, Any] = {
        "schema_version": GENERIC_PAIR_SUMMARY_SCHEMA if is_v3 else PAIR_SUMMARY_SCHEMA,
        "evaluation_id": plan["evaluation_id"],
        "coverage": {
            "official_manifest_rows": 2511,
            "evaluated_single_image_rows": 2240,
            "held_multi_image_rows": 271,
            "multi_image_policy": "unsupported_explicit_hold",
        },
        "materialization": materialization,
        "sampling": _sampling_report(plan, runtime),
        "arms": arm_reports,
    }
    if is_v3:
        report["identity_contracts"] = _identity_contract_report(plan, runtime)
    for arm, _step in arms:
        report[arm] = official_summaries[arm]
    return report


def _acquire_evaluation_process_lock(output_base: Path) -> None:
    """Hold one host-local lock for the canonical output identity."""

    identity = hashlib.sha256(str(output_base.resolve()).encode("utf-8")).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"tgvf-paired-eval-{identity}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            f"another paired evaluator owns {output_base.resolve()}"
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\noutput={output_base.resolve()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    _EVALUATION_LOCK_HANDLES.append(handle)


def _cleanup_evaluation_runtime(previous_handlers: dict[signal.Signals, Any]) -> None:
    """Drain children and locks while repeat termination signals stay blocked."""

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _CHILD_SPAWN_SIGNALS)
    cleanup_error: BaseException | None = None
    try:
        try:
            if _ACTIVE_PROCESS_GROUPS:
                _terminate_worker_groups(list(_ACTIVE_PROCESS_GROUPS.values()))
        except BaseException as error:
            cleanup_error = error
        while _EVALUATION_LOCK_HANDLES:
            handle = _EVALUATION_LOCK_HANDLES.pop()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except BaseException as error:
                cleanup_error = cleanup_error or error
            finally:
                try:
                    handle.close()
                except BaseException as error:
                    cleanup_error = cleanup_error or error
        for candidate, previous in previous_handlers.items():
            try:
                signal.signal(candidate, previous)
            except BaseException as error:
                cleanup_error = cleanup_error or error
    finally:
        # Restore the caller's handlers before unblocking, so a pending repeated
        # signal cannot re-enter controlled cleanup after all resources drain.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    if cleanup_error is not None:
        raise cleanup_error


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--mode",
        choices=("prepare", "infer", "run", "score", "status"),
        default="run",
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
    if args.wait_for_step8 and args.wait_for_final_arm:
        raise ValueError("select only one checkpoint-wait mode")
    if args.mode == "score" and (
        args.wait_for_step8 or args.wait_for_final_arm or args.wait_for_gpus
    ):
        raise ValueError("score mode cannot wait on checkpoints or GPUs")
    if args.mode != "score" and (
        len(args.gpu_ids) not in {4, 8} or len(set(args.gpu_ids)) != len(args.gpu_ids)
    ):
        raise ValueError("paired evaluator requires four or eight distinct GPU IDs")
    plan = _load_plan(args.plan.resolve())
    runtime = _load_evaluation_runtime(plan)
    judge = _load_judge_config(
        plan, require_local_model=args.mode not in {"infer", "score"}
    )
    judge_gpus = tuple(judge["devices"]["physical"])
    if (
        len(judge_gpus) != 2
        or len(set(judge_gpus)) != 2
        or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in judge_gpus)
    ):
        raise RuntimeError("pinned benchmark judge GPU binding is malformed")
    if args.mode not in {"infer", "score"} and any(
        gpu_id not in args.gpu_ids for gpu_id in judge_gpus
    ):
        raise ValueError("evaluation GPU set must include pinned judge GPUs")
    output_base = (
        args.output_root.resolve()
        if args.output_root is not None
        else runtime.output_root / "evaluation" / plan["evaluation_id"]
    )
    _acquire_evaluation_process_lock(output_base)
    if args.wait_for_step8:
        if runtime.backend == FULL_MODEL_BACKEND:
            _wait_for_full_model_arm(
                plan,
                runtime,
                arm="step8",
                optimizer_step=8,
                timeout_seconds=args.wait_timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
        else:
            _wait_for_step8(
                runtime.checkpoint_owner,
                timeout_seconds=args.wait_timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
    if args.wait_for_final_arm:
        final_arm = max(plan["arms"], key=lambda item: item["optimizer_step"])
        final_step = final_arm["optimizer_step"]
        if final_step <= 0:
            raise ValueError("final-arm wait requires a positive checkpoint arm")
        if runtime.backend == FULL_MODEL_BACKEND:
            _wait_for_full_model_arm(
                plan,
                runtime,
                arm=final_arm["name"],
                optimizer_step=final_step,
                timeout_seconds=args.wait_timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
        else:
            _wait_for_optimizer_step(
                runtime.checkpoint_owner,
                optimizer_step=final_step,
                timeout_seconds=args.wait_timeout_seconds,
                poll_seconds=args.poll_seconds,
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
                plan=plan,
                runtime=runtime,
                output_base=output_base,
                arm=arm,
                step=step,
                gpu_ids=None,
            )
            for arm, step in arms
        }
    else:
        configs = {
            arm: _materialize_arm(
                plan=plan,
                runtime=runtime,
                run=runtime.checkpoint_owner,
                arm=arm,
                step=step,
                output_base=output_base,
                gpu_ids=gpu_groups[index % len(gpu_groups)],
            )
            for index, (arm, step) in enumerate(arms)
        }
    _validate_materialized_frozen_pairing(plan, configs)
    if args.wait_for_gpus:
        _wait_for_gpus(
            tuple(args.gpu_ids),
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
            free_threshold_mib=args.gpu_free_threshold_mib,
        )
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
        if args.mode == "infer":
            receipt = _write_inference_complete(
                output_base,
                evaluation_id=plan["evaluation_id"],
                configs=configs,
                arms=arms,
                materialization=materialization,
            )
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    log_root = output_base / "logs"
    existing = {
        arm: _accepted_scored_arm(configs[arm], plan, judge) for arm, _step in arms
    }
    if any(value is None for value in existing.values()):
        _require_local_judge_runtime(judge)
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
    report = _build_paired_report(
        plan=plan,
        runtime=runtime,
        configs=configs,
        materialization=materialization,
        official_summaries=existing,
        arms=arms,
    )
    report_path = output_base / "paired-summary.json"
    write_json_atomic(report_path, report)
    _write_evaluation_complete(
        output_base, report_path, evaluation_id=plan["evaluation_id"]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    """Run the evaluator with signal-safe child and lock cleanup."""

    previous_handlers: dict[signal.Signals, Any] = {}

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        candidate = getattr(signal, name, None)
        if candidate is None:
            continue
        try:
            previous_handlers[candidate] = signal.getsignal(candidate)
            signal.signal(candidate, _controlled_interrupt)
        except ValueError:
            break
    try:
        return _main()
    finally:
        _cleanup_evaluation_runtime(previous_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
