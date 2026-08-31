#!/usr/bin/env python3
"""Run the minimal, hash-bound RP69R1 step-1000 evaluation.

This controller accepts exactly the RP69R1 continuation from the audited
RP69 step-500 checkpoint.  It verifies the horizon-extension lineage before
using the terminal Adapter and deliberately runs only:

* first-200 INT-DIAG;
* first-200 three-arm ACC generation;
* Qwen2.5-72B semantic rescore of those first-200 generations.

There is no Full-867 configuration, command, artifact, or completion claim in
this controller; status and completion metadata explicitly mark Full-867 as
not run.  Evaluation artifacts live outside the training artifact tree.
``dry-run`` performs static validation without writing files or
inspecting/starting GPU processes.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SHARED_CONTROLLER_PATH = (
    REPOSITORY_ROOT / "tools/run_rp69_post_training_evaluations.py"
)
SHARED_CONTROLLER_SHA256 = (
    "4a034cf550f951526dadfd260ec643e23a762c0ca3ffd7cd2da7e65e850a7021"
)


def _bootstrap_file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_shared_controller() -> ModuleType:
    if (
        _SHARED_CONTROLLER_PATH.is_symlink()
        or not _SHARED_CONTROLLER_PATH.is_file()
        or _bootstrap_file_sha256(_SHARED_CONTROLLER_PATH) != SHARED_CONTROLLER_SHA256
    ):
        raise RuntimeError("the pinned RP69 evaluation controller dependency differs")
    spec = importlib.util.spec_from_file_location(
        "_rp69_step1000_first200_shared", _SHARED_CONTROLLER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the RP69 evaluation controller")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_shared = _load_shared_controller()

PYTHON = _shared.PYTHON
CLEAN_RUNTIME = _shared.CLEAN_RUNTIME
CLEAN_RUNTIME_COMMIT = _shared.CLEAN_RUNTIME_COMMIT
CLEAN_LAUNCHER = _shared.CLEAN_LAUNCHER
CLEAN_SEMANTIC_TOOL = _shared.CLEAN_SEMANTIC_TOOL
JUDGE_CONFIG = _shared.JUDGE_CONFIG
JUDGE_CONFIG_SHA256 = _shared.JUDGE_CONFIG_SHA256
FIRST_MANIFEST = _shared.FIRST_MANIFEST
FIRST_MANIFEST_SHA256 = _shared.FIRST_MANIFEST_SHA256
COUNTERFACTUAL_MANIFEST = _shared.COUNTERFACTUAL_MANIFEST
COUNTERFACTUAL_MANIFEST_SHA256 = _shared.COUNTERFACTUAL_MANIFEST_SHA256
GROUNDING_MANIFEST = _shared.GROUNDING_MANIFEST
GROUNDING_MANIFEST_SHA256 = _shared.GROUNDING_MANIFEST_SHA256

EXPECTED_RUN_ID = (
    "RP-69R1-QWEN3-INSTRUCT-REP-BALANCED-T1-VISUAL-BARYCENTRIC-"
    "1000-FROM-STEP0500-GPU0123"
)
EXPECTED_MODEL = "Qwen3-VL-8B-Instruct"
EXPECTED_VARIANT = "full_d_deepstack_visual_barycentric"
EXPECTED_OBJECTIVE = "balanced-matrix-ce-l-gen-norm-v1"
EXPECTED_STEP = 1000
EXPECTED_RESUME_STEP = 500
EXPECTED_WORLD_SIZE = 4
EXPECTED_TRAIN_GPUS = (0, 1, 2, 3)
EXPECTED_GA = 2
EXPECTED_TENSOR_COUNT = 104
EXPECTED_TRAIN_SHA256 = (
    "c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c"
)
EXPECTED_VALIDATION_SHA256 = (
    "de61c731eb961825a77df587cd76c00eabfea75b5c6003096f3cc7f1a51dd82d"
)
EXPECTED_TRAINING_CONFIG_SHA256 = (
    "bb9f249881d2f7d069dcdab2a1f6ada1f1fedece0f68b3fa6edcd414ffa6216c"
)

EXPECTED_ARTIFACT_ROOT = REPOSITORY_ROOT / (
    "artifacts/representation/"
    "RP-69R1-qwen3-instruct-balanced-t1-visual-barycentric-"
    "1000-from-step0500-gpu0123"
)
DEFAULT_TRAINING_CONFIG = REPOSITORY_ROOT / (
    "configs/representation/experiments/visual_barycentric/"
    "rp69r1_qwen3_instruct_visual_barycentric_1000_from_step0500_gpu0123.toml"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "artifacts/representation_experiments/visual_barycentric/evaluation/"
    "rp69r1_step1000_first200_minimal"
)
EXPECTED_RESUME_CHECKPOINT = EXPECTED_ARTIFACT_ROOT / (
    "checkpoints/representation-qwen3-instruct-rp69r1-visual-barycentric-step-00000500"
)
EXPECTED_LINEAGE_MANIFEST = EXPECTED_ARTIFACT_ROOT / (
    "horizon-extension-step00000500-to00001000.json"
)

SOURCE_RUN_ID = "RP-69-QWEN3-INSTRUCT-REP-BALANCED-T1-VISUAL-BARYCENTRIC-500-GPU0123"
SOURCE_RUN_IDENTITY_SHA256 = (
    "b4594bbc4233430ff279c077ffc1e84a68635316937f64101a3567fc7ef03516"
)
SOURCE_TRAINING_CONFIG = REPOSITORY_ROOT / (
    "configs/representation/experiments/visual_barycentric/"
    "rp69_qwen3_instruct_visual_barycentric_500_gpu0123.toml"
)
SOURCE_TRAINING_CONFIG_SHA256 = (
    "5f444fa49346f395954da2b38b4261099599990238ccc3a1ccc5af7e13d86ee4"
)
SOURCE_ARTIFACT_ROOT = REPOSITORY_ROOT / (
    "artifacts/representation/"
    "RP-69-qwen3-instruct-balanced-t1-visual-barycentric-500-gpu0123"
)
SOURCE_CHECKPOINT = SOURCE_ARTIFACT_ROOT / (
    "checkpoints/representation-qwen3-instruct-rp69-visual-barycentric-step-00000500"
)
SOURCE_CHECKPOINT_METADATA_SHA256 = (
    "472e6901c031f105a4996d3832f629b6884e0ad4b0688c0f969b32171c53b294"
)
SOURCE_METRICS = SOURCE_ARTIFACT_ROOT / "metrics.jsonl"
SOURCE_METRICS_SHA256 = (
    "bab19d3cacce5585cde8eb9a4c1bb663d039209bd7a4f614a51813f91cd98d14"
)

PIPELINE_SCHEMA = "rp69r1-step1000-first200-minimal-evaluations-v1"
RECEIPT_SCHEMA = "rp69r1-step1000-first200-minimal-training-receipt-v1"
COMPLETE_SCHEMA = "rp69r1-step1000-first200-minimal-complete-v1"
LINEAGE_SCHEMA = "representation-horizon-extension-lineage-v1"
LINEAGE_EVENT_SCHEMA = "representation-horizon-extension-event-v1"
MINIMAL_SCOPE = "int_diag_first200_and_acc_val_first200_3arm"
ARMS = (
    "image_only",
    "image_correct_D",
    "image_same_target_wrong_image_D",
)

ControllerError = _shared.ControllerError
TrainingNotComplete = _shared.TrainingNotComplete
ControllerInterrupted = _shared.ControllerInterrupted


def _configure_shared_contract() -> None:
    """Configure one private import of the reusable RP69 machinery."""

    _shared.EXPECTED_RUN_ID = EXPECTED_RUN_ID
    _shared.EXPECTED_MODEL = EXPECTED_MODEL
    _shared.EXPECTED_VARIANT = EXPECTED_VARIANT
    _shared.EXPECTED_OBJECTIVE = EXPECTED_OBJECTIVE
    _shared.EXPECTED_STEP = EXPECTED_STEP
    _shared.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE
    _shared.EXPECTED_TRAIN_GPUS = EXPECTED_TRAIN_GPUS
    _shared.EXPECTED_GA = EXPECTED_GA
    _shared.EXPECTED_TENSOR_COUNT = EXPECTED_TENSOR_COUNT
    _shared.EXPECTED_TRAIN_SHA256 = EXPECTED_TRAIN_SHA256
    _shared.EXPECTED_VALIDATION_SHA256 = EXPECTED_VALIDATION_SHA256
    _shared.EXPECTED_TRAINING_CONFIG_SHA256 = EXPECTED_TRAINING_CONFIG_SHA256
    _shared.EXPECTED_ARTIFACT_ROOT = EXPECTED_ARTIFACT_ROOT
    _shared.DEFAULT_TRAINING_CONFIG = DEFAULT_TRAINING_CONFIG
    _shared.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    _shared.PIPELINE_SCHEMA = PIPELINE_SCHEMA
    _shared.RECEIPT_SCHEMA = RECEIPT_SCHEMA
    _shared.COMPLETE_SCHEMA = COMPLETE_SCHEMA
    _shared.ARMS = ARMS


_configure_shared_contract()

_append_event = _shared._append_event
_artifact_record = _shared._artifact_record
_artifact_record_current = _shared._artifact_record_current
_atomic_json = _shared._atomic_json
_canonical_bytes = _shared._canonical_bytes
_clean_environment = _shared._clean_environment
_endpoint_open = _shared._endpoint_open
_exclusive_lock = _shared._exclusive_lock
_file_sha256 = _shared._file_sha256
_generation_command = _shared._generation_command
_gpu_compute_pids = _shared._gpu_compute_pids
_int_command = _shared._int_command
_judge_command = _shared._judge_command
_judge_environment = _shared._judge_environment
_judge_settings = _shared._judge_settings
_load_jsonl = _shared._load_jsonl
_load_toml = _shared._load_toml
_owned_worker_pids_from_plan = _shared._owned_worker_pids_from_plan
_parse_gpu_list = _shared._parse_gpu_list
_regular_file = _shared._regular_file
_render_evaluation_config = _shared._render_evaluation_config
_semantic_current = _shared._semantic_current
_stop_owned_worker_session = _shared._stop_owned_worker_session
_stop_process_group = _shared._stop_process_group
_shared_training_protected_paths = _shared._training_protected_paths
_wait_gpus_empty = _shared._wait_gpus_empty
_wait_judge_ready = _shared._wait_judge_ready
_write_identical_or_new = _shared._write_identical_or_new


def _static_training_layout(path: Path) -> dict[str, Any]:
    value = _load_toml(path)
    try:
        output = value["output"]
        checkpoint = value["checkpoint"]
        training = value["training"]
        fsdp2 = value["fsdp2"]
        resume = value["resume"]
        post = value["post_training_internal_evaluation"]
        checks = {
            "run_id": (value["run_id"], EXPECTED_RUN_ID),
            "model.model_name": (value["model"]["model_name"], EXPECTED_MODEL),
            "adapter.variant": (value["adapter"]["variant"], EXPECTED_VARIANT),
            "objective.identity": (
                value["objective"]["identity"],
                EXPECTED_OBJECTIVE,
            ),
            "data.train.source_sha256": (
                value["data"]["train"]["source_sha256"],
                EXPECTED_TRAIN_SHA256,
            ),
            "data.validation.source_sha256": (
                value["data"]["validation"]["source_sha256"],
                EXPECTED_VALIDATION_SHA256,
            ),
            "training.target_optimizer_steps": (
                training["target_optimizer_steps"],
                EXPECTED_STEP,
            ),
            "training.gradient_accumulation_steps": (
                training["gradient_accumulation_steps"],
                EXPECTED_GA,
            ),
            "fsdp2.world_size": (fsdp2["world_size"], EXPECTED_WORLD_SIZE),
            "fsdp2.physical_gpu_ids": (
                tuple(fsdp2["physical_gpu_ids"]),
                EXPECTED_TRAIN_GPUS,
            ),
            "resume.enabled": (resume["enabled"], True),
            "resume.strict_identity": (resume["strict_identity"], True),
            "post_training_internal_evaluation.enabled": (post["enabled"], False),
            "checkpoint.filename_prefix": (
                checkpoint["filename_prefix"],
                "representation-qwen3-instruct-rp69r1-visual-barycentric",
            ),
            "checkpoint.strict_identity": (checkpoint["strict_identity"], True),
        }
        adapter_path = Path(output["final_artifact_path"])
        metrics_path = Path(output["metrics_jsonl_path"])
        checkpoint_directory = Path(checkpoint["directory"])
        resume_checkpoint_path = Path(resume["checkpoint_path"])
    except (KeyError, TypeError) as error:
        raise ControllerError("RP69R1 training config lacks required fields") from error
    mismatches = [
        name for name, (observed, expected) in checks.items() if observed != expected
    ]
    expected_paths = {
        "output.final_artifact_path": (
            adapter_path,
            EXPECTED_ARTIFACT_ROOT / "adapter.pt",
        ),
        "output.metrics_jsonl_path": (
            metrics_path,
            EXPECTED_ARTIFACT_ROOT / "metrics.jsonl",
        ),
        "checkpoint.directory": (
            checkpoint_directory,
            EXPECTED_ARTIFACT_ROOT / "checkpoints",
        ),
        "resume.checkpoint_path": (
            resume_checkpoint_path,
            EXPECTED_RESUME_CHECKPOINT,
        ),
    }
    mismatches.extend(
        name
        for name, (observed, expected) in expected_paths.items()
        if observed != expected
    )
    if mismatches:
        raise ControllerError(
            "training config is not the exact RP69R1 step1000 continuation: "
            + ", ".join(sorted(mismatches))
        )
    return {
        "run_id": EXPECTED_RUN_ID,
        "adapter_path": adapter_path,
        "metrics_path": metrics_path,
        "checkpoint_directory": checkpoint_directory,
        "resume_checkpoint_path": resume_checkpoint_path,
        "target_optimizer_steps": EXPECTED_STEP,
    }


_shared._static_training_layout = _static_training_layout


def _assert_file_sha256(path: Path, expected: str, *, label: str) -> None:
    _regular_file(path, label=label)
    if _file_sha256(path) != expected:
        raise ControllerError(f"{label} SHA256 differs")


def _assert_static_inputs(training_config_path: Path) -> dict[str, Any]:
    layout = _static_training_layout(training_config_path)
    _assert_file_sha256(
        training_config_path,
        EXPECTED_TRAINING_CONFIG_SHA256,
        label="RP69R1 step1000 training config",
    )
    _assert_file_sha256(
        _SHARED_CONTROLLER_PATH,
        SHARED_CONTROLLER_SHA256,
        label="pinned shared RP69 evaluation controller",
    )
    for path, expected, label in (
        (FIRST_MANIFEST, FIRST_MANIFEST_SHA256, "first-200 manifest"),
        (
            COUNTERFACTUAL_MANIFEST,
            COUNTERFACTUAL_MANIFEST_SHA256,
            "counterfactual manifest",
        ),
        (GROUNDING_MANIFEST, GROUNDING_MANIFEST_SHA256, "grounding manifest"),
        (JUDGE_CONFIG, JUDGE_CONFIG_SHA256, "semantic judge config"),
        (
            SOURCE_TRAINING_CONFIG,
            SOURCE_TRAINING_CONFIG_SHA256,
            "source RP69 training config",
        ),
        (SOURCE_METRICS, SOURCE_METRICS_SHA256, "source RP69 metrics"),
        (
            SOURCE_CHECKPOINT / "representation_metadata.pt",
            SOURCE_CHECKPOINT_METADATA_SHA256,
            "source RP69 checkpoint metadata",
        ),
    ):
        _assert_file_sha256(path, expected, label=label)
    source_sidecar = SOURCE_CHECKPOINT / "representation_metadata.sha256"
    _regular_file(source_sidecar, label="source RP69 checkpoint metadata sidecar")
    if source_sidecar.read_text(encoding="ascii").strip() != (
        SOURCE_CHECKPOINT_METADATA_SHA256
    ):
        raise ControllerError("source RP69 checkpoint metadata sidecar differs")
    if not PYTHON.is_file():
        raise ControllerError(f"Python runtime is missing: {PYTHON}")
    for path, label in (
        (CLEAN_LAUNCHER, "frozen answer-utility launcher"),
        (CLEAN_SEMANTIC_TOOL, "frozen semantic-rescore tool"),
    ):
        _regular_file(path, label=label)
    if _shared._git_output(CLEAN_RUNTIME, "rev-parse", "HEAD") != (
        CLEAN_RUNTIME_COMMIT
    ):
        raise ControllerError("frozen evaluation runtime commit differs")
    if _shared._git_output(CLEAN_RUNTIME, "status", "--porcelain"):
        raise ControllerError("frozen evaluation runtime is dirty")
    if EXPECTED_VARIANT not in (
        CLEAN_RUNTIME / "src/tgvf_rl/representation/adapter.py"
    ).read_text(encoding="utf-8"):
        raise ControllerError("frozen runtime lacks the RP69 adapter variant")
    return layout


def _assert_output_root_isolated(
    output_root: Path, training_layout: Mapping[str, Any]
) -> None:
    output = output_root.expanduser().resolve()
    protected = _training_protected_paths(training_layout)
    conflicts = tuple(
        directory
        for directory in protected
        if output == directory
        or directory in output.parents
        or output in directory.parents
    )
    if conflicts:
        raise ControllerError(
            "evaluation output root overlaps RP69/RP69R1 training artifact paths: "
            f"output={output}; protected=" + ",".join(str(path) for path in conflicts)
        )


def _training_protected_paths(
    training_layout: Mapping[str, Any],
) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            (
                *_shared_training_protected_paths(training_layout),
                SOURCE_ARTIFACT_ROOT.resolve(),
            )
        )
    )


def _dcp_tree(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ControllerError(f"resume DCP is not one regular directory: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ControllerError(f"resume DCP contains a symlink: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    if not files:
        raise ControllerError("resume DCP contains no files")
    identity_payload = {"files": files}
    return {
        "files": files,
        "file_count": len(files),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in files),
        "identity_sha256": sha256(_canonical_bytes(identity_payload)).hexdigest(),
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControllerError(f"{label} is malformed") from error
    if not isinstance(value, dict):
        raise ControllerError(f"{label} is not an object")
    return value


def _validate_resume_dcp_lineage(
    *, source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    """Rehash both step-500 payloads and require one exact lineage tree."""

    source_declared = source.get("dcp_payload")
    target_declared = target.get("dcp_payload")
    if (
        not isinstance(source_declared, dict)
        or not isinstance(target_declared, dict)
        or source_declared != target_declared
    ):
        raise ControllerError("RP69/RP69R1 lineage DCP declarations differ")
    source_actual = _dcp_tree(SOURCE_CHECKPOINT / "dcp")
    target_actual = _dcp_tree(EXPECTED_RESUME_CHECKPOINT / "dcp")
    if source_actual != target_declared or target_actual != target_declared:
        raise ControllerError(
            "RP69 source or RP69R1 migrated resume DCP differs from lineage"
        )
    return target_actual


def _validate_resume_lineage(
    *,
    training_config_path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact RP69 step500 -> RP69R1 step1000 migration."""

    lineage = _load_json_object(
        EXPECTED_LINEAGE_MANIFEST,
        label="RP69R1 horizon-extension lineage manifest",
    )
    source = lineage.get("source")
    target = lineage.get("target")
    preserved = lineage.get("preserved_state")
    rewritten = lineage.get("rewritten_state")
    if not all(
        isinstance(item, dict) for item in (source, target, preserved, rewritten)
    ):
        raise ControllerError("RP69R1 lineage manifest sections are malformed")
    assert isinstance(source, dict)
    assert isinstance(target, dict)
    assert isinstance(preserved, dict)
    assert isinstance(rewritten, dict)

    expected_source = {
        "training_config_path": str(SOURCE_TRAINING_CONFIG),
        "training_config_sha256": SOURCE_TRAINING_CONFIG_SHA256,
        "checkpoint_path": str(SOURCE_CHECKPOINT),
        "checkpoint_metadata_sha256": SOURCE_CHECKPOINT_METADATA_SHA256,
        "metrics_path": str(SOURCE_METRICS),
        "metrics_full_sha256": SOURCE_METRICS_SHA256,
        "run_id": SOURCE_RUN_ID,
        "run_identity_sha256": SOURCE_RUN_IDENTITY_SHA256,
        "planned_target_optimizer_steps": EXPECTED_RESUME_STEP,
        "checkpoint_global_step": EXPECTED_RESUME_STEP,
    }
    expected_target = {
        "training_config_path": str(training_config_path),
        "training_config_sha256": EXPECTED_TRAINING_CONFIG_SHA256,
        "output_root": str(EXPECTED_ARTIFACT_ROOT),
        "checkpoint_path": str(EXPECTED_RESUME_CHECKPOINT),
        "metrics_path": str(EXPECTED_ARTIFACT_ROOT / "metrics.jsonl"),
        "run_id": EXPECTED_RUN_ID,
        "run_identity_sha256": receipt["run_identity_sha256"],
        "planned_target_optimizer_steps": EXPECTED_STEP,
        "lineage_manifest_path": str(EXPECTED_LINEAGE_MANIFEST),
    }
    bad_source = sorted(
        key for key, expected in expected_source.items() if source.get(key) != expected
    )
    bad_target = sorted(
        key for key, expected in expected_target.items() if target.get(key) != expected
    )
    if bad_source or bad_target:
        raise ControllerError(
            "RP69R1 lineage identity differs: source="
            + repr(bad_source)
            + ", target="
            + repr(bad_target)
        )
    if (
        lineage.get("schema_version") != LINEAGE_SCHEMA
        or lineage.get("status") != "complete"
        or lineage.get("migration_kind") != "terminal_probe_horizon_extension"
        or preserved.get("dcp_payload_byte_identical") is not True
        or preserved.get("global_step") != EXPECTED_RESUME_STEP
        or preserved.get("scientific_metrics_equal_after_identity_rebind") is not True
        or rewritten.get("horizon_extension_event_appended") is not True
        or rewritten.get("source_complete_event_copied") is not False
        or rewritten.get("optimizer_updates_performed") != 0
    ):
        raise ControllerError("RP69R1 lineage transition contract differs")

    resume_metadata = EXPECTED_RESUME_CHECKPOINT / "representation_metadata.pt"
    resume_sidecar = EXPECTED_RESUME_CHECKPOINT / "representation_metadata.sha256"
    _regular_file(resume_metadata, label="RP69R1 migrated resume metadata")
    _regular_file(resume_sidecar, label="RP69R1 migrated resume metadata sidecar")
    metadata_sha256 = _file_sha256(resume_metadata)
    if (
        target.get("checkpoint_metadata_sha256") != metadata_sha256
        or target.get("expected_checkpoint_metadata_sha256") != metadata_sha256
        or resume_sidecar.read_text(encoding="ascii").strip() != metadata_sha256
    ):
        raise ControllerError("RP69R1 migrated resume metadata identity differs")
    resume_dcp = _validate_resume_dcp_lineage(source=source, target=target)

    clean_src = str(CLEAN_RUNTIME / "src")
    if clean_src not in sys.path:
        sys.path.insert(0, clean_src)
    from tgvf_rl.representation.training.distributed_checkpoint import (
        load_distributed_representation_checkpoint_metadata,
    )

    checkpoint = load_distributed_representation_checkpoint_metadata(
        EXPECTED_RESUME_CHECKPOINT
    )
    manifest = checkpoint.manifest
    if (
        manifest.global_step != EXPECTED_RESUME_STEP
        or manifest.run_identity.run_id != EXPECTED_RUN_ID
        or manifest.run_identity_sha256 != receipt["run_identity_sha256"]
        or manifest.run_identity.planned_target_optimizer_steps != EXPECTED_STEP
        or manifest.world_size != EXPECTED_WORLD_SIZE
        or len(manifest.owned_state_names) != EXPECTED_TENSOR_COUNT
    ):
        raise ControllerError("RP69R1 migrated resume checkpoint contract differs")

    records = _load_jsonl(Path(str(receipt["metrics_path"])))
    extensions = [
        record for record in records if record.get("event") == "horizon_extension"
    ]
    if len(extensions) != 1:
        raise ControllerError(
            "RP69R1 metrics requires exactly one horizon_extension event"
        )
    event = extensions[0]
    expected_event = {
        "schema_version": LINEAGE_EVENT_SCHEMA,
        "global_step": EXPECTED_RESUME_STEP,
        "run_id": EXPECTED_RUN_ID,
        "run_identity_sha256": receipt["run_identity_sha256"],
        "source_run_id": SOURCE_RUN_ID,
        "source_run_identity_sha256": SOURCE_RUN_IDENTITY_SHA256,
        "source_planned_target_optimizer_steps": EXPECTED_RESUME_STEP,
        "target_planned_target_optimizer_steps": EXPECTED_STEP,
        "source_checkpoint_path": str(SOURCE_CHECKPOINT),
        "source_checkpoint_metadata_sha256": SOURCE_CHECKPOINT_METADATA_SHA256,
        "target_training_config_path": str(training_config_path),
        "target_training_config_sha256": EXPECTED_TRAINING_CONFIG_SHA256,
        "state_transition": "identity_rebind_only_no_optimizer_update",
    }
    bad_event = sorted(
        key for key, expected in expected_event.items() if event.get(key) != expected
    )
    terminal = records[-1]
    if (
        bad_event
        or event.get("target_canonical_config_sha256")
        != target.get("canonical_config_sha256")
        or terminal.get("canonical_config_sha256")
        != target.get("canonical_config_sha256")
    ):
        raise ControllerError(
            "RP69R1 horizon_extension event differs: " + repr(bad_event)
        )
    return {
        "status": "complete",
        "migration_kind": "terminal_probe_horizon_extension",
        "source_run_id": SOURCE_RUN_ID,
        "source_run_identity_sha256": SOURCE_RUN_IDENTITY_SHA256,
        "source_global_step": EXPECTED_RESUME_STEP,
        "target_run_id": EXPECTED_RUN_ID,
        "target_run_identity_sha256": receipt["run_identity_sha256"],
        "target_global_step": EXPECTED_STEP,
        "lineage_manifest": _artifact_record(EXPECTED_LINEAGE_MANIFEST),
        "resume_checkpoint_path": str(EXPECTED_RESUME_CHECKPOINT),
        "resume_checkpoint_metadata": _artifact_record(resume_metadata),
        "resume_dcp_identity_sha256": resume_dcp["identity_sha256"],
        "horizon_extension_event_sha256": sha256(_canonical_bytes(event)).hexdigest(),
    }


_shared_load_training_completion = _shared._load_training_completion


def _load_training_completion(
    training_config_path: Path,
    *,
    expected_config_sha256: str,
    int_report_path: Path,
) -> dict[str, Any]:
    receipt = _shared_load_training_completion(
        training_config_path,
        expected_config_sha256=expected_config_sha256,
        int_report_path=int_report_path,
    )
    receipt["schema_version"] = RECEIPT_SCHEMA
    receipt["evaluation_scope"] = MINIMAL_SCOPE
    receipt["external_int_diag"] = {
        "status": "planned",
        "evaluation_id": "rp69r1-step1000-first200-int-diag-v1",
        "path": str(int_report_path),
    }
    receipt["resume_lineage"] = _validate_resume_lineage(
        training_config_path=training_config_path,
        receipt=receipt,
    )
    return receipt


def _default_output_root() -> Path:
    return DEFAULT_OUTPUT_ROOT


def _paths(root: Path) -> dict[str, Path]:
    generated = root / "generated-configs"
    return {
        "root": root,
        "lock": root / "pipeline.lock",
        "state": root / "state.json",
        "events": root / "events.jsonl",
        "receipt": root / "training-completion-receipt.json",
        "int_config": generated / "rp69r1-step1000-int-first200.toml",
        "first_config": generated / "rp69r1-step1000-acc-first200.toml",
        "int_report": root / "INT-DIAG-first200-step1000.json",
        "int_log": root / "INT-DIAG-first200-step1000.log",
        "first_generation": root / "ACC-VAL-first200-generation-3arm",
        "first_generation_log": root / "ACC-VAL-first200-generation-3arm.log",
        "first_semantic": root / "ACC-VAL-first200-semantic-3arm",
        "first_semantic_log": root / "ACC-VAL-first200-semantic-3arm.log",
        "judge_log": root / "semantic-judge.log",
        "judge_ownership": root / "semantic-judge-ownership.json",
        "complete": root / "complete-minimal-first200.json",
    }


def _materialize_configs(
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    int_gpu: int,
    first_gpus: tuple[int, ...],
) -> None:
    _write_identical_or_new(
        paths["receipt"],
        (
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    specifications = (
        (
            paths["int_config"],
            _render_evaluation_config(
                receipt=receipt,
                run_id=f"{EXPECTED_RUN_ID}-STEP1000-INT-FIRST200-GPU{int_gpu}",
                evaluation_id="rp69r1-step1000-first200-int-diag-v1",
                manifest_path=FIRST_MANIFEST,
                manifest_sha256=FIRST_MANIFEST_SHA256,
                report_path=paths["int_report"],
                physical_gpu_id=int_gpu,
            ),
        ),
        (
            paths["first_config"],
            _render_evaluation_config(
                receipt=receipt,
                run_id=f"{EXPECTED_RUN_ID}-STEP1000-ACC-FIRST200",
                evaluation_id="rp69r1-step1000-first200-acc-val-v1",
                manifest_path=FIRST_MANIFEST,
                manifest_sha256=FIRST_MANIFEST_SHA256,
                report_path=paths["root"] / "first200-internal-unused.json",
                physical_gpu_id=first_gpus[0],
            ),
        ),
    )
    for path, payload in specifications:
        _write_identical_or_new(path, payload)


def _int_report_current(receipt: Mapping[str, Any], paths: Mapping[str, Path]) -> bool:
    return _shared._int_report_current(receipt, paths)


def _generation_current(
    root: Path,
    *,
    gpu_ids: tuple[int, ...],
    receipt: Mapping[str, Any],
    source_config: Path,
) -> dict[str, Any] | None:
    return _shared._generation_current(
        root,
        samples=200,
        gpu_ids=gpu_ids,
        receipt=receipt,
        source_config=source_config,
    )


def _run_parallel_first200(
    *,
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
    int_gpu: int,
    first_gpus: tuple[int, ...],
    timeout_seconds: float,
) -> dict[str, Any]:
    first = _generation_current(
        paths["first_generation"],
        gpu_ids=first_gpus,
        receipt=receipt,
        source_config=paths["first_config"],
    )
    specifications = (
        (
            "int_diag",
            _int_report_current(receipt, paths),
            _int_command(paths["int_config"]),
            paths["int_log"],
            int_gpu,
            None,
        ),
        (
            "first200_generation",
            first is not None,
            _generation_command(
                config=paths["first_config"],
                root=paths["first_generation"],
                gpu_ids=first_gpus,
            ),
            paths["first_generation_log"],
            None,
            paths["first_generation"] / "launch-plan.json",
        ),
    )
    jobs: list[dict[str, Any]] = []
    try:
        for name, done, command, log_path, visible_gpu, worker_plan in specifications:
            if done:
                _append_event(paths["events"], "stage_already_complete", stage=name)
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("ab", buffering=0)
            environment = _clean_environment()
            if visible_gpu is not None:
                environment.update(
                    {
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "CUDA_VISIBLE_DEVICES": str(visible_gpu),
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    }
                )
            process: subprocess.Popen[bytes] | None = None
            try:
                process = subprocess.Popen(
                    command,
                    cwd=CLEAN_RUNTIME,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                jobs.append(
                    {
                        "name": name,
                        "process": process,
                        "log": log,
                        "log_path": log_path,
                        "worker_plan": worker_plan,
                    }
                )
            except BaseException:
                if process is not None:
                    _stop_process_group(process)
                log.close()
                raise
            _append_event(
                paths["events"],
                "stage_started",
                stage=name,
                pid=process.pid,
                command=command,
            )
        deadline = time.monotonic() + timeout_seconds
        pending = list(jobs)
        while pending:
            for job in tuple(pending):
                process = job["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.remove(job)
                job["log"].close()
                _append_event(
                    paths["events"],
                    "stage_exited",
                    stage=job["name"],
                    pid=process.pid,
                    returncode=return_code,
                )
                if return_code != 0:
                    raise ControllerError(
                        f"{job['name']} failed; inspect {job['log_path']}"
                    )
            if pending and time.monotonic() >= deadline:
                raise ControllerError("minimal first200 evaluation stage timed out")
            if pending:
                time.sleep(2)
    finally:
        for job in jobs:
            worker_plan = job["worker_plan"]
            if isinstance(worker_plan, Path):
                for worker_pid in _owned_worker_pids_from_plan(worker_plan):
                    _stop_owned_worker_session(worker_pid)
            _stop_process_group(job["process"])
            if not job["log"].closed:
                job["log"].close()
    if not _int_report_current(receipt, paths):
        raise ControllerError("INT-DIAG exited without a complete report")
    first = _generation_current(
        paths["first_generation"],
        gpu_ids=first_gpus,
        receipt=receipt,
        source_config=paths["first_config"],
    )
    if first is None:
        raise ControllerError("first200 ACC generation has no complete summary")
    return first


def _run_semantic_first200(
    *,
    first: Mapping[str, Any],
    paths: Mapping[str, Path],
    judge_gpus: tuple[int, int],
    gpu_wait_timeout: float,
) -> None:
    if _semantic_current(
        paths["first_semantic"],
        samples=200,
        generation=first,
        source_config=paths["first_config"],
    ):
        return
    settings = _judge_settings()
    if _endpoint_open(int(settings["port"])):
        raise ControllerError("semantic judge port is occupied by an unowned endpoint")
    _wait_gpus_empty(
        judge_gpus,
        timeout_seconds=gpu_wait_timeout,
        paths=paths,
    )
    log = paths["judge_log"].open("ab", buffering=0)
    judge: subprocess.Popen[bytes] | None = None
    ownership: dict[str, Any] | None = None
    try:
        judge = subprocess.Popen(
            _judge_command(settings),
            cwd=CLEAN_RUNTIME,
            env=_judge_environment(judge_gpus=judge_gpus, paths=paths),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ownership = {
            "schema_version": "rp69r1-step1000-minimal-owned-semantic-judge-v1",
            "status": "starting",
            "started_at": _shared._utc_now(),
            "pid": judge.pid,
            "pgid": judge.pid,
            "gpu_ids": list(judge_gpus),
            "port": settings["port"],
            "runtime_commit": CLEAN_RUNTIME_COMMIT,
            "judge_config_sha256": JUDGE_CONFIG_SHA256,
            "evaluation_scope": MINIMAL_SCOPE,
        }
        _atomic_json(paths["judge_ownership"], ownership)
        _wait_judge_ready(judge, settings=settings, timeout=600)
        ownership["status"] = "ready"
        ownership["ready_at"] = _shared._utc_now()
        _atomic_json(paths["judge_ownership"], ownership)
        _shared._run_semantic_one(
            split="first200",
            generation=first,
            config=paths["first_config"],
            output_root=paths["first_semantic"],
            samples=200,
            log_path=paths["first_semantic_log"],
            paths=paths,
        )
    finally:
        if judge is not None:
            _stop_process_group(judge)
        log.close()
        if ownership is not None and judge is not None:
            ownership["status"] = "stopped"
            ownership["stopped_at"] = _shared._utc_now()
            ownership["returncode"] = judge.returncode
            _atomic_json(paths["judge_ownership"], ownership)
    _wait_gpus_empty(judge_gpus, timeout_seconds=300, paths=paths)


def _complete_marker_current(
    path: Path,
    *,
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
    first_gpus: tuple[int, ...],
) -> bool:
    if not path.exists():
        return False
    value = _load_json_object(path, label="RP69R1 minimal completion marker")
    generated = value.get("generated_configs")
    first = value.get("first200")
    scope = value.get("evaluation_scope")
    if not all(isinstance(item, dict) for item in (generated, first, scope)):
        raise ControllerError("RP69R1 minimal completion sections are malformed")
    assert isinstance(generated, dict)
    assert isinstance(first, dict)
    assert isinstance(scope, dict)
    generation = _generation_current(
        paths["first_generation"],
        gpu_ids=first_gpus,
        receipt=receipt,
        source_config=paths["first_config"],
    )
    semantic_ok = _semantic_current(
        paths["first_semantic"],
        samples=200,
        generation=generation,
        source_config=paths["first_config"],
    )
    if (
        value.get("schema_version") != COMPLETE_SCHEMA
        or value.get("status") != "complete"
        or value.get("training_run_id") != EXPECTED_RUN_ID
        or value.get("training_run_identity_sha256") != receipt["run_identity_sha256"]
        or value.get("adapter_manifest_sha256") != receipt["adapter_manifest_sha256"]
        or value.get("runtime_commit") != CLEAN_RUNTIME_COMMIT
        or value.get("controller_file_sha256") != _file_sha256(Path(__file__).resolve())
        or value.get("clean_launcher_sha256") != _file_sha256(CLEAN_LAUNCHER)
        or value.get("clean_semantic_tool_sha256") != _file_sha256(CLEAN_SEMANTIC_TOOL)
        or value.get("shared_controller_file_sha256") != SHARED_CONTROLLER_SHA256
        or not _artifact_record_current(
            value.get("shared_controller"), _SHARED_CONTROLLER_PATH
        )
        or tuple(value.get("arms", ())) != ARMS
        or value.get("resume_lineage") != receipt["resume_lineage"]
        or set(generated) != {"int_diag", "first200"}
        or "full867" in value
        or first.get("samples") != 200
        or scope
        != {
            "kind": "minimal_recovery",
            "scope": MINIMAL_SCOPE,
            "full867": "not_run",
            "int_diag_samples": 200,
            "acc_val_samples": 200,
        }
        or not _artifact_record_current(value.get("receipt"), paths["receipt"])
        or not _artifact_record_current(generated.get("int_diag"), paths["int_config"])
        or not _artifact_record_current(
            generated.get("first200"), paths["first_config"]
        )
        or not _artifact_record_current(value.get("int_diag"), paths["int_report"])
        or not _artifact_record_current(
            first.get("generation"),
            paths["first_generation"] / "launch-summary.json",
        )
        or not _artifact_record_current(
            first.get("semantic_summary"),
            paths["first_semantic"] / "summary.json",
        )
        or not _artifact_record_current(
            first.get("semantic_manifest"),
            paths["first_semantic"] / "manifest.json",
        )
        or generation is None
        or not semantic_ok
    ):
        raise ControllerError("RP69R1 minimal completion marker drifted")
    return True


def _write_complete_marker(
    *,
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    marker = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "complete",
        "completed_at": _shared._utc_now(),
        "evaluation_scope": {
            "kind": "minimal_recovery",
            "scope": MINIMAL_SCOPE,
            "full867": "not_run",
            "int_diag_samples": 200,
            "acc_val_samples": 200,
        },
        "runtime_commit": CLEAN_RUNTIME_COMMIT,
        "controller_file_sha256": _file_sha256(Path(__file__).resolve()),
        "clean_launcher_sha256": _file_sha256(CLEAN_LAUNCHER),
        "clean_semantic_tool_sha256": _file_sha256(CLEAN_SEMANTIC_TOOL),
        "shared_controller_file_sha256": SHARED_CONTROLLER_SHA256,
        "shared_controller": _artifact_record(_SHARED_CONTROLLER_PATH),
        "training_run_id": EXPECTED_RUN_ID,
        "training_run_identity_sha256": receipt["run_identity_sha256"],
        "adapter_manifest_sha256": receipt["adapter_manifest_sha256"],
        "arms": list(ARMS),
        "receipt": _artifact_record(paths["receipt"]),
        "resume_lineage": receipt["resume_lineage"],
        "generated_configs": {
            "int_diag": _artifact_record(paths["int_config"]),
            "first200": _artifact_record(paths["first_config"]),
        },
        "int_diag": _artifact_record(paths["int_report"]),
        "first200": {
            "samples": 200,
            "generation": _artifact_record(
                paths["first_generation"] / "launch-summary.json"
            ),
            "semantic_summary": _artifact_record(
                paths["first_semantic"] / "summary.json"
            ),
            "semantic_manifest": _artifact_record(
                paths["first_semantic"] / "manifest.json"
            ),
        },
    }
    _atomic_json(paths["complete"], marker)


def _wait_training(
    training_config_path: Path,
    *,
    expected_config_sha256: str,
    poll_seconds: float,
    timeout_seconds: float,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_reason: str | None = None
    while True:
        try:
            receipt = _load_training_completion(
                training_config_path,
                expected_config_sha256=expected_config_sha256,
                int_report_path=paths["int_report"],
            )
            _append_event(paths["events"], "training_complete_verified")
            return receipt
        except TrainingNotComplete as error:
            reason = str(error)
            if reason != last_reason:
                _append_event(paths["events"], "waiting_for_training", reason=reason)
                last_reason = reason
            _atomic_json(
                paths["state"],
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "status": "waiting_for_training",
                    "pid": os.getpid(),
                    "training_config_path": str(training_config_path),
                    "training_config_sha256": expected_config_sha256,
                    "reason": reason,
                    "updated_at": _shared._utc_now(),
                },
            )
        if time.monotonic() >= deadline:
            raise ControllerError("timed out waiting for RP69R1 training completion")
        time.sleep(poll_seconds)


def _status(
    *,
    training_config_path: Path,
    expected_config_sha256: str,
    paths: Mapping[str, Path],
    first_gpus: tuple[int, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": PIPELINE_SCHEMA,
        "status": "waiting",
        "evaluation_scope": MINIMAL_SCOPE,
        "full867": "not_run",
        "training_config_path": str(training_config_path),
        "training_config_sha256": expected_config_sha256,
        "output_root": str(paths["root"]),
        "training_complete": False,
        "resume_lineage_complete": False,
        "int_diag_first200_complete": False,
        "acc_val_first200_generation_complete": False,
        "acc_val_first200_semantic_complete": False,
        "pipeline_complete": False,
    }
    try:
        receipt = _load_training_completion(
            training_config_path,
            expected_config_sha256=expected_config_sha256,
            int_report_path=paths["int_report"],
        )
        result["training_complete"] = True
        result["resume_lineage_complete"] = True
        result["training_run_identity_sha256"] = receipt["run_identity_sha256"]
        result["int_diag_first200_complete"] = _int_report_current(receipt, paths)
        generation = _generation_current(
            paths["first_generation"],
            gpu_ids=first_gpus,
            receipt=receipt,
            source_config=paths["first_config"],
        )
        result["acc_val_first200_generation_complete"] = generation is not None
        result["acc_val_first200_semantic_complete"] = _semantic_current(
            paths["first_semantic"],
            samples=200,
            generation=generation,
            source_config=paths["first_config"],
        )
        result["pipeline_complete"] = _complete_marker_current(
            paths["complete"],
            receipt=receipt,
            paths=paths,
            first_gpus=first_gpus,
        )
        result["status"] = "complete" if result["pipeline_complete"] else "ready"
    except TrainingNotComplete as error:
        result["waiting_reason"] = str(error)
    except (ControllerError, FileNotFoundError, ValueError) as error:
        result["status"] = "blocked"
        result["blocked_reason"] = str(error)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=24 * 60 * 60)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument("--evaluation-timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument("--int-gpu", type=int, default=0)
    parser.add_argument("--first-gpus", type=_parse_gpu_list, default=(1,))
    parser.add_argument("--judge-gpus", type=_parse_gpu_list, default=(0, 1))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run")
    subparsers.add_parser("status")
    run = subparsers.add_parser("run")
    run.add_argument("--execute", action="store_true", required=True)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    for name in (
        "poll_seconds",
        "wait_timeout_seconds",
        "gpu_wait_timeout_seconds",
        "evaluation_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            raise ControllerError(f"{name} must be positive")
    if args.int_gpu < 0:
        raise ControllerError("INT-DIAG GPU must be non-negative")
    if len(args.judge_gpus) != 2:
        raise ControllerError("semantic judge requires exactly two GPUs")
    if args.int_gpu in set(args.first_gpus):
        raise ControllerError("INT-DIAG GPU must be disjoint from ACC generation GPUs")
    if (args.int_gpu, args.first_gpus, args.judge_gpus) != (0, (1,), (0, 1)):
        raise ControllerError(
            "RP69R1 minimal evaluation requires exact GPU plan: "
            "int=0, first=(1,), judge=(0,1)"
        )


def _dry_run(
    *,
    training_config_path: Path,
    output_root: Path,
    int_gpu: int,
    first_gpus: tuple[int, ...],
    judge_gpus: tuple[int, int],
) -> dict[str, Any]:
    layout = _assert_static_inputs(training_config_path)
    _assert_output_root_isolated(output_root, layout)
    paths = _paths(output_root)
    placeholder = {
        "training_config_path": str(training_config_path),
        "training_config_sha256": _file_sha256(training_config_path),
        "adapter_path": str(layout["adapter_path"]),
        "adapter_file_sha256": "<materialized-after-training>",
        "adapter_manifest_sha256": "<materialized-after-training>",
        "run_identity_sha256": "<materialized-after-training>",
        "evaluation_data_path": "<bound-from-training-config>",
        "evaluation_data_source_sha256": EXPECTED_VALIDATION_SHA256,
        "conditioning_provider": "contextual_hidden_state",
        "prompt_identity": "qwen3-representation-image-question-v1",
        "prompt_sha256": "<bound-from-training-config>",
    }
    return {
        "schema_version": PIPELINE_SCHEMA,
        "status": "dry_run_complete",
        "gpu_processes_started": False,
        "files_written": False,
        "training_run_id": EXPECTED_RUN_ID,
        "training_config_sha256": _file_sha256(training_config_path),
        "evaluation_runtime_commit": CLEAN_RUNTIME_COMMIT,
        "evaluation_runtime_clean": True,
        "shared_controller": {
            "path": str(_SHARED_CONTROLLER_PATH),
            "sha256": SHARED_CONTROLLER_SHA256,
        },
        "adapter_variant": EXPECTED_VARIANT,
        "arms": list(ARMS),
        "evaluation_scope": {
            "scope": MINIMAL_SCOPE,
            "full867": "not_run",
        },
        "resume_lineage": {
            "source_run_id": SOURCE_RUN_ID,
            "source_run_identity_sha256": SOURCE_RUN_IDENTITY_SHA256,
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "source_checkpoint_metadata_sha256": (SOURCE_CHECKPOINT_METADATA_SHA256),
            "target_resume_checkpoint": str(EXPECTED_RESUME_CHECKPOINT),
            "target_lineage_manifest": str(EXPECTED_LINEAGE_MANIFEST),
        },
        "output_root": str(output_root),
        "output_isolation_preflight": {
            "status": "passed",
            "protected_paths": [
                str(path) for path in _training_protected_paths(layout)
            ],
        },
        "gpu_plan": {
            "parallel": {
                "int_diag_first200": [int_gpu],
                "acc_first200": list(first_gpus),
            },
            "semantic_judge_after_parallel_stage": list(judge_gpus),
        },
        "commands": {
            "int_diag": _int_command(paths["int_config"]),
            "acc_first200": _generation_command(
                config=paths["first_config"],
                root=paths["first_generation"],
                gpu_ids=first_gpus,
            ),
        },
        "config_preview_sha256": {
            "int_diag": sha256(
                _render_evaluation_config(
                    receipt=placeholder,
                    run_id=(f"{EXPECTED_RUN_ID}-STEP1000-INT-FIRST200-GPU{int_gpu}"),
                    evaluation_id="rp69r1-step1000-first200-int-diag-v1",
                    manifest_path=FIRST_MANIFEST,
                    manifest_sha256=FIRST_MANIFEST_SHA256,
                    report_path=paths["int_report"],
                    physical_gpu_id=int_gpu,
                )
            ).hexdigest(),
            "acc_first200": sha256(
                _render_evaluation_config(
                    receipt=placeholder,
                    run_id=f"{EXPECTED_RUN_ID}-STEP1000-ACC-FIRST200",
                    evaluation_id="rp69r1-step1000-first200-acc-val-v1",
                    manifest_path=FIRST_MANIFEST,
                    manifest_sha256=FIRST_MANIFEST_SHA256,
                    report_path=paths["root"] / "first200-internal-unused.json",
                    physical_gpu_id=first_gpus[0],
                )
            ).hexdigest(),
            "note": "preview contains explicit post-training hash placeholders",
        },
    }


def _run(
    args: argparse.Namespace,
    training_config_path: Path,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    _assert_output_root_isolated(
        paths["root"],
        _static_training_layout(training_config_path),
    )
    expected_config_sha256 = _file_sha256(training_config_path)
    paths["root"].mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(paths["lock"]):
        _append_event(
            paths["events"],
            "controller_started",
            pid=os.getpid(),
            evaluation_scope=MINIMAL_SCOPE,
            training_config_path=str(training_config_path),
            training_config_sha256=expected_config_sha256,
        )
        try:
            receipt = _wait_training(
                training_config_path,
                expected_config_sha256=expected_config_sha256,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.wait_timeout_seconds,
                paths=paths,
            )
            _materialize_configs(
                receipt,
                paths,
                int_gpu=args.int_gpu,
                first_gpus=args.first_gpus,
            )
            if _complete_marker_current(
                paths["complete"],
                receipt=receipt,
                paths=paths,
                first_gpus=args.first_gpus,
            ):
                return _status(
                    training_config_path=training_config_path,
                    expected_config_sha256=expected_config_sha256,
                    paths=paths,
                    first_gpus=args.first_gpus,
                )
            pending_gpus: list[int] = []
            if not _int_report_current(receipt, paths):
                pending_gpus.append(args.int_gpu)
            if (
                _generation_current(
                    paths["first_generation"],
                    gpu_ids=args.first_gpus,
                    receipt=receipt,
                    source_config=paths["first_config"],
                )
                is None
            ):
                pending_gpus.extend(args.first_gpus)
            if pending_gpus:
                _wait_gpus_empty(
                    pending_gpus,
                    timeout_seconds=args.gpu_wait_timeout_seconds,
                    paths=paths,
                )
            first = _run_parallel_first200(
                receipt=receipt,
                paths=paths,
                int_gpu=args.int_gpu,
                first_gpus=args.first_gpus,
                timeout_seconds=args.evaluation_timeout_seconds,
            )
            _run_semantic_first200(
                first=first,
                paths=paths,
                judge_gpus=args.judge_gpus,
                gpu_wait_timeout=args.gpu_wait_timeout_seconds,
            )
            _write_complete_marker(receipt=receipt, paths=paths)
            _atomic_json(
                paths["state"],
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "status": "complete",
                    "evaluation_scope": MINIMAL_SCOPE,
                    "full867": "not_run",
                    "pid": os.getpid(),
                    "marker": str(paths["complete"]),
                    "completed_at": _shared._utc_now(),
                },
            )
            _append_event(paths["events"], "controller_complete")
        except BaseException as error:
            _atomic_json(
                paths["state"],
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "status": "failed",
                    "evaluation_scope": MINIMAL_SCOPE,
                    "pid": os.getpid(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_at": _shared._utc_now(),
                },
            )
            _append_event(
                paths["events"],
                "controller_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
    return _status(
        training_config_path=training_config_path,
        expected_config_sha256=expected_config_sha256,
        paths=paths,
        first_gpus=args.first_gpus,
    )


def _install_signal_handlers() -> dict[signal.Signals, Any]:
    previous: dict[signal.Signals, Any] = {}

    def handler(signum: int, _frame: object) -> None:
        raise ControllerInterrupted(f"received {signal.Signals(signum).name}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous


def _restore_signal_handlers(previous: Mapping[signal.Signals, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def main() -> int:
    args = _parser().parse_args()
    _validate_arguments(args)
    training_config_path = args.training_config.expanduser().resolve()
    training_layout = _assert_static_inputs(training_config_path)
    output_root = (
        _default_output_root()
        if args.output_root is None
        else args.output_root.expanduser().resolve()
    )
    _assert_output_root_isolated(output_root, training_layout)
    paths = _paths(output_root)
    if args.command == "dry-run":
        result = _dry_run(
            training_config_path=training_config_path,
            output_root=output_root,
            int_gpu=args.int_gpu,
            first_gpus=args.first_gpus,
            judge_gpus=args.judge_gpus,
        )
    elif args.command == "status":
        result = _status(
            training_config_path=training_config_path,
            expected_config_sha256=_file_sha256(training_config_path),
            paths=paths,
            first_gpus=args.first_gpus,
        )
    else:
        previous = _install_signal_handlers()
        try:
            result = _run(args, training_config_path, paths)
        finally:
            _restore_signal_handlers(previous)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ControllerError as error:
        print(f"RP69R1_MINIMAL_EVALUATION_BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(3) from error
