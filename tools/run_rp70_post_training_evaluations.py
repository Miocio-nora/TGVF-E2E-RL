#!/usr/bin/env python3
"""Wait for one RP70 training config and run its complete evaluation chain.

The controller is deliberately separate from representation training.  It
waits for a durable terminal metrics record and a matching Adapter export,
then materializes evaluation TOMLs containing the hashes that cannot be known
before training completes.  Evaluation runs from the frozen RP70 evaluation
worktree, not from the mutable training checkout.

INT-DIAG is normally part of the accepted full-horizon training contract.  The
controller verifies and reuses that hash-bound report.  An explicit recovery
mode can instead run external INT-DIAG after proving optimizer/checkpoint/
Adapter completion and the known grounding-sidecar failure, without creating a
fake core completion record.  The recovery schedule is:

* GPU0: external recovery INT-DIAG;
* GPU1/2: first-200, three-arm ACC generation;
* GPU3-7: full-867, three-arm ACC generation;
* GPU6/7: semantic judge after every generation process has exited.

Every stage is restartable.  Existing publications are accepted only after
their identities and hashes have been revalidated.  The controller never
deletes or replaces a nonidentical result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPOSITORY_ROOT / ".venv312/bin/python"
CLEAN_RUNTIME = REPOSITORY_ROOT / ".eval-runtime-rp70-20260802"
CLEAN_RUNTIME_COMMIT = "2d61b07995b1d5b90c221fe1faf5090e8d985fef"
CLEAN_LAUNCHER = (
    CLEAN_RUNTIME / "tools/launch_representation_answer_utility_evaluation.py"
)
CLEAN_SEMANTIC_TOOL = (
    CLEAN_RUNTIME / "tools/run_representation_answer_utility_semantic_rescore.py"
)
PYTHON_HEADER_ROOT = REPOSITORY_ROOT / ".deps/python312-dev/root/usr/include"

FIRST_MANIFEST = REPOSITORY_ROOT / (
    "configs/representation/internal_evaluation/"
    "qwen3_v4_clean_imend_test_golden_first200_variable_k_v1.json"
)
FIRST_MANIFEST_SHA256 = (
    "55e2cde5e118de77e4bcc099a422844129cdff1caf328d349dae5c7f11a634d8"
)
FULL_MANIFEST = REPOSITORY_ROOT / (
    "configs/representation/internal_evaluation/"
    "qwen3_v4_clean_imend_test_full867_variable_k_v1.json"
)
FULL_MANIFEST_SHA256 = (
    "31cce579e919dccf3ba2702e09db3a7b2cfa65e412db079c7dcdcb15dcddbe78"
)
COUNTERFACTUAL_MANIFEST = REPOSITORY_ROOT / (
    "configs/representation/internal_evaluation/"
    "qwen3_v4_clean_imend_test_golden_counterfactual_v1.json"
)
COUNTERFACTUAL_MANIFEST_SHA256 = (
    "4589d14f196ccde48c3439405700220bc0fb63487edae0e74bc6b3713d7f4cc4"
)
GROUNDING_MANIFEST = REPOSITORY_ROOT / (
    "configs/representation/internal_evaluation/"
    "qwen3_v4_clean_imend_audited_grounding_v1.json"
)
GROUNDING_MANIFEST_SHA256 = (
    "a65aa6e6038ada1436302b60440136cc98b388552a7782b48ec95ed4324938c0"
)
JUDGE_CONFIG = (
    REPOSITORY_ROOT / "configs/policy/judges/qwen25_72b_rl_answer_judge_v1.json"
)
JUDGE_CONFIG_SHA256 = (
    "3737504858912a6392679d2c9720597cde58dd7d3218aa6f75b67ad00a769573"
)

PIPELINE_SCHEMA = "rp70-post-training-evaluations-v1"
RECEIPT_SCHEMA = "rp70-training-completion-receipt-v1"
COMPLETE_SCHEMA = "rp70-post-training-evaluations-complete-v1"
SEMANTIC_SCHEMA = "answer-utility-semantic-rescore-v2"
GENERATION_SCHEMA = "answer_utility_multi_worker_launch_result_v1"
ARMS = (
    "image_only",
    "image_correct_D",
    "image_same_target_wrong_image_D",
)
EXPECTED_MODEL = "Qwen3-VL-8B-Instruct"
EXPECTED_ADAPTER_VARIANT = "full_d_deepstack"
EXPECTED_TENSOR_COUNT = 104


class ControllerError(RuntimeError):
    """Fail-closed RP70 evaluation-controller error."""


class TrainingNotComplete(ControllerError):
    """Expected state while waiting for the terminal artifact boundary."""


class ControllerInterrupted(ControllerError):
    """Raised by an owned SIGINT or SIGTERM."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_prefix_sha256(path: Path, byte_count: int) -> str:
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ControllerError("file-prefix byte count is invalid")
    _regular_file(path, label="prefix-bound file")
    if path.stat().st_size < byte_count:
        raise ControllerError("prefix-bound file is shorter than its receipt")
    digest = sha256()
    remaining = byte_count
    with path.open("rb") as handle:
        while remaining:
            block = handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ControllerError("prefix-bound file was truncated during hashing")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ControllerError(f"{label} is not one regular file: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def _write_identical_or_new(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ControllerError(f"refusing to replace nonidentical file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_event(path: Path, event: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"at": _utc_now(), "event": event, **fields}
    with path.open("ab", buffering=0) as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        os.fsync(handle.fileno())


def _artifact_record(path: Path) -> dict[str, object]:
    _regular_file(path, label="completion artifact")
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _artifact_record_current(record: object, path: Path) -> bool:
    if not isinstance(record, dict) or path.is_symlink() or not path.is_file():
        return False
    return (
        record.get("path") == str(path.resolve())
        and record.get("sha256") == _file_sha256(path)
        and record.get("bytes") == path.stat().st_size
    )


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ControllerError(
            f"git {' '.join(arguments)} failed in {root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _assert_static_inputs() -> None:
    for path, expected, label in (
        (FIRST_MANIFEST, FIRST_MANIFEST_SHA256, "first-200 manifest"),
        (FULL_MANIFEST, FULL_MANIFEST_SHA256, "full-867 manifest"),
        (
            COUNTERFACTUAL_MANIFEST,
            COUNTERFACTUAL_MANIFEST_SHA256,
            "counterfactual manifest",
        ),
        (GROUNDING_MANIFEST, GROUNDING_MANIFEST_SHA256, "grounding manifest"),
        (JUDGE_CONFIG, JUDGE_CONFIG_SHA256, "judge config"),
    ):
        _regular_file(path, label=label)
        if _file_sha256(path) != expected:
            raise ControllerError(f"pinned {label} SHA256 differs")
    if not PYTHON.is_file():
        raise ControllerError(f"Python runtime is missing: {PYTHON}")
    for path, label in (
        (CLEAN_LAUNCHER, "clean answer-utility launcher"),
        (CLEAN_SEMANTIC_TOOL, "clean semantic-rescore tool"),
    ):
        _regular_file(path, label=label)
    if _git_output(CLEAN_RUNTIME, "rev-parse", "HEAD") != CLEAN_RUNTIME_COMMIT:
        raise ControllerError("clean RP70 runtime commit differs")
    if _git_output(CLEAN_RUNTIME, "status", "--porcelain"):
        raise ControllerError("clean RP70 runtime is dirty")


def _raw_training_layout(training_config_path: Path) -> dict[str, Any]:
    _regular_file(training_config_path, label="training config")
    try:
        value = tomllib.loads(training_config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ControllerError("training config TOML is unreadable") from error
    try:
        run_id = value["run_id"]
        metrics = Path(value["output"]["metrics_jsonl_path"])
        adapter = Path(value["output"]["final_artifact_path"])
        target = value["training"]["target_optimizer_steps"]
    except (KeyError, TypeError) as error:
        raise ControllerError("training config lacks required output/training fields") from error
    if not isinstance(run_id, str) or not run_id:
        raise ControllerError("training run_id is invalid")
    if not metrics.is_absolute() or not adapter.is_absolute():
        raise ControllerError("training output paths must be absolute")
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ControllerError("training target_optimizer_steps is invalid")
    raw_resume = value.get("resume", {}).get("checkpoint_path")
    resume_checkpoint = (
        None
        if raw_resume in (None, "none")
        else Path(raw_resume)
    )
    if resume_checkpoint is not None and not resume_checkpoint.is_absolute():
        raise ControllerError("training resume checkpoint path must be absolute")
    return {
        "run_id": run_id,
        "metrics_path": metrics,
        "adapter_path": adapter,
        "target_optimizer_steps": target,
        "resume_checkpoint_path": resume_checkpoint,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    _regular_file(path, label="training metrics ledger")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), 1):
        if not raw.strip():
            raise ControllerError(f"training metrics has a blank line at {line_number}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ControllerError(
                f"training metrics JSON is malformed at line {line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ControllerError(
                f"training metrics line {line_number} is not an object"
            )
        records.append(value)
    return records


def _validate_embedded_int_report(
    *,
    training: Any,
    terminal: Mapping[str, Any],
    adapter_manifest_sha256: str,
) -> dict[str, Any]:
    evaluation = training.post_training_internal_evaluation
    if evaluation is None or not evaluation.enabled:
        raise ControllerError("RP70 full-horizon training must embed INT-DIAG")
    report_path = evaluation.report_path
    if report_path is None:
        raise ControllerError("enabled embedded INT-DIAG has no report path")
    record = terminal.get("post_training_internal_evaluation")
    if not isinstance(record, dict) or record.get("status") != "complete":
        raise ControllerError("terminal training metrics has no complete embedded INT-DIAG")
    expected_path = str(report_path)
    if record.get("path") != expected_path:
        raise ControllerError("embedded INT-DIAG terminal path differs from config")
    _regular_file(report_path, label="embedded INT-DIAG report")
    payload_sha256 = _file_sha256(report_path)
    if record.get("payload_sha256") != payload_sha256:
        raise ControllerError("embedded INT-DIAG payload SHA256 differs")
    if record.get("byte_count") != report_path.stat().st_size:
        raise ControllerError("embedded INT-DIAG byte count differs")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControllerError("embedded INT-DIAG report is malformed") from error
    identity = report.get("identity") if isinstance(report, dict) else None
    expected_identity = {
        "schema_version": "representation_internal_evaluation_v1",
        "evaluation_id": evaluation.evaluation_id,
        "checkpoint_identity": adapter_manifest_sha256,
        "target_conditioning_provider": training.provider.provider.value,
        "random_seed": evaluation.random_seed,
        "prompt_identity": f"{training.prompt.identity}:{training.prompt.sha256}",
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != expected
        for key, expected in expected_identity.items()
    ):
        raise ControllerError("embedded INT-DIAG report identity differs")
    return {
        "status": "complete",
        "evaluation_id": evaluation.evaluation_id,
        "path": expected_path,
        "payload_sha256": payload_sha256,
        "byte_count": report_path.stat().st_size,
    }


def _load_training_completion(
    training_config_path: Path,
    *,
    expected_config_sha256: str,
    expected_global_step: int,
) -> dict[str, Any]:
    if _file_sha256(training_config_path) != expected_config_sha256:
        raise ControllerError("training config changed after controller startup")

    clean_src = str(CLEAN_RUNTIME / "src")
    if clean_src not in sys.path:
        sys.path.insert(0, clean_src)
    from tgvf_rl.checkpoint.coordinator import state_digest
    from tgvf_rl.representation.training.config import (
        load_representation_training_config,
    )
    from tgvf_rl.representation.training.distributed_checkpoint import (
        load_rank_zero_adapter_owned_state_export,
    )
    from tgvf_rl.representation.training.evaluation_runner import (
        _validate_training_artifact_binding,
    )

    training = load_representation_training_config(
        training_config_path,
        allow_existing_post_training_report=True,
    )
    mismatches: list[str] = []
    if training.training.target_optimizer_steps != expected_global_step:
        mismatches.append("target_optimizer_steps")
    if training.model.model_name != EXPECTED_MODEL:
        mismatches.append("model_name")
    if training.adapter_variant.value != EXPECTED_ADAPTER_VARIANT:
        mismatches.append("adapter_variant")
    if "answer-bearing-span" not in training.objective.objective.identity:
        mismatches.append("objective.identity")
    internal = training.post_training_internal_evaluation
    if internal is None or not internal.enabled:
        mismatches.append("post_training_internal_evaluation.enabled")
    if mismatches:
        raise ControllerError(
            "training config is not an accepted RP70 span run: "
            + ", ".join(mismatches)
        )

    metrics_path = training.output.metrics_jsonl_path
    if not metrics_path.exists():
        raise TrainingNotComplete("training metrics ledger does not exist yet")
    records = _load_jsonl(metrics_path)
    if not records:
        raise TrainingNotComplete("training metrics ledger is empty")
    complete = [record for record in records if record.get("event") == "complete"]
    if not complete:
        raise TrainingNotComplete("training has no terminal complete record")
    if len(complete) != 1 or records[-1] is not complete[0]:
        raise ControllerError(
            "training metrics requires exactly one terminal complete record"
        )
    terminal = complete[0]
    expected_terminal = {
        "status": "complete",
        "run_id": training.run_id,
        "global_step": expected_global_step,
        "source_toml_sha256": training.source_toml_sha256,
        "canonical_config_sha256": training.canonical_config_sha256,
        "final_artifact_path": str(training.output.final_artifact_path),
        "metrics_jsonl_path": str(metrics_path),
        "world_size": training.fsdp2.world_size,
        "physical_gpu_ids": list(training.fsdp2.physical_gpu_ids),
    }
    bad = sorted(
        key for key, expected in expected_terminal.items() if terminal.get(key) != expected
    )
    if bad:
        raise ControllerError(
            "terminal training metrics identity differs: " + ", ".join(bad)
        )
    adapter_path = training.output.final_artifact_path
    if not adapter_path.exists():
        raise TrainingNotComplete("terminal Adapter export does not exist yet")
    _regular_file(adapter_path, label="terminal Adapter export")
    export = load_rank_zero_adapter_owned_state_export(adapter_path)
    manifest = export.manifest
    manifest_sha256 = state_digest(manifest)
    if manifest.global_step != expected_global_step:
        raise ControllerError("Adapter export global step differs")
    if manifest.run_identity.run_id != training.run_id:
        raise ControllerError("Adapter export names another training run")
    manifest_variant = getattr(manifest.run_identity.adapter_contract, "variant", None)
    if manifest_variant not in (None, EXPECTED_ADAPTER_VARIANT):
        raise ControllerError("Adapter export names another structure")
    if len(manifest.tensor_names) != EXPECTED_TENSOR_COUNT:
        raise ControllerError("Adapter export tensor count differs")
    if terminal.get("run_identity_sha256") != manifest.run_identity_sha256:
        raise ControllerError("training metrics/Adapter run identity differs")
    if terminal.get("final_artifact_manifest_sha256") != manifest_sha256:
        raise ControllerError("training metrics/Adapter manifest differs")
    _validate_training_artifact_binding(training, manifest.run_identity)
    embedded_int = _validate_embedded_int_report(
        training=training,
        terminal=terminal,
        adapter_manifest_sha256=manifest_sha256,
    )

    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "training_completion_mode": "core_terminal_complete",
        "int_diag_mode": "embedded",
        "run_id": training.run_id,
        "global_step": expected_global_step,
        "run_identity_sha256": manifest.run_identity_sha256,
        "training_config_path": str(training_config_path),
        "training_config_sha256": training.source_toml_sha256,
        "training_code_commit": training.code.commit,
        "evaluation_data_path": str(training.data.validation.jsonl_path),
        "evaluation_data_source_sha256": training.data.validation.source_sha256,
        "conditioning_provider": training.provider.provider.value,
        "prompt_identity": training.prompt.identity,
        "prompt_sha256": training.prompt.sha256,
        "adapter_path": str(adapter_path),
        "adapter_file_sha256": _file_sha256(adapter_path),
        "adapter_manifest_sha256": manifest_sha256,
        "adapter_tensor_count": len(manifest.tensor_names),
        "metrics_path": str(metrics_path),
        "metrics_file_sha256": _file_sha256(metrics_path),
        "training_world_size": training.fsdp2.world_size,
        "training_physical_gpu_ids": list(training.fsdp2.physical_gpu_ids),
        "embedded_int_diag": embedded_int,
    }


_KNOWN_RP70_INT_FAILURE_SIGNATURES = (
    "run_post_training_internal_evaluation",
    "AnswerBearingSpanSupervisionError: span index has no record for sample '",
    "qwen3-v4-clean-imend-audited-grounding-v1:",
)


def _load_optimizer_complete_recovery(
    training_config_path: Path,
    *,
    expected_config_sha256: str,
    expected_global_step: int,
    failure_log_path: Path,
    external_int_report_path: Path,
) -> dict[str, Any]:
    """Authorize evaluation without inventing a core terminal completion.

    This accepts only the observed RP70 failure boundary: optimizer checkpoint,
    final validation, and Adapter export are all durable at the requested step,
    while the configured embedded INT report is absent and the owned training
    log contains the exact missing-grounding-sidecar exception.
    """

    if _file_sha256(training_config_path) != expected_config_sha256:
        raise ControllerError("training config changed after controller startup")
    clean_src = str(CLEAN_RUNTIME / "src")
    if clean_src not in sys.path:
        sys.path.insert(0, clean_src)
    from tgvf_rl.checkpoint.coordinator import state_digest
    from tgvf_rl.representation.training.config import (
        load_representation_training_config,
    )
    from tgvf_rl.representation.training.distributed_checkpoint import (
        load_distributed_representation_checkpoint_metadata,
        load_rank_zero_adapter_owned_state_export,
    )
    from tgvf_rl.representation.training.evaluation_runner import (
        _validate_training_artifact_binding,
    )

    training = load_representation_training_config(
        training_config_path,
        allow_existing_post_training_report=True,
    )
    mismatches: list[str] = []
    if training.training.target_optimizer_steps != expected_global_step:
        mismatches.append("target_optimizer_steps")
    if training.model.model_name != EXPECTED_MODEL:
        mismatches.append("model_name")
    if training.adapter_variant.value != EXPECTED_ADAPTER_VARIANT:
        mismatches.append("adapter_variant")
    if "answer-bearing-span" not in training.objective.objective.identity:
        mismatches.append("objective.identity")
    internal = training.post_training_internal_evaluation
    if internal is None or not internal.enabled:
        mismatches.append("post_training_internal_evaluation.enabled")
    if mismatches:
        raise ControllerError(
            "training config is not an accepted RP70 recovery source: "
            + ", ".join(mismatches)
        )
    assert internal is not None and internal.report_path is not None
    if internal.report_path.exists():
        raise ControllerError(
            "embedded INT report exists; optimizer-only recovery is not applicable"
        )

    metrics_path = training.output.metrics_jsonl_path
    if not metrics_path.exists():
        raise TrainingNotComplete("training metrics ledger does not exist yet")
    records = _load_jsonl(metrics_path)
    if not records:
        raise TrainingNotComplete("training metrics ledger is empty")
    if any(record.get("event") == "complete" for record in records):
        raise ControllerError("optimizer-only recovery refuses a core complete record")
    train_at_horizon = [
        record
        for record in records
        if record.get("event") == "train"
        and record.get("global_step") == expected_global_step
    ]
    train_steps = [
        record.get("global_step")
        for record in records
        if record.get("event") == "train"
        and isinstance(record.get("global_step"), int)
        and not isinstance(record.get("global_step"), bool)
    ]
    if (
        len(train_at_horizon) != 1
        or not train_steps
        or max(train_steps) != expected_global_step
        or records[-1].get("event") != "validation"
        or records[-1].get("global_step") != expected_global_step
        or records[-1].get("validation_event_index") != 0
    ):
        raise TrainingNotComplete(
            "optimizer horizon and terminal TRAIN-VAL are not both durable"
        )

    adapter_path = training.output.final_artifact_path
    if not adapter_path.exists():
        raise TrainingNotComplete("step-horizon Adapter export does not exist yet")
    _regular_file(adapter_path, label="step-horizon Adapter export")
    export = load_rank_zero_adapter_owned_state_export(adapter_path)
    adapter_manifest = export.manifest
    adapter_manifest_sha256 = state_digest(adapter_manifest)
    if (
        adapter_manifest.global_step != expected_global_step
        or adapter_manifest.run_identity.run_id != training.run_id
        or len(adapter_manifest.tensor_names) != EXPECTED_TENSOR_COUNT
    ):
        raise ControllerError("step-horizon Adapter export identity differs")
    _validate_training_artifact_binding(training, adapter_manifest.run_identity)

    checkpoint_path = training.checkpoint.directory / (
        f"{training.checkpoint.filename_prefix}-step-{expected_global_step:08d}"
    )
    if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
        raise TrainingNotComplete("step-horizon optimizer checkpoint is absent")
    checkpoint = load_distributed_representation_checkpoint_metadata(checkpoint_path)
    checkpoint_manifest = checkpoint.manifest
    if (
        checkpoint_manifest.global_step != expected_global_step
        or checkpoint_manifest.run_identity.run_id != training.run_id
        or checkpoint_manifest.run_identity_sha256
        != adapter_manifest.run_identity_sha256
        or checkpoint_manifest.run_identity != adapter_manifest.run_identity
        or checkpoint_manifest.world_size != training.fsdp2.world_size
        or tuple(checkpoint_manifest.owned_state_names)
        != tuple(adapter_manifest.tensor_names)
    ):
        raise ControllerError("optimizer checkpoint/Adapter identity differs")
    history = checkpoint_manifest.metrics_history
    if (
        history.run_id != training.run_id
        or history.run_identity_sha256 != adapter_manifest.run_identity_sha256
        or history.checkpoint_global_step != expected_global_step
        or history.byte_count <= 0
        or history.line_count <= 0
        or _file_prefix_sha256(metrics_path, history.byte_count)
        != history.raw_bytes_sha256
    ):
        raise ControllerError("optimizer checkpoint metrics-history binding differs")
    suffix = metrics_path.read_bytes()[history.byte_count :]
    try:
        suffix_records = [json.loads(line) for line in suffix.splitlines() if line]
    except json.JSONDecodeError as error:
        raise ControllerError("metrics suffix after optimizer checkpoint is malformed") from error
    if suffix_records != [records[-1]]:
        raise ControllerError(
            "metrics after optimizer checkpoint is not exactly terminal TRAIN-VAL"
        )

    metadata_path = checkpoint_path / "representation_metadata.pt"
    metadata_digest_path = checkpoint_path / "representation_metadata.sha256"
    _regular_file(metadata_path, label="optimizer checkpoint metadata")
    _regular_file(metadata_digest_path, label="optimizer checkpoint metadata digest")
    if metadata_digest_path.read_text(encoding="ascii").strip() != _file_sha256(
        metadata_path
    ):
        raise ControllerError("optimizer checkpoint metadata sidecar differs")
    checkpoint_files: list[dict[str, object]] = []
    for path in sorted(checkpoint_path.rglob("*")):
        if path.is_symlink():
            raise ControllerError("optimizer checkpoint contains a symlink")
        if path.is_file():
            record = _artifact_record(path)
            record["relative_path"] = str(path.relative_to(checkpoint_path))
            checkpoint_files.append(record)
    expected_dcp_shards = {
        f"dcp/__{rank}_0.distcp" for rank in range(training.fsdp2.world_size)
    }
    observed = {str(record["relative_path"]) for record in checkpoint_files}
    if not {
        "representation_metadata.pt",
        "representation_metadata.sha256",
        "dcp/.metadata",
        *expected_dcp_shards,
    }.issubset(observed):
        raise ControllerError("optimizer checkpoint files are incomplete")

    failure_log_path = failure_log_path.expanduser().resolve()
    _regular_file(failure_log_path, label="known embedded-INT failure log")
    failure_text = failure_log_path.read_text(encoding="utf-8", errors="replace")
    missing_signatures = tuple(
        signature
        for signature in _KNOWN_RP70_INT_FAILURE_SIGNATURES
        if signature not in failure_text
    )
    if missing_signatures:
        raise ControllerError(
            "training log does not prove the accepted embedded-INT failure: "
            + repr(missing_signatures)
        )
    if external_int_report_path == internal.report_path:
        raise ControllerError("external recovery INT must use a distinct report path")

    run_identity_sha256 = adapter_manifest.run_identity_sha256
    if any(
        record.get("run_identity_sha256") not in (None, run_identity_sha256)
        for record in records
    ):
        raise ControllerError("training metrics contains another run identity")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "optimizer_complete_recovery_authorized",
        "training_completion_mode": "optimizer_complete_without_core_terminal",
        "core_terminal_complete": False,
        "int_diag_mode": "external_recovery",
        "run_id": training.run_id,
        "global_step": expected_global_step,
        "run_identity_sha256": run_identity_sha256,
        "training_config_path": str(training_config_path),
        "training_config_sha256": training.source_toml_sha256,
        "training_code_commit": training.code.commit,
        "evaluation_data_path": str(training.data.validation.jsonl_path),
        "evaluation_data_source_sha256": training.data.validation.source_sha256,
        "conditioning_provider": training.provider.provider.value,
        "prompt_identity": training.prompt.identity,
        "prompt_sha256": training.prompt.sha256,
        "adapter_path": str(adapter_path),
        "adapter_file_sha256": _file_sha256(adapter_path),
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "adapter_tensor_count": len(adapter_manifest.tensor_names),
        "metrics_path": str(metrics_path),
        "metrics_file_sha256": _file_sha256(metrics_path),
        "training_world_size": training.fsdp2.world_size,
        "training_physical_gpu_ids": list(training.fsdp2.physical_gpu_ids),
        "optimizer_checkpoint": {
            "path": str(checkpoint_path),
            "global_step": checkpoint_manifest.global_step,
            "run_identity_sha256": checkpoint_manifest.run_identity_sha256,
            "metrics_history_identity_sha256": (
                checkpoint_manifest.metrics_history_identity_sha256
            ),
            "files": checkpoint_files,
        },
        "train_val": {
            "global_step": expected_global_step,
            "validation_event_index": records[-1]["validation_event_index"],
            "record_sha256": sha256(_canonical_bytes(records[-1])).hexdigest(),
        },
        "built_in_int_diag": {
            "status": "failed",
            "failure_kind": "rp70_span_sidecar_missing_grounding_sample",
            "configured_report_path": str(internal.report_path),
            "report_exists": False,
            "failure_log": _artifact_record(failure_log_path),
            "required_signatures": list(_KNOWN_RP70_INT_FAILURE_SIGNATURES),
        },
        "external_int_diag": {
            "status": "planned",
            "evaluation_id": (
                f"rp70-step{expected_global_step:04d}-external-int-recovery-v1"
            ),
            "path": str(external_int_report_path),
        },
    }


def _load_accepted_training_state(
    training_config_path: Path,
    *,
    expected_config_sha256: str,
    expected_global_step: int,
    recover_external_int: bool,
    failure_log_path: Path | None,
    external_int_report_path: Path,
) -> dict[str, Any]:
    try:
        return _load_training_completion(
            training_config_path,
            expected_config_sha256=expected_config_sha256,
            expected_global_step=expected_global_step,
        )
    except TrainingNotComplete as original:
        if not recover_external_int:
            raise
        if failure_log_path is None:
            raise ControllerError(
                "external-INT recovery requires --known-int-failure-log"
            ) from original
        return _load_optimizer_complete_recovery(
            training_config_path,
            expected_config_sha256=expected_config_sha256,
            expected_global_step=expected_global_step,
            failure_log_path=failure_log_path,
            external_int_report_path=external_int_report_path,
        )


def _default_output_root(training_config_path: Path) -> Path:
    layout = _raw_training_layout(training_config_path)
    target = int(layout["target_optimizer_steps"])
    resume_checkpoint = layout["resume_checkpoint_path"]
    source_step: int | None = None
    if isinstance(resume_checkpoint, Path):
        match = re.search(r"-step-(\d+)$", resume_checkpoint.name)
        if match is not None:
            source_step = int(match.group(1))
    name = f"rp70_step{target:04d}_post_training"
    if source_step is not None:
        name = f"rp70_step{target:04d}_from_step{source_step:04d}_post_training"
    return REPOSITORY_ROOT / (
        "artifacts/representation_experiments/answer_bearing_span/evaluation"
    ) / name


def _paths(output_root: Path) -> dict[str, Path]:
    generated = output_root / "generated-configs"
    return {
        "root": output_root,
        "lock": output_root / "pipeline.lock",
        "state": output_root / "state.json",
        "events": output_root / "events.jsonl",
        "receipt": output_root / "training-completion-receipt.json",
        "int_config": generated / "rp70-terminal-external-int-recovery.toml",
        "first_config": generated / "rp70-terminal-acc-first200.toml",
        "full_config": generated / "rp70-terminal-acc-full867.toml",
        "int_report": output_root / "INT-DIAG-first200-external-recovery.json",
        "int_log": output_root / "INT-DIAG-first200-external-recovery.log",
        "first_generation": output_root / "ACC-VAL-first200-generation-3arm",
        "full_generation": output_root / "ACC-VAL-full867-generation-3arm",
        "first_generation_log": output_root / "ACC-VAL-first200-generation-3arm.log",
        "full_generation_log": output_root / "ACC-VAL-full867-generation-3arm.log",
        "first_semantic": output_root / "ACC-VAL-first200-semantic-3arm",
        "full_semantic": output_root / "ACC-VAL-full867-semantic-3arm",
        "first_semantic_log": output_root / "ACC-VAL-first200-semantic-3arm.log",
        "full_semantic_log": output_root / "ACC-VAL-full867-semantic-3arm.log",
        "judge_log": output_root / "semantic-judge.log",
        "judge_ownership": output_root / "semantic-judge-ownership.json",
        "complete": output_root / "complete.json",
    }


def _toml_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _render_evaluation_config(
    *,
    receipt: Mapping[str, Any],
    run_id: str,
    evaluation_id: str,
    manifest_path: Path,
    manifest_sha256: str,
    report_path: Path,
    physical_gpu_id: int,
) -> bytes:
    return f'''schema_version = "representation-internal-evaluation-run-v3"
run_id = {_toml_string(run_id)}

[code]
repository = "Miocio-nora/TGVF-E2E-RL"
commit = {_toml_string(CLEAN_RUNTIME_COMMIT)}
dirty = false

[source]
training_config_path = {_toml_string(receipt["training_config_path"])}
training_config_sha256 = {_toml_string(receipt["training_config_sha256"])}

[artifact]
path = {_toml_string(receipt["adapter_path"])}
file_sha256 = {_toml_string(receipt["adapter_file_sha256"])}
manifest_sha256 = {_toml_string(receipt["adapter_manifest_sha256"])}
expected_run_identity_sha256 = {_toml_string(receipt["run_identity_sha256"])}
expected_global_step = {receipt["global_step"]}

[execution]
physical_gpu_id = {physical_gpu_id}

[evaluation_data]
jsonl_path = {_toml_string(receipt["evaluation_data_path"])}
source_sha256 = {_toml_string(receipt["evaluation_data_source_sha256"])}

[evaluation]
evaluation_id = {_toml_string(evaluation_id)}
ordered_group_manifest_path = {_toml_string(manifest_path)}
ordered_group_manifest_sha256 = {_toml_string(manifest_sha256)}
counterfactual_manifest_path = {_toml_string(COUNTERFACTUAL_MANIFEST)}
counterfactual_manifest_sha256 = {_toml_string(COUNTERFACTUAL_MANIFEST_SHA256)}
grounding_manifest_path = {_toml_string(GROUNDING_MANIFEST)}
grounding_manifest_sha256 = {_toml_string(GROUNDING_MANIFEST_SHA256)}
report_path = {_toml_string(report_path)}
random_seed = 42
max_new_tokens = 64
eos_token_ids = [151645]
'''.encode("utf-8")


def _materialize_configs(
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    int_gpu: int,
    first_gpus: tuple[int, ...],
    full_gpus: tuple[int, ...],
) -> None:
    run_slug = str(receipt["run_id"])
    step = int(receipt["global_step"])
    _write_identical_or_new(
        paths["receipt"],
        (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    if receipt.get("int_diag_mode") == "external_recovery":
        external = receipt.get("external_int_diag")
        if not isinstance(external, dict):
            raise ControllerError("external INT recovery receipt is malformed")
        _write_identical_or_new(
            paths["int_config"],
            _render_evaluation_config(
                receipt=receipt,
                run_id=f"{run_slug}-STEP{step}-EXTERNAL-INT-RECOVERY-GPU{int_gpu}",
                evaluation_id=str(external["evaluation_id"]),
                manifest_path=FIRST_MANIFEST,
                manifest_sha256=FIRST_MANIFEST_SHA256,
                report_path=Path(str(external["path"])),
                physical_gpu_id=int_gpu,
            ),
        )
    _write_identical_or_new(
        paths["first_config"],
        _render_evaluation_config(
            receipt=receipt,
            run_id=f"{run_slug}-STEP{step}-FIRST200-ACC-VAL",
            evaluation_id=f"rp70-step{step:04d}-first200-acc-val-v1",
            manifest_path=FIRST_MANIFEST,
            manifest_sha256=FIRST_MANIFEST_SHA256,
            report_path=paths["root"] / "first200-internal-unused.json",
            physical_gpu_id=first_gpus[0],
        ),
    )
    _write_identical_or_new(
        paths["full_config"],
        _render_evaluation_config(
            receipt=receipt,
            run_id=f"{run_slug}-STEP{step}-FULL867-ACC-VAL",
            evaluation_id=f"rp70-step{step:04d}-full867-acc-val-v1",
            manifest_path=FULL_MANIFEST,
            manifest_sha256=FULL_MANIFEST_SHA256,
            report_path=paths["root"] / "full867-internal-unused.json",
            physical_gpu_id=full_gpus[0],
        ),
    )


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(CLEAN_RUNTIME / "src"),
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def _int_report_path(receipt: Mapping[str, Any]) -> Path:
    mode = receipt.get("int_diag_mode")
    key = "embedded_int_diag" if mode == "embedded" else "external_int_diag"
    record = receipt.get(key)
    if not isinstance(record, dict):
        raise ControllerError("training receipt has no INT-DIAG specification")
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        raise ControllerError("INT-DIAG report path is not absolute")
    return path


def _int_report_current(receipt: Mapping[str, Any]) -> bool:
    path = _int_report_path(receipt)
    if not path.exists() and receipt.get("int_diag_mode") == "external_recovery":
        return False
    _regular_file(path, label="INT-DIAG report")
    mode = receipt.get("int_diag_mode")
    record_key = "embedded_int_diag" if mode == "embedded" else "external_int_diag"
    record = receipt.get(record_key)
    assert isinstance(record, dict)
    if mode == "embedded" and (
        record.get("payload_sha256") != _file_sha256(path)
        or record.get("byte_count") != path.stat().st_size
    ):
        raise ControllerError("embedded INT-DIAG receipt drifted")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControllerError("INT-DIAG report is malformed") from error
    identity = report.get("identity") if isinstance(report, dict) else None
    expected = {
        "schema_version": "representation_internal_evaluation_v1",
        "evaluation_id": record.get("evaluation_id"),
        "checkpoint_identity": receipt["adapter_manifest_sha256"],
        "target_conditioning_provider": receipt["conditioning_provider"],
        "random_seed": 42,
        "prompt_identity": f"{receipt['prompt_identity']}:{receipt['prompt_sha256']}",
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != value for key, value in expected.items()
    ):
        raise ControllerError("INT-DIAG report identity differs")
    return True


def _generation_current(
    root: Path, *, samples: int, shard_count: int
) -> dict[str, Any] | None:
    summary_path = root / "launch-summary.json"
    if not summary_path.exists():
        return None
    _regular_file(summary_path, label="ACC generation summary")
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControllerError("ACC generation summary is malformed") from error
    if (
        value.get("schema_version") != GENERATION_SCHEMA
        or value.get("status") != "complete"
        or tuple(value.get("arms", ())) != ARMS
        or value.get("sample_count") != samples
        or value.get("record_count") != samples * len(ARMS)
        or value.get("shard_count") != shard_count
    ):
        raise ControllerError(f"ACC generation summary differs: {summary_path}")
    shards = value.get("shards")
    if not isinstance(shards, list) or len(shards) != shard_count:
        raise ControllerError("ACC generation shard receipt count differs")
    for shard in shards:
        if not isinstance(shard, dict):
            raise ControllerError("ACC generation shard receipt is malformed")
        shard_root = Path(str(shard.get("output_root", "")))
        records = shard_root / "records.jsonl"
        summary = shard_root / "summary.json"
        identity = shard_root / "identity.json"
        for path in (records, summary, identity):
            _regular_file(path, label="ACC generation shard artifact")
        if shard.get("records_jsonl_sha256") != _file_sha256(records):
            raise ControllerError("ACC generation shard records SHA256 differs")
    return value


def _semantic_current(root: Path, *, samples: int) -> bool:
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    if not summary_path.exists() and not manifest_path.exists():
        return False
    _regular_file(summary_path, label="semantic summary")
    _regular_file(manifest_path, label="semantic manifest")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControllerError("semantic publication is malformed") from error
    by_arm = summary.get("by_arm")
    files = manifest.get("files")
    run_identity = summary.get("run_identity_sha256")
    if (
        summary.get("schema_version") != SEMANTIC_SCHEMA
        or manifest.get("schema_version") != SEMANTIC_SCHEMA
        or summary.get("status") != "complete"
        or manifest.get("status") != "complete"
        or not isinstance(run_identity, str)
        or len(run_identity) != 64
        or manifest.get("run_identity_sha256") != run_identity
        or not isinstance(by_arm, dict)
        or set(by_arm) != set(ARMS)
        or summary.get("overall", {}).get("total") != samples * len(ARMS)
        or any(
            not isinstance(by_arm.get(arm), dict) or by_arm[arm].get("total") != samples
            for arm in ARMS
        )
        or not isinstance(files, dict)
        or files.get("summary", {}).get("sha256") != _file_sha256(summary_path)
        or files.get("overlay_records", {}).get("rows") != samples * len(ARMS)
    ):
        raise ControllerError(f"semantic publication differs: {root}")
    return True


def _gpu_compute_pids(indices: Sequence[int]) -> dict[int, tuple[int, ...]]:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    uuid_to_index = {
        uuid.strip(): int(index.strip())
        for index, uuid in (row.split(",", 1) for row in gpu_rows)
    }
    result: dict[int, list[int]] = {index: [] for index in indices}
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    for row in rows:
        if not row.strip():
            continue
        pid_text, uuid = (item.strip() for item in row.split(",", 1))
        index = uuid_to_index.get(uuid)
        if index in result:
            result[index].append(int(pid_text))
    return {index: tuple(sorted(pids)) for index, pids in result.items()}


def _wait_gpus_empty(
    indices: Sequence[int], *, timeout_seconds: float, paths: Mapping[str, Path]
) -> None:
    unique = tuple(dict.fromkeys(indices))
    deadline = time.monotonic() + timeout_seconds
    last_busy: dict[int, tuple[int, ...]] = {}
    while time.monotonic() < deadline:
        busy = {gpu: pids for gpu, pids in _gpu_compute_pids(unique).items() if pids}
        if not busy:
            return
        if busy != last_busy:
            _append_event(paths["events"], "waiting_for_gpus", busy=busy)
            last_busy = busy
        time.sleep(5)
    raise ControllerError(f"GPUs did not become compute-empty: {last_busy}")


def _owned_session_pids(session_id: int) -> tuple[int, ...]:
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            value = (entry / "stat").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        closing = value.rfind(")")
        fields = value[closing + 2 :].split()
        if (
            len(fields) > 3
            and fields[0] not in {"Z", "X"}
            and int(fields[3]) == session_id
        ):
            result.append(int(entry.name))
    return tuple(sorted(result))


def _owned_worker_pids_from_plan(plan_path: Path) -> tuple[int, ...]:
    if not plan_path.exists():
        return ()
    _regular_file(plan_path, label="answer-utility launch plan")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControllerError("answer-utility launch plan is malformed") from error
    assignments = plan.get("assignments") if isinstance(plan, dict) else None
    if not isinstance(assignments, list):
        raise ControllerError("answer-utility launch plan has no assignments")
    expected_commands: set[tuple[str, ...]] = set()
    for assignment in assignments:
        command = assignment.get("command") if isinstance(assignment, dict) else None
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) for item in command)
        ):
            raise ControllerError("answer-utility launch assignment is malformed")
        expected_commands.add(tuple(command))
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            cwd = Path(os.readlink(entry / "cwd")).resolve()
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if cwd != CLEAN_RUNTIME.resolve() or not raw:
            continue
        argv = tuple(
            item.decode("utf-8", errors="surrogateescape")
            for item in raw.rstrip(b"\0").split(b"\0")
        )
        if argv in expected_commands:
            result.append(int(entry.name))
    return tuple(sorted(result))


def _stop_owned_worker_session(worker_pid: int) -> tuple[int, ...]:
    observed = set(_owned_session_pids(worker_pid))
    if worker_pid not in observed:
        return tuple(sorted(observed))
    try:
        os.killpg(worker_pid, signal.SIGTERM)
    except ProcessLookupError:
        return tuple(sorted(observed))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        remaining = _owned_session_pids(worker_pid)
        observed.update(remaining)
        if not remaining:
            return tuple(sorted(observed))
        time.sleep(0.25)
    remaining = _owned_session_pids(worker_pid)
    try:
        os.killpg(worker_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return tuple(sorted(observed))


def _stop_process_group(process: subprocess.Popen[bytes]) -> tuple[int, ...]:
    """Reap a complete owned session even after its launcher has exited."""

    session_id = process.pid
    observed = set(_owned_session_pids(session_id))
    try:
        os.killpg(session_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        remaining = _owned_session_pids(session_id)
        observed.update(remaining)
        if not remaining:
            break
        time.sleep(0.25)
    remaining = _owned_session_pids(session_id)
    if remaining:
        try:
            os.killpg(session_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and _owned_session_pids(session_id):
            time.sleep(0.25)
    if process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    return tuple(sorted(observed))


def _int_command(config: Path) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "tgvf_rl.cli",
        "run-representation-internal-evaluation",
        str(config),
    ]


def _generation_command(
    *, config: Path, root: Path, gpu_ids: Sequence[int]
) -> list[str]:
    command = [
        str(PYTHON),
        str(CLEAN_LAUNCHER),
        "--production-source",
        "--source-evaluation-config",
        str(config),
        "--output-root",
        str(root),
    ]
    for gpu in gpu_ids:
        command.extend(("--physical-gpu-id", str(gpu)))
    command.extend(
        (
            "--workers-per-gpu",
            "1",
            "--eos-token-id",
            "151645",
            "--eos-token-id",
            "151643",
            "--decode-mode",
            "cached",
            "--arm-batch-size",
            "1",
        )
    )
    for arm in ARMS:
        command.extend(("--arm", arm))
    return command


def _run_parallel_generations(
    *,
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
    int_gpu: int,
    first_gpus: tuple[int, ...],
    full_gpus: tuple[int, ...],
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = _generation_current(
        paths["first_generation"], samples=200, shard_count=len(first_gpus)
    )
    full = _generation_current(
        paths["full_generation"], samples=867, shard_count=len(full_gpus)
    )
    jobs: list[dict[str, Any]] = []
    specifications: list[tuple[str, bool, list[str], Path]] = []
    if receipt.get("int_diag_mode") == "external_recovery":
        specifications.append(
            (
                "external_int_diag",
                _int_report_current(receipt),
                _int_command(paths["int_config"]),
                paths["int_log"],
            )
        )
    elif not _int_report_current(receipt):
        raise ControllerError("embedded INT-DIAG report is incomplete")
    specifications.extend(
        (
        (
            "first200_generation",
            first is not None,
            _generation_command(
                config=paths["first_config"],
                root=paths["first_generation"],
                gpu_ids=first_gpus,
            ),
            paths["first_generation_log"],
        ),
        (
            "full867_generation",
            full is not None,
            _generation_command(
                config=paths["full_config"],
                root=paths["full_generation"],
                gpu_ids=full_gpus,
            ),
            paths["full_generation_log"],
        ),
        )
    )
    try:
        for name, done, command, log_path in specifications:
            if done:
                _append_event(paths["events"], "stage_already_complete", stage=name)
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("ab", buffering=0)
            environment = _clean_environment()
            if name == "external_int_diag":
                environment.update(
                    {
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "CUDA_VISIBLE_DEVICES": str(int_gpu),
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    }
                )
            process = subprocess.Popen(
                command,
                cwd=CLEAN_RUNTIME,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            worker_plan = None
            if name == "first200_generation":
                worker_plan = paths["first_generation"] / "launch-plan.json"
            elif name == "full867_generation":
                worker_plan = paths["full_generation"] / "launch-plan.json"
            jobs.append(
                {
                    "name": name,
                    "process": process,
                    "log": log,
                    "log_path": log_path,
                    "observed_child_pids": set(),
                    "worker_plan": worker_plan,
                }
            )
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
                children = {
                    pid
                    for pid in _owned_session_pids(process.pid)
                    if pid != process.pid
                }
                if job["worker_plan"] is not None:
                    children.update(
                        _owned_worker_pids_from_plan(job["worker_plan"])
                    )
                new_children = children - job["observed_child_pids"]
                if new_children:
                    job["observed_child_pids"].update(new_children)
                    _append_event(
                        paths["events"],
                        "stage_owned_children_observed",
                        stage=job["name"],
                        launcher_pid=process.pid,
                        child_pids=sorted(new_children),
                    )
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
                raise ControllerError("parallel evaluation stage timed out")
            if pending:
                time.sleep(2)
    finally:
        for job in jobs:
            process = job["process"]
            live_worker_pids: tuple[int, ...] = ()
            if job["worker_plan"] is not None:
                live_worker_pids = _owned_worker_pids_from_plan(job["worker_plan"])
                for worker_pid in live_worker_pids:
                    _stop_owned_worker_session(worker_pid)
            owned_pids = _stop_process_group(process)
            all_child_pids = tuple(
                sorted(
                    {
                        *job["observed_child_pids"],
                        *(pid for pid in owned_pids if pid != process.pid),
                    }
                )
            )
            live_at_cleanup = tuple(pid for pid in owned_pids if pid != process.pid)
            if all_child_pids:
                _append_event(
                    paths["events"],
                    "stage_owned_children_finalized",
                    stage=job["name"],
                    launcher_pid=process.pid,
                    observed_child_pids=list(all_child_pids),
                    live_at_cleanup_pids=list(
                        sorted({*live_at_cleanup, *live_worker_pids})
                    ),
                )
            if not job["log"].closed:
                job["log"].close()

    if not _int_report_current(receipt):
        raise ControllerError("external INT-DIAG exited without a complete report")
    first = _generation_current(
        paths["first_generation"], samples=200, shard_count=len(first_gpus)
    )
    full = _generation_current(
        paths["full_generation"], samples=867, shard_count=len(full_gpus)
    )
    if first is None or full is None:
        raise ControllerError("ACC generation exited without complete summaries")
    return first, full


def _python_header_cpath() -> str:
    python_headers = PYTHON_HEADER_ROOT / "python3.12"
    required = (
        python_headers / "Python.h",
        python_headers / "pyconfig.h",
        PYTHON_HEADER_ROOT / "x86_64-linux-gnu/python3.12/pyconfig.h",
    )
    missing = tuple(str(path) for path in required if not path.is_file())
    if missing:
        raise ControllerError(f"Python development headers missing: {missing}")
    return os.pathsep.join((str(PYTHON_HEADER_ROOT), str(python_headers)))


def _judge_settings() -> dict[str, Any]:
    value = json.loads(JUDGE_CONFIG.read_text(encoding="utf-8"))
    try:
        model_path = Path(value["model"]["local_path"])
        served_name = value["model"]["served_name"]
        base_url = value["service"]["base_url"]
        integration_devices = tuple(value["service"]["integration_devices"])
    except (KeyError, TypeError) as error:
        raise ControllerError("semantic judge config is malformed") from error
    parsed = urlparse(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port is None
        or integration_devices != (0, 1)
    ):
        raise ControllerError("semantic judge service binding differs")
    _regular_file(model_path / "config.json", label="semantic judge model config")
    return {
        "model_path": model_path,
        "served_name": served_name,
        "port": parsed.port,
    }


def _endpoint_open(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1) as response:
            response.read(1)
        return True
    except Exception:
        return False


def _judge_command(settings: Mapping[str, Any]) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(settings["model_path"]),
        "--served-model-name",
        str(settings["served_name"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(settings["port"]),
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.85",
        "--max-num-seqs",
        "64",
        "--seed",
        "42",
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
    ]


def _judge_environment(
    *, judge_gpus: tuple[int, int], paths: Mapping[str, Path]
) -> dict[str, str]:
    environment = _clean_environment()
    triton = paths["root"] / "cache/judge-triton"
    inductor = paths["root"] / "cache/judge-torchinductor"
    triton.mkdir(parents=True, exist_ok=True)
    inductor.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in judge_gpus),
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "CPATH": _python_header_cpath(),
            "LIBRARY_PATH": str(REPOSITORY_ROOT / ".venv312/lib"),
            "TRITON_CACHE_DIR": str(triton),
            "TORCHINDUCTOR_CACHE_DIR": str(inductor),
        }
    )
    return environment


def _wait_judge_ready(
    process: subprocess.Popen[bytes], *, settings: Mapping[str, Any], timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ControllerError(
                f"semantic judge exited during startup with {process.returncode}"
            )
        try:
            with urlopen(
                f"http://127.0.0.1:{settings['port']}/v1/models", timeout=2
            ) as response:
                value = json.loads(response.read())
            if {item.get("id") for item in value.get("data", [])} == {
                settings["served_name"]
            }:
                return
        except Exception:
            pass
        time.sleep(2)
    raise ControllerError("semantic judge readiness timed out")


def _semantic_command(
    *,
    generation: Mapping[str, Any],
    source_config: Path,
    output_root: Path,
) -> list[str]:
    command = [str(PYTHON), str(CLEAN_SEMANTIC_TOOL)]
    shards = generation.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ControllerError("generation summary has no semantic input shards")
    for shard in shards:
        command.extend(("--generation-output-root", str(shard["output_root"])))
    command.extend(
        (
            "--source-evaluation-config",
            str(source_config),
            "--judge-config",
            str(JUDGE_CONFIG),
            "--judge-config-sha256",
            JUDGE_CONFIG_SHA256,
            "--output-root",
            str(output_root),
            "--concurrency",
            "32",
        )
    )
    return command


def _run_semantic(
    *,
    name: str,
    generation: Mapping[str, Any],
    config: Path,
    output_root: Path,
    samples: int,
    log_path: Path,
    paths: Mapping[str, Path],
) -> None:
    if _semantic_current(output_root, samples=samples):
        _append_event(paths["events"], "semantic_already_complete", split=name)
        return
    command = _semantic_command(
        generation=generation, source_config=config, output_root=output_root
    )
    _append_event(paths["events"], "semantic_started", split=name, command=command)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=CLEAN_RUNTIME,
            env={**_clean_environment(), "CUDA_VISIBLE_DEVICES": ""},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=2 * 60 * 60)
        except BaseException:
            _stop_process_group(process)
            raise
    if return_code != 0 or not _semantic_current(output_root, samples=samples):
        raise ControllerError(f"{name} semantic rescore failed; inspect {log_path}")
    _append_event(paths["events"], "semantic_complete", split=name)


def _run_semantic_pair(
    *,
    first: Mapping[str, Any],
    full: Mapping[str, Any],
    paths: Mapping[str, Path],
    judge_gpus: tuple[int, int],
    gpu_wait_timeout: float,
) -> None:
    first_done = _semantic_current(paths["first_semantic"], samples=200)
    full_done = _semantic_current(paths["full_semantic"], samples=867)
    if first_done and full_done:
        return
    settings = _judge_settings()
    if _endpoint_open(int(settings["port"])):
        raise ControllerError("semantic judge port is occupied by an unowned endpoint")
    _wait_gpus_empty(
        judge_gpus, timeout_seconds=gpu_wait_timeout, paths=paths
    )
    log = paths["judge_log"].open("ab", buffering=0)
    judge = subprocess.Popen(
        _judge_command(settings),
        cwd=CLEAN_RUNTIME,
        env=_judge_environment(judge_gpus=judge_gpus, paths=paths),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    ownership = {
        "schema_version": "rp70-owned-semantic-judge-v1",
        "status": "starting",
        "started_at": _utc_now(),
        "pid": judge.pid,
        "pgid": judge.pid,
        "gpu_ids": list(judge_gpus),
        "port": settings["port"],
        "runtime_commit": CLEAN_RUNTIME_COMMIT,
        "judge_config_sha256": JUDGE_CONFIG_SHA256,
    }
    _atomic_json(paths["judge_ownership"], ownership)
    _append_event(paths["events"], "judge_started", pid=judge.pid)
    try:
        _wait_judge_ready(judge, settings=settings, timeout=600)
        ownership["status"] = "ready"
        ownership["ready_at"] = _utc_now()
        _atomic_json(paths["judge_ownership"], ownership)
        _run_semantic(
            name="first200",
            generation=first,
            config=paths["first_config"],
            output_root=paths["first_semantic"],
            samples=200,
            log_path=paths["first_semantic_log"],
            paths=paths,
        )
        _run_semantic(
            name="full867",
            generation=full,
            config=paths["full_config"],
            output_root=paths["full_semantic"],
            samples=867,
            log_path=paths["full_semantic_log"],
            paths=paths,
        )
    finally:
        _stop_process_group(judge)
        log.close()
        ownership["status"] = "stopped"
        ownership["stopped_at"] = _utc_now()
        ownership["returncode"] = judge.returncode
        _atomic_json(paths["judge_ownership"], ownership)
        _append_event(
            paths["events"],
            "judge_stopped",
            pid=judge.pid,
            returncode=judge.returncode,
        )
    _wait_gpus_empty(judge_gpus, timeout_seconds=300, paths=paths)


def _complete_marker_current(
    path: Path, *, receipt: Mapping[str, Any], paths: Mapping[str, Path]
) -> bool:
    if not path.exists():
        return False
    _regular_file(path, label="pipeline completion marker")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControllerError("pipeline completion marker is malformed") from error
    generated_configs = value.get("generated_configs")
    first200 = value.get("first200")
    full867 = value.get("full867")
    if not all(isinstance(item, dict) for item in (generated_configs, first200, full867)):
        raise ControllerError("pipeline completion marker sections are malformed")
    int_report = _int_report_path(receipt)
    recovery_config_current = True
    if receipt.get("int_diag_mode") == "external_recovery":
        recovery_config_current = _artifact_record_current(
            generated_configs.get("external_int_diag"), paths["int_config"]
        )
    if (
        value.get("schema_version") != COMPLETE_SCHEMA
        or value.get("status") != "complete"
        or value.get("training_run_id") != receipt["run_id"]
        or value.get("training_run_identity_sha256") != receipt["run_identity_sha256"]
        or value.get("adapter_manifest_sha256") != receipt["adapter_manifest_sha256"]
        or value.get("training_completion_mode")
        != receipt.get("training_completion_mode")
        or value.get("int_diag_mode") != receipt.get("int_diag_mode")
        or tuple(value.get("arms", ())) != ARMS
        or not _artifact_record_current(value.get("receipt"), paths["receipt"])
        or not _artifact_record_current(
            generated_configs.get("first200"), paths["first_config"]
        )
        or not _artifact_record_current(
            generated_configs.get("full867"), paths["full_config"]
        )
        or not recovery_config_current
        or not _artifact_record_current(value.get("int_diag"), int_report)
        or not _artifact_record_current(
            first200.get("generation"),
            paths["first_generation"] / "launch-summary.json",
        )
        or not _artifact_record_current(
            first200.get("semantic_summary"),
            paths["first_semantic"] / "summary.json",
        )
        or not _artifact_record_current(
            first200.get("semantic_manifest"),
            paths["first_semantic"] / "manifest.json",
        )
        or not _artifact_record_current(
            full867.get("generation"),
            paths["full_generation"] / "launch-summary.json",
        )
        or not _artifact_record_current(
            full867.get("semantic_summary"),
            paths["full_semantic"] / "summary.json",
        )
        or not _artifact_record_current(
            full867.get("semantic_manifest"),
            paths["full_semantic"] / "manifest.json",
        )
    ):
        raise ControllerError("pipeline completion marker drifted")
    return True


def _write_complete_marker(
    *, receipt: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    int_report = _int_report_path(receipt)
    generated_configs = {
        "first200": _artifact_record(paths["first_config"]),
        "full867": _artifact_record(paths["full_config"]),
    }
    if receipt.get("int_diag_mode") == "external_recovery":
        generated_configs["external_int_diag"] = _artifact_record(paths["int_config"])
    marker = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "complete",
        "completed_at": _utc_now(),
        "runtime_commit": CLEAN_RUNTIME_COMMIT,
        "controller_file_sha256": _file_sha256(Path(__file__).resolve()),
        "clean_launcher_sha256": _file_sha256(CLEAN_LAUNCHER),
        "clean_semantic_tool_sha256": _file_sha256(CLEAN_SEMANTIC_TOOL),
        "training_run_id": receipt["run_id"],
        "training_run_identity_sha256": receipt["run_identity_sha256"],
        "adapter_manifest_sha256": receipt["adapter_manifest_sha256"],
        "training_completion_mode": receipt.get("training_completion_mode"),
        "core_terminal_complete": receipt.get("core_terminal_complete", True),
        "int_diag_mode": receipt.get("int_diag_mode"),
        "built_in_int_diag": receipt.get("built_in_int_diag"),
        "arms": list(ARMS),
        "receipt": _artifact_record(paths["receipt"]),
        "generated_configs": generated_configs,
        "int_diag": _artifact_record(int_report),
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
        "full867": {
            "samples": 867,
            "generation": _artifact_record(
                paths["full_generation"] / "launch-summary.json"
            ),
            "semantic_summary": _artifact_record(
                paths["full_semantic"] / "summary.json"
            ),
            "semantic_manifest": _artifact_record(
                paths["full_semantic"] / "manifest.json"
            ),
        },
    }
    _atomic_json(paths["complete"], marker)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ControllerError("another RP70 evaluation controller owns the lock") from error
        yield


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


def _wait_training(
    training_config_path: Path,
    *,
    expected_config_sha256: str,
    expected_global_step: int,
    poll_seconds: float,
    timeout_seconds: float,
    paths: Mapping[str, Path],
    recover_external_int: bool,
    failure_log_path: Path | None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_reason: str | None = None
    while True:
        try:
            receipt = _load_accepted_training_state(
                training_config_path,
                expected_config_sha256=expected_config_sha256,
                expected_global_step=expected_global_step,
                recover_external_int=recover_external_int,
                failure_log_path=failure_log_path,
                external_int_report_path=paths["int_report"],
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
                    "expected_global_step": expected_global_step,
                    "reason": reason,
                    "updated_at": _utc_now(),
                },
            )
        if time.monotonic() >= deadline:
            raise ControllerError("timed out waiting for terminal training completion")
        time.sleep(poll_seconds)


def _parse_gpu_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from error
    if not values or len(set(values)) != len(values) or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("GPU IDs must be unique non-negative integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--expected-global-step", type=int, default=2000)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=7 * 24 * 60 * 60)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument("--evaluation-timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument(
        "--recover-external-int",
        action="store_true",
        help=(
            "Authorize the audited optimizer-complete/embedded-INT-failed recovery "
            "boundary instead of requiring a core terminal complete record."
        ),
    )
    parser.add_argument(
        "--known-int-failure-log",
        type=Path,
        help="Exact failed training log required by --recover-external-int.",
    )
    parser.add_argument(
        "--int-gpu",
        type=int,
        default=0,
        help="Deprecated compatibility option; INT-DIAG is embedded by training.",
    )
    parser.add_argument("--first-gpus", type=_parse_gpu_list, default=(1, 2))
    parser.add_argument("--full-gpus", type=_parse_gpu_list, default=(3, 4, 5, 6, 7))
    parser.add_argument("--judge-gpus", type=_parse_gpu_list, default=(6, 7))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    run = subparsers.add_parser("run")
    run.add_argument("--execute", action="store_true", required=True)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.expected_global_step <= 0:
        raise ControllerError("expected global step must be positive")
    for name in (
        "poll_seconds",
        "wait_timeout_seconds",
        "gpu_wait_timeout_seconds",
        "evaluation_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            raise ControllerError(f"{name} must be positive")
    if args.int_gpu < 0:
        raise ControllerError("int GPU must be non-negative")
    if args.recover_external_int != (args.known_int_failure_log is not None):
        raise ControllerError(
            "--recover-external-int and --known-int-failure-log must be supplied together"
        )
    if len(args.judge_gpus) != 2:
        raise ControllerError("semantic judge requires exactly two physical GPUs")
    if set(args.first_gpus).intersection(args.full_gpus):
        raise ControllerError("first200 and full867 generation GPUs must be disjoint")
    if args.recover_external_int and args.int_gpu in {
        *args.first_gpus,
        *args.full_gpus,
    }:
        raise ControllerError("external INT GPU must be disjoint from ACC generation GPUs")
    # Semantic scoring starts only after every generation worker has exited, so
    # the judge GPUs do not need to be members of the full-generation pool.
    # Keeping the pools artificially coupled can strand a restart even when a
    # different pair of GPUs is completely idle.


def _status(
    *,
    training_config_path: Path,
    expected_config_sha256: str,
    expected_global_step: int,
    paths: Mapping[str, Path],
    first_gpus: tuple[int, ...],
    full_gpus: tuple[int, ...],
    recover_external_int: bool,
    failure_log_path: Path | None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": PIPELINE_SCHEMA,
        "status": "waiting",
        "training_config_path": str(training_config_path),
        "training_config_sha256": expected_config_sha256,
        "expected_global_step": expected_global_step,
        "output_root": str(paths["root"]),
        "training_complete": False,
        "int_diag_complete": False,
        "first200_generation_complete": False,
        "full867_generation_complete": False,
        "first200_semantic_complete": False,
        "full867_semantic_complete": False,
        "pipeline_complete": False,
        "external_int_recovery_requested": recover_external_int,
    }
    try:
        receipt = _load_accepted_training_state(
            training_config_path,
            expected_config_sha256=expected_config_sha256,
            expected_global_step=expected_global_step,
            recover_external_int=recover_external_int,
            failure_log_path=failure_log_path,
            external_int_report_path=paths["int_report"],
        )
        value["training_complete"] = True
        value["training_completion_mode"] = receipt["training_completion_mode"]
        value["training_run_id"] = receipt["run_id"]
        value["training_run_identity_sha256"] = receipt["run_identity_sha256"]
        value["int_diag_complete"] = _int_report_current(receipt)
        value["first200_generation_complete"] = _generation_current(
            paths["first_generation"], samples=200, shard_count=len(first_gpus)
        ) is not None
        value["full867_generation_complete"] = _generation_current(
            paths["full_generation"], samples=867, shard_count=len(full_gpus)
        ) is not None
        value["first200_semantic_complete"] = _semantic_current(
            paths["first_semantic"], samples=200
        )
        value["full867_semantic_complete"] = _semantic_current(
            paths["full_semantic"], samples=867
        )
        if paths["receipt"].is_file():
            value["pipeline_complete"] = _complete_marker_current(
                paths["complete"], receipt=receipt, paths=paths
            )
        value["status"] = "complete" if value["pipeline_complete"] else "ready"
    except TrainingNotComplete as error:
        value["waiting_reason"] = str(error)
    except (ControllerError, FileNotFoundError, ValueError) as error:
        value["status"] = "blocked"
        value["blocked_reason"] = str(error)
    return value


def _run(args: argparse.Namespace) -> dict[str, Any]:
    training_config_path = args.training_config.expanduser().resolve()
    _assert_static_inputs()
    _raw_training_layout(training_config_path)
    expected_config_sha256 = _file_sha256(training_config_path)
    output_root = (
        _default_output_root(training_config_path)
        if args.output_root is None
        else args.output_root.expanduser().resolve()
    )
    paths = _paths(output_root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(paths["lock"]):
        _append_event(
            paths["events"],
            "controller_started",
            pid=os.getpid(),
            training_config_path=str(training_config_path),
            training_config_sha256=expected_config_sha256,
            expected_global_step=args.expected_global_step,
        )
        try:
            receipt = _wait_training(
                training_config_path,
                expected_config_sha256=expected_config_sha256,
                expected_global_step=args.expected_global_step,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.wait_timeout_seconds,
                paths=paths,
                recover_external_int=args.recover_external_int,
                failure_log_path=args.known_int_failure_log,
            )
            _materialize_configs(
                receipt,
                paths,
                int_gpu=args.int_gpu,
                first_gpus=args.first_gpus,
                full_gpus=args.full_gpus,
            )
            if _complete_marker_current(paths["complete"], receipt=receipt, paths=paths):
                return _status(
                    training_config_path=training_config_path,
                    expected_config_sha256=expected_config_sha256,
                    expected_global_step=args.expected_global_step,
                    paths=paths,
                    first_gpus=args.first_gpus,
                    full_gpus=args.full_gpus,
                    recover_external_int=args.recover_external_int,
                    failure_log_path=args.known_int_failure_log,
                )
            _atomic_json(
                paths["state"],
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "status": "waiting_for_generation_gpus",
                    "pid": os.getpid(),
                    "updated_at": _utc_now(),
                },
            )
            # A resumed controller must not wait for GPUs belonging to stages
            # whose durable outputs have already been validated.  Only reserve
            # devices for work that still needs to run.
            generation_gpus: list[int] = []
            if (
                receipt.get("int_diag_mode") == "external_recovery"
                and not _int_report_current(receipt)
            ):
                generation_gpus.append(args.int_gpu)
            if _generation_current(
                paths["first_generation"],
                samples=200,
                shard_count=len(args.first_gpus),
            ) is None:
                generation_gpus.extend(args.first_gpus)
            if _generation_current(
                paths["full_generation"],
                samples=867,
                shard_count=len(args.full_gpus),
            ) is None:
                generation_gpus.extend(args.full_gpus)
            if generation_gpus:
                _wait_gpus_empty(
                    generation_gpus,
                    timeout_seconds=args.gpu_wait_timeout_seconds,
                    paths=paths,
                )
            _atomic_json(
                paths["state"],
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "status": "running_external_int_and_acc_generation",
                    "pid": os.getpid(),
                    "updated_at": _utc_now(),
                },
            )
            first, full = _run_parallel_generations(
                receipt=receipt,
                paths=paths,
                int_gpu=args.int_gpu,
                first_gpus=args.first_gpus,
                full_gpus=args.full_gpus,
                timeout_seconds=args.evaluation_timeout_seconds,
            )
            _atomic_json(
                paths["state"],
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "status": "running_semantic_rescore",
                    "pid": os.getpid(),
                    "updated_at": _utc_now(),
                },
            )
            _run_semantic_pair(
                first=first,
                full=full,
                paths=paths,
                judge_gpus=args.judge_gpus,
                gpu_wait_timeout=args.gpu_wait_timeout_seconds,
            )
            if not _semantic_current(
                paths["first_semantic"], samples=200
            ) or not _semantic_current(paths["full_semantic"], samples=867):
                raise ControllerError("semantic publications are incomplete")
            _write_complete_marker(receipt=receipt, paths=paths)
            _atomic_json(
                paths["state"],
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "status": "complete",
                    "pid": os.getpid(),
                    "marker": str(paths["complete"]),
                    "completed_at": _utc_now(),
                },
            )
            _append_event(paths["events"], "controller_complete")
        except BaseException as error:
            _atomic_json(
                paths["state"],
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "status": "failed",
                    "pid": os.getpid(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_at": _utc_now(),
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
        expected_global_step=args.expected_global_step,
        paths=paths,
        first_gpus=args.first_gpus,
        full_gpus=args.full_gpus,
        recover_external_int=args.recover_external_int,
        failure_log_path=args.known_int_failure_log,
    )


def main() -> int:
    args = _parser().parse_args()
    _validate_arguments(args)
    training_config_path = args.training_config.expanduser().resolve()
    _assert_static_inputs()
    _raw_training_layout(training_config_path)
    expected_config_sha256 = _file_sha256(training_config_path)
    output_root = (
        _default_output_root(training_config_path)
        if args.output_root is None
        else args.output_root.expanduser().resolve()
    )
    paths = _paths(output_root)
    if args.command == "status":
        result = _status(
            training_config_path=training_config_path,
            expected_config_sha256=expected_config_sha256,
            expected_global_step=args.expected_global_step,
            paths=paths,
            first_gpus=args.first_gpus,
            full_gpus=args.full_gpus,
            recover_external_int=args.recover_external_int,
            failure_log_path=args.known_int_failure_log,
        )
    else:
        previous = _install_signal_handlers()
        try:
            result = _run(args)
        finally:
            _restore_signal_handlers(previous)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ControllerError as error:
        print(f"RP70_EVALUATION_CONTROLLER_BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(3) from error
