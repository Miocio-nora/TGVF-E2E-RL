#!/usr/bin/python3 -I
"""Wait for RP69 and run its isolated, hash-bound post-training evaluation.

This controller accepts exactly the 500-step Qwen3-VL-8B-Instruct RP69
``full_d_deepstack_visual_barycentric`` treatment.  It does not treat process
exit as scientific completion: the terminal metrics record, Adapter export,
run identity, manifest, training configuration, and evaluation inputs are all
verified before any GPU evaluation starts.

Once training is complete it materializes three immutable evaluation TOMLs:

* first-200 INT-DIAG;
* first-200 three-arm ACC-VAL;
* full-867 three-arm ACC-VAL.

INT-DIAG and both ACC generation jobs run concurrently on disjoint GPUs.  The
semantic judge starts only after all generation workers have exited.  Runtime
code is taken from the clean, frozen RP70 evaluation worktree because that
commit contains the RP69 visual-barycentric adapter implementation.  Outputs
are restartable and a completion marker is published only after every hash-
bound artifact has been revalidated.

``dry-run`` performs static validation and prints the exact process plan.  It
does not create output directories, inspect GPU occupancy, or start a process.
"""

from __future__ import annotations
# ruff: noqa: E402

# Direct script execution is stopped before legacy path/environment mutation or
# heavyweight runtime imports. Importing the module for read-only compatibility
# tests remains possible; its public ``main`` retains a second fail-closed guard.
if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range(2):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            "tools/run_rp69_post_training_evaluations.py",
        ),
    )

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)

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

DEFAULT_TRAINING_CONFIG = REPOSITORY_ROOT / (
    "configs/representation/experiments/visual_barycentric/"
    "rp69_qwen3_instruct_visual_barycentric_500_gpu0123.toml"
)
EXPECTED_RUN_ID = "RP-69-QWEN3-INSTRUCT-REP-BALANCED-T1-VISUAL-BARYCENTRIC-500-GPU0123"
EXPECTED_MODEL = "Qwen3-VL-8B-Instruct"
EXPECTED_VARIANT = "full_d_deepstack_visual_barycentric"
EXPECTED_OBJECTIVE = "balanced-matrix-ce-l-gen-norm-v1"
EXPECTED_STEP = 500
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
    "5f444fa49346f395954da2b38b4261099599990238ccc3a1ccc5af7e13d86ee4"
)
EXPECTED_ARTIFACT_ROOT = REPOSITORY_ROOT / (
    "artifacts/representation/"
    "RP-69-qwen3-instruct-balanced-t1-visual-barycentric-500-gpu0123"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "artifacts/representation_experiments/visual_barycentric/evaluation/"
    "rp69_step0500_gpu0123"
)

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
JUDGE_CONFIG_SHA256 = "3737504858912a6392679d2c9720597cde58dd7d3218aa6f75b67ad00a769573"

PIPELINE_SCHEMA = "rp69-post-training-evaluations-v1"
RECEIPT_SCHEMA = "rp69-training-completion-receipt-v1"
COMPLETE_SCHEMA = "rp69-post-training-evaluations-complete-v1"
GENERATION_SCHEMA = "answer_utility_multi_worker_launch_result_v1"
SEMANTIC_SCHEMA = "answer-utility-semantic-rescore-v2"
ARMS = (
    "image_only",
    "image_correct_D",
    "image_same_target_wrong_image_D",
)


class ControllerError(RuntimeError):
    """Fail-closed RP69 controller error."""


class TrainingNotComplete(ControllerError):
    """The accepted RP69 terminal boundary has not appeared yet."""


class ControllerInterrupted(ControllerError):
    """The controller received SIGINT or SIGTERM."""


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


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ControllerError(f"{label} is not one regular file: {path}")


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ControllerError(
            f"git {' '.join(arguments)} failed in {root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
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
    with path.open("ab", buffering=0) as handle:
        handle.write(
            _canonical_bytes({"at": _utc_now(), "event": event, **fields}) + b"\n"
        )
        os.fsync(handle.fileno())


def _artifact_record(path: Path) -> dict[str, object]:
    _regular_file(path, label="completion artifact")
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _artifact_record_current(record: object, path: Path) -> bool:
    return (
        isinstance(record, dict)
        and path.is_file()
        and not path.is_symlink()
        and record.get("path") == str(path.resolve())
        and record.get("sha256") == _file_sha256(path)
        and record.get("bytes") == path.stat().st_size
    )


def _load_toml(path: Path) -> dict[str, Any]:
    _regular_file(path, label="RP69 training config")
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ControllerError("RP69 training config is unreadable") from error
    if not isinstance(value, dict):
        raise ControllerError("RP69 training config is not a TOML table")
    return value


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
            "resume.enabled": (resume["enabled"], False),
            "post_training_internal_evaluation.enabled": (post["enabled"], False),
        }
        metrics_path = Path(output["metrics_jsonl_path"])
        adapter_path = Path(output["final_artifact_path"])
        checkpoint_directory = Path(checkpoint["directory"])
    except (KeyError, TypeError) as error:
        raise ControllerError("RP69 training config lacks required fields") from error
    mismatches = [
        name for name, (observed, expected) in checks.items() if observed != expected
    ]
    if adapter_path != EXPECTED_ARTIFACT_ROOT / "adapter.pt":
        mismatches.append("output.final_artifact_path")
    if metrics_path != EXPECTED_ARTIFACT_ROOT / "metrics.jsonl":
        mismatches.append("output.metrics_jsonl_path")
    if checkpoint_directory != EXPECTED_ARTIFACT_ROOT / "checkpoints":
        mismatches.append("checkpoint.directory")
    if mismatches:
        raise ControllerError(
            "training config is not the isolated RP69 treatment: "
            + ", ".join(sorted(mismatches))
        )
    return {
        "run_id": EXPECTED_RUN_ID,
        "adapter_path": adapter_path,
        "metrics_path": metrics_path,
        "checkpoint_directory": checkpoint_directory,
        "target_optimizer_steps": EXPECTED_STEP,
    }


def _assert_static_inputs(training_config_path: Path) -> dict[str, Any]:
    layout = _static_training_layout(training_config_path)
    if _file_sha256(training_config_path) != EXPECTED_TRAINING_CONFIG_SHA256:
        raise ControllerError(
            "RP69 training TOML differs from the exact configuration already launched"
        )
    for path, expected, label in (
        (FIRST_MANIFEST, FIRST_MANIFEST_SHA256, "first-200 manifest"),
        (FULL_MANIFEST, FULL_MANIFEST_SHA256, "full-867 manifest"),
        (
            COUNTERFACTUAL_MANIFEST,
            COUNTERFACTUAL_MANIFEST_SHA256,
            "counterfactual manifest",
        ),
        (GROUNDING_MANIFEST, GROUNDING_MANIFEST_SHA256, "grounding manifest"),
        (JUDGE_CONFIG, JUDGE_CONFIG_SHA256, "semantic judge config"),
    ):
        _regular_file(path, label=label)
        if _file_sha256(path) != expected:
            raise ControllerError(f"pinned {label} SHA256 differs")
    if not PYTHON.is_file():
        raise ControllerError(f"Python runtime is missing: {PYTHON}")
    _regular_file(CLEAN_LAUNCHER, label="frozen answer-utility launcher")
    _regular_file(CLEAN_SEMANTIC_TOOL, label="frozen semantic-rescore tool")
    if _git_output(CLEAN_RUNTIME, "rev-parse", "HEAD") != CLEAN_RUNTIME_COMMIT:
        raise ControllerError("frozen evaluation runtime commit differs")
    if _git_output(CLEAN_RUNTIME, "status", "--porcelain"):
        raise ControllerError("frozen evaluation runtime is dirty")
    if EXPECTED_VARIANT not in (
        CLEAN_RUNTIME / "src/tgvf_rl/representation/adapter.py"
    ).read_text(encoding="utf-8"):
        raise ControllerError("frozen runtime lacks the RP69 adapter variant")
    return layout


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    _regular_file(path, label="RP69 metrics ledger")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), 1):
        if not raw.strip():
            raise ControllerError(f"metrics has a blank line at {line_number}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ControllerError(
                f"metrics JSON is malformed at line {line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ControllerError(f"metrics line {line_number} is not an object")
        records.append(value)
    return records


def _load_training_completion(
    training_config_path: Path,
    *,
    expected_config_sha256: str,
    int_report_path: Path,
) -> dict[str, Any]:
    if _file_sha256(training_config_path) != expected_config_sha256:
        raise ControllerError("training config changed after controller startup")
    layout = _static_training_layout(training_config_path)
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

    training = load_representation_training_config(training_config_path)
    metrics_path = Path(layout["metrics_path"])
    if not metrics_path.exists():
        raise TrainingNotComplete("training metrics ledger does not exist yet")
    records = _load_jsonl(metrics_path)
    if not records:
        raise TrainingNotComplete("training metrics ledger is empty")
    complete = [record for record in records if record.get("event") == "complete"]
    if not complete:
        raise TrainingNotComplete("training has no terminal complete record")
    if len(complete) != 1 or records[-1] is not complete[0]:
        raise ControllerError("metrics requires exactly one terminal complete record")
    terminal = complete[0]
    expected_terminal = {
        "status": "complete",
        "run_id": EXPECTED_RUN_ID,
        "global_step": EXPECTED_STEP,
        "source_toml_sha256": training.source_toml_sha256,
        "canonical_config_sha256": training.canonical_config_sha256,
        "final_artifact_path": str(training.output.final_artifact_path),
        "metrics_jsonl_path": str(metrics_path),
        "world_size": EXPECTED_WORLD_SIZE,
        "physical_gpu_ids": list(EXPECTED_TRAIN_GPUS),
    }
    bad = sorted(
        key
        for key, expected in expected_terminal.items()
        if terminal.get(key) != expected
    )
    if bad:
        raise ControllerError("terminal metrics identity differs: " + ", ".join(bad))
    if terminal.get("post_training_internal_evaluation") != {"status": "disabled"}:
        raise ControllerError("terminal metrics unexpectedly embeds INT-DIAG")

    adapter_path = Path(layout["adapter_path"])
    if not adapter_path.exists():
        raise TrainingNotComplete("terminal Adapter export does not exist yet")
    _regular_file(adapter_path, label="terminal RP69 Adapter export")
    export = load_rank_zero_adapter_owned_state_export(adapter_path)
    manifest = export.manifest
    manifest_sha256 = state_digest(manifest)
    variant = getattr(manifest.run_identity.adapter_contract, "variant", None)
    if manifest.global_step != EXPECTED_STEP:
        raise ControllerError("Adapter export global step differs")
    if manifest.run_identity.run_id != EXPECTED_RUN_ID:
        raise ControllerError("Adapter export names another run")
    if variant != EXPECTED_VARIANT:
        raise ControllerError("Adapter export is not visual-barycentric RP69")
    if len(manifest.tensor_names) != EXPECTED_TENSOR_COUNT:
        raise ControllerError("Adapter export tensor count differs")
    if terminal.get("run_identity_sha256") != manifest.run_identity_sha256:
        raise ControllerError("metrics/Adapter run identity differs")
    if terminal.get("final_artifact_manifest_sha256") != manifest_sha256:
        raise ControllerError("metrics/Adapter manifest identity differs")
    _validate_training_artifact_binding(training, manifest.run_identity)

    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "training_completion_mode": "core_terminal_complete",
        "int_diag_mode": "external",
        "run_id": EXPECTED_RUN_ID,
        "global_step": EXPECTED_STEP,
        "run_identity_sha256": manifest.run_identity_sha256,
        "training_config_path": str(training_config_path),
        "training_config_sha256": training.source_toml_sha256,
        "training_code_commit": training.code.commit,
        "evaluation_runtime_commit": CLEAN_RUNTIME_COMMIT,
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
        "training_world_size": EXPECTED_WORLD_SIZE,
        "training_physical_gpu_ids": list(EXPECTED_TRAIN_GPUS),
        "external_int_diag": {
            "status": "planned",
            "evaluation_id": "rp69-step0500-first200-int-diag-v1",
            "path": str(int_report_path),
        },
    }


def _default_output_root() -> Path:
    return DEFAULT_OUTPUT_ROOT


def _training_protected_paths(
    training_layout: Mapping[str, Any],
) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            (
                Path(training_layout["adapter_path"]).resolve().parent,
                Path(training_layout["metrics_path"]).resolve().parent,
                Path(training_layout["checkpoint_directory"]).resolve(),
            )
        )
    )


def _assert_output_root_isolated(
    output_root: Path, training_layout: Mapping[str, Any]
) -> None:
    """Reject evaluation roots that overlap any RP69 training artifact tree."""

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
            "evaluation output root overlaps RP69 training/artifact paths: "
            f"output={output}; protected=" + ",".join(str(path) for path in conflicts)
        )


def _paths(root: Path) -> dict[str, Path]:
    generated = root / "generated-configs"
    return {
        "root": root,
        "lock": root / "pipeline.lock",
        "state": root / "state.json",
        "events": root / "events.jsonl",
        "receipt": root / "training-completion-receipt.json",
        "int_config": generated / "rp69-step0500-int-first200.toml",
        "first_config": generated / "rp69-step0500-acc-first200.toml",
        "full_config": generated / "rp69-step0500-acc-full867.toml",
        "int_report": root / "INT-DIAG-first200-step0500.json",
        "int_log": root / "INT-DIAG-first200-step0500.log",
        "first_generation": root / "ACC-VAL-first200-generation-3arm",
        "full_generation": root / "ACC-VAL-full867-generation-3arm",
        "first_generation_log": root / "ACC-VAL-first200-generation-3arm.log",
        "full_generation_log": root / "ACC-VAL-full867-generation-3arm.log",
        "first_semantic": root / "ACC-VAL-first200-semantic-3arm",
        "full_semantic": root / "ACC-VAL-full867-semantic-3arm",
        "first_semantic_log": root / "ACC-VAL-first200-semantic-3arm.log",
        "full_semantic_log": root / "ACC-VAL-full867-semantic-3arm.log",
        "judge_log": root / "semantic-judge.log",
        "judge_ownership": root / "semantic-judge-ownership.json",
        "complete": root / "complete.json",
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
    return f"""schema_version = "representation-internal-evaluation-run-v3"
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
expected_global_step = {EXPECTED_STEP}

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
""".encode("utf-8")


def _materialize_configs(
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    int_gpu: int,
    first_gpus: tuple[int, ...],
    full_gpus: tuple[int, ...],
) -> None:
    external = receipt.get("external_int_diag")
    if not isinstance(external, dict):
        raise ControllerError("RP69 receipt has no external INT-DIAG plan")
    _write_identical_or_new(
        paths["receipt"],
        (
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )
    specifications = (
        (
            paths["int_config"],
            _render_evaluation_config(
                receipt=receipt,
                run_id=f"{EXPECTED_RUN_ID}-STEP500-INT-FIRST200-GPU{int_gpu}",
                evaluation_id=str(external["evaluation_id"]),
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
                run_id=f"{EXPECTED_RUN_ID}-STEP500-ACC-FIRST200",
                evaluation_id="rp69-step0500-first200-acc-val-v1",
                manifest_path=FIRST_MANIFEST,
                manifest_sha256=FIRST_MANIFEST_SHA256,
                report_path=paths["root"] / "first200-internal-unused.json",
                physical_gpu_id=first_gpus[0],
            ),
        ),
        (
            paths["full_config"],
            _render_evaluation_config(
                receipt=receipt,
                run_id=f"{EXPECTED_RUN_ID}-STEP500-ACC-FULL867",
                evaluation_id="rp69-step0500-full867-acc-val-v1",
                manifest_path=FULL_MANIFEST,
                manifest_sha256=FULL_MANIFEST_SHA256,
                report_path=paths["root"] / "full867-internal-unused.json",
                physical_gpu_id=full_gpus[0],
            ),
        ),
    )
    for path, payload in specifications:
        _write_identical_or_new(path, payload)


def _int_report_current(receipt: Mapping[str, Any], paths: Mapping[str, Path]) -> bool:
    path = paths["int_report"]
    if not path.exists():
        return False
    _regular_file(path, label="RP69 INT-DIAG report")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControllerError("RP69 INT-DIAG report is malformed") from error
    external = receipt.get("external_int_diag")
    identity = report.get("identity") if isinstance(report, dict) else None
    expected = {
        "schema_version": "representation_internal_evaluation_v1",
        "evaluation_id": (
            external.get("evaluation_id") if isinstance(external, dict) else None
        ),
        "checkpoint_identity": receipt["adapter_manifest_sha256"],
        "target_conditioning_provider": receipt["conditioning_provider"],
        "random_seed": 42,
        "prompt_identity": f"{receipt['prompt_identity']}:{receipt['prompt_sha256']}",
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != value for key, value in expected.items()
    ):
        raise ControllerError("RP69 INT-DIAG report identity differs")
    return True


def _generation_current(
    root: Path,
    *,
    samples: int,
    gpu_ids: tuple[int, ...],
    receipt: Mapping[str, Any],
    source_config: Path,
) -> dict[str, Any] | None:
    shard_count = len(gpu_ids)
    summary_path = root / "launch-summary.json"
    if not summary_path.exists():
        return None
    _regular_file(summary_path, label="ACC generation summary")
    plan_path = root / "launch-plan.json"
    _regular_file(plan_path, label="ACC generation launch plan")
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControllerError("ACC generation publication is malformed") from error
    if not isinstance(plan, dict):
        raise ControllerError("ACC generation launch plan is not an object")
    declared_plan_sha256 = plan.get("plan_sha256")
    plan_without_digest = dict(plan)
    plan_without_digest.pop("plan_sha256", None)
    computed_plan_sha256 = sha256(_canonical_bytes(plan_without_digest)).hexdigest()
    if (
        value.get("schema_version") != GENERATION_SCHEMA
        or value.get("status") != "complete"
        or tuple(value.get("arms", ())) != ARMS
        or value.get("sample_count") != samples
        or value.get("record_count") != samples * len(ARMS)
        or value.get("shard_count") != shard_count
        or value.get("plan_sha256") != declared_plan_sha256
        or declared_plan_sha256 != computed_plan_sha256
        or plan.get("schema_version") != "answer_utility_multi_worker_launch_plan_v1"
        or Path(str(plan.get("output_root", ""))).resolve() != root.resolve()
        or tuple(plan.get("physical_gpu_ids", ())) != gpu_ids
        or plan.get("workers_per_gpu") != 1
    ):
        raise ControllerError(f"ACC generation plan/summary differs: {root}")
    whole = plan.get("whole_preflight")
    expected_manifest_sha256 = (
        FIRST_MANIFEST_SHA256 if samples == 200 else FULL_MANIFEST_SHA256
    )
    expected_binding = {
        "candidate_artifact_file_sha256": receipt["adapter_file_sha256"],
        "candidate_global_step": EXPECTED_STEP,
        "candidate_training_run_identity_sha256": receipt["run_identity_sha256"],
        "production_source_artifact_sha256": receipt["adapter_file_sha256"],
        "production_source_global_step": EXPECTED_STEP,
        "production_source_manifest_sha256": receipt["adapter_manifest_sha256"],
        "production_source_run_identity_sha256": receipt["run_identity_sha256"],
    }
    if (
        not isinstance(whole, dict)
        or tuple(whole.get("arms", ())) != ARMS
        or whole.get("selected_sample_count") != samples
        or any(whole.get(key) != expected for key, expected in expected_binding.items())
    ):
        raise ControllerError("ACC generation whole preflight names another source")
    shards = value.get("shards")
    assignments = plan.get("assignments")
    if (
        not isinstance(shards, list)
        or len(shards) != shard_count
        or not isinstance(assignments, list)
        or len(assignments) != shard_count
    ):
        raise ControllerError("ACC generation shard receipt count differs")
    source_config = source_config.resolve()
    selected_samples = 0
    evaluation_manifest_sha256: str | None = None
    for shard_index, (shard, assignment) in enumerate(zip(shards, assignments)):
        if not isinstance(shard, dict) or not isinstance(assignment, dict):
            raise ControllerError("ACC generation shard receipt is malformed")
        shard_root = Path(str(shard.get("output_root", "")))
        records = shard_root / "records.jsonl"
        summary = shard_root / "summary.json"
        identity = shard_root / "identity.json"
        for path in (records, summary, identity):
            _regular_file(path, label="ACC generation shard artifact")
        if shard.get("records_jsonl_sha256") != _file_sha256(records):
            raise ControllerError("ACC generation shard records SHA256 differs")
        try:
            identity_value = json.loads(identity.read_text(encoding="utf-8"))
            shard_summary = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ControllerError("ACC generation shard JSON is malformed") from error
        shard_identity = (
            identity_value.get("identity") if isinstance(identity_value, dict) else None
        )
        preflight = assignment.get("preflight")
        command = assignment.get("command")
        if (
            not isinstance(shard_identity, dict)
            or not isinstance(preflight, dict)
            or not isinstance(command, list)
            or not all(isinstance(item, str) for item in command)
        ):
            raise ControllerError("ACC generation shard identity is malformed")
        try:
            config_position = command.index("--source-evaluation-config") + 1
            command_source_config = Path(command[config_position]).resolve()
        except (ValueError, IndexError) as error:
            raise ControllerError("ACC worker command has no source config") from error
        expected_shard = {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "physical_gpu_id": gpu_ids[shard_index],
        }
        expected_shard_receipt = {
            "shard_index": shard_index,
            "physical_gpu_id": gpu_ids[shard_index],
        }
        expected_shard_root = root / f"shard-{shard_index:04d}-of-{shard_count:04d}"
        if (
            command_source_config != source_config
            or any(
                assignment.get(key) != expected
                for key, expected in expected_shard.items()
            )
            or any(
                shard.get(key) != expected
                for key, expected in expected_shard_receipt.items()
            )
            or shard_root.resolve() != expected_shard_root.resolve()
            or Path(str(assignment.get("output_root", ""))).resolve()
            != expected_shard_root.resolve()
            or any(
                preflight.get(key) != expected
                for key, expected in expected_binding.items()
            )
            or any(
                shard_identity.get(key) != expected
                for key, expected in expected_binding.items()
            )
            or shard_identity.get("ordered_group_manifest_sha256")
            != expected_manifest_sha256
            or shard_summary.get("run_identity_sha256")
            != shard.get("run_identity_sha256")
            or identity_value.get("identity_sha256") != shard.get("run_identity_sha256")
            or shard_summary.get("records_jsonl_sha256") != _file_sha256(records)
        ):
            raise ControllerError("ACC generation shard names another RP69 source")
        shard_data_manifest = shard_identity.get("data_manifest_sha256")
        if not isinstance(shard_data_manifest, str) or len(shard_data_manifest) != 64:
            raise ControllerError("ACC shard data manifest identity is invalid")
        if evaluation_manifest_sha256 is None:
            evaluation_manifest_sha256 = shard_data_manifest
        elif shard_data_manifest != evaluation_manifest_sha256:
            raise ControllerError("ACC shards name different evaluation data")
        selected_samples += int(shard.get("sample_count", -1))
    if selected_samples != samples:
        raise ControllerError("ACC shard sample counts do not cover the split")
    return value


def _semantic_current(
    root: Path,
    *,
    samples: int,
    generation: Mapping[str, Any] | None,
    source_config: Path,
) -> bool:
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
    semantic_identity = manifest.get("identity")
    run_identity = summary.get("run_identity_sha256")
    if generation is None:
        raise ControllerError("semantic publication exists without current generation")
    generation_shards = generation.get("shards")
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
        or not isinstance(semantic_identity, dict)
        or not isinstance(generation_shards, list)
        or semantic_identity.get("source_evaluation_config_path")
        != str(source_config.resolve())
        or semantic_identity.get("source_evaluation_config_sha256")
        != _file_sha256(source_config)
        or semantic_identity.get("judge", {}).get("config_file_sha256")
        != JUDGE_CONFIG_SHA256
    ):
        raise ControllerError(f"semantic publication differs: {root}")
    generation_sources = semantic_identity.get("generation_sources")
    if not isinstance(generation_sources, list) or len(generation_sources) != len(
        generation_shards
    ):
        raise ControllerError("semantic generation-source count differs")
    sources_by_root = {
        str(source.get("root")): source
        for source in generation_sources
        if isinstance(source, dict)
    }
    data_manifest_sha256: str | None = None
    for shard in generation_shards:
        if not isinstance(shard, dict):
            raise ControllerError("semantic source generation shard is malformed")
        shard_root = Path(str(shard.get("output_root", ""))).resolve()
        source = sources_by_root.get(str(shard_root))
        if not isinstance(source, dict):
            raise ControllerError("semantic publication omits one generation shard")
        identity_path = shard_root / "identity.json"
        records_path = shard_root / "records.jsonl"
        shard_summary_path = shard_root / "summary.json"
        for path in (identity_path, records_path, shard_summary_path):
            _regular_file(path, label="semantic generation source")
        identity_value = json.loads(identity_path.read_text(encoding="utf-8"))
        shard_identity = (
            identity_value.get("identity") if isinstance(identity_value, dict) else None
        )
        if not isinstance(shard_identity, dict):
            raise ControllerError("semantic generation identity is malformed")
        observed_data_manifest = shard_identity.get("data_manifest_sha256")
        if data_manifest_sha256 is None:
            data_manifest_sha256 = observed_data_manifest
        elif observed_data_manifest != data_manifest_sha256:
            raise ControllerError("semantic generation sources use different data")
        if (
            source.get("generation_identity_sha256") != shard.get("run_identity_sha256")
            or source.get("record_count") != shard.get("record_count")
            or source.get("records_file_sha256") != _file_sha256(records_path)
            or source.get("identity_file_sha256") != _file_sha256(identity_path)
            or source.get("summary_file_sha256") != _file_sha256(shard_summary_path)
        ):
            raise ControllerError("semantic publication names another generation")
    if semantic_identity.get("evaluation_data_manifest_sha256") != data_manifest_sha256:
        raise ControllerError("semantic publication names another evaluation dataset")
    for name in ("overlay_records", "blind_requests", "judge_evidence"):
        record = files.get(name)
        if not isinstance(record, dict):
            raise ControllerError(f"semantic manifest omits {name}")
        artifact = root / str(record.get("path", ""))
        _regular_file(artifact, label=f"semantic {name}")
        if record.get("sha256") != _file_sha256(artifact):
            raise ControllerError(f"semantic {name} SHA256 differs")
    return True


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
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
    """Find live worker sessions whose exact argv appears in our launch plan."""

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


def _stop_owned_worker_session(worker_pid: int) -> None:
    if not _owned_session_pids(worker_pid):
        return
    try:
        os.killpg(worker_pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not _owned_session_pids(worker_pid):
            return
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


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None and not _owned_session_pids(process.pid):
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and _owned_session_pids(process.pid):
        time.sleep(0.25)
    remaining = _owned_session_pids(process.pid)
    if remaining:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _run_parallel_generation(
    *,
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
    int_gpu: int,
    first_gpus: tuple[int, ...],
    full_gpus: tuple[int, ...],
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = _generation_current(
        paths["first_generation"],
        samples=200,
        gpu_ids=first_gpus,
        receipt=receipt,
        source_config=paths["first_config"],
    )
    full = _generation_current(
        paths["full_generation"],
        samples=867,
        gpu_ids=full_gpus,
        receipt=receipt,
        source_config=paths["full_config"],
    )
    specifications = (
        (
            "int_diag",
            _int_report_current(receipt, paths),
            _int_command(paths["int_config"]),
            paths["int_log"],
            int_gpu,
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
            None,
        ),
    )
    jobs: list[dict[str, Any]] = []
    try:
        for name, done, command, log_path, visible_gpu in specifications:
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
                        "worker_plan": (
                            paths["first_generation"] / "launch-plan.json"
                            if name == "first200_generation"
                            else (
                                paths["full_generation"] / "launch-plan.json"
                                if name == "full867_generation"
                                else None
                            )
                        ),
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
                raise ControllerError("parallel evaluation stage timed out")
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
        samples=200,
        gpu_ids=first_gpus,
        receipt=receipt,
        source_config=paths["first_config"],
    )
    full = _generation_current(
        paths["full_generation"],
        samples=867,
        gpu_ids=full_gpus,
        receipt=receipt,
        source_config=paths["full_config"],
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
    return {"model_path": model_path, "served_name": served_name, "port": parsed.port}


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
    *, generation: Mapping[str, Any], source_config: Path, output_root: Path
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


def _run_semantic_one(
    *,
    split: str,
    generation: Mapping[str, Any],
    config: Path,
    output_root: Path,
    samples: int,
    log_path: Path,
    paths: Mapping[str, Path],
) -> None:
    if _semantic_current(
        output_root,
        samples=samples,
        generation=generation,
        source_config=config,
    ):
        return
    command = _semantic_command(
        generation=generation, source_config=config, output_root=output_root
    )
    _append_event(paths["events"], "semantic_started", split=split, command=command)
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
    if return_code != 0 or not _semantic_current(
        output_root,
        samples=samples,
        generation=generation,
        source_config=config,
    ):
        raise ControllerError(f"{split} semantic rescore failed; inspect {log_path}")
    _append_event(paths["events"], "semantic_complete", split=split)


def _run_semantic_pair(
    *,
    first: Mapping[str, Any],
    full: Mapping[str, Any],
    paths: Mapping[str, Path],
    judge_gpus: tuple[int, int],
    gpu_wait_timeout: float,
) -> None:
    if _semantic_current(
        paths["first_semantic"],
        samples=200,
        generation=first,
        source_config=paths["first_config"],
    ) and _semantic_current(
        paths["full_semantic"],
        samples=867,
        generation=full,
        source_config=paths["full_config"],
    ):
        return
    settings = _judge_settings()
    if _endpoint_open(int(settings["port"])):
        raise ControllerError("semantic judge port is occupied by an unowned endpoint")
    _wait_gpus_empty(judge_gpus, timeout_seconds=gpu_wait_timeout, paths=paths)
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
            "schema_version": "rp69-owned-semantic-judge-v1",
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
        _wait_judge_ready(judge, settings=settings, timeout=600)
        ownership["status"] = "ready"
        ownership["ready_at"] = _utc_now()
        _atomic_json(paths["judge_ownership"], ownership)
        _run_semantic_one(
            split="first200",
            generation=first,
            config=paths["first_config"],
            output_root=paths["first_semantic"],
            samples=200,
            log_path=paths["first_semantic_log"],
            paths=paths,
        )
        _run_semantic_one(
            split="full867",
            generation=full,
            config=paths["full_config"],
            output_root=paths["full_semantic"],
            samples=867,
            log_path=paths["full_semantic_log"],
            paths=paths,
        )
    finally:
        if judge is not None:
            _stop_process_group(judge)
        log.close()
        if ownership is not None and judge is not None:
            ownership["status"] = "stopped"
            ownership["stopped_at"] = _utc_now()
            ownership["returncode"] = judge.returncode
            _atomic_json(paths["judge_ownership"], ownership)
    _wait_gpus_empty(judge_gpus, timeout_seconds=300, paths=paths)


def _complete_marker_current(
    path: Path, *, receipt: Mapping[str, Any], paths: Mapping[str, Path]
) -> bool:
    if not path.exists():
        return False
    _regular_file(path, label="RP69 completion marker")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControllerError("RP69 completion marker is malformed") from error
    generated = value.get("generated_configs")
    first = value.get("first200")
    full = value.get("full867")
    if not all(isinstance(item, dict) for item in (generated, first, full)):
        raise ControllerError("RP69 completion marker sections are malformed")
    if (
        value.get("schema_version") != COMPLETE_SCHEMA
        or value.get("status") != "complete"
        or value.get("training_run_id") != EXPECTED_RUN_ID
        or value.get("training_run_identity_sha256") != receipt["run_identity_sha256"]
        or value.get("adapter_manifest_sha256") != receipt["adapter_manifest_sha256"]
        or tuple(value.get("arms", ())) != ARMS
        or not _artifact_record_current(value.get("receipt"), paths["receipt"])
        or not _artifact_record_current(generated.get("int_diag"), paths["int_config"])
        or not _artifact_record_current(
            generated.get("first200"), paths["first_config"]
        )
        or not _artifact_record_current(generated.get("full867"), paths["full_config"])
        or not _artifact_record_current(value.get("int_diag"), paths["int_report"])
        or not _artifact_record_current(
            first.get("generation"), paths["first_generation"] / "launch-summary.json"
        )
        or not _artifact_record_current(
            first.get("semantic_summary"), paths["first_semantic"] / "summary.json"
        )
        or not _artifact_record_current(
            first.get("semantic_manifest"), paths["first_semantic"] / "manifest.json"
        )
        or not _artifact_record_current(
            full.get("generation"), paths["full_generation"] / "launch-summary.json"
        )
        or not _artifact_record_current(
            full.get("semantic_summary"), paths["full_semantic"] / "summary.json"
        )
        or not _artifact_record_current(
            full.get("semantic_manifest"), paths["full_semantic"] / "manifest.json"
        )
    ):
        raise ControllerError("RP69 completion marker drifted")
    return True


def _write_complete_marker(
    *, receipt: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    marker = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "complete",
        "completed_at": _utc_now(),
        "runtime_commit": CLEAN_RUNTIME_COMMIT,
        "controller_file_sha256": _file_sha256(Path(__file__).resolve()),
        "clean_launcher_sha256": _file_sha256(CLEAN_LAUNCHER),
        "clean_semantic_tool_sha256": _file_sha256(CLEAN_SEMANTIC_TOOL),
        "training_run_id": EXPECTED_RUN_ID,
        "training_run_identity_sha256": receipt["run_identity_sha256"],
        "adapter_manifest_sha256": receipt["adapter_manifest_sha256"],
        "arms": list(ARMS),
        "receipt": _artifact_record(paths["receipt"]),
        "generated_configs": {
            "int_diag": _artifact_record(paths["int_config"]),
            "first200": _artifact_record(paths["first_config"]),
            "full867": _artifact_record(paths["full_config"]),
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
            raise ControllerError(
                "another RP69 evaluation controller owns the lock"
            ) from error
        yield


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
                    "updated_at": _utc_now(),
                },
            )
        if time.monotonic() >= deadline:
            raise ControllerError("timed out waiting for RP69 training completion")
        time.sleep(poll_seconds)


def _status(
    *,
    training_config_path: Path,
    expected_config_sha256: str,
    paths: Mapping[str, Path],
    first_gpus: tuple[int, ...],
    full_gpus: tuple[int, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": PIPELINE_SCHEMA,
        "status": "waiting",
        "training_config_path": str(training_config_path),
        "training_config_sha256": expected_config_sha256,
        "output_root": str(paths["root"]),
        "training_complete": False,
        "int_diag_complete": False,
        "first200_generation_complete": False,
        "full867_generation_complete": False,
        "first200_semantic_complete": False,
        "full867_semantic_complete": False,
        "pipeline_complete": False,
    }
    try:
        receipt = _load_training_completion(
            training_config_path,
            expected_config_sha256=expected_config_sha256,
            int_report_path=paths["int_report"],
        )
        result["training_complete"] = True
        result["training_run_identity_sha256"] = receipt["run_identity_sha256"]
        result["int_diag_complete"] = _int_report_current(receipt, paths)
        first_generation = _generation_current(
            paths["first_generation"],
            samples=200,
            gpu_ids=first_gpus,
            receipt=receipt,
            source_config=paths["first_config"],
        )
        full_generation = _generation_current(
            paths["full_generation"],
            samples=867,
            gpu_ids=full_gpus,
            receipt=receipt,
            source_config=paths["full_config"],
        )
        result["first200_generation_complete"] = first_generation is not None
        result["full867_generation_complete"] = full_generation is not None
        result["first200_semantic_complete"] = _semantic_current(
            paths["first_semantic"],
            samples=200,
            generation=first_generation,
            source_config=paths["first_config"],
        )
        result["full867_semantic_complete"] = _semantic_current(
            paths["full_semantic"],
            samples=867,
            generation=full_generation,
            source_config=paths["full_config"],
        )
        result["pipeline_complete"] = _complete_marker_current(
            paths["complete"], receipt=receipt, paths=paths
        )
        result["status"] = "complete" if result["pipeline_complete"] else "ready"
    except TrainingNotComplete as error:
        result["waiting_reason"] = str(error)
    except (ControllerError, FileNotFoundError, ValueError) as error:
        result["status"] = "blocked"
        result["blocked_reason"] = str(error)
    return result


def _parse_gpu_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU IDs must be comma-separated integers"
        ) from error
    if (
        not values
        or len(set(values)) != len(values)
        or any(item < 0 for item in values)
    ):
        raise argparse.ArgumentTypeError("GPU IDs must be unique non-negative integers")
    return values


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
    parser.add_argument("--full-gpus", type=_parse_gpu_list, default=(2, 3))
    parser.add_argument("--judge-gpus", type=_parse_gpu_list, default=(2, 3))
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
    if set(args.first_gpus).intersection(args.full_gpus):
        raise ControllerError("first200 and full867 generation GPUs must be disjoint")
    if args.int_gpu in {*args.first_gpus, *args.full_gpus}:
        raise ControllerError("INT-DIAG GPU must be disjoint from generation GPUs")


def _dry_run(
    *,
    training_config_path: Path,
    output_root: Path,
    int_gpu: int,
    first_gpus: tuple[int, ...],
    full_gpus: tuple[int, ...],
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
        "adapter_variant": EXPECTED_VARIANT,
        "arms": list(ARMS),
        "output_root": str(output_root),
        "output_isolation_preflight": {
            "status": "passed",
            "protected_paths": [
                str(path) for path in _training_protected_paths(layout)
            ],
        },
        "gpu_plan": {
            "int_diag_first200": [int_gpu],
            "acc_first200": list(first_gpus),
            "acc_full867": list(full_gpus),
            "semantic_judge_after_generation": list(judge_gpus),
        },
        "commands": {
            "int_diag": _int_command(paths["int_config"]),
            "acc_first200": _generation_command(
                config=paths["first_config"],
                root=paths["first_generation"],
                gpu_ids=first_gpus,
            ),
            "acc_full867": _generation_command(
                config=paths["full_config"],
                root=paths["full_generation"],
                gpu_ids=full_gpus,
            ),
        },
        "config_preview_sha256": {
            "int_diag": sha256(
                _render_evaluation_config(
                    receipt=placeholder,
                    run_id=f"{EXPECTED_RUN_ID}-STEP500-INT-FIRST200-GPU{int_gpu}",
                    evaluation_id="rp69-step0500-first200-int-diag-v1",
                    manifest_path=FIRST_MANIFEST,
                    manifest_sha256=FIRST_MANIFEST_SHA256,
                    report_path=paths["int_report"],
                    physical_gpu_id=int_gpu,
                )
            ).hexdigest(),
            "note": "preview contains explicit post-training hash placeholders",
        },
    }


def _run(
    args: argparse.Namespace, training_config_path: Path, paths: Mapping[str, Path]
) -> dict[str, Any]:
    _assert_output_root_isolated(
        paths["root"], _static_training_layout(training_config_path)
    )
    expected_config_sha256 = _file_sha256(training_config_path)
    paths["root"].mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(paths["lock"]):
        _append_event(
            paths["events"],
            "controller_started",
            pid=os.getpid(),
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
                full_gpus=args.full_gpus,
            )
            if _complete_marker_current(
                paths["complete"], receipt=receipt, paths=paths
            ):
                return _status(
                    training_config_path=training_config_path,
                    expected_config_sha256=expected_config_sha256,
                    paths=paths,
                    first_gpus=args.first_gpus,
                    full_gpus=args.full_gpus,
                )
            pending_gpus: list[int] = []
            if not _int_report_current(receipt, paths):
                pending_gpus.append(args.int_gpu)
            if (
                _generation_current(
                    paths["first_generation"],
                    samples=200,
                    gpu_ids=args.first_gpus,
                    receipt=receipt,
                    source_config=paths["first_config"],
                )
                is None
            ):
                pending_gpus.extend(args.first_gpus)
            if (
                _generation_current(
                    paths["full_generation"],
                    samples=867,
                    gpu_ids=args.full_gpus,
                    receipt=receipt,
                    source_config=paths["full_config"],
                )
                is None
            ):
                pending_gpus.extend(args.full_gpus)
            if pending_gpus:
                _wait_gpus_empty(
                    pending_gpus,
                    timeout_seconds=args.gpu_wait_timeout_seconds,
                    paths=paths,
                )
            first, full = _run_parallel_generation(
                receipt=receipt,
                paths=paths,
                int_gpu=args.int_gpu,
                first_gpus=args.first_gpus,
                full_gpus=args.full_gpus,
                timeout_seconds=args.evaluation_timeout_seconds,
            )
            _run_semantic_pair(
                first=first,
                full=full,
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
        paths=paths,
        first_gpus=args.first_gpus,
        full_gpus=args.full_gpus,
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
    assert_legacy_standalone_execution_quarantined(
        "tools/run_rp69_post_training_evaluations.py"
    )
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
            full_gpus=args.full_gpus,
            judge_gpus=args.judge_gpus,
        )
    elif args.command == "status":
        result = _status(
            training_config_path=training_config_path,
            expected_config_sha256=_file_sha256(training_config_path),
            paths=paths,
            first_gpus=args.first_gpus,
            full_gpus=args.full_gpus,
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
    raise SystemExit(main())
