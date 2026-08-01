#!/usr/bin/env python3
"""Run RP68 INT-DIAG and ACC-VAL as two restartable, fail-closed stages.

The tool is intentionally specific to the routing-only RP68 experiment.  It
does not wait for training and it never infers success from an exited process:
each command first validates the terminal training ledger and the exported
Adapter, then publishes a hash-bound completion marker only after its own
outputs are complete.

``int-diag`` runs the established first-200 internal diagnostic.  ``acc-val``
runs the two decision arms (original image and image + correct D) on first-200
and full-867, followed by the pinned semantic rescore.  The two markers are
separate so they can be used directly as the ``int_diag`` and ``acc_val``
acceptance artifacts in ``run_overnight_pipeline.py``.
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
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence
from urllib.request import urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPOSITORY_ROOT / ".venv312/bin/python"
DEFAULT_TRAINING_CONFIG = REPOSITORY_ROOT / (
    "configs/representation/"
    "qwen3_instruct_balanced_t1_vision_routing_2000step_gpu0123_resume_step10.toml"
)
EXPECTED_RUN_ID = "RP-68-QWEN3-INSTRUCT-REP-BALANCED-T1-VISION-ROUTING-2000-GPU0123"
EXPECTED_MODEL_NAME = "Qwen3-VL-8B-Instruct"
EXPECTED_ADAPTER_VARIANT = "full_d_deepstack_vision_routing"
EXPECTED_GLOBAL_STEP = 2000
EXPECTED_WORLD_SIZE = 4
EXPECTED_PHYSICAL_GPU_IDS = (0, 1, 2, 3)
EXPECTED_GRADIENT_ACCUMULATION_STEPS = 2
EXPECTED_TENSOR_COUNT = 104
EVALUATION_CODE_IDENTITY_PATHS = (
    "src/tgvf_rl",
    "pyproject.toml",
    "requirements/compatibility.lock",
    "requirements/compatibility-torch211-cu129.lock",
    "uv.lock",
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
PYTHON_HEADER_ROOT = REPOSITORY_ROOT / ".deps/python312-dev/root/usr/include"

PIPELINE_SCHEMA = "rp68-post-training-evaluations-v1"
INT_MARKER_SCHEMA = "rp68-int-diag-complete-v1"
ACC_MARKER_SCHEMA = "rp68-acc-val-complete-v1"
SEMANTIC_SCHEMA = "answer-utility-semantic-rescore-v2"
MAIN_ARMS = ("image_only", "image_correct_D")


class EvaluationBlockedError(RuntimeError):
    """Raised when an identity or durable-output check refuses continuation."""


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
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )


def _write_identical_or_new(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise EvaluationBlockedError(f"refusing to replace nonidentical {path}")
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
    value = {"event": event, "at": _utc_now(), **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        os.fsync(handle.fileno())


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise EvaluationBlockedError(f"{label} is not one regular file: {path}")


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvaluationBlockedError(
            "could not resolve the live evaluation code identity"
        ) from error
    return completed.stdout.strip()


def _live_evaluation_code_commit() -> str:
    """Bind generated configs to the latest commit touching evaluator code.

    The training commit remains part of the checkpoint receipt.  Evaluation is
    a later, separately executed program, so its config must instead name the
    code actually eligible to run now.  Using the latest commit that touched
    the evaluator's identity paths keeps this value stable across ledger-only
    commits between INT-DIAG and ACC-VAL.
    """

    local_patch = _git_output(
        "diff", "--name-only", "HEAD", "--", *EVALUATION_CODE_IDENTITY_PATHS
    )
    untracked = _git_output(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *EVALUATION_CODE_IDENTITY_PATHS,
    )
    if local_patch or untracked:
        raise EvaluationBlockedError(
            "RP68 post-training evaluation requires clean live code paths"
        )
    commit = _git_output(
        "log", "-1", "--format=%H", "HEAD", "--", *EVALUATION_CODE_IDENTITY_PATHS
    )
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise EvaluationBlockedError("live evaluation code commit is invalid")
    return commit


def _assert_pinned_inputs() -> None:
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
            raise EvaluationBlockedError(f"pinned {label} SHA256 differs")


def _load_training_completion(training_config_path: Path) -> dict[str, Any]:
    """Return one independently verified RP68 terminal artifact receipt."""

    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
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

    training_config_path = training_config_path.expanduser().resolve()
    _regular_file(training_config_path, label="RP68 training config")
    training = load_representation_training_config(training_config_path)
    mismatches: list[str] = []
    if training.run_id != EXPECTED_RUN_ID:
        mismatches.append("run_id")
    if training.model.model_name != EXPECTED_MODEL_NAME:
        mismatches.append("model")
    if training.adapter_variant.value != EXPECTED_ADAPTER_VARIANT:
        mismatches.append("adapter_variant")
    if training.training.target_optimizer_steps != EXPECTED_GLOBAL_STEP:
        mismatches.append("target_optimizer_steps")
    if training.fsdp2.world_size != EXPECTED_WORLD_SIZE:
        mismatches.append("world_size")
    if tuple(training.fsdp2.physical_gpu_ids) != EXPECTED_PHYSICAL_GPU_IDS:
        mismatches.append("physical_gpu_ids")
    if (
        training.training.gradient_accumulation_steps
        != EXPECTED_GRADIENT_ACCUMULATION_STEPS
    ):
        mismatches.append("gradient_accumulation_steps")
    internal = training.post_training_internal_evaluation
    if internal is None or internal.enabled:
        mismatches.append("post_training_internal_evaluation.enabled")
    if mismatches:
        raise EvaluationBlockedError(
            "training config is not the accepted RP68 world4/GA2 identity: "
            + ", ".join(mismatches)
        )

    metrics_path = training.output.metrics_jsonl_path
    _regular_file(metrics_path, label="RP68 metrics ledger")
    records: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(metrics_path.read_bytes().splitlines(), 1):
        if not raw.strip():
            raise EvaluationBlockedError(
                f"RP68 metrics contains a blank line at {line_number}"
            )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise EvaluationBlockedError(
                f"RP68 metrics JSON is malformed at line {line_number}"
            ) from error
        if not isinstance(value, dict):
            raise EvaluationBlockedError(
                f"RP68 metrics line {line_number} is not an object"
            )
        records.append(value)
    complete = [record for record in records if record.get("event") == "complete"]
    if len(complete) != 1 or records[-1] is not complete[0]:
        raise EvaluationBlockedError(
            "RP68 metrics requires exactly one terminal complete event"
        )
    terminal = complete[0]
    expected_terminal = {
        "status": "complete",
        "run_id": EXPECTED_RUN_ID,
        "global_step": EXPECTED_GLOBAL_STEP,
        "source_toml_sha256": training.source_toml_sha256,
        "canonical_config_sha256": training.canonical_config_sha256,
        "final_artifact_path": str(training.output.final_artifact_path),
        "metrics_jsonl_path": str(metrics_path),
        "world_size": EXPECTED_WORLD_SIZE,
        "physical_gpu_ids": list(EXPECTED_PHYSICAL_GPU_IDS),
    }
    bad_terminal = sorted(
        key
        for key, expected in expected_terminal.items()
        if terminal.get(key) != expected
    )
    if bad_terminal:
        raise EvaluationBlockedError(
            "RP68 terminal metrics identity differs: " + ", ".join(bad_terminal)
        )
    if terminal.get("post_training_internal_evaluation") != {"status": "disabled"}:
        raise EvaluationBlockedError(
            "RP68 terminal metrics unexpectedly embeds INT-DIAG"
        )

    adapter_path = training.output.final_artifact_path
    _regular_file(adapter_path, label="RP68 Adapter export")
    export = load_rank_zero_adapter_owned_state_export(adapter_path)
    manifest = export.manifest
    manifest_sha256 = state_digest(manifest)
    if manifest.global_step != EXPECTED_GLOBAL_STEP:
        raise EvaluationBlockedError("RP68 Adapter export is not step 2000")
    if manifest.run_identity.run_id != EXPECTED_RUN_ID:
        raise EvaluationBlockedError("RP68 Adapter export names another run")
    if manifest.run_identity.adapter_contract.variant != EXPECTED_ADAPTER_VARIANT:
        raise EvaluationBlockedError("RP68 Adapter export names another structure")
    if len(manifest.tensor_names) != EXPECTED_TENSOR_COUNT:
        raise EvaluationBlockedError("RP68 Adapter export tensor count differs")
    if terminal.get("run_identity_sha256") != manifest.run_identity_sha256:
        raise EvaluationBlockedError("metrics/Adapter run identity differs")
    if terminal.get("final_artifact_manifest_sha256") != manifest_sha256:
        raise EvaluationBlockedError("metrics/Adapter manifest identity differs")
    _validate_training_artifact_binding(training, manifest.run_identity)

    return {
        "schema_version": "rp68-training-completion-receipt-v1",
        "status": "complete",
        "run_id": EXPECTED_RUN_ID,
        "run_identity_sha256": manifest.run_identity_sha256,
        "global_step": EXPECTED_GLOBAL_STEP,
        "training_config_path": str(training_config_path),
        "training_config_sha256": training.source_toml_sha256,
        "code_commit": training.code.commit,
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
    }


def _toml_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _render_evaluation_config(
    *,
    receipt: Mapping[str, Any],
    evaluation_code_commit: str,
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
commit = {_toml_string(evaluation_code_commit)}
dirty = false

[source]
training_config_path = {_toml_string(receipt["training_config_path"])}
training_config_sha256 = {_toml_string(receipt["training_config_sha256"])}

[artifact]
path = {_toml_string(receipt["adapter_path"])}
file_sha256 = {_toml_string(receipt["adapter_file_sha256"])}
manifest_sha256 = {_toml_string(receipt["adapter_manifest_sha256"])}
expected_run_identity_sha256 = {_toml_string(receipt["run_identity_sha256"])}
expected_global_step = {EXPECTED_GLOBAL_STEP}

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


def _paths(training_config_path: Path) -> dict[str, Path]:
    # The output root is fixed by the accepted RP68 run identity, while the
    # exact training config path remains explicit in every generated config.
    training_root = REPOSITORY_ROOT / (
        "artifacts/representation/"
        "RP-68-qwen3-instruct-balanced-t1-vision-routing-2000-gpu0123"
    )
    root = training_root / "post-training-evaluation"
    configs = root / "generated-configs"
    return {
        "root": root,
        "events": root / "events.jsonl",
        "lock": root / "pipeline.lock",
        "receipt": root / "training-completion-receipt.json",
        "int_config": configs / "rp68-step2000-int-diag.toml",
        "first_config": configs / "rp68-step2000-acc-first200.toml",
        "full_config": configs / "rp68-step2000-acc-full867.toml",
        "int_report": root / "INT-DIAG-first200-step2000.json",
        "int_marker": root / "INT-DIAG-complete.json",
        "first_generation": root / "ACC-VAL-first200-generation",
        "full_generation": root / "ACC-VAL-full867-generation",
        "first_semantic": root / "ACC-VAL-first200-semantic",
        "full_semantic": root / "ACC-VAL-full867-semantic",
        "acc_marker": root / "ACC-VAL-complete.json",
        "int_log": root / "INT-DIAG.log",
        "first_generation_log": root / "ACC-VAL-first200-generation.log",
        "full_generation_log": root / "ACC-VAL-full867-generation.log",
        "judge_log": root / "ACC-VAL-judge-server.log",
        "first_semantic_log": root / "ACC-VAL-first200-semantic.log",
        "full_semantic_log": root / "ACC-VAL-full867-semantic.log",
    }


def _materialize_configs(
    training_config_path: Path,
    *,
    physical_gpu_ids: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Path]]:
    _assert_pinned_inputs()
    receipt = _load_training_completion(training_config_path)
    evaluation_code_commit = _live_evaluation_code_commit()
    paths = _paths(training_config_path)
    paths["root"].mkdir(parents=True, exist_ok=True)
    _write_identical_or_new(
        paths["receipt"],
        (
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )
    _write_identical_or_new(
        paths["int_config"],
        _render_evaluation_config(
            receipt=receipt,
            evaluation_code_commit=evaluation_code_commit,
            run_id="RP-68-STEP2000-INT-DIAG-FIRST200-GPU0",
            evaluation_id="rp68-step2000-int-diag-v1",
            manifest_path=FIRST_MANIFEST,
            manifest_sha256=FIRST_MANIFEST_SHA256,
            report_path=paths["int_report"],
            physical_gpu_id=physical_gpu_ids[0],
        ),
    )
    _write_identical_or_new(
        paths["first_config"],
        _render_evaluation_config(
            receipt=receipt,
            evaluation_code_commit=evaluation_code_commit,
            run_id="RP-68-STEP2000-ACC-VAL-FIRST200-GPU01",
            evaluation_id="rp68-step2000-acc-val-first200-v1",
            manifest_path=FIRST_MANIFEST,
            manifest_sha256=FIRST_MANIFEST_SHA256,
            report_path=paths["root"] / "ACC-VAL-first200-internal-unused.json",
            physical_gpu_id=physical_gpu_ids[0],
        ),
    )
    _write_identical_or_new(
        paths["full_config"],
        _render_evaluation_config(
            receipt=receipt,
            evaluation_code_commit=evaluation_code_commit,
            run_id="RP-68-STEP2000-ACC-VAL-FULL867-GPU01",
            evaluation_id="rp68-step2000-acc-val-full867-v1",
            manifest_path=FULL_MANIFEST,
            manifest_sha256=FULL_MANIFEST_SHA256,
            report_path=paths["root"] / "ACC-VAL-full867-internal-unused.json",
            physical_gpu_id=physical_gpu_ids[0],
        ),
    )
    return receipt, paths


def _artifact_record(path: Path) -> dict[str, object]:
    _regular_file(path, label="completion artifact")
    return {
        "status": "complete",
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _artifact_record_is_current(record: object, path: Path) -> bool:
    if not isinstance(record, dict):
        return False
    return (
        record.get("status") == "complete"
        and record.get("path") == str(path.resolve())
        and path.is_file()
        and not path.is_symlink()
        and record.get("sha256") == _file_sha256(path)
        and record.get("bytes") == path.stat().st_size
    )


def _int_report_is_current(path: Path, receipt: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    _regular_file(path, label="INT-DIAG report")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationBlockedError("INT-DIAG report JSON is malformed") from error
    identity = value.get("identity") if isinstance(value, dict) else None
    expected = {
        "schema_version": "representation_internal_evaluation_v1",
        "evaluation_id": "rp68-step2000-int-diag-v1",
        "checkpoint_identity": receipt["adapter_manifest_sha256"],
        "target_conditioning_provider": receipt["conditioning_provider"],
        "random_seed": 42,
        "prompt_identity": (f"{receipt['prompt_identity']}:{receipt['prompt_sha256']}"),
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise EvaluationBlockedError("INT-DIAG report identity differs")
    return True


def _int_marker_is_current(
    marker_path: Path,
    *,
    receipt: Mapping[str, Any],
    report_path: Path,
    config_path: Path,
) -> bool:
    if not marker_path.exists():
        return False
    _regular_file(marker_path, label="INT-DIAG completion marker")
    value = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != INT_MARKER_SCHEMA
        or value.get("status") != "complete"
        or value.get("run_id") != EXPECTED_RUN_ID
        or value.get("run_identity_sha256") != receipt["run_identity_sha256"]
        or value.get("adapter_manifest_sha256") != receipt["adapter_manifest_sha256"]
        or value.get("evaluation_config_sha256") != _file_sha256(config_path)
        or not _artifact_record_is_current(value.get("report"), report_path)
    ):
        raise EvaluationBlockedError("INT-DIAG completion marker drifted")
    _int_report_is_current(report_path, receipt)
    return True


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise EvaluationBlockedError(
                "another RP68 post-training evaluation process is active"
            ) from error
        yield


def _run_int_diag(
    *,
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
    physical_gpu_ids: tuple[int, int],
) -> None:
    if _int_marker_is_current(
        paths["int_marker"],
        receipt=receipt,
        report_path=paths["int_report"],
        config_path=paths["int_config"],
    ):
        return
    if not _int_report_is_current(paths["int_report"], receipt):
        command = [
            str(PYTHON),
            "-m",
            "tgvf_rl.cli",
            "run-representation-internal-evaluation",
            str(paths["int_config"]),
        ]
        environment = {
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(physical_gpu_ids[0]),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
        }
        _append_event(paths["events"], "int_diag_started", command=command)
        with paths["int_log"].open("ab", buffering=0) as log:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60 * 60,
            )
        if completed.returncode != 0 or not _int_report_is_current(
            paths["int_report"], receipt
        ):
            raise EvaluationBlockedError(f"INT-DIAG failed; inspect {paths['int_log']}")
        _append_event(paths["events"], "int_diag_finished")
    marker = {
        "schema_version": INT_MARKER_SCHEMA,
        "status": "complete",
        "completed_at": _utc_now(),
        "run_id": EXPECTED_RUN_ID,
        "run_identity_sha256": receipt["run_identity_sha256"],
        "adapter_manifest_sha256": receipt["adapter_manifest_sha256"],
        "evaluation_config_sha256": _file_sha256(paths["int_config"]),
        "report": _artifact_record(paths["int_report"]),
    }
    _atomic_json(paths["int_marker"], marker)


def _load_complete_generation(root: Path, *, samples: int) -> dict[str, Any] | None:
    summary_path = root / "launch-summary.json"
    if not summary_path.exists():
        return None
    _regular_file(summary_path, label="ACC generation launch summary")
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "complete"
        or tuple(value.get("arms", ())) != MAIN_ARMS
        or value.get("sample_count") != samples
        or value.get("record_count") != samples * len(MAIN_ARMS)
    ):
        raise EvaluationBlockedError(f"ACC generation summary differs: {summary_path}")
    shards = value.get("shards")
    if not isinstance(shards, list) or len(shards) != 2:
        raise EvaluationBlockedError("ACC generation did not use exactly two shards")
    for shard in shards:
        if not isinstance(shard, dict):
            raise EvaluationBlockedError("ACC generation shard record is invalid")
        shard_root = Path(str(shard.get("output_root", "")))
        for name in ("identity.json", "records.jsonl", "summary.json"):
            _regular_file(shard_root / name, label="ACC generation shard artifact")
    return value


def _launch_generation(
    *,
    source_config: Path,
    output_root: Path,
    samples: int,
    physical_gpu_ids: tuple[int, int],
    log_path: Path,
    events: Path,
) -> dict[str, Any]:
    existing = _load_complete_generation(output_root, samples=samples)
    if existing is not None:
        return existing
    command = [
        str(PYTHON),
        str(
            REPOSITORY_ROOT / "tools/launch_representation_answer_utility_evaluation.py"
        ),
        "--production-source",
        "--source-evaluation-config",
        str(source_config),
        "--output-root",
        str(output_root),
    ]
    for gpu in physical_gpu_ids:
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
    for arm in MAIN_ARMS:
        command.extend(("--arm", arm))
    _append_event(
        events,
        "acc_generation_started",
        samples=samples,
        output_root=str(output_root),
    )
    with log_path.open("ab", buffering=0) as log:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=3 * 60 * 60,
        )
    if completed.returncode != 0:
        raise EvaluationBlockedError(f"ACC generation failed; inspect {log_path}")
    result = _load_complete_generation(output_root, samples=samples)
    if result is None:
        raise EvaluationBlockedError("ACC generation exited without completion")
    _append_event(events, "acc_generation_finished", samples=samples)
    return result


def _semantic_complete(root: Path, *, samples: int) -> bool:
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    if not summary_path.exists() and not manifest_path.exists():
        return False
    _regular_file(summary_path, label="ACC semantic summary")
    _regular_file(manifest_path, label="ACC semantic manifest")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_arm = summary.get("by_arm")
    run_identity = summary.get("run_identity_sha256")
    files = manifest.get("files")
    if (
        summary.get("schema_version") != SEMANTIC_SCHEMA
        or manifest.get("schema_version") != SEMANTIC_SCHEMA
        or summary.get("status") != "complete"
        or manifest.get("status") != "complete"
        or not isinstance(run_identity, str)
        or len(run_identity) != 64
        or manifest.get("run_identity_sha256") != run_identity
        or not isinstance(by_arm, dict)
        or set(by_arm) != set(MAIN_ARMS)
        or summary.get("overall", {}).get("total") != samples * len(MAIN_ARMS)
        or any(
            not isinstance(by_arm.get(arm), dict) or by_arm[arm].get("total") != samples
            for arm in MAIN_ARMS
        )
        or not isinstance(files, dict)
        or files.get("summary", {}).get("path") != "summary.json"
        or files.get("summary", {}).get("sha256") != _file_sha256(summary_path)
        or files.get("overlay_records", {}).get("rows") != samples * len(MAIN_ARMS)
    ):
        raise EvaluationBlockedError(f"ACC semantic publication differs: {root}")
    return True


def _semantic_record(root: Path, *, samples: int) -> dict[str, object]:
    if not _semantic_complete(root, samples=samples):
        raise EvaluationBlockedError(f"ACC semantic publication is missing: {root}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return {
        "status": "complete",
        "root": str(root.resolve()),
        "sample_count": samples,
        "arms": list(MAIN_ARMS),
        "run_identity_sha256": summary["run_identity_sha256"],
        "summary": _artifact_record(root / "summary.json"),
        "manifest": _artifact_record(root / "manifest.json"),
    }


def _semantic_record_is_current(record: object, *, root: Path, samples: int) -> bool:
    if not isinstance(record, dict) or record.get("status") != "complete":
        return False
    if (
        record.get("root") != str(root.resolve())
        or record.get("sample_count") != samples
        or tuple(record.get("arms", ())) != MAIN_ARMS
        or not _semantic_complete(root, samples=samples)
    ):
        return False
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return (
        record.get("run_identity_sha256") == summary.get("run_identity_sha256")
        and _artifact_record_is_current(record.get("summary"), root / "summary.json")
        and _artifact_record_is_current(record.get("manifest"), root / "manifest.json")
    )


def _acc_marker_is_current(
    marker_path: Path,
    *,
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> bool:
    if not marker_path.exists():
        return False
    _regular_file(marker_path, label="ACC-VAL completion marker")
    value = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != ACC_MARKER_SCHEMA
        or value.get("status") != "complete"
        or value.get("run_id") != EXPECTED_RUN_ID
        or value.get("run_identity_sha256") != receipt["run_identity_sha256"]
        or value.get("adapter_manifest_sha256") != receipt["adapter_manifest_sha256"]
        or value.get("first200_evaluation_config_sha256")
        != _file_sha256(paths["first_config"])
        or value.get("full867_evaluation_config_sha256")
        != _file_sha256(paths["full_config"])
        or not _semantic_record_is_current(
            value.get("first200"), root=paths["first_semantic"], samples=200
        )
        or not _semantic_record_is_current(
            value.get("full867"), root=paths["full_semantic"], samples=867
        )
    ):
        raise EvaluationBlockedError("ACC-VAL completion marker drifted")
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
        pid_text, uuid = (part.strip() for part in row.split(",", 1))
        index = uuid_to_index[uuid]
        if index in result:
            result[index].append(int(pid_text))
    return {index: tuple(sorted(pids)) for index, pids in result.items()}


def _wait_gpus_empty(indices: tuple[int, int], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not any(_gpu_compute_pids(indices).values()):
            return
        time.sleep(5)
    raise EvaluationBlockedError(f"GPUs {indices} did not become compute-empty")


def _judge_command() -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct",
        "--served-model-name",
        "Qwen2.5-72B-Instruct",
        "--host",
        "127.0.0.1",
        "--port",
        "8013",
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


def _python_header_cpath() -> str:
    python_headers = PYTHON_HEADER_ROOT / "python3.12"
    required = (
        python_headers / "Python.h",
        python_headers / "pyconfig.h",
        PYTHON_HEADER_ROOT / "x86_64-linux-gnu/python3.12/pyconfig.h",
    )
    missing = tuple(str(path) for path in required if not path.is_file())
    if missing:
        raise EvaluationBlockedError(f"Python development headers missing: {missing}")
    return os.pathsep.join((str(PYTHON_HEADER_ROOT), str(python_headers)))


def _judge_endpoint_is_open() -> bool:
    try:
        with urlopen("http://127.0.0.1:8013/v1/models", timeout=1) as response:
            response.read(1)
        return True
    except Exception:
        return False


def _wait_judge_ready(process: subprocess.Popen[bytes], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise EvaluationBlockedError("semantic judge exited during startup")
        try:
            with urlopen("http://127.0.0.1:8013/v1/models", timeout=2) as response:
                value = json.loads(response.read())
            if {item.get("id") for item in value.get("data", [])} == {
                "Qwen2.5-72B-Instruct"
            }:
                return
        except Exception:
            pass
        time.sleep(2)
    raise EvaluationBlockedError("semantic judge readiness timed out")


def _stop_owned_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _run_semantic(
    *,
    generation: Mapping[str, Any],
    source_config: Path,
    output_root: Path,
    samples: int,
    log_path: Path,
    events: Path,
) -> None:
    if _semantic_complete(output_root, samples=samples):
        return
    command = [
        str(PYTHON),
        str(
            REPOSITORY_ROOT
            / "tools/run_representation_answer_utility_semantic_rescore.py"
        ),
    ]
    for shard in generation["shards"]:
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
    _append_event(events, "acc_semantic_started", samples=samples)
    with log_path.open("ab", buffering=0) as log:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=2 * 60 * 60,
        )
    if completed.returncode != 0 or not _semantic_complete(
        output_root, samples=samples
    ):
        raise EvaluationBlockedError(f"semantic rescore failed; inspect {log_path}")
    _append_event(events, "acc_semantic_finished", samples=samples)


def _run_acc_val(
    *,
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
    physical_gpu_ids: tuple[int, int],
) -> None:
    if _acc_marker_is_current(paths["acc_marker"], receipt=receipt, paths=paths):
        return
    first = _launch_generation(
        source_config=paths["first_config"],
        output_root=paths["first_generation"],
        samples=200,
        physical_gpu_ids=physical_gpu_ids,
        log_path=paths["first_generation_log"],
        events=paths["events"],
    )
    full = _launch_generation(
        source_config=paths["full_config"],
        output_root=paths["full_generation"],
        samples=867,
        physical_gpu_ids=physical_gpu_ids,
        log_path=paths["full_generation_log"],
        events=paths["events"],
    )
    _wait_gpus_empty(physical_gpu_ids, 300)
    jobs = (
        (
            first,
            paths["first_config"],
            paths["first_semantic"],
            200,
            paths["first_semantic_log"],
        ),
        (
            full,
            paths["full_config"],
            paths["full_semantic"],
            867,
            paths["full_semantic_log"],
        ),
    )
    pending = any(
        not _semantic_complete(root, samples=samples) for _, _, root, samples, _ in jobs
    )
    if pending and _judge_endpoint_is_open():
        raise EvaluationBlockedError("port 8013 is occupied by an unowned endpoint")
    judge: subprocess.Popen[bytes] | None = None
    judge_log = paths["judge_log"].open("ab", buffering=0)
    try:
        if pending:
            environment = {
                **os.environ,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in physical_gpu_ids),
                "VLLM_USE_V1": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONHASHSEED": "42",
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "CC": "/usr/bin/gcc",
                "CXX": "/usr/bin/g++",
                "CPATH": _python_header_cpath(),
                "LIBRARY_PATH": str(REPOSITORY_ROOT / ".venv312/lib"),
                "TRITON_CACHE_DIR": str(paths["root"] / "cache/judge-triton"),
                "TORCHINDUCTOR_CACHE_DIR": str(
                    paths["root"] / "cache/judge-torchinductor"
                ),
            }
            Path(environment["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            Path(environment["TORCHINDUCTOR_CACHE_DIR"]).mkdir(
                parents=True, exist_ok=True
            )
            judge = subprocess.Popen(
                _judge_command(),
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=judge_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _append_event(paths["events"], "acc_judge_started", pgid=judge.pid)
            _wait_judge_ready(judge, 300)
            for generation, config, root, samples, log_path in jobs:
                _run_semantic(
                    generation=generation,
                    source_config=config,
                    output_root=root,
                    samples=samples,
                    log_path=log_path,
                    events=paths["events"],
                )
    finally:
        if judge is not None:
            _stop_owned_group(judge)
            _append_event(paths["events"], "acc_judge_stopped", pgid=judge.pid)
        judge_log.close()
    _wait_gpus_empty(physical_gpu_ids, 300)
    marker = {
        "schema_version": ACC_MARKER_SCHEMA,
        "status": "complete",
        "completed_at": _utc_now(),
        "run_id": EXPECTED_RUN_ID,
        "run_identity_sha256": receipt["run_identity_sha256"],
        "adapter_manifest_sha256": receipt["adapter_manifest_sha256"],
        "arms": list(MAIN_ARMS),
        "first200_evaluation_config_sha256": _file_sha256(paths["first_config"]),
        "full867_evaluation_config_sha256": _file_sha256(paths["full_config"]),
        "first200": _semantic_record(paths["first_semantic"], samples=200),
        "full867": _semantic_record(paths["full_semantic"], samples=867),
    }
    _atomic_json(paths["acc_marker"], marker)


def _parse_gpu_ids(value: str) -> tuple[int, int]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU IDs must be two comma-separated integers"
        ) from error
    if len(values) != 2 or len(set(values)) != 2 or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "GPU IDs must be two distinct non-negative integers"
        )
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--gpu-ids",
        type=_parse_gpu_ids,
        default=(0, 1),
        help="Two physical GPUs used by ACC and its TP=2 judge (default: 0,1).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    for name in ("int-diag", "acc-val"):
        command = subparsers.add_parser(name)
        command.add_argument("--execute", action="store_true", required=True)
    return parser


def _status(
    *,
    training_config_path: Path,
    physical_gpu_ids: tuple[int, int],
) -> dict[str, object]:
    paths = _paths(training_config_path)
    result: dict[str, object] = {
        "schema_version": PIPELINE_SCHEMA,
        "run_id": EXPECTED_RUN_ID,
        "training_complete": False,
        "int_diag_complete": False,
        "acc_val_complete": False,
        "int_diag_marker": str(paths["int_marker"]),
        "acc_val_marker": str(paths["acc_marker"]),
    }
    try:
        receipt, paths = _materialize_configs(
            training_config_path, physical_gpu_ids=physical_gpu_ids
        )
        result["training_complete"] = True
        result["run_identity_sha256"] = receipt["run_identity_sha256"]
        result["int_diag_complete"] = _int_marker_is_current(
            paths["int_marker"],
            receipt=receipt,
            report_path=paths["int_report"],
            config_path=paths["int_config"],
        )
        result["acc_val_complete"] = _acc_marker_is_current(
            paths["acc_marker"], receipt=receipt, paths=paths
        )
    except (EvaluationBlockedError, FileNotFoundError, ValueError) as error:
        result["blocked_reason"] = str(error)
    return result


def main() -> int:
    args = _parser().parse_args()
    training_config_path = args.training_config.expanduser().resolve()
    physical_gpu_ids = args.gpu_ids
    if args.command == "status":
        print(
            json.dumps(
                _status(
                    training_config_path=training_config_path,
                    physical_gpu_ids=physical_gpu_ids,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    receipt, paths = _materialize_configs(
        training_config_path, physical_gpu_ids=physical_gpu_ids
    )
    with _exclusive_lock(paths["lock"]):
        _append_event(paths["events"], "command_started", command=args.command)
        if args.command == "int-diag":
            _run_int_diag(
                receipt=receipt,
                paths=paths,
                physical_gpu_ids=physical_gpu_ids,
            )
            marker = paths["int_marker"]
        else:
            _run_acc_val(
                receipt=receipt,
                paths=paths,
                physical_gpu_ids=physical_gpu_ids,
            )
            marker = paths["acc_marker"]
        _append_event(paths["events"], "command_finished", command=args.command)
    print(
        json.dumps(
            {
                "schema_version": PIPELINE_SCHEMA,
                "status": "complete",
                "command": args.command,
                "marker": str(marker),
                "marker_sha256": _file_sha256(marker),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationBlockedError as error:
        print(f"RP68_EVALUATION_BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(3) from error
