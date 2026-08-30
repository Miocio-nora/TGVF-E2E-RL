#!/usr/bin/python3 -I
"""Wait for RP67 completion, run every accepted ACC check, and release GPU0/1.

This one-off controller is intentionally strict and restartable.  It waits for
the exact live RP67 launcher to disappear, verifies the durable training
completion with the image-axis handoff gate, materializes step-2000 evaluation
configs, runs first-200/full-867 main ACC plus the first-200 six-arm diagnostic,
applies the pinned semantic judge, and only then publishes the marker consumed
by ``supervise_rp67_t1_schedule.py``.
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
            "tools/run_rp67_step2000_acc_pipeline.py",
        ),
    )

import argparse
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
from typing import Any, Iterable, Mapping, Sequence
from urllib.request import urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)

PYTHON = REPOSITORY_ROOT / ".venv312/bin/python"
RUN_ID = "RP-67-QWEN3-INSTRUCT-REP-BALANCED-T1-IMAGE-AXIS-GROUNDED-2000-GPU01"
RUN_IDENTITY_SHA256 = "0b53d04cf8e4c8b665e76279da1df8d1e6ebabee63318c644a3bff5bad099b44"
GLOBAL_STEP = 2000
EXPERIMENT_CONFIG = REPOSITORY_ROOT / (
    "configs/representation/experiments/image_axis_grounding/"
    "rp67_image_axis_grounding_2000_finalize_step2000_v1.toml"
)
EXPERIMENT_CONFIG_SHA256 = (
    "c7242f0b259ac0ceb3416ce0e2d33deef1736b32e6f9988a55ba217386a84672"
)
TRAINING_CONFIG = REPOSITORY_ROOT / (
    "configs/representation/experiments/image_axis_grounding/"
    "rp67_qwen3_instruct_image_axis_grounded_2000_gpu01_finalize_step2000.toml"
)
TRAINING_CONFIG_SHA256 = (
    "cbbfc146b97263f5529603c462a4b46400deea283f307a89d8db7cae264aa5ee"
)
TRAINING_ROOT = REPOSITORY_ROOT / (
    "artifacts/representation/"
    "RP-67-qwen3-instruct-balanced-t1-image-axis-grounded-2000-gpu01"
)
OUTER_LOG = TRAINING_ROOT / "finalize-step2000.log"
METRICS = TRAINING_ROOT / "metrics.jsonl"
ADAPTER = TRAINING_ROOT / "adapter.pt"
INT_DIAG = TRAINING_ROOT / "int-diag-step2000.json"
TRAINING_PID = 2907805
TRAINING_STARTTIME_TICKS = 109938523
TRAINING_BOOT_ID = "6b9dc539-015e-4968-9ae2-29419cef00f2"
EVALUATION_CONFIG_ROOT = REPOSITORY_ROOT / (
    "configs/representation/experiments/image_axis_grounding/evaluation"
)
EVALUATION_ROOT = REPOSITORY_ROOT / (
    "artifacts/representation_experiments/image_axis_grounding/evaluation"
)
PIPELINE_ROOT = EVALUATION_ROOT / "rp67_step2000_acc_pipeline_20260801"
COMPLETE_MARKER = EVALUATION_ROOT / "rp67_step2000_all_validations_complete_v2.json"
PIPELINE_COMPLETE = PIPELINE_ROOT / "complete-v2.json"
FIRST_CONFIG = EVALUATION_CONFIG_ROOT / "rp67_step2000_first200_gpu0.toml"
FULL_CONFIG = EVALUATION_CONFIG_ROOT / "rp67_step2000_full867_gpu0.toml"
FIRST_MAIN_ROOT = EVALUATION_ROOT / "rp67_step2000_first200_acc_main_mw2_20260801"
FULL_MAIN_ROOT = EVALUATION_ROOT / "rp67_step2000_full867_acc_main_mw2_20260801"
FIRST_SIX_ROOT = EVALUATION_ROOT / "rp67_step2000_first200_6arm_scalar_mw4_20260801"
FIRST_MAIN_SEMANTIC = (
    EVALUATION_ROOT / "rp67_step2000_first200_acc_main_semantic_v2_20260801"
)
FULL_MAIN_SEMANTIC = (
    EVALUATION_ROOT / "rp67_step2000_full867_acc_main_semantic_v2_20260801"
)
FIRST_SIX_SEMANTIC = EVALUATION_ROOT / "rp67_step2000_first200_6arm_semantic_20260801"
JUDGE_CONFIG = (
    REPOSITORY_ROOT / "configs/policy/judges/qwen25_72b_rl_answer_judge_v1.json"
)
JUDGE_CONFIG_SHA256 = "3737504858912a6392679d2c9720597cde58dd7d3218aa6f75b67ad00a769573"
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
COUNTERFACTUAL_SHA256 = (
    "4589d14f196ccde48c3439405700220bc0fb63487edae0e74bc6b3713d7f4cc4"
)
GROUNDING_MANIFEST = REPOSITORY_ROOT / (
    "configs/representation/internal_evaluation/"
    "qwen3_v4_clean_imend_audited_grounding_v1.json"
)
GROUNDING_SHA256 = "a65aa6e6038ada1436302b60440136cc98b388552a7782b48ec95ed4324938c0"
EVALUATION_DATA = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/data/tgvf_teacher/generated/"
    "runs/tgvf_v4_teacher_50k_clean_imend/splits/"
    "tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl"
)
EVALUATION_DATA_SHA256 = (
    "de61c731eb961825a77df587cd76c00eabfea75b5c6003096f3cc7f1a51dd82d"
)
MARKER_SCHEMA = "rp67-all-validations-complete-v2"
PIPELINE_SCHEMA = "rp67-step2000-acc-pipeline-v1"
SEMANTIC_SCHEMA = "answer-utility-semantic-rescore-v2"
MAIN_ARMS = ("image_only", "image_correct_D")
SIX_ARMS = (
    "target_zero_D_only",
    "correct_D_only",
    "matched_wrong_D",
    "direct_zero_D_replacement",
    "direct_correct_D_replacement",
    "direct_matched_wrong_D_replacement",
)


class PipelineBlockedError(RuntimeError):
    """Raised when a fail-closed identity or ownership check fails."""


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )


def _append_event(path: Path, event: str, **fields: object) -> None:
    value = {
        "event": event,
        "at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    with path.open("ab", buffering=0) as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        os.fsync(handle.fileno())


def _assert_pinned_files() -> None:
    for path, expected, label in (
        (EXPERIMENT_CONFIG, EXPERIMENT_CONFIG_SHA256, "experiment config"),
        (TRAINING_CONFIG, TRAINING_CONFIG_SHA256, "training config"),
        (FIRST_MANIFEST, FIRST_MANIFEST_SHA256, "first200 manifest"),
        (FULL_MANIFEST, FULL_MANIFEST_SHA256, "full867 manifest"),
        (COUNTERFACTUAL_MANIFEST, COUNTERFACTUAL_SHA256, "counterfactual manifest"),
        (GROUNDING_MANIFEST, GROUNDING_SHA256, "grounding manifest"),
        (EVALUATION_DATA, EVALUATION_DATA_SHA256, "evaluation split"),
        (JUDGE_CONFIG, JUDGE_CONFIG_SHA256, "judge config"),
    ):
        if path.is_symlink() or not path.is_file() or _file_sha256(path) != expected:
            raise PipelineBlockedError(f"pinned {label} failed its SHA256 binding")


def _proc_starttime(pid: int) -> int | None:
    try:
        value = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    closing = value.rfind(")")
    fields = value[closing + 2 :].split()
    return int(fields[19])


def _training_is_live() -> bool:
    observed = _proc_starttime(TRAINING_PID)
    if observed is None:
        return False
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    if boot != TRAINING_BOOT_ID or observed != TRAINING_STARTTIME_TICKS:
        raise PipelineBlockedError("training PID identity changed before handoff")
    argv = (Path("/proc") / str(TRAINING_PID) / "cmdline").read_bytes()
    if b"run_representation_image_axis_grounding.py" not in argv:
        raise PipelineBlockedError("training PID no longer names the RP67 launcher")
    return True


def _last_metrics_completion() -> Mapping[str, Any]:
    rows = [line for line in METRICS.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise PipelineBlockedError("RP67 metrics ledger is empty")
    value = json.loads(rows[-1])
    if not isinstance(value, dict) or value.get("event") != "complete":
        raise PipelineBlockedError("RP67 metrics ledger has no terminal completion")
    return value


def _run_handoff() -> dict[str, Any]:
    command = [
        str(PYTHON),
        str(REPOSITORY_ROOT / "tools/handoff_representation_image_axis_evaluation.py"),
        "--training-exit-code",
        "0",
        "--outer-result-log",
        str(OUTER_LOG),
        "--expected-run-id",
        RUN_ID,
        "--expected-run-identity-sha256",
        RUN_IDENTITY_SHA256,
        "--expected-global-step",
        str(GLOBAL_STEP),
        "--expected-experiment-config-sha256",
        EXPERIMENT_CONFIG_SHA256,
        "--expected-training-config-sha256",
        TRAINING_CONFIG_SHA256,
        "--expected-artifact-path",
        str(ADAPTER),
        "--expected-metrics-path",
        str(METRICS),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    if completed.returncode != 0:
        raise PipelineBlockedError(
            "RP67 handoff rejected completion: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PipelineBlockedError("handoff did not return one JSON receipt") from error
    if receipt.get("status") != "authorized":
        raise PipelineBlockedError("handoff receipt is not authorized")
    completion = _last_metrics_completion()
    internal = completion.get("post_training_internal_evaluation")
    if not isinstance(internal, dict) or internal.get("status") != "complete":
        raise PipelineBlockedError("INT-DIAG is not complete")
    if Path(str(internal.get("path", ""))).resolve() != INT_DIAG.resolve():
        raise PipelineBlockedError("INT-DIAG completion identifies another path")
    if (
        INT_DIAG.is_symlink()
        or not INT_DIAG.is_file()
        or _file_sha256(INT_DIAG) != internal.get("payload_sha256")
    ):
        raise PipelineBlockedError("INT-DIAG payload failed its metrics binding")
    return receipt


def _toml_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _render_source_config(
    *,
    split: str,
    receipt: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
) -> bytes:
    if split not in {"first200", "full867"}:
        raise ValueError("unknown ACC split")
    run_id = f"RP-67-STEP2000-{split.upper()}-IMAGE-AXIS-UTILITY-EVAL-GPU01"
    evaluation_id = f"rp67-step2000-{split}-answer-utility-v1"
    report_path = PIPELINE_ROOT / f"{split}-internal-evaluation-unused.json"
    values = f"""schema_version = "representation-internal-evaluation-run-v3"
run_id = {_toml_string(run_id)}

[code]
repository = "Miocio-nora/TGVF-E2E-RL"
commit = "d46715ff916f60a5fac1410a0b013d665f8a1f99"
dirty = false

[source]
training_config_path = {_toml_string(TRAINING_CONFIG)}
training_config_sha256 = {_toml_string(TRAINING_CONFIG_SHA256)}

[artifact]
path = {_toml_string(receipt["artifact_path"])}
file_sha256 = {_toml_string(receipt["artifact_file_sha256"])}
manifest_sha256 = {_toml_string(receipt["artifact_manifest_sha256"])}
expected_run_identity_sha256 = {_toml_string(RUN_IDENTITY_SHA256)}
expected_global_step = 2000

[execution]
physical_gpu_id = 0

[evaluation_data]
jsonl_path = {_toml_string(EVALUATION_DATA)}
source_sha256 = {_toml_string(EVALUATION_DATA_SHA256)}

[evaluation]
evaluation_id = {_toml_string(evaluation_id)}
ordered_group_manifest_path = {_toml_string(manifest_path)}
ordered_group_manifest_sha256 = {_toml_string(manifest_sha256)}
counterfactual_manifest_path = {_toml_string(COUNTERFACTUAL_MANIFEST)}
counterfactual_manifest_sha256 = {_toml_string(COUNTERFACTUAL_SHA256)}
grounding_manifest_path = {_toml_string(GROUNDING_MANIFEST)}
grounding_manifest_sha256 = {_toml_string(GROUNDING_SHA256)}
report_path = {_toml_string(report_path)}
random_seed = 42
max_new_tokens = 64
eos_token_ids = [151645]
"""
    return values.encode("utf-8")


def _write_identical_or_new(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise PipelineBlockedError(f"refusing to replace nonidentical {path}")
        return
    _atomic_bytes(path, payload)


def _load_complete_launch(
    root: Path, *, expected_arms: Sequence[str], samples: int
) -> dict[str, Any] | None:
    path = root / "launch-summary.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise PipelineBlockedError(f"launch summary is not regular: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "complete"
        or tuple(value.get("arms", ())) != tuple(expected_arms)
        or value.get("sample_count") != samples
        or value.get("record_count") != samples * len(expected_arms)
    ):
        raise PipelineBlockedError(f"completed generation summary mismatch: {path}")
    return value


def _launch_generation(
    *,
    source_config: Path,
    output_root: Path,
    arms: Sequence[str],
    samples: int,
    workers_per_gpu: int,
    events: Path,
) -> dict[str, Any]:
    existing = _load_complete_launch(output_root, expected_arms=arms, samples=samples)
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
        "--physical-gpu-id",
        "0",
        "--physical-gpu-id",
        "1",
        "--workers-per-gpu",
        str(workers_per_gpu),
        "--eos-token-id",
        "151645",
        "--eos-token-id",
        "151643",
        "--decode-mode",
        "cached",
        "--arm-batch-size",
        "1",
    ]
    for arm in arms:
        command.extend(("--arm", arm))
    _append_event(
        events, "generation_started", output_root=str(output_root), arms=list(arms)
    )
    log_path = PIPELINE_ROOT / f"{output_root.name}.log"
    with log_path.open("ab", buffering=0) as log:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=4 * 60 * 60,
        )
    if completed.returncode != 0:
        raise PipelineBlockedError(f"ACC generation failed; inspect {log_path}")
    value = _load_complete_launch(output_root, expected_arms=arms, samples=samples)
    if value is None:
        raise PipelineBlockedError("ACC launcher exited without a complete summary")
    _append_event(events, "generation_complete", output_root=str(output_root))
    return value


def _gpu_compute_pids(indices: Iterable[int]) -> dict[int, tuple[int, ...]]:
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


def _wait_gpu01_empty(timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not any(_gpu_compute_pids((0, 1)).values()):
            return
        time.sleep(5)
    raise PipelineBlockedError("GPU0/1 did not become compute-empty")


def _semantic_complete(root: Path, *, arms: Sequence[str], samples: int) -> bool:
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    if not summary_path.exists() and not manifest_path.exists():
        return False
    if any(
        path.is_symlink() or not path.is_file()
        for path in (summary_path, manifest_path)
    ):
        raise PipelineBlockedError(f"semantic publication is incomplete: {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_arm = summary.get("by_arm")
    run_identity = summary.get("run_identity_sha256")
    manifest_files = manifest.get("files")
    manifest_summary = (
        manifest_files.get("summary") if isinstance(manifest_files, dict) else None
    )
    if (
        summary.get("schema_version") != SEMANTIC_SCHEMA
        or manifest.get("schema_version") != SEMANTIC_SCHEMA
        or summary.get("status") != "complete"
        or manifest.get("status") != "complete"
        or not isinstance(run_identity, str)
        or len(run_identity) != 64
        or manifest.get("run_identity_sha256") != run_identity
        or not isinstance(by_arm, dict)
        or set(by_arm) != set(arms)
        or summary.get("overall", {}).get("total") != samples * len(arms)
        or any(
            not isinstance(by_arm.get(arm), dict) or by_arm[arm].get("total") != samples
            for arm in arms
        )
        or not isinstance(manifest_summary, dict)
        or manifest_summary.get("path") != "summary.json"
        or manifest_summary.get("sha256") != _file_sha256(summary_path)
        or manifest.get("files", {}).get("overlay_records", {}).get("rows")
        != samples * len(arms)
    ):
        raise PipelineBlockedError(f"semantic publication mismatch: {root}")
    return True


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
        raise PipelineBlockedError(
            f"Python 3.12 development headers are missing: {missing}"
        )
    return os.pathsep.join((str(PYTHON_HEADER_ROOT), str(python_headers)))


def _wait_judge_ready(process: subprocess.Popen[bytes], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise PipelineBlockedError("semantic judge server exited during startup")
        try:
            with urlopen("http://127.0.0.1:8013/v1/models", timeout=2) as response:
                value = json.loads(response.read())
            ids = {item.get("id") for item in value.get("data", [])}
            if ids == {"Qwen2.5-72B-Instruct"}:
                return
        except Exception:
            pass
        time.sleep(2)
    raise PipelineBlockedError("semantic judge server readiness timed out")


def _judge_endpoint_is_open() -> bool:
    try:
        with urlopen("http://127.0.0.1:8013/v1/models", timeout=1) as response:
            response.read(1)
        return True
    except Exception:
        return False


def _run_semantic(
    *,
    generation_summary: Mapping[str, Any],
    source_config: Path,
    output_root: Path,
    arms: Sequence[str],
    samples: int,
    events: Path,
) -> None:
    if _semantic_complete(output_root, arms=arms, samples=samples):
        return
    roots = [str(shard["output_root"]) for shard in generation_summary["shards"]]
    command = [
        str(PYTHON),
        str(
            REPOSITORY_ROOT
            / "tools/run_representation_answer_utility_semantic_rescore.py"
        ),
    ]
    for root in roots:
        command.extend(("--generation-output-root", root))
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
    _append_event(events, "semantic_started", output_root=str(output_root))
    log_path = PIPELINE_ROOT / f"{output_root.name}.log"
    with log_path.open("ab", buffering=0) as log:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=2 * 60 * 60,
        )
    if completed.returncode != 0 or not _semantic_complete(
        output_root, arms=arms, samples=samples
    ):
        raise PipelineBlockedError(f"semantic rescore failed; inspect {log_path}")
    _append_event(events, "semantic_complete", output_root=str(output_root))


def _stop_owned_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _artifact_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise PipelineBlockedError(f"completion artifact is missing: {path}")
    return {
        "status": "complete",
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
    }


def _semantic_artifact_record(
    root: Path, *, arms: Sequence[str], samples: int
) -> dict[str, object]:
    if not _semantic_complete(root, arms=arms, samples=samples):
        raise PipelineBlockedError(f"semantic publication is missing: {root}")
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "status": "complete",
        "root": str(root.resolve()),
        "run_identity_sha256": summary["run_identity_sha256"],
        "summary": _artifact_record(summary_path),
        "manifest": _artifact_record(manifest_path),
    }


def _artifact_record_is_current(record: object, *, expected_path: Path) -> bool:
    if not isinstance(record, dict):
        return False
    path = Path(str(record.get("path", "")))
    return (
        record.get("status") == "complete"
        and path == expected_path.resolve()
        and not path.is_symlink()
        and path.is_file()
        and _file_sha256(path) == record.get("sha256")
    )


def _semantic_artifact_record_is_current(
    record: object,
    *,
    expected_root: Path,
    arms: Sequence[str],
    samples: int,
) -> bool:
    if not isinstance(record, dict) or record.get("status") != "complete":
        return False
    if Path(str(record.get("root", ""))) != expected_root.resolve():
        return False
    if not _semantic_complete(expected_root, arms=arms, samples=samples):
        return False
    summary = json.loads((expected_root / "summary.json").read_text(encoding="utf-8"))
    return (
        record.get("run_identity_sha256") == summary.get("run_identity_sha256")
        and _artifact_record_is_current(
            record.get("summary"), expected_path=expected_root / "summary.json"
        )
        and _artifact_record_is_current(
            record.get("manifest"), expected_path=expected_root / "manifest.json"
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    run = subparsers.add_parser("run")
    run.add_argument("--execute", action="store_true", required=True)
    return parser


def _status() -> dict[str, object]:
    summaries = {
        "first200_main": FIRST_MAIN_SEMANTIC / "summary.json",
        "full867_main": FULL_MAIN_SEMANTIC / "summary.json",
        "first200_sixarm": FIRST_SIX_SEMANTIC / "summary.json",
    }
    return {
        "training_live": _training_is_live(),
        "adapter_exists": ADAPTER.is_file(),
        "int_diag_exists": INT_DIAG.is_file(),
        "semantic_summaries": {
            name: path.is_file() for name, path in summaries.items()
        },
        "complete_marker_exists": COMPLETE_MARKER.is_file(),
    }


def _existing_complete_marker_is_valid() -> bool:
    if not COMPLETE_MARKER.exists():
        return False
    if COMPLETE_MARKER.is_symlink() or not COMPLETE_MARKER.is_file():
        raise PipelineBlockedError("existing completion marker is not regular")
    value = json.loads(COMPLETE_MARKER.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != MARKER_SCHEMA
        or value.get("status") != "complete"
        or value.get("rp67_run_id") != RUN_ID
    ):
        raise PipelineBlockedError("existing completion marker identity mismatch")
    artifacts = value.get("artifacts")
    expected = {
        "int_diag",
        "acc_first200",
        "acc_full867",
        "diag_first200_sixarm",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise PipelineBlockedError("existing completion marker artifact set mismatch")
    if not _artifact_record_is_current(artifacts["int_diag"], expected_path=INT_DIAG):
        raise PipelineBlockedError("existing completion marker INT-DIAG drifted")
    semantic_bindings = (
        ("acc_first200", FIRST_MAIN_SEMANTIC, MAIN_ARMS, 200),
        ("acc_full867", FULL_MAIN_SEMANTIC, MAIN_ARMS, 867),
        ("diag_first200_sixarm", FIRST_SIX_SEMANTIC, SIX_ARMS, 200),
    )
    for name, root, arms, samples in semantic_bindings:
        if not _semantic_artifact_record_is_current(
            artifacts[name], expected_root=root, arms=arms, samples=samples
        ):
            raise PipelineBlockedError(
                f"existing completion marker semantic artifact {name} drifted"
            )
    return True


def main() -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/run_rp67_step2000_acc_pipeline.py"
    )
    args = _parser().parse_args()
    _assert_pinned_files()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.command == "status":
        print(json.dumps(_status(), indent=2, sort_keys=True))
        return 0
    if _existing_complete_marker_is_valid():
        return 0

    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = (PIPELINE_ROOT / "pipeline.lock").open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise PipelineBlockedError("another ACC pipeline instance is active") from error
    events = PIPELINE_ROOT / "events.jsonl"
    heartbeat = PIPELINE_ROOT / "heartbeat.json"
    _append_event(events, "pipeline_started", pid=os.getpid())
    while _training_is_live():
        _atomic_json(
            heartbeat,
            {
                "schema_version": PIPELINE_SCHEMA,
                "status": "waiting_for_rp67",
                "pid": os.getpid(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        time.sleep(args.poll_seconds)

    # The original launcher did not persist a wait(2) exit status.  Passing
    # zero here is an explicit inference, subsequently guarded by independent
    # outer-result, terminal-metrics, Adapter and INT-DIAG completion records.
    # A failed closeout can be repaired from its durable checkpoint; retain the
    # controller and retry the strict gate instead of silently abandoning ACC.
    previous_handoff_error: str | None = None
    while True:
        try:
            receipt = _run_handoff()
        except PipelineBlockedError as error:
            message = str(error)
            if message != previous_handoff_error:
                _append_event(events, "handoff_waiting_for_recovery", error=message)
                previous_handoff_error = message
            _atomic_json(
                heartbeat,
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "status": "waiting_for_strict_handoff",
                    "pid": os.getpid(),
                    "last_error": message,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            time.sleep(args.poll_seconds)
            continue
        break
    _atomic_json(PIPELINE_ROOT / "handoff-receipt.json", receipt)
    _append_event(events, "handoff_authorized", receipt=receipt)
    _write_identical_or_new(
        FIRST_CONFIG,
        _render_source_config(
            split="first200",
            receipt=receipt,
            manifest_path=FIRST_MANIFEST,
            manifest_sha256=FIRST_MANIFEST_SHA256,
        ),
    )
    _write_identical_or_new(
        FULL_CONFIG,
        _render_source_config(
            split="full867",
            receipt=receipt,
            manifest_path=FULL_MANIFEST,
            manifest_sha256=FULL_MANIFEST_SHA256,
        ),
    )
    _wait_gpu01_empty(600)
    first_main = _launch_generation(
        source_config=FIRST_CONFIG,
        output_root=FIRST_MAIN_ROOT,
        arms=MAIN_ARMS,
        samples=200,
        workers_per_gpu=1,
        events=events,
    )
    full_main = _launch_generation(
        source_config=FULL_CONFIG,
        output_root=FULL_MAIN_ROOT,
        arms=MAIN_ARMS,
        samples=867,
        workers_per_gpu=1,
        events=events,
    )
    first_six = _launch_generation(
        source_config=FIRST_CONFIG,
        output_root=FIRST_SIX_ROOT,
        arms=SIX_ARMS,
        samples=200,
        workers_per_gpu=2,
        events=events,
    )
    _wait_gpu01_empty(300)
    if any(_gpu_compute_pids((0, 1)).values()):
        raise PipelineBlockedError("GPU0/1 is occupied before judge launch")
    semantic_jobs = (
        (first_main, FIRST_CONFIG, FIRST_MAIN_SEMANTIC, MAIN_ARMS, 200),
        (full_main, FULL_CONFIG, FULL_MAIN_SEMANTIC, MAIN_ARMS, 867),
        (first_six, FIRST_CONFIG, FIRST_SIX_SEMANTIC, SIX_ARMS, 200),
    )
    semantic_pending = any(
        not _semantic_complete(root, arms=arms, samples=samples)
        for _, _, root, arms, samples in semantic_jobs
    )
    if semantic_pending and _judge_endpoint_is_open():
        raise PipelineBlockedError("port 8013 is already serving an unowned endpoint")
    judge_log = (PIPELINE_ROOT / "judge-server.log").open("ab", buffering=0)
    judge_env = {
        **os.environ,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1",
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
        "TRITON_CACHE_DIR": str(PIPELINE_ROOT / "cache/judge-triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(PIPELINE_ROOT / "cache/judge-torchinductor"),
    }
    Path(judge_env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(judge_env["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    judge: subprocess.Popen[bytes] | None = None
    try:
        if semantic_pending:
            judge = subprocess.Popen(
                _judge_command(),
                cwd=REPOSITORY_ROOT,
                env=judge_env,
                stdout=judge_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _append_event(events, "judge_started", pgid=judge.pid)
            _wait_judge_ready(judge, 300)
            for generation, config, root, arms, samples in semantic_jobs:
                _run_semantic(
                    generation_summary=generation,
                    source_config=config,
                    output_root=root,
                    arms=arms,
                    samples=samples,
                    events=events,
                )
    finally:
        if judge is not None:
            _stop_owned_group(judge)
            _append_event(events, "judge_stopped", pgid=judge.pid)
        judge_log.close()
    _wait_gpu01_empty(300)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "rp67_run_id": RUN_ID,
        "rp67_run_identity_sha256": RUN_IDENTITY_SHA256,
        "handoff_receipt_sha256": _file_sha256(PIPELINE_ROOT / "handoff-receipt.json"),
        "evaluation_tool_sha256": _file_sha256(
            REPOSITORY_ROOT / "tools/run_representation_answer_utility_evaluation.py"
        ),
        "launcher_tool_sha256": _file_sha256(
            REPOSITORY_ROOT / "tools/launch_representation_answer_utility_evaluation.py"
        ),
        "semantic_tool_sha256": _file_sha256(
            REPOSITORY_ROOT
            / "tools/run_representation_answer_utility_semantic_rescore.py"
        ),
        "artifacts": {
            "int_diag": _artifact_record(INT_DIAG),
            "acc_first200": _semantic_artifact_record(
                FIRST_MAIN_SEMANTIC, arms=MAIN_ARMS, samples=200
            ),
            "acc_full867": _semantic_artifact_record(
                FULL_MAIN_SEMANTIC, arms=MAIN_ARMS, samples=867
            ),
            "diag_first200_sixarm": _semantic_artifact_record(
                FIRST_SIX_SEMANTIC, arms=SIX_ARMS, samples=200
            ),
        },
    }
    _atomic_json(COMPLETE_MARKER, marker)
    _atomic_json(
        PIPELINE_COMPLETE,
        {**marker, "complete_marker_sha256": _file_sha256(COMPLETE_MARKER)},
    )
    _append_event(events, "pipeline_complete", marker=str(COMPLETE_MARKER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
