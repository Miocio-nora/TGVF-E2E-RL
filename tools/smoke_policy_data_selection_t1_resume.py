#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the GPU-3 Instruct T1 stop/resume equivalence smoke.

The commands are intentionally split.  ``plan`` is CPU-only.  GPU execution is
never started unless ``baseline``, ``interrupt``, or ``resume`` is requested.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from tgvf_rl.data.policy_selection_t1_resume_smoke import (
    archive_t1_continuous_baseline,
    build_t1_resume_smoke_plan,
    compare_t1_resume_with_continuous,
    t1_resume_smoke_core_digest,
    validate_t1_resume_smoke_prefix,
    write_t1_resume_smoke_artifact,
)
from tgvf_rl.data.policy_selection_vllm import prepare_output_root
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_mode_quarantined,
)

_WORKER = _REPO_ROOT / "tools" / "run_policy_data_selection_t1.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "baseline", "interrupt", "resume"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    return parser


def _worker_command(config: Path) -> list[str]:
    return [
        sys.executable,
        str(_WORKER),
        "worker",
        "--config",
        str(config),
        "--rank",
        "3",
        "--budget-revision",
        "0",
        "--max-chunks",
        "2",
    ]


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "3",
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        }
    )
    return environment


def _decode_json_stream(payload: str) -> list[dict[str, Any]]:
    """Extract JSON objects while ignoring vLLM's human-readable log stream."""

    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    offset = 0
    while offset < len(payload):
        start = payload.find("{", offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(payload, start)
        except json.JSONDecodeError:
            offset = start + 1
            continue
        offset = end
        if not isinstance(value, dict):
            continue
        records.append(value)
    return records


def _run_worker(config: Path, log_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        _worker_command(config),
        cwd=_REPO_ROOT,
        env=_worker_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"T1 worker failed with {completed.returncode}; see {log_path}"
        )
    records = _decode_json_stream(completed.stdout)
    if not records:
        raise RuntimeError("T1 worker returned no JSON result")
    return records[-1]


def _interrupt_worker_after_first_commit(config: Path, log_path: Path) -> int:
    process = subprocess.Popen(
        _worker_command(config),
        cwd=_REPO_ROOT,
        env=_worker_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    lines: list[str] = []
    committed = 0
    signalled = False
    for line in process.stdout:
        lines.append(line)
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event") == "chunk_committed":
            committed += 1
            if committed == 1:
                os.killpg(process.pid, signal.SIGINT)
                signalled = True
    returncode = process.wait()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(lines), encoding="utf-8")
    if not signalled:
        raise RuntimeError("worker exited before the first chunk-commit signal")
    if returncode == 0:
        raise RuntimeError("interrupted worker unexpectedly exited successfully")
    return returncode


def _plan_record(config: Path) -> dict[str, Any]:
    plan = build_t1_resume_smoke_plan(config)
    record = plan.as_record()
    record["commands"] = {
        "baseline": _worker_command(plan.config_path),
        "interrupt": _worker_command(plan.config_path),
        "resume": _worker_command(plan.config_path),
        "idempotency": _worker_command(plan.config_path),
    }
    record["prompt_assertion"] = (
        "one user turn: image then canonical question; no system; no tools; "
        "no RL think trigger; add_generation_prompt=true"
    )
    return record


def _run_baseline(config: Path) -> dict[str, Any]:
    plan = build_t1_resume_smoke_plan(config)
    if plan.continuous_baseline_root.exists():
        raise FileExistsError("continuous baseline archive already exists")
    worker_log = plan.audit_root / "continuous-worker.log"
    recovered = plan.output_root.exists()
    if recovered:
        validate_t1_resume_smoke_prefix(plan, committed_chunks=plan.max_chunks)
        if not worker_log.is_file():
            raise FileNotFoundError(worker_log)
        records = _decode_json_stream(worker_log.read_text(encoding="utf-8"))
        if not records:
            raise RuntimeError("completed baseline log contains no JSON result")
        worker = records[-1]
    else:
        prepare_output_root(plan.config_path)
        worker = _run_worker(plan.config_path, worker_log)
    if worker.get("chunks_written") != 2 or worker.get("records_written") != 64:
        raise RuntimeError("continuous worker did not write the exact two-chunk plan")
    snapshot = archive_t1_continuous_baseline(plan)
    return {
        "worker": worker,
        "snapshot": snapshot,
        "recovered_completed_worker": recovered,
    }


def _run_interrupt(config: Path) -> dict[str, Any]:
    plan = build_t1_resume_smoke_plan(config)
    if not plan.continuous_baseline_root.is_dir():
        raise FileNotFoundError("continuous baseline must be archived first")
    if plan.output_root.exists():
        raise FileExistsError("interrupt phase requires a fresh active root")
    prepare_output_root(plan.config_path)
    returncode = _interrupt_worker_after_first_commit(
        plan.config_path, plan.audit_root / "interrupted-worker.log"
    )
    snapshot = validate_t1_resume_smoke_prefix(plan, committed_chunks=1)
    result = {"worker_returncode": returncode, "snapshot": snapshot}
    write_t1_resume_smoke_artifact(
        plan.audit_root / "interrupted-prefix-snapshot.json", result
    )
    return result


def _run_resume(config: Path) -> dict[str, Any]:
    plan = build_t1_resume_smoke_plan(config)
    validate_t1_resume_smoke_prefix(plan, committed_chunks=1)
    resumed = _run_worker(plan.config_path, plan.audit_root / "resumed-worker.log")
    if (
        resumed.get("chunks_written") != 1
        or resumed.get("records_written") != 32
        or resumed.get("records_resumed") != 32
    ):
        raise RuntimeError("resume worker did not skip one and write one chunk")
    validate_t1_resume_smoke_prefix(plan, committed_chunks=2)
    digest_before = t1_resume_smoke_core_digest(plan.output_root, plan)
    idempotent = _run_worker(
        plan.config_path, plan.audit_root / "idempotent-worker.log"
    )
    digest_after = t1_resume_smoke_core_digest(plan.output_root, plan)
    if idempotent != {
        "run_id": plan.run.run_id,
        "rank": 3,
        "budget_revision": 0,
        "chunks_written": 0,
        "records_written": 0,
        "records_resumed": 64,
    }:
        raise RuntimeError("idempotent worker result differs from the exact contract")
    if digest_before != digest_after:
        raise RuntimeError("idempotent rerun changed immutable run/chunk state")
    comparison = compare_t1_resume_with_continuous(plan)
    report = {
        **comparison,
        "resume_worker": resumed,
        "idempotent_worker": idempotent,
        "idempotent_core_sha256_before": digest_before,
        "idempotent_core_sha256_after": digest_after,
    }
    write_t1_resume_smoke_artifact(plan.audit_root / "resume-smoke-report.json", report)
    return report


def main() -> None:
    args = _parser().parse_args()
    assert_legacy_standalone_mode_quarantined(
        "tools/smoke_policy_data_selection_t1_resume.py",
        selected_mode=args.command,
        read_only_modes=("plan",),
        blocked_modes=("baseline", "interrupt", "resume"),
    )
    if args.command == "plan":
        result = _plan_record(args.config)
    elif args.command == "baseline":
        result = _run_baseline(args.config)
    elif args.command == "interrupt":
        result = _run_interrupt(args.config)
    else:
        result = _run_resume(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
