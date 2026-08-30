#!/usr/bin/python3 -I
"""Prove that a completed one-step Policy run resumes without another update."""

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
            "tools/prove_policy_auto_resume.py",
        ),
    )

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


PROOF_SCHEMA_VERSION = "tgvf-policy-auto-resume-proof-v1"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise RuntimeError(f"checkpoint directory is missing: {root}")
    digest = sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"checkpoint directory is empty: {root}")
    for path in files:
        if path.is_symlink():
            raise RuntimeError(f"checkpoint contains a symlink: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _metrics_snapshot(path: Path, *, expected_step: int) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Policy metrics are missing: {path}")
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Policy metrics are invalid at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Policy metrics record is not an object: {path}:{line_number}"
            )
        records.append(record)
    if not records:
        raise RuntimeError(f"Policy metrics are empty: {path}")
    observed_step = records[-1].get("optimizer_step")
    if observed_step != expected_step:
        raise RuntimeError(
            f"Policy metrics last optimizer_step is {observed_step!r}, "
            f"expected {expected_step}"
        )
    return {
        "sha256": sha256(raw).hexdigest(),
        "bytes": len(raw),
        "records": len(records),
        "last_optimizer_step": observed_step,
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def prove_auto_resume(config_path: Path, proof_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    proof_path = proof_path.resolve(strict=False)
    config = load_policy_e2e_smoke_run_config(config_path)
    expected_step = config.training.maximum_optimizer_steps
    if expected_step != 1:
        raise RuntimeError("auto-resume proof is restricted to a one-step smoke")
    checkpoint = config.output.checkpoint_directory / f"global_step_{expected_step}"
    before_metrics = _metrics_snapshot(
        config.output.metrics_path, expected_step=expected_step
    )
    before_checkpoint = _tree_sha256(checkpoint)
    completed = subprocess.run(
        [sys.executable, "-m", "tgvf_rl.cli", "run-policy", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Policy auto-resume invocation exited {completed.returncode}"
        )
    after_metrics = _metrics_snapshot(
        config.output.metrics_path, expected_step=expected_step
    )
    after_checkpoint = _tree_sha256(checkpoint)
    if before_metrics != after_metrics:
        raise RuntimeError("auto-resume changed the completed one-step metrics")
    if before_checkpoint != after_checkpoint:
        raise RuntimeError("auto-resume changed the completed step-1 checkpoint")
    if (config.output.checkpoint_directory / "global_step_2").exists():
        raise RuntimeError("auto-resume unexpectedly created global_step_2")
    proof = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "run_identity_sha256": config.identity_sha256,
        "resume": {
            "proven": True,
            "expected_optimizer_step": expected_step,
            "metrics": after_metrics,
            "checkpoint_tree_sha256": after_checkpoint,
            "extra_optimizer_step_absent": True,
        },
    }
    _atomic_json(proof_path, proof)
    return proof


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--proof", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    assert_legacy_standalone_execution_quarantined("tools/prove_policy_auto_resume.py")
    arguments = _parser().parse_args(argv)
    try:
        proof = prove_auto_resume(arguments.config, arguments.proof)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"auto-resume proof failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(proof, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
