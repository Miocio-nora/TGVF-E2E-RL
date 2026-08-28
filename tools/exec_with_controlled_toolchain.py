#!/usr/bin/env python3
"""Exec a command after applying the shared fail-closed compiler environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.controlled_toolchain import (  # noqa: E402
    build_controlled_toolchain_environment,
    controlled_toolchain_contract,
    controlled_toolchain_verification,
    python312_toolchain_environment,
)


EXECUTION_CONTRACT_SCHEMA = "tgvf.controlled-toolchain-exec.v1"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise RuntimeError(f"controlled toolchain contract differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _parse_overlay(values: list[str]) -> dict[str, str]:
    overlay: dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or not name or name in overlay:
            raise ValueError("controlled environment overlay is malformed")
        overlay[name] = item
    return overlay


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-environment-root", type=Path, required=True)
    parser.add_argument("--python-header-root", type=Path, required=True)
    parser.add_argument("--environment", action="append", default=[])
    parser.add_argument("--contract-out", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    if not command or not Path(command[0]).is_file():
        raise RuntimeError("controlled toolchain command is absent")
    controlled = python312_toolchain_environment(
        python_environment_root=args.python_environment_root,
        python_header_root=args.python_header_root,
    )
    overlay = _parse_overlay(args.environment)
    environment = build_controlled_toolchain_environment(
        controlled=controlled,
        overlay=overlay,
    )
    content: dict[str, object] = {
        "schema_version": EXECUTION_CONTRACT_SCHEMA,
        "toolchain": controlled_toolchain_contract(controlled),
        "verification": controlled_toolchain_verification(
            environment,
            controlled=controlled,
        ),
        "runtime_overlay": overlay,
        "command": command,
    }
    contract = {**content, "identity_sha256": _canonical_sha256(content)}
    _write_immutable_json(args.contract_out.resolve(), contract)
    if args.validate_only:
        print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    os.execvpe(command[0], command, environment)
    raise AssertionError("os.execvpe returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
