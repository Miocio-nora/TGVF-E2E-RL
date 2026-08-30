#!/usr/bin/python3 -I
"""Summarize answer-relevant versus bonus-only PRL13 GRPO group signal."""

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
            "tools/analyze_prl13_group_signal.py",
        ),
    )

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from tgvf_rl.policy.prl13_group_signal import summarize_group_signal  # noqa: E402
from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", nargs="+", type=Path)
    parser.add_argument("--include-groups", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _load(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            handle = path.open(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("record is not an object")
                    records.append(value)
                except (json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"{path}:{line_number}: {exc}")
    return records, errors


def main(argv: Sequence[str] | None = None) -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/analyze_prl13_group_signal.py"
    )
    args = _parser().parse_args(argv)
    records, errors = _load(args.trajectories)
    if not records:
        raise SystemExit("no valid trajectory records")
    report = summarize_group_signal(records, include_groups=args.include_groups)
    report["input_files"] = [str(path.resolve()) for path in args.trajectories]
    report["malformed_records"] = len(errors)
    report["malformed_examples"] = errors[:10]
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
