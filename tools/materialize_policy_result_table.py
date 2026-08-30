#!/usr/bin/python3 -I
"""Validate the policy-result registry and materialize its Markdown tables."""

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
            "tools/materialize_policy_result_table.py",
        ),
    )

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.result_registry import (  # noqa: E402
    RegistryValidationError,
    load_result_registry,
)
from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)


DEFAULT_REGISTRY = REPOSITORY_ROOT / "evidence/policy/result_registry_v2.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate evidence tables; cross-contract deltas are refused by the "
            "typed registry."
        )
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        help="Materialize only this table ID; may be supplied more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write Markdown atomically instead of printing it to stdout.",
    )
    parser.add_argument(
        "--unsafe-skip-artifact-verification",
        action="store_true",
        help=(
            "Diagnostic-only escape hatch: render without verifying score and "
            "preregistration artifacts. Cannot be combined with --output."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository root used to resolve score-artifact paths.",
    )
    return parser


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/materialize_policy_result_table.py"
    )
    args = _parser().parse_args()
    if args.unsafe_skip_artifact_verification and args.output is not None:
        print(
            "result registry rejected: --unsafe-skip-artifact-verification "
            "cannot be combined with --output",
            file=sys.stderr,
        )
        return 2
    try:
        registry = load_result_registry(args.registry)
        if not args.unsafe_skip_artifact_verification:
            registry.verify_artifacts(args.artifact_root)
        rendered = registry.render_markdown(args.tables)
    except RegistryValidationError as error:
        print(f"result registry rejected: {error}", file=sys.stderr)
        return 2
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        _write_atomic(args.output, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
