#!/usr/bin/python3 -I
"""Validate and aggregate the seven independently scored CoreDev-2511 slices."""

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
            "tools/summarize_coredev_2511.py",
        ),
    )

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.coredev_results import (  # noqa: E402
    summarize_coredev_results,
    write_json_atomic,
)
from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)


def main() -> int:
    assert_legacy_standalone_execution_quarantined("tools/summarize_coredev_2511.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("infer", "eval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-base-url")
    args = parser.parse_args()

    summary = summarize_coredev_results(
        work_dir=args.work_dir.resolve(),
        repository_root=REPOSITORY_ROOT,
        phase=args.phase,
        expected_judge_base_url=args.judge_base_url,
    )
    write_json_atomic(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/python3 -I
