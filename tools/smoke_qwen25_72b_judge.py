#!/usr/bin/python3 -I
"""Validate the pinned Qwen2.5-72B OpenAI-compatible judge service."""

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
            "tools/smoke_qwen25_72b_judge.py",
        ),
    )

import argparse
import json
from pathlib import Path

from tgvf_rl.evaluation.coredev_results import (
    check_qwen25_72b_judge,
    write_json_atomic,
)
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)


EXPECTED_MODEL = "Qwen2.5-72B-Instruct"


def main() -> int:
    assert_legacy_standalone_execution_quarantined("tools/smoke_qwen25_72b_judge.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8012/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_qwen25_72b_judge(
        base_url=args.base_url,
        expected_model=EXPECTED_MODEL,
    )
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/python3 -I
