#!/usr/bin/python3 -I
"""Quarantined legacy handoff to image-axis evaluation."""
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
            "tools/handoff_representation_image_axis_evaluation.py",
        ),
    )

from collections.abc import Sequence

from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)
from tgvf_rl.representation.experiments.image_axis_grounding.handoff import (
    main as _legacy_main,
)


def main(argv: Sequence[str] | None = None) -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/handoff_representation_image_axis_evaluation.py"
    )
    return _legacy_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
