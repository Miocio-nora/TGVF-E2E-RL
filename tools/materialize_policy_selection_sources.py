#!/usr/bin/python3 -I
"""Materialize pinned Policy RL source candidates without loading a model."""

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
            "tools/materialize_policy_selection_sources.py",
        ),
    )

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.data.policy_selection_sources import (  # noqa: E402
    materialize_arxivqa_candidates,
    materialize_thinklite_candidates,
    materialize_vstar_candidates,
    materialize_vstar_images,
)
from tgvf_rl.data.policy_selection_screening import (  # noqa: E402
    screen_policy_selection_candidates,
)
from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)


def main(argv: list[str] | None = None) -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/materialize_policy_selection_sources.py"
    )
    parser = argparse.ArgumentParser(
        description="CPU-only materialization of pinned RL selection candidates"
    )
    subparsers = parser.add_subparsers(dest="source", required=True)

    vstar = subparsers.add_parser("vstar")
    vstar.add_argument("--annotations-root", type=Path, required=True)
    vstar.add_argument("--image-root", type=Path, required=True)
    vstar.add_argument("--output-root", type=Path, required=True)

    vstar_images = subparsers.add_parser("vstar-images")
    vstar_images.add_argument("--annotations-root", type=Path, required=True)
    vstar_images.add_argument("--archives-root", type=Path, required=True)
    vstar_images.add_argument("--output-root", type=Path, required=True)
    vstar_images.add_argument("--gqa-mirror-archive", type=Path)

    arxivqa = subparsers.add_parser("arxivqa")
    arxivqa.add_argument("--annotations", type=Path, required=True)
    arxivqa.add_argument("--images-archive", type=Path, required=True)
    arxivqa.add_argument("--image-root", type=Path, required=True)
    arxivqa.add_argument("--output-root", type=Path, required=True)

    thinklite = subparsers.add_parser("thinklite")
    thinklite.add_argument("--parquet", type=Path, required=True)
    thinklite.add_argument("--output-root", type=Path, required=True)

    screen = subparsers.add_parser("screen-heldout")
    screen.add_argument("--candidates", type=Path, required=True)
    screen.add_argument("--heldout-tasks", type=Path, required=True)
    screen.add_argument("--output-root", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.source == "vstar":
        result = materialize_vstar_candidates(
            args.annotations_root, args.image_root, args.output_root
        )
    elif args.source == "vstar-images":
        result = materialize_vstar_images(
            args.annotations_root,
            args.archives_root,
            args.output_root,
            gqa_mirror_archive=args.gqa_mirror_archive,
        )
    elif args.source == "arxivqa":
        result = materialize_arxivqa_candidates(
            args.annotations,
            args.images_archive,
            args.image_root,
            args.output_root,
        )
    elif args.source == "screen-heldout":
        result = screen_policy_selection_candidates(
            args.candidates, args.heldout_tasks, args.output_root
        )
    else:
        result = materialize_thinklite_candidates(args.parquet, args.output_root)
    print(json.dumps(result.as_record(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/python3 -I
