#!/usr/bin/python3 -I
"""Quarantined legacy T1 retained-pool materialization entry point.

The checked-in final-scoring producer emits schema v1, while the retained-pool
consumer requires schema v2.  Keep the underlying historical implementations
unchanged and fail closed here until a content-bound v1-to-v2 migration or a
native v2 producer is reviewed end to end.
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
            "tools/materialize_policy_t1_mixed_retained_pool.py",
        ),
    )

import argparse
import json
from pathlib import Path

from tgvf_rl.data.policy_t1_mixed_rl_dataset import (
    T1_04_EXPECTED_SOURCE_COUNTS,
    materialize_policy_t1_mixed_retained_pool,
)
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)


def main() -> None:
    assert_legacy_standalone_execution_quarantined(
        "tools/materialize_policy_t1_mixed_retained_pool.py"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--expected-vstar-count",
        type=int,
        default=T1_04_EXPECTED_SOURCE_COUNTS["vstar"],
    )
    parser.add_argument(
        "--expected-arxivqa-count",
        type=int,
        default=T1_04_EXPECTED_SOURCE_COUNTS["arxivqa"],
    )
    parser.add_argument(
        "--expected-thinklite-count",
        type=int,
        default=T1_04_EXPECTED_SOURCE_COUNTS["thinklite"],
    )
    args = parser.parse_args()
    result = materialize_policy_t1_mixed_retained_pool(
        args.candidates,
        args.final_manifest,
        args.output_root,
        shuffle_seed=args.shuffle_seed,
        expected_source_counts={
            "vstar": args.expected_vstar_count,
            "arxivqa": args.expected_arxivqa_count,
            "thinklite": args.expected_thinklite_count,
        },
    )
    print(json.dumps(result.as_record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
