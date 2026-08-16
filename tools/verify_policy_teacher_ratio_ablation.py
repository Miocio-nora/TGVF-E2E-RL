#!/usr/bin/env python3
"""Verify PRL22/23 Teacher25/50/100 artifacts and nested populations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgvf_rl.data.policy_teacher_quarter_mix import (  # noqa: E402
    PolicyTeacherQuarterMixRuntimeBinding,
    load_policy_teacher_quarter_mix_runtime,
)
from tgvf_rl.data.policy_teacher_ratio_mix import (  # noqa: E402
    PolicyTeacherRatioMixRuntimeBinding,
    load_policy_teacher_ratio_mix_runtime,
)


_DATA_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_rl"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher25-root",
        type=Path,
        default=_DATA_ROOT / "PRL22-TEACHER25-MIXED-SCHEDULE-v1",
    )
    parser.add_argument(
        "--teacher50-root",
        type=Path,
        default=_DATA_ROOT / "PRL23-TEACHER50-MIXED-SCHEDULE-v1",
    )
    parser.add_argument(
        "--teacher100-root",
        type=Path,
        default=_DATA_ROOT / "PRL23-TEACHER100-MIXED-SCHEDULE-v1",
    )
    return parser


def _manifest(root: Path) -> tuple[dict[str, object], str]:
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest is not an object: {root}")
    return manifest, hashlib.sha256(raw).hexdigest()


def _roles(samples: object) -> tuple[set[str], set[str]]:
    teacher: set[str] = set()
    base: set[str] = set()
    for sample in samples:
        target = teacher if sample.data_source == "teacher" else base
        if sample.sample_id in target:
            raise ValueError(f"duplicate sample_id: {sample.sample_id}")
        target.add(sample.sample_id)
    return base, teacher


def main() -> int:
    args = _parser().parse_args()
    manifest25, manifest_file25 = _manifest(args.teacher25_root)
    runtime25 = load_policy_teacher_quarter_mix_runtime(
        args.teacher25_root,
        binding=PolicyTeacherQuarterMixRuntimeBinding(
            manifest_file_sha256=manifest_file25,
            content_sha256=str(manifest25["content_sha256"]),
            schedule_seed=42,
            expected_sample_count=20_480,
        ),
    )
    runtimes = {25: runtime25}
    records: dict[int, dict[str, object]] = {
        25: {
            "root": str(args.teacher25_root),
            "manifest_file_sha256": manifest_file25,
            "content_sha256": manifest25["content_sha256"],
            "samples_sha256": runtime25.samples_sha256,
            "iteration_identity_sha256": runtime25.iteration_identity_sha256,
        }
    }
    for percentage, root in (
        (50, args.teacher50_root),
        (100, args.teacher100_root),
    ):
        manifest, manifest_file = _manifest(root)
        runtime = load_policy_teacher_ratio_mix_runtime(
            root,
            binding=PolicyTeacherRatioMixRuntimeBinding(
                manifest_file_sha256=manifest_file,
                content_sha256=str(manifest["content_sha256"]),
                schedule_seed=42,
                expected_sample_count=20_480,
                teacher_percentage=percentage,
            ),
        )
        runtimes[percentage] = runtime
        records[percentage] = {
            "root": str(root),
            "manifest_file_sha256": manifest_file,
            "content_sha256": manifest["content_sha256"],
            "samples_sha256": runtime.samples_sha256,
            "iteration_identity_sha256": runtime.iteration_identity_sha256,
        }

    role_sets = {
        percentage: _roles(runtime.samples)
        for percentage, runtime in runtimes.items()
    }
    base25, teacher25 = role_sets[25]
    base50, teacher50 = role_sets[50]
    base100, teacher100 = role_sets[100]
    if not teacher25 < teacher50 < teacher100:
        raise ValueError("teacher populations are not strict nested prefixes")
    if not base50 < base25 or base100:
        raise ValueError("base populations do not shrink as contracted")
    proof = {
        "schema_version": "tgvf.policy-teacher-ratio-ablation-proof.v1",
        "status": "pass",
        "artifacts": {str(key): value for key, value in records.items()},
        "population_counts": {
            "teacher25": len(teacher25),
            "teacher50": len(teacher50),
            "teacher100": len(teacher100),
            "teacher50_minus_teacher25": len(teacher50 - teacher25),
            "teacher100_minus_teacher50": len(teacher100 - teacher50),
            "base25": len(base25),
            "base50": len(base50),
            "base100": len(base100),
        },
        "strict_teacher_nesting": True,
        "strict_base_shrinkage": True,
    }
    print(json.dumps(proof, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
