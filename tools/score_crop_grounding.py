#!/usr/bin/env python3
"""Score one behavior step against a frozen Crop grounding probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.crop_grounding import score_crop_grounding  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--probe-manifest-sha256", required=True)
    parser.add_argument("--trajectory-audit-root", type=Path, required=True)
    parser.add_argument("--behavior-step", type=int, required=True)
    parser.add_argument(
        "--audit-mode", choices=("training", "benchmark"), default="training"
    )
    parser.add_argument("--evaluation-identity", type=Path)
    parser.add_argument("--evaluation-identity-file-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_json_exclusive(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"output is not a regular file: {path}")
        if path.read_bytes() != encoded:
            raise RuntimeError(f"output already exists with different content: {path}")
        return
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
            if path.read_bytes() != encoded:
                raise RuntimeError(
                    f"output already exists with different content: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = score_crop_grounding(
        candidate_manifest_path=args.candidate_manifest,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        probe_manifest_path=args.probe_manifest,
        probe_manifest_sha256=args.probe_manifest_sha256,
        trajectory_audit_root=args.trajectory_audit_root,
        behavior_step=args.behavior_step,
        audit_mode=args.audit_mode,
        evaluation_identity_path=args.evaluation_identity,
        evaluation_identity_file_sha256=args.evaluation_identity_file_sha256,
    )
    _write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
