#!/usr/bin/env python3
"""Preflight, materialize, and validate PRL13 full-model evaluation snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_full_model_snapshot import (  # noqa: E402
    build_full_model_snapshot_manifest,
    full_model_materialization_preflight,
    full_model_snapshot_identity_record,
    load_full_model_evaluation_snapshot,
    materialize_full_model_snapshot,
    write_full_model_materialization_receipt,
    write_full_model_snapshot_manifest,
)
from tgvf_rl.policy.deepeyes_native_contract import (  # noqa: E402
    load_deepeyes_native_run_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a PRL13 base/full FSDP checkpoint and expose standalone HF "
            "weights without constructing a LoRA adapter."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_source_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--run-config", type=Path, required=True)
        command.add_argument("--optimizer-step", type=int, required=True)
        command.add_argument(
            "--source",
            type=Path,
            required=True,
            help="Step 0: base HF directory. Step N: global_step_N directory.",
        )
        command.add_argument(
            "--allow-template-run",
            action="store_true",
            help="Test-only: permit launch_enabled=false/placeholder run contracts.",
        )
        command.add_argument(
            "--runtime-fsdp-world-size",
            type=int,
            help=(
                "Explicitly bind the actual launcher/checkpoint FSDP world size "
                "when it differs from the formal run TOML."
            ),
        )

    preflight = subparsers.add_parser("preflight")
    add_source_arguments(preflight)
    preflight.add_argument("--target-dir", type=Path)

    materialize = subparsers.add_parser("materialize")
    add_source_arguments(materialize)
    materialize.add_argument("--target-dir", type=Path)
    materialize.add_argument("--snapshot-manifest", type=Path, required=True)
    materialize.add_argument("--receipt", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--snapshot-manifest", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--allow-template-run", action="store_true")
    return parser


def _manifest(args: argparse.Namespace):
    contract = load_deepeyes_native_run_contract(
        args.run_config, allow_template=args.allow_template_run
    )
    return build_full_model_snapshot_manifest(
        contract,
        source_path=args.source,
        optimizer_step=args.optimizer_step,
        runtime_fsdp_world_size=args.runtime_fsdp_world_size,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "preflight":
        manifest = _manifest(args)
        record = {
            "snapshot": manifest.as_record(),
            "preflight": full_model_materialization_preflight(
                manifest, target_dir=args.target_dir
            ),
        }
    elif args.mode == "materialize":
        manifest = _manifest(args)
        preflight = full_model_materialization_preflight(
            manifest, target_dir=args.target_dir
        )
        receipt = materialize_full_model_snapshot(manifest, target_dir=args.target_dir)
        write_full_model_snapshot_manifest(args.snapshot_manifest, manifest)
        write_full_model_materialization_receipt(args.receipt, receipt)
        record = {
            "snapshot": manifest.as_record(),
            "preflight": preflight,
            "materialization": receipt.as_record(),
        }
    elif args.mode == "validate":
        snapshot = load_full_model_evaluation_snapshot(
            args.snapshot_manifest,
            args.receipt,
            require_launchable_run=not args.allow_template_run,
        )
        record = {
            "valid": True,
            "snapshot": full_model_snapshot_identity_record(snapshot),
            "model_path": str(snapshot.model_path),
        }
    else:  # pragma: no cover - argparse exhaustiveness guard
        raise RuntimeError(f"unsupported mode {args.mode}")
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
