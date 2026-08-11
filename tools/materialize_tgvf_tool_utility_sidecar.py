#!/usr/bin/env python3
"""Export the TGVF-80 prefix or aggregate real counterfactual attempts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgvf_rl.data.tgvf_tool_utility import (
    materialize_tgvf_tool_utility_schedule,
    materialize_tgvf_tool_utility_sidecar,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    schedule = commands.add_parser(
        "schedule", help="export the exact fresh-run sequential training prefix"
    )
    schedule.add_argument("--dataset-root", type=Path, required=True)
    schedule.add_argument("--output-root", type=Path, required=True)
    schedule.add_argument("--global-prompt-batch-size", type=int, default=16)
    schedule.add_argument("--optimizer-steps", type=int, default=80)
    schedule.add_argument("--canary-sample-count", type=int, default=128)

    aggregate = commands.add_parser(
        "aggregate", help="assign labels from complete forced-TGVF attempts"
    )
    aggregate.add_argument("--schedule-root", type=Path, required=True)
    aggregate.add_argument("--attempts", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--run-identity-sha256", required=True)
    aggregate.add_argument("--sample-count", type=int)
    aggregate.add_argument("--attempts-per-sample", type=int, default=8)
    aggregate.add_argument("--needed-threshold", type=float, default=0.25)
    aggregate.add_argument("--unnecessary-threshold", type=float, default=-0.25)
    aggregate.add_argument("--confidence", type=float, default=0.5)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "schedule":
        result = materialize_tgvf_tool_utility_schedule(
            args.dataset_root,
            args.output_root,
            global_prompt_batch_size=args.global_prompt_batch_size,
            optimizer_steps=args.optimizer_steps,
            canary_sample_count=args.canary_sample_count,
        )
    else:
        result = materialize_tgvf_tool_utility_sidecar(
            args.schedule_root,
            args.attempts,
            args.output_root,
            run_id=args.run_id,
            run_identity_sha256=args.run_identity_sha256,
            sample_count=args.sample_count,
            attempts_per_sample=args.attempts_per_sample,
            needed_threshold=args.needed_threshold,
            unnecessary_threshold=args.unnecessary_threshold,
            confidence=args.confidence,
        )
    print(json.dumps(result.as_record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
