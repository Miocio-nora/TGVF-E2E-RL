#!/usr/bin/env python3
"""Materialize the RP66 matched-run utility schedule without changing T1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.data.deepeyes_official_schedule import DEEPEYES_T1_ROOT  # noqa: E402
from tgvf_rl.data.deepeyes_official_schedule_index import (  # noqa: E402
    DEEPEYES_SCHEDULE_INDEX_PATH,
    load_deepeyes_schedule_index,
)
from tgvf_rl.data.tgvf_tool_utility import (  # noqa: E402
    materialize_indexed_tgvf_tool_utility_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the exact first N matched DeepEyes/RP66 prompt rows for "
            "counterfactual tool-utility labeling."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEEPEYES_T1_ROOT)
    parser.add_argument(
        "--schedule-index", type=Path, default=DEEPEYES_SCHEDULE_INDEX_PATH
    )
    parser.add_argument("--global-prompt-batch-size", type=int, default=16)
    parser.add_argument("--optimizer-steps", type=int, default=8)
    parser.add_argument("--canary-sample-count", type=int, default=16)
    args = parser.parse_args()

    index = load_deepeyes_schedule_index(args.schedule_index)
    sample_count = args.global_prompt_batch_size * args.optimizer_steps
    if sample_count > len(index.train):
        raise ValueError("requested prefix exceeds the verified DeepEyes schedule")
    result = materialize_indexed_tgvf_tool_utility_schedule(
        args.dataset_root,
        args.output_root,
        [sample.sample_id for sample in index.train[:sample_count]],
        source_selection="deepeyes-stratified-prefix-v1",
        source_schedule_identity_sha256=index.schedule_identity_sha256,
        global_prompt_batch_size=args.global_prompt_batch_size,
        optimizer_steps=args.optimizer_steps,
        canary_sample_count=args.canary_sample_count,
    )
    print(json.dumps(result.as_record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
