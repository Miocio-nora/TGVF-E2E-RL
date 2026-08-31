#!/usr/bin/env python3
"""Republish unchanged deterministic-v2/judge-v1 results into judge-v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tgvf_rl.data.policy_selection_t1_judge_reuse import (
    reuse_t1_legacy_judge_results,
)


def _progress(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()
    result = reuse_t1_legacy_judge_results(
        args.config,
        judge_config_path=args.judge_config,
        dry_run=args.dry_run,
        workers=args.workers,
        progress_every=args.progress_every,
        progress_callback=_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
