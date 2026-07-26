#!/usr/bin/env python3
"""Materialize the accepted 271,842-candidate Qwen3-Instruct T1 population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgvf_rl.data.policy_selection_recommended import (
    materialize_t1_recommended_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vstar", type=Path, required=True)
    parser.add_argument("--arxivqa", type=Path, required=True)
    parser.add_argument("--thinklite", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_t1_recommended_selection(
        [args.vstar, args.arxivqa, args.thinklite],
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
