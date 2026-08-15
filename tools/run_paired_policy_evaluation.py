#!/usr/bin/env python3
"""Generic entrypoint for the versioned paired-policy evaluation executor."""

from __future__ import annotations

from pathlib import Path
import runpy


_IMPLEMENTATION = Path(__file__).with_name("run_prl15_paired_evaluation.py")


def main() -> int:
    namespace = runpy.run_path(str(_IMPLEMENTATION))
    return namespace["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
