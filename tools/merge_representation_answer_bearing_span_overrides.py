#!/usr/bin/env python3
"""Merge reviewed RP70 annotation overrides into one complete JSONL."""

import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:
    import tomli

    sys.modules.setdefault("tomllib", tomli)

from tgvf_rl.representation.experiments.answer_bearing_span.annotation_overrides import (
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
