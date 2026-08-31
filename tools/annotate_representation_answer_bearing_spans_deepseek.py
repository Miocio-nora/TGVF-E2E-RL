#!/usr/bin/env python3
"""Annotate RP70 evidence spans with DeepSeek V4 Flash."""

import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # Python 3.10 runtime compatibility
    import tomli

    sys.modules.setdefault("tomllib", tomli)

from tgvf_rl.representation.experiments.answer_bearing_span.deepseek_annotator import (
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
