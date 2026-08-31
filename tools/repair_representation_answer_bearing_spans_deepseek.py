#!/usr/bin/env python3
"""Repair failed RP70 evidence spans through DeepSeek token indices."""

import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:
    import tomli

    sys.modules.setdefault("tomllib", tomli)

from tgvf_rl.representation.experiments.answer_bearing_span.deepseek_index_repair import (
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
