#!/usr/bin/env python3
"""Compose or launch the PRL15 world-4/micro-1 native-Crop control."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgvf_rl.framework.verl.crop_matched_control_launcher import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
