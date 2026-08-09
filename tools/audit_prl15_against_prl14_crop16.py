#!/usr/bin/env python3
"""Audit PRL15 against the hash-verified PRL14 Crop-16 control; never relaunch it."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgvf_rl.framework.verl.crop_matched_control_launcher import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
