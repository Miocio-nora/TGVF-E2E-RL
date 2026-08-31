"""Explicit development-only Hydra boundary for Policy training."""

from __future__ import annotations

import os
import sys
from typing import Sequence

from tgvf_rl.policy.dev_launch import (
    POLICY_DEV_EXECUTION_PROFILE,
    POLICY_EXECUTION_PROFILE_ENVIRONMENT,
)

from .policy_main import compose_and_run_pinned_verl


def main(argv: Sequence[str] | None = None) -> None:
    """Verify the dev launcher marker, then enter the shared veRL core."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if (
        os.environ.get(POLICY_EXECUTION_PROFILE_ENVIRONMENT)
        != POLICY_DEV_EXECUTION_PROFILE
    ):
        raise RuntimeError("policy_dev_main requires the explicit dev launch profile")
    del os.environ[POLICY_EXECUTION_PROFILE_ENVIRONMENT]
    compose_and_run_pinned_verl(arguments)


if __name__ == "__main__":
    main()


__all__ = ["main"]
