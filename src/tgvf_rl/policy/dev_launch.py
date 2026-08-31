"""Explicit low-friction Policy development launch boundary.

This is the normal ``run-policy`` path.  It deliberately omits the historical
canonical-config gate, one-time launch token, compile manifest, and
runtime-locator closure retained by ``strict-run-policy``.  It still loads the
ordinary Policy run config, builds the same veRL launch plan, permits only the
plan's exact missing-compile-manifest blocker, and starts the child from the
existing sanitized Policy environment profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

from tgvf_rl.framework.verl.launcher import (
    UpstreamVerlLaunchPlan,
    build_policy_e2e_smoke_verl_plan,
)
from tgvf_rl.framework.verl.method_matrix_launcher import (
    route_policy_method_matrix_plan,
)
from tgvf_rl.ops.policy_compile_prerequisites import (
    POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER,
)

from .run_config import PolicyE2ESmokeRunConfig
from .launch import policy_child_environment


POLICY_DEV_DRIVER_MAIN_MODULE = "tgvf_rl.framework.verl.policy_dev_main"
POLICY_DEV_EXECUTION_PROFILE = "policy-development-direct-v1"
POLICY_EXECUTION_PROFILE_ENVIRONMENT = "TGVF_POLICY_EXECUTION_PROFILE"


@dataclass(frozen=True, slots=True)
class PreparedPolicyDevLaunch:
    """Config-derived argv and sanitized environment for one dev process."""

    plan: UpstreamVerlLaunchPlan
    command: tuple[str, ...]
    environment: Mapping[str, str]


def prepare_policy_dev_launch(
    config: PolicyE2ESmokeRunConfig,
    *,
    python_executable: str | Path,
    host_environment: Mapping[str, str] | None = None,
) -> PreparedPolicyDevLaunch:
    """Build one executable dev launch without consulting strict authority."""

    if not isinstance(config, PolicyE2ESmokeRunConfig):
        raise TypeError("development launch requires PolicyE2ESmokeRunConfig")
    base_plan = build_policy_e2e_smoke_verl_plan(config)
    plan = route_policy_method_matrix_plan(config, base_plan)
    _assert_dev_plan(plan)
    executable = Path(python_executable).expanduser().absolute()
    strict_command = plan.command(executable, allow_blocked=True)
    if strict_command[1:3] != ("-m", plan.main_module):
        raise RuntimeError("Policy plan command lost its strict driver coordinate")
    command = (
        strict_command[0],
        "-m",
        POLICY_DEV_DRIVER_MAIN_MODULE,
        *strict_command[3:],
    )
    environment = policy_child_environment(plan, base=host_environment)
    if not isinstance(environment, dict):  # pragma: no cover - fixed API mode
        raise RuntimeError("sanitized Policy environment shape differs")
    environment[POLICY_EXECUTION_PROFILE_ENVIRONMENT] = POLICY_DEV_EXECUTION_PROFILE
    return PreparedPolicyDevLaunch(
        plan=plan,
        command=command,
        environment=MappingProxyType(environment),
    )


def execute_policy_dev_launch(prepared: PreparedPolicyDevLaunch) -> NoReturn:
    """Replace the CLI process with its explicitly prepared dev driver."""

    if type(prepared) is not PreparedPolicyDevLaunch:
        raise TypeError("prepared must be exactly PreparedPolicyDevLaunch")
    os.execve(prepared.command[0], prepared.command, dict(prepared.environment))


def _assert_dev_plan(plan: UpstreamVerlLaunchPlan) -> None:
    if not isinstance(plan, UpstreamVerlLaunchPlan):
        raise TypeError("development launch requires UpstreamVerlLaunchPlan")
    expected = (POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER,)
    if plan.compile_prerequisites is not None or plan.launch_blockers != expected:
        raise RuntimeError(
            "development Policy launch permits only the missing compile-manifest "
            f"blocker; observed {plan.launch_blockers!r}"
        )


__all__ = [
    "POLICY_DEV_DRIVER_MAIN_MODULE",
    "POLICY_DEV_EXECUTION_PROFILE",
    "POLICY_EXECUTION_PROFILE_ENVIRONMENT",
    "PreparedPolicyDevLaunch",
    "execute_policy_dev_launch",
    "prepare_policy_dev_launch",
]
