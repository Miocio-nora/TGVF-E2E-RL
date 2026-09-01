"""Executable Hydra boundary for the repo-owned pinned-veRL v0 TaskRunner."""

from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path
import sys
from typing import Sequence

from tgvf_rl.ops.child_environment import (
    scrub_policy_driver_authorization_environment,
    verify_policy_driver_child_environment,
)
from tgvf_rl.ops.cli_authorization import (
    CLIExecutionAuthorizationIdentity,
    assert_canonical_runtime_launch_enabled,
    bind_current_python_executable,
    verify_cli_worker_authorization_from_environment,
)
from tgvf_rl.ops.policy_compile_prerequisites import (
    POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY,
    verify_policy_compile_prerequisites_from_environment,
)


_POLICY_PHASE = "policy_training"
_POLICY_COMMAND_ID = "tgvf-rl:run-policy:v4"


def compose_pinned_verl_config(overrides: Sequence[str]) -> object:
    """Compose e003's own config tree without importing its upstream main."""

    spec = find_spec("verl.trainer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("pinned veRL trainer package is not importable")
    trainer_directory = Path(next(iter(spec.submodule_search_locations))).resolve()
    config_directory = trainer_directory / "config"
    if not config_directory.is_dir():
        raise RuntimeError("pinned veRL trainer config directory is missing")
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        return compose(config_name="ppo_trainer", overrides=list(overrides))


def main(argv: Sequence[str] | None = None) -> None:
    """Run upstream orchestration with the project lifecycle TaskRunner class."""

    # This must be the first runtime action.  A compile receipt by itself is not
    # launch authority: require the consumed one-time CLI authorization and its
    # live parent before touching Hydra, veRL, Ray, or lazy compilation.
    launch_identity = verify_cli_worker_authorization_from_environment(
        expected_phase=_POLICY_PHASE,
        expected_command_id=_POLICY_COMMAND_ID,
    )
    assert_canonical_runtime_launch_enabled()
    verify_policy_driver_child_environment(
        os.environ,
        dict(launch_identity.parameters),
    )
    _verify_launch_identity_against_current_process(launch_identity)
    verify_policy_compile_prerequisites_from_environment(
        required=True,
        require_closure_complete=True,
    )
    scrub_policy_driver_authorization_environment(os.environ)
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    compose_and_run_pinned_verl(arguments)


def compose_and_run_pinned_verl(overrides: Sequence[str]) -> None:
    """Compose and execute the shared pinned-veRL Policy orchestration core."""

    config = compose_pinned_verl_config(overrides)
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    from .policy_task_runner import (
        create_policy_pilot_task_runner_class,
        policy_reference_replay_mode,
    )

    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=(
            need_reference_policy(config)
            or policy_reference_replay_mode(config) == "full_diagnostic"
        ),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=create_policy_pilot_task_runner_class())


def _verify_launch_identity_against_current_process(
    identity: CLIExecutionAuthorizationIdentity,
) -> None:
    parameters = dict(identity.parameters)
    python_identity = bind_current_python_executable(sys.executable)
    for name, expected in python_identity.authorization_parameters().items():
        if parameters.get(name) != expected:
            raise RuntimeError(
                f"Policy worker Python identity differs from launcher parameter {name}"
            )
    compile_environment_parameters = {
        "compile_prerequisite_receipt_sha256": (
            "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_SHA256"
        ),
        "compile_prerequisite_binding_sha256": (
            "TGVF_POLICY_COMPILE_PREREQUISITE_BINDING_SHA256"
        ),
        "compile_prerequisite_manifest_sha256": (
            "TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_SHA256"
        ),
    }
    for parameter_name, environment_name in compile_environment_parameters.items():
        if parameters.get(parameter_name) != os.environ.get(environment_name):
            raise RuntimeError(
                "Policy compile receipt environment differs from consumed launch identity"
            )
    if (
        parameters.get("compile_prerequisite_closure_policy")
        != POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY
    ):
        raise RuntimeError(
            "Policy compile closure policy differs from consumed launch identity"
        )


if __name__ == "__main__":
    main()


__all__ = ["compose_and_run_pinned_verl", "compose_pinned_verl_config", "main"]
