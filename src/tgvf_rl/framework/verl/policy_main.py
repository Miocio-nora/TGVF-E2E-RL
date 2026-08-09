"""Executable Hydra boundary for the repo-owned pinned-veRL v0 TaskRunner."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys
from typing import Sequence


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

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    config = compose_pinned_verl_config(arguments)
    run_pinned_verl_config(config)


def run_pinned_verl_config(config: object) -> None:
    """Run one already-composed config without a second Hydra composition."""

    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    from .policy_task_runner import (
        POLICY_REFERENCE_DIAGNOSTIC_ENABLED,
        create_policy_pilot_task_runner_class,
    )

    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=POLICY_REFERENCE_DIAGNOSTIC_ENABLED,
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=create_policy_pilot_task_runner_class())


if __name__ == "__main__":
    main()


__all__ = ["compose_pinned_verl_config", "main", "run_pinned_verl_config"]
