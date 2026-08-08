"""Thin upstream-e003 main for PRL13 native DeepEyes training."""

from __future__ import annotations

from contextlib import redirect_stdout
from importlib.util import find_spec
import io
from pathlib import Path
import sys
from typing import Sequence


def pinned_verl_config_directory() -> Path:
    """Resolve the installed pinned veRL trainer config tree."""

    spec = find_spec("verl.trainer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("pinned veRL trainer package is not importable")
    directory = Path(next(iter(spec.submodule_search_locations))).resolve() / "config"
    if not directory.is_dir():
        raise RuntimeError("pinned veRL trainer config directory is missing")
    return directory


def compose_pinned_deepeyes_config(
    overrides: Sequence[str], *, config_directory: str | Path | None = None
) -> object:
    """Compose e003's real ``ppo_trainer`` tree without Ray/CUDA startup."""

    directory = (
        pinned_verl_config_directory()
        if config_directory is None
        else Path(config_directory).resolve(strict=True)
    )
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        return compose(config_name="ppo_trainer", overrides=list(overrides))


def preflight_pinned_deepeyes_config(config: object) -> dict[str, object]:
    """Validate config and resolve custom Dataset/Reward classes without I/O."""

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as error:
        raise RuntimeError(
            "PRL13 Qwen-VL rollout requires qwen-vl-utils; install the qwen extra"
        ) from error

    if not callable(process_vision_info):
        raise RuntimeError("qwen-vl-utils does not expose process_vision_info")

    from torch.utils.data import Dataset
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
    from verl.trainer.ppo.reward import resolve_reward_manager_cls
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.config import validate_config
    from verl.experimental.agent_loop import AgentLoopManager
    from verl.utils.import_utils import (
        import_external_libs,
        load_class_from_fqn,
        load_extern_object,
    )

    from .native_deepeyes_manager import PRL13_AGENT_LOOP_MANAGER_FQN
    from .deepeyes_actor_loss import (
        DEEPEYES_OFFICIAL_POLICY_LOSS_MODE,
        DEEPEYES_OFFICIAL_POLICY_LOSS_MODULE,
        compute_deepeyes_official_micro_token_mean_loss,
    )
    from .torch_bert_padding import (
        require_prl13_torch_bert_padding,
    )

    use_reference = need_reference_policy(config)
    use_critic = need_critic(config)
    # Upstream prints a success banner; keep the compose-only CLI JSON clean.
    with redirect_stdout(io.StringIO()):
        validate_config(
            config=config,
            use_reference_policy=use_reference,
            use_critic=use_critic,
        )
    # ``validate_config`` does not instantiate the strict RolloutConfig used
    # later by every GPU worker.  Mirror that runtime conversion here so an
    # unknown project field fails during the CPU-only preflight, before model
    # loading or Ray startup.
    omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    dataset_cls = load_extern_object(
        module_path=config.data.custom_cls.path,
        object_name=config.data.custom_cls.name,
    )
    if not isinstance(dataset_cls, type) or not issubclass(dataset_cls, Dataset):
        raise TypeError("PRL13 custom Dataset does not inherit torch Dataset")
    reward_cls = resolve_reward_manager_cls(config)
    if not isinstance(reward_cls, type):
        raise TypeError("PRL13 reward manager did not resolve to a class")
    if config.actor_rollout_ref.rollout.get("checkpoint_manager_class") is not None:
        raise ValueError("PRL13 must use the upstream checkpoint manager")
    manager_fqn = config.actor_rollout_ref.rollout.agent.get(
        "agent_loop_manager_class"
    )
    if manager_fqn != PRL13_AGENT_LOOP_MANAGER_FQN:
        raise ValueError("PRL13 heterogeneous manager identity differs")
    manager_cls = load_class_from_fqn(manager_fqn, "PRL13 AgentLoopManager")
    if not issubclass(manager_cls, AgentLoopManager):
        raise TypeError("PRL13 manager does not inherit upstream AgentLoopManager")
    external_lib = config.actor_rollout_ref.model.get("external_lib")
    if external_lib != DEEPEYES_OFFICIAL_POLICY_LOSS_MODULE:
        raise ValueError("PRL13 actor-loss external module identity differs")
    loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode")
    if loss_mode != DEEPEYES_OFFICIAL_POLICY_LOSS_MODE:
        raise ValueError("PRL13 actor policy-loss mode identity differs")
    import_external_libs(external_lib)
    if get_policy_loss_fn(loss_mode) is not (
        compute_deepeyes_official_micro_token_mean_loss
    ):
        raise RuntimeError("PRL13 actor-loss registry binding differs")
    padding_backend = require_prl13_torch_bert_padding()
    return {
        "need_reference_policy": use_reference,
        "need_critic": use_critic,
        "dataset_class": f"{dataset_cls.__module__}.{dataset_cls.__name__}",
        "reward_manager_class": f"{reward_cls.__module__}.{reward_cls.__name__}",
        "agent_loop_manager_class": manager_fqn,
        "padding_backend": padding_backend,
        "policy_loss_mode": loss_mode,
    }


def run_prl13_task_runner(
    upstream_run: object, runner: object, config: object
) -> object:
    """Install padding in the CPU TaskRunner, then call upstream ``run``.

    A global Ray worker setup hook is intentionally forbidden here: it imports
    Torch before Ray assigns per-actor CUDA visibility and can make distinct
    FSDP ranks bind the same physical GPU.  Model workers receive the same
    backend later through ``ModelConfig.external_lib`` after resource binding.
    """

    if not callable(upstream_run):
        raise TypeError("PRL13 upstream TaskRunner.run is not callable")
    from .torch_bert_padding import (
        install_prl13_torch_bert_padding,
        require_prl13_torch_bert_padding,
    )

    install_prl13_torch_bert_padding()
    require_prl13_torch_bert_padding()
    return upstream_run(runner, config)


def create_prl13_task_runner_class() -> object:
    """Wrap only upstream TaskRunner's entry point at the Ray actor boundary."""

    import ray
    from verl.trainer.main_ppo_v0 import TaskRunner

    upstream_class = getattr(TaskRunner, "__ray_actor_class__", None)
    if (
        not isinstance(upstream_class, type)
        or upstream_class.__module__ != "verl.trainer.main_ppo_v0"
        or upstream_class.__name__ != "TaskRunner"
    ):
        raise RuntimeError("pinned upstream TaskRunner class identity differs")
    upstream_run = upstream_class.run

    class PRL13TaskRunner(upstream_class):
        def run(self, config: object) -> object:
            return run_prl13_task_runner(upstream_run, self, config)

    PRL13TaskRunner.__name__ = "PRL13TaskRunner"
    PRL13TaskRunner.__qualname__ = "PRL13TaskRunner"
    PRL13TaskRunner.__module__ = __name__
    return ray.remote(PRL13TaskRunner)


def run_pinned_deepeyes_config(config: object) -> None:
    """Launch the standard upstream v0 TaskRunner after full preflight."""

    preflight_pinned_deepeyes_config(config)
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    run_ppo(config, task_runner_class=create_prl13_task_runner_class())


def main(argv: Sequence[str] | None = None) -> None:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    config = compose_pinned_deepeyes_config(arguments)
    run_pinned_deepeyes_config(config)


if __name__ == "__main__":
    main()


__all__ = [
    "compose_pinned_deepeyes_config",
    "create_prl13_task_runner_class",
    "main",
    "pinned_verl_config_directory",
    "preflight_pinned_deepeyes_config",
    "run_pinned_deepeyes_config",
    "run_prl13_task_runner",
]
