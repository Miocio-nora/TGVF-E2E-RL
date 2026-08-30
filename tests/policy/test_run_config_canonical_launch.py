from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import get_type_hints

from tgvf_rl.policy import run_config
from tgvf_rl.policy import run_config_canonical_launch as canonical_launch
from tgvf_rl.policy import run_config_reward
from tests.policy.test_run_config import _write_config


def test_public_loader_remains_owned_by_the_run_config_facade() -> None:
    loader = run_config.load_policy_e2e_smoke_run_config

    assert loader.__module__ == "tgvf_rl.policy.run_config"
    assert tuple(inspect.signature(loader).parameters) == (
        "path",
        "allow_external_agent_loop_config",
        "allow_historical_reward_contract",
        "allow_historical_read_only_contract",
    )
    assert get_type_hints(loader)


def test_facade_injects_current_loader_globals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path, _, _ = _write_config(tmp_path)
    original = canonical_launch.bind_canonical_policy_launch
    observed: list[dict[str, object]] = []

    def unused_openai_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bounded MCQ config must not load an external judge")

    def unused_utility_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bounded MCQ config must not load tool utility")

    def unused_visual_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bounded MCQ config must not load a visual judge")

    monkeypatch.setattr(
        run_config,
        "load_openai_compatible_judge",
        unused_openai_loader,
    )
    monkeypatch.setattr(
        run_config,
        "load_tgvf_tool_utility_runtime_binding",
        unused_utility_loader,
    )
    monkeypatch.setattr(
        run_config,
        "load_tgvf_visual_quality_judge",
        unused_visual_loader,
    )

    def recording_bind(*args: object, **kwargs: object):
        observed.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(run_config, "bind_canonical_policy_launch", recording_bind)

    run_config.load_policy_e2e_smoke_run_config(config_path)

    assert len(observed) == 1
    assert observed[0]["load_openai_compatible_judge"] is unused_openai_loader
    assert (
        observed[0]["load_tgvf_tool_utility_runtime_binding"]
        is unused_utility_loader
    )
    assert observed[0]["load_tgvf_visual_quality_judge"] is unused_visual_loader


def test_canonical_leaves_do_not_import_facade_tomllib_or_judges() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    policy_directory = repository_root / "src" / "tgvf_rl" / "policy"
    script = textwrap.dedent(
        """
        import importlib
        import sys
        import types

        import tgvf_rl

        package = types.ModuleType("tgvf_rl.policy")
        package.__package__ = "tgvf_rl.policy"
        package.__path__ = [sys.argv[1]]
        sys.modules["tgvf_rl.policy"] = package
        canonical = importlib.import_module(
            "tgvf_rl.policy.run_config_canonical_launch"
        )
        reward = importlib.import_module("tgvf_rl.policy.run_config_reward")

        assert canonical._CanonicalLaunchBindings.__module__ == (
            "tgvf_rl.policy.run_config_canonical_launch"
        )
        assert reward.bind_policy_reward.__module__ == (
            "tgvf_rl.policy.run_config_reward"
        )
        assert "tgvf_rl.policy.run_config" not in sys.modules
        assert "tomllib" not in sys.modules
        assert not any(name.startswith("tgvf_rl.judges") for name in sys.modules)
        """
    )
    environment = os.environ.copy()
    source_root = str(repository_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )

    subprocess.run(
        [sys.executable, "-c", script, str(policy_directory)],
        check=True,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_canonical_binding_type_hints_resolve() -> None:
    assert get_type_hints(canonical_launch._CanonicalLaunchBindings)  # noqa: SLF001
    assert get_type_hints(canonical_launch.bind_canonical_policy_launch)
    assert get_type_hints(run_config_reward.bind_policy_reward)
