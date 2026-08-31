from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tgvf_rl import cli
from tgvf_rl.framework.verl import policy_dev_main, policy_main
from tgvf_rl.framework.verl.launcher import build_policy_e2e_smoke_verl_plan
from tgvf_rl.ops.policy_compile_prerequisites import (
    POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER,
)
from tgvf_rl.policy import dev_launch
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config
from tests.policy.test_run_config import (
    _with_generic_gpu_topology,
    _write_config,
)


_STRICT_POLICY_ARGUMENTS = [
    "run-policy",
    "/canonical/policy.toml",
    "--compile-prerequisite-manifest",
    "/compile.json",
    "--runtime-locator-manifest",
    "/runtime.json",
    "--runtime-locator-manifest-sha256",
    "d" * 64,
    "--runtime-locator-manifest-byte-length",
    "1",
    "--gate-directory",
    "/gate",
    "--authorization-token",
    "/gate/token.json",
]


def _write_two_gpu_policy_config(tmp_path: Path) -> Path:
    path, text, _ = _write_config(tmp_path)
    path.write_text(
        _with_generic_gpu_topology(
            text,
            physical_gpu_ids=(4, 6),
            tensor_parallel_size=2,
        ),
        encoding="utf-8",
    )
    return path


def test_dev_parser_is_minimal_and_strict_parser_contract_is_unchanged() -> None:
    parser = cli._parser()  # noqa: SLF001
    dev = parser.parse_args(["dev-run-policy", "/work/policy.toml"])
    strict = parser.parse_args(_STRICT_POLICY_ARGUMENTS)

    assert set(vars(dev)) == {"command", "path", "python"}
    assert dev.command == "dev-run-policy"
    assert dev.path == Path("/work/policy.toml")
    assert not hasattr(dev, "gate_directory")
    assert not hasattr(dev, "authorization_token")
    assert not hasattr(dev, "freeze_override")
    assert not hasattr(dev, "compile_prerequisite_manifest")
    assert not hasattr(dev, "runtime_locator_manifest")

    assert strict.command == "run-policy"
    assert strict.path == Path("/canonical/policy.toml")
    assert strict.compile_prerequisite_manifest == Path("/compile.json")
    assert strict.runtime_locator_manifest == Path("/runtime.json")
    assert strict.gate_directory == Path("/gate")
    assert strict.authorization_token == Path("/gate/token.json")
    assert strict.freeze_override is None


def test_dev_cli_exec_uses_same_plan_and_sanitized_topology_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_two_gpu_policy_config(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-cross-exec")
    monkeypatch.setenv("HTTPS_PROXY", "https://secret@example.invalid")
    captured: dict[str, object] = {}

    def fake_execve(
        executable: str,
        argv: tuple[str, ...],
        environment: dict[str, str],
    ) -> None:
        captured.update(
            executable=executable,
            argv=argv,
            environment=environment,
        )

    monkeypatch.setattr(dev_launch.os, "execve", fake_execve)

    assert (
        cli.main(
            [
                "dev-run-policy",
                str(path),
                "--python",
                str(Path(sys.executable).absolute()),
            ]
        )
        == 0
    )

    config = load_policy_e2e_smoke_run_config(path)
    plan = build_policy_e2e_smoke_verl_plan(config)
    argv = captured["argv"]
    environment = captured["environment"]
    assert isinstance(argv, tuple)
    assert isinstance(environment, dict)
    assert captured["executable"] == str(Path(sys.executable).absolute())
    assert argv[:3] == (
        str(Path(sys.executable).absolute()),
        "-m",
        dev_launch.POLICY_DEV_DRIVER_MAIN_MODULE,
    )
    assert argv[3:] == plan.hydra_override_args()
    assert "++trainer.n_gpus_per_node=2" in argv
    assert "++actor_rollout_ref.rollout.n_gpus_per_node=2" in argv
    assert "++actor_rollout_ref.rollout.agent.num_workers=2" in argv
    assert "++actor_rollout_ref.rollout.tensor_model_parallel_size=2" in argv

    assert environment["CUDA_VISIBLE_DEVICES"] == "4,6"
    assert environment["TGVF_POLICY_RUN_CONFIG_PATH"] == str(path)
    assert environment["TGVF_POLICY_RUN_ID"] == config.run_id
    assert environment["TGVF_POLICY_RUN_IDENTITY_SHA256"] == config.identity_sha256
    assert (
        environment[dev_launch.POLICY_EXECUTION_PROFILE_ENVIRONMENT]
        == dev_launch.POLICY_DEV_EXECUTION_PROFILE
    )
    assert "OPENROUTER_API_KEY" not in environment
    assert "HTTPS_PROXY" not in environment
    assert not any(name.startswith("TGVF_CLI_") for name in environment)
    assert not any(
        name.startswith("TGVF_POLICY_COMPILE_PREREQUISITE_")
        for name in environment
    )


def test_policy_dev_main_requires_and_consumes_explicit_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment: dict[str, str] = {}
    events: list[tuple[str, ...]] = []
    monkeypatch.setattr(policy_dev_main, "os", SimpleNamespace(environ=environment))
    monkeypatch.setattr(
        policy_dev_main,
        "compose_and_run_pinned_verl",
        lambda overrides: events.append(tuple(overrides)),
    )

    with pytest.raises(RuntimeError, match="explicit dev launch profile"):
        policy_dev_main.main(("++trainer.n_gpus_per_node=2",))
    assert events == []

    environment[dev_launch.POLICY_EXECUTION_PROFILE_ENVIRONMENT] = (
        dev_launch.POLICY_DEV_EXECUTION_PROFILE
    )
    policy_dev_main.main(("++trainer.n_gpus_per_node=2",))

    assert events == [("++trainer.n_gpus_per_node=2",)]
    assert dev_launch.POLICY_EXECUTION_PROFILE_ENVIRONMENT not in environment


def test_strict_policy_main_never_switches_on_the_dev_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        dev_launch.POLICY_EXECUTION_PROFILE_ENVIRONMENT: (
            dev_launch.POLICY_DEV_EXECUTION_PROFILE
        )
    }
    events: list[str] = []
    monkeypatch.setattr(policy_main, "os", SimpleNamespace(environ=environment))
    monkeypatch.setattr(
        policy_main,
        "verify_cli_worker_authorization_from_environment",
        lambda **_kwargs: (
            events.append("strict-authorization")
            or (_ for _ in ()).throw(RuntimeError("strict authorization required"))
        ),
    )
    monkeypatch.setattr(
        policy_main,
        "compose_and_run_pinned_verl",
        lambda _overrides: events.append("shared-core"),
    )

    with pytest.raises(RuntimeError, match="strict authorization required"):
        policy_main.main(())

    assert events == ["strict-authorization"]
    assert environment[dev_launch.POLICY_EXECUTION_PROFILE_ENVIRONMENT] == (
        dev_launch.POLICY_DEV_EXECUTION_PROFILE
    )


def test_dev_launcher_rejects_any_blocker_beyond_missing_compile_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_two_gpu_policy_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    plan = build_policy_e2e_smoke_verl_plan(config)
    assert plan.launch_blockers == (POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER,)
    broadened = replace(
        plan,
        launch_blockers=(
            POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER,
            "synthetic non-compile blocker",
        ),
    )
    monkeypatch.setattr(
        dev_launch,
        "build_policy_e2e_smoke_verl_plan",
        lambda _config: broadened,
    )

    with pytest.raises(RuntimeError, match="permits only the missing compile"):
        dev_launch.prepare_policy_dev_launch(
            config,
            python_executable=Path(sys.executable).absolute(),
            host_environment={},
        )


def test_dev_cli_rejects_invalid_config_before_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, text, _ = _write_config(tmp_path)
    path.write_text(
        _with_generic_gpu_topology(text, physical_gpu_ids=(4, 4)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dev_launch.os,
        "execve",
        lambda *_args: pytest.fail("invalid config reached exec"),
    )

    assert cli.main(["dev-run-policy", str(path)]) == 2
    assert "physical_gpu_ids must be non-empty and unique" in capsys.readouterr().err


def test_strict_plan_command_still_targets_authorized_policy_main(tmp_path: Path) -> None:
    path = _write_two_gpu_policy_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    plan = build_policy_e2e_smoke_verl_plan(config)

    assert plan.command(
        Path(sys.executable).absolute(), allow_blocked=True
    )[1:3] == ("-m", "tgvf_rl.framework.verl.policy_main")
