from __future__ import annotations

from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tgvf_rl import cli
from tgvf_rl.ops.cli_authorization import (
    CLIExecutionAuthorizationIdentity,
    CLIWorkerAuthorization,
    CanonicalConfigBinding,
    PythonExecutableIdentity,
)
from tgvf_rl.ops.launch_gate import LaunchAuthorizationError


AUTHORIZATION_ARGUMENTS = [
    "--gate-directory",
    "/gate",
    "--authorization-token",
    "/gate/token.json",
    "--freeze-override",
    "/gate/override.json",
]
POLICY_COMPILE_ARGUMENTS = [
    "--compile-prerequisite-manifest",
    "/compile-prerequisites.json",
]


def _binding(path: str, digest: str = "b" * 64) -> CanonicalConfigBinding:
    source = Path(path)
    return CanonicalConfigBinding(
        canonical_root=source.parent,
        source_path=source,
        resolved_path=source,
        source_sha256=digest,
        byte_length=10,
        device=1,
        inode=2,
        mode=stat.S_IFREG | 0o644,
    )


def _python_identity(path: str = "/audited/python") -> PythonExecutableIdentity:
    return PythonExecutableIdentity(
        declared_path=Path(path),
        resolved_path=Path(path),
        sha256="d" * 64,
        byte_length=100,
        device=3,
        inode=4,
        mode=stat.S_IFREG | 0o755,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["launch-representation", "/run.toml"],
        ["run-representation", "/run.toml"],
        ["run-representation-internal-evaluation", "/eval.toml"],
        ["run-policy", "/policy.toml"],
    ],
)
def test_execution_commands_require_explicit_outer_gate_or_worker_proof(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(arguments)


@pytest.mark.parametrize(
    ("command", "config_loader"),
    [
        ("launch-representation", "load_representation_training_config"),
        (
            "run-representation-internal-evaluation",
            "_load_representation_internal_evaluation_config",
        ),
        ("run-policy", "_load_policy_run_config"),
    ],
)
def test_legacy_config_is_blocked_before_load_or_token_consumption(
    command: str,
    config_loader: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli, "assert_canonical_runtime_launch_enabled", lambda: None)
    monkeypatch.setattr(cli, config_loader, lambda _path: events.append("load"))
    monkeypatch.setattr(
        cli,
        "_consume_command_authorization",
        lambda *_args, **_kwargs: events.append("consume"),
    )

    compile_arguments = POLICY_COMPILE_ARGUMENTS if command == "run-policy" else []
    result = cli.main(
        [
            command,
            f"/legacy/{command}.toml",
            *compile_arguments,
            *AUTHORIZATION_ARGUMENTS,
        ]
    )

    assert result == 2
    assert events == []
    assert "outside canonical root" in capsys.readouterr().err


def test_policy_compile_refusal_precedes_token_consumption(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli, "assert_canonical_runtime_launch_enabled", lambda: None)
    binding = _binding("/canonical/policy.toml")
    config = SimpleNamespace(
        run_id="POLICY-RUN",
        identity_sha256="a" * 64,
        source_sha256="b" * 64,
        source_path=binding.source_path,
    )
    monkeypatch.setattr(cli, "bind_canonical_config_path", lambda *_a, **_k: binding)
    monkeypatch.setattr(cli, "_load_policy_run_config", lambda _path: config)
    monkeypatch.setattr(
        cli, "assert_loaded_config_matches_binding", lambda *_a, **_k: None
    )

    def refuse(*_args: object, **_kwargs: object) -> object:
        events.append("compile-preflight")
        raise RuntimeError("system-toolchain remain unbound")

    monkeypatch.setattr(cli, "_preflight_policy_run", refuse)
    monkeypatch.setattr(
        cli,
        "_consume_command_authorization",
        lambda *_args, **_kwargs: events.append("consume"),
    )

    result = cli.main(
        [
            "run-policy",
            str(binding.source_path),
            *POLICY_COMPILE_ARGUMENTS,
            *AUTHORIZATION_ARGUMENTS,
        ]
    )

    assert result == 2
    assert events == ["compile-preflight"]
    assert "system-toolchain remain unbound" in capsys.readouterr().err


def test_runtime_closure_blocks_unsafe_evaluation_before_artifact_or_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    binding = _binding("/canonical/evaluation.toml")
    config = SimpleNamespace(source_path=binding.source_path, source_sha256="b" * 64)
    monkeypatch.setattr(cli, "bind_canonical_config_path", lambda *_a, **_k: binding)
    monkeypatch.setattr(
        cli,
        "_load_representation_internal_evaluation_config",
        lambda _path: events.append("load-safe-config") or config,
    )
    monkeypatch.setattr(
        cli, "assert_loaded_config_matches_binding", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        cli,
        "_run_representation_internal_evaluation",
        lambda *_a, **_k: events.append("unsafe-artifact-load"),
    )
    monkeypatch.setattr(
        cli,
        "_consume_command_authorization",
        lambda *_a, **_k: events.append("consume-token"),
    )

    result = cli.main(
        [
            "run-representation-internal-evaluation",
            str(binding.source_path),
            *AUTHORIZATION_ARGUMENTS,
        ]
    )

    assert result == 2
    assert events == []
    assert "runtime closure is incomplete" in capsys.readouterr().err


def test_policy_uses_one_prepared_identity_after_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    monkeypatch.setattr(cli, "assert_canonical_runtime_launch_enabled", lambda: None)
    binding = _binding("/canonical/policy.toml")
    config = SimpleNamespace(
        run_id="POLICY-RUN",
        identity_sha256="a" * 64,
        source_sha256="b" * 64,
        source_path=binding.source_path,
    )
    prepared = SimpleNamespace(
        config=config,
        horizon_extension=None,
        authorization_parameters=lambda: {"prepared_policy_launch_sha256": "e" * 64},
        close_python_binding=lambda: None,
    )
    worker = CLIWorkerAuthorization(
        consumption_receipt_path=Path("/gate/consumptions/token.json"),
        consumption_receipt_sha256="c" * 64,
        launcher_liveness_receipt_path=Path("/gate/cli-launches/token/live.json"),
    )
    monkeypatch.setattr(cli, "bind_canonical_config_path", lambda *_a, **_k: binding)
    monkeypatch.setattr(cli, "_load_policy_run_config", lambda _path: config)
    monkeypatch.setattr(
        cli, "assert_loaded_config_matches_binding", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        cli,
        "_preflight_policy_run",
        lambda *_a, **_k: events.append("preflight") or prepared,
    )
    monkeypatch.setattr(
        cli, "_assert_installed_stack_identity", lambda *_a: events.append("stack")
    )
    monkeypatch.setattr(
        cli, "verify_verl_distribution_identity", lambda **_k: events.append("verl")
    )
    monkeypatch.setattr(
        cli,
        "_consume_command_authorization",
        lambda _a, identity: events.append(("consume", identity)) or {},
    )
    monkeypatch.setattr(
        cli,
        "materialize_cli_worker_authorization",
        lambda *_a, **_k: events.append("materialize") or worker,
    )
    monkeypatch.setattr(
        cli,
        "verify_canonical_config_binding",
        lambda observed: events.append(("reverify", observed)),
    )
    monkeypatch.setattr(
        cli,
        "_execute_policy_run",
        lambda observed, **kwargs: events.append(("execute", observed, kwargs)),
    )

    assert (
        cli.main(
            [
                "run-policy",
                str(binding.source_path),
                *POLICY_COMPILE_ARGUMENTS,
                *AUTHORIZATION_ARGUMENTS,
            ]
        )
        == 0
    )

    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "preflight",
        "stack",
        "verl",
        "consume",
        "materialize",
        "reverify",
        "execute",
    ]
    consumed_identity = events[3][1]
    executed = events[-1]
    assert executed[1] is prepared
    assert executed[2]["launch_identity"] is consumed_identity
    assert "compile_prerequisite_manifest_path" not in executed[2]


def test_representation_preflights_then_consumes_then_executes_same_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    monkeypatch.setattr(cli, "assert_canonical_runtime_launch_enabled", lambda: None)
    binding = _binding("/canonical/representation.toml")
    python_identity = _python_identity()
    config = SimpleNamespace(
        run_id="REPRESENTATION-RUN",
        canonical_config_sha256="a" * 64,
        source_toml_sha256="b" * 64,
        source_path=binding.source_path,
        fsdp2=SimpleNamespace(world_size=4),
    )
    prepared = cli.PreparedRepresentationLaunch(
        config=config,
        config_binding=binding,
        python_identity=python_identity,
        stop_after_global_step=32,
        command_prefix=("/audited/python", "-m", "torch.distributed.run"),
        child_environment=(("WORLD_SIZE", "4"),),
        stripped_environment_names=(),
    )
    worker = CLIWorkerAuthorization(
        consumption_receipt_path=Path("/gate/consumptions/token.json"),
        consumption_receipt_sha256="c" * 64,
        launcher_liveness_receipt_path=Path("/gate/cli-launches/token/live.json"),
    )
    monkeypatch.setattr(cli, "bind_canonical_config_path", lambda *_a, **_k: binding)
    monkeypatch.setattr(cli, "load_representation_training_config", lambda _p: config)
    monkeypatch.setattr(
        cli,
        "_preflight_representation_launch",
        lambda *_a, **_k: events.append("preflight") or prepared,
    )
    monkeypatch.setattr(
        cli,
        "_consume_command_authorization",
        lambda _a, identity: events.append(("consume", identity)) or {},
    )
    monkeypatch.setattr(
        cli,
        "materialize_cli_worker_authorization",
        lambda *_a, **_k: events.append("materialize") or worker,
    )
    monkeypatch.setattr(
        cli,
        "_execute_representation_torchrun",
        lambda observed, **kwargs: events.append(("execute", observed, kwargs)),
    )

    assert (
        cli.main(
            [
                "launch-representation",
                str(binding.source_path),
                "--stop-after-global-step",
                "32",
                "--python",
                "/audited/python",
                *AUTHORIZATION_ARGUMENTS,
            ]
        )
        == 0
    )
    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "preflight",
        "consume",
        "materialize",
        "execute",
    ]
    assert events[-1][1] is prepared
    assert events[-1][2]["launch_identity"] is events[1][1]


def test_representation_worker_authorization_is_first_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    binding = _binding("/canonical/representation.toml")
    config = SimpleNamespace(
        source_path=binding.source_path,
        source_toml_sha256="b" * 64,
        fsdp2=SimpleNamespace(world_size=4),
    )
    identity = CLIExecutionAuthorizationIdentity.create(
        run_id="REP",
        phase=cli.REPRESENTATION_TRAINING_PHASE,
        command_id=cli._REPRESENTATION_COMMAND_ID,
        run_identity_sha256="a" * 64,
    )
    monkeypatch.setenv("TGVF_CLI_GATE_DIRECTORY", "/gate")
    monkeypatch.setenv("TGVF_CLI_CONSUMPTION_RECEIPT_PATH", "/gate/receipt.json")
    monkeypatch.setenv("TGVF_CLI_CONSUMPTION_RECEIPT_SHA256", "c" * 64)
    monkeypatch.setenv("TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH", "/gate/live.json")
    monkeypatch.setattr(
        cli,
        "verify_cli_worker_authorization_from_environment",
        lambda **_k: events.append("authorize") or identity,
    )
    monkeypatch.setattr(
        cli,
        "assert_canonical_runtime_launch_enabled",
        lambda: events.append("runtime-closure"),
    )
    monkeypatch.setattr(
        cli,
        "bind_canonical_config_path",
        lambda *_a, **_k: events.append("bind-config") or binding,
    )
    monkeypatch.setattr(
        cli,
        "load_representation_training_config",
        lambda _p: events.append("load-config") or config,
    )
    monkeypatch.setattr(
        cli,
        "assert_loaded_config_matches_binding",
        lambda *_a, **_k: events.append("check-config"),
    )
    monkeypatch.setattr(
        cli,
        "bind_current_python_executable",
        lambda _p: events.append("check-python") or _python_identity(),
    )
    monkeypatch.setattr(
        cli,
        "_assert_worker_identity_parameters",
        lambda *_a, **_k: events.append("check-identity"),
    )
    monkeypatch.setattr(
        cli, "_run_representation_training", lambda *_a, **_k: events.append("runner")
    )

    assert (
        cli.main(
            [
                "run-representation",
                str(binding.source_path),
                "--launcher-python-executable",
                "/audited/python",
                "--gate-directory",
                "/gate",
                "--launch-consumption-receipt",
                "/gate/receipt.json",
                "--launch-consumption-sha256",
                "c" * 64,
                "--launcher-liveness-receipt",
                "/gate/live.json",
            ]
        )
        == 0
    )
    assert events == [
        "authorize",
        "runtime-closure",
        "bind-config",
        "load-config",
        "check-config",
        "check-python",
        "check-identity",
        "runner",
    ]


def test_read_only_commands_keep_legacy_config_access_without_auth_arguments() -> None:
    read_only_arguments = [
        ["compat-info"],
        ["validate-representation-config", "/legacy/representation.toml"],
        ["validate-policy-config", "/legacy/policy.toml"],
        ["plan-policy", "/legacy/policy.toml"],
    ]
    for arguments in read_only_arguments:
        parsed = cli._parser().parse_args(arguments)
        assert not hasattr(parsed, "gate_directory")
        assert not hasattr(parsed, "authorization_token")
        assert not hasattr(parsed, "freeze_override")


def test_freeze_override_argument_is_policy_conditional() -> None:
    parsed = cli._parser().parse_args(
        [
            "launch-representation",
            "/canonical/run.toml",
            "--gate-directory",
            "/gate",
            "--authorization-token",
            "/gate/token.json",
        ]
    )
    assert parsed.freeze_override is None


def test_authorization_error_is_caught_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "assert_canonical_runtime_launch_enabled", lambda: None)
    monkeypatch.setattr(
        cli,
        "bind_canonical_config_path",
        lambda *_a, **_k: (_ for _ in ()).throw(
            LaunchAuthorizationError("synthetic refusal")
        ),
    )
    assert (
        cli.main(["launch-representation", "/run.toml", *AUTHORIZATION_ARGUMENTS]) == 2
    )
    assert "synthetic refusal" in capsys.readouterr().err
