from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from tgvf_rl.ops import cli_authorization as implementation
from tgvf_rl.ops.cli_authorization import (
    CLIExecutionAuthorizationIdentity,
    REPOSITORY_EXECUTION_POLICY_PATH,
    assert_legacy_standalone_execution_quarantined,
    assert_legacy_standalone_mode_quarantined,
    bind_canonical_config_path,
    bind_current_python_executable,
    assert_canonical_runtime_launch_enabled,
    consume_cli_execution_authorization,
    materialize_cli_worker_authorization,
    sanitized_child_environment,
    verify_canonical_config_binding,
    verify_cli_worker_authorization,
)
from tgvf_rl.ops.launch_gate import (
    LaunchAuthorizationError,
    issue_freeze_override,
    issue_launch_authorization,
    materialize_ready_receipt,
)


def _identity(
    *, stop_after_global_step: str = "32"
) -> CLIExecutionAuthorizationIdentity:
    return CLIExecutionAuthorizationIdentity.create(
        run_id="REPRESENTATION-RUN-1",
        phase="representation_training",
        command_id="tgvf-rl:launch-representation:v1",
        run_identity_sha256="a" * 64,
        parameters={
            "config_source_sha256": "b" * 64,
            "stop_after_global_step": stop_after_global_step,
        },
    )


def _authorized_gate(
    tmp_path: Path,
    identity: CLIExecutionAuthorizationIdentity,
) -> tuple[Path, Path, Path]:
    evidence = tmp_path / "validated-config.json"
    evidence.write_text('{"status":"validated"}\n', encoding="utf-8")
    gate = tmp_path / "gate"
    materialize_ready_receipt(
        gate,
        run_identity=identity.gate_run_identity,
        evidence_paths={"validated_config": evidence},
    )
    token_path, _ = issue_launch_authorization(
        gate,
        ttl_seconds=300,
        authorized_by="test-operator",
    )
    override_path, _ = issue_freeze_override(
        gate,
        REPOSITORY_EXECUTION_POLICY_PATH,
        reason="hermetic public CLI authorization test",
        ttl_seconds=300,
        authorized_by="test-operator",
    )
    return gate, token_path, override_path


def test_cli_authorization_consumes_token_and_frozen_override_once(
    tmp_path: Path,
) -> None:
    identity = _identity()
    gate, token_path, override_path = _authorized_gate(tmp_path, identity)

    consumption = consume_cli_execution_authorization(
        identity,
        gate_directory=gate,
        authorization_token_path=token_path,
        freeze_override_path=override_path,
    )

    assert consumption["run_id"] == identity.run_id
    assert consumption["phase"] == identity.phase
    assert (
        consumption["run_identity_sha256"]
        == (identity.gate_run_identity["identity_sha256"])
    )
    assert consumption["execution_mode"] == "frozen"
    worker = materialize_cli_worker_authorization(
        identity,
        consumption,
        gate_directory=gate,
    )
    verified = verify_cli_worker_authorization(
        identity,
        gate_directory=gate,
        consumption_receipt_path=worker.consumption_receipt_path,
        expected_consumption_receipt_sha256=worker.consumption_receipt_sha256,
        launcher_liveness_receipt_path=worker.launcher_liveness_receipt_path,
    )
    assert verified["consumer_pid"] == consumption["consumer_pid"]
    with pytest.raises(LaunchAuthorizationError, match="already consumed"):
        consume_cli_execution_authorization(
            identity,
            gate_directory=gate,
            authorization_token_path=token_path,
            freeze_override_path=override_path,
        )


def test_repository_runtime_closure_is_non_overridably_blocked() -> None:
    with pytest.raises(
        LaunchAuthorizationError,
        match="canonical runtime closure is incomplete.*cannot be bypassed",
    ):
        assert_canonical_runtime_launch_enabled()


def test_cli_authorization_rejects_parameter_drift_before_consumption(
    tmp_path: Path,
) -> None:
    gate, token_path, override_path = _authorized_gate(tmp_path, _identity())

    with pytest.raises(LaunchAuthorizationError, match="execution parameters"):
        consume_cli_execution_authorization(
            _identity(stop_after_global_step="80"),
            gate_directory=gate,
            authorization_token_path=token_path,
            freeze_override_path=override_path,
        )

    assert not list((gate / "consumptions").glob("*.json"))
    assert not list((gate / "freeze-override-consumptions").glob("*.json"))


def test_cli_authorization_requires_existing_gate_without_creating_it(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-gate"

    with pytest.raises(LaunchAuthorizationError, match="existing non-symlink"):
        consume_cli_execution_authorization(
            _identity(),
            gate_directory=missing,
            authorization_token_path=tmp_path / "missing-token.json",
            freeze_override_path=tmp_path / "missing-override.json",
        )

    assert not missing.exists()


def test_worker_rejects_receipt_when_consumer_is_not_in_process_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    gate, token_path, override_path = _authorized_gate(tmp_path, identity)
    consumption = consume_cli_execution_authorization(
        identity,
        gate_directory=gate,
        authorization_token_path=token_path,
        freeze_override_path=override_path,
    )
    worker = materialize_cli_worker_authorization(
        identity,
        consumption,
        gate_directory=gate,
    )
    monkeypatch.setattr(implementation.os, "getpid", lambda: 1)

    with pytest.raises(LaunchAuthorizationError, match="not a descendant"):
        verify_cli_worker_authorization(
            identity,
            gate_directory=gate,
            consumption_receipt_path=worker.consumption_receipt_path,
            expected_consumption_receipt_sha256=worker.consumption_receipt_sha256,
            launcher_liveness_receipt_path=worker.launcher_liveness_receipt_path,
        )


def test_legacy_standalone_guard_rejects_current_frozen_policy() -> None:
    with pytest.raises(
        LaunchAuthorizationError,
        match=r"legacy standalone tool tools/legacy_gpu.py is quarantined by frozen",
    ):
        assert_legacy_standalone_execution_quarantined("tools/legacy_gpu.py")


def test_legacy_standalone_guard_remains_closed_when_policy_is_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = json.loads(REPOSITORY_EXECUTION_POLICY_PATH.read_text(encoding="utf-8"))
    policy["execution_mode"] = "open"
    policy_path = tmp_path / "open-policy.json"
    policy_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        implementation,
        "REPOSITORY_EXECUTION_POLICY_PATH",
        policy_path,
    )

    with pytest.raises(
        LaunchAuthorizationError,
        match="remains quarantined because it has no run-bound canonical",
    ):
        assert_legacy_standalone_execution_quarantined("tools/legacy_gpu.py")


def test_legacy_mode_guard_allows_only_enumerated_read_only_mode() -> None:
    assert_legacy_standalone_mode_quarantined(
        "tools/mixed.py",
        selected_mode="status",
        read_only_modes=("status", "validate"),
        blocked_modes=("prepare", "worker"),
    )

    with pytest.raises(LaunchAuthorizationError, match="quarantined by frozen"):
        assert_legacy_standalone_mode_quarantined(
            "tools/mixed.py",
            selected_mode="worker",
            read_only_modes=("status", "validate"),
            blocked_modes=("prepare", "worker"),
        )

    with pytest.raises(LaunchAuthorizationError, match="unclassified mode"):
        assert_legacy_standalone_mode_quarantined(
            "tools/mixed.py",
            selected_mode="future-mode",
            read_only_modes=("status", "validate"),
            blocked_modes=("prepare", "worker"),
        )


def test_canonical_config_binding_rejects_legacy_and_symlink_ancestor(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    config = canonical / "run.toml"
    config.write_text("run_id = 'canonical'\n", encoding="utf-8")

    binding = bind_canonical_config_path(config, canonical_root=canonical)
    assert binding.source_sha256
    verify_canonical_config_binding(binding)

    legacy = tmp_path / "legacy.toml"
    legacy.write_text("run_id = 'legacy'\n", encoding="utf-8")
    with pytest.raises(LaunchAuthorizationError, match="outside canonical root"):
        bind_canonical_config_path(legacy, canonical_root=canonical)

    real_directory = canonical / "real"
    real_directory.mkdir()
    linked_config = real_directory / "linked.toml"
    linked_config.write_text("run_id = 'linked'\n", encoding="utf-8")
    (canonical / "alias").symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(
        LaunchAuthorizationError,
        match="symlink|real path",
    ):
        bind_canonical_config_path(
            canonical / "alias" / "linked.toml",
            canonical_root=canonical,
        )


def test_canonical_config_replacement_is_detected(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    config = canonical / "run.toml"
    config.write_text("version = 1\n", encoding="utf-8")
    binding = bind_canonical_config_path(config, canonical_root=canonical)
    replacement = canonical / "replacement.toml"
    replacement.write_text("version = 2\n", encoding="utf-8")
    os.replace(replacement, config)

    with pytest.raises(LaunchAuthorizationError, match="identity changed"):
        verify_canonical_config_binding(binding)


def test_python_binding_accepts_only_current_declared_interpreter() -> None:
    identity = bind_current_python_executable(sys.executable)
    assert identity.declared_path == Path(sys.executable).expanduser().absolute()
    assert identity.resolved_path.is_file()
    assert identity.mode & 0o111

    alternate = identity.resolved_path
    if alternate == identity.declared_path:
        alternate = identity.declared_path.with_name(
            identity.declared_path.name + "-other"
        )
    with pytest.raises(LaunchAuthorizationError, match="exact current audited"):
        bind_current_python_executable(alternate)


def test_child_environment_strips_credentials_and_python_injection() -> None:
    environment, stripped = sanitized_child_environment(
        {
            "PATH": "/usr/bin",
            "OPENROUTER_API_KEY": "secret",
            "WANDB_API_KEY": "secret",
            "HF_TOKEN": "secret",
            "PYTHONPATH": "/injected",
            "SAFE_VALUE": "kept",
        }
    )

    assert environment == {"PATH": "/usr/bin", "SAFE_VALUE": "kept"}
    assert stripped == (
        "HF_TOKEN",
        "OPENROUTER_API_KEY",
        "PYTHONPATH",
        "WANDB_API_KEY",
    )
