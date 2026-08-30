from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tgvf_rl.framework.verl import policy_main
from tgvf_rl.framework.verl.compile_prerequisites import (
    POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY,
    POLICY_COMPILE_PREREQUISITE_MANIFEST_SCHEMA,
    POLICY_COMPILE_PREREQUISITE_RESIDUAL_BLOCKER,
    load_policy_compile_prerequisite_manifest,
    materialize_policy_compile_prerequisite_receipt,
    preflight_policy_compile_prerequisites,
    verify_policy_compile_prerequisites_from_environment,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_manifest(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "minimum-prerequisites"
    include = root / "include" / "python3.12"
    include.mkdir(parents=True)
    paths = {
        "c_compiler": root / "gcc",
        "cxx_compiler": root / "g++",
        "python_h": include / "Python.h",
        "pyconfig_h": include / "pyconfig.h",
    }
    for name, path in paths.items():
        path.write_bytes(f"fixture:{name}\n".encode())
        path.chmod(0o755 if name in {"c_compiler", "cxx_compiler"} else 0o644)
    manifest = {
        "schema_version": POLICY_COMPILE_PREREQUISITE_MANIFEST_SCHEMA,
        "closure_policy": POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY,
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "byte_length": path.stat().st_size,
                "executable_required": name in {"c_compiler", "cxx_compiler"},
            }
            for name, path in paths.items()
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    return manifest_path, paths


def test_explicit_manifest_preflight_is_content_bound_but_not_closure_complete(
    tmp_path: Path,
) -> None:
    manifest_path, paths = _write_manifest(tmp_path)

    binding = load_policy_compile_prerequisite_manifest(manifest_path)
    receipt = preflight_policy_compile_prerequisites(binding)

    assert binding.manifest_source_path == manifest_path
    assert binding.c_compiler == paths["c_compiler"]
    assert binding.cxx_compiler == paths["cxx_compiler"]
    assert binding.launch_blockers == (POLICY_COMPILE_PREREQUISITE_RESIDUAL_BLOCKER,)
    assert receipt.binding == binding
    assert receipt.as_record()["closure_complete"] is False
    assert len(receipt.receipt_sha256) == 64


def test_manifest_missing_tampered_and_symlink_inputs_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="missing, unreadable, or a symlink"):
        load_policy_compile_prerequisite_manifest(tmp_path / "missing.json")

    manifest_path, paths = _write_manifest(tmp_path)
    binding = load_policy_compile_prerequisite_manifest(manifest_path)
    paths["python_h"].write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="size differs|SHA256 differs"):
        preflight_policy_compile_prerequisites(binding)

    manifest_path, paths = _write_manifest(tmp_path / "second")
    binding = load_policy_compile_prerequisite_manifest(manifest_path)
    target = paths["pyconfig_h"].with_name("real-pyconfig.h")
    paths["pyconfig_h"].rename(target)
    paths["pyconfig_h"].symlink_to(target)
    with pytest.raises(RuntimeError, match="missing, unreadable, or a symlink"):
        preflight_policy_compile_prerequisites(binding)

    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(manifest_path)
    with pytest.raises(RuntimeError, match="missing, unreadable, or a symlink"):
        load_policy_compile_prerequisite_manifest(manifest_link)


def test_manifest_rejects_unknown_fields_and_duplicate_keys(tmp_path: Path) -> None:
    manifest_path, _ = _write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["implicit_default"] = True
    manifest_path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(ValueError, match="fields differ"):
        load_policy_compile_prerequisite_manifest(manifest_path)

    manifest_path.write_text(
        '{"schema_version":"x","schema_version":"y",'
        '"closure_policy":"minimum-declared-prerequisites-v1","files":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate field"):
        load_policy_compile_prerequisite_manifest(manifest_path)


def test_private_receipt_is_revalidated_in_child_environment(tmp_path: Path) -> None:
    manifest_path, paths = _write_manifest(tmp_path)
    binding = load_policy_compile_prerequisite_manifest(manifest_path)
    receipt = preflight_policy_compile_prerequisites(binding)
    receipt_path = materialize_policy_compile_prerequisite_receipt(
        receipt,
        state_directory=(tmp_path / "private-state").resolve(),
    )
    environment = {
        "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_PATH": str(receipt_path),
        "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_SHA256": receipt.receipt_sha256,
        "TGVF_POLICY_COMPILE_PREREQUISITE_BINDING_SHA256": binding.identity_sha256,
        "TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_SHA256": (
            binding.manifest_source_sha256
        ),
    }

    observed = verify_policy_compile_prerequisites_from_environment(
        environment, required=True
    )

    assert observed == receipt
    with pytest.raises(RuntimeError, match="closure is incomplete"):
        verify_policy_compile_prerequisites_from_environment(
            environment,
            required=True,
            require_closure_complete=True,
        )
    assert stat_mode(receipt_path) & 0o077 == 0
    paths["cxx_compiler"].write_bytes(b"changed after launcher preflight\n")
    with pytest.raises(RuntimeError, match="size differs|SHA256 differs"):
        verify_policy_compile_prerequisites_from_environment(environment, required=True)


def test_receipt_environment_is_required_and_cannot_be_partial() -> None:
    with pytest.raises(RuntimeError, match="environment is missing"):
        verify_policy_compile_prerequisites_from_environment({}, required=True)
    assert (
        verify_policy_compile_prerequisites_from_environment({}, required=False) is None
    )
    with pytest.raises(RuntimeError, match="environment is partial"):
        verify_policy_compile_prerequisites_from_environment(
            {"TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_PATH": "/missing"},
            required=False,
        )


def test_policy_main_requires_worker_authorization_before_compile_or_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        policy_main,
        "verify_cli_worker_authorization_from_environment",
        lambda **_kwargs: (
            events.append("worker-authorization")
            or (_ for _ in ()).throw(RuntimeError("synthetic worker refusal"))
        ),
    )
    monkeypatch.setattr(
        policy_main,
        "verify_policy_compile_prerequisites_from_environment",
        lambda *_args, **_kwargs: events.append("compile-receipt"),
    )
    monkeypatch.setattr(
        policy_main,
        "compose_pinned_verl_config",
        lambda *_args, **_kwargs: events.append("compose"),
    )

    with pytest.raises(RuntimeError, match="synthetic worker refusal"):
        policy_main.main(())
    assert events == ["worker-authorization"]


def test_policy_main_requires_complete_compile_closure_before_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        policy_main,
        "verify_cli_worker_authorization_from_environment",
        lambda **_kwargs: (
            events.append("worker-authorization") or SimpleNamespace(parameters=())
        ),
    )
    monkeypatch.setattr(
        policy_main,
        "assert_canonical_runtime_launch_enabled",
        lambda: events.append("runtime-closure"),
    )
    monkeypatch.setattr(
        policy_main,
        "verify_policy_driver_child_environment",
        lambda *_args: events.append("child-environment"),
    )
    monkeypatch.setattr(
        policy_main,
        "_verify_launch_identity_against_current_process",
        lambda _identity: events.append("process-identity"),
    )

    def refuse(*_args: object, **kwargs: object) -> None:
        events.append("compile-receipt")
        assert kwargs["required"] is True
        assert kwargs["require_closure_complete"] is True
        raise RuntimeError("compile closure is incomplete")

    monkeypatch.setattr(
        policy_main,
        "verify_policy_compile_prerequisites_from_environment",
        refuse,
    )
    monkeypatch.setattr(
        policy_main,
        "compose_pinned_verl_config",
        lambda *_args, **_kwargs: events.append("compose"),
    )

    with pytest.raises(RuntimeError, match="closure is incomplete"):
        policy_main.main(())
    assert events == [
        "worker-authorization",
        "runtime-closure",
        "child-environment",
        "process-identity",
        "compile-receipt",
    ]


def test_policy_main_refuses_child_environment_before_process_or_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        policy_main,
        "verify_cli_worker_authorization_from_environment",
        lambda **_kwargs: (
            events.append("worker-authorization") or SimpleNamespace(parameters=())
        ),
    )
    monkeypatch.setattr(
        policy_main,
        "assert_canonical_runtime_launch_enabled",
        lambda: events.append("runtime-closure"),
    )
    monkeypatch.setattr(
        policy_main,
        "verify_policy_driver_child_environment",
        lambda *_args: (
            events.append("child-environment")
            or (_ for _ in ()).throw(RuntimeError("synthetic child-env refusal"))
        ),
    )
    monkeypatch.setattr(
        policy_main,
        "_verify_launch_identity_against_current_process",
        lambda _identity: events.append("process-identity"),
    )
    monkeypatch.setattr(
        policy_main,
        "verify_policy_compile_prerequisites_from_environment",
        lambda *_args, **_kwargs: events.append("compile-receipt"),
    )

    with pytest.raises(RuntimeError, match="synthetic child-env refusal"):
        policy_main.main(())
    assert events == [
        "worker-authorization",
        "runtime-closure",
        "child-environment",
    ]


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode
