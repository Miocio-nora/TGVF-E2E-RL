from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import json
import pickle

import pytest

from tgvf_rl.ops import worker_startup as worker_startup_module
from tgvf_rl.ops.worker_startup import (
    POLICY_DRIVER_ROLE,
    REPRESENTATION_LAUNCHER_ROLE,
    REPRESENTATION_MEMBER_ROLE,
    SUPPORTED_WORKER_STARTUP_ROLES,
    WORKER_STARTUP_SCHEMA,
    VerifiedWorkerStartup,
    WorkerStartupIdentity,
)


def _identity(*, role: str = POLICY_DRIVER_ROLE) -> WorkerStartupIdentity:
    return WorkerStartupIdentity(
        role=role,
        command=(
            "/runtime/python",
            "-B",
            "-P",
            "-S",
            "-m",
            "tgvf_rl.worker_bootstrap",
            "run-policy",
        ),
        target="tgvf_rl.framework.verl.policy_main:main",
        runtime_package_sha256="a" * 64,
        dependency_roots_sha256="b" * 64,
    )


def _mint(
    identity: WorkerStartupIdentity,
    *,
    required_role: str,
) -> VerifiedWorkerStartup:
    return worker_startup_module._mint_verified_worker_startup_for_bootstrap(
        identity,
        required_role=required_role,
    )


def test_roles_and_authorization_parameters_are_exact_and_deterministic() -> None:
    identity = _identity()

    assert SUPPORTED_WORKER_STARTUP_ROLES == (
        "policy-driver",
        "representation-launcher",
        "representation-member",
    )
    assert identity.authorization_parameters() == {
        "worker_startup_schema": WORKER_STARTUP_SCHEMA,
        "worker_startup_role": POLICY_DRIVER_ROLE,
        "worker_startup_command_json": (
            '["/runtime/python","-B","-P","-S","-m",'
            '"tgvf_rl.worker_bootstrap","run-policy"]'
        ),
        "worker_startup_command_sha256": (
            "e92c6887a9b345f2da16a49b928d56a131832feb9208fcf5fc8ca9e7f90dcb76"
        ),
        "worker_startup_target": "tgvf_rl.framework.verl.policy_main:main",
        "worker_startup_runtime_package_sha256": "a" * 64,
        "worker_startup_dependency_roots_sha256": "b" * 64,
        "worker_startup_identity_sha256": (
            "7daa782eaba0b206a15c95cbc9cdcc6cb672bb610fe754ae874d9fab7a86dcee"
        ),
    }
    json.dumps(identity.authorization_parameters(), sort_keys=True)
    assert (
        identity.identity_sha256
        == identity.authorization_parameters()["worker_startup_identity_sha256"]
    )


@pytest.mark.parametrize(
    "role",
    [
        "policy",
        "policy-driver ",
        "POLICY-DRIVER",
        "representation",
        "representation-member-child",
        "",
        None,
        1,
    ],
)
def test_identity_rejects_every_non_exact_role(role: object) -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        _identity(role=role)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "command",
    [
        (),
        ["python", "-m", "bootstrap"],
        ("python", 1),
        ("python", ""),
        ("python", "bad\x00argument"),
        ("python", "bad\rargument"),
        ("python", "bad\nargument"),
    ],
)
def test_identity_rejects_non_exact_command(command: object) -> None:
    values: dict[str, object] = {
        "role": POLICY_DRIVER_ROLE,
        "command": command,
        "target": "tgvf_rl.framework.verl.policy_main:main",
        "runtime_package_sha256": "a" * 64,
        "dependency_roots_sha256": "b" * 64,
    }

    with pytest.raises(ValueError, match="command"):
        WorkerStartupIdentity(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "target",
    [
        "",
        "/absolute/target",
        "../relative",
        "module:main ",
        "module:main\x00",
        None,
    ],
)
def test_identity_rejects_ambiguous_target(target: object) -> None:
    with pytest.raises(ValueError, match="target"):
        WorkerStartupIdentity(
            role=POLICY_DRIVER_ROLE,
            command=("python", "-m", "tgvf_rl.worker_bootstrap"),
            target=target,  # type: ignore[arg-type]
            runtime_package_sha256="a" * 64,
            dependency_roots_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("runtime_package_sha256", "a" * 63),
        ("runtime_package_sha256", "A" * 64),
        ("runtime_package_sha256", "g" * 64),
        ("runtime_package_sha256", None),
        ("dependency_roots_sha256", "b" * 65),
        ("dependency_roots_sha256", "B" * 64),
        ("dependency_roots_sha256", 1),
    ],
)
def test_identity_rejects_noncanonical_digests(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "role": POLICY_DRIVER_ROLE,
        "command": ("python", "-m", "tgvf_rl.worker_bootstrap"),
        "target": "tgvf_rl.framework.verl.policy_main:main",
        "runtime_package_sha256": "a" * 64,
        "dependency_roots_sha256": "b" * 64,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        WorkerStartupIdentity(**values)  # type: ignore[arg-type]


def test_identity_is_frozen_but_serializable_authorization_data() -> None:
    identity = _identity()

    with pytest.raises(FrozenInstanceError):
        identity.role = REPRESENTATION_MEMBER_ROLE  # type: ignore[misc]
    assert pickle.loads(pickle.dumps(identity)) == identity


def test_command_order_and_content_are_bound_into_authorization() -> None:
    original = _identity()
    reordered = WorkerStartupIdentity(
        role=original.role,
        command=(original.command[0], "-P", "-B", *original.command[3:]),
        target=original.target,
        runtime_package_sha256=original.runtime_package_sha256,
        dependency_roots_sha256=original.dependency_roots_sha256,
    )
    changed = WorkerStartupIdentity(
        role=original.role,
        command=(*original.command[:-1], "run-representation"),
        target=original.target,
        runtime_package_sha256=original.runtime_package_sha256,
        dependency_roots_sha256=original.dependency_roots_sha256,
    )

    parameters = original.authorization_parameters()
    assert json.loads(parameters["worker_startup_command_json"]) == list(
        original.command
    )
    assert len(parameters["worker_startup_command_sha256"]) == 64
    for different in (reordered, changed):
        different_parameters = different.authorization_parameters()
        assert (
            different_parameters["worker_startup_command_sha256"]
            != parameters["worker_startup_command_sha256"]
        )
        assert different.identity_sha256 != original.identity_sha256


def test_verified_startup_direct_construction_always_fails_closed() -> None:
    identity = _identity()

    with pytest.raises(TypeError, match="only be minted by the worker bootstrap"):
        VerifiedWorkerStartup(identity, required_role=POLICY_DRIVER_ROLE)
    with pytest.raises(TypeError):
        VerifiedWorkerStartup(
            identity,
            required_role=POLICY_DRIVER_ROLE,
            _sentinel=object(),
        )
    with pytest.raises(TypeError, match="mint sentinel differs"):
        VerifiedWorkerStartup._mint_for_bootstrap(
            identity,
            required_role=POLICY_DRIVER_ROLE,
            _sentinel=object(),
        )
    assert "_mint_verified_worker_startup_for_bootstrap" not in (
        worker_startup_module.__all__
    )
    assert "_VERIFIED_WORKER_STARTUP_SENTINEL" not in worker_startup_module.__all__


@pytest.mark.parametrize(
    "role",
    [POLICY_DRIVER_ROLE, REPRESENTATION_LAUNCHER_ROLE, REPRESENTATION_MEMBER_ROLE],
)
def test_verified_startup_requires_and_returns_one_exact_role(role: str) -> None:
    identity = _identity(role=role)
    verified = _mint(identity, required_role=role)

    assert verified.identity is identity
    assert verified.require_role(role) is identity
    assert verified.authorization_parameters() == identity.authorization_parameters()


def test_verified_startup_rejects_wrong_or_unknown_required_role() -> None:
    identity = _identity()

    with pytest.raises(PermissionError, match="role differs"):
        _mint(
            identity,
            required_role=REPRESENTATION_LAUNCHER_ROLE,
        )
    with pytest.raises(ValueError, match="exactly one of"):
        _mint(identity, required_role="policy")

    verified = _mint(identity, required_role=POLICY_DRIVER_ROLE)
    with pytest.raises(PermissionError, match="role differs"):
        verified.require_role(REPRESENTATION_MEMBER_ROLE)
    with pytest.raises(ValueError, match="exactly one of"):
        verified.require_role("policy-driver-child")


def test_verified_startup_requires_exact_identity_type() -> None:
    with pytest.raises(TypeError, match="exactly WorkerStartupIdentity"):
        _mint(object(), required_role=POLICY_DRIVER_ROLE)  # type: ignore[arg-type]


def test_verified_startup_is_immutable_noncopyable_and_nonserializable() -> None:
    verified = _mint(
        _identity(),
        required_role=POLICY_DRIVER_ROLE,
    )

    with pytest.raises(AttributeError, match="immutable"):
        verified.extra = "value"  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="process-local"):
        copy.copy(verified)
    with pytest.raises(TypeError, match="process-local"):
        copy.deepcopy(verified)
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(verified)
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class _Derived(VerifiedWorkerStartup):
            pass


def test_verified_startup_rejects_use_from_another_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _mint(
        _identity(),
        required_role=POLICY_DRIVER_ROLE,
    )
    monkeypatch.setattr(
        worker_startup_module.os,
        "getpid",
        lambda: 999_999_999,
    )

    with pytest.raises(RuntimeError, match="different process"):
        _ = verified.identity
    with pytest.raises(RuntimeError, match="different process"):
        verified.require_role(POLICY_DRIVER_ROLE)
    with pytest.raises(RuntimeError, match="different process"):
        verified.authorization_parameters()
