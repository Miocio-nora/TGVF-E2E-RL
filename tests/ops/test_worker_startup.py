from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
import pickle

import pytest

from tgvf_rl.ops import worker_startup as worker_startup_module
from tgvf_rl.ops.worker_startup import (
    POLICY_DRIVER_ROLE,
    REPRESENTATION_LAUNCHER_ROLE,
    REPRESENTATION_MEMBER_ROLE,
    SUPPORTED_WORKER_STARTUP_ROLES,
    WORKER_STARTUP_ENVELOPE_SCHEMA,
    WORKER_STARTUP_SCHEMA,
    VerifiedWorkerStartup,
    WorkerStartupEnvelope,
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


def _representation_identity(*, role: str) -> WorkerStartupIdentity:
    if role == REPRESENTATION_LAUNCHER_ROLE:
        command_name = "run-representation-launcher"
        target = "tgvf_rl.ops.representation_launcher:main"
        runtime_sha256 = "c" * 64
        dependency_sha256 = "d" * 64
    elif role == REPRESENTATION_MEMBER_ROLE:
        command_name = "run-representation-member"
        target = "tgvf_rl.representation.training.runner:run_representation_training"
        runtime_sha256 = "e" * 64
        dependency_sha256 = "f" * 64
    else:  # pragma: no cover - test helper owns its role domain
        raise AssertionError(f"unexpected representation role {role!r}")
    return WorkerStartupIdentity(
        role=role,
        command=(
            "/runtime/python",
            "-B",
            "-P",
            "-S",
            "-m",
            "tgvf_rl.worker_bootstrap",
            command_name,
            "/runtime/config.toml",
        ),
        target=target,
        runtime_package_sha256=runtime_sha256,
        dependency_roots_sha256=dependency_sha256,
    )


def _representation_envelope() -> WorkerStartupEnvelope:
    return WorkerStartupEnvelope(
        entry_role=REPRESENTATION_LAUNCHER_ROLE,
        identities=(
            _representation_identity(role=REPRESENTATION_LAUNCHER_ROLE),
            _representation_identity(role=REPRESENTATION_MEMBER_ROLE),
        ),
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


def test_identity_and_envelope_reject_lone_unicode_command_surrogate() -> None:
    with pytest.raises(ValueError, match="valid UTF-8 text"):
        WorkerStartupIdentity(
            role=POLICY_DRIVER_ROLE,
            command=("/runtime/python", "\ud800"),
            target="tgvf_rl.framework.verl.policy_main:main",
            runtime_package_sha256="a" * 64,
            dependency_roots_sha256="b" * 64,
        )

    malformed = (
        _representation_envelope()
        .to_json()
        .replace(
            '"command":[',
            '"command":["\\ud800",',
            1,
        )
    )
    with pytest.raises(ValueError, match="valid UTF-8 text"):
        WorkerStartupEnvelope.from_json(malformed)


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


def test_identity_record_is_exact_digest_bound_and_round_trips() -> None:
    identity = _identity()
    record = identity.as_record()

    assert set(record) == {
        "schema",
        "role",
        "command",
        "target",
        "runtime_package_sha256",
        "dependency_roots_sha256",
        "identity_sha256",
    }
    assert record["schema"] == WORKER_STARTUP_SCHEMA
    assert record["command"] == list(identity.command)
    assert record["identity_sha256"] == identity.identity_sha256
    assert WorkerStartupIdentity.from_record(record) == identity


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-field",
        "missing-field",
        "wrong-schema",
        "wrong-command-container",
        "wrong-digest",
    ],
)
def test_identity_record_rejects_nonexact_or_unbound_content(mutation: str) -> None:
    record = _identity().as_record()
    if mutation == "extra-field":
        record["extra"] = "forbidden"
    elif mutation == "missing-field":
        del record["target"]
    elif mutation == "wrong-schema":
        record["schema"] = "tgvf-worker-startup-v0"
    elif mutation == "wrong-command-container":
        record["command"] = tuple(record["command"])  # type: ignore[arg-type]
    elif mutation == "wrong-digest":
        record["identity_sha256"] = "0" * 64
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="record|schema|command|digest"):
        WorkerStartupIdentity.from_record(record)


def test_policy_envelope_has_one_atomic_exact_authorization_group() -> None:
    identity = _identity()
    envelope = WorkerStartupEnvelope(
        entry_role=POLICY_DRIVER_ROLE,
        identities=(identity,),
    )

    assert envelope.identity_for_role(POLICY_DRIVER_ROLE) is identity
    assert envelope.as_record() == {
        "schema": WORKER_STARTUP_ENVELOPE_SCHEMA,
        "entry_role": POLICY_DRIVER_ROLE,
        "identities": {POLICY_DRIVER_ROLE: identity.as_record()},
    }
    parameters = envelope.authorization_parameters()
    assert set(parameters) == {
        "worker_startup_envelope_schema",
        "worker_startup_envelope_json",
        "worker_startup_envelope_sha256",
    }
    assert (
        parameters["worker_startup_envelope_schema"] == WORKER_STARTUP_ENVELOPE_SCHEMA
    )
    assert parameters["worker_startup_envelope_json"] == envelope.to_json()
    assert parameters["worker_startup_envelope_sha256"] == envelope.envelope_sha256
    assert (
        sha256(envelope.to_json().encode("utf-8")).hexdigest()
        == envelope.envelope_sha256
    )
    assert WorkerStartupEnvelope.from_json(envelope.to_json()) == envelope
    assert WorkerStartupEnvelope.from_record(envelope.as_record()) == envelope


def test_envelope_reconstructs_from_broader_authorization_parameters() -> None:
    envelope = WorkerStartupEnvelope(
        entry_role=POLICY_DRIVER_ROLE,
        identities=(_identity(),),
    )
    parameters = {
        "canonical_config_sha256": "c" * 64,
        "prepared_policy_launch_sha256": "d" * 64,
        **envelope.authorization_parameters(),
    }

    assert (
        WorkerStartupEnvelope.from_authorization_parameters(
            parameters,
            expected_entry_role=POLICY_DRIVER_ROLE,
        )
        == envelope
    )


@pytest.mark.parametrize(
    "retained_names",
    [
        (),
        ("worker_startup_envelope_schema",),
        (
            "worker_startup_envelope_schema",
            "worker_startup_envelope_json",
        ),
    ],
)
def test_envelope_authorization_rejects_missing_or_partial_group(
    retained_names: tuple[str, ...],
) -> None:
    complete = _representation_envelope().authorization_parameters()
    parameters = {name: complete[name] for name in retained_names}
    parameters["unrelated_cli_parameter"] = "ignored"

    with pytest.raises(ValueError, match="parameter group differs.*missing"):
        WorkerStartupEnvelope.from_authorization_parameters(
            parameters,
            expected_entry_role=REPRESENTATION_LAUNCHER_ROLE,
        )


@pytest.mark.parametrize(
    "extra_name",
    [
        "worker_startup_role",
        "worker_startup_envelope_extra",
        "worker_startup_future_authority",
    ],
)
def test_envelope_authorization_rejects_any_extra_startup_parameter(
    extra_name: str,
) -> None:
    parameters = _representation_envelope().authorization_parameters()
    parameters[extra_name] = "forbidden"

    with pytest.raises(ValueError, match="parameter group differs.*extra"):
        WorkerStartupEnvelope.from_authorization_parameters(
            parameters,
            expected_entry_role=REPRESENTATION_LAUNCHER_ROLE,
        )


def test_envelope_authorization_requires_exact_container_and_value_types() -> None:
    envelope = _representation_envelope()
    parameters = envelope.authorization_parameters()

    class _DictSubclass(dict[str, str]):
        pass

    with pytest.raises(TypeError, match="exact dict"):
        WorkerStartupEnvelope.from_authorization_parameters(
            _DictSubclass(parameters),
            expected_entry_role=REPRESENTATION_LAUNCHER_ROLE,
        )

    parameters["worker_startup_envelope_schema"] = 1  # type: ignore[assignment]
    with pytest.raises(TypeError, match="exactly str"):
        WorkerStartupEnvelope.from_authorization_parameters(
            parameters,
            expected_entry_role=REPRESENTATION_LAUNCHER_ROLE,
        )


def test_envelope_authorization_rejects_nonexact_keys_before_namespace_scan() -> None:
    class _StringSubclass(str):
        pass

    envelope = _representation_envelope()
    subclass_required = {
        _StringSubclass(name): value
        for name, value in envelope.authorization_parameters().items()
    }
    with pytest.raises(TypeError, match="keys must be exactly str"):
        WorkerStartupEnvelope.from_authorization_parameters(
            subclass_required,
            expected_entry_role=REPRESENTATION_LAUNCHER_ROLE,
        )

    subclass_extra = envelope.authorization_parameters()
    subclass_extra[_StringSubclass("worker_startup_future_authority")] = "forbidden"
    with pytest.raises(TypeError, match="keys must be exactly str"):
        WorkerStartupEnvelope.from_authorization_parameters(
            subclass_extra,
            expected_entry_role=REPRESENTATION_LAUNCHER_ROLE,
        )

    nonstring_unrelated = envelope.authorization_parameters()
    nonstring_unrelated[1] = object()  # type: ignore[index]
    with pytest.raises(TypeError, match="keys must be exactly str"):
        WorkerStartupEnvelope.from_authorization_parameters(
            nonstring_unrelated,
            expected_entry_role=REPRESENTATION_LAUNCHER_ROLE,
        )


def test_envelope_authorization_rejects_schema_or_digest_drift() -> None:
    envelope = _representation_envelope()
    wrong_schema = envelope.authorization_parameters()
    wrong_schema["worker_startup_envelope_schema"] = "tgvf-worker-startup-envelope-v0"
    with pytest.raises(ValueError, match="authorization schema differs"):
        WorkerStartupEnvelope.from_authorization_parameters(
            wrong_schema,
            expected_entry_role=REPRESENTATION_LAUNCHER_ROLE,
        )

    wrong_digest = envelope.authorization_parameters()
    wrong_digest["worker_startup_envelope_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authorization SHA256 differs"):
        WorkerStartupEnvelope.from_authorization_parameters(
            wrong_digest,
            expected_entry_role=REPRESENTATION_LAUNCHER_ROLE,
        )


@pytest.mark.parametrize("mutation", ["noncanonical", "duplicate", "unknown"])
def test_envelope_authorization_reuses_strict_nested_json_contract(
    mutation: str,
) -> None:
    envelope = _representation_envelope()
    parameters = envelope.authorization_parameters()
    canonical = parameters["worker_startup_envelope_json"]
    if mutation == "noncanonical":
        changed = canonical + " "
        expected = "not canonical"
    elif mutation == "duplicate":
        changed = canonical.replace(
            '"entry_role":',
            f'"entry_role":"{REPRESENTATION_LAUNCHER_ROLE}","entry_role":',
            1,
        )
        expected = "duplicate key"
    elif mutation == "unknown":
        record = envelope.as_record()
        record["unknown"] = "forbidden"
        changed = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected = "record field set differs"
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)
    parameters["worker_startup_envelope_json"] = changed

    with pytest.raises(ValueError, match=expected):
        WorkerStartupEnvelope.from_authorization_parameters(
            parameters,
            expected_entry_role=REPRESENTATION_LAUNCHER_ROLE,
        )


def test_envelope_authorization_rejects_wrong_expected_entry_role() -> None:
    parameters = _representation_envelope().authorization_parameters()

    with pytest.raises(PermissionError, match="entry role differs"):
        WorkerStartupEnvelope.from_authorization_parameters(
            parameters,
            expected_entry_role=POLICY_DRIVER_ROLE,
        )


def test_representation_envelope_preserves_both_roles_without_flat_key_loss() -> None:
    envelope = _representation_envelope()
    launcher = envelope.identity_for_role(REPRESENTATION_LAUNCHER_ROLE)
    member = envelope.identity_for_role(REPRESENTATION_MEMBER_ROLE)
    record = json.loads(envelope.to_json())

    assert tuple(identity.role for identity in envelope.identities) == (
        REPRESENTATION_LAUNCHER_ROLE,
        REPRESENTATION_MEMBER_ROLE,
    )
    assert set(record["identities"]) == {
        REPRESENTATION_LAUNCHER_ROLE,
        REPRESENTATION_MEMBER_ROLE,
    }
    assert (
        record["identities"][REPRESENTATION_LAUNCHER_ROLE]["identity_sha256"]
        == launcher.identity_sha256
    )
    assert (
        record["identities"][REPRESENTATION_MEMBER_ROLE]["identity_sha256"]
        == member.identity_sha256
    )
    assert launcher.identity_sha256 != member.identity_sha256
    flat_legacy_merge = {
        **launcher.authorization_parameters(),
        **member.authorization_parameters(),
    }
    assert flat_legacy_merge["worker_startup_role"] == REPRESENTATION_MEMBER_ROLE
    assert len(envelope.authorization_parameters()) == 3


def test_representation_identity_order_does_not_change_canonical_envelope() -> None:
    forward = _representation_envelope()
    reverse = WorkerStartupEnvelope(
        entry_role=REPRESENTATION_LAUNCHER_ROLE,
        identities=tuple(reversed(forward.identities)),
    )

    assert reverse.identities == forward.identities
    assert reverse.as_record() == forward.as_record()
    assert reverse.to_json() == forward.to_json()
    assert reverse.envelope_sha256 == forward.envelope_sha256


def test_member_command_change_changes_complete_representation_envelope() -> None:
    original = _representation_envelope()
    launcher = original.identity_for_role(REPRESENTATION_LAUNCHER_ROLE)
    member = original.identity_for_role(REPRESENTATION_MEMBER_ROLE)
    changed_member = WorkerStartupIdentity(
        role=member.role,
        command=(*member.command[:-1], member.command[-1][:-1] + "m"),
        target=member.target,
        runtime_package_sha256=member.runtime_package_sha256,
        dependency_roots_sha256=member.dependency_roots_sha256,
    )
    changed = WorkerStartupEnvelope(
        entry_role=REPRESENTATION_LAUNCHER_ROLE,
        identities=(launcher, changed_member),
    )

    assert changed.identity_for_role(REPRESENTATION_LAUNCHER_ROLE) == launcher
    original_command = "\0".join(member.command).encode("ascii")
    changed_command = "\0".join(changed_member.command).encode("ascii")
    assert len(changed_command) == len(original_command)
    assert (
        sum(left != right for left, right in zip(original_command, changed_command))
        == 1
    )
    assert changed_member.identity_sha256 != member.identity_sha256
    assert changed.to_json() != original.to_json()
    assert changed.envelope_sha256 != original.envelope_sha256


@pytest.mark.parametrize(
    ("entry_role", "identities"),
    [
        (POLICY_DRIVER_ROLE, ()),
        (
            POLICY_DRIVER_ROLE,
            (
                _identity(),
                _representation_identity(role=REPRESENTATION_LAUNCHER_ROLE),
            ),
        ),
        (
            POLICY_DRIVER_ROLE,
            (_representation_identity(role=REPRESENTATION_LAUNCHER_ROLE),),
        ),
        (
            REPRESENTATION_LAUNCHER_ROLE,
            (_representation_identity(role=REPRESENTATION_LAUNCHER_ROLE),),
        ),
        (
            REPRESENTATION_LAUNCHER_ROLE,
            (
                _identity(),
                _representation_identity(role=REPRESENTATION_LAUNCHER_ROLE),
                _representation_identity(role=REPRESENTATION_MEMBER_ROLE),
            ),
        ),
    ],
)
def test_envelope_rejects_missing_extra_or_mismatched_role_sets(
    entry_role: str,
    identities: tuple[WorkerStartupIdentity, ...],
) -> None:
    with pytest.raises(ValueError, match="role set differs"):
        WorkerStartupEnvelope(entry_role=entry_role, identities=identities)


def test_envelope_rejects_duplicate_role_and_member_entry_role() -> None:
    launcher = _representation_identity(role=REPRESENTATION_LAUNCHER_ROLE)
    with pytest.raises(ValueError, match="repeats role"):
        WorkerStartupEnvelope(
            entry_role=REPRESENTATION_LAUNCHER_ROLE,
            identities=(launcher, launcher),
        )
    with pytest.raises(ValueError, match="entry role"):
        WorkerStartupEnvelope(
            entry_role=REPRESENTATION_MEMBER_ROLE,
            identities=(_representation_identity(role=REPRESENTATION_MEMBER_ROLE),),
        )


def test_envelope_requires_exact_tuple_and_identity_types() -> None:
    with pytest.raises(TypeError, match="exact tuple"):
        WorkerStartupEnvelope(
            entry_role=POLICY_DRIVER_ROLE,
            identities=[_identity()],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="exactly WorkerStartupIdentity"):
        WorkerStartupEnvelope(
            entry_role=POLICY_DRIVER_ROLE,
            identities=(object(),),  # type: ignore[arg-type]
        )


def test_envelope_identity_lookup_rejects_absent_or_unknown_role() -> None:
    envelope = WorkerStartupEnvelope(
        entry_role=POLICY_DRIVER_ROLE,
        identities=(_identity(),),
    )

    with pytest.raises(PermissionError, match="does not contain"):
        envelope.identity_for_role(REPRESENTATION_MEMBER_ROLE)
    with pytest.raises(ValueError, match="exactly one of"):
        envelope.identity_for_role("policy")


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-field",
        "missing-field",
        "wrong-schema",
        "non-object-identities",
        "role-key-mismatch",
        "nested-extra-field",
        "nested-digest-drift",
    ],
)
def test_envelope_record_rejects_nonexact_nested_content(mutation: str) -> None:
    record = _representation_envelope().as_record()
    identities = record["identities"]
    assert isinstance(identities, dict)
    if mutation == "extra-field":
        record["extra"] = "forbidden"
    elif mutation == "missing-field":
        del record["entry_role"]
    elif mutation == "wrong-schema":
        record["schema"] = "tgvf-worker-startup-envelope-v0"
    elif mutation == "non-object-identities":
        record["identities"] = list(identities.values())
    elif mutation == "role-key-mismatch":
        identities[POLICY_DRIVER_ROLE] = identities.pop(REPRESENTATION_MEMBER_ROLE)
    elif mutation == "nested-extra-field":
        member = identities[REPRESENTATION_MEMBER_ROLE]
        assert isinstance(member, dict)
        member["extra"] = "forbidden"
    elif mutation == "nested-digest-drift":
        member = identities[REPRESENTATION_MEMBER_ROLE]
        assert isinstance(member, dict)
        member["identity_sha256"] = "0" * 64
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="record|schema|identities|key|digest"):
        WorkerStartupEnvelope.from_record(record)


def test_envelope_json_rejects_duplicate_keys_at_every_depth() -> None:
    canonical = _representation_envelope().to_json()
    duplicate_outer = canonical.replace(
        '"entry_role":',
        f'"entry_role":"{REPRESENTATION_LAUNCHER_ROLE}","entry_role":',
        1,
    )
    duplicate_nested = canonical.replace(
        '"command":',
        '"command":[],"command":',
        1,
    )

    for value in (duplicate_outer, duplicate_nested):
        with pytest.raises(ValueError, match="duplicate key"):
            WorkerStartupEnvelope.from_json(value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_envelope_json_rejects_nonfinite_constants(constant: str) -> None:
    canonical = _representation_envelope().to_json()
    malformed = canonical.replace(
        f'"entry_role":"{REPRESENTATION_LAUNCHER_ROLE}"',
        f'"entry_role":{constant}',
        1,
    )

    with pytest.raises(ValueError, match="non-finite"):
        WorkerStartupEnvelope.from_json(malformed)


def test_envelope_json_rejects_noncanonical_spelling_and_non_string_input() -> None:
    canonical = _representation_envelope().to_json()

    with pytest.raises(ValueError, match="not canonical"):
        WorkerStartupEnvelope.from_json(canonical + " ")
    with pytest.raises(TypeError, match="exactly str"):
        WorkerStartupEnvelope.from_json(json.loads(canonical))


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
