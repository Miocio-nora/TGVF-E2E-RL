from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tgvf_rl.ops.representation_startup import (
    REPRESENTATION_MEMBER_CLAIM_SCHEMA,
    REPRESENTATION_STARTUP_PLAN_SCHEMA,
    RepresentationMemberClaim,
    RepresentationStartupPlan,
    build_representation_startup_plan,
)
from tgvf_rl.ops.worker_startup import (
    POLICY_DRIVER_ROLE,
    REPRESENTATION_LAUNCHER_ROLE,
    REPRESENTATION_MEMBER_ROLE,
    WorkerStartupEnvelope,
    WorkerStartupIdentity,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_SHA256 = "a" * 64
CONFIG_SHA256 = "b" * 64


def _identity(*, role: str, unicode_command: bool = False) -> WorkerStartupIdentity:
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
    elif role == POLICY_DRIVER_ROLE:
        command_name = "run-policy"
        target = "tgvf_rl.framework.verl.policy_main:main"
        runtime_sha256 = "1" * 64
        dependency_sha256 = "2" * 64
    else:  # pragma: no cover - test helper owns its role domain
        raise AssertionError(role)
    suffix = "/配置.toml" if unicode_command else "/config.toml"
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
            suffix,
        ),
        target=target,
        runtime_package_sha256=runtime_sha256,
        dependency_roots_sha256=dependency_sha256,
    )


def _envelope(*, unicode_command: bool = False) -> WorkerStartupEnvelope:
    return WorkerStartupEnvelope(
        entry_role=REPRESENTATION_LAUNCHER_ROLE,
        identities=(
            _identity(
                role=REPRESENTATION_LAUNCHER_ROLE,
                unicode_command=unicode_command,
            ),
            _identity(
                role=REPRESENTATION_MEMBER_ROLE,
                unicode_command=unicode_command,
            ),
        ),
    )


def _plan(
    physical_gpu_ids: tuple[int, ...] = (4, 7),
    *,
    envelope: WorkerStartupEnvelope | None = None,
) -> RepresentationStartupPlan:
    return build_representation_startup_plan(
        _envelope() if envelope is None else envelope,
        run_identity_sha256=RUN_SHA256,
        config_identity_sha256=CONFIG_SHA256,
        physical_gpu_ids=physical_gpu_ids,
    )


def test_builder_binds_complete_envelope_and_exact_world_two_mapping() -> None:
    envelope = _envelope()
    member_identity = envelope.identity_for_role(REPRESENTATION_MEMBER_ROLE)
    plan = _plan(envelope=envelope)

    assert plan.envelope is envelope
    assert plan.envelope_sha256 == envelope.envelope_sha256
    assert plan.member_identity_sha256 == member_identity.identity_sha256
    assert plan.run_identity_sha256 == RUN_SHA256
    assert plan.config_identity_sha256 == CONFIG_SHA256
    assert plan.world_size == 2
    assert tuple(
        (member.global_rank, member.local_rank, member.physical_gpu_id)
        for member in plan.members
    ) == ((0, 0, 4), (1, 1, 7))
    for member in plan.members:
        assert member.envelope_sha256 == envelope.envelope_sha256
        assert member.member_identity_sha256 == member_identity.identity_sha256
        assert member.run_identity_sha256 == RUN_SHA256
        assert member.config_identity_sha256 == CONFIG_SHA256
        assert member.world_size == 2

    record = plan.as_record()
    assert record["schema"] == REPRESENTATION_STARTUP_PLAN_SCHEMA
    assert record["envelope"] == envelope.as_record()
    assert record["envelope_sha256"] == envelope.envelope_sha256
    assert record["member_identity_sha256"] == member_identity.identity_sha256
    assert len(plan.plan_sha256) == 64


def test_builder_accepts_complete_world_four_mapping() -> None:
    plan = _plan((0, 2, 5, 9))

    assert plan.world_size == 4
    assert tuple(member.global_rank for member in plan.members) == (0, 1, 2, 3)
    assert tuple(member.local_rank for member in plan.members) == (0, 1, 2, 3)
    assert tuple(member.physical_gpu_id for member in plan.members) == (0, 2, 5, 9)


def test_direct_plan_normalizes_member_order_deterministically() -> None:
    original = _plan()
    reverse = RepresentationStartupPlan(
        envelope=original.envelope,
        run_identity_sha256=original.run_identity_sha256,
        config_identity_sha256=original.config_identity_sha256,
        world_size=original.world_size,
        members=tuple(reversed(original.members)),
    )

    assert reverse.members == original.members
    assert reverse.as_record() == original.as_record()
    assert reverse.to_json() == original.to_json()
    assert reverse.plan_sha256 == original.plan_sha256


def test_envelope_or_member_command_change_changes_plan_identity() -> None:
    original = _plan()
    envelope = original.envelope
    launcher = envelope.identity_for_role(REPRESENTATION_LAUNCHER_ROLE)
    member = envelope.identity_for_role(REPRESENTATION_MEMBER_ROLE)
    changed_member = WorkerStartupIdentity(
        role=member.role,
        command=(*member.command, "--changed"),
        target=member.target,
        runtime_package_sha256=member.runtime_package_sha256,
        dependency_roots_sha256=member.dependency_roots_sha256,
    )
    changed_envelope = WorkerStartupEnvelope(
        entry_role=REPRESENTATION_LAUNCHER_ROLE,
        identities=(launcher, changed_member),
    )
    changed = _plan(envelope=changed_envelope)

    assert changed.envelope_sha256 != original.envelope_sha256
    assert changed.member_identity_sha256 != original.member_identity_sha256
    assert changed.plan_sha256 != original.plan_sha256


@pytest.mark.parametrize(
    "physical_gpu_ids",
    [
        (),
        (0,),
        (0, 1, 2),
        (0, 1, 2, 3, 4),
        (0, 0),
        (0, -1),
        (0, True),
    ],
)
def test_builder_rejects_invalid_or_nonunique_gpu_topology(
    physical_gpu_ids: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="world_size|distinct|physical_gpu_id"):
        _plan(physical_gpu_ids)


def test_builder_requires_exact_envelope_and_gpu_tuple_types() -> None:
    with pytest.raises(TypeError, match="envelope type"):
        build_representation_startup_plan(
            object(),  # type: ignore[arg-type]
            run_identity_sha256=RUN_SHA256,
            config_identity_sha256=CONFIG_SHA256,
            physical_gpu_ids=(0, 1),
        )
    with pytest.raises(TypeError, match="exact tuple"):
        build_representation_startup_plan(
            _envelope(),
            run_identity_sha256=RUN_SHA256,
            config_identity_sha256=CONFIG_SHA256,
            physical_gpu_ids=[0, 1],  # type: ignore[arg-type]
        )


def test_builder_rejects_policy_envelope() -> None:
    policy_envelope = WorkerStartupEnvelope(
        entry_role=POLICY_DRIVER_ROLE,
        identities=(_identity(role=POLICY_DRIVER_ROLE),),
    )

    with pytest.raises(ValueError, match="entry role"):
        _plan(envelope=policy_envelope)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("run_identity_sha256", "A" * 64),
        ("run_identity_sha256", "a" * 63),
        ("run_identity_sha256", 1),
        ("config_identity_sha256", "g" * 64),
        ("config_identity_sha256", None),
    ],
)
def test_builder_rejects_malformed_run_or_config_identity(
    name: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "run_identity_sha256": RUN_SHA256,
        "config_identity_sha256": CONFIG_SHA256,
    }
    arguments[name] = value
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_representation_startup_plan(
            _envelope(),
            physical_gpu_ids=(0, 1),
            **arguments,  # type: ignore[arg-type]
        )


def test_member_claim_record_is_exact_and_digest_bound() -> None:
    member = _plan().members[0]
    record = member.as_record()

    assert record["schema"] == REPRESENTATION_MEMBER_CLAIM_SCHEMA
    assert record["claim_sha256"] == member.claim_sha256
    assert RepresentationMemberClaim.from_record(record) == member

    record["physical_gpu_id"] = 8
    with pytest.raises(ValueError, match="digest differs"):
        RepresentationMemberClaim.from_record(record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("envelope_sha256", "A" * 64, "lowercase SHA-256"),
        ("member_identity_sha256", "x", "lowercase SHA-256"),
        ("run_identity_sha256", None, "lowercase SHA-256"),
        ("config_identity_sha256", 7, "lowercase SHA-256"),
        ("world_size", 3, "exactly 2 or 4"),
        ("world_size", True, "exactly 2 or 4"),
        ("global_rank", -1, r"\[0, world_size\)"),
        ("global_rank", True, r"\[0, world_size\)"),
        ("local_rank", 2, r"\[0, world_size\)"),
        ("physical_gpu_id", -1, "non-negative integer"),
        ("physical_gpu_id", False, "non-negative integer"),
    ],
)
def test_member_claim_requires_exact_field_domains(
    field: str,
    value: object,
    message: str,
) -> None:
    values = {
        "envelope_sha256": "c" * 64,
        "member_identity_sha256": "d" * 64,
        "run_identity_sha256": RUN_SHA256,
        "config_identity_sha256": CONFIG_SHA256,
        "world_size": 2,
        "global_rank": 0,
        "local_rank": 0,
        "physical_gpu_id": 4,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        RepresentationMemberClaim(**values)  # type: ignore[arg-type]


def test_member_claim_requires_single_node_global_local_rank_identity() -> None:
    member = _plan().members[1]
    with pytest.raises(ValueError, match="global_rank must equal local_rank"):
        replace(member, local_rank=0)


@pytest.mark.parametrize(
    "mutation",
    ["extra", "missing", "schema", "digest", "wrong-container"],
)
def test_member_claim_record_rejects_nonexact_content(mutation: str) -> None:
    record: object = _plan().members[0].as_record()
    assert isinstance(record, dict)
    if mutation == "extra":
        record["extra"] = "forbidden"
    elif mutation == "missing":
        del record["local_rank"]
    elif mutation == "schema":
        record["schema"] = "future"
    elif mutation == "digest":
        record["claim_sha256"] = "0" * 64
    elif mutation == "wrong-container":
        record = list(record.items())
    else:  # pragma: no cover - exhaustive parametrization
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="field set|schema|digest"):
        RepresentationMemberClaim.from_record(record)


def test_plan_rejects_incomplete_duplicate_rank_or_gpu_mapping() -> None:
    original = _plan()
    member_zero, member_one = original.members

    with pytest.raises(ValueError, match="member count"):
        replace(original, members=(member_zero,))
    with pytest.raises(ValueError, match="global ranks"):
        replace(original, members=(member_zero, member_zero))
    with pytest.raises(ValueError, match="one-to-one"):
        replace(
            original,
            members=(
                member_zero,
                replace(member_one, physical_gpu_id=member_zero.physical_gpu_id),
            ),
        )


def test_plan_defensively_rejects_tampered_local_rank_set() -> None:
    original = _plan()
    member_zero, member_one = original.members
    object.__setattr__(member_one, "local_rank", 0)

    with pytest.raises(ValueError, match="local ranks"):
        replace(original, members=(member_zero, member_one))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("envelope_sha256", "0" * 64, "envelope_sha256"),
        ("member_identity_sha256", "0" * 64, "member_identity_sha256"),
        ("run_identity_sha256", "0" * 64, "run_identity_sha256"),
        ("config_identity_sha256", "0" * 64, "config_identity_sha256"),
        ("world_size", 4, "world_size"),
    ],
)
def test_plan_rejects_member_shared_identity_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    original = _plan()
    changed = replace(original.members[0], **{field: value})
    with pytest.raises(ValueError, match=message):
        replace(original, members=(changed, original.members[1]))


def test_plan_requires_exact_envelope_member_tuple_and_member_types() -> None:
    original = _plan()
    with pytest.raises(TypeError, match="envelope type"):
        replace(original, envelope=object())
    with pytest.raises(TypeError, match="exact tuple"):
        replace(original, members=list(original.members))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exactly RepresentationMemberClaim"):
        replace(original, members=(object(), object()))  # type: ignore[arg-type]


def test_plan_json_and_authorization_round_trip_are_exact() -> None:
    plan = _plan((1, 3, 5, 7))
    encoded = plan.to_json()
    parameters = plan.authorization_parameters()

    assert RepresentationStartupPlan.from_json(encoded) == plan
    assert parameters == {
        "representation_startup_plan_schema": REPRESENTATION_STARTUP_PLAN_SCHEMA,
        "representation_startup_plan_json": encoded,
        "representation_startup_plan_sha256": plan.plan_sha256,
    }
    assert RepresentationStartupPlan.from_authorization_parameters(parameters) == plan


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-field",
        "missing-field",
        "wrong-schema",
        "wrong-members-container",
        "envelope-digest",
        "member-identity-digest",
        "member-claim-digest",
        "nested-envelope-extra",
    ],
)
def test_plan_record_rejects_nonexact_or_tampered_content(mutation: str) -> None:
    record = _plan().as_record()
    if mutation == "extra-field":
        record["extra"] = "forbidden"
    elif mutation == "missing-field":
        del record["config_identity_sha256"]
    elif mutation == "wrong-schema":
        record["schema"] = "future"
    elif mutation == "wrong-members-container":
        record["members"] = tuple(record["members"])  # type: ignore[arg-type]
    elif mutation == "envelope-digest":
        record["envelope_sha256"] = "0" * 64
    elif mutation == "member-identity-digest":
        record["member_identity_sha256"] = "0" * 64
    elif mutation == "member-claim-digest":
        members = record["members"]
        assert isinstance(members, list) and isinstance(members[0], dict)
        members[0]["claim_sha256"] = "0" * 64
    elif mutation == "nested-envelope-extra":
        envelope = record["envelope"]
        assert isinstance(envelope, dict)
        envelope["extra"] = "forbidden"
    else:  # pragma: no cover - exhaustive parametrization
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="field set|schema|array|digest"):
        RepresentationStartupPlan.from_record(record)


def test_json_rejects_duplicate_keys_at_top_and_nested_depths() -> None:
    encoded = _plan().to_json()
    duplicate_top = encoded.replace(
        '{"config_identity_sha256":',
        '{"config_identity_sha256":"b","config_identity_sha256":',
        1,
    )
    duplicate_member = encoded.replace(
        '"global_rank":0,',
        '"global_rank":0,"global_rank":0,',
        1,
    )
    duplicate_envelope = encoded.replace(
        '"entry_role":"representation-launcher",',
        '"entry_role":"representation-launcher",'
        '"entry_role":"representation-launcher",',
        1,
    )

    for value in (duplicate_top, duplicate_member, duplicate_envelope):
        with pytest.raises(ValueError, match="duplicate key"):
            RepresentationStartupPlan.from_json(value)


def test_json_rejects_unknown_nonfinite_malformed_and_noncanonical_spelling() -> None:
    plan = _plan()
    encoded = plan.to_json()
    unknown = plan.as_record()
    unknown["future"] = "forbidden"
    nonfinite = encoded.replace('"world_size":2', '"world_size":NaN', 1)
    pretty = json.dumps(plan.as_record(), ensure_ascii=False, sort_keys=True)

    with pytest.raises(ValueError, match="field set"):
        RepresentationStartupPlan.from_json(
            json.dumps(
                unknown, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
    with pytest.raises(ValueError, match="non-finite"):
        RepresentationStartupPlan.from_json(nonfinite)
    with pytest.raises(ValueError, match="malformed"):
        RepresentationStartupPlan.from_json(encoded[:-1])
    for value in (encoded + " ", pretty):
        with pytest.raises(ValueError, match="not canonical"):
            RepresentationStartupPlan.from_json(value)


def test_json_canonical_utf8_accepts_unicode_and_rejects_escaped_or_surrogate() -> None:
    plan = _plan(envelope=_envelope(unicode_command=True))
    encoded = plan.to_json()

    assert "配置" in encoded
    assert RepresentationStartupPlan.from_json(encoded) == plan
    escaped = json.dumps(
        plan.as_record(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises(ValueError, match="not canonical"):
        RepresentationStartupPlan.from_json(escaped)
    with pytest.raises(ValueError, match="valid UTF-8"):
        RepresentationStartupPlan.from_json(encoded[:-1] + "\ud800")
    with pytest.raises(TypeError, match="exactly str"):
        RepresentationStartupPlan.from_json(encoded.encode("utf-8"))


@pytest.mark.parametrize(
    "mutation",
    ["extra", "missing", "schema", "digest", "json"],
)
def test_authorization_parameters_reject_nonexact_or_tampered_group(
    mutation: str,
) -> None:
    parameters: object = _plan().authorization_parameters()
    assert isinstance(parameters, dict)
    if mutation == "extra":
        parameters["extra"] = "forbidden"
    elif mutation == "missing":
        del parameters["representation_startup_plan_sha256"]
    elif mutation == "schema":
        parameters["representation_startup_plan_schema"] = "future"
    elif mutation == "digest":
        parameters["representation_startup_plan_sha256"] = "0" * 64
    elif mutation == "json":
        parameters["representation_startup_plan_json"] += " "
    else:  # pragma: no cover - exhaustive parametrization
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="field set|schema|digest|canonical"):
        RepresentationStartupPlan.from_authorization_parameters(parameters)


def test_authorization_parameters_require_exact_container_keys_and_values() -> None:
    class _StringSubclass(str):
        pass

    parameters = _plan().authorization_parameters()
    with pytest.raises(TypeError, match="exact dict"):
        RepresentationStartupPlan.from_authorization_parameters(
            list(parameters.items())
        )

    subclass_keys = {_StringSubclass(name): value for name, value in parameters.items()}
    with pytest.raises(TypeError, match="keys must be exactly str"):
        RepresentationStartupPlan.from_authorization_parameters(subclass_keys)

    subclass_schema = dict(parameters)
    subclass_schema["representation_startup_plan_schema"] = _StringSubclass(
        REPRESENTATION_STARTUP_PLAN_SCHEMA
    )
    with pytest.raises(TypeError, match="values must be exactly str"):
        RepresentationStartupPlan.from_authorization_parameters(subclass_schema)

    future_authority = dict(parameters)
    future_authority[_StringSubclass("representation_startup_future_authority")] = (
        "forbidden"
    )
    with pytest.raises(TypeError, match="keys must be exactly str"):
        RepresentationStartupPlan.from_authorization_parameters(future_authority)


def test_plan_reconstructs_from_complete_broader_cli_authorization_map() -> None:
    plan = _plan()
    parameters = {
        "canonical_config_sha256": CONFIG_SHA256,
        "prepared_representation_launch_sha256": "c" * 64,
        **plan.authorization_parameters(),
    }

    assert RepresentationStartupPlan.from_cli_authorization_parameters(parameters) == (
        plan
    )


@pytest.mark.parametrize(
    "retained_names",
    [
        (),
        ("representation_startup_plan_schema",),
        (
            "representation_startup_plan_schema",
            "representation_startup_plan_json",
        ),
    ],
)
def test_cli_authorization_rejects_missing_or_partial_plan_group(
    retained_names: tuple[str, ...],
) -> None:
    complete = _plan().authorization_parameters()
    parameters = {name: complete[name] for name in retained_names}
    parameters["unrelated_cli_parameter"] = "ignored"

    with pytest.raises(ValueError, match="parameter group differs.*missing"):
        RepresentationStartupPlan.from_cli_authorization_parameters(parameters)


@pytest.mark.parametrize(
    "extra_name",
    [
        "representation_startup_plan_future",
        "representation_startup_member_claim",
        "representation_startup_future_authority",
    ],
)
def test_cli_authorization_rejects_extra_protected_plan_parameter(
    extra_name: str,
) -> None:
    parameters = _plan().authorization_parameters()
    parameters[extra_name] = "forbidden"

    with pytest.raises(ValueError, match="parameter group differs.*extra"):
        RepresentationStartupPlan.from_cli_authorization_parameters(parameters)


def test_cli_authorization_checks_exact_keys_before_protected_namespace_scan() -> None:
    class _StringSubclass(str):
        pass

    plan = _plan()
    subclass_required = {
        _StringSubclass(name): value
        for name, value in plan.authorization_parameters().items()
    }
    with pytest.raises(TypeError, match="keys must be exactly str"):
        RepresentationStartupPlan.from_cli_authorization_parameters(subclass_required)

    subclass_extra = plan.authorization_parameters()
    subclass_extra[_StringSubclass("representation_startup_future_authority")] = (
        "forbidden"
    )
    with pytest.raises(TypeError, match="keys must be exactly str"):
        RepresentationStartupPlan.from_cli_authorization_parameters(subclass_extra)

    nonstring_unrelated = plan.authorization_parameters()
    nonstring_unrelated[1] = object()  # type: ignore[index]
    with pytest.raises(TypeError, match="keys must be exactly str"):
        RepresentationStartupPlan.from_cli_authorization_parameters(nonstring_unrelated)


def test_cli_authorization_requires_exact_dict_and_delegates_value_checks() -> None:
    class _DictSubclass(dict[str, str]):
        pass

    parameters = _plan().authorization_parameters()
    with pytest.raises(TypeError, match="exact dict"):
        RepresentationStartupPlan.from_cli_authorization_parameters(
            _DictSubclass(parameters)
        )

    parameters["representation_startup_plan_json"] = 1  # type: ignore[assignment]
    with pytest.raises(TypeError, match="values must be exactly str"):
        RepresentationStartupPlan.from_cli_authorization_parameters(parameters)


def test_contract_objects_are_frozen() -> None:
    plan = _plan()
    with pytest.raises(FrozenInstanceError):
        plan.world_size = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.members[0].global_rank = 1  # type: ignore[misc]


def test_representation_startup_leaf_has_isolated_python_firebreak() -> None:
    script = """
import sys
import tgvf_rl.ops.representation_startup as startup
assert startup.REPRESENTATION_STARTUP_PLAN_SCHEMA
for prefix in ('torch', 'numpy', 'tgvf_rl.framework', 'tgvf_rl.representation'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)
"""
    completed = subprocess.run(
        (sys.executable, "-B", "-P", "-S", "-c", script),
        cwd=REPOSITORY_ROOT,
        env={
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.defpath,
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
            "PYTHONUTF8": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
