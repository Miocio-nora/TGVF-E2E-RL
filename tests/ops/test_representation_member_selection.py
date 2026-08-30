from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from importlib import metadata, util
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import UUID

import pytest

from tgvf_rl.ops.cli_authorization_identity import (
    CLIExecutionAuthorizationIdentity,
)
from tgvf_rl.ops.child_environment import (
    CLI_WORKER_LATE_ENVIRONMENT_NAMES,
    REPRESENTATION_TORCHRUN_PROFILE,
    TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES,
    build_child_environment,
)
from tgvf_rl.ops.representation_member_selection import (
    REPRESENTATION_MEMBER_SELECTION_SCHEMA,
    REPRESENTATION_TRAINING_COMMAND_ID,
    REPRESENTATION_TRAINING_PHASE,
    RepresentationMemberSelection,
    select_representation_member,
)
from tgvf_rl.ops.representation_startup import (
    RepresentationStartupPlan,
    build_representation_startup_plan,
)
from tgvf_rl.ops.worker_startup import (
    REPRESENTATION_LAUNCHER_ROLE,
    REPRESENTATION_MEMBER_ROLE,
    WorkerStartupEnvelope,
    WorkerStartupIdentity,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_SHA256 = "a" * 64
CONFIG_SHA256 = "b" * 64
TORCHELASTIC_RUN_ID = "12345678-1234-4abc-8def-1234567890ab"


def _worker_identity(role: str) -> WorkerStartupIdentity:
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
    else:  # pragma: no cover - helper has a closed role domain
        raise AssertionError(role)
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
            "/config.toml",
        ),
        target=target,
        runtime_package_sha256=runtime_sha256,
        dependency_roots_sha256=dependency_sha256,
    )


def _plan(
    physical_gpu_ids: tuple[int, ...] = (4, 7),
) -> RepresentationStartupPlan:
    envelope = WorkerStartupEnvelope(
        entry_role=REPRESENTATION_LAUNCHER_ROLE,
        identities=(
            _worker_identity(REPRESENTATION_LAUNCHER_ROLE),
            _worker_identity(REPRESENTATION_MEMBER_ROLE),
        ),
    )
    return build_representation_startup_plan(
        envelope,
        run_identity_sha256=RUN_SHA256,
        config_identity_sha256=CONFIG_SHA256,
        physical_gpu_ids=physical_gpu_ids,
    )


def _cli_identity(
    plan: RepresentationStartupPlan | None = None,
    *,
    parameter_updates: dict[str, str] | None = None,
    phase: str = REPRESENTATION_TRAINING_PHASE,
    command_id: str = REPRESENTATION_TRAINING_COMMAND_ID,
    run_identity_sha256: str | None = None,
) -> CLIExecutionAuthorizationIdentity:
    selected = _plan() if plan is None else plan
    child_environment = _child_environment_binding(selected)
    parameters = {
        "canonical_config_sha256": selected.config_identity_sha256,
        "config_source_sha256": selected.config_identity_sha256,
        "nproc_per_node": str(selected.world_size),
        "prepared_representation_launch_sha256": "1" * 64,
        "python_executable_sha256": "2" * 64,
        "stop_after_global_step": "none",
        "world_size": str(selected.world_size),
        **child_environment.authorization_parameters(),
        **selected.authorization_parameters(),
    }
    parameters.update(parameter_updates or {})
    return CLIExecutionAuthorizationIdentity.create(
        run_id="REPRESENTATION-SELECTION-TEST",
        phase=phase,
        command_id=command_id,
        run_identity_sha256=(
            selected.run_identity_sha256
            if run_identity_sha256 is None
            else run_identity_sha256
        ),
        parameters=parameters,
    )


def _child_environment_binding(plan: RepresentationStartupPlan):
    return build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        owned_environment={
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(member.physical_gpu_id) for member in plan.members
            ),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        },
        host_environment={},
    )


def _canonical_identity_json(identity: CLIExecutionAuthorizationIdentity) -> str:
    return json.dumps(
        identity.as_record(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _environment(
    plan: RepresentationStartupPlan | None = None,
    *,
    identity: CLIExecutionAuthorizationIdentity | None = None,
    rank: int = 0,
    updates: dict[str, str] | None = None,
) -> dict[str, str]:
    selected = _plan() if plan is None else plan
    selected_identity = _cli_identity(selected) if identity is None else identity
    world_size = str(selected.world_size)
    cli_overlay = {
        "TGVF_CLI_CONSUMPTION_RECEIPT_PATH": "/gate/consumptions/token.json",
        "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256": "3" * 64,
        "TGVF_CLI_EXECUTION_IDENTITY_JSON": _canonical_identity_json(selected_identity),
        "TGVF_CLI_GATE_DIRECTORY": "/gate",
        "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH": (
            "/gate/cli-launches/token/launcher-liveness.json"
        ),
        "TGVF_CLI_WORKER_AUTHORIZATION_SCHEMA": (
            "tgvf-cli-worker-authorization-environment-v1"
        ),
    }
    assert set(cli_overlay) == set(CLI_WORKER_LATE_ENVIRONMENT_NAMES)
    values = (
        _child_environment_binding(selected)
        .with_late_overlay(cli_overlay)
        .as_environment()
    )
    torchrun_overlay = {
        "GROUP_RANK": "0",
        "GROUP_WORLD_SIZE": "1",
        "LOCAL_RANK": str(rank),
        "LOCAL_WORLD_SIZE": world_size,
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29400",
        "RANK": str(rank),
        "ROLE_NAME": "default",
        "ROLE_RANK": str(rank),
        "ROLE_WORLD_SIZE": world_size,
        "TORCHELASTIC_ERROR_FILE": f"/tmp/torchelastic/rank-{rank}/error.json",
        "TORCHELASTIC_MAX_RESTARTS": "0",
        "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_RUN_ID": TORCHELASTIC_RUN_ID,
        "TORCHELASTIC_USE_AGENT_STORE": "True",
        "WORLD_SIZE": world_size,
    }
    assert set(torchrun_overlay) == set(TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES)
    values.update(torchrun_overlay)
    values.update(updates or {})
    return values


def test_selects_only_rank_claim_from_complete_cli_plan() -> None:
    plan = _plan()
    identity = _cli_identity(plan)
    environment = _environment(plan, identity=identity, rank=1)
    selection = select_representation_member(
        identity,
        environment,
    )

    assert type(selection) is RepresentationMemberSelection
    assert selection.identity == identity
    assert selection.plan == plan
    assert selection.claim == plan.members[1]
    assert selection.claim.global_rank == 1
    assert selection.claim.local_rank == 1
    assert selection.claim.physical_gpu_id == 7
    assert dict(selection.full_environment) == environment
    assert len(selection.full_environment_sha256) == 64
    assert selection.replay_protected is False
    assert not hasattr(selection, "authorization_parameters")

    record = selection.as_record()
    assert set(record) == {
        "schema",
        "authorization_scope",
        "cli_run_id",
        "cli_phase",
        "cli_command_id",
        "cli_gate_run_identity_sha256",
        "run_identity_sha256",
        "config_identity_sha256",
        "plan_sha256",
        "claim",
        "torchrun_environment",
        "full_environment_sha256",
        "replay_protected",
    }
    assert record["schema"] == REPRESENTATION_MEMBER_SELECTION_SCHEMA
    assert record["authorization_scope"] == "selection-only"
    assert record["cli_run_id"] == "REPRESENTATION-SELECTION-TEST"
    assert record["cli_phase"] == REPRESENTATION_TRAINING_PHASE
    assert record["cli_command_id"] == REPRESENTATION_TRAINING_COMMAND_ID
    assert (
        record["cli_gate_run_identity_sha256"]
        == identity.gate_run_identity["identity_sha256"]
    )
    assert record["run_identity_sha256"] == RUN_SHA256
    assert record["config_identity_sha256"] == CONFIG_SHA256
    assert record["plan_sha256"] == plan.plan_sha256
    assert record["claim"] == plan.members[1].as_record()
    assert record["torchrun_environment"] == {
        name: environment[name]
        for name in {
            "CUDA_VISIBLE_DEVICES",
            *TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES,
        }
    }
    assert record["full_environment_sha256"] == selection.full_environment_sha256
    assert record["replay_protected"] is False


def test_world_four_selects_each_unique_rank_and_gpu_mapping() -> None:
    plan = _plan((0, 2, 5, 9))
    identity = _cli_identity(plan)
    selections = tuple(
        select_representation_member(
            identity,
            _environment(plan, identity=identity, rank=rank),
        )
        for rank in range(4)
    )

    assert tuple(item.claim.global_rank for item in selections) == (0, 1, 2, 3)
    assert tuple(item.claim.physical_gpu_id for item in selections) == (0, 2, 5, 9)
    assert len({item.claim.claim_sha256 for item in selections}) == 4
    assert all(item.replay_protected is False for item in selections)


def test_individual_claim_is_not_accepted_as_authority() -> None:
    plan = _plan()

    with pytest.raises(TypeError, match="CLIExecutionAuthorizationIdentity"):
        select_representation_member(plan.members[0], _environment(plan))


def test_requires_exact_cli_identity_and_fixed_phase_and_command() -> None:
    class _IdentitySubclass(CLIExecutionAuthorizationIdentity):
        pass

    identity = _cli_identity()
    subclass = _IdentitySubclass(
        run_id=identity.run_id,
        phase=identity.phase,
        command_id=identity.command_id,
        run_identity_sha256=identity.run_identity_sha256,
        parameters=identity.parameters,
    )
    with pytest.raises(TypeError, match="exactly CLIExecutionAuthorizationIdentity"):
        select_representation_member(subclass, _environment())

    for changed in (
        _cli_identity(phase="representation_internal_evaluation"),
        _cli_identity(command_id="tgvf-rl:launch-representation:v1"),
    ):
        with pytest.raises(ValueError, match="phase differs|command differs"):
            select_representation_member(changed, _environment())


def test_rejects_plan_run_identity_different_from_cli_identity() -> None:
    with pytest.raises(ValueError, match="plan run identity differs"):
        select_representation_member(
            _cli_identity(run_identity_sha256="0" * 64),
            _environment(),
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("config_source_sha256", "0" * 64),
        ("canonical_config_sha256", "0" * 64),
    ],
)
def test_rejects_either_toml_source_binding_drift(
    parameter: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="bound TOML source"):
        select_representation_member(
            _cli_identity(parameter_updates={parameter: value}),
            _environment(),
        )


@pytest.mark.parametrize(
    "parameter",
    [
        "canonical_config_sha256",
        "config_source_sha256",
        "nproc_per_node",
        "world_size",
    ],
)
def test_rejects_missing_required_cli_parameter(parameter: str) -> None:
    identity = _cli_identity()
    parameters = dict(identity.parameters)
    del parameters[parameter]
    changed = CLIExecutionAuthorizationIdentity.create(
        run_id=identity.run_id,
        phase=identity.phase,
        command_id=identity.command_id,
        run_identity_sha256=identity.run_identity_sha256,
        parameters=parameters,
    )

    with pytest.raises(ValueError, match=f"missing: {parameter}"):
        select_representation_member(changed, _environment())


@pytest.mark.parametrize("parameter", ["nproc_per_node", "world_size"])
def test_rejects_cli_world_size_drift(parameter: str) -> None:
    with pytest.raises(ValueError, match=f"CLI {parameter}"):
        select_representation_member(
            _cli_identity(parameter_updates={parameter: "4"}),
            _environment(),
        )


def test_rejects_nonexact_required_cli_parameter_value() -> None:
    class _StringSubclass(str):
        pass

    with pytest.raises(TypeError, match="CLI parameters.*exact string pairs"):
        select_representation_member(
            _cli_identity(
                parameter_updates={
                    "config_source_sha256": _StringSubclass(CONFIG_SHA256)
                }
            ),
            _environment(),
        )


def test_rejects_extra_protected_cli_parameter_without_silent_projection() -> None:
    with pytest.raises(ValueError, match="parameter group differs.*extra"):
        select_representation_member(
            _cli_identity(
                parameter_updates={
                    "representation_startup_future_authority": "forbidden"
                }
            ),
            _environment(),
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("prepared_representation_launch_sha256", "4" * 64),
        ("python_executable_sha256", "5" * 64),
        ("stop_after_global_step", "7"),
        ("child_environment_ignored_host_names_sha256", "6" * 64),
    ],
)
def test_complete_cli_parameter_change_changes_selection_record(
    parameter: str,
    value: str,
) -> None:
    plan = _plan()
    original_identity = _cli_identity(plan)
    changed_identity = _cli_identity(plan, parameter_updates={parameter: value})
    original = select_representation_member(
        original_identity,
        _environment(plan, identity=original_identity),
    )
    changed = select_representation_member(
        changed_identity,
        _environment(plan, identity=changed_identity),
    )

    assert original.plan == changed.plan
    assert original.claim == changed.claim
    assert (
        original.as_record()["cli_gate_run_identity_sha256"]
        != (changed.as_record()["cli_gate_run_identity_sha256"])
    )
    assert original.as_record() != changed.as_record()


def test_raw_receipt_and_liveness_changes_bind_distinct_full_environments() -> None:
    plan = _plan()
    identity = _cli_identity(plan)
    original = select_representation_member(
        identity,
        _environment(plan, identity=identity),
    )
    changed = select_representation_member(
        identity,
        _environment(
            plan,
            identity=identity,
            updates={
                "TGVF_CLI_CONSUMPTION_RECEIPT_PATH": (
                    "/gate/consumptions/another-token.json"
                ),
                "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256": "7" * 64,
                "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH": (
                    "/gate/cli-launches/another-token/launcher-liveness.json"
                ),
            },
        ),
    )

    assert original.identity == changed.identity
    assert original.plan == changed.plan
    assert original.claim == changed.claim
    assert original.full_environment_sha256 != changed.full_environment_sha256
    assert original.as_record() != changed.as_record()


@pytest.mark.parametrize("mutation", ["pretty", "different-identity"])
def test_rejects_noncanonical_or_different_cli_identity_environment_json(
    mutation: str,
) -> None:
    plan = _plan()
    identity = _cli_identity(plan)
    if mutation == "pretty":
        changed_json = json.dumps(identity.as_record(), sort_keys=True)
    elif mutation == "different-identity":
        changed_json = _canonical_identity_json(
            _cli_identity(
                plan,
                parameter_updates={"stop_after_global_step": "7"},
            )
        )
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="environment identity JSON differs"):
        select_representation_member(
            identity,
            _environment(
                plan,
                identity=identity,
                updates={"TGVF_CLI_EXECUTION_IDENTITY_JSON": changed_json},
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("PYTHONPATH", "/tampered/runtime"),
        ("HF_HUB_DISABLE_TELEMETRY", "0"),
        ("UNDECLARED_BASE_FIELD", "forbidden"),
    ],
)
def test_existing_child_environment_verifier_rejects_base_or_extra_tamper(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="entry count differs|identity differs"):
        select_representation_member(
            _cli_identity(),
            _environment(updates={field: value}),
        )


def test_requires_exact_copied_environment_container_keys_and_values() -> None:
    class _DictSubclass(dict[str, str]):
        pass

    class _StringSubclass(str):
        pass

    environment = _environment()
    with pytest.raises(TypeError, match="exact copied dict"):
        select_representation_member(_cli_identity(), _DictSubclass(environment))

    subclass_key = dict(environment)
    value = subclass_key.pop("RANK")
    subclass_key[_StringSubclass("RANK")] = value
    with pytest.raises(TypeError, match="keys must be exactly str"):
        select_representation_member(_cli_identity(), subclass_key)

    subclass_value = dict(environment)
    subclass_value["RANK"] = _StringSubclass("0")
    with pytest.raises(TypeError, match="values must be exactly str"):
        select_representation_member(_cli_identity(), subclass_value)


def test_rejects_missing_selection_environment_field() -> None:
    environment = _environment()
    del environment["TORCHELASTIC_RUN_ID"]

    with pytest.raises(ValueError, match="missing fields.*TORCHELASTIC_RUN_ID"):
        select_representation_member(_cli_identity(), environment)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("WORLD_SIZE", "4", "WORLD_SIZE differs"),
        ("LOCAL_WORLD_SIZE", "4", "LOCAL_WORLD_SIZE differs"),
        ("ROLE_WORLD_SIZE", "4", "ROLE_WORLD_SIZE differs"),
        ("RANK", "2", "RANK is outside"),
        ("LOCAL_RANK", "1", "LOCAL_RANK differs"),
        ("ROLE_RANK", "1", "ROLE_RANK differs"),
        ("GROUP_RANK", "1", "GROUP_RANK must be exactly 0"),
        ("GROUP_WORLD_SIZE", "2", "GROUP_WORLD_SIZE must be exactly 1"),
        ("ROLE_NAME", "trainer", "ROLE_NAME must be exactly"),
        (
            "TORCHELASTIC_MAX_RESTARTS",
            "1",
            "TORCHELASTIC_MAX_RESTARTS must be exactly 0",
        ),
        (
            "TORCHELASTIC_RESTART_COUNT",
            "1",
            "TORCHELASTIC_RESTART_COUNT must be exactly 0",
        ),
        (
            "CUDA_VISIBLE_DEVICES",
            "7,4",
            "materialized child environment identity differs",
        ),
    ],
)
def test_rejects_rank_topology_restart_or_gpu_drift(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_representation_member(
            _cli_identity(),
            _environment(updates={field: value}),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("RANK", "00"),
        ("RANK", "+0"),
        ("RANK", "-0"),
        ("RANK", "٠"),
        ("WORLD_SIZE", "02"),
        ("TORCHELASTIC_RESTART_COUNT", "٠"),
    ],
)
def test_rejects_noncanonical_torchrun_integers(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="canonical non-negative ASCII decimal"):
        select_representation_member(
            _cli_identity(),
            _environment(updates={field: value}),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("TORCHELASTIC_RUN_ID", "", "non-empty canonical text"),
        ("TORCHELASTIC_RUN_ID", "run\nother", "non-empty canonical text"),
        (
            "TORCHELASTIC_RUN_ID",
            "12345678-1234-1abc-8def-1234567890ab",
            "canonical lowercase UUID4",
        ),
        (
            "TORCHELASTIC_RUN_ID",
            TORCHELASTIC_RUN_ID.upper(),
            "canonical lowercase UUID4",
        ),
        ("MASTER_ADDR", " localhost", "non-empty canonical text"),
        ("MASTER_PORT", "0", r"\[1, 65535\]"),
        ("MASTER_PORT", "65536", r"\[1, 65535\]"),
        ("MASTER_PORT", "029400", "canonical non-negative ASCII decimal"),
        (
            "TORCHELASTIC_USE_AGENT_STORE",
            "False",
            "must be exactly True",
        ),
        (
            "TORCHELASTIC_ERROR_FILE",
            "relative/error.json",
            "canonical lexical absolute",
        ),
        (
            "TORCHELASTIC_ERROR_FILE",
            "/tmp/../error.json",
            "canonical lexical absolute",
        ),
        (
            "TORCHELASTIC_ERROR_FILE",
            "/tmp/./error.json",
            "canonical lexical absolute",
        ),
        (
            "TORCHELASTIC_ERROR_FILE",
            "/",
            "canonical lexical absolute",
        ),
    ],
)
def test_rejects_malformed_dynamic_torchrun_receipt_fields(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        select_representation_member(
            _cli_identity(),
            _environment(updates={field: value}),
        )


def test_selection_is_frozen_but_explicitly_not_replay_evidence() -> None:
    selection = select_representation_member(_cli_identity(), _environment())

    with pytest.raises(FrozenInstanceError):
        selection.identity = _cli_identity()  # type: ignore[misc]
    changed_environment = dict(selection.full_environment)
    changed_environment["RANK"] = "1"
    with pytest.raises(ValueError, match="LOCAL_RANK differs"):
        replace(
            selection,
            full_environment=tuple(sorted(changed_environment.items())),
        )
    with pytest.raises(ValueError, match="claim differs"):
        replace(selection, claim=selection.plan.members[1])
    assert selection.replay_protected is False
    assert "Verified" not in type(selection).__name__


def test_direct_selection_constructor_revalidates_complete_identity_and_plan() -> None:
    selection = select_representation_member(_cli_identity(), _environment())
    wrong_phase = _cli_identity(phase="representation_internal_evaluation")
    with pytest.raises(ValueError, match="CLI phase differs"):
        replace(selection, identity=wrong_phase)

    changed_identity = _cli_identity(parameter_updates={"stop_after_global_step": "7"})
    with pytest.raises(ValueError, match="environment identity JSON differs"):
        replace(selection, identity=changed_identity)

    different_plan = _plan((0, 1))
    with pytest.raises(ValueError, match="plan differs from complete CLI identity"):
        replace(selection, plan=different_plan)


def test_pinned_torchrun_emits_selection_topology_contract(tmp_path: Path) -> None:
    if util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    assert metadata.version("torch").startswith("2.9")

    worker = tmp_path / "capture_selection_environment.py"
    worker.write_text(
        """\
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

names = (
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "ROLE_NAME",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_ERROR_FILE",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "WORLD_SIZE",
)
output = Path(sys.argv[1])
rank = os.environ["RANK"]
(output / f"rank-{rank}.json").write_text(
    json.dumps({name: os.environ[name] for name in names}, sort_keys=True),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    output = tmp_path / "workers"
    output.mkdir()
    parent_binding = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        owned_environment={
            "CUDA_VISIBLE_DEVICES": "",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        },
        host_environment={},
    )
    cli_overlay = {
        "TGVF_CLI_CONSUMPTION_RECEIPT_PATH": "/gate/consumptions/token.json",
        "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256": "3" * 64,
        "TGVF_CLI_EXECUTION_IDENTITY_JSON": "{}",
        "TGVF_CLI_GATE_DIRECTORY": "/gate",
        "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH": (
            "/gate/cli-launches/token/launcher-liveness.json"
        ),
        "TGVF_CLI_WORKER_AUTHORIZATION_SCHEMA": (
            "tgvf-cli-worker-authorization-environment-v1"
        ),
    }
    parent_environment = parent_binding.with_late_overlay(cli_overlay).as_environment()

    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker),
            str(output),
        ),
        cwd=tmp_path,
        env=parent_environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    workers = tuple(
        json.loads((output / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    )
    assert {worker_environment["RANK"] for worker_environment in workers} == {
        "0",
        "1",
    }
    assert {worker_environment["LOCAL_RANK"] for worker_environment in workers} == {
        "0",
        "1",
    }
    assert {worker_environment["ROLE_RANK"] for worker_environment in workers} == {
        "0",
        "1",
    }
    for worker_environment in workers:
        assert worker_environment["WORLD_SIZE"] == "2"
        assert worker_environment["LOCAL_WORLD_SIZE"] == "2"
        assert worker_environment["ROLE_WORLD_SIZE"] == "2"
        assert worker_environment["GROUP_RANK"] == "0"
        assert worker_environment["GROUP_WORLD_SIZE"] == "1"
        assert worker_environment["ROLE_NAME"] == "default"
        assert worker_environment["TORCHELASTIC_RESTART_COUNT"] == "0"
        assert worker_environment["TORCHELASTIC_MAX_RESTARTS"] == "0"
        assert worker_environment["TORCHELASTIC_USE_AGENT_STORE"] == "True"
        run_id = worker_environment["TORCHELASTIC_RUN_ID"]
        assert UUID(run_id).version == 4
        assert str(UUID(run_id)) == run_id
    for name in ("MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
        assert workers[0][name] == workers[1][name]
    assert (
        workers[0]["TORCHELASTIC_ERROR_FILE"] != workers[1]["TORCHELASTIC_ERROR_FILE"]
    )


def test_member_selection_leaf_has_isolated_python_firebreak() -> None:
    script = """
import sys
import tgvf_rl.ops.representation_member_selection as selection
assert selection.REPRESENTATION_MEMBER_SELECTION_SCHEMA
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
