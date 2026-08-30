from __future__ import annotations

import copy
import multiprocessing
import os
from pathlib import Path
import pickle

import pytest

import tgvf_rl.secure_file_read as secure_file_read
import tgvf_rl.ops.representation_member_consumption as consumption_implementation
from tgvf_rl.ops.cli_authorization import (
    REPOSITORY_EXECUTION_POLICY_PATH,
    cli_worker_authorization_environment,
    consume_cli_execution_authorization,
    materialize_cli_worker_authorization,
)
from tgvf_rl.ops.cli_authorization_identity import (
    CLIExecutionAuthorizationIdentity,
)
from tgvf_rl.ops.child_environment import (
    REPRESENTATION_TORCHRUN_PROFILE,
    build_child_environment,
)
from tgvf_rl.ops.launch_gate import (
    issue_freeze_override,
    issue_launch_authorization,
    materialize_ready_receipt,
)
from tgvf_rl.ops.representation_member_consumption import (
    REPRESENTATION_MEMBER_CONSUMPTION_SCHEMA,
    ConsumedRepresentationMemberAuthorization,
    RepresentationMemberConsumptionError,
    consume_representation_member_selection,
)
from tgvf_rl.ops.representation_member_selection import (
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


RUN_SHA256 = "a" * 64
CONFIG_SHA256 = "b" * 64
TORCHELASTIC_RUN_ID = "12345678-1234-4abc-8def-1234567890ab"

pytestmark = pytest.mark.filterwarnings(
    r"ignore:This process .* is multi-threaded, use of fork\(\) may lead to "
    r"deadlocks in the child\.:DeprecationWarning"
)


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
    else:  # pragma: no cover
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


def _plan() -> RepresentationStartupPlan:
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
        physical_gpu_ids=(4, 7),
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


def _identity(plan: RepresentationStartupPlan) -> CLIExecutionAuthorizationIdentity:
    child_environment = _child_environment_binding(plan)
    return CLIExecutionAuthorizationIdentity.create(
        run_id="REPRESENTATION-CONSUMPTION-TEST",
        phase=REPRESENTATION_TRAINING_PHASE,
        command_id=REPRESENTATION_TRAINING_COMMAND_ID,
        run_identity_sha256=plan.run_identity_sha256,
        parameters={
            "canonical_config_sha256": plan.config_identity_sha256,
            "config_source_sha256": plan.config_identity_sha256,
            "nproc_per_node": str(plan.world_size),
            "prepared_representation_launch_sha256": "1" * 64,
            "python_executable_sha256": "2" * 64,
            "stop_after_global_step": "none",
            "world_size": str(plan.world_size),
            **child_environment.authorization_parameters(),
            **plan.authorization_parameters(),
        },
    )


def _authorized_selections(
    tmp_path: Path,
) -> tuple[tuple[RepresentationMemberSelection, ...], Path, str]:
    plan = _plan()
    identity = _identity(plan)
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
        reason="representation member receipt CPU contract test",
        ttl_seconds=300,
        authorized_by="test-operator",
    )
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
    token_id = consumption["token_id"]
    assert isinstance(token_id, str)
    token_directory = gate / "cli-launches" / token_id
    os.chmod(token_directory, 0o700)
    receipt_directory = token_directory / "representation-members"
    receipt_directory.mkdir(mode=0o700)
    os.chmod(receipt_directory, 0o700)

    cli_overlay = cli_worker_authorization_environment(
        identity,
        worker,
        gate_directory=gate,
    )
    selections: list[RepresentationMemberSelection] = []
    for rank in range(plan.world_size):
        environment = (
            _child_environment_binding(plan)
            .with_late_overlay(cli_overlay)
            .as_environment()
        )
        environment.update(
            {
                "GROUP_RANK": "0",
                "GROUP_WORLD_SIZE": "1",
                "LOCAL_RANK": str(rank),
                "LOCAL_WORLD_SIZE": str(plan.world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29400",
                "RANK": str(rank),
                "ROLE_NAME": "default",
                "ROLE_RANK": str(rank),
                "ROLE_WORLD_SIZE": str(plan.world_size),
                "TORCHELASTIC_ERROR_FILE": (
                    f"/tmp/torchelastic/rank-{rank}/error.json"
                ),
                "TORCHELASTIC_MAX_RESTARTS": "0",
                "TORCHELASTIC_RESTART_COUNT": "0",
                "TORCHELASTIC_RUN_ID": TORCHELASTIC_RUN_ID,
                "TORCHELASTIC_USE_AGENT_STORE": "True",
                "WORLD_SIZE": str(plan.world_size),
            }
        )
        selections.append(select_representation_member(identity, environment))
    return tuple(selections), receipt_directory, token_id


def _concurrent_consume(
    selection: RepresentationMemberSelection,
    start: object,
    results: object,
) -> None:
    start.wait()  # type: ignore[attr-defined]
    try:
        authorization = consume_representation_member_selection(selection)
    except Exception as error:
        results.put(("error", type(error).__name__, str(error)))  # type: ignore[attr-defined]
    else:
        results.put(  # type: ignore[attr-defined]
            (
                "ok",
                authorization.selection.claim.global_rank,
                str(authorization.receipt_path),
            )
        )


def _run_concurrent(
    selections: tuple[RepresentationMemberSelection, ...],
) -> list[tuple[object, ...]]:
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_concurrent_consume, args=(item, start, results))
        for item in selections
    ]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    return observed


def test_consumes_one_rank_with_exact_cooperative_receipt(tmp_path: Path) -> None:
    selections, receipt_directory, token_id = _authorized_selections(tmp_path)

    authorization = consume_representation_member_selection(selections[1])

    assert type(authorization) is ConsumedRepresentationMemberAuthorization
    assert authorization.selection == selections[1]
    assert authorization.replay_protected is True
    assert authorization.receipt_path == receipt_directory / "rank-1.json"
    assert authorization.receipt_path.stat().st_mode & 0o777 == 0o600
    assert not hasattr(authorization, "authorization_parameters")
    record = authorization.receipt_record()
    assert record["schema"] == REPRESENTATION_MEMBER_CONSUMPTION_SCHEMA
    assert record["status"] == "consumed"
    assert record["security_model"] == "cooperative-same-uid-v1"
    assert record["hostile_same_uid_protected"] is False
    assert record["retry_policy"] == "new-launch-token-required"
    assert record["token_id"] == token_id
    assert record["claim"] == selections[1].claim.as_record()
    assert record["selection_sha256"]
    assert record["full_environment_sha256"] == (selections[1].full_environment_sha256)
    assert record["token_directory_identity"]["mode"] == "0700"  # type: ignore[index]
    assert record["receipt_directory_identity"]["mode"] == "0700"  # type: ignore[index]


def test_same_rank_concurrency_allows_exactly_one_consumer(tmp_path: Path) -> None:
    selections, receipt_directory, _ = _authorized_selections(tmp_path)

    observed = _run_concurrent((selections[0],) * 8)

    assert sum(item[0] == "ok" for item in observed) == 1
    assert sum(item[0] == "error" for item in observed) == 7
    assert all(
        item[0] == "ok" or "already consumed" in str(item[2]) for item in observed
    )
    assert tuple(path.name for path in receipt_directory.iterdir()) == ("rank-0.json",)


def test_distinct_ranks_consume_concurrently_without_global_lock(
    tmp_path: Path,
) -> None:
    selections, receipt_directory, _ = _authorized_selections(tmp_path)

    observed = _run_concurrent(selections)

    assert sorted(item[1] for item in observed if item[0] == "ok") == [0, 1]
    assert sorted(path.name for path in receipt_directory.iterdir()) == [
        "rank-0.json",
        "rank-1.json",
    ]


def test_same_rank_can_be_consumed_once_under_each_distinct_outer_token(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, _, first_token = _authorized_selections(first_root)
    second, _, second_token = _authorized_selections(second_root)

    first_authorization = consume_representation_member_selection(first[0])
    second_authorization = consume_representation_member_selection(second[0])

    assert first_token != second_token
    assert first_authorization.receipt_path != second_authorization.receipt_path
    assert first_authorization.replay_protected is True
    assert second_authorization.replay_protected is True


def test_receipt_write_failure_burns_rank_without_same_token_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selections, receipt_directory, _ = _authorized_selections(tmp_path)
    original_write_all = secure_file_read._write_all

    def _fail_after_reservation(_descriptor: int, _payload: bytes) -> None:
        raise OSError("injected receipt write failure")

    monkeypatch.setattr(secure_file_read, "_write_all", _fail_after_reservation)
    with pytest.raises(
        RepresentationMemberConsumptionError, match="permanently burned"
    ):
        consume_representation_member_selection(selections[0])
    monkeypatch.setattr(secure_file_read, "_write_all", original_write_all)

    tombstone = receipt_directory / "rank-0.json"
    assert tombstone.exists()
    assert tombstone.read_bytes() == b""
    with pytest.raises(RepresentationMemberConsumptionError, match="already consumed"):
        consume_representation_member_selection(selections[0])


def test_existing_rank_leaf_or_symlink_is_a_permanent_replay_refusal(
    tmp_path: Path,
) -> None:
    selections, receipt_directory, _ = _authorized_selections(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("untouched", encoding="utf-8")
    (receipt_directory / "rank-0.json").symlink_to(outside)

    with pytest.raises(RepresentationMemberConsumptionError, match="already consumed"):
        consume_representation_member_selection(selections[0])

    assert outside.read_text(encoding="utf-8") == "untouched"


def test_receipt_directory_symlink_is_rejected_without_writing_target(
    tmp_path: Path,
) -> None:
    selections, receipt_directory, _ = _authorized_selections(tmp_path)
    token_directory = receipt_directory.parent
    receipt_directory.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    receipt_directory.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RepresentationMemberConsumptionError, match="creation failed"):
        consume_representation_member_selection(selections[0])

    assert not (outside / "rank-0.json").exists()
    assert token_directory.stat().st_mode & 0o777 == 0o700


def test_receipt_directory_inode_swap_is_detected_after_exclusive_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selections, receipt_directory, _ = _authorized_selections(tmp_path)
    original_create = (
        consumption_implementation.create_regular_file_exclusive_beneath_nofollow
    )
    moved = receipt_directory.with_name("moved-representation-members")

    def _create_then_swap(*args: object, **kwargs: object):
        creation = original_create(*args, **kwargs)  # type: ignore[arg-type]
        receipt_directory.rename(moved)
        receipt_directory.mkdir(mode=0o700)
        return creation

    monkeypatch.setattr(
        consumption_implementation,
        "create_regular_file_exclusive_beneath_nofollow",
        _create_then_swap,
    )

    with pytest.raises(
        RepresentationMemberConsumptionError,
        match="receipt directory binding changed",
    ):
        consume_representation_member_selection(selections[0])

    assert (moved / "rank-0.json").is_file()
    assert not (receipt_directory / "rank-0.json").exists()


def test_consumed_authorization_rejects_tamper_copy_mutation_and_subclass(
    tmp_path: Path,
) -> None:
    selections, _, _ = _authorized_selections(tmp_path)
    authorization = consume_representation_member_selection(selections[0])

    with pytest.raises(AttributeError, match="immutable"):
        authorization._receipt_sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(authorization)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(authorization)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(authorization)
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class _Subclass(ConsumedRepresentationMemberAuthorization):
            pass

    authorization.receipt_path.write_bytes(b'{"tampered":true}\n')
    with pytest.raises(RepresentationMemberConsumptionError, match="bytes changed"):
        authorization.assert_current_process_and_receipt()


def test_consumed_authorization_does_not_retain_mutable_selection_reference(
    tmp_path: Path,
) -> None:
    selections, _, _ = _authorized_selections(tmp_path)
    authorization = consume_representation_member_selection(selections[0])
    leaked_snapshot = authorization.selection

    object.__setattr__(leaked_snapshot, "claim", selections[1].claim)
    object.__setattr__(selections[0], "claim", selections[1].claim)

    assert authorization.selection.claim.global_rank == 0
    assert authorization.replay_protected is True


def test_consumed_authorization_cross_checks_internal_snapshot_against_receipt(
    tmp_path: Path,
) -> None:
    selections, _, _ = _authorized_selections(tmp_path)
    authorization = consume_representation_member_selection(selections[0])

    object.__setattr__(authorization, "_selection_rank", 1)

    with pytest.raises(
        RepresentationMemberConsumptionError,
        match="canonical selection",
    ):
        authorization.assert_current_process_and_receipt()


def test_consumed_authorization_is_rejected_after_fork(tmp_path: Path) -> None:
    selections, _, _ = _authorized_selections(tmp_path)
    authorization = consume_representation_member_selection(selections[0])
    reader, writer = os.pipe()
    child = os.fork()
    if child == 0:  # pragma: no cover - assertion observed by parent
        os.close(reader)
        try:
            authorization.assert_current_process_and_receipt()
        except RepresentationMemberConsumptionError as error:
            os.write(writer, str(error).encode("utf-8"))
            os._exit(0)
        os._exit(1)

    os.close(writer)
    message = os.read(reader, 4096).decode("utf-8")
    os.close(reader)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert "process-local" in message


def test_fork_cannot_rebind_internal_pid_fields_away_from_receipt(
    tmp_path: Path,
) -> None:
    selections, _, _ = _authorized_selections(tmp_path)
    authorization = consume_representation_member_selection(selections[0])
    reader, writer = os.pipe()
    child = os.fork()
    if child == 0:  # pragma: no cover - assertion observed by parent
        os.close(reader)
        child_pid = os.getpid()
        object.__setattr__(authorization, "_worker_pid", child_pid)
        object.__setattr__(
            authorization,
            "_worker_process_start_ticks",
            consumption_implementation._process_start_ticks(child_pid),
        )
        try:
            authorization.assert_current_process_and_receipt()
        except RepresentationMemberConsumptionError as error:
            os.write(writer, str(error).encode("utf-8"))
            os._exit(0)
        os._exit(1)

    os.close(writer)
    message = os.read(reader, 4096).decode("utf-8")
    os.close(reader)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert "receipt worker process identity differs" in message
