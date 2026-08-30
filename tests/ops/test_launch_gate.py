from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil

import pytest

from tgvf_rl.ops.launch_gate import (
    LaunchAuthorizationError,
    LaunchGateError,
    LaunchLivenessError,
    LaunchTimeoutError,
    assert_process_liveness,
    consume_launch_authorization,
    issue_freeze_override,
    issue_launch_authorization,
    make_run_identity,
    materialize_ready_receipt,
    wait_for_artifact,
    write_process_liveness_receipt,
)


NOW = datetime(2026, 8, 30, 0, 0, tzinfo=timezone.utc)


def _identity(*, run_id: str = "RUN-1", phase: str = "evaluate") -> dict[str, object]:
    return make_run_identity(
        run_id=run_id,
        phase=phase,
        command_id="tests:launch",
        parameters={"contract_sha256": "a" * 64},
    )


def _policy(path: Path, *, mode: str = "frozen") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "tgvf-experiment-execution-policy-v2",
                "policy_id": "TEST-POLICY-V1",
                "revision": 1,
                "execution_mode": mode,
                "reason": "unit test execution policy",
                "freeze_override": {
                    "required_when_frozen": True,
                    "max_ttl_seconds": 600,
                    "reason_required": True,
                },
                "runtime_closure": {
                    "launch_enabled": True,
                    "blocker_ids": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _ready(tmp_path: Path) -> tuple[Path, Path]:
    evidence = tmp_path / "completion.json"
    evidence.write_text('{"status":"complete"}\n', encoding="utf-8")
    gate = tmp_path / "gate"
    materialize_ready_receipt(
        gate,
        run_identity=_identity(),
        evidence_paths={"completion": evidence},
        now=NOW,
    )
    return gate, evidence


def test_ready_receipt_is_idempotent_but_identity_bound(tmp_path: Path) -> None:
    gate, evidence = _ready(tmp_path)
    first = json.loads((gate / "ready.json").read_text(encoding="utf-8"))
    second = materialize_ready_receipt(
        gate,
        run_identity=_identity(),
        evidence_paths={"completion": evidence},
        now=NOW + timedelta(hours=1),
    )
    assert second == first
    with pytest.raises(LaunchGateError, match="different run identity"):
        materialize_ready_receipt(
            gate,
            run_identity=_identity(run_id="OTHER"),
            evidence_paths={"completion": evidence},
            now=NOW,
        )


def test_changed_ready_evidence_blocks_authorization(tmp_path: Path) -> None:
    gate, evidence = _ready(tmp_path)
    evidence.write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(LaunchAuthorizationError, match="evidence changed"):
        issue_launch_authorization(
            gate, ttl_seconds=300, authorized_by="operator", now=NOW
        )


def test_frozen_policy_rejects_plain_per_run_token(tmp_path: Path) -> None:
    gate, _ = _ready(tmp_path)
    policy = _policy(tmp_path / "policy.json")
    token_path, _ = issue_launch_authorization(
        gate, ttl_seconds=300, authorized_by="operator", now=NOW
    )
    with pytest.raises(LaunchAuthorizationError, match="execution is frozen"):
        consume_launch_authorization(
            gate,
            token_path,
            policy,
            expected_run_id="RUN-1",
            expected_phase="evaluate",
            now=NOW + timedelta(seconds=1),
        )
    assert not list((gate / "consumptions").glob("*.json"))


def test_frozen_launch_requires_and_consumes_two_bound_authorizations(
    tmp_path: Path,
) -> None:
    gate, _ = _ready(tmp_path)
    policy = _policy(tmp_path / "policy.json")
    token_path, token = issue_launch_authorization(
        gate, ttl_seconds=300, authorized_by="operator", now=NOW
    )
    override_path, override = issue_freeze_override(
        gate,
        policy,
        reason="approved stabilization smoke",
        ttl_seconds=120,
        authorized_by="operator",
        now=NOW,
    )
    consumption = consume_launch_authorization(
        gate,
        token_path,
        policy,
        expected_run_id="RUN-1",
        expected_phase="evaluate",
        freeze_override_path=override_path,
        consumed_by="controller",
        now=NOW + timedelta(seconds=1),
    )
    assert consumption["token_id"] == token["token_id"]
    assert consumption["freeze_override_id"] == override["override_id"]
    assert consumption["execution_mode"] == "frozen"
    assert Path(consumption["consumption_path"]).is_file()
    assert (
        gate / "freeze-override-consumptions" / f"{override['override_id']}.json"
    ).is_file()
    with pytest.raises(LaunchAuthorizationError, match="already consumed"):
        consume_launch_authorization(
            gate,
            token_path,
            policy,
            expected_run_id="RUN-1",
            expected_phase="evaluate",
            freeze_override_path=override_path,
            now=NOW + timedelta(seconds=2),
        )
    replacement_token, _ = issue_launch_authorization(
        gate, ttl_seconds=300, authorized_by="operator", now=NOW
    )
    with pytest.raises(LaunchAuthorizationError, match="freeze override .* consumed"):
        consume_launch_authorization(
            gate,
            replacement_token,
            policy,
            expected_run_id="RUN-1",
            expected_phase="evaluate",
            freeze_override_path=override_path,
            now=NOW + timedelta(seconds=2),
        )


def test_copied_token_cannot_be_replayed(tmp_path: Path) -> None:
    gate, _ = _ready(tmp_path)
    policy = _policy(tmp_path / "policy.json", mode="open")
    token_path, _ = issue_launch_authorization(
        gate, ttl_seconds=300, authorized_by="operator", now=NOW
    )
    copied = tmp_path / "copied-token.json"
    shutil.copyfile(token_path, copied)
    consume_launch_authorization(
        gate,
        copied,
        policy,
        expected_run_id="RUN-1",
        expected_phase="evaluate",
        now=NOW + timedelta(seconds=1),
    )
    with pytest.raises(LaunchAuthorizationError, match="already consumed"):
        consume_launch_authorization(
            gate,
            token_path,
            policy,
            expected_run_id="RUN-1",
            expected_phase="evaluate",
            now=NOW + timedelta(seconds=2),
        )


def test_expired_token_and_wrong_run_are_rejected(tmp_path: Path) -> None:
    gate, _ = _ready(tmp_path)
    policy = _policy(tmp_path / "policy.json", mode="open")
    wrong_token, _ = issue_launch_authorization(
        gate, ttl_seconds=300, authorized_by="operator", now=NOW
    )
    with pytest.raises(LaunchAuthorizationError, match="expected run identity"):
        consume_launch_authorization(
            gate,
            wrong_token,
            policy,
            expected_run_id="WRONG",
            expected_phase="evaluate",
            now=NOW + timedelta(seconds=1),
        )
    expired_token, _ = issue_launch_authorization(
        gate, ttl_seconds=1, authorized_by="operator", now=NOW
    )
    with pytest.raises(LaunchAuthorizationError, match="expired"):
        consume_launch_authorization(
            gate,
            expired_token,
            policy,
            expected_run_id="RUN-1",
            expected_phase="evaluate",
            now=NOW + timedelta(seconds=2),
        )


def test_policy_change_invalidates_freeze_override(tmp_path: Path) -> None:
    gate, _ = _ready(tmp_path)
    policy = _policy(tmp_path / "policy.json")
    token_path, _ = issue_launch_authorization(
        gate, ttl_seconds=300, authorized_by="operator", now=NOW
    )
    override_path, _ = issue_freeze_override(
        gate,
        policy,
        reason="approved test",
        ttl_seconds=120,
        authorized_by="operator",
        now=NOW,
    )
    value = json.loads(policy.read_text(encoding="utf-8"))
    value["revision"] = 2
    policy.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(LaunchAuthorizationError, match="current policy"):
        consume_launch_authorization(
            gate,
            token_path,
            policy,
            expected_run_id="RUN-1",
            expected_phase="evaluate",
            freeze_override_path=override_path,
            now=NOW + timedelta(seconds=1),
        )


def test_liveness_receipt_binds_pid_start_time(tmp_path: Path) -> None:
    receipt_path = tmp_path / "liveness.json"
    written = write_process_liveness_receipt(
        receipt_path, run_identity=_identity(phase="training"), now=NOW
    )
    observed = assert_process_liveness(
        receipt_path, expected_run_id="RUN-1", expected_phase="training"
    )
    assert observed == written
    tampered = dict(written)
    tampered["process_start_ticks"] += 1
    receipt_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(LaunchLivenessError, match="replaced"):
        assert_process_liveness(receipt_path)


def test_bounded_wait_succeeds_or_times_out_without_unbounded_sleep(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    assert (
        wait_for_artifact(artifact, timeout_seconds=1, poll_seconds=0.1)
        == artifact.absolute()
    )

    clock = iter((0.0, 0.0, 1.0))
    sleeps: list[float] = []
    with pytest.raises(LaunchTimeoutError, match="within 1s"):
        wait_for_artifact(
            tmp_path / "missing.json",
            timeout_seconds=1,
            poll_seconds=0.5,
            monotonic=lambda: next(clock),
            sleep=sleeps.append,
        )
    assert sleeps == [0.5]


def test_token_file_is_owner_only(tmp_path: Path) -> None:
    gate, _ = _ready(tmp_path)
    token_path, _ = issue_launch_authorization(
        gate, ttl_seconds=300, authorized_by="operator", now=NOW
    )
    assert token_path.stat().st_mode & 0o777 == 0o600
    assert os.access(token_path, os.R_OK)
