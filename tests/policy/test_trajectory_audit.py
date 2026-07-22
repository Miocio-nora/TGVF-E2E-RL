from __future__ import annotations

import json
from types import SimpleNamespace

from tests.framework.test_verl_bridges import _record
from tests.rewards.test_verl_adapter import _scorer

from tgvf_rl.policy.trajectory_audit import (
    POLICY_TRAJECTORY_AUDIT_SCHEMA,
    PolicyTrajectoryAuditWriter,
)


def test_writer_retains_human_readable_selected_trajectory(tmp_path) -> None:
    trajectory = _record(tool_call_count=0).trajectory_payload
    scorer, _judge = _scorer()
    reward = scorer.score(
        request=SimpleNamespace(identity=trajectory.identity),
        trajectory=trajectory,
    )

    writer = PolicyTrajectoryAuditWriter(tmp_path)
    writer.record(trajectory, reward)
    writer.record(trajectory, reward)

    paths = list(tmp_path.glob("step-*/*.json"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == POLICY_TRAJECTORY_AUDIT_SCHEMA
    assert payload["selection_reasons"] == [
        "representative_rollout_zero",
        "correct_answer",
    ]
    assert payload["candidate_answer"] == "fixture answer"
    assert payload["expected_answer"] == "fixture answer"
    assert payload["assistant_turns"][0]["raw_text"] == "reason</think>answer"
    assert payload["reward_components"]["answer_reward"] == 1.0
