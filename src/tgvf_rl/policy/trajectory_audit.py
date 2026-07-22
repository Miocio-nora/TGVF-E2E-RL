"""Bounded, local-only human-readable Policy RL trajectory audit records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from tgvf_rl.rewards.verl_adapter import PilotVerlTrajectoryReward
from tgvf_rl.trajectories.schema import (
    CropTGVFToolCallRecord,
    CropToolCallRecord,
    ToolCallRecord,
    TrajectoryRecord,
    TrajectoryStop,
    trajectory_checksum,
)


POLICY_TRAJECTORY_AUDIT_SCHEMA = "policy-trajectory-audit-v1"


@dataclass(frozen=True, slots=True)
class PolicyTrajectoryAuditWriter:
    """Retain a bounded diagnostic subset without sending text to W&B."""

    root: Path

    def record(
        self,
        trajectory: TrajectoryRecord,
        reward: PilotVerlTrajectoryReward,
    ) -> None:
        reasons = _selection_reasons(trajectory, reward)
        if not reasons:
            return
        step = trajectory.behavior_policy.optimizer_step
        step_dir = self.root / f"step-{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(
            trajectory.identity.canonical_id.encode("utf-8")
        ).hexdigest()
        path = step_dir / f"{name}.json"
        payload = _audit_payload(trajectory, reward, reasons=reasons)
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != serialized:
                raise RuntimeError("trajectory audit path contains different content")


def _selection_reasons(
    trajectory: TrajectoryRecord,
    reward: PilotVerlTrajectoryReward,
) -> tuple[str, ...]:
    components = dict(reward.raw_components)
    reasons: list[str] = []
    if trajectory.identity.rollout_index == 0:
        reasons.append("representative_rollout_zero")
    if components["answer_reward"] == 1.0:
        reasons.append("correct_answer")
    if components["format_reward"] == -1.0:
        reasons.append("format_error")
    if trajectory.stop is TrajectoryStop.MAX_TOKENS:
        reasons.append("max_tokens")
    return tuple(reasons)


def _audit_payload(
    trajectory: TrajectoryRecord,
    reward: PilotVerlTrajectoryReward,
    *,
    reasons: tuple[str, ...],
) -> dict[str, object]:
    context = reward.context
    return {
        "schema_version": POLICY_TRAJECTORY_AUDIT_SCHEMA,
        "selection_reasons": reasons,
        "optimizer_step": trajectory.behavior_policy.optimizer_step,
        "trajectory_id": trajectory.identity.canonical_id,
        "trajectory_sha256": trajectory_checksum(trajectory),
        "sample_id": trajectory.identity.sample_id,
        "group_uid": trajectory.identity.group_id,
        "rollout_index": trajectory.identity.rollout_index,
        "question": context.question,
        "expected_answer": context.expected_answer,
        "candidate_answer": context.candidate_answer,
        "stop": trajectory.stop.value,
        "protocol_valid": context.protocol_valid,
        "has_valid_final_answer": context.has_valid_final_answer,
        "reward_total": reward.total,
        "reward_components": dict(reward.raw_components),
        "answer_verifier_evidence": reward.result.components[0].evidence,
        "assistant_turns": [
            {
                "turn_index": turn.turn_index,
                "raw_text": turn.raw_text,
                "sampled_token_count": len(turn.tokens.token_ids),
                "is_tool_call": turn.is_tool_call,
                "has_think_span": turn.think_span is not None,
                "stop_reason": turn.stop_reason,
            }
            for turn in trajectory.assistant_turns
        ],
        "tool_calls": [_tool_call_payload(call) for call in trajectory.tool_calls],
        "tool_errors": [
            {
                "attempt_index": error.attempt_index,
                "assistant_turn_index": error.assistant_turn_index,
                "function_name": error.function_name,
                "code": error.code,
                "payload_json": error.payload_json,
                "recoverable": error.recoverable,
            }
            for error in trajectory.tool_errors
        ],
        "successful_observation_count": len(trajectory.observations),
    }


def _tool_call_payload(call: object) -> dict[str, object]:
    common: dict[str, object]
    if isinstance(call, ToolCallRecord):
        common = {"target": call.target}
    elif isinstance(call, CropToolCallRecord):
        common = {"bbox_2d": call.bbox_2d, "label": call.label}
    elif isinstance(call, CropTGVFToolCallRecord):
        common = {"bbox_2d": call.bbox_2d, "target": call.target}
    else:  # pragma: no cover - TrajectoryRecord owns the closed union
        raise TypeError("unsupported native tool-call record")
    return {
        "call_index": call.call_index,
        "assistant_turn_index": call.assistant_turn_index,
        "function_name": call.function_name,
        "raw_call_text": call.raw_call_text,
        **common,
    }


__all__ = [
    "POLICY_TRAJECTORY_AUDIT_SCHEMA",
    "PolicyTrajectoryAuditWriter",
]
