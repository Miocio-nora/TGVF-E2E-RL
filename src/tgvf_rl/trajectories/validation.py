"""Fail-closed invariants for actual-logprob multi-call trajectories."""

from __future__ import annotations

import hashlib

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.tokens import TokenOwnership
from tgvf_rl.observations.store import ObservationStore
from tgvf_rl.observations.schema import (
    CropObservationRecord,
    CropTGVFObservationRecord,
    FocusedObservationRecord,
)
from tgvf_rl.protocol.schema import ToolErrorCode

from .behavior import BehaviorTraceStore
from .schema import (
    CropToolCallRecord,
    CropTGVFToolCallRecord,
    TrajectoryBatch,
    TrajectoryRecord,
    TrajectoryStop,
)


class TrajectoryValidator:
    def __init__(
        self,
        observation_store: ObservationStore,
        behavior_store: BehaviorTraceStore,
        *,
        maximum_policy_staleness: int = 0,
    ) -> None:
        if maximum_policy_staleness < 0:
            raise ValueError("maximum_policy_staleness must be non-negative")
        self.store = observation_store
        self.behavior_store = behavior_store
        self.maximum_policy_staleness = maximum_policy_staleness

    def validate(self, trajectory: TrajectoryRecord) -> None:
        if tuple(turn.turn_index for turn in trajectory.assistant_turns) != tuple(
            range(len(trajectory.assistant_turns))
        ):
            raise ReplayMismatchError("assistant turns must have contiguous indices")
        call_indices = tuple(call.call_index for call in trajectory.tool_calls)
        observation_indices = tuple(item.call_index for item in trajectory.observations)
        if call_indices != tuple(range(len(call_indices))):
            raise ReplayMismatchError("tool calls must have contiguous call indices")
        if observation_indices != call_indices:
            raise ReplayMismatchError(
                "every valid tool call must have one ordered observation"
            )
        attempts = tuple(
            sorted(
                (
                    (call.attempt_index, call.assistant_turn_index, "success")
                    for call in trajectory.tool_calls
                ),
            )
        ) + tuple(
            sorted(
                (
                    (error.attempt_index, error.assistant_turn_index, "error")
                    for error in trajectory.tool_errors
                ),
            )
        )
        attempts = tuple(sorted(attempts, key=lambda item: item[0]))
        attempt_indices = tuple(item[0] for item in attempts)
        if any(index is None for index in attempt_indices):
            raise ReplayMismatchError("tool attempt identity must be materialized")
        if attempt_indices != tuple(range(len(attempt_indices))):
            raise ReplayMismatchError("tool attempts must have contiguous indices")
        attempt_turns = tuple(item[1] for item in attempts)
        if len(set(attempt_turns)) != len(attempt_turns):
            raise ReplayMismatchError(
                "each assistant action turn may produce at most one tool event"
            )
        if attempt_turns != tuple(sorted(attempt_turns)):
            raise ReplayMismatchError("tool attempts must follow assistant-turn order")
        representation = None
        for item, call in zip(
            trajectory.observations, trajectory.tool_calls, strict=True
        ):
            record = self.store.resolve_record(item.handle)
            if record.call_index != item.call_index:
                raise ReplayMismatchError(
                    "observation call index differs from trajectory"
                )
            if record.model != trajectory.model:
                raise IdentityMismatchError(
                    "observation model differs from trajectory model"
                )
            if isinstance(call, CropToolCallRecord):
                if not isinstance(record, CropObservationRecord):
                    raise IdentityMismatchError(
                        "crop call received a non-crop observation"
                    )
                if record.policy_version != trajectory.behavior_policy:
                    raise IdentityMismatchError(
                        "crop materialization policy differs from behavior policy"
                    )
                if record.trajectory_id != trajectory.identity.canonical_id:
                    raise IdentityMismatchError(
                        "crop observation differs from trajectory identity"
                    )
                if record.requested_bbox_2d != call.bbox_2d:
                    raise IdentityMismatchError(
                        "crop observation bbox differs from sampled tool call"
                    )
            elif isinstance(call, CropTGVFToolCallRecord):
                if not isinstance(record, CropTGVFObservationRecord):
                    raise IdentityMismatchError(
                        "atomic crop+TGVF call received a different observation type"
                    )
                if record.condition.policy_version != trajectory.behavior_policy:
                    raise IdentityMismatchError(
                        "atomic crop+TGVF policy differs from behavior policy"
                    )
                if record.condition.trajectory_ids != (
                    trajectory.identity.canonical_id,
                ):
                    raise IdentityMismatchError(
                        "atomic crop+TGVF observation differs from trajectory"
                    )
                if record.requested_bbox_2d != call.bbox_2d:
                    raise IdentityMismatchError(
                        "atomic crop+TGVF bbox differs from sampled tool call"
                    )
                if record.sampled_target_char_span != call.target_char_span:
                    raise IdentityMismatchError(
                        "atomic crop+TGVF target char span differs from sampled tool call"
                    )
                expected_target_sha = hashlib.sha256(
                    call.target.encode("utf-8")
                ).hexdigest()
                if record.condition.sampled_target_text_sha256 != expected_target_sha:
                    raise IdentityMismatchError(
                        "atomic crop+TGVF target differs from sampled tool call"
                    )
                if (
                    record.condition.sampled_target_token_start,
                    record.condition.sampled_target_token_end,
                ) != (call.target_token_span.start, call.target_token_span.end):
                    raise IdentityMismatchError(
                        "atomic crop+TGVF target span differs from sampled tool call"
                    )
                if representation is None:
                    representation = record.representation
                elif record.representation != representation:
                    raise IdentityMismatchError(
                        "representation artifact changed within one trajectory"
                    )
            else:
                if not isinstance(record, FocusedObservationRecord):
                    raise IdentityMismatchError(
                        "TGVF call received a non-TGVF observation"
                    )
                if record.condition.policy_version != trajectory.behavior_policy:
                    raise IdentityMismatchError(
                        "observation materialization policy differs from behavior policy"
                    )
                if record.condition.trajectory_ids != (
                    trajectory.identity.canonical_id,
                ):
                    raise IdentityMismatchError(
                        "observation provenance differs from trajectory identity"
                    )
                if record.condition.call_indices != (call.call_index,):
                    raise IdentityMismatchError(
                        "observation provenance differs from trajectory call"
                    )
                expected_target_sha = hashlib.sha256(
                    call.target.encode("utf-8")
                ).hexdigest()
                if record.condition.sampled_target_text_sha256 != expected_target_sha:
                    raise IdentityMismatchError(
                        "observation target differs from sampled tool target"
                    )
                if (
                    record.condition.sampled_target_token_start,
                    record.condition.sampled_target_token_end,
                ) != (call.target_token_span.start, call.target_token_span.end):
                    raise IdentityMismatchError(
                        "observation target token span differs from sampled tool call"
                    )
                if representation is None:
                    representation = record.representation
                elif record.representation != representation:
                    raise IdentityMismatchError(
                        "representation artifact changed within one trajectory"
                    )
            if call.assistant_turn_index >= len(trajectory.assistant_turns):
                raise ReplayMismatchError(
                    "tool call references a missing assistant turn"
                )
            assistant_turn = trajectory.assistant_turns[call.assistant_turn_index]
            if not assistant_turn.is_tool_call:
                raise ReplayMismatchError(
                    "tool call references a non-tool assistant turn"
                )
            if isinstance(
                record, (FocusedObservationRecord, CropTGVFObservationRecord)
            ):
                sampled_length = len(assistant_turn.tokens.token_ids)
                if call.target_token_span.end > sampled_length:
                    raise ReplayMismatchError(
                        "sampled target span lies outside its assistant turn"
                    )
                prefix_length = record.condition.source_sequence_length - sampled_length
                expected_full_span = (
                    prefix_length + call.target_token_span.start,
                    prefix_length + call.target_token_span.end,
                )
                if (
                    prefix_length < 0
                    or (
                        record.condition.conditioning_target_token_start,
                        record.condition.conditioning_target_token_end,
                    )
                    != expected_full_span
                ):
                    raise IdentityMismatchError(
                        "conditioning target span differs from sampled-turn offset"
                    )
        for error in trajectory.tool_errors:
            if error.assistant_turn_index >= len(trajectory.assistant_turns):
                raise ReplayMismatchError(
                    "tool error references a missing assistant turn"
                )
            if not trajectory.assistant_turns[error.assistant_turn_index].is_tool_call:
                raise ReplayMismatchError(
                    "tool error references a non-tool assistant turn"
                )

        for turn in trajectory.assistant_turns:
            trace = self.behavior_store.resolve(turn.behavior_trace)
            if trace.trajectory_id != trajectory.identity.canonical_id:
                raise IdentityMismatchError(
                    "behavior trace differs from trajectory identity"
                )
            if trace.assistant_turn_index != turn.turn_index:
                raise IdentityMismatchError(
                    "behavior trace differs from assistant turn index"
                )
            if trace.tokens != turn.tokens:
                raise ReplayMismatchError(
                    "assistant tokens differ from content-addressed behavior trace"
                )
            if trace.behavior_policy != trajectory.behavior_policy:
                raise IdentityMismatchError(
                    "behavior trace policy differs from trajectory"
                )
            policy_indices = turn.tokens.policy_indices
            if policy_indices != trace.behavior.sampled_token_indices:
                raise ReplayMismatchError(
                    "policy ownership mask and behavior logprobs differ"
                )
            sampled_ids = tuple(
                turn.tokens.token_ids[index] for index in policy_indices
            )
            if sampled_ids != trace.behavior.sampled_token_ids:
                raise ReplayMismatchError(
                    "sampled token IDs differ from behavior record"
                )
            sampling = trace.behavior.sampling
            if sampling.policy_version != trajectory.behavior_policy:
                raise IdentityMismatchError(
                    "turn behavior policy differs from trajectory"
                )
            if sampling.asynchronous_staleness_steps > self.maximum_policy_staleness:
                raise ReplayMismatchError("rollout staleness exceeds accepted bound")
            if turn.think_span is not None:
                span = turn.think_span
                if span.end > len(turn.tokens.token_ids):
                    raise ReplayMismatchError("think span lies outside assistant turn")
            if not all(
                isinstance(owner, TokenOwnership) for owner in turn.tokens.ownership
            ):
                raise ReplayMismatchError("unknown token ownership value")

        if trajectory.stop in {
            TrajectoryStop.MALFORMED_CALL,
            TrajectoryStop.TOOL_ERROR,
            TrajectoryStop.CALL_CAP,
        }:
            if (
                not trajectory.assistant_turns
                or not trajectory.assistant_turns[-1].is_tool_call
            ):
                raise ReplayMismatchError(
                    "tool-terminal trajectory must end in a tool-call assistant turn"
                )
            final_turn_index = len(trajectory.assistant_turns) - 1
            final_attempt_is_recorded = bool(
                attempts and attempts[-1][1] == final_turn_index
            )
            cap_recovery_retry = (
                trajectory.stop is TrajectoryStop.CALL_CAP
                and _is_unadmitted_cap_recovery_retry(
                    trajectory,
                    attempts=attempts,
                    final_turn_index=final_turn_index,
                )
            )
            if not final_attempt_is_recorded and not cap_recovery_retry:
                raise ReplayMismatchError(
                    "tool-terminal trajectory must record its final tool attempt"
                )
        if trajectory.stop is TrajectoryStop.INVALID_FORMAT:
            if not trajectory.assistant_turns:
                raise ReplayMismatchError(
                    "invalid-format trajectory must retain its assistant turn"
                )
            final_turn = trajectory.assistant_turns[-1]
            if final_turn.is_tool_call or (
                final_turn.think_span is not None
                and trajectory.final_answer is not None
            ):
                raise ReplayMismatchError(
                    "invalid-format terminal turn must be a malformed final response"
                )

    def validate_batch(self, batch: TrajectoryBatch) -> None:
        seen: set[tuple[str, int]] = set()
        for trajectory in batch.trajectories:
            key = (trajectory.identity.sample_id, trajectory.identity.rollout_index)
            if key in seen:
                raise IdentityMismatchError("duplicate trajectory batch identity")
            seen.add(key)
            self.validate(trajectory)


def _is_unadmitted_cap_recovery_retry(
    trajectory: TrajectoryRecord,
    *,
    attempts: tuple[tuple[int | None, int, str], ...],
    final_turn_index: int,
) -> bool:
    """Recognize the one tool action forbidden after the emitted cap error."""

    cap_errors = tuple(
        error
        for error in trajectory.tool_errors
        if error.code == ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value
    )
    if len(cap_errors) != 1 or not attempts:
        return False
    cap_error = cap_errors[0]
    return cap_error.assistant_turn_index == final_turn_index - 1 and attempts[-1] == (
        cap_error.attempt_index,
        cap_error.assistant_turn_index,
        "error",
    )


class TrajectoryBatcher:
    def __init__(self, validator: TrajectoryValidator) -> None:
        self.validator = validator

    def batch(self, trajectories: tuple[TrajectoryRecord, ...]) -> TrajectoryBatch:
        batch = TrajectoryBatch(trajectories)
        self.validator.validate_batch(batch)
        return batch
