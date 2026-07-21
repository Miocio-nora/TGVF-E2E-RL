"""Lossless boundary from verified project trajectories to public AgentLoop APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from tgvf_rl.contracts.tokens import TokenOwnership
from tgvf_rl.observations.store import (
    ObservationHandle,
    TrajectoryReplayBundle,
    TrajectoryReplayHandle,
)
from tgvf_rl.trajectories.behavior import (
    BehaviorTraceHandle,
    BehaviorTraceRecord,
    verify_behavior_trace_pair,
)
from tgvf_rl.trajectories.schema import TrajectoryRecord, trajectory_checksum
from tgvf_rl.trajectories.validation import TrajectoryValidator

from .compatibility import (
    SPIKE_CANDIDATE_VERL_COMMIT,
    TORCH211_CANDIDATE_VERL_COMMIT,
    VERL_AGENT_LOOP_RETURN_TRANSPORT,
    VERL_AGENT_LOOP_TRANSFER_QUEUE_TRANSPORT,
    VerlCompatibilityError,
    VerlPublicAPI,
    load_verl_public_api,
)
from .objective_bridge import validate_objective_sentinels


BRIDGE_SCHEMA_VERSION = "tgvf-verl-rollout-bridge-v2"
BRIDGE_SCHEMA_FIELD = "tgvf_bridge_schema_version"
EXACT_PROMPT_IDS_FIELD = "tgvf_exact_prompt_ids"
EXACT_RESPONSE_IDS_FIELD = "tgvf_exact_response_ids"
EXACT_OBSERVATION_HANDLES_FIELD = "tgvf_exact_observation_handles"
ACTUAL_RESPONSE_LOGPROBS_FIELD = "tgvf_actual_response_logprobs"
BEHAVIOR_TRACE_HANDLES_FIELD = "tgvf_behavior_trace_handles"
BEHAVIOR_TRACE_RECORDS_FIELD = "tgvf_behavior_trace_records"
OBJECTIVE_SENTINELS_FIELD = "tgvf_objective_sentinels"
TRAJECTORY_PAYLOAD_FIELD = "tgvf_trajectory_payload"
TRAJECTORY_ID_FIELD = "tgvf_trajectory_id"
TRAJECTORY_SHA256_FIELD = "tgvf_trajectory_sha256"
TRAJECTORY_REPLAY_HANDLE_FIELD = "tgvf_trajectory_replay_handle"
TRAJECTORY_REPLAY_BUNDLE_FIELD = "tgvf_trajectory_replay_bundle"
TOKEN_OWNERSHIP_SHA256_FIELD = "tgvf_token_ownership_sha256"
ROLLOUT_PROVENANCE_SHA256_FIELD = "tgvf_rollout_provenance_sha256"

# The live AgentLoop manager builds its DataProto directly from
# ``AgentLoopOutput.extra_fields``; it does not pass through DataProtoPayload.
# Carry the release lease as ordinary non-tensor fields so the subsequent
# DataProto -> TensorDict conversion can preserve it across the Ray boundary.
DATAPROTO_META_SCHEMA_VERSION = "tgvf-verl-dataproto-meta-v1"
DATAPROTO_META_SCHEMA_FIELD = "tgvf_dataproto_meta_schema_version"
SIDECAR_RELEASE_SCHEMA_VERSION = "tgvf-dataproto-sidecar-release-v1"
SIDECAR_RELEASE_SCHEMA_FIELD = "tgvf_sidecar_release_schema_version"
SIDECAR_RELEASE_FIELDS_FIELD = "tgvf_sidecar_release_fields"

AGENT_LOOP_EXACT_SIDECAR_FIELDS = (
    BRIDGE_SCHEMA_FIELD,
    EXACT_PROMPT_IDS_FIELD,
    EXACT_RESPONSE_IDS_FIELD,
    EXACT_OBSERVATION_HANDLES_FIELD,
    ACTUAL_RESPONSE_LOGPROBS_FIELD,
    BEHAVIOR_TRACE_HANDLES_FIELD,
    BEHAVIOR_TRACE_RECORDS_FIELD,
    OBJECTIVE_SENTINELS_FIELD,
    TRAJECTORY_PAYLOAD_FIELD,
    TRAJECTORY_ID_FIELD,
    TRAJECTORY_SHA256_FIELD,
    TRAJECTORY_REPLAY_HANDLE_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
    TOKEN_OWNERSHIP_SHA256_FIELD,
    ROLLOUT_PROVENANCE_SHA256_FIELD,
)

_RESERVED_EXTRA_FIELDS = {
    BRIDGE_SCHEMA_FIELD,
    EXACT_PROMPT_IDS_FIELD,
    EXACT_RESPONSE_IDS_FIELD,
    EXACT_OBSERVATION_HANDLES_FIELD,
    ACTUAL_RESPONSE_LOGPROBS_FIELD,
    BEHAVIOR_TRACE_HANDLES_FIELD,
    BEHAVIOR_TRACE_RECORDS_FIELD,
    OBJECTIVE_SENTINELS_FIELD,
    TRAJECTORY_PAYLOAD_FIELD,
    TRAJECTORY_ID_FIELD,
    TRAJECTORY_SHA256_FIELD,
    TRAJECTORY_REPLAY_HANDLE_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
    TOKEN_OWNERSHIP_SHA256_FIELD,
    ROLLOUT_PROVENANCE_SHA256_FIELD,
}
_BRIDGE_MINT_TOKEN = object()


def _validate_handle(handle: object) -> ObservationHandle:
    if not isinstance(handle, ObservationHandle):
        raise TypeError("exact observation handles must be ObservationHandle values")
    if not handle.observation_id:
        raise ValueError("observation handle ID must be non-empty")
    _validate_lower_sha256(handle.record_sha256, "observation handle")
    return handle


def _validate_replay_handle(handle: object) -> TrajectoryReplayHandle:
    if not isinstance(handle, TrajectoryReplayHandle):
        raise TypeError("trajectory replay handle must be a TrajectoryReplayHandle")
    if not handle.replay_id:
        raise ValueError("trajectory replay ID must be non-empty")
    _validate_lower_sha256(handle.record_sha256, "trajectory replay handle")
    return handle


def _validate_lower_sha256(value: str, owner: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{owner} must carry a lowercase SHA256")


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trajectory_response_materialization(
    trajectory: TrajectoryRecord,
    behavior_records: tuple[BehaviorTraceRecord, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...], tuple[str, ...]]:
    if len(behavior_records) != len(trajectory.assistant_turns):
        raise ValueError("every assistant turn requires one behavior trace record")
    calls_by_turn = {
        call.assistant_turn_index: call.call_index for call in trajectory.tool_calls
    }
    observations_by_call = {
        observation.call_index: observation for observation in trajectory.observations
    }
    errors_by_turn = {
        error.assistant_turn_index: error for error in trajectory.tool_errors
    }
    if set(calls_by_turn) & set(errors_by_turn):
        raise ValueError("one assistant turn cannot carry success and error responses")
    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []
    ownership: list[str] = []

    for turn, trace in zip(trajectory.assistant_turns, behavior_records, strict=True):
        if trace.trajectory_id != trajectory.identity.canonical_id:
            raise ValueError("behavior trace is bound to another trajectory")
        if trace.assistant_turn_index != turn.turn_index:
            raise ValueError("behavior trace is bound to another assistant turn")
        if (
            trace.tokens != turn.tokens
            or trace.behavior_policy != trajectory.behavior_policy
        ):
            raise ValueError("behavior trace tokens/policy differ from trajectory")
        sampled_logprobs = dict(
            zip(
                trace.behavior.sampled_token_indices,
                trace.behavior.logprobs,
                strict=True,
            )
        )
        for token_index, (token_id, owner) in enumerate(
            zip(turn.tokens.token_ids, turn.tokens.ownership, strict=True)
        ):
            response_ids.append(token_id)
            ownership.append(owner.value)
            if owner is TokenOwnership.POLICY_SAMPLED:
                try:
                    response_logprobs.append(sampled_logprobs[token_index])
                except KeyError as error:
                    raise ValueError(
                        "policy-owned token lacks actual behavior logprob"
                    ) from error
                response_mask.append(1)
            else:
                if token_index in sampled_logprobs:
                    raise ValueError(
                        "non-policy token carries an actual behavior logprob"
                    )
                response_logprobs.append(0.0)
                response_mask.append(0)

        call_index = calls_by_turn.get(turn.turn_index)
        if call_index is not None:
            try:
                observation = observations_by_call[call_index]
            except KeyError as error:
                raise ValueError(
                    "successful tool call lacks appended observation tokens"
                ) from error
            response_ids.extend(observation.template_token_ids)
            response_mask.extend(0 for _ in observation.template_token_ids)
            response_logprobs.extend(0.0 for _ in observation.template_token_ids)
            ownership.extend(
                TokenOwnership.TOOL_OBSERVATION.value
                for _ in observation.template_token_ids
            )
        error = errors_by_turn.get(turn.turn_index)
        if error is not None:
            response_ids.extend(error.template_token_ids)
            response_mask.extend(0 for _ in error.template_token_ids)
            response_logprobs.extend(0.0 for _ in error.template_token_ids)
            ownership.extend(
                TokenOwnership.TOOL_OBSERVATION.value
                for _ in error.template_token_ids
            )

    return (
        tuple(response_ids),
        tuple(response_mask),
        tuple(response_logprobs),
        tuple(ownership),
    )


def token_ownership_checksum(
    *,
    trajectory_id: str,
    prompt_ids: tuple[int, ...],
    response_ids: tuple[int, ...],
    response_ownership: tuple[str, ...],
) -> str:
    return _json_sha256(
        {
            "schema": "tgvf-global-token-ownership-v1",
            "trajectory_id": trajectory_id,
            "prompt_ids": prompt_ids,
            "prompt_ownership": [TokenOwnership.TEMPLATE.value] * len(prompt_ids),
            "response_ids": response_ids,
            "response_ownership": response_ownership,
        }
    )


def rollout_provenance_checksum(
    *,
    trajectory_id: str,
    trajectory_sha256: str,
    replay_handle: TrajectoryReplayHandle,
    replay_bundle_sha256: str,
    observation_handles: tuple[ObservationHandle, ...],
    behavior_trace_handles: tuple[BehaviorTraceHandle, ...],
    token_ownership_sha256: str,
) -> str:
    """Bind every independently addressed rollout/replay artifact together."""

    return _json_sha256(
        {
            "schema": "tgvf-rollout-provenance-v2",
            "trajectory_id": trajectory_id,
            "trajectory_sha256": trajectory_sha256,
            "replay": {
                "id": replay_handle.replay_id,
                "sha256": replay_handle.record_sha256,
                "bundle_sha256": replay_bundle_sha256,
            },
            "observations": [
                {"id": handle.observation_id, "sha256": handle.record_sha256}
                for handle in observation_handles
            ],
            "behavior_traces": [
                {"id": handle.trace_id, "sha256": handle.record_sha256}
                for handle in behavior_trace_handles
            ],
            "token_ownership_sha256": token_ownership_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class RolloutBridgeRecord:
    """One internally minted AgentLoop result with content-bound replay state."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    exact_observation_handles: tuple[ObservationHandle, ...]
    behavior_trace_handles: tuple[BehaviorTraceHandle, ...]
    behavior_trace_records: tuple[BehaviorTraceRecord, ...]
    sentinel_fields: Mapping[str, object]
    num_turns: int
    trajectory_id: str
    trajectory_sha256: str
    replay_handle: TrajectoryReplayHandle
    replay_bundle: TrajectoryReplayBundle
    token_ownership_sha256: str
    rollout_provenance_sha256: str
    trajectory_payload: TrajectoryRecord
    _mint_token: object = field(repr=False, compare=False)
    extra_fields: Mapping[str, object] = field(default_factory=dict)
    reward_score: float | None = None
    multi_modal_data: Mapping[str, Any] | None = None
    mm_processor_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self._mint_token is not _BRIDGE_MINT_TOKEN:
            raise TypeError(
                "RolloutBridgeRecord must be minted by trajectory_to_rollout_bridge"
            )
        object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))
        object.__setattr__(self, "response_ids", tuple(self.response_ids))
        object.__setattr__(self, "response_mask", tuple(self.response_mask))
        object.__setattr__(self, "response_logprobs", tuple(self.response_logprobs))
        object.__setattr__(
            self,
            "exact_observation_handles",
            tuple(
                _validate_handle(handle) for handle in self.exact_observation_handles
            ),
        )
        object.__setattr__(
            self, "behavior_trace_handles", tuple(self.behavior_trace_handles)
        )
        object.__setattr__(
            self, "behavior_trace_records", tuple(self.behavior_trace_records)
        )
        object.__setattr__(
            self, "replay_handle", _validate_replay_handle(self.replay_handle)
        )
        if not isinstance(self.replay_bundle, TrajectoryReplayBundle):
            raise TypeError("replay_bundle must be a TrajectoryReplayBundle")
        if self.replay_bundle.replay_handle != self.replay_handle:
            raise ValueError("replay provenance bundle and replay handle differ")
        object.__setattr__(
            self,
            "sentinel_fields",
            validate_objective_sentinels(self.sentinel_fields),
        )
        if any(
            type(token_id) is not int or token_id < 0
            for token_id in self.prompt_ids + self.response_ids
        ):
            raise ValueError(
                "prompt and response token IDs must be non-negative integers"
            )
        if not self.prompt_ids:
            raise ValueError("AgentLoop prompt IDs must be non-empty")
        if not self.response_ids:
            raise ValueError("AgentLoop response IDs must be non-empty")
        if not isinstance(self.trajectory_payload, TrajectoryRecord):
            raise TypeError("trajectory_payload must be a complete TrajectoryRecord")
        lengths = {
            len(self.response_ids),
            len(self.response_mask),
            len(self.response_logprobs),
        }
        if len(lengths) != 1:
            raise ValueError(
                "response IDs, masks, and actual log probabilities must align exactly"
            )
        if any(
            type(value) is not int or value not in {0, 1}
            for value in self.response_mask
        ):
            raise ValueError("response_mask values must be integer 0 or 1")
        if not any(self.response_mask):
            raise ValueError(
                "response_mask must identify at least one policy-sampled token"
            )
        if any(not math.isfinite(value) for value in self.response_logprobs):
            raise ValueError("actual response log probabilities must be finite")
        if any(
            mask == 0 and logprob != 0.0
            for mask, logprob in zip(
                self.response_mask, self.response_logprobs, strict=True
            )
        ):
            raise ValueError("template/tool tokens cannot carry behavior logprobs")
        if type(self.num_turns) is not int or self.num_turns <= 0:
            raise ValueError("num_turns must be a positive integer")
        if self.reward_score is not None and not math.isfinite(self.reward_score):
            raise ValueError("reward_score must be finite when present")
        if not isinstance(self.extra_fields, Mapping):
            raise TypeError("extra_fields must be a mapping")
        collisions = _RESERVED_EXTRA_FIELDS & set(self.extra_fields)
        if collisions:
            raise ValueError(
                f"extra_fields collide with reserved bridge fields: {sorted(collisions)}"
            )
        object.__setattr__(
            self, "extra_fields", MappingProxyType(dict(self.extra_fields))
        )
        if self.multi_modal_data is not None:
            object.__setattr__(
                self, "multi_modal_data", MappingProxyType(dict(self.multi_modal_data))
            )
        if self.mm_processor_kwargs is not None:
            object.__setattr__(
                self,
                "mm_processor_kwargs",
                MappingProxyType(dict(self.mm_processor_kwargs)),
            )
        self._validate_content_bindings()

    def _validate_content_bindings(self) -> None:
        trajectory = self.trajectory_payload
        if self.trajectory_id != trajectory.identity.canonical_id:
            raise ValueError("bridge trajectory ID differs from trajectory payload")
        _validate_lower_sha256(self.trajectory_sha256, "trajectory payload")
        if self.trajectory_sha256 != trajectory_checksum(trajectory):
            raise ValueError("bridge trajectory checksum differs from payload")
        if self.num_turns != len(trajectory.assistant_turns):
            raise ValueError("bridge turn count differs from trajectory payload")
        expected_observations = tuple(item.handle for item in trajectory.observations)
        if self.exact_observation_handles != expected_observations:
            raise ValueError(
                "bridge observation handles differ from trajectory payload"
            )
        expected_behavior_handles = tuple(
            turn.behavior_trace for turn in trajectory.assistant_turns
        )
        if self.behavior_trace_handles != expected_behavior_handles:
            raise ValueError("bridge behavior handles differ from trajectory payload")
        if len(self.behavior_trace_handles) != len(self.behavior_trace_records):
            raise ValueError("behavior trace handles and records must align")
        for handle, record in zip(
            self.behavior_trace_handles, self.behavior_trace_records, strict=True
        ):
            verify_behavior_trace_pair(handle, record)
        expected_ids, expected_mask, expected_logprobs, ownership = (
            _trajectory_response_materialization(
                trajectory, self.behavior_trace_records
            )
        )
        if self.response_ids != expected_ids or self.response_mask != expected_mask:
            raise ValueError("bridge response order/ownership differs from trajectory")
        if self.response_logprobs != expected_logprobs:
            raise ValueError("bridge logprobs differ from behavior trace records")
        expected_ownership_sha = token_ownership_checksum(
            trajectory_id=self.trajectory_id,
            prompt_ids=self.prompt_ids,
            response_ids=self.response_ids,
            response_ownership=ownership,
        )
        _validate_lower_sha256(self.token_ownership_sha256, "token ownership")
        if self.token_ownership_sha256 != expected_ownership_sha:
            raise ValueError("bridge token ownership checksum differs from trajectory")
        expected_provenance_sha = rollout_provenance_checksum(
            trajectory_id=self.trajectory_id,
            trajectory_sha256=self.trajectory_sha256,
            replay_handle=self.replay_handle,
            replay_bundle_sha256=self.replay_bundle.bundle_sha256,
            observation_handles=self.exact_observation_handles,
            behavior_trace_handles=self.behavior_trace_handles,
            token_ownership_sha256=self.token_ownership_sha256,
        )
        _validate_lower_sha256(self.rollout_provenance_sha256, "rollout provenance")
        if self.rollout_provenance_sha256 != expected_provenance_sha:
            raise ValueError("bridge rollout provenance checksum differs from sidecars")

    def agent_loop_extra_fields(self) -> dict[str, object]:
        """Return a fresh mapping; exact sidecars remain immutable values."""

        fields = dict(self.extra_fields)
        fields.update(
            {
                BRIDGE_SCHEMA_FIELD: BRIDGE_SCHEMA_VERSION,
                EXACT_PROMPT_IDS_FIELD: self.prompt_ids,
                EXACT_RESPONSE_IDS_FIELD: self.response_ids,
                EXACT_OBSERVATION_HANDLES_FIELD: self.exact_observation_handles,
                ACTUAL_RESPONSE_LOGPROBS_FIELD: self.response_logprobs,
                BEHAVIOR_TRACE_HANDLES_FIELD: self.behavior_trace_handles,
                BEHAVIOR_TRACE_RECORDS_FIELD: self.behavior_trace_records,
                OBJECTIVE_SENTINELS_FIELD: dict(self.sentinel_fields),
                TRAJECTORY_PAYLOAD_FIELD: self.trajectory_payload,
                TRAJECTORY_ID_FIELD: self.trajectory_id,
                TRAJECTORY_SHA256_FIELD: self.trajectory_sha256,
                TRAJECTORY_REPLAY_HANDLE_FIELD: self.replay_handle,
                TRAJECTORY_REPLAY_BUNDLE_FIELD: self.replay_bundle,
                TOKEN_OWNERSHIP_SHA256_FIELD: self.token_ownership_sha256,
                ROLLOUT_PROVENANCE_SHA256_FIELD: self.rollout_provenance_sha256,
            }
        )
        return fields


def _mint_rollout_bridge_record(**kwargs: object) -> RolloutBridgeRecord:
    return RolloutBridgeRecord(_mint_token=_BRIDGE_MINT_TOKEN, **kwargs)  # type: ignore[arg-type]


def trajectory_to_rollout_bridge(
    trajectory: TrajectoryRecord,
    *,
    validator: TrajectoryValidator,
    initial_prompt_token_ids: tuple[int, ...],
    native_tool_appended_token_ids: tuple[tuple[int, ...], ...],
    replay_handle: TrajectoryReplayHandle,
    sentinel_fields: Mapping[str, object],
    extra_fields: Mapping[str, object] | None = None,
    reward_score: float | None = None,
    multi_modal_data: Mapping[str, Any] | None = None,
    mm_processor_kwargs: Mapping[str, Any] | None = None,
) -> RolloutBridgeRecord:
    """The sole trajectory-to-veRL converter; no token is rendered or regenerated."""

    validator.validate(trajectory)
    prompt_ids = tuple(initial_prompt_token_ids)
    native_tokens = tuple(tuple(row) for row in native_tool_appended_token_ids)
    observation_by_turn = {
        call.assistant_turn_index: observation.template_token_ids
        for call, observation in zip(
            trajectory.tool_calls, trajectory.observations, strict=True
        )
    }
    error_by_turn = {
        error.assistant_turn_index: error.template_token_ids
        for error in trajectory.tool_errors
    }
    recorded_native_tokens = tuple(
        observation_by_turn.get(turn.turn_index, error_by_turn.get(turn.turn_index))
        for turn in trajectory.assistant_turns
        if turn.turn_index in observation_by_turn or turn.turn_index in error_by_turn
    )
    if native_tokens != recorded_native_tokens:
        raise ValueError("native appended tool tokens differ from trajectory record")
    behavior_records = tuple(
        validator.behavior_store.resolve(turn.behavior_trace)
        for turn in trajectory.assistant_turns
    )
    response_ids, response_mask, response_logprobs, ownership = (
        _trajectory_response_materialization(trajectory, behavior_records)
    )
    replay = validator.store.resolve_replay(replay_handle)
    replay_bundle = validator.store.export_replay_bundle(replay_handle)
    if replay.trajectory_id != trajectory.identity.canonical_id:
        raise ValueError("trajectory replay is bound to another trajectory")
    if replay.model != trajectory.model:
        raise ValueError("trajectory replay model differs from trajectory")
    if replay.behavior_policy != trajectory.behavior_policy:
        raise ValueError("trajectory replay policy differs from trajectory")
    observation_handles = tuple(item.handle for item in trajectory.observations)
    if replay.observation_handles != observation_handles:
        raise ValueError("trajectory replay observation handles differ from trajectory")
    replay_input_ids = validator.store.resolve_verified(replay.tensors.input_ids)
    if tuple(replay_input_ids.shape[:1]) != (1,):
        raise ValueError("trajectory replay converter requires one unpadded sequence")
    exact_final_ids = tuple(int(token_id) for token_id in replay_input_ids[0].tolist())
    if exact_final_ids != prompt_ids + response_ids:
        raise ValueError(
            "replay input IDs differ from exact prompt/trajectory token order"
        )
    trajectory_id = trajectory.identity.canonical_id
    ownership_sha256 = token_ownership_checksum(
        trajectory_id=trajectory_id,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_ownership=ownership,
    )
    behavior_handles = tuple(turn.behavior_trace for turn in trajectory.assistant_turns)
    trajectory_sha256 = trajectory_checksum(trajectory)
    return _mint_rollout_bridge_record(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        exact_observation_handles=observation_handles,
        behavior_trace_handles=behavior_handles,
        behavior_trace_records=behavior_records,
        sentinel_fields=sentinel_fields,
        num_turns=len(trajectory.assistant_turns),
        trajectory_id=trajectory_id,
        trajectory_sha256=trajectory_sha256,
        replay_handle=replay_handle,
        replay_bundle=replay_bundle,
        token_ownership_sha256=ownership_sha256,
        rollout_provenance_sha256=rollout_provenance_checksum(
            trajectory_id=trajectory_id,
            trajectory_sha256=trajectory_sha256,
            replay_handle=replay_handle,
            replay_bundle_sha256=replay_bundle.bundle_sha256,
            observation_handles=observation_handles,
            behavior_trace_handles=behavior_handles,
            token_ownership_sha256=ownership_sha256,
        ),
        trajectory_payload=trajectory,
        extra_fields={} if extra_fields is None else extra_fields,
        reward_score=reward_score,
        multi_modal_data=multi_modal_data,
        mm_processor_kwargs=mm_processor_kwargs,
    )


def build_agent_loop_output(
    record: RolloutBridgeRecord,
    *,
    metrics: object,
    agent_loop_output_cls: type[Any] | None = None,
) -> Any:
    """Construct the public ``AgentLoopOutput`` without rerendering any token."""

    if agent_loop_output_cls is None:
        agent_loop_output_cls = load_verl_public_api().agent_loop_output
    return agent_loop_output_cls(
        prompt_ids=list(record.prompt_ids),
        response_ids=list(record.response_ids),
        response_mask=list(record.response_mask),
        response_logprobs=list(record.response_logprobs),
        multi_modal_data=(
            dict(record.multi_modal_data)
            if record.multi_modal_data is not None
            else None
        ),
        reward_score=record.reward_score,
        num_turns=record.num_turns,
        metrics=metrics,
        extra_fields=record.agent_loop_extra_fields(),
        mm_processor_kwargs=(
            dict(record.mm_processor_kwargs)
            if record.mm_processor_kwargs is not None
            else None
        ),
    )


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        try:
            return value[name]
        except KeyError as error:
            raise ValueError(f"AgentLoop output is missing {name!r}") from error
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise ValueError(f"AgentLoop output is missing {name!r}") from error


def parse_agent_loop_output(output: object) -> RolloutBridgeRecord:
    """Recover and cross-check a previously minted AgentLoop output."""

    extra = _field(output, "extra_fields")
    if not isinstance(extra, Mapping):
        raise TypeError("AgentLoop extra_fields must be a mapping")
    if extra.get(BRIDGE_SCHEMA_FIELD) != BRIDGE_SCHEMA_VERSION:
        raise ValueError("AgentLoop output has an absent or incompatible bridge schema")
    response_logprobs = _field(output, "response_logprobs")
    if response_logprobs is None:
        raise ValueError("AgentLoop output is missing actual response_logprobs")
    exact_logprobs = tuple(extra.get(ACTUAL_RESPONSE_LOGPROBS_FIELD, ()))
    if tuple(response_logprobs) != exact_logprobs:
        raise ValueError("AgentLoop response_logprobs changed after rollout recording")
    exact_prompt_ids = tuple(extra.get(EXACT_PROMPT_IDS_FIELD, ()))
    exact_response_ids = tuple(extra.get(EXACT_RESPONSE_IDS_FIELD, ()))
    if tuple(_field(output, "prompt_ids")) != exact_prompt_ids:
        raise ValueError("AgentLoop prompt_ids changed after trajectory conversion")
    if tuple(_field(output, "response_ids")) != exact_response_ids:
        raise ValueError("AgentLoop response_ids changed after trajectory conversion")
    ordinary_extra = {
        key: item for key, item in extra.items() if key not in _RESERVED_EXTRA_FIELDS
    }
    return _mint_rollout_bridge_record(
        prompt_ids=exact_prompt_ids,
        response_ids=exact_response_ids,
        response_mask=tuple(_field(output, "response_mask")),
        response_logprobs=exact_logprobs,
        exact_observation_handles=tuple(extra.get(EXACT_OBSERVATION_HANDLES_FIELD, ())),
        behavior_trace_handles=tuple(extra.get(BEHAVIOR_TRACE_HANDLES_FIELD, ())),
        behavior_trace_records=tuple(extra.get(BEHAVIOR_TRACE_RECORDS_FIELD, ())),
        sentinel_fields=extra.get(OBJECTIVE_SENTINELS_FIELD, {}),
        num_turns=int(_field(output, "num_turns")),
        trajectory_id=extra.get(TRAJECTORY_ID_FIELD),
        trajectory_sha256=extra.get(TRAJECTORY_SHA256_FIELD),
        replay_handle=extra.get(TRAJECTORY_REPLAY_HANDLE_FIELD),
        replay_bundle=extra.get(TRAJECTORY_REPLAY_BUNDLE_FIELD),
        token_ownership_sha256=extra.get(TOKEN_OWNERSHIP_SHA256_FIELD),
        rollout_provenance_sha256=extra.get(ROLLOUT_PROVENANCE_SHA256_FIELD),
        trajectory_payload=extra.get(TRAJECTORY_PAYLOAD_FIELD),
        extra_fields=ordinary_extra,
        reward_score=_field(output, "reward_score"),
        multi_modal_data=_field(output, "multi_modal_data"),
        mm_processor_kwargs=_field(output, "mm_processor_kwargs"),
    )


def _map_maybe_awaitable(value: object, transform: Any) -> object:
    if inspect.isawaitable(value):

        async def deferred() -> object:
            return transform(await value)

        return deferred()
    return transform(value)


class _LosslessAgentLoopManagerBase:
    """Version-bound composition wrapper for one audited veRL transport."""

    expected_verl_commit: str
    expected_transport: str

    def __init__(
        self,
        *args: object,
        _delegate: object | None = None,
        _public_api: VerlPublicAPI | None = None,
        **kwargs: object,
    ) -> None:
        self._public_api = _public_api or load_verl_public_api(
            expected_commit=self.expected_verl_commit
        )
        if self._public_api.agent_loop_transport != self.expected_transport:
            raise VerlCompatibilityError(
                "veRL agent-loop manager transport differs from the selected runtime"
            )
        self._delegate = (
            _delegate
            if _delegate is not None
            else self._public_api.agent_loop_manager(*args, **kwargs)
        )

    @classmethod
    def create(cls, *args: object, **kwargs: object) -> object:
        api = kwargs.pop("_public_api", None) or load_verl_public_api(
            expected_commit=cls.expected_verl_commit
        )
        if api.agent_loop_transport != cls.expected_transport:
            raise VerlCompatibilityError(
                "veRL agent-loop manager transport differs from the selected runtime"
            )
        created = api.agent_loop_manager.create(*args, **kwargs)
        return _map_maybe_awaitable(
            created,
            lambda delegate: cls(_delegate=delegate, _public_api=api),
        )

    def generate_sequences(self, prompts: object) -> object:
        generated = self._delegate.generate_sequences(prompts)
        if self.expected_transport == VERL_AGENT_LOOP_TRANSFER_QUEUE_TRANSPORT:
            return _map_maybe_awaitable(generated, _require_transfer_queue_dispatch)

        from .data_bridge import (
            bind_agent_loop_data_proto_sidecar_lease,
            validate_data_proto_integrity,
        )

        return _map_maybe_awaitable(
            generated,
            lambda output: _validate_and_return(
                bind_agent_loop_data_proto_sidecar_lease(output),
                validate_data_proto_integrity,
            ),
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class LosslessAgentLoopManager(_LosslessAgentLoopManagerBase):
    """Lossless DataProto-return manager for the accepted control veRL."""

    expected_verl_commit = SPIKE_CANDIDATE_VERL_COMMIT
    expected_transport = VERL_AGENT_LOOP_RETURN_TRANSPORT


class LosslessTransferQueueAgentLoopManager(_LosslessAgentLoopManagerBase):
    """TransferQueue-dispatch manager for the Torch 2.11 veRL candidate."""

    expected_verl_commit = TORCH211_CANDIDATE_VERL_COMMIT
    expected_transport = VERL_AGENT_LOOP_TRANSFER_QUEUE_TRANSPORT


def _require_transfer_queue_dispatch(value: object) -> None:
    if value is not None:
        raise ValueError(
            "the selected veRL TransferQueue manager must dispatch in-place and "
            "return None"
        )
    return None


def _validate_and_return(value: object, validator: Any) -> object:
    validator(value)
    return value
