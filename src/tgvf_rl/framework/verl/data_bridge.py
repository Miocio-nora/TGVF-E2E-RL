"""DataProto transport for exact log probabilities and observation handles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import torch

from tgvf_rl.observations.store import ObservationHandle, TrajectoryReplayHandle
from tgvf_rl.trajectories.behavior import BehaviorTraceHandle, BehaviorTraceRecord
from tgvf_rl.trajectories.schema import TrajectoryRecord

from .compatibility import load_verl_public_api
from .objective_bridge import validate_objective_sentinels
from .rollout_bridge import (
    ACTUAL_RESPONSE_LOGPROBS_FIELD,
    BEHAVIOR_TRACE_HANDLES_FIELD,
    BEHAVIOR_TRACE_RECORDS_FIELD,
    BRIDGE_SCHEMA_FIELD,
    BRIDGE_SCHEMA_VERSION,
    EXACT_PROMPT_IDS_FIELD,
    EXACT_RESPONSE_IDS_FIELD,
    EXACT_OBSERVATION_HANDLES_FIELD,
    OBJECTIVE_SENTINELS_FIELD,
    ROLLOUT_PROVENANCE_SHA256_FIELD,
    TOKEN_OWNERSHIP_SHA256_FIELD,
    TRAJECTORY_ID_FIELD,
    TRAJECTORY_PAYLOAD_FIELD,
    TRAJECTORY_REPLAY_HANDLE_FIELD,
    TRAJECTORY_SHA256_FIELD,
    RolloutBridgeRecord,
    _mint_rollout_bridge_record,
    _validate_handle,
)


@dataclass(frozen=True, slots=True)
class DataProtoPayload:
    """Neutral constructor payload so CPU tests do not need veRL/TensorDict."""

    tensor_batch: Mapping[str, torch.Tensor]
    non_tensor_batch: Mapping[str, object]
    meta_info: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tensor_batch", MappingProxyType(dict(self.tensor_batch))
        )
        object.__setattr__(
            self, "non_tensor_batch", MappingProxyType(dict(self.non_tensor_batch))
        )
        object.__setattr__(self, "meta_info", MappingProxyType(dict(self.meta_info)))


@dataclass(frozen=True, slots=True)
class DataProtoIntegrityView:
    """Validated exact fields recovered from a DataProto-like object."""

    observation_handles: tuple[tuple[ObservationHandle, ...], ...]
    behavior_trace_handles: tuple[tuple[BehaviorTraceHandle, ...], ...]
    behavior_trace_records: tuple[tuple[BehaviorTraceRecord, ...], ...]
    actual_response_logprobs: tuple[tuple[float, ...], ...]
    objective_sentinels: tuple[Mapping[str, object], ...]
    trajectory_payloads: tuple[TrajectoryRecord, ...]
    replay_handles: tuple[TrajectoryReplayHandle, ...]
    trajectory_ids: tuple[str, ...]
    trajectory_sha256s: tuple[str, ...]
    token_ownership_sha256s: tuple[str, ...]
    rollout_provenance_sha256s: tuple[str, ...]


def _object_array(values: list[object]) -> object:
    try:
        import numpy as np
    except (
        ImportError
    ) as error:  # pragma: no cover - torch environments normally include numpy
        raise RuntimeError("DataProto object fields require numpy") from error
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def build_data_proto_payload(
    records: Iterable[RolloutBridgeRecord],
) -> DataProtoPayload:
    """Build an unmodified, equal-width batch; padding belongs to AgentLoop."""

    rows = tuple(records)
    if not rows:
        raise ValueError("at least one rollout record is required")
    if any(not isinstance(row, RolloutBridgeRecord) for row in rows):
        raise TypeError("all rows must be RolloutBridgeRecord values")
    prompt_widths = {len(row.prompt_ids) for row in rows}
    response_widths = {len(row.response_ids) for row in rows}
    if len(prompt_widths) != 1 or len(response_widths) != 1:
        raise ValueError("the neutral bridge never pads or truncates rollout tokens")

    tensors = {
        "prompts": torch.tensor([row.prompt_ids for row in rows], dtype=torch.int64),
        "responses": torch.tensor(
            [row.response_ids for row in rows], dtype=torch.int64
        ),
        "response_mask": torch.tensor(
            [row.response_mask for row in rows], dtype=torch.int64
        ),
        # Match AgentLoopOutput.as_dict/_postprocess.  The exact Python values
        # remain alongside this public tensor so float32 transport is auditable.
        "rollout_log_probs": torch.tensor(
            [row.response_logprobs for row in rows], dtype=torch.float32
        ),
    }
    non_tensors: dict[str, object] = {
        BRIDGE_SCHEMA_FIELD: _object_array([BRIDGE_SCHEMA_VERSION for _ in rows]),
        EXACT_PROMPT_IDS_FIELD: _object_array([row.prompt_ids for row in rows]),
        EXACT_RESPONSE_IDS_FIELD: _object_array([row.response_ids for row in rows]),
        EXACT_OBSERVATION_HANDLES_FIELD: _object_array(
            [row.exact_observation_handles for row in rows]
        ),
        BEHAVIOR_TRACE_HANDLES_FIELD: _object_array(
            [row.behavior_trace_handles for row in rows]
        ),
        BEHAVIOR_TRACE_RECORDS_FIELD: _object_array(
            [row.behavior_trace_records for row in rows]
        ),
        ACTUAL_RESPONSE_LOGPROBS_FIELD: _object_array(
            [row.response_logprobs for row in rows]
        ),
        OBJECTIVE_SENTINELS_FIELD: _object_array(
            [dict(row.sentinel_fields) for row in rows]
        ),
        TRAJECTORY_PAYLOAD_FIELD: _object_array(
            [row.trajectory_payload for row in rows]
        ),
        TRAJECTORY_ID_FIELD: _object_array([row.trajectory_id for row in rows]),
        TRAJECTORY_SHA256_FIELD: _object_array([row.trajectory_sha256 for row in rows]),
        TRAJECTORY_REPLAY_HANDLE_FIELD: _object_array(
            [row.replay_handle for row in rows]
        ),
        TOKEN_OWNERSHIP_SHA256_FIELD: _object_array(
            [row.token_ownership_sha256 for row in rows]
        ),
        ROLLOUT_PROVENANCE_SHA256_FIELD: _object_array(
            [row.rollout_provenance_sha256 for row in rows]
        ),
        "__num_turns__": _object_array([row.num_turns for row in rows]),
    }
    extra_names = sorted({name for row in rows for name in row.extra_fields})
    for name in extra_names:
        non_tensors[name] = _object_array([row.extra_fields.get(name) for row in rows])
    return DataProtoPayload(
        tensor_batch=tensors,
        non_tensor_batch=non_tensors,
        meta_info={"tgvf_bridge_schema_version": BRIDGE_SCHEMA_VERSION},
    )


def to_verl_data_proto(
    payload: DataProtoPayload,
    *,
    data_proto_cls: type[Any] | None = None,
) -> Any:
    """Use the public ``DataProto.from_dict`` constructor when veRL is live."""

    if data_proto_cls is None:
        data_proto_cls = load_verl_public_api().data_proto
    from_dict = getattr(data_proto_cls, "from_dict", None)
    if not callable(from_dict):
        raise TypeError("public DataProto type must expose from_dict")
    return from_dict(
        tensors=dict(payload.tensor_batch),
        non_tensors=dict(payload.non_tensor_batch),
        meta_info=dict(payload.meta_info),
    )


def build_verl_data_proto(
    records: Iterable[RolloutBridgeRecord],
    *,
    data_proto_cls: type[Any] | None = None,
) -> Any:
    return to_verl_data_proto(
        build_data_proto_payload(records), data_proto_cls=data_proto_cls
    )


def _data_parts(data: object) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    batch = getattr(data, "batch", None)
    non_tensor_batch = getattr(data, "non_tensor_batch", None)
    if (
        batch is None
        or not hasattr(batch, "__getitem__")
        or not hasattr(batch, "__contains__")
    ):
        raise TypeError("DataProto.batch must be a TensorDict/mapping-like value")
    if not isinstance(non_tensor_batch, Mapping):
        raise TypeError("DataProto.non_tensor_batch must be a mapping")
    return batch, non_tensor_batch


def _required(mapping: Mapping[str, Any], name: str, owner: str) -> Any:
    if name not in mapping:
        raise ValueError(f"{owner} is missing required field {name!r}")
    return mapping[name]


def _row_values(value: object, batch_size: int, field_name: str) -> tuple[object, ...]:
    if not hasattr(value, "__len__") or not hasattr(value, "__getitem__"):
        raise TypeError(f"non-tensor field {field_name!r} must be an indexable batch")
    if len(value) != batch_size:
        raise ValueError(f"non-tensor field {field_name!r} has the wrong batch size")
    return tuple(value[index] for index in range(batch_size))


def validate_data_proto_integrity(data: object) -> DataProtoIntegrityView:
    """Prove public DataProto transport did not overwrite project-owned state."""

    batch, non_tensors = _data_parts(data)
    prompts = _required(batch, "prompts", "DataProto.batch")
    responses = _required(batch, "responses", "DataProto.batch")
    response_mask = _required(batch, "response_mask", "DataProto.batch")
    rollout_log_probs = _required(batch, "rollout_log_probs", "DataProto.batch")
    if any(
        not isinstance(value, torch.Tensor)
        for value in (prompts, responses, response_mask, rollout_log_probs)
    ):
        raise TypeError(
            "prompts, responses, response_mask, and rollout_log_probs must be tensors"
        )
    if (
        prompts.ndim != 2
        or responses.ndim != 2
        or prompts.shape[0] != responses.shape[0]
        or response_mask.shape != responses.shape
        or rollout_log_probs.shape != responses.shape
    ):
        raise ValueError("DataProto response fields must share [batch, response] shape")
    if not rollout_log_probs.dtype.is_floating_point:
        raise TypeError("rollout_log_probs must use a floating dtype")
    if not bool(torch.isfinite(rollout_log_probs).all().item()):
        raise ValueError("rollout_log_probs must be finite")
    if not bool(((response_mask == 0) | (response_mask == 1)).all().item()):
        raise ValueError("response_mask must remain binary")

    batch_size, width = responses.shape
    schemas = _row_values(
        _required(non_tensors, BRIDGE_SCHEMA_FIELD, "DataProto.non_tensor_batch"),
        batch_size,
        BRIDGE_SCHEMA_FIELD,
    )
    if any(schema != BRIDGE_SCHEMA_VERSION for schema in schemas):
        raise ValueError("DataProto bridge schema was lost or changed")
    exact_prompt_rows = _row_values(
        _required(non_tensors, EXACT_PROMPT_IDS_FIELD, "DataProto.non_tensor_batch"),
        batch_size,
        EXACT_PROMPT_IDS_FIELD,
    )
    exact_response_rows = _row_values(
        _required(non_tensors, EXACT_RESPONSE_IDS_FIELD, "DataProto.non_tensor_batch"),
        batch_size,
        EXACT_RESPONSE_IDS_FIELD,
    )
    handle_rows = _row_values(
        _required(
            non_tensors, EXACT_OBSERVATION_HANDLES_FIELD, "DataProto.non_tensor_batch"
        ),
        batch_size,
        EXACT_OBSERVATION_HANDLES_FIELD,
    )
    behavior_handle_rows = _row_values(
        _required(
            non_tensors, BEHAVIOR_TRACE_HANDLES_FIELD, "DataProto.non_tensor_batch"
        ),
        batch_size,
        BEHAVIOR_TRACE_HANDLES_FIELD,
    )
    behavior_record_rows = _row_values(
        _required(
            non_tensors, BEHAVIOR_TRACE_RECORDS_FIELD, "DataProto.non_tensor_batch"
        ),
        batch_size,
        BEHAVIOR_TRACE_RECORDS_FIELD,
    )
    exact_logprob_rows = _row_values(
        _required(
            non_tensors, ACTUAL_RESPONSE_LOGPROBS_FIELD, "DataProto.non_tensor_batch"
        ),
        batch_size,
        ACTUAL_RESPONSE_LOGPROBS_FIELD,
    )
    sentinel_rows = _row_values(
        _required(non_tensors, OBJECTIVE_SENTINELS_FIELD, "DataProto.non_tensor_batch"),
        batch_size,
        OBJECTIVE_SENTINELS_FIELD,
    )
    trajectory_rows = _row_values(
        _required(non_tensors, TRAJECTORY_PAYLOAD_FIELD, "DataProto.non_tensor_batch"),
        batch_size,
        TRAJECTORY_PAYLOAD_FIELD,
    )
    trajectory_id_rows = _row_values(
        _required(non_tensors, TRAJECTORY_ID_FIELD, "DataProto.non_tensor_batch"),
        batch_size,
        TRAJECTORY_ID_FIELD,
    )
    trajectory_sha_rows = _row_values(
        _required(non_tensors, TRAJECTORY_SHA256_FIELD, "DataProto.non_tensor_batch"),
        batch_size,
        TRAJECTORY_SHA256_FIELD,
    )
    replay_handle_rows = _row_values(
        _required(
            non_tensors, TRAJECTORY_REPLAY_HANDLE_FIELD, "DataProto.non_tensor_batch"
        ),
        batch_size,
        TRAJECTORY_REPLAY_HANDLE_FIELD,
    )
    ownership_sha_rows = _row_values(
        _required(
            non_tensors, TOKEN_OWNERSHIP_SHA256_FIELD, "DataProto.non_tensor_batch"
        ),
        batch_size,
        TOKEN_OWNERSHIP_SHA256_FIELD,
    )
    provenance_sha_rows = _row_values(
        _required(
            non_tensors,
            ROLLOUT_PROVENANCE_SHA256_FIELD,
            "DataProto.non_tensor_batch",
        ),
        batch_size,
        ROLLOUT_PROVENANCE_SHA256_FIELD,
    )
    turn_rows = _row_values(
        _required(non_tensors, "__num_turns__", "DataProto.non_tensor_batch"),
        batch_size,
        "__num_turns__",
    )

    validated_handles: list[tuple[ObservationHandle, ...]] = []
    validated_behavior_handles: list[tuple[BehaviorTraceHandle, ...]] = []
    validated_behavior_records: list[tuple[BehaviorTraceRecord, ...]] = []
    validated_logprobs: list[tuple[float, ...]] = []
    validated_sentinels: list[Mapping[str, object]] = []
    validated_trajectories: list[TrajectoryRecord] = []
    validated_replays: list[TrajectoryReplayHandle] = []
    validated_trajectory_ids: list[str] = []
    validated_trajectory_shas: list[str] = []
    validated_ownership_shas: list[str] = []
    validated_provenance_shas: list[str] = []
    for row_index in range(batch_size):
        exact_prompt_ids = tuple(exact_prompt_rows[row_index])
        exact_response_ids = tuple(exact_response_rows[row_index])
        if not exact_prompt_ids or len(exact_prompt_ids) > prompts.shape[1]:
            raise ValueError("exact prompt token sidecar is malformed")
        if not exact_response_ids or len(exact_response_ids) > width:
            raise ValueError("exact response token sidecar is malformed")
        prompt_tensor_values = tuple(
            int(value)
            for value in prompts[row_index, -len(exact_prompt_ids) :].tolist()
        )
        if prompt_tensor_values != exact_prompt_ids:
            raise ValueError("prompts tensor differs from exact prompt token sidecar")
        response_tensor_values = tuple(
            int(value)
            for value in responses[row_index, : len(exact_response_ids)].tolist()
        )
        if response_tensor_values != exact_response_ids:
            raise ValueError(
                "responses tensor differs from exact trajectory token order"
            )
        handles = tuple(
            _validate_handle(handle) for handle in tuple(handle_rows[row_index])
        )
        behavior_handles = tuple(behavior_handle_rows[row_index])
        behavior_records = tuple(behavior_record_rows[row_index])
        exact_values = tuple(exact_logprob_rows[row_index])
        if len(exact_values) != len(exact_response_ids) or any(
            not math.isfinite(value) for value in exact_values
        ):
            raise ValueError("exact response_logprobs sidecar is malformed")
        tensor_row = rollout_log_probs[row_index]
        expected = torch.tensor(
            exact_values, dtype=tensor_row.dtype, device=tensor_row.device
        )
        if not torch.equal(tensor_row[: len(exact_values)], expected):
            raise ValueError(
                "rollout_log_probs tensor differs from the recorded response_logprobs"
            )
        if len(exact_values) < width:
            if not bool((tensor_row[len(exact_values) :] == 0).all().item()):
                raise ValueError("padded rollout_log_probs must be zero")
            if not bool(
                (response_mask[row_index, len(exact_values) :] == 0).all().item()
            ):
                raise ValueError(
                    "padded response tokens must be excluded by response_mask"
                )
        sentinels = validate_objective_sentinels(sentinel_rows[row_index])
        trajectory = trajectory_rows[row_index]
        if not isinstance(trajectory, TrajectoryRecord):
            raise TypeError("trajectory sidecar must be a complete TrajectoryRecord")
        replay_handle = replay_handle_rows[row_index]
        if not isinstance(replay_handle, TrajectoryReplayHandle):
            raise TypeError("replay sidecar must be a TrajectoryReplayHandle")
        bridge_record = _mint_rollout_bridge_record(
            prompt_ids=exact_prompt_ids,
            response_ids=exact_response_ids,
            response_mask=tuple(
                int(value)
                for value in response_mask[
                    row_index, : len(exact_response_ids)
                ].tolist()
            ),
            response_logprobs=exact_values,
            exact_observation_handles=handles,
            behavior_trace_handles=behavior_handles,
            behavior_trace_records=behavior_records,
            sentinel_fields=sentinels,
            num_turns=int(turn_rows[row_index]),
            trajectory_id=trajectory_id_rows[row_index],
            trajectory_sha256=trajectory_sha_rows[row_index],
            replay_handle=replay_handle,
            token_ownership_sha256=ownership_sha_rows[row_index],
            rollout_provenance_sha256=provenance_sha_rows[row_index],
            trajectory_payload=trajectory,
        )
        validated_handles.append(handles)
        validated_behavior_handles.append(bridge_record.behavior_trace_handles)
        validated_behavior_records.append(bridge_record.behavior_trace_records)
        validated_logprobs.append(exact_values)
        validated_sentinels.append(sentinels)
        validated_trajectories.append(trajectory)
        validated_replays.append(replay_handle)
        validated_trajectory_ids.append(bridge_record.trajectory_id)
        validated_trajectory_shas.append(bridge_record.trajectory_sha256)
        validated_ownership_shas.append(bridge_record.token_ownership_sha256)
        validated_provenance_shas.append(bridge_record.rollout_provenance_sha256)

    return DataProtoIntegrityView(
        observation_handles=tuple(validated_handles),
        behavior_trace_handles=tuple(validated_behavior_handles),
        behavior_trace_records=tuple(validated_behavior_records),
        actual_response_logprobs=tuple(validated_logprobs),
        objective_sentinels=tuple(validated_sentinels),
        trajectory_payloads=tuple(validated_trajectories),
        replay_handles=tuple(validated_replays),
        trajectory_ids=tuple(validated_trajectory_ids),
        trajectory_sha256s=tuple(validated_trajectory_shas),
        token_ownership_sha256s=tuple(validated_ownership_shas),
        rollout_provenance_sha256s=tuple(validated_provenance_shas),
    )
