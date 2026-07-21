"""DataProto transport for exact log probabilities and observation handles."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from threading import RLock
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import torch

from tgvf_rl.contracts.tokens import TokenOwnership
from tgvf_rl.observations.store import (
    ObservationHandle,
    TrajectoryReplayBundle,
    TrajectoryReplayHandle,
    validate_replay_bundle,
)
from tgvf_rl.trajectories.behavior import BehaviorTraceHandle, BehaviorTraceRecord
from tgvf_rl.trajectories.schema import TrajectoryRecord

from .compatibility import load_verl_public_api
from .objective_bridge import validate_objective_sentinels
from .rollout_bridge import (
    ACTUAL_RESPONSE_LOGPROBS_FIELD,
    AGENT_LOOP_EXACT_SIDECAR_FIELDS,
    DATAPROTO_META_SCHEMA_FIELD,
    DATAPROTO_META_SCHEMA_VERSION,
    BEHAVIOR_TRACE_HANDLES_FIELD,
    BEHAVIOR_TRACE_RECORDS_FIELD,
    BRIDGE_SCHEMA_FIELD,
    BRIDGE_SCHEMA_VERSION,
    EXACT_PROMPT_IDS_FIELD,
    EXACT_RESPONSE_IDS_FIELD,
    EXACT_OBSERVATION_HANDLES_FIELD,
    OBJECTIVE_SENTINELS_FIELD,
    ROLLOUT_PROVENANCE_SHA256_FIELD,
    SIDECAR_RELEASE_FIELDS_FIELD,
    SIDECAR_RELEASE_SCHEMA_FIELD,
    SIDECAR_RELEASE_SCHEMA_VERSION,
    TOKEN_OWNERSHIP_SHA256_FIELD,
    TRAJECTORY_ID_FIELD,
    TRAJECTORY_PAYLOAD_FIELD,
    TRAJECTORY_REPLAY_HANDLE_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
    TRAJECTORY_SHA256_FIELD,
    RolloutBridgeRecord,
    _mint_rollout_bridge_record,
    _trajectory_response_materialization,
    _validate_handle,
)


VARIABLE_LENGTH_PADDING_SCHEMA_VERSION = "tgvf-verl-variable-padding-v1"
PADDING_SCHEMA_FIELD = "tgvf_padding_schema_version"
PAD_TOKEN_ID_FIELD = "tgvf_explicit_pad_token_id"
PROMPT_TOKEN_OWNERSHIP_FIELD = "tgvf_batched_prompt_token_ownership"
RESPONSE_TOKEN_OWNERSHIP_FIELD = "tgvf_batched_response_token_ownership"
_PADDING_FIELD_ORDER = (
    PADDING_SCHEMA_FIELD,
    PAD_TOKEN_ID_FIELD,
    PROMPT_TOKEN_OWNERSHIP_FIELD,
    RESPONSE_TOKEN_OWNERSHIP_FIELD,
)
_PADDING_FIELDS = frozenset(_PADDING_FIELD_ORDER)

_SIDECAR_RELEASE_SCHEMA_VERSION = SIDECAR_RELEASE_SCHEMA_VERSION
_SIDECAR_RELEASE_SCHEMA_FIELD = SIDECAR_RELEASE_SCHEMA_FIELD
_SIDECAR_RELEASE_FIELDS_FIELD = SIDECAR_RELEASE_FIELDS_FIELD


@dataclass(slots=True)
class _TrackedVerlDataProto:
    data: object
    non_tensor_batch: dict[str, object]
    field_values: dict[str, object]


@dataclass(frozen=True, slots=True)
class DataProtoPayload:
    """Neutral constructor payload so CPU tests do not need veRL/TensorDict."""

    tensor_batch: Mapping[str, torch.Tensor]
    non_tensor_batch: Mapping[str, object]
    meta_info: Mapping[str, object]
    _non_tensor_storage: dict[str, object] = field(
        init=False, repr=False, compare=False
    )
    _sidecar_lock: RLock = field(init=False, repr=False, compare=False)
    _sidecars_released: bool = field(
        init=False, repr=False, compare=False, default=False
    )
    _tracked_verl_data: list[_TrackedVerlDataProto] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        tensor_storage = dict(self.tensor_batch)
        non_tensor_storage = dict(self.non_tensor_batch)
        meta_storage = dict(self.meta_info)
        object.__setattr__(self, "tensor_batch", MappingProxyType(tensor_storage))
        object.__setattr__(
            self, "non_tensor_batch", MappingProxyType(non_tensor_storage)
        )
        object.__setattr__(self, "meta_info", MappingProxyType(meta_storage))
        object.__setattr__(self, "_non_tensor_storage", non_tensor_storage)
        object.__setattr__(self, "_sidecar_lock", RLock())
        object.__setattr__(self, "_sidecars_released", False)
        object.__setattr__(self, "_tracked_verl_data", [])

    @property
    def sidecars_released(self) -> bool:
        with self._sidecar_lock:
            return self._sidecars_released

    def assert_sidecars_available(self) -> None:
        with self._sidecar_lock:
            self._assert_sidecars_available_locked()

    def _assert_sidecars_available_locked(self) -> None:
        if self._sidecars_released:
            raise RuntimeError("DataProto exact-replay sidecars have been released")
        for tracked in self._tracked_verl_data:
            if getattr(tracked.data, "non_tensor_batch", None) is not (
                tracked.non_tensor_batch
            ):
                raise RuntimeError(
                    "tracked veRL DataProto replaced its exact-replay sidecar mapping"
                )
            for name, expected in tracked.field_values.items():
                if tracked.non_tensor_batch.get(name) is not expected:
                    raise RuntimeError(
                        "tracked veRL DataProto changed an exact-replay sidecar "
                        f"before lifecycle release: {name!r}"
                    )

    def _materialize_verl_data_proto(self, from_dict: Any) -> Any:
        """Construct and lease the local public DataProto under one lock."""

        with self._sidecar_lock:
            self._assert_sidecars_available_locked()
            non_tensors = dict(self.non_tensor_batch)
            meta_info = dict(self.meta_info)
            existing_meta_schema = meta_info.get(DATAPROTO_META_SCHEMA_FIELD)
            if existing_meta_schema not in {
                None,
                DATAPROTO_META_SCHEMA_VERSION,
            }:
                raise ValueError("DataProto meta schema identity was changed")
            meta_info[DATAPROTO_META_SCHEMA_FIELD] = DATAPROTO_META_SCHEMA_VERSION
            reserved = {
                _SIDECAR_RELEASE_SCHEMA_FIELD,
                _SIDECAR_RELEASE_FIELDS_FIELD,
            }
            collisions = reserved & set(meta_info)
            if collisions:
                raise ValueError(
                    "DataProto meta_info collides with sidecar release fields: "
                    f"{sorted(collisions)!r}"
                )
            meta_info[_SIDECAR_RELEASE_SCHEMA_FIELD] = _SIDECAR_RELEASE_SCHEMA_VERSION
            meta_info[_SIDECAR_RELEASE_FIELDS_FIELD] = tuple(non_tensors)
            data = from_dict(
                tensors=dict(self.tensor_batch),
                non_tensors=non_tensors,
                meta_info=meta_info,
            )
            production = getattr(data, "non_tensor_batch", None)
            if type(production) is not dict:
                non_tensors.clear()
                raise TypeError(
                    "public DataProto.non_tensor_batch must remain a mutable "
                    "built-in dict for exact-replay lifecycle release"
                )
            if set(non_tensors) != set(production):
                non_tensors.clear()
                production.clear()
                raise RuntimeError(
                    "public DataProto changed sidecar fields during construction"
                )
            _sidecar_release_fields(data)
            self._tracked_verl_data.append(
                _TrackedVerlDataProto(
                    data=data,
                    non_tensor_batch=production,
                    field_values={name: production[name] for name in production},
                )
            )
            return data

    def release_sidecars(self) -> bool:
        """Drop transient exact-replay objects after the batch update barrier.

        Tensor fields contain only ordinary token/log-probability batch tensors;
        the potentially large replay bundles live in ``non_tensor_batch``.
        Returning ``False`` on a repeated call makes cleanup idempotent.
        """

        with self._sidecar_lock:
            if self._sidecars_released:
                return False
            self._assert_sidecars_available_locked()
            for tracked in self._tracked_verl_data:
                release_verl_data_proto_sidecars(tracked.data)
            self._non_tensor_storage.clear()
            self._tracked_verl_data.clear()
            object.__setattr__(self, "_sidecars_released", True)
            return True


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
    replay_bundles: tuple[TrajectoryReplayBundle, ...]
    trajectory_ids: tuple[str, ...]
    trajectory_sha256s: tuple[str, ...]
    token_ownership_sha256s: tuple[str, ...]
    rollout_provenance_sha256s: tuple[str, ...]
    pad_token_id: int | None
    prompt_token_ownership: tuple[tuple[TokenOwnership, ...], ...]
    response_token_ownership: tuple[tuple[TokenOwnership, ...], ...]


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
    """Build the legacy unmodified equal-width batch."""

    rows = _validated_rows(records)
    prompt_widths = {len(row.prompt_ids) for row in rows}
    response_widths = {len(row.response_ids) for row in rows}
    if len(prompt_widths) != 1 or len(response_widths) != 1:
        raise ValueError("the neutral bridge never pads or truncates rollout tokens")

    return _build_payload(
        rows,
        prompt_rows=[row.prompt_ids for row in rows],
        response_rows=[row.response_ids for row in rows],
        response_mask_rows=[row.response_mask for row in rows],
        response_logprob_rows=[row.response_logprobs for row in rows],
    )


def build_padded_data_proto_payload(
    records: Iterable[RolloutBridgeRecord],
    *,
    pad_token_id: int,
) -> DataProtoPayload:
    """Losslessly batch variable lengths with explicit left/right padding.

    Prompts are left-padded and responses are right-padded.  The padding token
    is a caller-owned run binding: this function never reads tokenizer EOS or
    padding defaults.  Exact token/logprob/replay sidecars remain untouched.
    """

    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be an explicit non-negative integer")
    rows = _validated_rows(records)
    prompt_width = max(len(row.prompt_ids) for row in rows)
    response_width = max(len(row.response_ids) for row in rows)
    prompt_rows = [
        (pad_token_id,) * (prompt_width - len(row.prompt_ids)) + row.prompt_ids
        for row in rows
    ]
    response_rows = [
        row.response_ids + (pad_token_id,) * (response_width - len(row.response_ids))
        for row in rows
    ]
    response_mask_rows = [
        row.response_mask + (0,) * (response_width - len(row.response_mask))
        for row in rows
    ]
    response_logprob_rows = [
        row.response_logprobs + (0.0,) * (response_width - len(row.response_logprobs))
        for row in rows
    ]
    prompt_ownership_rows = [
        (TokenOwnership.PADDING.value,) * (prompt_width - len(row.prompt_ids))
        + (TokenOwnership.TEMPLATE.value,) * len(row.prompt_ids)
        for row in rows
    ]
    response_ownership_rows = []
    for row in rows:
        _, _, _, exact_ownership = _trajectory_response_materialization(
            row.trajectory_payload, row.behavior_trace_records
        )
        response_ownership_rows.append(
            exact_ownership
            + (TokenOwnership.PADDING.value,) * (response_width - len(row.response_ids))
        )
    return _build_payload(
        rows,
        prompt_rows=prompt_rows,
        response_rows=response_rows,
        response_mask_rows=response_mask_rows,
        response_logprob_rows=response_logprob_rows,
        padding_fields={
            PADDING_SCHEMA_FIELD: _object_array(
                [VARIABLE_LENGTH_PADDING_SCHEMA_VERSION for _ in rows]
            ),
            PAD_TOKEN_ID_FIELD: _object_array([pad_token_id for _ in rows]),
            PROMPT_TOKEN_OWNERSHIP_FIELD: _object_array(prompt_ownership_rows),
            RESPONSE_TOKEN_OWNERSHIP_FIELD: _object_array(response_ownership_rows),
        },
    )


def _validated_rows(
    records: Iterable[RolloutBridgeRecord],
) -> tuple[RolloutBridgeRecord, ...]:
    rows = tuple(records)
    if not rows:
        raise ValueError("at least one rollout record is required")
    if any(not isinstance(row, RolloutBridgeRecord) for row in rows):
        raise TypeError("all rows must be RolloutBridgeRecord values")
    return rows


def _build_payload(
    rows: tuple[RolloutBridgeRecord, ...],
    *,
    prompt_rows: list[tuple[int, ...]],
    response_rows: list[tuple[int, ...]],
    response_mask_rows: list[tuple[int, ...]],
    response_logprob_rows: list[tuple[float, ...]],
    padding_fields: Mapping[str, object] | None = None,
) -> DataProtoPayload:
    """Assemble tensors while preserving every unpadded sidecar verbatim."""

    tensors = {
        "prompts": torch.tensor(prompt_rows, dtype=torch.int64),
        "responses": torch.tensor(response_rows, dtype=torch.int64),
        "response_mask": torch.tensor(response_mask_rows, dtype=torch.int64),
        # Match AgentLoopOutput.as_dict/_postprocess.  The exact Python values
        # remain alongside this public tensor so float32 transport is auditable.
        "rollout_log_probs": torch.tensor(response_logprob_rows, dtype=torch.float32),
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
        TRAJECTORY_REPLAY_BUNDLE_FIELD: _object_array(
            [row.replay_bundle for row in rows]
        ),
        TOKEN_OWNERSHIP_SHA256_FIELD: _object_array(
            [row.token_ownership_sha256 for row in rows]
        ),
        ROLLOUT_PROVENANCE_SHA256_FIELD: _object_array(
            [row.rollout_provenance_sha256 for row in rows]
        ),
        "__num_turns__": _object_array([row.num_turns for row in rows]),
    }
    if padding_fields is not None:
        collisions = set(padding_fields) & set(non_tensors)
        if collisions:
            raise RuntimeError(
                f"padding fields collide with bridge fields: {collisions}"
            )
        non_tensors.update(padding_fields)
    extra_names = sorted({name for row in rows for name in row.extra_fields})
    padding_collisions = _PADDING_FIELDS & set(extra_names)
    if padding_collisions:
        raise ValueError(
            "rollout extra_fields collide with variable-padding fields: "
            f"{sorted(padding_collisions)}"
        )
    for name in extra_names:
        non_tensors[name] = _object_array([row.extra_fields.get(name) for row in rows])
    return DataProtoPayload(
        tensor_batch=tensors,
        non_tensor_batch=non_tensors,
        meta_info={DATAPROTO_META_SCHEMA_FIELD: DATAPROTO_META_SCHEMA_VERSION},
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
    return payload._materialize_verl_data_proto(from_dict)


def compact_agent_loop_data_proto_response_width(data: object) -> object:
    """Trim the upstream transport envelope to this batch's exact width.

    Pinned veRL must receive a finite response envelope large enough to avoid
    truncating environment-owned tool tokens.  Its AgentLoop manager pads every
    row to that envelope, however, so forwarding the result unchanged would
    make actor/reference replay pay for the worst-case context on every batch.
    Exact response sidecars let this boundary shrink all aligned tensors before
    worker dispatch without guessing from token values or attention masks.
    """

    batch, non_tensors = _data_parts(data)
    keys = set(batch.keys())
    known = {
        "prompts",
        "responses",
        "response_mask",
        "rollout_log_probs",
        "input_ids",
        "attention_mask",
        "position_ids",
        "rm_scores",
        "routed_experts",
        "teacher_ids",
        "teacher_logprobs",
    }
    unknown = sorted(str(name) for name in keys - known)
    if unknown:
        raise ValueError(
            "live AgentLoop DataProto has unknown tensors whose response-axis "
            f"alignment cannot be compacted safely: {unknown!r}"
        )
    required = {
        "prompts",
        "responses",
        "response_mask",
        "rollout_log_probs",
        "input_ids",
        "attention_mask",
        "position_ids",
    }
    missing = sorted(required - keys)
    if missing:
        raise ValueError(
            f"live AgentLoop DataProto lacks compaction tensors: {missing!r}"
        )

    prompts = batch["prompts"]
    responses = batch["responses"]
    if (
        not isinstance(prompts, torch.Tensor)
        or not isinstance(responses, torch.Tensor)
        or prompts.ndim != 2
        or responses.ndim != 2
        or prompts.shape[0] != responses.shape[0]
    ):
        raise ValueError(
            "live AgentLoop prompts/responses must be aligned rank-two tensors"
        )
    batch_size = int(responses.shape[0])
    prompt_width = int(prompts.shape[1])
    transport_width = int(responses.shape[1])
    exact_rows = _row_values(
        _required(non_tensors, EXACT_RESPONSE_IDS_FIELD, "DataProto.non_tensor_batch"),
        batch_size,
        EXACT_RESPONSE_IDS_FIELD,
    )
    exact_lengths = tuple(len(tuple(row)) for row in exact_rows)
    if any(length <= 0 or length > transport_width for length in exact_lengths):
        raise ValueError("live AgentLoop exact response lengths are invalid")
    exact_width = max(exact_lengths)
    full_transport_width = prompt_width + transport_width
    full_exact_width = prompt_width + exact_width

    response_fields = ("responses", "response_mask", "rollout_log_probs", "rm_scores")
    full_fields = ("input_ids", "attention_mask")
    replacements: dict[str, torch.Tensor] = {}
    for name in response_fields:
        if name not in batch:
            continue
        value = batch[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or tuple(value.shape) != (batch_size, transport_width)
        ):
            raise ValueError(f"live AgentLoop {name} is not response-width aligned")
        replacements[name] = _clone_prefix(value, exact_width, axis=1)
    for name in full_fields:
        value = batch[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or tuple(value.shape) != (batch_size, full_transport_width)
        ):
            raise ValueError(f"live AgentLoop {name} is not full-sequence aligned")
        replacements[name] = _clone_prefix(value, full_exact_width, axis=1)

    position_ids = batch["position_ids"]
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.ndim not in {2, 3}
        or int(position_ids.shape[0]) != batch_size
        or int(position_ids.shape[-1]) != full_transport_width
    ):
        raise ValueError(
            "live AgentLoop position_ids must align on the full-sequence axis"
        )
    replacements["position_ids"] = _clone_prefix(
        position_ids, full_exact_width, axis=position_ids.ndim - 1
    )

    for name in ("routed_experts", "teacher_ids", "teacher_logprobs"):
        if name not in batch:
            continue
        value = batch[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim < 2
            or int(value.shape[0]) != batch_size
            or int(value.shape[1]) != full_transport_width
        ):
            raise ValueError(f"live AgentLoop {name} is not full-sequence aligned")
        replacements[name] = _clone_prefix(value, full_exact_width, axis=1)

    for name, value in replacements.items():
        batch[name] = value
    return data


def _clone_prefix(value: torch.Tensor, width: int, *, axis: int) -> torch.Tensor:
    if int(value.shape[axis]) == width:
        return value
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(0, width)
    return value[tuple(slices)].clone(memory_format=torch.contiguous_format)


def bind_agent_loop_data_proto_sidecar_lease(data: object) -> object:
    """Attach the exact-sidecar lease to a live AgentLoop-manager DataProto.

    Pinned veRL constructs this DataProto itself, so the neutral
    :class:`DataProtoPayload` constructor never has an opportunity to add its
    transport metadata.  The repo-owned lossless manager invokes this hook on
    the driver before validation or any TensorDict/Ray dispatch.
    """

    non_tensors = getattr(data, "non_tensor_batch", None)
    meta_info = getattr(data, "meta_info", None)
    if type(non_tensors) is not dict or type(meta_info) is not dict:
        raise TypeError(
            "live AgentLoop DataProto must expose mutable built-in mappings"
        )
    missing = tuple(
        name for name in AGENT_LOOP_EXACT_SIDECAR_FIELDS if name not in non_tensors
    )
    if missing:
        raise RuntimeError(
            f"live AgentLoop DataProto is missing exact sidecars: {missing!r}"
        )
    padding_fields = _bind_live_agent_loop_padding_contract(data)
    leased_fields = AGENT_LOOP_EXACT_SIDECAR_FIELDS + padding_fields
    existing_release_fields = meta_info.get(_SIDECAR_RELEASE_FIELDS_FIELD)
    if existing_release_fields is not None:
        existing = _sidecar_release_fields(data)
        if not set(leased_fields).issubset(existing):
            raise RuntimeError(
                "existing DataProto lease omits AgentLoop exact sidecars"
            )
        return data
    expected = {
        DATAPROTO_META_SCHEMA_FIELD: DATAPROTO_META_SCHEMA_VERSION,
        _SIDECAR_RELEASE_SCHEMA_FIELD: _SIDECAR_RELEASE_SCHEMA_VERSION,
        _SIDECAR_RELEASE_FIELDS_FIELD: leased_fields,
    }
    for name, value in expected.items():
        existing_value = meta_info.get(name)
        if existing_value is not None and existing_value != value:
            raise RuntimeError(f"live AgentLoop DataProto changed lease field {name!r}")
        meta_info[name] = value
    _sidecar_release_fields(data)
    return data


def _bind_live_agent_loop_padding_contract(data: object) -> tuple[str, ...]:
    """Describe and verify padding already materialized by pinned veRL.

    The live AgentLoop manager pads token tensors before it constructs its
    ``DataProto``, while the project-owned exact sidecars deliberately retain
    unpadded token rows.  Bind the existing tensor layout here; never repad or
    mutate a token tensor at this boundary.
    """

    batch, non_tensors = _data_parts(data)
    prompts = _required(batch, "prompts", "DataProto.batch")
    responses = _required(batch, "responses", "DataProto.batch")
    if (
        not isinstance(prompts, torch.Tensor)
        or not isinstance(responses, torch.Tensor)
        or prompts.ndim != 2
        or responses.ndim != 2
        or prompts.shape[0] != responses.shape[0]
    ):
        raise ValueError(
            "live AgentLoop prompts/responses must be aligned rank-two tensors"
        )
    batch_size = int(prompts.shape[0])
    prompt_width = int(prompts.shape[1])
    response_width = int(responses.shape[1])
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
    prompt_lengths = tuple(len(tuple(row)) for row in exact_prompt_rows)
    response_lengths = tuple(len(tuple(row)) for row in exact_response_rows)
    if any(length <= 0 or length > prompt_width for length in prompt_lengths):
        raise ValueError("live AgentLoop exact prompt lengths are invalid")
    if any(length <= 0 or length > response_width for length in response_lengths):
        raise ValueError("live AgentLoop exact response lengths are invalid")

    present = _PADDING_FIELDS & set(non_tensors)
    if present:
        if present != _PADDING_FIELDS:
            missing = sorted(_PADDING_FIELDS - present)
            raise ValueError(
                f"live AgentLoop variable-padding contract is incomplete: {missing}"
            )
        return _PADDING_FIELD_ORDER

    padding_required = any(
        prompt_length != prompt_width or response_length != response_width
        for prompt_length, response_length in zip(
            prompt_lengths, response_lengths, strict=True
        )
    )
    if not padding_required:
        return ()

    padding_values: list[int] = []
    for row_index, (prompt_length, response_length) in enumerate(
        zip(prompt_lengths, response_lengths, strict=True)
    ):
        padding_values.extend(
            int(value)
            for value in prompts[row_index, : prompt_width - prompt_length].tolist()
        )
        padding_values.extend(
            int(value) for value in responses[row_index, response_length:].tolist()
        )
    if not padding_values or len(set(padding_values)) != 1:
        raise ValueError(
            "live AgentLoop prompt/response padding must use one explicit token ID"
        )
    pad_token_id = padding_values[0]
    if pad_token_id < 0:
        raise ValueError("live AgentLoop pad token ID must be non-negative")

    trajectory_rows = _row_values(
        _required(non_tensors, TRAJECTORY_PAYLOAD_FIELD, "DataProto.non_tensor_batch"),
        batch_size,
        TRAJECTORY_PAYLOAD_FIELD,
    )
    behavior_rows = _row_values(
        _required(
            non_tensors,
            BEHAVIOR_TRACE_RECORDS_FIELD,
            "DataProto.non_tensor_batch",
        ),
        batch_size,
        BEHAVIOR_TRACE_RECORDS_FIELD,
    )
    prompt_ownership_rows: list[tuple[str, ...]] = []
    response_ownership_rows: list[tuple[str, ...]] = []
    for row_index, (prompt_length, response_length) in enumerate(
        zip(prompt_lengths, response_lengths, strict=True)
    ):
        prompt_ownership_rows.append(
            (TokenOwnership.PADDING.value,) * (prompt_width - prompt_length)
            + (TokenOwnership.TEMPLATE.value,) * prompt_length
        )
        _, _, _, exact_response_ownership = _trajectory_response_materialization(
            trajectory_rows[row_index], tuple(behavior_rows[row_index])
        )
        if len(exact_response_ownership) != response_length:
            raise ValueError(
                "live AgentLoop response ownership differs from exact response length"
            )
        response_ownership_rows.append(
            exact_response_ownership
            + (TokenOwnership.PADDING.value,) * (response_width - response_length)
        )

    bound_fields = {
        PADDING_SCHEMA_FIELD: _object_array(
            [VARIABLE_LENGTH_PADDING_SCHEMA_VERSION] * batch_size
        ),
        PAD_TOKEN_ID_FIELD: _object_array([pad_token_id] * batch_size),
        PROMPT_TOKEN_OWNERSHIP_FIELD: _object_array(prompt_ownership_rows),
        RESPONSE_TOKEN_OWNERSHIP_FIELD: _object_array(response_ownership_rows),
    }
    collisions = set(bound_fields) & set(non_tensors)
    if collisions:  # pragma: no cover - guarded by the complete/partial check above
        raise RuntimeError(
            f"live AgentLoop padding fields collide: {sorted(collisions)}"
        )
    non_tensors.update(bound_fields)
    return _PADDING_FIELD_ORDER


def _sidecar_release_fields(data: object) -> tuple[str, ...]:
    _validate_dataproto_meta_schema(data)
    meta_info = getattr(data, "meta_info", None)
    if not isinstance(meta_info, Mapping):
        raise TypeError("DataProto.meta_info must preserve sidecar release metadata")
    if meta_info.get(_SIDECAR_RELEASE_SCHEMA_FIELD) != (
        _SIDECAR_RELEASE_SCHEMA_VERSION
    ):
        raise RuntimeError("DataProto sidecar release schema is missing or changed")
    fields = meta_info.get(_SIDECAR_RELEASE_FIELDS_FIELD)
    if (
        not isinstance(fields, tuple)
        or not fields
        or any(not isinstance(name, str) or not name for name in fields)
        or len(set(fields)) != len(fields)
    ):
        raise RuntimeError("DataProto sidecar release field lease is malformed")
    return fields


def release_verl_data_proto_sidecars(data: object) -> int:
    """Release one local/worker DataProto's project-owned sidecar references.

    Ray-deserialized DataProto copies are separate owners.  Their worker must
    call this function in its own ``finally`` block; releasing the driver
    ``DataProtoPayload`` cannot reach remote-process copies.
    """

    non_tensors = getattr(data, "non_tensor_batch", None)
    if type(non_tensors) is not dict:
        raise TypeError(
            "DataProto.non_tensor_batch must be a mutable built-in dict for release"
        )
    fields = _sidecar_release_fields(data)
    present = tuple(name for name in fields if name in non_tensors)
    if not present:
        return 0
    if len(present) != len(fields):
        missing = tuple(name for name in fields if name not in non_tensors)
        raise RuntimeError(
            "DataProto sidecars were partially released outside the lifecycle: "
            f"missing={missing!r}"
        )
    for name in fields:
        del non_tensors[name]
    return len(fields)


@contextmanager
def worker_data_proto_sidecar_scope(data: object) -> Iterator[object]:
    """Validate and release one worker-local DataProto in ``finally``.

    Ray serialization creates a new owner of every object in
    ``non_tensor_batch``.  The driver's retained :class:`DataProtoPayload`
    cannot release that copy.  A veRL worker entry point must therefore wrap
    its complete use of the deserialized DataProto in this scope.  Validation
    happens before the consumer runs, and cleanup is attempted for success,
    consumer failure, and validation failure alike.

    If both the consumer and cleanup fail, the consumer exception remains the
    primary error and receives a cleanup note.  This prevents a release defect
    from hiding the failure that may have left optimizer commit status unknown.
    """

    active_error: BaseException | None = None
    try:
        validate_data_proto_integrity(data)
        yield data
    except BaseException as error:
        active_error = error
        raise
    finally:
        try:
            release_verl_data_proto_sidecars(data)
        except BaseException as cleanup_error:
            if active_error is None:
                raise
            active_error.add_note(
                "veRL worker DataProto sidecar cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


def release_verl_worker_tensordict_sidecars(data: object) -> int:
    """Release exact-replay objects after DataProto-to-TensorDict dispatch.

    The pinned veRL driver converts a DataProto to TensorDict before Ray actor
    and reference calls.  Its conversion carries the release lease through
    non-tensor metadata, so the worker can delete exactly the project-owned
    fields without touching ordinary tensors or worker metrics.
    """

    if not hasattr(data, "__contains__") or not hasattr(data, "__delitem__"):
        raise TypeError("veRL worker TensorDict must be mutable and mapping-like")
    schema = _worker_non_tensor_value(
        data[DATAPROTO_META_SCHEMA_FIELD]
        if DATAPROTO_META_SCHEMA_FIELD in data
        else None
    )
    if schema != DATAPROTO_META_SCHEMA_VERSION:
        raise RuntimeError("worker TensorDict lost the DataProto meta schema")
    release_schema = _worker_non_tensor_value(
        data[_SIDECAR_RELEASE_SCHEMA_FIELD]
        if _SIDECAR_RELEASE_SCHEMA_FIELD in data
        else None
    )
    if release_schema != _SIDECAR_RELEASE_SCHEMA_VERSION:
        raise RuntimeError("worker TensorDict lost the sidecar release schema")
    fields = _worker_non_tensor_value(
        data[_SIDECAR_RELEASE_FIELDS_FIELD]
        if _SIDECAR_RELEASE_FIELDS_FIELD in data
        else None
    )
    if (
        not isinstance(fields, tuple)
        or not fields
        or any(not isinstance(name, str) or not name for name in fields)
        or len(set(fields)) != len(fields)
    ):
        raise RuntimeError("worker TensorDict sidecar release lease is malformed")
    present = tuple(name for name in fields if name in data)
    if not present:
        return 0
    if len(present) != len(fields):
        missing = tuple(name for name in fields if name not in data)
        raise RuntimeError(
            f"worker TensorDict sidecars were partially released: missing={missing!r}"
        )
    for name in fields:
        del data[name]
    return len(fields)


@contextmanager
def worker_tensordict_sidecar_scope(data: object) -> Iterator[object]:
    """Own one Ray-local TensorDict sidecar lease through a worker call."""

    active_error: BaseException | None = None
    try:
        # Validate the lease before invoking model code; exact replay performs
        # the full semantic integrity check on each microbatch.
        _worker_sidecar_fields(data)
        yield data
    except BaseException as error:
        active_error = error
        raise
    finally:
        try:
            release_verl_worker_tensordict_sidecars(data)
        except BaseException as cleanup_error:
            if active_error is None:
                raise
            active_error.add_note(
                "veRL worker TensorDict sidecar cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


def make_sidecar_releasing_training_worker_class(
    upstream_worker_cls: type[Any],
) -> type[Any]:
    """Wrap pinned veRL TrainingWorker entry points with worker ``finally``.

    The returned class is intended for ``ActorRolloutRefWorker.actor_worker_cls``
    and ``ref_worker_cls``.  All model execution remains upstream-owned; only
    the lifetime of the Ray-local exact-replay sidecars changes.
    """

    if not isinstance(upstream_worker_cls, type):
        raise TypeError("upstream TrainingWorker must be a class")
    for method_name in ("train_mini_batch", "infer_batch"):
        if not callable(getattr(upstream_worker_cls, method_name, None)):
            raise TypeError(f"upstream TrainingWorker is missing {method_name}()")

    class SidecarReleasingTrainingWorker(upstream_worker_cls):
        def train_mini_batch(self, data, *args, **kwargs):
            with worker_tensordict_sidecar_scope(data):
                return super().train_mini_batch(data, *args, **kwargs)

        def infer_batch(self, data, *args, **kwargs):
            with worker_tensordict_sidecar_scope(data):
                return super().infer_batch(data, *args, **kwargs)

    SidecarReleasingTrainingWorker.__name__ = "SidecarReleasingTrainingWorker"
    SidecarReleasingTrainingWorker.__qualname__ = "SidecarReleasingTrainingWorker"
    SidecarReleasingTrainingWorker.__module__ = __name__
    return SidecarReleasingTrainingWorker


def make_sidecar_releasing_actor_rollout_ref_worker_class(
    upstream_worker_cls: type[Any],
    *,
    upstream_training_worker_cls: type[Any] | None = None,
) -> type[Any]:
    """Bind the cleanup TrainingWorker into veRL's public role worker class."""

    if not isinstance(upstream_worker_cls, type):
        raise TypeError("upstream ActorRolloutRefWorker must be a class")
    training_cls = upstream_training_worker_cls or getattr(
        upstream_worker_cls, "actor_worker_cls", None
    )
    wrapped_training_cls = make_sidecar_releasing_training_worker_class(training_cls)

    class SidecarReleasingActorRolloutRefWorker(upstream_worker_cls):
        actor_worker_cls = wrapped_training_cls
        ref_worker_cls = wrapped_training_cls

    SidecarReleasingActorRolloutRefWorker.__name__ = (
        "SidecarReleasingActorRolloutRefWorker"
    )
    SidecarReleasingActorRolloutRefWorker.__qualname__ = (
        "SidecarReleasingActorRolloutRefWorker"
    )
    SidecarReleasingActorRolloutRefWorker.__module__ = __name__
    return SidecarReleasingActorRolloutRefWorker


def _worker_sidecar_fields(data: object) -> tuple[str, ...]:
    if not hasattr(data, "__contains__") or not hasattr(data, "__getitem__"):
        raise TypeError("veRL worker TensorDict must be mapping-like")
    schema = _worker_non_tensor_value(
        data[DATAPROTO_META_SCHEMA_FIELD]
        if DATAPROTO_META_SCHEMA_FIELD in data
        else None
    )
    if schema != DATAPROTO_META_SCHEMA_VERSION:
        raise RuntimeError("worker TensorDict lost the DataProto meta schema")
    release_schema = _worker_non_tensor_value(
        data[_SIDECAR_RELEASE_SCHEMA_FIELD]
        if _SIDECAR_RELEASE_SCHEMA_FIELD in data
        else None
    )
    if release_schema != _SIDECAR_RELEASE_SCHEMA_VERSION:
        raise RuntimeError("worker TensorDict lost the sidecar release schema")
    fields = _worker_non_tensor_value(
        data[_SIDECAR_RELEASE_FIELDS_FIELD]
        if _SIDECAR_RELEASE_FIELDS_FIELD in data
        else None
    )
    if (
        not isinstance(fields, tuple)
        or not fields
        or any(not isinstance(name, str) or not name for name in fields)
        or len(set(fields)) != len(fields)
    ):
        raise RuntimeError("worker TensorDict sidecar release lease is malformed")
    missing = tuple(name for name in fields if name not in data)
    if missing:
        raise RuntimeError(f"worker TensorDict is missing leased sidecars: {missing!r}")
    return fields


def _worker_non_tensor_value(value: object) -> object:
    unwrapped = getattr(value, "data", value)
    if unwrapped is value:
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except (TypeError, ValueError):
                return value
    return unwrapped


def build_verl_data_proto(
    records: Iterable[RolloutBridgeRecord],
    *,
    data_proto_cls: type[Any] | None = None,
) -> Any:
    """Build a standalone caller-owned DataProto.

    This convenience path has no retained ``DataProtoPayload`` owner.  The
    caller must invoke :func:`release_verl_data_proto_sidecars` in ``finally``;
    Policy Pilot lifecycle code must use the retained-payload path instead.
    """

    return to_verl_data_proto(
        build_data_proto_payload(records), data_proto_cls=data_proto_cls
    )


def build_padded_verl_data_proto(
    records: Iterable[RolloutBridgeRecord],
    *,
    pad_token_id: int,
    data_proto_cls: type[Any] | None = None,
) -> Any:
    """Construct a standalone caller-owned, explicitly padded DataProto.

    The caller owns worker/local sidecar release through
    :func:`release_verl_data_proto_sidecars`; this is not the retained Policy
    Pilot lifecycle path.
    """

    return to_verl_data_proto(
        build_padded_data_proto_payload(records, pad_token_id=pad_token_id),
        data_proto_cls=data_proto_cls,
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

    if hasattr(data, "meta_info"):
        _validate_dataproto_meta_schema(data)
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
    prompt_width = prompts.shape[1]
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
    replay_bundle_rows = _row_values(
        _required(
            non_tensors, TRAJECTORY_REPLAY_BUNDLE_FIELD, "DataProto.non_tensor_batch"
        ),
        batch_size,
        TRAJECTORY_REPLAY_BUNDLE_FIELD,
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
    present_padding_fields = _PADDING_FIELDS & set(non_tensors)
    if present_padding_fields and present_padding_fields != _PADDING_FIELDS:
        missing = sorted(_PADDING_FIELDS - present_padding_fields)
        raise ValueError(
            f"DataProto variable-padding contract is incomplete: {missing}"
        )
    padding_enabled = bool(present_padding_fields)
    pad_token_id: int | None = None
    prompt_ownership_rows: tuple[object, ...] | None = None
    response_ownership_rows: tuple[object, ...] | None = None
    if padding_enabled:
        padding_schemas = _row_values(
            _required(non_tensors, PADDING_SCHEMA_FIELD, "DataProto.non_tensor_batch"),
            batch_size,
            PADDING_SCHEMA_FIELD,
        )
        if any(
            schema != VARIABLE_LENGTH_PADDING_SCHEMA_VERSION
            for schema in padding_schemas
        ):
            raise ValueError("DataProto variable-padding schema was lost or changed")
        pad_token_rows = _row_values(
            _required(non_tensors, PAD_TOKEN_ID_FIELD, "DataProto.non_tensor_batch"),
            batch_size,
            PAD_TOKEN_ID_FIELD,
        )
        if any(type(value) is not int or value < 0 for value in pad_token_rows):
            raise ValueError("DataProto pad_token_id binding is malformed")
        if len(set(pad_token_rows)) != 1:
            raise ValueError("one DataProto batch cannot mix pad_token_id bindings")
        pad_token_id = pad_token_rows[0]
        prompt_ownership_rows = _row_values(
            _required(
                non_tensors,
                PROMPT_TOKEN_OWNERSHIP_FIELD,
                "DataProto.non_tensor_batch",
            ),
            batch_size,
            PROMPT_TOKEN_OWNERSHIP_FIELD,
        )
        response_ownership_rows = _row_values(
            _required(
                non_tensors,
                RESPONSE_TOKEN_OWNERSHIP_FIELD,
                "DataProto.non_tensor_batch",
            ),
            batch_size,
            RESPONSE_TOKEN_OWNERSHIP_FIELD,
        )

    validated_handles: list[tuple[ObservationHandle, ...]] = []
    validated_behavior_handles: list[tuple[BehaviorTraceHandle, ...]] = []
    validated_behavior_records: list[tuple[BehaviorTraceRecord, ...]] = []
    validated_logprobs: list[tuple[float, ...]] = []
    validated_sentinels: list[Mapping[str, object]] = []
    validated_trajectories: list[TrajectoryRecord] = []
    validated_replays: list[TrajectoryReplayHandle] = []
    validated_replay_bundles: list[TrajectoryReplayBundle] = []
    validated_trajectory_ids: list[str] = []
    validated_trajectory_shas: list[str] = []
    validated_ownership_shas: list[str] = []
    validated_provenance_shas: list[str] = []
    validated_prompt_ownership: list[tuple[TokenOwnership, ...]] = []
    validated_response_ownership: list[tuple[TokenOwnership, ...]] = []
    for row_index in range(batch_size):
        exact_prompt_ids = tuple(exact_prompt_rows[row_index])
        exact_response_ids = tuple(exact_response_rows[row_index])
        if not exact_prompt_ids or len(exact_prompt_ids) > prompts.shape[1]:
            raise ValueError("exact prompt token sidecar is malformed")
        if not exact_response_ids or len(exact_response_ids) > width:
            raise ValueError("exact response token sidecar is malformed")
        if not padding_enabled and (
            len(exact_prompt_ids) != prompt_width or len(exact_response_ids) != width
        ):
            raise ValueError(
                "short exact token sidecars require the explicit variable-padding "
                "contract"
            )
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
        if padding_enabled:
            assert pad_token_id is not None
            prompt_padding = prompt_width - len(exact_prompt_ids)
            if not bool(
                (prompts[row_index, :prompt_padding] == pad_token_id).all().item()
            ):
                raise ValueError(
                    "left prompt padding differs from explicit pad_token_id"
                )
            if not bool(
                (responses[row_index, len(exact_response_ids) :] == pad_token_id)
                .all()
                .item()
            ):
                raise ValueError(
                    "right response padding differs from explicit pad_token_id"
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
        replay_bundle = replay_bundle_rows[row_index]
        if not isinstance(replay_bundle, TrajectoryReplayBundle):
            raise TypeError("replay bundle sidecar must be a TrajectoryReplayBundle")
        validate_replay_bundle(replay_bundle)
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
            replay_bundle=replay_bundle,
            token_ownership_sha256=ownership_sha_rows[row_index],
            rollout_provenance_sha256=provenance_sha_rows[row_index],
            trajectory_payload=trajectory,
        )
        _, _, _, exact_response_ownership = _trajectory_response_materialization(
            bridge_record.trajectory_payload,
            bridge_record.behavior_trace_records,
        )
        expected_prompt_ownership = (TokenOwnership.PADDING,) * (
            prompt_width - len(exact_prompt_ids)
        ) + (TokenOwnership.TEMPLATE,) * len(exact_prompt_ids)
        expected_response_ownership = tuple(
            TokenOwnership(value) for value in exact_response_ownership
        ) + (TokenOwnership.PADDING,) * (width - len(exact_response_ids))
        if padding_enabled:
            assert prompt_ownership_rows is not None
            assert response_ownership_rows is not None
            actual_prompt_ownership = _ownership_row(
                prompt_ownership_rows[row_index],
                width=prompt_width,
                field_name=PROMPT_TOKEN_OWNERSHIP_FIELD,
            )
            actual_response_ownership = _ownership_row(
                response_ownership_rows[row_index],
                width=width,
                field_name=RESPONSE_TOKEN_OWNERSHIP_FIELD,
            )
            if actual_prompt_ownership != expected_prompt_ownership:
                raise ValueError("batched prompt token ownership was changed")
            if actual_response_ownership != expected_response_ownership:
                raise ValueError("batched response token ownership was changed")
        else:
            actual_prompt_ownership = expected_prompt_ownership
            actual_response_ownership = expected_response_ownership
        if any(
            owner is TokenOwnership.PADDING
            and (owner.policy_loss_mask != 0 or owner.requires_behavior_logprob)
            for owner in actual_prompt_ownership + actual_response_ownership
        ):
            raise RuntimeError("TokenOwnership.PADDING contract is internally invalid")
        validated_handles.append(handles)
        validated_behavior_handles.append(bridge_record.behavior_trace_handles)
        validated_behavior_records.append(bridge_record.behavior_trace_records)
        validated_logprobs.append(exact_values)
        validated_sentinels.append(sentinels)
        validated_trajectories.append(trajectory)
        validated_replays.append(replay_handle)
        validated_replay_bundles.append(replay_bundle)
        validated_trajectory_ids.append(bridge_record.trajectory_id)
        validated_trajectory_shas.append(bridge_record.trajectory_sha256)
        validated_ownership_shas.append(bridge_record.token_ownership_sha256)
        validated_provenance_shas.append(bridge_record.rollout_provenance_sha256)
        validated_prompt_ownership.append(actual_prompt_ownership)
        validated_response_ownership.append(actual_response_ownership)

    return DataProtoIntegrityView(
        observation_handles=tuple(validated_handles),
        behavior_trace_handles=tuple(validated_behavior_handles),
        behavior_trace_records=tuple(validated_behavior_records),
        actual_response_logprobs=tuple(validated_logprobs),
        objective_sentinels=tuple(validated_sentinels),
        trajectory_payloads=tuple(validated_trajectories),
        replay_handles=tuple(validated_replays),
        replay_bundles=tuple(validated_replay_bundles),
        trajectory_ids=tuple(validated_trajectory_ids),
        trajectory_sha256s=tuple(validated_trajectory_shas),
        token_ownership_sha256s=tuple(validated_ownership_shas),
        rollout_provenance_sha256s=tuple(validated_provenance_shas),
        pad_token_id=pad_token_id,
        prompt_token_ownership=tuple(validated_prompt_ownership),
        response_token_ownership=tuple(validated_response_ownership),
    )


def _validate_dataproto_meta_schema(data: object) -> None:
    meta_info = getattr(data, "meta_info", None)
    if not isinstance(meta_info, Mapping):
        raise TypeError("DataProto.meta_info must preserve the transport schema")
    if meta_info.get(DATAPROTO_META_SCHEMA_FIELD) != DATAPROTO_META_SCHEMA_VERSION:
        raise RuntimeError("DataProto meta transport schema was lost or changed")


def _ownership_row(
    value: object,
    *,
    width: int,
    field_name: str,
) -> tuple[TokenOwnership, ...]:
    if not hasattr(value, "__len__") or not hasattr(value, "__getitem__"):
        raise TypeError(f"{field_name} row must be an indexable sequence")
    if len(value) != width:
        raise ValueError(f"{field_name} row has the wrong width")
    try:
        return tuple(TokenOwnership(value[index]) for index in range(width))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} row contains unknown ownership") from error
