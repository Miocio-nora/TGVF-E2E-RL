"""Public rollout bridge facade and veRL manager composition.

The immutable wire contract lives in :mod:`rollout_contract`.  Keeping the
manager composition here preserves the established import surface while making
the runtime dependency on :mod:`data_bridge` explicit and one-way.
"""

from __future__ import annotations

import inspect
from typing import Any

from .compatibility import (
    SPIKE_CANDIDATE_VERL_COMMIT,
    TORCH211_CANDIDATE_VERL_COMMIT,
    VERL_AGENT_LOOP_RETURN_TRANSPORT,
    VERL_AGENT_LOOP_TRANSFER_QUEUE_TRANSPORT,
    VerlCompatibilityError,
    VerlPublicAPI,
    load_verl_public_api,
)
from .data_bridge import (
    bind_agent_loop_data_proto_sidecar_lease,
    compact_agent_loop_data_proto_response_width,
    validate_data_proto_integrity,
)
from .rollout_contract import (
    ACTUAL_RESPONSE_LOGPROBS_FIELD as ACTUAL_RESPONSE_LOGPROBS_FIELD,
    AGENT_LOOP_EXACT_SIDECAR_FIELDS as AGENT_LOOP_EXACT_SIDECAR_FIELDS,
    BEHAVIOR_TRACE_HANDLES_FIELD as BEHAVIOR_TRACE_HANDLES_FIELD,
    BEHAVIOR_TRACE_RECORDS_FIELD as BEHAVIOR_TRACE_RECORDS_FIELD,
    BRIDGE_SCHEMA_FIELD as BRIDGE_SCHEMA_FIELD,
    BRIDGE_SCHEMA_VERSION as BRIDGE_SCHEMA_VERSION,
    DATAPROTO_META_SCHEMA_FIELD as DATAPROTO_META_SCHEMA_FIELD,
    DATAPROTO_META_SCHEMA_VERSION as DATAPROTO_META_SCHEMA_VERSION,
    EXACT_OBSERVATION_HANDLES_FIELD as EXACT_OBSERVATION_HANDLES_FIELD,
    EXACT_PROMPT_IDS_FIELD as EXACT_PROMPT_IDS_FIELD,
    EXACT_RESPONSE_IDS_FIELD as EXACT_RESPONSE_IDS_FIELD,
    OBJECTIVE_SENTINELS_FIELD as OBJECTIVE_SENTINELS_FIELD,
    ROLLOUT_PROVENANCE_SHA256_FIELD as ROLLOUT_PROVENANCE_SHA256_FIELD,
    SIDECAR_RELEASE_FIELDS_FIELD as SIDECAR_RELEASE_FIELDS_FIELD,
    SIDECAR_RELEASE_SCHEMA_FIELD as SIDECAR_RELEASE_SCHEMA_FIELD,
    SIDECAR_RELEASE_SCHEMA_VERSION as SIDECAR_RELEASE_SCHEMA_VERSION,
    TOKEN_OWNERSHIP_SHA256_FIELD as TOKEN_OWNERSHIP_SHA256_FIELD,
    TRAJECTORY_ID_FIELD as TRAJECTORY_ID_FIELD,
    TRAJECTORY_PAYLOAD_FIELD as TRAJECTORY_PAYLOAD_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD as TRAJECTORY_REPLAY_BUNDLE_FIELD,
    TRAJECTORY_REPLAY_HANDLE_FIELD as TRAJECTORY_REPLAY_HANDLE_FIELD,
    TRAJECTORY_SHA256_FIELD as TRAJECTORY_SHA256_FIELD,
    RolloutBridgeRecord as RolloutBridgeRecord,
    _BRIDGE_MINT_TOKEN as _BRIDGE_MINT_TOKEN,
    _RESERVED_EXTRA_FIELDS as _RESERVED_EXTRA_FIELDS,
    _field as _field,
    _json_sha256 as _json_sha256,
    _mint_rollout_bridge_record as _mint_rollout_bridge_record,
    _trajectory_response_materialization as _trajectory_response_materialization,
    _validate_handle as _validate_handle,
    _validate_lower_sha256 as _validate_lower_sha256,
    _validate_replay_handle as _validate_replay_handle,
    build_agent_loop_output as build_agent_loop_output,
    parse_agent_loop_output as parse_agent_loop_output,
    rollout_provenance_checksum as rollout_provenance_checksum,
    token_ownership_checksum as token_ownership_checksum,
    trajectory_to_rollout_bridge as trajectory_to_rollout_bridge,
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

        return _map_maybe_awaitable(
            generated,
            lambda output: _validate_and_return(
                bind_agent_loop_data_proto_sidecar_lease(
                    compact_agent_loop_data_proto_response_width(output)
                ),
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
