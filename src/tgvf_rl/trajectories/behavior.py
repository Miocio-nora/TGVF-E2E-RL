"""Content-addressed evidence for tokens sampled by the vLLM behavior policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Mapping

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion, _validate_sha256
from tgvf_rl.contracts.tokens import (
    BehaviorLogProbBlock,
    OwnedTokenSequence,
    SamplingIdentity,
)


BEHAVIOR_TRACE_SCHEMA_VERSION = "vllm-behavior-trace-v1"
_TRACE_ID_PREFIX = "behavior-sha256:"


@dataclass(frozen=True, slots=True)
class BehaviorTraceHandle:
    """A content address, not a caller-selected rollout label."""

    trace_id: str
    record_sha256: str

    def __post_init__(self) -> None:
        _validate_sha256(self.record_sha256)
        if self.trace_id != f"{_TRACE_ID_PREFIX}{self.record_sha256}":
            raise ValueError("behavior trace ID must be derived from its SHA256")


@dataclass(frozen=True, slots=True)
class BehaviorTraceRecord:
    """Immutable proof of what vLLM sampled and under which probability measure."""

    schema_version: str
    trajectory_id: str
    assistant_turn_index: int
    tokens: OwnedTokenSequence
    behavior_policy: PolicyVersion
    behavior: BehaviorLogProbBlock
    backend_request_sha256: str
    backend_response_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != BEHAVIOR_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported behavior trace schema")
        if not self.trajectory_id:
            raise ValueError("behavior trace trajectory ID must be non-empty")
        if self.assistant_turn_index < 0:
            raise ValueError("assistant turn index must be non-negative")
        _validate_sha256(self.backend_request_sha256)
        _validate_sha256(self.backend_response_sha256)
        policy_indices = self.tokens.policy_indices
        if policy_indices != self.behavior.sampled_token_indices:
            raise ReplayMismatchError(
                "behavior trace ownership and sampled token indices differ"
            )
        sampled_ids = tuple(self.tokens.token_ids[index] for index in policy_indices)
        if sampled_ids != self.behavior.sampled_token_ids:
            raise ReplayMismatchError(
                "behavior trace ownership and sampled token IDs differ"
            )
        if self.behavior.sampling.policy_version != self.behavior_policy:
            raise IdentityMismatchError(
                "behavior trace sampling and policy versions differ"
            )

    @property
    def ownership_sha256(self) -> str:
        payload = {
            "schema": "behavior-token-ownership-v1",
            "token_ids": self.tokens.token_ids,
            "ownership": tuple(owner.value for owner in self.tokens.ownership),
        }
        return _json_sha256(payload)


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        _canonical(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def behavior_trace_checksum(record: BehaviorTraceRecord) -> str:
    return _json_sha256(asdict(record))


def verify_behavior_trace_pair(
    handle: BehaviorTraceHandle, record: BehaviorTraceRecord
) -> None:
    digest = behavior_trace_checksum(record)
    if (
        handle.record_sha256 != digest
        or handle.trace_id != f"{_TRACE_ID_PREFIX}{digest}"
    ):
        raise ReplayMismatchError("behavior trace handle/content checksum mismatch")


class BehaviorTraceStore:
    """Content-addressed behavior evidence minted only by a recorder boundary."""

    def __init__(self) -> None:
        self._records: dict[str, BehaviorTraceRecord] = {}

    def _put_from_recorder(self, record: BehaviorTraceRecord) -> BehaviorTraceHandle:
        digest = behavior_trace_checksum(record)
        existing = self._records.get(digest)
        if existing is not None and existing != record:
            raise IdentityMismatchError("SHA256 collision for behavior trace")
        self._records[digest] = record
        return BehaviorTraceHandle(f"{_TRACE_ID_PREFIX}{digest}", digest)

    def resolve(self, handle: BehaviorTraceHandle) -> BehaviorTraceRecord:
        try:
            record = self._records[handle.record_sha256]
        except KeyError as error:
            raise ReplayMismatchError("unknown behavior trace") from error
        verify_behavior_trace_pair(handle, record)
        return record

    def checkpoint_state(self) -> dict[str, object]:
        return {"records": dict(self._records)}

    @classmethod
    def from_checkpoint_state(cls, state: Mapping[str, object]) -> "BehaviorTraceStore":
        records = state.get("records")
        if not isinstance(records, dict):
            raise ReplayMismatchError("malformed behavior-trace checkpoint")
        store = cls()
        for digest, record in records.items():
            if not isinstance(digest, str) or not isinstance(
                record, BehaviorTraceRecord
            ):
                raise ReplayMismatchError("invalid behavior-trace checkpoint entry")
            if digest != behavior_trace_checksum(record):
                raise ReplayMismatchError("behavior-trace checkpoint checksum mismatch")
            store._records[digest] = record
        return store


class VLLMBehaviorRecorder:
    """The sole minting boundary for actual vLLM behavior probabilities."""

    def __init__(self, store: BehaviorTraceStore) -> None:
        self.store = store

    def record(
        self,
        *,
        trajectory_id: str,
        assistant_turn_index: int,
        tokens: OwnedTokenSequence,
        actual_sampled_logprobs: tuple[float, ...],
        sampling: SamplingIdentity,
        behavior_policy: PolicyVersion,
        backend_request_sha256: str,
        backend_response_sha256: str,
    ) -> BehaviorTraceHandle:
        if sampling.backend.lower() != "vllm":
            raise ValueError("VLLMBehaviorRecorder accepts only vLLM samples")
        policy_indices = tokens.policy_indices
        block = BehaviorLogProbBlock(
            sampled_token_indices=policy_indices,
            sampled_token_ids=tuple(
                tokens.token_ids[index] for index in policy_indices
            ),
            logprobs=tuple(actual_sampled_logprobs),
            sampling=sampling,
        )
        record = BehaviorTraceRecord(
            schema_version=BEHAVIOR_TRACE_SCHEMA_VERSION,
            trajectory_id=trajectory_id,
            assistant_turn_index=assistant_turn_index,
            tokens=tokens,
            behavior_policy=behavior_policy,
            behavior=block,
            backend_request_sha256=backend_request_sha256,
            backend_response_sha256=backend_response_sha256,
        )
        return self.store._put_from_recorder(record)
