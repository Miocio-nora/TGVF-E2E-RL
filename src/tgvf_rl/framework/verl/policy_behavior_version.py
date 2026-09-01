"""Typed rollout-behavior publication for full-Qwen method-matrix runs.

The historical Policy path publishes decoder LoRA tensors and derives its
``PolicyVersion`` directly from that exact snapshot.  Method-matrix runs do
not use decoder LoRA: NoTool/Crop serve a synchronized full Qwen model, while
TGVF/Atomic serve that same Qwen synchronization plus an RP66-owned (RP67
checkpoint) Adapter snapshot.  This module records the identity of the
*accepted behavior state* after the existing upstream synchronization returns.

The full-Qwen receipt is deliberately request scoped.  It is created only
after ``CheckpointEngineManager.update_weights`` completes, so the receipt
means "the upstream transport accepted this run/step/request".  The behavior
version uses a stable run/optimizer-step sync identity so an idempotent
same-step publication retry does not create two policy versions.  Neither
value is a second tensor transport or a claim that upstream exposed a Qwen
content hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Mapping

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion

from .policy_weight_sync import (
    PolicyWeightSyncRequest,
    PolicyWeightSyncState,
    _atomic_replace_bytes,
    _canonical_json_bytes,
    _nonnegative_step,
    _read_bytes,
    _require_run_identity,
    _require_sha256,
    _verify_integrity_field,
    _with_integrity,
)


POLICY_BEHAVIOR_LATEST_FILENAME = "latest-behavior-snapshot.json"
POLICY_BEHAVIOR_LATEST_SCHEMA = "tgvf-policy-behavior-latest-v1"
FULL_QWEN_SYNC_ACK_SCHEMA = "tgvf-full-qwen-sync-ack-v1"
FULL_QWEN_IDENTITY_KIND = "accepted_full_qwen_sync_receipt_v1"


class PolicyBehaviorPayload(str, Enum):
    """Trainable state accepted by rollout workers for one optimizer step."""

    FULL_QWEN = "full_qwen"
    FULL_QWEN_PLUS_RP66 = "full_qwen_plus_rp66"


@dataclass(frozen=True, slots=True)
class FullQwenSyncReceipt:
    """Identity of one request after upstream full-Qwen sync returned."""

    run_id: str
    run_identity_sha256: str
    optimizer_step: int
    request_sha256: str
    sync_identity_sha256: str
    acknowledgement_sha256: str
    schema_version: str = FULL_QWEN_SYNC_ACK_SCHEMA
    identity_kind: str = FULL_QWEN_IDENTITY_KIND

    def __post_init__(self) -> None:
        if self.schema_version != FULL_QWEN_SYNC_ACK_SCHEMA:
            raise ValueError("unsupported full-Qwen sync receipt schema")
        if self.identity_kind != FULL_QWEN_IDENTITY_KIND:
            raise ValueError("unsupported full-Qwen identity kind")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("full-Qwen sync receipt requires run_id")
        _nonnegative_step(self.optimizer_step)
        _require_sha256(self.run_identity_sha256, "run identity")
        _require_sha256(self.request_sha256, "weight-sync request identity")
        _require_sha256(self.sync_identity_sha256, "full-Qwen sync identity")
        _require_sha256(self.acknowledgement_sha256, "full-Qwen sync ACK identity")
        if not hmac.compare_digest(
            self.sync_identity_sha256,
            _full_qwen_sync_identity_sha256(
                run_id=self.run_id,
                run_identity_sha256=self.run_identity_sha256,
                optimizer_step=self.optimizer_step,
            ),
        ):
            raise ValueError("full-Qwen sync identity differs")
        if not hmac.compare_digest(
            self.acknowledgement_sha256,
            _sha256_mapping(self.content_mapping()),
        ):
            raise ValueError("full-Qwen sync ACK identity differs")

    def content_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity_kind": self.identity_kind,
            "run_id": self.run_id,
            "run_identity_sha256": self.run_identity_sha256,
            "optimizer_step": self.optimizer_step,
            "request_sha256": self.request_sha256,
            "upstream_update_weights_returned": True,
        }

    @classmethod
    def from_acknowledged_request(
        cls, request: PolicyWeightSyncRequest, /
    ) -> "FullQwenSyncReceipt":
        """Construct only at the successful upstream-return boundary."""

        if not isinstance(request, PolicyWeightSyncRequest):
            raise TypeError("full-Qwen receipt requires PolicyWeightSyncRequest")
        content = {
            "schema_version": FULL_QWEN_SYNC_ACK_SCHEMA,
            "identity_kind": FULL_QWEN_IDENTITY_KIND,
            "run_id": request.run_id,
            "run_identity_sha256": request.run_identity_sha256,
            "optimizer_step": request.optimizer_step,
            "request_sha256": request.request_sha256,
            "upstream_update_weights_returned": True,
        }
        return cls(
            run_id=request.run_id,
            run_identity_sha256=request.run_identity_sha256,
            optimizer_step=request.optimizer_step,
            request_sha256=request.request_sha256,
            sync_identity_sha256=_full_qwen_sync_identity_sha256(
                run_id=request.run_id,
                run_identity_sha256=request.run_identity_sha256,
                optimizer_step=request.optimizer_step,
            ),
            acknowledgement_sha256=_sha256_mapping(content),
        )


@dataclass(frozen=True, slots=True)
class PolicyBehaviorSnapshot:
    """Strictly validated latest full-Qwen/composite behavior identity."""

    policy_version: PolicyVersion
    run_identity_sha256: str
    payload: PolicyBehaviorPayload
    request_sha256: str
    full_qwen_sync_identity_sha256: str
    full_qwen_sync_ack_sha256: str
    full_qwen_identity_kind: str
    adapter_state_sha256: str | None
    adapter_snapshot_storage_sha256: str | None
    adapter_acknowledgement_sha256: str | None
    adapter_acknowledgement_count: int | None
    adapter_applied_count: int | None
    pointer_file: Path
    pointer_file_sha256: str
    pointer_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, PolicyVersion):
            raise TypeError("behavior snapshot requires PolicyVersion")
        if not isinstance(self.payload, PolicyBehaviorPayload):
            raise TypeError("behavior snapshot payload is invalid")
        if self.full_qwen_identity_kind != FULL_QWEN_IDENTITY_KIND:
            raise ValueError("behavior snapshot full-Qwen identity kind is invalid")
        for owner, value in (
            ("run identity", self.run_identity_sha256),
            ("request identity", self.request_sha256),
            ("full-Qwen sync identity", self.full_qwen_sync_identity_sha256),
            ("full-Qwen sync ACK", self.full_qwen_sync_ack_sha256),
            ("pointer file", self.pointer_file_sha256),
        ):
            _require_sha256(value, owner)
        optional = (
            self.adapter_state_sha256,
            self.adapter_snapshot_storage_sha256,
            self.adapter_acknowledgement_sha256,
            self.adapter_acknowledgement_count,
            self.adapter_applied_count,
        )
        if self.payload is PolicyBehaviorPayload.FULL_QWEN:
            if any(value is not None for value in optional):
                raise ValueError("full-Qwen behavior forbids Adapter identities")
        elif any(value is None for value in optional):
            raise ValueError("composite behavior requires complete Adapter identities")
        for value in optional:
            if isinstance(value, str):
                _require_sha256(value, "Adapter behavior identity")
        if self.adapter_acknowledgement_count is not None:
            _validate_adapter_counts(
                self.adapter_acknowledgement_count,
                self.adapter_applied_count,
            )
        path = Path(self.pointer_file)
        if not path.is_absolute():
            raise ValueError("behavior pointer path must be absolute")
        if not isinstance(self.pointer_bytes, bytes) or not self.pointer_bytes:
            raise ValueError("behavior pointer bytes must be non-empty")


def _adapter_acknowledgement_sha256(
    *,
    optimizer_step: int,
    state_sha256: str,
    snapshot_storage_sha256: str,
    request_sha256: str,
    acknowledgement_count: int,
    applied_count: int,
) -> str:
    """Bind the already-validated RP66/RP67 fan-out acknowledgements."""

    _nonnegative_step(optimizer_step)
    for owner, value in (
        ("Adapter state", state_sha256),
        ("Adapter snapshot storage", snapshot_storage_sha256),
        ("weight-sync request", request_sha256),
    ):
        _require_sha256(value, owner)
    if type(acknowledgement_count) is not int or acknowledgement_count <= 0:
        raise ValueError("Adapter ACK count must be positive")
    if (
        type(applied_count) is not int
        or applied_count < 0
        or applied_count > acknowledgement_count
    ):
        raise ValueError("Adapter applied ACK count is invalid")
    return _sha256_mapping(
        {
            "schema_version": "tgvf-adapter-sync-ack-summary-v1",
            "optimizer_step": optimizer_step,
            "state_sha256": state_sha256,
            "snapshot_storage_sha256": snapshot_storage_sha256,
            "request_sha256": request_sha256,
            "acknowledgement_count": acknowledgement_count,
            "applied_count": applied_count,
        }
    )


def adapter_acknowledgement_sha256(
    *,
    optimizer_step: int,
    state_sha256: str,
    snapshot_storage_sha256: str,
    request_sha256: str,
    acknowledgement_count: int,
    applied_count: int,
) -> str:
    """Public typed helper for one validated Adapter ACK summary."""

    return _adapter_acknowledgement_sha256(
        optimizer_step=optimizer_step,
        state_sha256=state_sha256,
        snapshot_storage_sha256=snapshot_storage_sha256,
        request_sha256=request_sha256,
        acknowledgement_count=acknowledgement_count,
        applied_count=applied_count,
    )


def publish_policy_behavior_snapshot(
    state: PolicyWeightSyncState,
    *,
    full_qwen: FullQwenSyncReceipt,
    payload: PolicyBehaviorPayload,
    adapter_state_sha256: str | None = None,
    adapter_snapshot_storage_sha256: str | None = None,
    adapter_acknowledgement_sha256: str | None = None,
    adapter_acknowledgement_count: int | None = None,
    adapter_applied_count: int | None = None,
) -> PolicyBehaviorSnapshot:
    """Publish only after all payload-specific rollout ACKs have completed."""

    if not isinstance(state, PolicyWeightSyncState):
        raise TypeError("state must be PolicyWeightSyncState")
    if not isinstance(full_qwen, FullQwenSyncReceipt):
        raise TypeError("full_qwen must be FullQwenSyncReceipt")
    if not isinstance(payload, PolicyBehaviorPayload):
        raise TypeError("payload must be PolicyBehaviorPayload")
    _require_run_identity(state, full_qwen.run_id, full_qwen.run_identity_sha256)
    optional = (
        adapter_state_sha256,
        adapter_snapshot_storage_sha256,
        adapter_acknowledgement_sha256,
        adapter_acknowledgement_count,
        adapter_applied_count,
    )
    if payload is PolicyBehaviorPayload.FULL_QWEN:
        if any(value is not None for value in optional):
            raise ValueError("full-Qwen behavior forbids Adapter identities")
    elif any(value is None for value in optional):
        raise ValueError("composite behavior requires complete Adapter identities")
    for value in optional:
        if isinstance(value, str):
            _require_sha256(value, "Adapter behavior identity")
    if payload is PolicyBehaviorPayload.FULL_QWEN_PLUS_RP66:
        assert adapter_state_sha256 is not None
        assert adapter_snapshot_storage_sha256 is not None
        assert adapter_acknowledgement_sha256 is not None
        assert adapter_acknowledgement_count is not None
        assert adapter_applied_count is not None
        _validate_adapter_counts(
            adapter_acknowledgement_count,
            adapter_applied_count,
        )
        expected_adapter_ack = _adapter_acknowledgement_sha256(
            optimizer_step=full_qwen.optimizer_step,
            state_sha256=adapter_state_sha256,
            snapshot_storage_sha256=adapter_snapshot_storage_sha256,
            request_sha256=full_qwen.request_sha256,
            acknowledgement_count=adapter_acknowledgement_count,
            applied_count=adapter_applied_count,
        )
        if not hmac.compare_digest(
            adapter_acknowledgement_sha256, expected_adapter_ack
        ):
            raise ValueError("Adapter acknowledgement identity differs")
        _verify_adapter_snapshot_closure(
            state,
            optimizer_step=full_qwen.optimizer_step,
            request_sha256=full_qwen.request_sha256,
            adapter_state_sha256=adapter_state_sha256,
            adapter_snapshot_storage_sha256=adapter_snapshot_storage_sha256,
        )
    content = {
        "schema_version": POLICY_BEHAVIOR_LATEST_SCHEMA,
        "run_id": state.run_id,
        "run_identity_sha256": state.run_identity_sha256,
        "optimizer_step": full_qwen.optimizer_step,
        "payload": payload.value,
        "request_sha256": full_qwen.request_sha256,
        "full_qwen_identity_kind": full_qwen.identity_kind,
        "full_qwen_sync_identity_sha256": full_qwen.sync_identity_sha256,
        "full_qwen_sync_ack_sha256": full_qwen.acknowledgement_sha256,
        "adapter_state_sha256": adapter_state_sha256,
        "adapter_snapshot_storage_sha256": adapter_snapshot_storage_sha256,
        "adapter_acknowledgement_sha256": adapter_acknowledgement_sha256,
        "adapter_acknowledgement_count": adapter_acknowledgement_count,
        "adapter_applied_count": adapter_applied_count,
    }
    content["weights_sha256"] = _behavior_weights_sha256(content)
    pointer_bytes = _canonical_json_bytes(_with_integrity(content)) + b"\n"
    _atomic_replace_bytes(
        state.directory / POLICY_BEHAVIOR_LATEST_FILENAME, pointer_bytes
    )
    return load_latest_policy_behavior_snapshot(
        state, expected_optimizer_step=full_qwen.optimizer_step
    )


def load_latest_policy_behavior_snapshot(
    state: PolicyWeightSyncState,
    *,
    expected_optimizer_step: int | None = None,
) -> PolicyBehaviorSnapshot:
    """Load the latest full-Qwen behavior pointer and verify its identity."""

    if not isinstance(state, PolicyWeightSyncState):
        raise TypeError("state must be PolicyWeightSyncState")
    if expected_optimizer_step is not None:
        _nonnegative_step(expected_optimizer_step)
    pointer = Path(
        os.path.abspath(os.fspath(state.directory / POLICY_BEHAVIOR_LATEST_FILENAME))
    )
    pointer_bytes = _read_bytes(pointer, "latest Policy behavior pointer")
    pointer_sha256 = hashlib.sha256(pointer_bytes).hexdigest()
    try:
        decoded = json.loads(pointer_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReplayMismatchError("Policy behavior pointer is unreadable") from error
    expected_fields = {
        "schema_version",
        "run_id",
        "run_identity_sha256",
        "optimizer_step",
        "payload",
        "request_sha256",
        "full_qwen_identity_kind",
        "full_qwen_sync_identity_sha256",
        "full_qwen_sync_ack_sha256",
        "adapter_state_sha256",
        "adapter_snapshot_storage_sha256",
        "adapter_acknowledgement_sha256",
        "adapter_acknowledgement_count",
        "adapter_applied_count",
        "weights_sha256",
        "integrity_sha256",
    }
    if not isinstance(decoded, Mapping) or set(decoded) != expected_fields:
        raise ReplayMismatchError("Policy behavior pointer fields differ")
    _verify_integrity_field(decoded, "integrity_sha256", "Policy behavior pointer")
    if decoded["schema_version"] != POLICY_BEHAVIOR_LATEST_SCHEMA:
        raise ReplayMismatchError("Policy behavior pointer schema differs")
    run_id = _required_text(decoded, "run_id")
    run_identity = _required_digest(decoded, "run_identity_sha256")
    _require_run_identity(state, run_id, run_identity)
    step = decoded["optimizer_step"]
    try:
        _nonnegative_step(step)
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError(
            "Policy behavior optimizer step is invalid"
        ) from error
    if expected_optimizer_step is not None and step != expected_optimizer_step:
        raise IdentityMismatchError("latest Policy behavior optimizer step differs")
    try:
        payload = PolicyBehaviorPayload(decoded["payload"])
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError("Policy behavior payload is invalid") from error
    request_sha256 = _required_digest(decoded, "request_sha256")
    identity_kind = _required_text(decoded, "full_qwen_identity_kind")
    if identity_kind != FULL_QWEN_IDENTITY_KIND:
        raise ReplayMismatchError("Policy behavior full-Qwen identity kind differs")
    full_qwen_identity = _required_digest(decoded, "full_qwen_sync_identity_sha256")
    expected_full_qwen_identity = _full_qwen_sync_identity_sha256(
        run_id=run_id,
        run_identity_sha256=run_identity,
        optimizer_step=step,
    )
    if not hmac.compare_digest(full_qwen_identity, expected_full_qwen_identity):
        raise ReplayMismatchError("Policy behavior full-Qwen sync identity differs")
    full_qwen_ack = _required_digest(decoded, "full_qwen_sync_ack_sha256")
    receipt_content = {
        "schema_version": FULL_QWEN_SYNC_ACK_SCHEMA,
        "identity_kind": FULL_QWEN_IDENTITY_KIND,
        "run_id": run_id,
        "run_identity_sha256": run_identity,
        "optimizer_step": step,
        "request_sha256": request_sha256,
        "upstream_update_weights_returned": True,
    }
    if not hmac.compare_digest(full_qwen_ack, _sha256_mapping(receipt_content)):
        raise ReplayMismatchError("Policy behavior full-Qwen ACK identity differs")
    adapter_state = _optional_digest(decoded, "adapter_state_sha256")
    adapter_storage = _optional_digest(decoded, "adapter_snapshot_storage_sha256")
    adapter_ack = _optional_digest(decoded, "adapter_acknowledgement_sha256")
    adapter_ack_count = _optional_count(decoded, "adapter_acknowledgement_count")
    adapter_applied_count = _optional_count(decoded, "adapter_applied_count")
    optional = (
        adapter_state,
        adapter_storage,
        adapter_ack,
        adapter_ack_count,
        adapter_applied_count,
    )
    if payload is PolicyBehaviorPayload.FULL_QWEN:
        if any(value is not None for value in optional):
            raise ReplayMismatchError("full-Qwen behavior carries Adapter identity")
    elif any(value is None for value in optional):
        raise ReplayMismatchError("composite behavior omitted Adapter identity")
    if payload is PolicyBehaviorPayload.FULL_QWEN_PLUS_RP66:
        assert adapter_state is not None
        assert adapter_storage is not None
        assert adapter_ack is not None
        assert adapter_ack_count is not None
        assert adapter_applied_count is not None
        try:
            _validate_adapter_counts(adapter_ack_count, adapter_applied_count)
            expected_adapter_ack = _adapter_acknowledgement_sha256(
                optimizer_step=step,
                state_sha256=adapter_state,
                snapshot_storage_sha256=adapter_storage,
                request_sha256=request_sha256,
                acknowledgement_count=adapter_ack_count,
                applied_count=adapter_applied_count,
            )
        except ValueError as error:
            raise ReplayMismatchError(
                "Policy behavior Adapter acknowledgement is invalid"
            ) from error
        if not hmac.compare_digest(adapter_ack, expected_adapter_ack):
            raise ReplayMismatchError(
                "Policy behavior Adapter acknowledgement identity differs"
            )
        _verify_adapter_snapshot_closure(
            state,
            optimizer_step=step,
            request_sha256=request_sha256,
            adapter_state_sha256=adapter_state,
            adapter_snapshot_storage_sha256=adapter_storage,
        )
    content = {name: decoded[name] for name in expected_fields - {"integrity_sha256"}}
    weights = _required_digest(decoded, "weights_sha256")
    if not hmac.compare_digest(weights, _behavior_weights_sha256(content)):
        raise ReplayMismatchError("Policy behavior composite identity differs")
    return PolicyBehaviorSnapshot(
        policy_version=PolicyVersion(run_id, step, weights),
        run_identity_sha256=run_identity,
        payload=payload,
        request_sha256=request_sha256,
        full_qwen_sync_identity_sha256=full_qwen_identity,
        full_qwen_sync_ack_sha256=full_qwen_ack,
        full_qwen_identity_kind=identity_kind,
        adapter_state_sha256=adapter_state,
        adapter_snapshot_storage_sha256=adapter_storage,
        adapter_acknowledgement_sha256=adapter_ack,
        adapter_acknowledgement_count=adapter_ack_count,
        adapter_applied_count=adapter_applied_count,
        pointer_file=pointer,
        pointer_file_sha256=pointer_sha256,
        pointer_bytes=pointer_bytes,
    )


def load_latest_policy_behavior_version(
    state: PolicyWeightSyncState,
    *,
    expected_optimizer_step: int | None = None,
) -> PolicyVersion:
    return load_latest_policy_behavior_snapshot(
        state, expected_optimizer_step=expected_optimizer_step
    ).policy_version


def _behavior_weights_sha256(content: Mapping[str, object]) -> str:
    identity = {
        name: content[name]
        for name in (
            "schema_version",
            "run_id",
            "run_identity_sha256",
            "optimizer_step",
            "payload",
            "full_qwen_identity_kind",
            "full_qwen_sync_identity_sha256",
            "adapter_state_sha256",
            "adapter_snapshot_storage_sha256",
        )
    }
    return _sha256_mapping(
        {
            "identity_schema_version": "tgvf-policy-behavior-weights-v1",
            **identity,
        }
    )


def _sha256_mapping(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _full_qwen_sync_identity_sha256(
    *, run_id: str, run_identity_sha256: str, optimizer_step: int
) -> str:
    """Stable optimizer-state identity; not a Qwen tensor content hash."""

    return _sha256_mapping(
        {
            "schema_version": "tgvf-accepted-full-qwen-sync-identity-v1",
            "identity_kind": FULL_QWEN_IDENTITY_KIND,
            "run_id": run_id,
            "run_identity_sha256": run_identity_sha256,
            "optimizer_step": optimizer_step,
            "upstream_update_weights_returned": True,
        }
    )


def _verify_adapter_snapshot_closure(
    state: PolicyWeightSyncState,
    *,
    optimizer_step: int,
    request_sha256: str,
    adapter_state_sha256: str,
    adapter_snapshot_storage_sha256: str,
) -> None:
    """Re-open the exact RP66/RP67 tensor closure named by the behavior."""

    from .trainable_tgvf_weight_sync import load_latest_trainable_rp66_snapshot
    from .vllm_tool_runtime import adapter_owned_state_sha256

    snapshot = load_latest_trainable_rp66_snapshot(
        state,
        expected_optimizer_step=optimizer_step,
        expected_request_sha256=request_sha256,
    )
    if not hmac.compare_digest(
        snapshot.storage_sha256, adapter_snapshot_storage_sha256
    ):
        raise ReplayMismatchError("Policy behavior Adapter storage identity differs")
    actual_state = adapter_owned_state_sha256(snapshot.tensors)
    if not hmac.compare_digest(actual_state, adapter_state_sha256):
        raise ReplayMismatchError("Policy behavior Adapter state identity differs")


def _validate_adapter_counts(
    acknowledgement_count: int,
    applied_count: int | None,
) -> None:
    if type(acknowledgement_count) is not int or acknowledgement_count <= 0:
        raise ValueError("Adapter ACK count must be positive")
    if (
        type(applied_count) is not int
        or applied_count < 0
        or applied_count > acknowledgement_count
    ):
        raise ValueError("Adapter applied ACK count is invalid")


def _required_text(value: Mapping[str, object], name: str) -> str:
    selected = value[name]
    if not isinstance(selected, str) or not selected:
        raise ReplayMismatchError(f"Policy behavior {name} is invalid")
    return selected


def _required_digest(value: Mapping[str, object], name: str) -> str:
    selected = _required_text(value, name)
    try:
        _require_sha256(selected, name)
    except ValueError as error:
        raise ReplayMismatchError(f"Policy behavior {name} is invalid") from error
    return selected


def _optional_digest(value: Mapping[str, object], name: str) -> str | None:
    selected = value[name]
    if selected is None:
        return None
    if not isinstance(selected, str):
        raise ReplayMismatchError(f"Policy behavior {name} is invalid")
    try:
        _require_sha256(selected, name)
    except ValueError as error:
        raise ReplayMismatchError(f"Policy behavior {name} is invalid") from error
    return selected


def _optional_count(value: Mapping[str, object], name: str) -> int | None:
    selected = value[name]
    if selected is None:
        return None
    if type(selected) is not int:
        raise ReplayMismatchError(f"Policy behavior {name} is invalid")
    return selected


__all__ = [
    "FULL_QWEN_SYNC_ACK_SCHEMA",
    "FULL_QWEN_IDENTITY_KIND",
    "POLICY_BEHAVIOR_LATEST_FILENAME",
    "POLICY_BEHAVIOR_LATEST_SCHEMA",
    "FullQwenSyncReceipt",
    "PolicyBehaviorPayload",
    "PolicyBehaviorSnapshot",
    "adapter_acknowledgement_sha256",
    "load_latest_policy_behavior_snapshot",
    "load_latest_policy_behavior_version",
    "publish_policy_behavior_snapshot",
]
