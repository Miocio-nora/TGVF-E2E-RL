"""Project-owned checkpoint state for the Policy Pilot v1 runtime.

This payload is an adjunct to, never a replacement for, the veRL/FSDP2
checkpoint.  In particular, policy LoRA parameters and optimizer, scheduler,
and scaler tensors are deliberately absent.  Their loaded identities are
checked against this state before any project-owned state is restored.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.policy.metrics import (
    PilotMetricsAccumulator,
    PilotMetricsCheckpointState,
)


POLICY_PILOT_V1_PROJECT_CHECKPOINT_SCHEMA = (
    "policy-pilot-v1-project-extra-checkpoint-v1"
)

DATA_CURSOR_OWNER = "data_cursor"
ROLLOUT_SAMPLER_OWNER = "rollout_sampler"
ROLLOUT_RNG_OWNER = "rollout_rng"


@dataclass(frozen=True, order=True, slots=True)
class CheckpointIdentityHash:
    """One named, immutable run-identity digest."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        _nonempty_string(self.name, "identity hash name")
        _sha256(self.sha256, "identity hash")


@dataclass(frozen=True, slots=True)
class PilotRunIdentityHashes:
    """The exact set of hashes that identifies one Policy Pilot run."""

    run_id: str
    hashes: tuple[CheckpointIdentityHash, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.run_id, "run_id")
        if not isinstance(self.hashes, Sequence) or isinstance(
            self.hashes, (str, bytes)
        ):
            raise TypeError("run identity hashes must be a sequence")
        object.__setattr__(self, "hashes", tuple(self.hashes))
        if not self.hashes:
            raise ValueError("at least one run identity hash is required")
        if any(not isinstance(item, CheckpointIdentityHash) for item in self.hashes):
            raise TypeError("hashes must contain CheckpointIdentityHash values")
        names = tuple(item.name for item in self.hashes)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("run identity hashes must be unique and sorted by name")

    @classmethod
    def from_hashes(
        cls, run_id: str, hashes: Mapping[str, str]
    ) -> "PilotRunIdentityHashes":
        if not isinstance(hashes, Mapping):
            raise TypeError("hashes must be a mapping")
        if any(type(name) is not str for name in hashes):
            raise TypeError("run identity hash names must be strings")
        return cls(
            run_id=run_id,
            hashes=tuple(
                CheckpointIdentityHash(name, hashes[name]) for name in sorted(hashes)
            ),
        )

    def to_checkpoint_mapping(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "hashes": [[item.name, item.sha256] for item in self.hashes],
        }

    @classmethod
    def from_checkpoint_mapping(
        cls, payload: object
    ) -> "PilotRunIdentityHashes":
        mapping = _strict_mapping(payload, {"run_id", "hashes"}, "run identity")
        rows = mapping["hashes"]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ReplayMismatchError("run identity hashes must be a sequence")
        hashes: list[CheckpointIdentityHash] = []
        for row in rows:
            if (
                not isinstance(row, Sequence)
                or isinstance(row, (str, bytes))
                or len(row) != 2
            ):
                raise ReplayMismatchError(
                    "each run identity hash must be [name, sha256]"
                )
            try:
                hashes.append(CheckpointIdentityHash(row[0], row[1]))
            except (TypeError, ValueError) as error:
                raise ReplayMismatchError("run identity hash is malformed") from error
        try:
            return cls(run_id=mapping["run_id"], hashes=tuple(hashes))
        except (TypeError, ValueError) as error:
            raise ReplayMismatchError("run identity is malformed") from error


@dataclass(frozen=True, slots=True)
class OpaqueProjectState:
    """Owner-serialized bytes whose content the checkpoint layer never interprets."""

    owner: str
    codec: str
    payload: bytes

    def __post_init__(self) -> None:
        _nonempty_string(self.owner, "opaque state owner")
        _nonempty_string(self.codec, "opaque state codec")
        if type(self.payload) is not bytes:
            raise TypeError("opaque state payload must be immutable bytes")
        if not self.payload:
            raise ValueError("opaque state payload must not be empty")

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def to_checkpoint_mapping(self) -> dict[str, object]:
        return {
            "owner": self.owner,
            "codec": self.codec,
            "payload_base64": base64.b64encode(self.payload).decode("ascii"),
            "payload_sha256": self.payload_sha256,
        }

    @classmethod
    def from_checkpoint_mapping(cls, payload: object) -> "OpaqueProjectState":
        mapping = _strict_mapping(
            payload,
            {"owner", "codec", "payload_base64", "payload_sha256"},
            "opaque project state",
        )
        encoded = mapping["payload_base64"]
        if type(encoded) is not str:
            raise ReplayMismatchError("opaque payload_base64 must be a string")
        try:
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error) as error:
            raise ReplayMismatchError("opaque payload_base64 is malformed") from error
        try:
            state = cls(
                owner=mapping["owner"], codec=mapping["codec"], payload=raw
            )
            _sha256(mapping["payload_sha256"], "opaque payload digest")
        except (TypeError, ValueError) as error:
            raise ReplayMismatchError("opaque project state is malformed") from error
        if not hmac.compare_digest(state.payload_sha256, mapping["payload_sha256"]):
            raise ReplayMismatchError("opaque project state digest mismatch")
        return state


@dataclass(frozen=True, slots=True)
class PilotOptimizerDataCursor:
    """Completed optimizer step plus the owner's exact next-data cursor."""

    optimizer_step: int
    data_cursor: OpaqueProjectState

    def __post_init__(self) -> None:
        _nonnegative_int(self.optimizer_step, "optimizer_step")
        if not isinstance(self.data_cursor, OpaqueProjectState):
            raise TypeError("data_cursor must be OpaqueProjectState")
        if self.data_cursor.owner != DATA_CURSOR_OWNER:
            raise ValueError(f"data cursor owner must be {DATA_CURSOR_OWNER!r}")

    def to_checkpoint_mapping(self) -> dict[str, object]:
        return {
            "optimizer_step": self.optimizer_step,
            "data_cursor": self.data_cursor.to_checkpoint_mapping(),
        }

    @classmethod
    def from_checkpoint_mapping(
        cls, payload: object
    ) -> "PilotOptimizerDataCursor":
        mapping = _strict_mapping(
            payload, {"optimizer_step", "data_cursor"}, "optimizer/data cursor"
        )
        try:
            return cls(
                optimizer_step=mapping["optimizer_step"],
                data_cursor=OpaqueProjectState.from_checkpoint_mapping(
                    mapping["data_cursor"]
                ),
            )
        except ReplayMismatchError:
            raise
        except (TypeError, ValueError) as error:
            raise ReplayMismatchError("optimizer/data cursor is malformed") from error


@dataclass(frozen=True, slots=True)
class PilotRolloutBarrier:
    """Checkpoint boundary: no stale or sampled-but-unconsumed rollout exists."""

    asynchronous_staleness_steps: int = 0
    outstanding_rollout_count: int = 0

    def __post_init__(self) -> None:
        _nonnegative_int(
            self.asynchronous_staleness_steps, "asynchronous_staleness_steps"
        )
        _nonnegative_int(self.outstanding_rollout_count, "outstanding_rollout_count")
        if self.asynchronous_staleness_steps != 0:
            raise ValueError("Policy Pilot checkpoint requires zero rollout staleness")
        if self.outstanding_rollout_count != 0:
            raise ValueError(
                "Policy Pilot checkpoint requires no outstanding rollouts"
            )

    def to_checkpoint_mapping(self) -> dict[str, object]:
        return {
            "asynchronous_staleness_steps": self.asynchronous_staleness_steps,
            "outstanding_rollout_count": self.outstanding_rollout_count,
        }

    @classmethod
    def from_checkpoint_mapping(cls, payload: object) -> "PilotRolloutBarrier":
        mapping = _strict_mapping(
            payload,
            {"asynchronous_staleness_steps", "outstanding_rollout_count"},
            "rollout barrier",
        )
        try:
            return cls(
                asynchronous_staleness_steps=mapping[
                    "asynchronous_staleness_steps"
                ],
                outstanding_rollout_count=mapping["outstanding_rollout_count"],
            )
        except (TypeError, ValueError) as error:
            raise ReplayMismatchError("rollout barrier is not checkpoint-safe") from error


@dataclass(frozen=True, slots=True)
class PilotProjectCheckpointState:
    """Immutable project-extra state paired with one upstream checkpoint."""

    run_identity: PilotRunIdentityHashes
    progress: PilotOptimizerDataCursor
    rollout_sampler_state: OpaqueProjectState
    rollout_rng_state: OpaqueProjectState
    metrics_state: PilotMetricsCheckpointState
    policy_version: PolicyVersion
    reference_version: PolicyVersion
    rollout_barrier: PilotRolloutBarrier
    schema_version: str = POLICY_PILOT_V1_PROJECT_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_PILOT_V1_PROJECT_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported Policy Pilot project checkpoint schema")
        expected_types = (
            (self.run_identity, PilotRunIdentityHashes, "run_identity"),
            (self.progress, PilotOptimizerDataCursor, "progress"),
            (
                self.rollout_sampler_state,
                OpaqueProjectState,
                "rollout_sampler_state",
            ),
            (self.rollout_rng_state, OpaqueProjectState, "rollout_rng_state"),
            (self.metrics_state, PilotMetricsCheckpointState, "metrics_state"),
            (self.policy_version, PolicyVersion, "policy_version"),
            (self.reference_version, PolicyVersion, "reference_version"),
            (self.rollout_barrier, PilotRolloutBarrier, "rollout_barrier"),
        )
        for value, expected_type, name in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")
        for version, name in (
            (self.policy_version, "policy_version"),
            (self.reference_version, "reference_version"),
        ):
            _nonempty_string(version.run_id, f"{name} run_id")
            _nonnegative_int(version.optimizer_step, f"{name} optimizer_step")
            _sha256(version.weights_sha256, f"{name} weights digest")
        if self.rollout_sampler_state.owner != ROLLOUT_SAMPLER_OWNER:
            raise ValueError(
                f"rollout sampler owner must be {ROLLOUT_SAMPLER_OWNER!r}"
            )
        if self.rollout_rng_state.owner != ROLLOUT_RNG_OWNER:
            raise ValueError(f"rollout RNG owner must be {ROLLOUT_RNG_OWNER!r}")
        step = self.progress.optimizer_step
        if self.metrics_state.optimizer_steps != step:
            raise ValueError("metrics optimizer step differs from the data cursor")
        if self.policy_version.optimizer_step != step:
            raise ValueError("policy version differs from the data cursor step")
        if self.policy_version.run_id != self.run_identity.run_id:
            raise ValueError("policy version run_id differs from checkpoint run_id")
        if self.reference_version.optimizer_step != 0:
            raise ValueError("the frozen Pilot reference must remain at optimizer step 0")

    @property
    def integrity_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self._content_mapping())).hexdigest()

    def _content_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_identity": self.run_identity.to_checkpoint_mapping(),
            "progress": self.progress.to_checkpoint_mapping(),
            "rollout_sampler_state": (
                self.rollout_sampler_state.to_checkpoint_mapping()
            ),
            "rollout_rng_state": self.rollout_rng_state.to_checkpoint_mapping(),
            "metrics_state": self.metrics_state.to_checkpoint_mapping(),
            "policy_version": _policy_version_mapping(self.policy_version),
            "reference_version": _policy_version_mapping(self.reference_version),
            "rollout_barrier": self.rollout_barrier.to_checkpoint_mapping(),
        }

    def to_checkpoint_mapping(self) -> dict[str, object]:
        content = self._content_mapping()
        content["integrity_sha256"] = self.integrity_sha256
        return content

    @classmethod
    def from_checkpoint_mapping(
        cls, payload: object
    ) -> "PilotProjectCheckpointState":
        expected = {
            "schema_version",
            "run_identity",
            "progress",
            "rollout_sampler_state",
            "rollout_rng_state",
            "metrics_state",
            "policy_version",
            "reference_version",
            "rollout_barrier",
            "integrity_sha256",
        }
        mapping = _strict_mapping(payload, expected, "project checkpoint")
        if mapping["schema_version"] != POLICY_PILOT_V1_PROJECT_CHECKPOINT_SCHEMA:
            raise ReplayMismatchError("project checkpoint schema mismatch")
        try:
            _sha256(mapping["integrity_sha256"], "checkpoint integrity digest")
            content = {key: mapping[key] for key in mapping if key != "integrity_sha256"}
            actual_digest = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
        except (TypeError, ValueError) as error:
            raise ReplayMismatchError("project checkpoint digest is malformed") from error
        if not hmac.compare_digest(actual_digest, mapping["integrity_sha256"]):
            raise ReplayMismatchError("project checkpoint integrity mismatch")

        try:
            metrics_payload = mapping["metrics_state"]
            if not isinstance(metrics_payload, Mapping):
                raise ReplayMismatchError("metrics checkpoint state must be a mapping")
            return cls(
                schema_version=mapping["schema_version"],
                run_identity=PilotRunIdentityHashes.from_checkpoint_mapping(
                    mapping["run_identity"]
                ),
                progress=PilotOptimizerDataCursor.from_checkpoint_mapping(
                    mapping["progress"]
                ),
                rollout_sampler_state=OpaqueProjectState.from_checkpoint_mapping(
                    mapping["rollout_sampler_state"]
                ),
                rollout_rng_state=OpaqueProjectState.from_checkpoint_mapping(
                    mapping["rollout_rng_state"]
                ),
                metrics_state=PilotMetricsCheckpointState.from_checkpoint_mapping(
                    metrics_payload
                ),
                policy_version=_policy_version_from_mapping(
                    mapping["policy_version"], "policy version"
                ),
                reference_version=_policy_version_from_mapping(
                    mapping["reference_version"], "reference version"
                ),
                rollout_barrier=PilotRolloutBarrier.from_checkpoint_mapping(
                    mapping["rollout_barrier"]
                ),
            )
        except ReplayMismatchError:
            raise
        except (TypeError, ValueError) as error:
            raise ReplayMismatchError("project checkpoint content is corrupt") from error

    def validate_restore_identity(
        self,
        *,
        expected_run_identity: PilotRunIdentityHashes,
        loaded_policy_version: PolicyVersion,
        loaded_reference_version: PolicyVersion,
    ) -> None:
        """Match this adjunct to the run and framework-owned loaded weights."""

        if not isinstance(expected_run_identity, PilotRunIdentityHashes):
            raise TypeError("expected_run_identity must be PilotRunIdentityHashes")
        if not isinstance(loaded_policy_version, PolicyVersion):
            raise TypeError("loaded_policy_version must be PolicyVersion")
        if not isinstance(loaded_reference_version, PolicyVersion):
            raise TypeError("loaded_reference_version must be PolicyVersion")
        mismatches: dict[str, object] = {}
        if self.run_identity != expected_run_identity:
            mismatches["run_identity"] = (
                self.run_identity,
                expected_run_identity,
            )
        if self.policy_version != loaded_policy_version:
            mismatches["policy_version"] = (
                self.policy_version,
                loaded_policy_version,
            )
        if self.reference_version != loaded_reference_version:
            mismatches["reference_version"] = (
                self.reference_version,
                loaded_reference_version,
            )
        if mismatches:
            raise IdentityMismatchError(
                "Policy Pilot project checkpoint identity mismatch: "
                f"{mismatches!r}"
            )


def capture_pilot_project_checkpoint(
    *,
    run_identity: PilotRunIdentityHashes,
    progress: PilotOptimizerDataCursor,
    rollout_sampler_state: OpaqueProjectState,
    rollout_rng_state: OpaqueProjectState,
    metrics_accumulator: PilotMetricsAccumulator,
    policy_version: PolicyVersion,
    reference_version: PolicyVersion,
    rollout_barrier: PilotRolloutBarrier,
) -> PilotProjectCheckpointState:
    """Capture only project-owned state at a proven quiescent boundary."""

    if not isinstance(metrics_accumulator, PilotMetricsAccumulator):
        raise TypeError("metrics_accumulator must be PilotMetricsAccumulator")
    return PilotProjectCheckpointState(
        run_identity=run_identity,
        progress=progress,
        rollout_sampler_state=rollout_sampler_state,
        rollout_rng_state=rollout_rng_state,
        metrics_state=metrics_accumulator.state,
        policy_version=policy_version,
        reference_version=reference_version,
        rollout_barrier=rollout_barrier,
    )


def restore_pilot_project_checkpoint(
    payload: object,
    *,
    expected_run_identity: PilotRunIdentityHashes,
    loaded_policy_version: PolicyVersion,
    loaded_reference_version: PolicyVersion,
    metrics_accumulator: PilotMetricsAccumulator,
) -> PilotProjectCheckpointState:
    """Validate completely, then atomically restore project-owned metrics.

    The returned immutable state gives the owners their exact data-cursor,
    sampler, and RNG bytes.  This function does not load or mutate framework-
    owned model, optimizer, scheduler, or scaler state.
    """

    if not isinstance(metrics_accumulator, PilotMetricsAccumulator):
        raise TypeError("metrics_accumulator must be PilotMetricsAccumulator")
    state = validate_pilot_project_checkpoint_restore(
        payload,
        expected_run_identity=expected_run_identity,
        loaded_policy_version=loaded_policy_version,
        loaded_reference_version=loaded_reference_version,
    )
    metrics_accumulator.restore_checkpoint_state(state.metrics_state)
    return state


def validate_pilot_project_checkpoint_restore(
    payload: object,
    *,
    expected_run_identity: PilotRunIdentityHashes,
    loaded_policy_version: PolicyVersion,
    loaded_reference_version: PolicyVersion,
) -> PilotProjectCheckpointState:
    """Validate a project adjunct without mutating any runtime owner.

    A clean-process veRL resume must load the framework-owned LoRA, optimizer,
    and scheduler before it can prove their policy/reference identities.  This
    split validation entry point lets that orchestration validate the complete
    adjunct and the loaded framework state before restoring the data cursor,
    rollout sampler/RNG, or metrics.
    """

    state = _coerce_project_checkpoint(payload)
    state.validate_restore_identity(
        expected_run_identity=expected_run_identity,
        loaded_policy_version=loaded_policy_version,
        loaded_reference_version=loaded_reference_version,
    )
    return state


def _coerce_project_checkpoint(payload: object) -> PilotProjectCheckpointState:
    if isinstance(payload, PilotProjectCheckpointState):
        return payload
    if isinstance(payload, Mapping):
        return PilotProjectCheckpointState.from_checkpoint_mapping(payload)
    raise TypeError("project checkpoint must be a state object or mapping")


def _policy_version_mapping(version: PolicyVersion) -> dict[str, object]:
    return {
        "run_id": version.run_id,
        "optimizer_step": version.optimizer_step,
        "weights_sha256": version.weights_sha256,
    }


def _policy_version_from_mapping(payload: object, name: str) -> PolicyVersion:
    mapping = _strict_mapping(
        payload, {"run_id", "optimizer_step", "weights_sha256"}, name
    )
    try:
        _nonnegative_int(mapping["optimizer_step"], f"{name} optimizer_step")
        _nonempty_string(mapping["run_id"], f"{name} run_id")
        _sha256(mapping["weights_sha256"], f"{name} weights digest")
        return PolicyVersion(
            run_id=mapping["run_id"],
            optimizer_step=mapping["optimizer_step"],
            weights_sha256=mapping["weights_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError(f"{name} is malformed") from error


def _strict_mapping(
    payload: object, expected: set[str], name: str
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ReplayMismatchError(f"{name} must be a mapping")
    keys = tuple(payload)
    if any(type(key) is not str for key in keys):
        raise ReplayMismatchError(f"{name} field names must be strings")
    actual = set(keys)
    if actual != expected:
        missing = tuple(sorted(expected - actual))
        extra = tuple(sorted(actual - expected))
        raise ReplayMismatchError(
            f"{name} fields differ: missing={missing!r} extra={extra!r}"
        )
    return payload


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _nonempty_string(value: object, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
