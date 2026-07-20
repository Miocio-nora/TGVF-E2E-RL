"""Bit-preserving in-memory observation store with strict checksum replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from threading import RLock
from typing import Iterable, Mapping

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tensors import (
    ContentAddress,
    TensorArtifactRef,
    TensorDescriptor,
)

from .schema import (
    CropObservationRecord,
    FocusedObservationRecord,
    ObservationRecord,
    SourceVisualState,
    TrajectorySourceVisual,
)


@dataclass(frozen=True, slots=True)
class ObservationHandle:
    observation_id: str
    record_sha256: str


@dataclass(frozen=True, slots=True)
class TrajectoryReplayHandle:
    replay_id: str
    record_sha256: str


@dataclass(frozen=True, slots=True)
class ObservationReleaseCounts:
    records: int
    replays: int
    tensors: int


@dataclass(frozen=True, slots=True)
class ReplayTensorPayload:
    """One bit-preserving CPU tensor carried across a worker boundary."""

    sha256: str
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("replay tensor payload must be a torch.Tensor")
        stored = self.tensor.detach().to(device="cpu").contiguous().clone()
        object.__setattr__(self, "tensor", stored)
        if tensor_checksum(stored) != self.sha256:
            raise ReplayMismatchError("replay tensor payload checksum mismatch")


@dataclass(frozen=True, slots=True)
class TrajectoryReplayBundle:
    """Self-contained exact replay payload transferable to policy/reference."""

    schema_version: str
    replay_handle: TrajectoryReplayHandle
    replay_record: TrajectoryReplayRecord
    observation_records: tuple[ObservationRecord, ...]
    tensor_payloads: tuple[ReplayTensorPayload, ...]
    bundle_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "trajectory-replay-bundle-v1":
            raise ValueError("unknown trajectory replay bundle schema")
        object.__setattr__(self, "observation_records", tuple(self.observation_records))
        object.__setattr__(self, "tensor_payloads", tuple(self.tensor_payloads))
        if replay_checksum(self.replay_record) != self.replay_handle.record_sha256:
            raise ReplayMismatchError("bundle replay handle/record checksum mismatch")
        if self.replay_record.replay_id != self.replay_handle.replay_id:
            raise ReplayMismatchError("bundle replay handle/record ID mismatch")
        record_digests = tuple(
            record_checksum(record) for record in self.observation_records
        )
        expected_handles = tuple(
            ObservationHandle(record.observation_id, digest)
            for record, digest in zip(
                self.observation_records, record_digests, strict=True
            )
        )
        if expected_handles != self.replay_record.observation_handles:
            raise ReplayMismatchError(
                "bundle observation records differ from replay handles"
            )
        payload_digests = tuple(payload.sha256 for payload in self.tensor_payloads)
        if payload_digests != tuple(sorted(set(payload_digests))):
            raise ReplayMismatchError(
                "bundle tensor payloads must be unique and digest-sorted"
            )
        expected_tensor_digests = tuple(
            sorted(
                {
                    ref.address.digest
                    for value in (
                        self.replay_record.tensors,
                        self.replay_record.source_visual,
                        *self.observation_records,
                    )
                    for ref in _walk_tensor_refs(value)
                }
            )
        )
        if payload_digests != expected_tensor_digests:
            raise ReplayMismatchError(
                "bundle tensors differ from replay/observation references"
            )
        if (
            _replay_bundle_checksum(
                self.replay_handle,
                self.observation_records,
                self.tensor_payloads,
            )
            != self.bundle_sha256
        ):
            raise ReplayMismatchError("trajectory replay bundle checksum mismatch")


@dataclass(frozen=True, slots=True)
class TrajectoryReplayTensorRefs:
    """Exact final-sequence tensors materialized by rollout, never by replay."""

    input_ids: TensorArtifactRef
    position_ids: TensorArtifactRef
    attention_mask: TensorArtifactRef
    policy_attention_mask: TensorArtifactRef
    reference_attention_mask: TensorArtifactRef
    teacher_attention_mask: TensorArtifactRef
    token_type_ids: TensorArtifactRef | None = None
    original_image_key_block: TensorArtifactRef | None = None
    cache_position: TensorArtifactRef | None = None
    rope_delta: TensorArtifactRef | None = None

    def __post_init__(self) -> None:
        input_shape = self.input_ids.descriptor.shape
        if len(input_shape) != 2:
            raise ValueError("trajectory replay input_ids must have shape [B,S]")
        batch, sequence = input_shape
        for name, ref in (
            ("attention_mask", self.attention_mask),
            ("policy_attention_mask", self.policy_attention_mask),
            ("reference_attention_mask", self.reference_attention_mask),
            ("teacher_attention_mask", self.teacher_attention_mask),
        ):
            if ref.descriptor.shape != (batch, sequence):
                raise ValueError(f"{name} must have shape [B,S]")
        position_shape = self.position_ids.descriptor.shape
        if position_shape != (batch, sequence) and not (
            len(position_shape) == 3 and position_shape[-2:] == (batch, sequence)
        ):
            raise ValueError("position_ids must have shape [B,S] or [R,B,S]")
        if self.token_type_ids is not None and self.token_type_ids.descriptor.shape != (
            batch,
            sequence,
        ):
            raise ValueError("token_type_ids must have shape [B,S]")


@dataclass(frozen=True, slots=True)
class TrajectoryReplayRecord:
    """Content-identified final replay state shared by policy/ref/teacher."""

    schema_version: str
    replay_id: str
    trajectory_id: str
    model: ModelIdentity
    behavior_policy: PolicyVersion
    source_visual: TrajectorySourceVisual
    observation_handles: tuple[ObservationHandle, ...]
    tensors: TrajectoryReplayTensorRefs
    crop_vision_replay_mode: str = "no_crop"
    cache_mode: str = "no_cache"
    cache_prefix_length: int = 0
    deterministic_forward: bool = True
    adapter_dropout: float = 0.0

    def __post_init__(self) -> None:
        if not self.schema_version or not self.replay_id or not self.trajectory_id:
            raise ValueError("trajectory replay identities must be non-empty")
        if not isinstance(self.source_visual, TrajectorySourceVisual):
            raise TypeError(
                "trajectory replay requires a mandatory source visual artifact"
            )
        if self.crop_vision_replay_mode not in {
            "no_crop",
            "shared_frozen_recorded_features",
        }:
            raise ValueError("unknown crop vision replay mode")
        if self.cache_mode not in {"no_cache", "prefill_decode", "recorded_cache"}:
            raise ValueError("unknown trajectory replay cache mode")
        if self.cache_prefix_length < 0:
            raise ValueError("cache prefix length must be non-negative")
        if not self.deterministic_forward:
            raise ValueError("trajectory replay forward must be deterministic")
        if self.adapter_dropout != 0.0:
            raise ValueError("trajectory replay requires zero adapter dropout")


def _raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.layout is not torch.strided:
        raise TypeError("only strided tensors can be materialized")
    cpu = tensor.detach().to(device="cpu").contiguous()
    return cpu.view(torch.uint8).numpy().tobytes()


def tensor_checksum(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_raw_tensor_bytes(tensor)).hexdigest()


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def record_checksum(record: ObservationRecord) -> str:
    payload = json.dumps(
        _canonical(asdict(record)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay_checksum(record: TrajectoryReplayRecord) -> str:
    payload = json.dumps(
        _canonical(asdict(record)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _replay_bundle_checksum(
    replay_handle: TrajectoryReplayHandle,
    observation_records: tuple[ObservationRecord, ...],
    tensor_payloads: tuple[ReplayTensorPayload, ...],
) -> str:
    payload = {
        "schema": "trajectory-replay-bundle-v1",
        "replay": {
            "id": replay_handle.replay_id,
            "sha256": replay_handle.record_sha256,
        },
        "observations": [
            {"id": record.observation_id, "sha256": record_checksum(record)}
            for record in observation_records
        ],
        "tensors": [
            {
                "sha256": item.sha256,
                "shape": tuple(item.tensor.shape),
                "dtype": str(item.tensor.dtype).removeprefix("torch."),
                "stride": tuple(item.tensor.stride()),
                "bytes": item.tensor.numel() * item.tensor.element_size(),
            }
            for item in tensor_payloads
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_replay_bundle(
    bundle: TrajectoryReplayBundle,
) -> TrajectoryReplayBundle:
    """Revalidate a transported bundle, including its mutable tensor bytes."""

    if not isinstance(bundle, TrajectoryReplayBundle):
        raise TypeError("bundle must be TrajectoryReplayBundle")
    for payload in bundle.tensor_payloads:
        if tensor_checksum(payload.tensor) != payload.sha256:
            raise ReplayMismatchError("transport changed replay tensor payload")
    # Re-run all graph/manifest checks after the transport boundary.
    TrajectoryReplayBundle(
        schema_version=bundle.schema_version,
        replay_handle=bundle.replay_handle,
        replay_record=bundle.replay_record,
        observation_records=bundle.observation_records,
        tensor_payloads=bundle.tensor_payloads,
        bundle_sha256=bundle.bundle_sha256,
    )
    return bundle


class ObservationStore:
    """Stores cloned CPU tensors; callers can never mutate recorded rollout data."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._tensors: dict[str, torch.Tensor] = {}
        self._tensor_owners: dict[str, set[str]] = {}
        self._records: dict[str, ObservationRecord] = {}
        self._record_digests: dict[str, str] = {}
        self._replays: dict[str, TrajectoryReplayRecord] = {}
        self._replay_digests: dict[str, str] = {}
        self._released_trajectory_ids: set[str] = set()

    def put_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        alias_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> TensorArtifactRef:
        if not name:
            raise ValueError("tensor name must be non-empty")
        if trajectory_id is not None and (
            not isinstance(trajectory_id, str) or not trajectory_id
        ):
            raise ValueError("tensor trajectory_id must be a non-empty string")
        stored = tensor.detach().to(device="cpu").contiguous().clone()
        raw = _raw_tensor_bytes(stored)
        checksum = hashlib.sha256(raw).hexdigest()
        descriptor = TensorDescriptor(
            shape=tuple(stored.shape),
            dtype=str(stored.dtype).removeprefix("torch."),
            layout="strided",
            stride=tuple(stored.stride()),
            checksum=checksum,
            alias_id=alias_id,
        )
        ref = TensorArtifactRef(
            name=name,
            address=ContentAddress("sha256", checksum, len(raw)),
            descriptor=descriptor,
        )
        with self._lock:
            if trajectory_id in self._released_trajectory_ids:
                raise ReplayMismatchError(
                    "cannot add a tensor for a released trajectory"
                )
            existing = self._tensors.get(checksum)
            if existing is not None:
                if not _same_stored_tensor_semantics(existing, stored):
                    raise IdentityMismatchError(
                        "raw tensor SHA256 was reused by a different dtype/shape/"
                        "stride payload"
                    )
            else:
                self._tensors[checksum] = stored
            if trajectory_id is not None:
                self._tensor_owners.setdefault(checksum, set()).add(trajectory_id)
        return ref

    def put(self, record: ObservationRecord) -> ObservationHandle:
        trajectory_id = _record_trajectory_id(record)
        with self._lock:
            self._assert_trajectory_live(trajectory_id)
            refs = tuple(_walk_tensor_refs(record))
            missing = [
                ref.address.digest
                for ref in refs
                if ref.address.digest not in self._tensors
            ]
            if missing:
                raise ReplayMismatchError(
                    f"observation references missing tensors: {missing}"
                )
            for ref in refs:
                self._verify_ref(ref)
            digest = record_checksum(record)
            old = self._record_digests.get(record.observation_id)
            if old is not None and old != digest:
                raise IdentityMismatchError(
                    "observation ID was reused with different content"
                )
            self._records[record.observation_id] = record
            self._record_digests[record.observation_id] = digest
            self._claim_refs(trajectory_id, refs)
            return ObservationHandle(record.observation_id, digest)

    def resolve_verified(self, ref: TensorArtifactRef) -> torch.Tensor:
        with self._lock:
            self._verify_ref(ref)
            return self._tensors[ref.address.digest].clone()

    def resolve_verified_for_trajectory(
        self, ref: TensorArtifactRef, *, trajectory_id: str
    ) -> torch.Tensor:
        """Resolve a tensor while enforcing the batch's liveness boundary."""

        with self._lock:
            self._assert_trajectory_live(trajectory_id)
            owners = self._tensor_owners.get(ref.address.digest, set())
            if trajectory_id not in owners:
                raise ReplayMismatchError(
                    "tensor is not owned by the requested trajectory"
                )
            self._verify_ref(ref)
            return self._tensors[ref.address.digest].clone()

    def resolve_record(self, handle: ObservationHandle) -> ObservationRecord:
        with self._lock:
            try:
                record = self._records[handle.observation_id]
            except KeyError as exc:
                raise ReplayMismatchError(
                    f"unknown observation {handle.observation_id!r}"
                ) from exc
            actual = record_checksum(record)
            if (
                actual != handle.record_sha256
                or actual != self._record_digests[handle.observation_id]
            ):
                raise ReplayMismatchError("observation record checksum mismatch")
            return record

    def batch(
        self, handles: Iterable[ObservationHandle]
    ) -> tuple[ObservationRecord, ...]:
        return tuple(self.resolve_record(handle) for handle in handles)

    def put_replay(self, replay: TrajectoryReplayRecord) -> TrajectoryReplayHandle:
        with self._lock:
            self._assert_trajectory_live(replay.trajectory_id)
            return self._put_replay_locked(replay)

    def _put_replay_locked(
        self, replay: TrajectoryReplayRecord
    ) -> TrajectoryReplayHandle:
        refs = tuple(_walk_tensor_refs((replay.tensors, replay.source_visual)))
        for ref in refs:
            self._verify_ref(ref)
        sequence = replay.tensors.input_ids.descriptor.shape[-1]
        _validate_trajectory_source_visual(replay.source_visual, sequence=sequence)
        observations = tuple(
            self.resolve_record(handle) for handle in replay.observation_handles
        )
        call_indices = tuple(record.call_index for record in observations)
        if call_indices != tuple(range(len(call_indices))):
            raise ReplayMismatchError(
                "trajectory replay observations must have contiguous call indices"
            )
        source_identity = _source_state_identity(replay.source_visual.state)
        occupied = set(replay.source_visual.positions)
        representation = None
        source_pixels_sha256 = None
        for record in observations:
            if record.model != replay.model:
                raise IdentityMismatchError(
                    "trajectory replay model differs from observation"
                )
            if _record_policy_version(record) != replay.behavior_policy:
                raise IdentityMismatchError(
                    "trajectory replay policy differs from observation materialization"
                )
            if _source_state_identity(record.source_visual) != source_identity:
                raise ReplayMismatchError(
                    "tool observation source visual differs from trajectory source"
                )
            if _original_image_positions(record) != replay.source_visual.positions:
                raise ReplayMismatchError(
                    "tool observation source positions differ from trajectory source"
                )
            if (
                _record_branch_layers(record)
                != replay.source_visual.deepstack_branch_layers
            ):
                raise ReplayMismatchError(
                    "tool observation source DeepStack layers differ from trajectory source"
                )
            if isinstance(record, FocusedObservationRecord):
                if representation is None:
                    representation = record.representation
                elif record.representation != representation:
                    raise IdentityMismatchError(
                        "representation artifact changed within one replay"
                    )
            elif source_pixels_sha256 is None:
                source_pixels_sha256 = record.source_pixels_sha256
            elif record.source_pixels_sha256 != source_pixels_sha256:
                raise ReplayMismatchError(
                    "multi-crop replay changed the immutable source pixels"
                )
            positions = set(_tool_visual_positions(record))
            if occupied & positions:
                raise ReplayMismatchError("multi-call replay visual positions overlap")
            occupied.update(positions)
        has_crop = any(
            isinstance(record, CropObservationRecord) for record in observations
        )
        if has_crop and replay.crop_vision_replay_mode != (
            "shared_frozen_recorded_features"
        ):
            raise ReplayMismatchError(
                "crop replay requires an explicit shared frozen-vision contract"
            )
        if not has_crop and replay.crop_vision_replay_mode != "no_crop":
            raise ReplayMismatchError(
                "crop vision replay mode was set without a crop observation"
            )
        if any(
            position >= sequence
            for record in observations
            for position in _tool_visual_positions(record)
        ):
            raise ReplayMismatchError(
                "observation visual position lies outside final replay sequence"
            )
        digest = replay_checksum(replay)
        old = self._replay_digests.get(replay.replay_id)
        if old is not None and old != digest:
            raise IdentityMismatchError("replay ID was reused with different content")
        self._replays[replay.replay_id] = replay
        self._replay_digests[replay.replay_id] = digest
        self._claim_refs(replay.trajectory_id, refs)
        return TrajectoryReplayHandle(replay.replay_id, digest)

    def resolve_replay(self, handle: TrajectoryReplayHandle) -> TrajectoryReplayRecord:
        with self._lock:
            try:
                replay = self._replays[handle.replay_id]
            except KeyError as exc:
                raise ReplayMismatchError(
                    f"unknown replay {handle.replay_id!r}"
                ) from exc
            actual = replay_checksum(replay)
            if (
                actual != handle.record_sha256
                or actual != self._replay_digests[handle.replay_id]
            ):
                raise ReplayMismatchError("trajectory replay checksum mismatch")
            for ref in _walk_tensor_refs((replay.tensors, replay.source_visual)):
                self._verify_ref(ref)
            _validate_trajectory_source_visual(
                replay.source_visual,
                sequence=replay.tensors.input_ids.descriptor.shape[-1],
            )
            for observation in replay.observation_handles:
                self.resolve_record(observation)
            return replay

    def export_replay_bundle(
        self, handle: TrajectoryReplayHandle
    ) -> TrajectoryReplayBundle:
        """Export only tensors/records reachable from one exact trajectory replay."""

        with self._lock:
            replay = self.resolve_replay(handle)
            records = tuple(
                self.resolve_record(observation)
                for observation in replay.observation_handles
            )
            digests = tuple(
                sorted(
                    {
                        ref.address.digest
                        for value in (replay.tensors, replay.source_visual, *records)
                        for ref in _walk_tensor_refs(value)
                    }
                )
            )
            payloads = tuple(
                ReplayTensorPayload(digest, self._tensors[digest]) for digest in digests
            )
            bundle_sha256 = _replay_bundle_checksum(handle, records, payloads)
            return TrajectoryReplayBundle(
                schema_version="trajectory-replay-bundle-v1",
                replay_handle=handle,
                replay_record=replay,
                observation_records=records,
                tensor_payloads=payloads,
                bundle_sha256=bundle_sha256,
            )

    @classmethod
    def from_replay_bundle(
        cls, bundle: TrajectoryReplayBundle
    ) -> tuple["ObservationStore", TrajectoryReplayHandle]:
        """Reconstruct and re-verify an isolated worker-local exact store."""

        validate_replay_bundle(bundle)
        # Reconstructing the immutable dataclass reruns its full manifest and
        # tensor checksum validation after any transport/pickle boundary.
        verified = TrajectoryReplayBundle(
            schema_version=bundle.schema_version,
            replay_handle=bundle.replay_handle,
            replay_record=bundle.replay_record,
            observation_records=bundle.observation_records,
            tensor_payloads=tuple(
                ReplayTensorPayload(payload.sha256, payload.tensor)
                for payload in bundle.tensor_payloads
            ),
            bundle_sha256=bundle.bundle_sha256,
        )
        store = cls()
        for payload in verified.tensor_payloads:
            tensor = payload.tensor.detach().cpu().contiguous().clone()
            if tensor_checksum(tensor) != payload.sha256:
                raise ReplayMismatchError("transport changed replay tensor payload")
            store._tensors[payload.sha256] = tensor
        handles = tuple(store.put(record) for record in verified.observation_records)
        if handles != verified.replay_record.observation_handles:
            raise ReplayMismatchError("imported observation handles changed")
        replay_handle = store.put_replay(verified.replay_record)
        if replay_handle != verified.replay_handle:
            raise ReplayMismatchError("imported replay handle changed")
        return store, replay_handle

    def prepare_release_trajectories(
        self, trajectory_ids: Iterable[str]
    ) -> tuple[str, ...]:
        """Validate a selective release without mutating stored evidence."""

        identities = _trajectory_id_set(trajectory_ids)
        with self._lock:
            self._prepare_release_locked(identities)
        return identities

    def release_trajectories(
        self, trajectory_ids: Iterable[str]
    ) -> ObservationReleaseCounts:
        """Release one batch while retaining content shared by live batches."""

        identities = _trajectory_id_set(trajectory_ids)
        with self._lock:
            self._prepare_release_locked(identities)
            replay_ids = tuple(
                replay_id
                for replay_id, replay in self._replays.items()
                if replay.trajectory_id in identities
            )
            record_ids = tuple(
                observation_id
                for observation_id, record in self._records.items()
                if _record_trajectory_id(record) in identities
            )
            for replay_id in replay_ids:
                del self._replays[replay_id]
                del self._replay_digests[replay_id]
            for observation_id in record_ids:
                del self._records[observation_id]
                del self._record_digests[observation_id]

            candidate_digests: set[str] = set()
            for digest, owners in tuple(self._tensor_owners.items()):
                if owners.intersection(identities):
                    candidate_digests.add(digest)
                    owners.difference_update(identities)
                    if not owners:
                        del self._tensor_owners[digest]
            live_digests = {
                ref.address.digest
                for value in (*self._records.values(), *self._replays.values())
                for ref in _walk_tensor_refs(value)
            }
            removed_tensors = 0
            for digest in candidate_digests:
                if digest not in self._tensor_owners and digest not in live_digests:
                    if self._tensors.pop(digest, None) is not None:
                        removed_tensors += 1
            self._released_trajectory_ids.update(identities)
            return ObservationReleaseCounts(
                records=len(record_ids),
                replays=len(replay_ids),
                tensors=removed_tensors,
            )

    def resource_counts(self) -> ObservationReleaseCounts:
        with self._lock:
            return ObservationReleaseCounts(
                records=len(self._records),
                replays=len(self._replays),
                tensors=len(self._tensors),
            )

    def checkpoint_state(self) -> dict[str, object]:
        with self._lock:
            return {
                "records": dict(self._records),
                "record_digests": dict(self._record_digests),
                "replays": dict(self._replays),
                "replay_digests": dict(self._replay_digests),
                "tensors": {
                    key: tensor.clone() for key, tensor in self._tensors.items()
                },
            }

    @classmethod
    def from_checkpoint_state(cls, state: Mapping[str, object]) -> "ObservationStore":
        store = cls()
        records = state.get("records")
        digests = state.get("record_digests")
        tensors = state.get("tensors")
        replays = state.get("replays", {})
        replay_digests = state.get("replay_digests", {})
        if (
            not isinstance(records, dict)
            or not isinstance(digests, dict)
            or not isinstance(tensors, dict)
            or not isinstance(replays, dict)
            or not isinstance(replay_digests, dict)
        ):
            raise ReplayMismatchError("malformed observation-store checkpoint")
        store._records = dict(records)
        store._record_digests = dict(digests)
        store._replays = dict(replays)
        store._replay_digests = dict(replay_digests)
        store._tensors = {
            str(key): value.detach().cpu().contiguous().clone()
            for key, value in tensors.items()
            if isinstance(value, torch.Tensor)
        }
        if len(store._tensors) != len(tensors):
            raise ReplayMismatchError("checkpoint contains non-tensor payload")
        for observation_id, record in store._records.items():
            if not isinstance(
                record, (FocusedObservationRecord, CropObservationRecord)
            ):
                raise ReplayMismatchError(
                    "checkpoint contains invalid observation record"
                )
            expected = store._record_digests.get(observation_id)
            if expected != record_checksum(record):
                raise ReplayMismatchError("checkpoint record checksum mismatch")
            refs = tuple(_walk_tensor_refs(record))
            for ref in refs:
                store._verify_ref(ref)
            store._claim_refs(_record_trajectory_id(record), refs)
        for replay_id, replay in store._replays.items():
            if not isinstance(replay, TrajectoryReplayRecord):
                raise ReplayMismatchError(
                    "checkpoint contains invalid trajectory replay"
                )
            if store._replay_digests.get(replay_id) != replay_checksum(replay):
                raise ReplayMismatchError("checkpoint replay checksum mismatch")
            refs = tuple(_walk_tensor_refs((replay.tensors, replay.source_visual)))
            for ref in refs:
                store._verify_ref(ref)
            store._claim_refs(replay.trajectory_id, refs)
            _validate_trajectory_source_visual(
                replay.source_visual,
                sequence=replay.tensors.input_ids.descriptor.shape[-1],
            )
            for handle in replay.observation_handles:
                store.resolve_record(handle)
        return store

    def _prepare_release_locked(self, trajectory_ids: tuple[str, ...]) -> None:
        already_released = set(trajectory_ids) & self._released_trajectory_ids
        if already_released:
            # Releasing the same trajectories is an idempotent no-op.  A batch
            # manager prevents an old identity from being opened again.
            return
        record_owners = {
            observation_id: _record_trajectory_id(record)
            for observation_id, record in self._records.items()
        }
        for replay in self._replays.values():
            if replay.trajectory_id in trajectory_ids:
                continue
            foreign = tuple(
                handle.observation_id
                for handle in replay.observation_handles
                if record_owners.get(handle.observation_id) in trajectory_ids
            )
            if foreign:
                raise ReplayMismatchError(
                    "cannot release observations referenced by another batch"
                )

    def _assert_trajectory_live(self, trajectory_id: str) -> None:
        if trajectory_id in self._released_trajectory_ids:
            raise ReplayMismatchError("trajectory observation resources were released")

    def _claim_refs(
        self, trajectory_id: str, refs: Iterable[TensorArtifactRef]
    ) -> None:
        for ref in refs:
            self._tensor_owners.setdefault(ref.address.digest, set()).add(trajectory_id)

    def _verify_ref(self, ref: TensorArtifactRef) -> None:
        try:
            tensor = self._tensors[ref.address.digest]
        except KeyError as exc:
            raise ReplayMismatchError(f"missing tensor {ref.name!r}") from exc
        actual = tensor_checksum(tensor)
        if actual != ref.address.digest or actual != ref.descriptor.checksum:
            raise ReplayMismatchError(f"tensor checksum mismatch for {ref.name!r}")
        if tuple(tensor.shape) != ref.descriptor.shape:
            raise ReplayMismatchError(f"tensor shape mismatch for {ref.name!r}")
        if str(tensor.dtype).removeprefix("torch.") != ref.descriptor.dtype:
            raise ReplayMismatchError(f"tensor dtype mismatch for {ref.name!r}")
        if tuple(tensor.stride()) != ref.descriptor.stride:
            raise ReplayMismatchError(f"tensor stride mismatch for {ref.name!r}")


def _same_stored_tensor_semantics(first: torch.Tensor, second: torch.Tensor) -> bool:
    return (
        first.dtype == second.dtype
        and tuple(first.shape) == tuple(second.shape)
        and tuple(first.stride()) == tuple(second.stride())
        and _raw_tensor_bytes(first) == _raw_tensor_bytes(second)
    )


def _trajectory_id_set(trajectory_ids: Iterable[str]) -> tuple[str, ...]:
    identities = tuple(trajectory_ids)
    if not identities or any(
        not isinstance(identity, str) or not identity for identity in identities
    ):
        raise ValueError("trajectory_ids must contain non-empty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("trajectory_ids must be unique")
    return identities


def _record_trajectory_id(record: ObservationRecord) -> str:
    if isinstance(record, FocusedObservationRecord):
        return record.condition.trajectory_ids[0]
    if isinstance(record, CropObservationRecord):
        return record.trajectory_id
    raise TypeError("unknown observation record type")


def _walk_tensor_refs(value: object) -> Iterable[TensorArtifactRef]:
    if isinstance(value, TensorArtifactRef):
        yield value
    elif hasattr(value, "__dataclass_fields__"):
        for field_name in value.__dataclass_fields__:
            yield from _walk_tensor_refs(getattr(value, field_name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_tensor_refs(item)


def _record_policy_version(record: ObservationRecord) -> PolicyVersion:
    if isinstance(record, FocusedObservationRecord):
        return record.condition.policy_version
    return record.policy_version


def _source_state_identity(source: SourceVisualState) -> tuple[object, ...]:
    return (
        source.image_sha256,
        source.premerge_main.address.digest,
        tuple(ref.address.digest for ref in source.premerge_deepstack),
        source.merged_main.address.digest,
        tuple(ref.address.digest for ref in source.merged_deepstack),
        source.image_grid_thw,
        source.spatial_merge_size,
    )


def _original_image_positions(record: ObservationRecord) -> tuple[int, ...]:
    if isinstance(record, FocusedObservationRecord):
        return record.layout.original_image_positions
    return record.original_image_positions


def _tool_visual_positions(record: ObservationRecord) -> tuple[int, ...]:
    if isinstance(record, FocusedObservationRecord):
        return record.layout.d_positions
    return record.crop_visual.positions


def _record_branch_layers(record: ObservationRecord) -> tuple[int, ...]:
    if isinstance(record, FocusedObservationRecord):
        return record.layout.deepstack_branch_layers
    return record.crop_visual.deepstack_branch_layers


def _validate_trajectory_source_visual(
    source: TrajectorySourceVisual, *, sequence: int
) -> None:
    positions = source.positions + tuple(
        position
        for branch_positions in source.deepstack_injection_positions
        for position in branch_positions
    )
    if any(position >= sequence for position in positions):
        raise ReplayMismatchError(
            "trajectory source visual position lies outside final replay sequence"
        )
