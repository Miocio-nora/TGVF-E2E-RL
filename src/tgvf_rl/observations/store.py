"""Bit-preserving in-memory observation store with strict checksum replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tensors import (
    ContentAddress,
    TensorArtifactRef,
    TensorDescriptor,
)

from .schema import CropObservationRecord, FocusedObservationRecord, ObservationRecord


@dataclass(frozen=True, slots=True)
class ObservationHandle:
    observation_id: str
    record_sha256: str


@dataclass(frozen=True, slots=True)
class TrajectoryReplayHandle:
    replay_id: str
    record_sha256: str


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
        if (
            self.token_type_ids is not None
            and self.token_type_ids.descriptor.shape
            != (
                batch,
                sequence,
            )
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


class ObservationStore:
    """Stores cloned CPU tensors; callers can never mutate recorded rollout data."""

    def __init__(self) -> None:
        self._tensors: dict[str, torch.Tensor] = {}
        self._records: dict[str, ObservationRecord] = {}
        self._record_digests: dict[str, str] = {}
        self._replays: dict[str, TrajectoryReplayRecord] = {}
        self._replay_digests: dict[str, str] = {}

    def put_tensor(
        self, name: str, tensor: torch.Tensor, *, alias_id: str | None = None
    ) -> TensorArtifactRef:
        if not name:
            raise ValueError("tensor name must be non-empty")
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
        existing = self._tensors.get(checksum)
        if existing is not None and not torch.equal(existing, stored):
            raise IdentityMismatchError("SHA256 collision for tensor payload")
        self._tensors[checksum] = stored
        return ref

    def put(self, record: ObservationRecord) -> ObservationHandle:
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
        return ObservationHandle(record.observation_id, digest)

    def resolve_verified(self, ref: TensorArtifactRef) -> torch.Tensor:
        self._verify_ref(ref)
        return self._tensors[ref.address.digest].clone()

    def resolve_record(self, handle: ObservationHandle) -> ObservationRecord:
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
        refs = tuple(_walk_tensor_refs(replay.tensors))
        for ref in refs:
            self._verify_ref(ref)
        observations = tuple(
            self.resolve_record(handle) for handle in replay.observation_handles
        )
        call_indices = tuple(record.call_index for record in observations)
        if call_indices != tuple(range(len(call_indices))):
            raise ReplayMismatchError(
                "trajectory replay observations must have contiguous call indices"
            )
        if observations:
            first = observations[0]
            source_identity = _source_identity(first)
            occupied = set(_original_image_positions(first))
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
                if _source_identity(record) != source_identity:
                    raise ReplayMismatchError(
                        "multi-call replay changed the original visual state"
                    )
                if _original_image_positions(record) != _original_image_positions(
                    first
                ):
                    raise ReplayMismatchError(
                        "multi-call replay changed original-image positions"
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
                    raise ReplayMismatchError(
                        "multi-call replay visual positions overlap"
                    )
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
        sequence = replay.tensors.input_ids.descriptor.shape[-1]
        if observations and any(
            position >= sequence
            for record in observations
            for position in (
                _original_image_positions(record) + _tool_visual_positions(record)
            )
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
        return TrajectoryReplayHandle(replay.replay_id, digest)

    def resolve_replay(self, handle: TrajectoryReplayHandle) -> TrajectoryReplayRecord:
        try:
            replay = self._replays[handle.replay_id]
        except KeyError as exc:
            raise ReplayMismatchError(f"unknown replay {handle.replay_id!r}") from exc
        actual = replay_checksum(replay)
        if (
            actual != handle.record_sha256
            or actual != self._replay_digests[handle.replay_id]
        ):
            raise ReplayMismatchError("trajectory replay checksum mismatch")
        for ref in _walk_tensor_refs(replay.tensors):
            self._verify_ref(ref)
        for observation in replay.observation_handles:
            self.resolve_record(observation)
        return replay

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "records": dict(self._records),
            "record_digests": dict(self._record_digests),
            "replays": dict(self._replays),
            "replay_digests": dict(self._replay_digests),
            "tensors": {key: tensor.clone() for key, tensor in self._tensors.items()},
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
            for ref in _walk_tensor_refs(record):
                store._verify_ref(ref)
        for replay_id, replay in store._replays.items():
            if not isinstance(replay, TrajectoryReplayRecord):
                raise ReplayMismatchError(
                    "checkpoint contains invalid trajectory replay"
                )
            if store._replay_digests.get(replay_id) != replay_checksum(replay):
                raise ReplayMismatchError("checkpoint replay checksum mismatch")
            for ref in _walk_tensor_refs(replay.tensors):
                store._verify_ref(ref)
            for handle in replay.observation_handles:
                store.resolve_record(handle)
        return store

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


def _source_identity(record: ObservationRecord) -> tuple[object, ...]:
    source = record.source_visual
    return (
        source.image_sha256,
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
