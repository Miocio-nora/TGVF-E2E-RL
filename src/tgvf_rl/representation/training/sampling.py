"""Deterministic same-image multi-target batch sampling."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha1, sha256
import json
import random

from .schema import RepresentationTrainingSample


SAMPLER_STATE_SCHEMA_VERSION = "same_image_batch_sampler_state_v1"
SAMPLER_IDENTITY_SCHEMA_VERSION = "same_image_batch_sampler_identity_v1"


@dataclass(frozen=True, slots=True)
class _OwnedImageGroup:
    key: str
    sample_indices: tuple[int, ...]


def same_image_group_owner(image_group_key: str, *, world_size: int) -> int:
    """Return the legacy whole-group owner for an image key."""

    if not isinstance(image_group_key, str) or not image_group_key.strip():
        raise ValueError("image_group_key must be a non-empty string")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise ValueError("world_size must be a positive integer")
    return int(sha1(image_group_key.encode("utf-8")).hexdigest(), 16) % world_size


def partition_same_image_group(*, group_size: int, batch_size: int) -> tuple[int, ...]:
    """Partition as much of one group as possible into permitted batch sizes.

    Golden clean training allowed sizes four and five when the configured local
    batch size was five. All other configurations require full local batches.
    Any suffix that has no exact permitted partition is omitted.
    """

    _validate_non_negative_int(group_size, field_name="group_size")
    _validate_positive_int(batch_size, field_name="batch_size")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for same-image Matrix CE")
    if group_size == 0:
        return ()

    minimum = 4 if batch_size == 5 else batch_size
    exact_partitions: list[tuple[int, ...] | None] = [None] * (group_size + 1)
    exact_partitions[0] = ()
    for used in range(1, group_size + 1):
        for size in range(batch_size, minimum - 1, -1):
            if used < size:
                continue
            prefix = exact_partitions[used - size]
            if prefix is not None:
                exact_partitions[used] = (*prefix, size)
                break

    for used in range(group_size, minimum - 1, -1):
        partition = exact_partitions[used]
        if partition is not None:
            return partition
    return ()


class SameImageBatchSampler:
    """Stateful batch sampler that never falls back to independent shuffling.

    A group is assigned wholesale to one data-parallel rank. Iterating yields
    the unconsumed part of exactly one epoch; :meth:`next_batch` additionally
    provides the clean cursor-style infinite interface used by training loops.
    """

    def __init__(
        self,
        samples: Sequence[RepresentationTrainingSample],
        *,
        batch_size: int,
        seed: int,
        data_manifest_sha256: str,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not samples:
            raise ValueError("same-image sampling requires at least one sample")
        _validate_positive_int(batch_size, field_name="batch_size")
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 for same-image Matrix CE")
        _validate_int(seed, field_name="seed")
        _validate_positive_int(world_size, field_name="world_size")
        _validate_int(rank, field_name="rank")
        _validate_sha256(data_manifest_sha256, field_name="data_manifest_sha256")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if not all(
            isinstance(sample, RepresentationTrainingSample) for sample in samples
        ):
            raise TypeError(
                "samples must contain only RepresentationTrainingSample values"
            )

        sample_ids = tuple(sample.sample_id for sample in samples)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_id values must be unique")
        _validate_distinct_targets_within_image_groups(samples)

        self._samples = tuple(samples)
        self.batch_size = batch_size
        self.seed = seed
        self.data_manifest_sha256 = data_manifest_sha256
        self.rank = rank
        self.world_size = world_size
        groups_by_rank = tuple(
            self._build_owned_groups(owner_rank) for owner_rank in range(world_size)
        )
        empty_ranks = tuple(
            owner_rank for owner_rank, groups in enumerate(groups_by_rank) if not groups
        )
        if empty_ranks:
            raise ValueError(
                "same-image distributed preflight found ranks with no permitted "
                f"image group: {empty_ranks}"
            )
        self._groups = groups_by_rank[rank]
        self._usable_group_counts_by_rank = tuple(map(len, groups_by_rank))

        self._identity_sha256 = self._build_identity_sha256()
        self._next_epoch = 0
        self._active_epoch: int | None = None
        self._epoch_batches: tuple[tuple[int, ...], ...] = ()
        self._batch_cursor = 0

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    @property
    def owned_group_keys(self) -> tuple[str, ...]:
        return tuple(group.key for group in self._groups)

    @property
    def next_epoch(self) -> int:
        return self._next_epoch

    @property
    def active_epoch(self) -> int | None:
        return self._active_epoch

    @property
    def batch_cursor(self) -> int:
        return self._batch_cursor

    @property
    def local_epoch_batch_count(self) -> int:
        """Number of local batches before this rank refreshes its shuffle epoch."""

        return sum(
            len(
                partition_same_image_group(
                    group_size=len(group.sample_indices),
                    batch_size=self.batch_size,
                )
            )
            for group in self._groups
        )

    @property
    def usable_group_counts_by_rank(self) -> tuple[int, ...]:
        return self._usable_group_counts_by_rank

    def __len__(self) -> int:
        if self.world_size > 1:
            raise RuntimeError(
                "distributed representation training has rank-local epoch lengths; "
                "use globally fixed training steps with next_batch()"
            )
        return self.local_epoch_batch_count

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        if self.world_size > 1:
            raise RuntimeError(
                "distributed representation training must not terminate on a "
                "rank-local iterator; use globally fixed steps with next_batch()"
            )
        if not self._epoch_batches or self._batch_cursor >= len(self._epoch_batches):
            self._materialize_next_epoch()
        while self._batch_cursor < len(self._epoch_batches):
            batch = self._epoch_batches[self._batch_cursor]
            self._batch_cursor += 1
            yield batch

    def next_batch(self) -> tuple[int, ...]:
        """Return one batch, crossing epoch boundaries when necessary."""

        if not self._epoch_batches or self._batch_cursor >= len(self._epoch_batches):
            self._materialize_next_epoch()
        batch = self._epoch_batches[self._batch_cursor]
        self._batch_cursor += 1
        return batch

    def state_dict(self) -> dict[str, object]:
        """Return JSON-serializable state sufficient for the exact next batch."""

        return {
            "schema_version": SAMPLER_STATE_SCHEMA_VERSION,
            "identity_sha256": self.identity_sha256,
            "next_epoch": self._next_epoch,
            "active_epoch": self._active_epoch,
            "epoch_batches": [list(batch) for batch in self._epoch_batches],
            "batch_cursor": self._batch_cursor,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate and atomically restore exact same-image sampler progress."""

        if not isinstance(state, Mapping):
            raise TypeError("sampler state must be a mapping")
        required_keys = {
            "schema_version",
            "identity_sha256",
            "next_epoch",
            "active_epoch",
            "epoch_batches",
            "batch_cursor",
        }
        if set(state) != required_keys:
            raise ValueError("sampler state fields do not match the v1 schema")
        if state["schema_version"] != SAMPLER_STATE_SCHEMA_VERSION:
            raise ValueError("sampler state schema_version mismatch")
        if state["identity_sha256"] != self.identity_sha256:
            raise ValueError("sampler state identity does not match this sampler")

        next_epoch = _state_non_negative_int(
            state["next_epoch"], field_name="next_epoch"
        )
        batch_cursor = _state_non_negative_int(
            state["batch_cursor"], field_name="batch_cursor"
        )
        active_epoch_value = state["active_epoch"]
        active_epoch = (
            None
            if active_epoch_value is None
            else _state_non_negative_int(active_epoch_value, field_name="active_epoch")
        )
        epoch_batches = _state_batches(state["epoch_batches"])

        if active_epoch is None:
            if next_epoch != 0 or epoch_batches or batch_cursor != 0:
                raise ValueError(
                    "unmaterialized sampler state must be at its initial position"
                )
        else:
            if next_epoch != active_epoch + 1:
                raise ValueError("next_epoch must immediately follow active_epoch")
            expected_batches = self._materialize_epoch(active_epoch)
            if epoch_batches != expected_batches:
                raise ValueError(
                    "materialized epoch batches do not match sampler identity"
                )
            if batch_cursor > len(epoch_batches):
                raise ValueError("batch_cursor exceeds the materialized epoch")

        self._next_epoch = next_epoch
        self._active_epoch = active_epoch
        self._epoch_batches = epoch_batches
        self._batch_cursor = batch_cursor

    def _build_owned_groups(self, owner_rank: int) -> tuple[_OwnedImageGroup, ...]:
        by_key: dict[str, list[int]] = {}
        for index, sample in enumerate(self._samples):
            key = sample.image_group_key
            if same_image_group_owner(key, world_size=self.world_size) != owner_rank:
                continue
            by_key.setdefault(key, []).append(index)

        groups = []
        for key, sample_indices in by_key.items():
            if partition_same_image_group(
                group_size=len(sample_indices), batch_size=self.batch_size
            ):
                groups.append(
                    _OwnedImageGroup(key=key, sample_indices=tuple(sample_indices))
                )
        return tuple(groups)

    def _build_identity_sha256(self) -> str:
        payload = {
            "schema_version": SAMPLER_IDENTITY_SCHEMA_VERSION,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "data_manifest_sha256": self.data_manifest_sha256,
            "rank": self.rank,
            "world_size": self.world_size,
            "samples": [
                {
                    "index": index,
                    "sample_id": sample.sample_id,
                    "image_group_key": sample.image_group_key,
                    "content_sha256": sample.content_sha256,
                }
                for index, sample in enumerate(self._samples)
            ],
            "usable_group_counts_by_rank": list(self.usable_group_counts_by_rank),
            "owned_groups": [
                {"key": group.key, "sample_indices": list(group.sample_indices)}
                for group in self._groups
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _materialize_next_epoch(self) -> None:
        epoch = self._next_epoch
        batches = self._materialize_epoch(epoch)
        if not batches:
            raise RuntimeError("same-image sampler produced no permitted batches")
        self._active_epoch = epoch
        self._next_epoch = epoch + 1
        self._epoch_batches = batches
        self._batch_cursor = 0

    def _materialize_epoch(self, epoch: int) -> tuple[tuple[int, ...], ...]:
        rng = random.Random(self.seed + epoch)
        groups = list(self._groups)
        rng.shuffle(groups)
        batches: list[tuple[int, ...]] = []
        for group in groups:
            indices = list(group.sample_indices)
            rng.shuffle(indices)
            start = 0
            for take in partition_same_image_group(
                group_size=len(indices), batch_size=self.batch_size
            ):
                batches.append(tuple(indices[start : start + take]))
                start += take
        return tuple(batches)


def _state_batches(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("epoch_batches must be a sequence")
    batches = []
    for raw_batch in value:
        if not isinstance(raw_batch, (list, tuple)) or not raw_batch:
            raise ValueError(
                "each materialized epoch batch must be a non-empty sequence"
            )
        batch = tuple(
            _state_non_negative_int(index, field_name="sample index")
            for index in raw_batch
        )
        batches.append(batch)
    return tuple(batches)


def _validate_distinct_targets_within_image_groups(
    samples: Sequence[RepresentationTrainingSample],
) -> None:
    seen: dict[str, dict[str, str]] = {}
    for sample in samples:
        group_targets = seen.setdefault(sample.image_group_key, {})
        prior_sample_id = group_targets.get(sample.target)
        if prior_sample_id is not None:
            raise ValueError(
                "same-image Matrix CE requires exact distinct target strings; "
                f"group={sample.image_group_key!r} target={sample.target!r} "
                f"samples=({prior_sample_id!r}, {sample.sample_id!r})"
            )
        group_targets[sample.target] = sample.sample_id


def _state_non_negative_int(value: object, *, field_name: str) -> int:
    _validate_int(value, field_name=field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _validate_positive_int(value: object, *, field_name: str) -> None:
    _validate_int(value, field_name=field_name)
    if value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_non_negative_int(value: object, *, field_name: str) -> None:
    _validate_int(value, field_name=field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")


def _validate_sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")
