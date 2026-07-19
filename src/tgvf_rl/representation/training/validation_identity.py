"""Content-bound validation and data-split identities.

The retained JSONL manifest binds resolved image paths, but it deliberately
does not read image bytes.  This module adds the missing byte-level audit and
combines it with the validation sampler/cadence and exact split-overlap
contract.  Builders are read-only: they resolve and hash existing regular
files and never rewrite a dataset or image.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat

from .data import (
    RepresentationDataManifest,
    RepresentationDataset,
    SplitOverlapKind,
    SplitOverlapPolicy,
    SplitOverlapReport,
    train_validation_group_overlap,
)


REPRESENTATION_IMAGE_RAW_BYTE_MANIFEST_SCHEMA_VERSION = (
    "representation_image_raw_byte_manifest_v1"
)
REPRESENTATION_VALIDATION_DATA_IDENTITY_SCHEMA_VERSION = (
    "representation_validation_data_identity_v1"
)
REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION = (
    "representation_validation_event_v1"
)

_HASH_CHUNK_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class RepresentationValidationIdentityError(ValueError):
    """A validation/data-split identity cannot be established exactly."""


@dataclass(frozen=True, slots=True)
class ImageRawByteManifestEntry:
    """One unique, fully resolved image path and its exact current bytes."""

    resolved_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _resolved_absolute_path_text(self.resolved_path, field_name="resolved_path")
        _non_negative_int(self.size_bytes, field_name="size_bytes")
        _lowercase_sha256(self.sha256, field_name="sha256")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "resolved_path": self.resolved_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ImageRawByteManifest:
    """Path-deduplicated, path-sorted image-byte manifest.

    Distinct resolved paths remain distinct entries even when their bytes are
    equal.  This binds both the path-to-content association and every byte read
    by the representation data pipeline.
    """

    entries: tuple[ImageRawByteManifestEntry, ...]
    schema_version: str = REPRESENTATION_IMAGE_RAW_BYTE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPRESENTATION_IMAGE_RAW_BYTE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("image raw-byte manifest schema mismatch")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError(
                "image raw-byte manifest entries must be a non-empty tuple"
            )
        if any(
            not isinstance(entry, ImageRawByteManifestEntry) for entry in self.entries
        ):
            raise TypeError("image raw-byte manifest contains an untyped entry")
        paths = tuple(entry.resolved_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError(
                "image raw-byte manifest entries must have unique sorted paths"
            )

    @property
    def file_count(self) -> int:
        return len(self.entries)

    @property
    def total_size_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "entries": [entry.canonical_payload() for entry in self.entries],
        }

    @property
    def manifest_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class RepresentationValidationDataIdentity:
    """Restore-invariant validation and exact train/validation split contract."""

    train_retained_manifest_sha256: str
    validation_retained_manifest_sha256: str
    validation_batch_k: int
    validation_sampler_seed: int
    validation_every_optimizer_steps: int
    evaluator_schema_version: str
    overlap_policy: SplitOverlapPolicy
    overlap_report_sha256: str
    overlap_record_count: int
    overlap_kinds: tuple[SplitOverlapKind, ...]
    train_image_manifest_sha256: str
    train_image_file_count: int
    train_image_total_size_bytes: int
    validation_image_manifest_sha256: str
    validation_image_file_count: int
    validation_image_total_size_bytes: int
    schema_version: str = REPRESENTATION_VALIDATION_DATA_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != REPRESENTATION_VALIDATION_DATA_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError("representation validation-data identity schema mismatch")
        for field_name in (
            "train_retained_manifest_sha256",
            "validation_retained_manifest_sha256",
            "overlap_report_sha256",
            "train_image_manifest_sha256",
            "validation_image_manifest_sha256",
        ):
            _lowercase_sha256(getattr(self, field_name), field_name=field_name)
        _positive_int(self.validation_batch_k, field_name="validation_batch_k")
        if self.validation_batch_k < 2:
            raise ValueError("validation_batch_k must be at least two")
        _integer(self.validation_sampler_seed, field_name="validation_sampler_seed")
        _positive_int(
            self.validation_every_optimizer_steps,
            field_name="validation_every_optimizer_steps",
        )
        if (
            self.evaluator_schema_version
            != REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION
        ):
            raise ValueError("representation validation evaluator schema mismatch")
        if not isinstance(self.overlap_policy, SplitOverlapPolicy):
            raise TypeError("overlap_policy must be an explicit SplitOverlapPolicy")
        _non_negative_int(self.overlap_record_count, field_name="overlap_record_count")
        if not isinstance(self.overlap_kinds, tuple) or any(
            not isinstance(kind, SplitOverlapKind) for kind in self.overlap_kinds
        ):
            raise TypeError("overlap_kinds must be an immutable typed tuple")
        if len(self.overlap_kinds) != len(set(self.overlap_kinds)):
            raise ValueError("overlap_kinds must be unique")
        if self.overlap_policy is SplitOverlapPolicy.REQUIRE_DISJOINT:
            if self.overlap_record_count or self.overlap_kinds:
                raise ValueError("disjoint policy requires an empty overlap report")
        elif self.overlap_policy is SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH:
            if self.overlap_record_count < 1:
                raise ValueError("recorded image-path policy requires overlap records")
            if self.overlap_kinds != (SplitOverlapKind.IMAGE_PATH,):
                raise ValueError(
                    "recorded image-path policy permits only image_path overlap"
                )
        else:  # pragma: no cover - typed enum exhaustiveness guard
            raise ValueError("unsupported train/validation overlap policy")
        for field_name in ("train_image_file_count", "validation_image_file_count"):
            _positive_int(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "train_image_total_size_bytes",
            "validation_image_total_size_bytes",
        ):
            _non_negative_int(getattr(self, field_name), field_name=field_name)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "train_retained_manifest_sha256": self.train_retained_manifest_sha256,
            "validation_retained_manifest_sha256": (
                self.validation_retained_manifest_sha256
            ),
            "validation_batch_k": self.validation_batch_k,
            "validation_sampler_seed": self.validation_sampler_seed,
            "validation_every_optimizer_steps": (self.validation_every_optimizer_steps),
            "evaluator_schema_version": self.evaluator_schema_version,
            "overlap_policy": self.overlap_policy.value,
            "overlap_report_sha256": self.overlap_report_sha256,
            "overlap_record_count": self.overlap_record_count,
            "overlap_kinds": [kind.value for kind in self.overlap_kinds],
            "train_image_manifest_sha256": self.train_image_manifest_sha256,
            "train_image_file_count": self.train_image_file_count,
            "train_image_total_size_bytes": self.train_image_total_size_bytes,
            "validation_image_manifest_sha256": (self.validation_image_manifest_sha256),
            "validation_image_file_count": self.validation_image_file_count,
            "validation_image_total_size_bytes": (
                self.validation_image_total_size_bytes
            ),
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class RepresentationValidationDataAudit:
    """Materialized read-only evidence used to construct one identity."""

    identity: RepresentationValidationDataIdentity
    overlap_report: SplitOverlapReport
    train_image_manifest: ImageRawByteManifest
    validation_image_manifest: ImageRawByteManifest

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RepresentationValidationDataIdentity):
            raise TypeError("identity must be RepresentationValidationDataIdentity")
        if not isinstance(self.overlap_report, SplitOverlapReport):
            raise TypeError("overlap_report must be SplitOverlapReport")
        for field_name in ("train_image_manifest", "validation_image_manifest"):
            if not isinstance(getattr(self, field_name), ImageRawByteManifest):
                raise TypeError(f"{field_name} must be ImageRawByteManifest")
        overlap_kinds = _overlap_kinds(self.overlap_report)
        if self.identity.overlap_report_sha256 != self.overlap_report.identity_sha256:
            raise RepresentationValidationIdentityError(
                "overlap report differs from validation-data identity"
            )
        if self.identity.overlap_record_count != len(self.overlap_report.records):
            raise RepresentationValidationIdentityError(
                "overlap record count differs from validation-data identity"
            )
        if self.identity.overlap_kinds != overlap_kinds:
            raise RepresentationValidationIdentityError(
                "overlap kinds differ from validation-data identity"
            )
        _validate_overlap_policy(
            self.overlap_report,
            policy=self.identity.overlap_policy,
            expected_report_sha256=self.identity.overlap_report_sha256,
        )
        _require_image_binding(
            self.train_image_manifest,
            expected_sha256=self.identity.train_image_manifest_sha256,
            expected_file_count=self.identity.train_image_file_count,
            expected_total_size_bytes=self.identity.train_image_total_size_bytes,
            split_name="train",
        )
        _require_image_binding(
            self.validation_image_manifest,
            expected_sha256=self.identity.validation_image_manifest_sha256,
            expected_file_count=self.identity.validation_image_file_count,
            expected_total_size_bytes=(self.identity.validation_image_total_size_bytes),
            split_name="validation",
        )


def build_image_raw_byte_manifest(
    resolved_paths: Sequence[str | os.PathLike[str]],
) -> ImageRawByteManifest:
    """Read and hash unique, already-resolved regular files without writing.

    Duplicate input paths are collapsed.  Symlink aliases, relative paths, and
    lexically non-canonical paths are rejected rather than silently changing
    the retained manifest identity.
    """

    if isinstance(resolved_paths, (str, bytes, os.PathLike)) or not isinstance(
        resolved_paths, Sequence
    ):
        raise TypeError("resolved_paths must be a non-string sequence")
    if not resolved_paths:
        raise ValueError("resolved_paths must be non-empty")

    unique_paths: dict[str, Path] = {}
    for index, raw_path in enumerate(resolved_paths):
        path = _require_existing_resolved_file(raw_path, index=index)
        unique_paths[str(path)] = path

    entries = tuple(
        _hash_image_file(unique_paths[path_text]) for path_text in sorted(unique_paths)
    )
    return ImageRawByteManifest(entries=entries)


def build_retained_image_raw_byte_manifest(
    manifest: RepresentationDataManifest,
) -> ImageRawByteManifest:
    """Hash every unique image path consumed by one retained data manifest."""

    if not isinstance(manifest, RepresentationDataManifest):
        raise TypeError("manifest must be a RepresentationDataManifest")
    return build_image_raw_byte_manifest(
        tuple(entry.resolved_image_path for entry in manifest.accepted_rows)
    )


def build_representation_validation_data_audit(
    *,
    train_dataset: RepresentationDataset,
    validation_dataset: RepresentationDataset,
    validation_batch_k: int,
    validation_sampler_seed: int,
    validation_every_optimizer_steps: int,
    evaluator_schema_version: str,
    overlap_policy: SplitOverlapPolicy,
    expected_overlap_report_sha256: str,
) -> RepresentationValidationDataAudit:
    """Build the exact validation/split identity and all byte-level evidence."""

    if not isinstance(train_dataset, RepresentationDataset):
        raise TypeError("train_dataset must be a RepresentationDataset")
    if not isinstance(validation_dataset, RepresentationDataset):
        raise TypeError("validation_dataset must be a RepresentationDataset")
    _positive_int(validation_batch_k, field_name="validation_batch_k")
    if validation_batch_k < 2:
        raise ValueError("validation_batch_k must be at least two")
    _integer(validation_sampler_seed, field_name="validation_sampler_seed")
    _positive_int(
        validation_every_optimizer_steps,
        field_name="validation_every_optimizer_steps",
    )
    if evaluator_schema_version != REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION:
        raise ValueError("representation validation evaluator schema mismatch")
    if not isinstance(overlap_policy, SplitOverlapPolicy):
        raise TypeError("overlap_policy must be an explicit SplitOverlapPolicy")
    _lowercase_sha256(
        expected_overlap_report_sha256,
        field_name="expected_overlap_report_sha256",
    )

    overlap_report = train_validation_group_overlap(
        train_dataset.samples, validation_dataset.samples
    )
    if overlap_report.identity_sha256 != expected_overlap_report_sha256:
        raise RepresentationValidationIdentityError(
            "actual train/validation overlap report differs from expected SHA256"
        )
    _validate_overlap_policy(
        overlap_report,
        policy=overlap_policy,
        expected_report_sha256=expected_overlap_report_sha256,
    )

    train_images = build_retained_image_raw_byte_manifest(train_dataset.manifest)
    validation_images = build_retained_image_raw_byte_manifest(
        validation_dataset.manifest
    )
    identity = RepresentationValidationDataIdentity(
        train_retained_manifest_sha256=train_dataset.manifest.manifest_sha256,
        validation_retained_manifest_sha256=(
            validation_dataset.manifest.manifest_sha256
        ),
        validation_batch_k=validation_batch_k,
        validation_sampler_seed=validation_sampler_seed,
        validation_every_optimizer_steps=validation_every_optimizer_steps,
        evaluator_schema_version=evaluator_schema_version,
        overlap_policy=overlap_policy,
        overlap_report_sha256=overlap_report.identity_sha256,
        overlap_record_count=len(overlap_report.records),
        overlap_kinds=_overlap_kinds(overlap_report),
        train_image_manifest_sha256=train_images.manifest_sha256,
        train_image_file_count=train_images.file_count,
        train_image_total_size_bytes=train_images.total_size_bytes,
        validation_image_manifest_sha256=validation_images.manifest_sha256,
        validation_image_file_count=validation_images.file_count,
        validation_image_total_size_bytes=validation_images.total_size_bytes,
    )
    return RepresentationValidationDataAudit(
        identity=identity,
        overlap_report=overlap_report,
        train_image_manifest=train_images,
        validation_image_manifest=validation_images,
    )


def _validate_overlap_policy(
    report: SplitOverlapReport,
    *,
    policy: SplitOverlapPolicy,
    expected_report_sha256: str,
) -> None:
    if policy is SplitOverlapPolicy.REQUIRE_DISJOINT:
        report.validate_policy(policy, expected_report_sha256=None)
        return
    if policy is SplitOverlapPolicy.ALLOW_RECORDED_IMAGE_PATH:
        if report.is_disjoint:
            raise RepresentationValidationIdentityError(
                "recorded image-path policy requires a non-empty exact report"
            )
        report.validate_policy(policy, expected_report_sha256=expected_report_sha256)
        return
    raise ValueError("unsupported train/validation overlap policy")


def _overlap_kinds(report: SplitOverlapReport) -> tuple[SplitOverlapKind, ...]:
    return tuple(dict.fromkeys(record.kind for record in report.records))


def _require_image_binding(
    manifest: ImageRawByteManifest,
    *,
    expected_sha256: str,
    expected_file_count: int,
    expected_total_size_bytes: int,
    split_name: str,
) -> None:
    if (
        manifest.manifest_sha256 != expected_sha256
        or manifest.file_count != expected_file_count
        or manifest.total_size_bytes != expected_total_size_bytes
    ):
        raise RepresentationValidationIdentityError(
            f"{split_name} image raw-byte manifest differs from validation-data identity"
        )


def _require_existing_resolved_file(
    raw_path: str | os.PathLike[str], *, index: int
) -> Path:
    try:
        path_text = os.fspath(raw_path)
    except TypeError as exc:
        raise TypeError(f"resolved_paths[{index}] must be str or PathLike") from exc
    if not isinstance(path_text, str):
        raise TypeError(f"resolved_paths[{index}] must resolve to text, not bytes")
    _resolved_absolute_path_text(path_text, field_name=f"resolved_paths[{index}]")
    path = Path(path_text)
    try:
        actual = path.resolve(strict=True)
    except OSError as exc:
        raise RepresentationValidationIdentityError(
            f"resolved_paths[{index}] does not exist: {path_text}"
        ) from exc
    if actual != path:
        raise RepresentationValidationIdentityError(
            f"resolved_paths[{index}] is not already fully resolved: {path_text}"
        )
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise RepresentationValidationIdentityError(
            f"cannot stat resolved_paths[{index}]: {path_text}"
        ) from exc
    if not stat.S_ISREG(mode):
        raise RepresentationValidationIdentityError(
            f"resolved_paths[{index}] is not a regular file: {path_text}"
        )
    return path


def _hash_image_file(path: Path) -> ImageRawByteManifestEntry:
    digest = sha256()
    bytes_read = 0
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RepresentationValidationIdentityError(
                    f"image path is not a regular file: {path}"
                )
            while True:
                chunk = stream.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                bytes_read += len(chunk)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except RepresentationValidationIdentityError:
        raise
    except OSError as exc:
        raise RepresentationValidationIdentityError(
            f"cannot read image bytes exactly: {path}"
        ) from exc

    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    before_state = tuple(getattr(before, name) for name in stable_fields)
    after_state = tuple(getattr(after, name) for name in stable_fields)
    current_state = tuple(getattr(current, name) for name in stable_fields)
    if before_state != after_state or after_state != current_state:
        raise RepresentationValidationIdentityError(
            f"image changed while its raw bytes were being hashed: {path}"
        )
    if bytes_read != before.st_size:
        raise RepresentationValidationIdentityError(
            f"image size changed while its raw bytes were being hashed: {path}"
        )
    return ImageRawByteManifestEntry(
        resolved_path=str(path),
        size_bytes=bytes_read,
        sha256=digest.hexdigest(),
    )


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _resolved_absolute_path_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty path text")
    if not os.path.isabs(value) or os.path.normpath(value) != value:
        raise ValueError(f"{field_name} must be an absolute normalized path")


def _lowercase_sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")


def _integer(value: object, *, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int")


def _positive_int(value: object, *, field_name: str) -> None:
    _integer(value, field_name=field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _non_negative_int(value: object, *, field_name: str) -> None:
    _integer(value, field_name=field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
