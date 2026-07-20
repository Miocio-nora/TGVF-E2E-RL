"""Fail-closed runtime loader for prompt-free DeepEyes-47K materializations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .deepeyes47k import (
    DEEPEYES47K_DATASET_ID,
    DEEPEYES47K_MANIFEST_FILE,
    DEEPEYES47K_SAMPLES_FILE,
    DEEPEYES47K_SCHEMA_VERSION,
    DEEPEYES47K_SHUFFLE_ALGORITHM,
    DEEPEYES47K_SNAPSHOT,
    DEEPEYES47K_SOURCE_FILES,
    DEEPEYES47K_TOTAL_ROWS,
    DeepEyesSourceFileSpec,
    DeepEyesSourceValidationError,
    DeepEyesTaskKind,
    validate_materialized_deepeyes47k,
)


DEEPEYES47K_RUNTIME_SCHEMA_VERSION = "tgvf.deepeyes47k.runtime.v1"
DEEPEYES47K_PROMPT_GROUP_UID_SCHEMA = "tgvf.policy-pilot.prompt-group.v1"
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "snapshot",
        "fixture",
        "source_files",
        "source_total_rows",
        "sample_count",
        "shuffle",
        "samples",
        "images",
        "content_sha256",
    }
)
_SOURCE_FILE_KEYS = frozenset({"filename", "rows", "lfs_sha256", "byte_size"})
_ROW_KEYS = frozenset(
    {
        "sample_id",
        "image",
        "extra_info",
        "reward_model",
        "data_source",
        "task_kind",
        "provenance",
        "ability",
        "style",
        "split",
    }
)
_REQUIRED_ROW_KEYS = _ROW_KEYS.difference({"ability", "style", "split"})
_PROVENANCE_KEYS = frozenset(
    {
        "dataset_id",
        "snapshot",
        "source_file",
        "source_file_sha256",
        "source_row_index",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise DeepEyesSourceValidationError(
            "runtime artifact contains non-canonical JSON data"
        ) from error
    return rendered.encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DeepEyesSourceValidationError(
            f"runtime artifact file is unreadable: {path.name}"
        ) from error
    return digest.hexdigest()


def _validate_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeepEyesSourceValidationError(
            f"runtime {field_name} must be a non-empty string"
        )
    return value


@dataclass(frozen=True, slots=True)
class DeepEyes47KRuntimeBinding:
    """External run identity required before opening a materialized dataset."""

    manifest_file_sha256: str
    content_sha256: str
    shuffle_seed: int
    fixture: bool = False
    expected_sample_count: int = DEEPEYES47K_TOTAL_ROWS

    def __post_init__(self) -> None:
        _validate_sha256(
            self.manifest_file_sha256, field_name="manifest_file_sha256"
        )
        _validate_sha256(self.content_sha256, field_name="content_sha256")
        if type(self.shuffle_seed) is not int:
            raise TypeError("shuffle_seed must be an integer")
        if type(self.fixture) is not bool:
            raise TypeError("fixture must be bool")
        if type(self.expected_sample_count) is not int or (
            self.expected_sample_count <= 0
        ):
            raise ValueError("expected_sample_count must be a positive integer")
        if not self.fixture and self.expected_sample_count != DEEPEYES47K_TOTAL_ROWS:
            raise ValueError(
                f"formal DeepEyes-47K requires exactly {DEEPEYES47K_TOTAL_ROWS} samples"
            )
        if self.fixture and self.expected_sample_count >= DEEPEYES47K_TOTAL_ROWS:
            raise ValueError("fixture runtime bindings must remain smaller than 47,052")

    @classmethod
    def formal(
        cls,
        *,
        manifest_file_sha256: str,
        content_sha256: str,
        shuffle_seed: int,
    ) -> DeepEyes47KRuntimeBinding:
        return cls(
            manifest_file_sha256=manifest_file_sha256,
            content_sha256=content_sha256,
            shuffle_seed=shuffle_seed,
            fixture=False,
            expected_sample_count=DEEPEYES47K_TOTAL_ROWS,
        )

    @classmethod
    def fixture_binding(
        cls,
        *,
        manifest_file_sha256: str,
        content_sha256: str,
        shuffle_seed: int,
        expected_sample_count: int,
    ) -> DeepEyes47KRuntimeBinding:
        return cls(
            manifest_file_sha256=manifest_file_sha256,
            content_sha256=content_sha256,
            shuffle_seed=shuffle_seed,
            fixture=True,
            expected_sample_count=expected_sample_count,
        )


@dataclass(frozen=True, slots=True)
class DeepEyes47KRuntimeSample:
    """Prompt-free fields admitted to the policy runtime."""

    sample_id: str
    prompt_group_uid: str
    image_path: Path
    image_sha256: str
    question: str
    ground_truth: Any
    data_source: str
    task_kind: DeepEyesTaskKind
    metadata: Mapping[str, str | int | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_path", Path(self.image_path))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class DeepEyes47KRuntimeDataset(Sequence[DeepEyes47KRuntimeSample]):
    """A verified, seed-bound and deterministically ordered runtime view."""

    root: Path
    binding: DeepEyes47KRuntimeBinding
    dataset_id: str
    snapshot: str
    samples_sha256: str
    iteration_identity_sha256: str
    _samples: tuple[DeepEyes47KRuntimeSample, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "_samples", tuple(self._samples))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, index: int | slice
    ) -> DeepEyes47KRuntimeSample | tuple[DeepEyes47KRuntimeSample, ...]:
        return self._samples[index]

    def __iter__(self) -> Iterator[DeepEyes47KRuntimeSample]:
        return iter(self._samples)


def _load_manifest(root: Path, binding: DeepEyes47KRuntimeBinding) -> dict[str, Any]:
    manifest_path = root / DEEPEYES47K_MANIFEST_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DeepEyesSourceValidationError(
            "runtime manifest must be a regular file inside the artifact root"
        )
    manifest_bytes = manifest_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != binding.manifest_file_sha256:
        raise DeepEyesSourceValidationError("runtime manifest file hash mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise DeepEyesSourceValidationError("runtime manifest JSON is invalid") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise DeepEyesSourceValidationError("runtime manifest schema is invalid")
    if manifest_bytes != _canonical_json_bytes(manifest) + b"\n":
        raise DeepEyesSourceValidationError("runtime manifest is not canonical JSON")

    declared_content_sha256 = manifest.get("content_sha256")
    descriptor = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    computed_content_sha256 = _sha256_bytes(_canonical_json_bytes(descriptor))
    if (
        declared_content_sha256 != computed_content_sha256
        or declared_content_sha256 != binding.content_sha256
    ):
        raise DeepEyesSourceValidationError("runtime manifest content hash mismatch")
    if (
        manifest.get("schema_version") != DEEPEYES47K_SCHEMA_VERSION
        or manifest.get("dataset_id") != DEEPEYES47K_DATASET_ID
    ):
        raise DeepEyesSourceValidationError("runtime dataset/schema identity mismatch")
    if manifest.get("fixture") is not binding.fixture:
        raise DeepEyesSourceValidationError("runtime fixture/formal identity mismatch")
    expected_snapshot = "fixture" if binding.fixture else DEEPEYES47K_SNAPSHOT
    if manifest.get("snapshot") != expected_snapshot:
        raise DeepEyesSourceValidationError("runtime snapshot identity mismatch")

    source_files = manifest.get("source_files")
    if (
        not isinstance(source_files, list)
        or not source_files
        or any(not isinstance(item, dict) or set(item) != _SOURCE_FILE_KEYS for item in source_files)
    ):
        raise DeepEyesSourceValidationError("runtime source-file schema is invalid")
    try:
        source_specs = tuple(
            DeepEyesSourceFileSpec(
                filename=item["filename"],
                rows=item["rows"],
                lfs_sha256=item["lfs_sha256"],
                byte_size=item["byte_size"],
            )
            for item in source_files
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DeepEyesSourceValidationError(
            "runtime source-file identity is invalid"
        ) from error
    if not binding.fixture and source_specs != DEEPEYES47K_SOURCE_FILES:
        raise DeepEyesSourceValidationError("runtime production source pin mismatch")
    source_total_rows = manifest.get("source_total_rows")
    if (
        type(source_total_rows) is not int
        or source_total_rows != sum(item.rows for item in source_specs)
        or source_total_rows != binding.expected_sample_count
        or type(manifest.get("sample_count")) is not int
        or manifest.get("sample_count") != binding.expected_sample_count
    ):
        raise DeepEyesSourceValidationError("runtime row-count identity mismatch")

    shuffle = manifest.get("shuffle")
    if (
        not isinstance(shuffle, dict)
        or set(shuffle) != {"algorithm", "seed"}
        or shuffle.get("algorithm") != DEEPEYES47K_SHUFFLE_ALGORITHM
        or shuffle.get("seed") != binding.shuffle_seed
    ):
        raise DeepEyesSourceValidationError("runtime shuffle-seed identity mismatch")
    samples = manifest.get("samples")
    images = manifest.get("images")
    if (
        not isinstance(samples, dict)
        or set(samples) != {"path", "rows", "sha256"}
        or samples.get("path") != DEEPEYES47K_SAMPLES_FILE
        or samples.get("rows") != binding.expected_sample_count
        or not isinstance(samples.get("sha256"), str)
        or not isinstance(images, dict)
        or images
        != {"directory": "images", "address": "sha256-of-original-bytes"}
    ):
        raise DeepEyesSourceValidationError("runtime sample/image descriptor is invalid")
    return manifest


def _prompt_group_uid(
    *,
    sample_id: str,
    snapshot: str,
    content_sha256: str,
    shuffle_seed: int,
) -> str:
    identity = {
        "schema_version": DEEPEYES47K_PROMPT_GROUP_UID_SCHEMA,
        "dataset_id": DEEPEYES47K_DATASET_ID,
        "snapshot": snapshot,
        "content_sha256": content_sha256,
        "shuffle_seed": shuffle_seed,
        "sample_id": sample_id,
    }
    return f"tgvf-pilot-group:{_sha256_bytes(_canonical_json_bytes(identity))}"


def _resolve_image_path(
    relative_image: object,
    *,
    root: Path,
    images_root: Path,
) -> Path:
    if not isinstance(relative_image, str):
        raise DeepEyesSourceValidationError("runtime image path is invalid")
    pure_image = PurePosixPath(relative_image)
    if (
        pure_image.is_absolute()
        or pure_image.parts[:1] != ("images",)
        or ".." in pure_image.parts
    ):
        raise DeepEyesSourceValidationError("runtime image path escapes images/")
    image_path = root.joinpath(*pure_image.parts)
    if image_path.is_symlink():
        raise DeepEyesSourceValidationError("runtime image path must not be a symlink")
    try:
        resolved_image = image_path.resolve(strict=True)
        resolved_image.relative_to(images_root)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise DeepEyesSourceValidationError(
            "runtime image path escapes the verified image root"
        ) from error
    return resolved_image


def _preflight_image_paths(
    samples_path: Path,
    *,
    root: Path,
    images_root: Path,
    expected_sample_count: int,
) -> None:
    """Reject escaping paths before the shared validator opens image bytes."""

    row_count = 0
    try:
        with samples_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if row_count >= expected_sample_count:
                    raise DeepEyesSourceValidationError(
                        "runtime samples exceed the bound row count"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DeepEyesSourceValidationError(
                        f"runtime sample JSON is invalid at line {line_number}"
                    ) from error
                image = record.get("image") if isinstance(record, dict) else None
                if not isinstance(image, dict) or set(image) != {"path", "sha256"}:
                    raise DeepEyesSourceValidationError(
                        "runtime sample image schema is invalid"
                    )
                _resolve_image_path(
                    image.get("path"), root=root, images_root=images_root
                )
                row_count += 1
    except OSError as error:
        raise DeepEyesSourceValidationError("runtime samples are unreadable") from error
    if row_count != expected_sample_count:
        raise DeepEyesSourceValidationError("runtime preflight row count mismatch")


def _runtime_sample(
    record: object,
    *,
    root: Path,
    images_root: Path,
    binding: DeepEyes47KRuntimeBinding,
    snapshot: str,
) -> DeepEyes47KRuntimeSample:
    if (
        not isinstance(record, dict)
        or not _REQUIRED_ROW_KEYS.issubset(record)
        or not set(record).issubset(_ROW_KEYS)
    ):
        raise DeepEyesSourceValidationError("runtime sample schema is invalid")
    image = record.get("image")
    extra_info = record.get("extra_info")
    reward_model = record.get("reward_model")
    provenance = record.get("provenance")
    if (
        not isinstance(image, dict)
        or set(image) != {"path", "sha256"}
        or not isinstance(extra_info, dict)
        or set(extra_info) != {"question"}
        or not isinstance(reward_model, dict)
        or set(reward_model) != {"ground_truth"}
        or not isinstance(provenance, dict)
        or set(provenance) != _PROVENANCE_KEYS
    ):
        raise DeepEyesSourceValidationError("runtime sample fields are invalid")

    resolved_image = _resolve_image_path(
        image.get("path"), root=root, images_root=images_root
    )

    try:
        task_kind = DeepEyesTaskKind(record.get("task_kind"))
    except ValueError as error:
        raise DeepEyesSourceValidationError("runtime task kind is invalid") from error
    sample_id = _required_string(record.get("sample_id"), field_name="sample_id")
    image_sha256 = _validate_sha256(
        image.get("sha256"), field_name="runtime image sha256"
    )
    if _sha256_file(resolved_image) != image_sha256:
        raise DeepEyesSourceValidationError(
            "runtime image SHA-256 changed after materialization validation"
        )
    ground_truth = json.loads(_canonical_json_bytes(reward_model["ground_truth"]))
    optional_metadata: dict[str, str | None] = {}
    for field_name in ("ability", "style", "split"):
        value = record.get(field_name)
        optional_metadata[field_name] = (
            None
            if value is None
            else _required_string(value, field_name=field_name)
        )
    metadata: dict[str, str | int | None] = {
        **optional_metadata,
        "dataset_id": provenance.get("dataset_id"),
        "snapshot": provenance.get("snapshot"),
        "source_file": provenance.get("source_file"),
        "source_file_sha256": provenance.get("source_file_sha256"),
        "source_row_index": provenance.get("source_row_index"),
    }
    return DeepEyes47KRuntimeSample(
        sample_id=sample_id,
        prompt_group_uid=_prompt_group_uid(
            sample_id=sample_id,
            snapshot=snapshot,
            content_sha256=binding.content_sha256,
            shuffle_seed=binding.shuffle_seed,
        ),
        image_path=resolved_image,
        image_sha256=image_sha256,
        question=_required_string(
            extra_info.get("question"), field_name="question"
        ),
        ground_truth=ground_truth,
        data_source=_required_string(
            record.get("data_source"), field_name="data_source"
        ),
        task_kind=task_kind,
        metadata=metadata,
    )


def load_deepeyes47k_runtime(
    output_root: Path,
    *,
    binding: DeepEyes47KRuntimeBinding,
) -> DeepEyes47KRuntimeDataset:
    """Load only ``manifest.json``/``samples.jsonl`` plus addressed images."""

    if not isinstance(binding, DeepEyes47KRuntimeBinding):
        raise TypeError("binding must be DeepEyes47KRuntimeBinding")
    try:
        root = Path(output_root).resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise DeepEyesSourceValidationError(
            "runtime materialization root is missing"
        ) from error
    if not root.is_dir():
        raise DeepEyesSourceValidationError("runtime materialization root is not a directory")
    manifest = _load_manifest(root, binding)

    samples_path = root / DEEPEYES47K_SAMPLES_FILE
    if samples_path.is_symlink() or not samples_path.is_file():
        raise DeepEyesSourceValidationError(
            "runtime samples must be a regular file inside the artifact root"
        )
    samples_sha256 = manifest["samples"]["sha256"]
    if _sha256_file(samples_path) != samples_sha256:
        raise DeepEyesSourceValidationError("runtime samples file hash mismatch")

    unresolved_images_root = root / "images"
    if unresolved_images_root.is_symlink() or not unresolved_images_root.is_dir():
        raise DeepEyesSourceValidationError(
            "runtime images root must be a real directory"
        )
    try:
        images_root = unresolved_images_root.resolve(strict=True)
        images_root.relative_to(root)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise DeepEyesSourceValidationError(
            "runtime images root escapes the artifact root"
        ) from error
    _preflight_image_paths(
        samples_path,
        root=root,
        images_root=images_root,
        expected_sample_count=binding.expected_sample_count,
    )

    # Reuse the materializer's streaming row/provenance/task/image/order verifier.
    # It has no source-parquet or historical-prompt input path.
    validated = validate_materialized_deepeyes47k(root)
    if (
        validated.get("sample_count") != binding.expected_sample_count
        or validated.get("samples_sha256") != samples_sha256
        or validated.get("content_sha256") != binding.content_sha256
        or validated.get("fixture") is not binding.fixture
    ):
        raise DeepEyesSourceValidationError("runtime materialization validation mismatch")

    samples: list[DeepEyes47KRuntimeSample] = []
    try:
        with samples_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if len(samples) >= binding.expected_sample_count:
                    raise DeepEyesSourceValidationError(
                        "runtime samples exceed the bound row count"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DeepEyesSourceValidationError(
                        f"runtime sample JSON is invalid at line {line_number}"
                    ) from error
                samples.append(
                    _runtime_sample(
                        record,
                        root=root,
                        images_root=images_root,
                        binding=binding,
                        snapshot=manifest["snapshot"],
                    )
                )
    except OSError as error:
        raise DeepEyesSourceValidationError("runtime samples are unreadable") from error
    if len(samples) != binding.expected_sample_count:
        raise DeepEyesSourceValidationError("runtime sample count changed during load")
    if _sha256_file(samples_path) != samples_sha256:
        raise DeepEyesSourceValidationError("runtime samples changed during load")

    iteration_identity = {
        "schema_version": DEEPEYES47K_RUNTIME_SCHEMA_VERSION,
        "dataset_id": DEEPEYES47K_DATASET_ID,
        "snapshot": manifest["snapshot"],
        "fixture": binding.fixture,
        "sample_count": len(samples),
        "shuffle_algorithm": DEEPEYES47K_SHUFFLE_ALGORITHM,
        "shuffle_seed": binding.shuffle_seed,
        "manifest_file_sha256": binding.manifest_file_sha256,
        "content_sha256": binding.content_sha256,
        "samples_sha256": samples_sha256,
    }
    return DeepEyes47KRuntimeDataset(
        root=root,
        binding=binding,
        dataset_id=DEEPEYES47K_DATASET_ID,
        snapshot=manifest["snapshot"],
        samples_sha256=samples_sha256,
        iteration_identity_sha256=_sha256_bytes(
            _canonical_json_bytes(iteration_identity)
        ),
        _samples=tuple(samples),
    )


__all__ = [
    "DEEPEYES47K_PROMPT_GROUP_UID_SCHEMA",
    "DEEPEYES47K_RUNTIME_SCHEMA_VERSION",
    "DeepEyes47KRuntimeBinding",
    "DeepEyes47KRuntimeDataset",
    "DeepEyes47KRuntimeSample",
    "load_deepeyes47k_runtime",
]
