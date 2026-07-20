"""Pinned DeepEyes-47K source verification and prompt-free materialization.

The official parquet files are treated as immutable inputs.  This module never
reads the source ``prompt`` field: project-native prompts are rendered later by
the policy runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any


DEEPEYES47K_DATASET_ID = "ChenShawn/DeepEyes-Datasets-47k"
DEEPEYES47K_SNAPSHOT = "5546681e28fa2eda9f60a9ea9dd0cf291216ded3"
DEEPEYES47K_SCHEMA_VERSION = "tgvf.deepeyes47k.materialized.v1"
DEEPEYES47K_TOTAL_ROWS = 47_052
DEEPEYES47K_SAMPLES_FILE = "samples.jsonl"
DEEPEYES47K_MANIFEST_FILE = "manifest.json"
DEEPEYES47K_SHUFFLE_ALGORITHM = "sha256-sort-v1"


class DeepEyesTaskKind(str, Enum):
    MATH = "math"
    MCQ = "mcq"
    OPEN = "open"


class DeepEyesSourceValidationError(ValueError):
    """A source or materialized artifact differs from its declared identity."""


class DeepEyesDependencyError(RuntimeError):
    """An optional dependency required for real parquet I/O is unavailable."""


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class DeepEyesSourceFileSpec:
    filename: str
    rows: int
    lfs_sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        if Path(self.filename).name != self.filename or not self.filename.endswith(
            ".parquet"
        ):
            raise ValueError("source filename must be a basename ending in .parquet")
        if type(self.rows) is not int or self.rows <= 0:
            raise ValueError("source rows must be a positive integer")
        if type(self.byte_size) is not int or self.byte_size <= 0:
            raise ValueError("source byte_size must be a positive integer")
        _validate_sha256(self.lfs_sha256, field_name="source lfs_sha256")

    def as_record(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "rows": self.rows,
            "lfs_sha256": self.lfs_sha256,
            "byte_size": self.byte_size,
        }


DEEPEYES47K_SOURCE_FILES = (
    DeepEyesSourceFileSpec(
        filename="data_0.1.2_visual_toolbox_v2.parquet",
        rows=22_362,
        lfs_sha256="42992bf5de25e8d766f820fb9730ece275563ba80dd41e3377bf678c9ba2c2c1",
        byte_size=990_263_397,
    ),
    DeepEyesSourceFileSpec(
        filename="data_thinklite_reasoning_acc.parquet",
        rows=11_031,
        lfs_sha256="660cea5ff8f74d19f993b575f30b6f5406b6c330dd8f9aacc6be59e299238967",
        byte_size=1_656_152_904,
    ),
    DeepEyesSourceFileSpec(
        filename="data_v0.8_visual_toolbox_v2.parquet",
        rows=13_659,
        lfs_sha256="96fc256e6f73e098c1b586f1c37baad616ecbddf1105bfca71aa07a5dda7da5a",
        byte_size=2_198_504_506,
    ),
)

if sum(spec.rows for spec in DEEPEYES47K_SOURCE_FILES) != DEEPEYES47K_TOTAL_ROWS:
    raise RuntimeError("the pinned DeepEyes-47K row contract is internally invalid")


@dataclass(frozen=True, slots=True)
class VerifiedDeepEyesSource:
    filename: str
    rows: int
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class DeepEyesSourceVerification:
    snapshot: str
    files: tuple[VerifiedDeepEyesSource, ...]
    total_rows: int

    def as_record(self) -> dict[str, object]:
        return {
            "dataset_id": DEEPEYES47K_DATASET_ID,
            "snapshot": self.snapshot,
            "total_rows": self.total_rows,
            "files": [
                {
                    "filename": item.filename,
                    "rows": item.rows,
                    "sha256": item.sha256,
                    "byte_size": item.byte_size,
                }
                for item in self.files
            ],
        }


@dataclass(frozen=True, slots=True)
class SanitizedDeepEyesSample:
    sample_id: str
    image_bytes: bytes
    image_sha256: str
    question: str
    ground_truth: Any
    data_source: str
    task_kind: DeepEyesTaskKind
    source_file: str
    source_file_sha256: str
    source_row_index: int
    snapshot: str
    ability: str | None = None
    style: str | None = None
    split: str | None = None

    def manifest_record(self, *, image_path: str) -> dict[str, object]:
        record: dict[str, object] = {
            "sample_id": self.sample_id,
            "image": {
                "path": image_path,
                "sha256": self.image_sha256,
            },
            "extra_info": {"question": self.question},
            "reward_model": {"ground_truth": self.ground_truth},
            "data_source": self.data_source,
            "task_kind": self.task_kind.value,
            "provenance": {
                "dataset_id": DEEPEYES47K_DATASET_ID,
                "snapshot": self.snapshot,
                "source_file": self.source_file,
                "source_file_sha256": self.source_file_sha256,
                "source_row_index": self.source_row_index,
            },
        }
        for field_name in ("ability", "style", "split"):
            value = getattr(self, field_name)
            if value is not None:
                record[field_name] = value
        return record


@dataclass(frozen=True, slots=True)
class DeepEyesMaterializationResult:
    output_root: Path
    sample_count: int
    samples_sha256: str
    content_sha256: str
    manifest_file_sha256: str
    shuffle_seed: int
    fixture: bool

    def as_record(self) -> dict[str, object]:
        return {
            "output_root": str(self.output_root),
            "sample_count": self.sample_count,
            "samples_sha256": self.samples_sha256,
            "content_sha256": self.content_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "shuffle_seed": self.shuffle_seed,
            "fixture": self.fixture,
        }


RowCountReader = Callable[[Path], int]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON data") from exc
    return encoded.encode("utf-8")


def _normalized_json_value(value: Any, *, field_name: str) -> Any:
    try:
        return json.loads(_canonical_json_bytes(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain finite JSON data") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_deepeyes47k_sample_id(
    *, source_file: str, source_row_index: int, snapshot: str = DEEPEYES47K_SNAPSHOT
) -> str:
    if Path(source_file).name != source_file or not source_file:
        raise ValueError("source_file must be a non-empty basename")
    if type(source_row_index) is not int or source_row_index < 0:
        raise ValueError("source_row_index must be a non-negative integer")
    if not isinstance(snapshot, str) or not snapshot:
        raise ValueError("snapshot must be a non-empty string")
    identity = {
        "dataset_id": DEEPEYES47K_DATASET_ID,
        "snapshot": snapshot,
        "source_file": source_file,
        "source_row_index": source_row_index,
    }
    return f"deepeyes47k:{_sha256_bytes(_canonical_json_bytes(identity))}"


def _required_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_metadata(field_name: str, *containers: Mapping[str, Any]) -> str | None:
    candidates = [
        container[field_name]
        for container in containers
        if container.get(field_name) is not None
    ]
    if candidates and any(value != candidates[0] for value in candidates[1:]):
        raise ValueError(f"conflicting {field_name} metadata")
    value = candidates[0] if candidates else None
    if value is None:
        return None
    return _required_string(value, field_name=field_name)


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, Mapping):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    elif isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes) or not value:
        raise ValueError("image must contain non-empty embedded bytes")
    return value


def _single_source_image_bytes(value: Any) -> bytes:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError("images must contain exactly one source image")
    return _image_bytes(value[0])


_OPTION_PATTERN = re.compile(r"(?im)(?:^|\n)\s*(?:\(([A-H])\)|([A-H])[.):.])\s+\S")
_LETTER_ANSWER = re.compile(r"^\(?[A-H]\)?(?:[.):])?$", re.IGNORECASE)
_MATH_METADATA_MARKERS = (
    "algebra",
    "arithmetic",
    "calculus",
    "geometry",
    "math",
    "olympiad",
    "proof",
    "theorem",
)
_MCQ_METADATA_MARKERS = ("multiple_choice", "multiple-choice", "mcq")


def classify_deepeyes_task_kind(
    *,
    question: str,
    ground_truth: Any,
    data_source: str,
    ability: str | None = None,
    style: str | None = None,
) -> DeepEyesTaskKind:
    """Classify the verifier route using a deterministic, versioned heuristic."""

    metadata = " ".join(
        value.casefold() for value in (data_source, ability, style) if value is not None
    )
    option_labels = {
        next(label for label in match.groups() if label is not None)
        for match in _OPTION_PATTERN.finditer(question)
    }
    answer = ground_truth.strip() if isinstance(ground_truth, str) else None
    if any(marker in metadata for marker in _MCQ_METADATA_MARKERS) or (
        answer is not None
        and _LETTER_ANSWER.fullmatch(answer) is not None
        and len(option_labels) >= 2
    ):
        return DeepEyesTaskKind.MCQ
    if any(marker in metadata for marker in _MATH_METADATA_MARKERS):
        return DeepEyesTaskKind.MATH
    if isinstance(ground_truth, (int, float)) and not isinstance(ground_truth, bool):
        if math.isfinite(float(ground_truth)):
            return DeepEyesTaskKind.MATH
    if re.search(r"\\(?:frac|sqrt|begin\{|boxed\{|angle\b)", question):
        return DeepEyesTaskKind.MATH
    return DeepEyesTaskKind.OPEN


def sanitize_deepeyes47k_row(
    row: Mapping[str, Any],
    *,
    source_spec: DeepEyesSourceFileSpec,
    source_row_index: int,
    snapshot: str = DEEPEYES47K_SNAPSHOT,
) -> SanitizedDeepEyesSample:
    """Extract only the accepted fields; the source ``prompt`` is never read."""

    if not isinstance(row, Mapping):
        raise TypeError("source row must be a mapping")
    if (
        type(source_row_index) is not int
        or not 0 <= source_row_index < source_spec.rows
    ):
        raise ValueError("source_row_index is outside the declared source file")
    extra_info = _required_mapping(row.get("extra_info"), field_name="extra_info")
    reward_model = _required_mapping(row.get("reward_model"), field_name="reward_model")
    question = _required_string(
        extra_info.get("question"), field_name="extra_info.question"
    )
    if "ground_truth" not in reward_model:
        raise ValueError("reward_model.ground_truth is required")
    ground_truth = _normalized_json_value(
        reward_model["ground_truth"], field_name="reward_model.ground_truth"
    )
    if ground_truth is None or (
        isinstance(ground_truth, str) and not ground_truth.strip()
    ):
        raise ValueError("reward_model.ground_truth must be non-empty")
    data_source = _required_string(row.get("data_source"), field_name="data_source")
    image_bytes = _single_source_image_bytes(row.get("images"))
    image_sha256 = _sha256_bytes(image_bytes)
    ability = _optional_metadata("ability", row, extra_info)
    style = _optional_metadata("style", row, reward_model, extra_info)
    split = _optional_metadata("split", row, extra_info)
    return SanitizedDeepEyesSample(
        sample_id=stable_deepeyes47k_sample_id(
            source_file=source_spec.filename,
            source_row_index=source_row_index,
            snapshot=snapshot,
        ),
        image_bytes=image_bytes,
        image_sha256=image_sha256,
        question=question,
        ground_truth=ground_truth,
        data_source=data_source,
        task_kind=classify_deepeyes_task_kind(
            question=question,
            ground_truth=ground_truth,
            data_source=data_source,
            ability=ability,
            style=style,
        ),
        source_file=source_spec.filename,
        source_file_sha256=source_spec.lfs_sha256,
        source_row_index=source_row_index,
        snapshot=snapshot,
        ability=ability,
        style=style,
        split=split,
    )


def _load_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DeepEyesDependencyError(
            "pyarrow is required for DeepEyes parquet verification/materialization"
        ) from exc
    return parquet


def _parquet_row_count(path: Path) -> int:
    parquet = _load_pyarrow_parquet()
    return int(parquet.ParquetFile(path).metadata.num_rows)


def _parquet_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    parquet = _load_pyarrow_parquet()
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            if not isinstance(row, Mapping):
                raise DeepEyesSourceValidationError(
                    f"{path.name} produced a non-mapping parquet row"
                )
            yield row


def verify_deepeyes47k_source_files(
    source_root: Path,
    *,
    source_specs: Sequence[DeepEyesSourceFileSpec] = DEEPEYES47K_SOURCE_FILES,
    row_count_reader: RowCountReader | None = None,
    snapshot: str = DEEPEYES47K_SNAPSHOT,
) -> DeepEyesSourceVerification:
    """Verify byte size, LFS SHA-256, and parquet row count for every source."""

    source_root = Path(source_root)
    specs = tuple(source_specs)
    if not specs or len({spec.filename for spec in specs}) != len(specs):
        raise ValueError("source_specs must be non-empty with unique filenames")
    count_rows = row_count_reader or _parquet_row_count
    verified: list[VerifiedDeepEyesSource] = []
    for spec in specs:
        path = source_root / spec.filename
        if not path.is_file():
            raise DeepEyesSourceValidationError(f"missing source file: {path}")
        actual_size = path.stat().st_size
        if actual_size != spec.byte_size:
            raise DeepEyesSourceValidationError(
                f"{spec.filename} byte size mismatch: {actual_size} != {spec.byte_size}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != spec.lfs_sha256:
            raise DeepEyesSourceValidationError(
                f"{spec.filename} SHA-256 mismatch: {actual_sha256} != {spec.lfs_sha256}"
            )
        actual_rows = count_rows(path)
        if type(actual_rows) is not int or actual_rows != spec.rows:
            raise DeepEyesSourceValidationError(
                f"{spec.filename} row count mismatch: {actual_rows} != {spec.rows}"
            )
        verified.append(
            VerifiedDeepEyesSource(
                filename=spec.filename,
                rows=actual_rows,
                sha256=actual_sha256,
                byte_size=actual_size,
            )
        )
    total_rows = sum(item.rows for item in verified)
    if total_rows != sum(spec.rows for spec in specs):
        raise DeepEyesSourceValidationError("verified source total row count mismatch")
    return DeepEyesSourceVerification(
        snapshot=snapshot,
        files=tuple(verified),
        total_rows=total_rows,
    )


def _validate_shuffle_seed(shuffle_seed: int) -> None:
    if type(shuffle_seed) is not int:
        raise TypeError("shuffle_seed must be an explicitly supplied integer")


def _shuffle_key(sample_id: str, shuffle_seed: int) -> tuple[str, str]:
    payload = (f"{DEEPEYES47K_SHUFFLE_ALGORITHM}\0{shuffle_seed}\0{sample_id}").encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest(), sample_id


def _image_extension(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    if payload.startswith(b"BM"):
        return ".bmp"
    if payload.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"
    return ".bin"


def _manifest_descriptor(
    *,
    source_specs: Sequence[DeepEyesSourceFileSpec],
    snapshot: str,
    sample_count: int,
    samples_sha256: str,
    shuffle_seed: int,
    fixture: bool,
) -> dict[str, object]:
    return {
        "schema_version": DEEPEYES47K_SCHEMA_VERSION,
        "dataset_id": DEEPEYES47K_DATASET_ID,
        "snapshot": snapshot,
        "fixture": fixture,
        "source_files": [spec.as_record() for spec in source_specs],
        "source_total_rows": sum(spec.rows for spec in source_specs),
        "sample_count": sample_count,
        "shuffle": {
            "algorithm": DEEPEYES47K_SHUFFLE_ALGORITHM,
            "seed": shuffle_seed,
        },
        "samples": {
            "path": DEEPEYES47K_SAMPLES_FILE,
            "rows": sample_count,
            "sha256": samples_sha256,
        },
        "images": {
            "directory": "images",
            "address": "sha256-of-original-bytes",
        },
    }


def _materialize_rows(
    *,
    rows_by_file: Mapping[str, Iterable[Mapping[str, Any]]],
    source_specs: Sequence[DeepEyesSourceFileSpec],
    output_root: Path,
    shuffle_seed: int,
    snapshot: str,
    fixture: bool,
) -> DeepEyesMaterializationResult:
    _validate_shuffle_seed(shuffle_seed)
    specs = tuple(source_specs)
    if not specs or len({spec.filename for spec in specs}) != len(specs):
        raise ValueError("source_specs must be non-empty with unique filenames")
    if set(rows_by_file) != {spec.filename for spec in specs}:
        raise DeepEyesSourceValidationError(
            "rows_by_file must contain exactly the declared source filenames"
        )
    output_root = Path(output_root)
    if os.path.lexists(output_root):
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".deepeyes47k-", dir=output_root.parent)
    )
    try:
        image_root = temporary_root / "images"
        image_root.mkdir()
        records: list[dict[str, object]] = []
        for spec in specs:
            row_count = 0
            for row_index, row in enumerate(rows_by_file[spec.filename]):
                sample = sanitize_deepeyes47k_row(
                    row,
                    source_spec=spec,
                    source_row_index=row_index,
                    snapshot=snapshot,
                )
                extension = _image_extension(sample.image_bytes)
                image_name = f"{sample.image_sha256}{extension}"
                image_path = image_root / image_name
                if image_path.exists():
                    if _sha256_file(image_path) != sample.image_sha256:
                        raise DeepEyesSourceValidationError(
                            f"content-address collision for image {sample.image_sha256}"
                        )
                else:
                    image_path.write_bytes(sample.image_bytes)
                records.append(
                    sample.manifest_record(image_path=f"images/{image_name}")
                )
                row_count += 1
            if row_count != spec.rows:
                raise DeepEyesSourceValidationError(
                    f"{spec.filename} yielded {row_count} rows, expected {spec.rows}"
                )

        records.sort(
            key=lambda record: _shuffle_key(str(record["sample_id"]), shuffle_seed)
        )
        samples_digest = hashlib.sha256()
        samples_path = temporary_root / DEEPEYES47K_SAMPLES_FILE
        with samples_path.open("wb") as handle:
            for record in records:
                line = _canonical_json_bytes(record) + b"\n"
                handle.write(line)
                samples_digest.update(line)
        samples_sha256 = samples_digest.hexdigest()
        descriptor = _manifest_descriptor(
            source_specs=specs,
            snapshot=snapshot,
            sample_count=len(records),
            samples_sha256=samples_sha256,
            shuffle_seed=shuffle_seed,
            fixture=fixture,
        )
        content_sha256 = _sha256_bytes(_canonical_json_bytes(descriptor))
        manifest = {**descriptor, "content_sha256": content_sha256}
        manifest_bytes = _canonical_json_bytes(manifest) + b"\n"
        (temporary_root / DEEPEYES47K_MANIFEST_FILE).write_bytes(manifest_bytes)
        manifest_file_sha256 = _sha256_bytes(manifest_bytes)
        temporary_root.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return DeepEyesMaterializationResult(
        output_root=output_root,
        sample_count=len(records),
        samples_sha256=samples_sha256,
        content_sha256=content_sha256,
        manifest_file_sha256=manifest_file_sha256,
        shuffle_seed=shuffle_seed,
        fixture=fixture,
    )


def materialize_deepeyes47k(
    source_root: Path,
    output_root: Path,
    *,
    shuffle_seed: int,
) -> DeepEyesMaterializationResult:
    """Verify and materialize the complete pinned snapshot using pyarrow."""

    _validate_shuffle_seed(shuffle_seed)
    source_root = Path(source_root)
    verification = verify_deepeyes47k_source_files(source_root)
    if verification.total_rows != DEEPEYES47K_TOTAL_ROWS:
        raise DeepEyesSourceValidationError(
            f"pinned source must contain {DEEPEYES47K_TOTAL_ROWS} rows"
        )
    return _materialize_rows(
        rows_by_file={
            spec.filename: _parquet_rows(source_root / spec.filename)
            for spec in DEEPEYES47K_SOURCE_FILES
        },
        source_specs=DEEPEYES47K_SOURCE_FILES,
        output_root=output_root,
        shuffle_seed=shuffle_seed,
        snapshot=DEEPEYES47K_SNAPSHOT,
        fixture=False,
    )


def materialize_deepeyes47k_fixture(
    rows_by_file: Mapping[str, Iterable[Mapping[str, Any]]],
    source_specs: Sequence[DeepEyesSourceFileSpec],
    output_root: Path,
    *,
    shuffle_seed: int,
) -> DeepEyesMaterializationResult:
    """Run the exact sanitizer/writer on explicit small in-memory fixtures."""

    return _materialize_rows(
        rows_by_file=rows_by_file,
        source_specs=source_specs,
        output_root=output_root,
        shuffle_seed=shuffle_seed,
        snapshot="fixture",
        fixture=True,
    )


def validate_materialized_deepeyes47k(output_root: Path) -> dict[str, object]:
    """Validate manifest, row bytes, stable identities, and image addresses."""

    output_root = Path(output_root)
    manifest_path = output_root / DEEPEYES47K_MANIFEST_FILE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DeepEyesSourceValidationError(
            "materialized manifest is missing or invalid"
        ) from exc
    if not isinstance(manifest, dict):
        raise DeepEyesSourceValidationError("materialized manifest must be an object")
    content_sha256 = manifest.pop("content_sha256", None)
    if not isinstance(content_sha256, str) or content_sha256 != _sha256_bytes(
        _canonical_json_bytes(manifest)
    ):
        raise DeepEyesSourceValidationError(
            "materialized manifest content hash mismatch"
        )
    if manifest.get("schema_version") != DEEPEYES47K_SCHEMA_VERSION:
        raise DeepEyesSourceValidationError("materialized schema version mismatch")
    if manifest.get("dataset_id") != DEEPEYES47K_DATASET_ID:
        raise DeepEyesSourceValidationError("materialized dataset identity mismatch")
    fixture = manifest.get("fixture")
    if type(fixture) is not bool:
        raise DeepEyesSourceValidationError("materialized fixture marker is invalid")
    raw_source_specs = manifest.get("source_files")
    if not isinstance(raw_source_specs, list):
        raise DeepEyesSourceValidationError("materialized source list is invalid")
    try:
        source_specs = tuple(
            DeepEyesSourceFileSpec(
                filename=item["filename"],
                rows=item["rows"],
                lfs_sha256=item["lfs_sha256"],
                byte_size=item["byte_size"],
            )
            for item in raw_source_specs
            if isinstance(item, dict)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DeepEyesSourceValidationError(
            "materialized source identity is invalid"
        ) from exc
    if (
        len(source_specs) != len(raw_source_specs)
        or not source_specs
        or len({spec.filename for spec in source_specs}) != len(source_specs)
    ):
        raise DeepEyesSourceValidationError("materialized source identity is invalid")
    if manifest.get("source_total_rows") != sum(spec.rows for spec in source_specs):
        raise DeepEyesSourceValidationError("materialized source total is invalid")
    if not fixture and (
        manifest.get("snapshot") != DEEPEYES47K_SNAPSHOT
        or source_specs != DEEPEYES47K_SOURCE_FILES
        or manifest.get("source_total_rows") != DEEPEYES47K_TOTAL_ROWS
    ):
        raise DeepEyesSourceValidationError("production source pin mismatch")
    shuffle = manifest.get("shuffle")
    if (
        not isinstance(shuffle, dict)
        or shuffle.get("algorithm") != DEEPEYES47K_SHUFFLE_ALGORITHM
        or type(shuffle.get("seed")) is not int
    ):
        raise DeepEyesSourceValidationError("materialized shuffle identity is invalid")
    samples = manifest.get("samples")
    if not isinstance(samples, dict) or samples.get("path") != DEEPEYES47K_SAMPLES_FILE:
        raise DeepEyesSourceValidationError(
            "materialized samples descriptor is invalid"
        )
    samples_path = output_root / DEEPEYES47K_SAMPLES_FILE
    if _sha256_file(samples_path) != samples.get("sha256"):
        raise DeepEyesSourceValidationError("materialized samples SHA-256 mismatch")

    sample_count = 0
    sample_ids: set[str] = set()
    ordered_sample_ids: list[str] = []
    specs_by_name = {spec.filename: spec for spec in source_specs}
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DeepEyesSourceValidationError(
                    "invalid materialized JSONL row"
                ) from exc
            allowed_keys = {
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
            required_keys = allowed_keys.difference({"ability", "style", "split"})
            if (
                not isinstance(record, dict)
                or not required_keys.issubset(record)
                or not set(record).issubset(allowed_keys)
            ):
                raise DeepEyesSourceValidationError(
                    "materialized row has a forbidden schema"
                )
            sample_id = record.get("sample_id")
            image = record.get("image")
            provenance = record.get("provenance")
            extra_info = record.get("extra_info")
            reward_model = record.get("reward_model")
            if (
                not isinstance(sample_id, str)
                or not isinstance(image, dict)
                or set(image) != {"path", "sha256"}
                or not isinstance(provenance, dict)
                or set(provenance)
                != {
                    "dataset_id",
                    "snapshot",
                    "source_file",
                    "source_file_sha256",
                    "source_row_index",
                }
                or not isinstance(extra_info, dict)
                or set(extra_info) != {"question"}
                or not isinstance(reward_model, dict)
                or set(reward_model) != {"ground_truth"}
            ):
                raise DeepEyesSourceValidationError(
                    "materialized row identity is incomplete"
                )
            source_file = provenance.get("source_file")
            source_spec = (
                specs_by_name.get(source_file) if isinstance(source_file, str) else None
            )
            source_row_index = provenance.get("source_row_index")
            if (
                source_spec is None
                or provenance.get("dataset_id") != DEEPEYES47K_DATASET_ID
                or provenance.get("snapshot") != manifest.get("snapshot")
                or provenance.get("source_file_sha256") != source_spec.lfs_sha256
                or type(source_row_index) is not int
                or not 0 <= source_row_index < source_spec.rows
            ):
                raise DeepEyesSourceValidationError("materialized provenance mismatch")
            expected_sample_id = stable_deepeyes47k_sample_id(
                source_file=source_file,
                source_row_index=source_row_index,
                snapshot=provenance["snapshot"],
            )
            if sample_id != expected_sample_id or sample_id in sample_ids:
                raise DeepEyesSourceValidationError(
                    "sample identity mismatch or duplicate"
                )
            sample_ids.add(sample_id)
            ordered_sample_ids.append(sample_id)
            try:
                ability = record.get("ability")
                style = record.get("style")
                if ability is not None:
                    ability = _required_string(ability, field_name="ability")
                if style is not None:
                    style = _required_string(style, field_name="style")
                ground_truth = _normalized_json_value(
                    reward_model.get("ground_truth"),
                    field_name="reward_model.ground_truth",
                )
                expected_task_kind = classify_deepeyes_task_kind(
                    question=_required_string(
                        extra_info.get("question"), field_name="extra_info.question"
                    ),
                    ground_truth=ground_truth,
                    data_source=_required_string(
                        record.get("data_source"), field_name="data_source"
                    ),
                    ability=ability,
                    style=style,
                )
            except ValueError as exc:
                raise DeepEyesSourceValidationError(
                    "materialized task fields are invalid"
                ) from exc
            if record.get("task_kind") != expected_task_kind.value:
                raise DeepEyesSourceValidationError("materialized task kind mismatch")
            relative_image = image.get("path")
            image_sha256 = image.get("sha256")
            if not isinstance(relative_image, str) or not isinstance(image_sha256, str):
                raise DeepEyesSourceValidationError(
                    "materialized image identity is incomplete"
                )
            pure_path = PurePosixPath(relative_image)
            if (
                pure_path.is_absolute()
                or pure_path.parts[:1] != ("images",)
                or ".." in pure_path.parts
            ):
                raise DeepEyesSourceValidationError(
                    "materialized image path escapes images/"
                )
            image_path = output_root.joinpath(*pure_path.parts)
            if not image_path.is_file() or _sha256_file(image_path) != image_sha256:
                raise DeepEyesSourceValidationError(
                    "materialized image SHA-256 mismatch"
                )
            sample_count += 1
    if sample_count != samples.get("rows") or sample_count != manifest.get(
        "sample_count"
    ):
        raise DeepEyesSourceValidationError("materialized sample count mismatch")
    expected_order = sorted(
        ordered_sample_ids,
        key=lambda sample_id: _shuffle_key(sample_id, shuffle["seed"]),
    )
    if ordered_sample_ids != expected_order:
        raise DeepEyesSourceValidationError("materialized shuffle order mismatch")
    return {
        "sample_count": sample_count,
        "samples_sha256": samples["sha256"],
        "content_sha256": content_sha256,
        "fixture": fixture,
    }
