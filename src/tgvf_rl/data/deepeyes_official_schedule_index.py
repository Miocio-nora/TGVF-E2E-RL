"""Immutable, compact runtime index for the PRL13 stratified schedule.

The canonical T1 artifact is deliberately expensive to validate: reconstructing
the PRL13 schedule scans every candidate and retained row and hashes every
retained image.  That work belongs in a one-time materialization gate, not in
every trainer process.  This module stores only the rows that PRL13 can consume
(formal train, held-out probe, and isolated smoke), cryptographically binds the
index to all parent artifacts, and validates the complete schedule semantics at
startup.

Image bytes remain fail-closed without returning to eager all-pool hashing.  A
consumer verifies the exact SHA-256 of a selected image on first access; the
dataset owns that lazy cache because it knows which rows were actually read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tgvf_rl.data.deepeyes_official_schedule import (
    DEEPEYES_BATCH_COUNTS,
    DEEPEYES_CANDIDATE_ROWS,
    DEEPEYES_CANDIDATE_SHA256,
    DEEPEYES_CANDIDATE_SIDECAR,
    DEEPEYES_PROBE_NAME,
    DEEPEYES_PROBE_SEED,
    DEEPEYES_PROMPTS_PER_STEP,
    DEEPEYES_T1_CONTENT_SHA256,
    DEEPEYES_T1_MANIFEST_FILE_SHA256,
    DEEPEYES_T1_ROOT,
    DEEPEYES_T1_SAMPLE_COUNT,
    DEEPEYES_T1_SAMPLES_SHA256,
    DEEPEYES_T1_SCHEDULE_SCHEMA,
    DEEPEYES_T1_SHUFFLE_SEED,
    DEEPEYES_TOTAL_STEPS,
    DEEPEYES_TRAIN_SEED,
    DeepEyesOfficialSample,
    DeepEyesSchedule,
    assert_verl_route_contract,
    build_deepeyes_schedule,
)


DEEPEYES_SCHEDULE_INDEX_SCHEMA = "tgvf.deepeyes-native-schedule-index.v1"
DEEPEYES_SCHEDULE_INDEX_PATH = (
    DEEPEYES_T1_ROOT / "prl13-stratified-schedule-index-v1.json"
)

# These two values are replaced only after the one-time materializer validates
# every canonical parent row and image.  Keeping them in source makes a changed
# or locally regenerated index fail closed instead of silently becoming data.
DEEPEYES_SCHEDULE_INDEX_FILE_SHA256 = (
    "52ef77788b822526a3c1becae82bc07cb06915d1a20aaa0b6a8c65360d89617e"
)
DEEPEYES_SCHEDULE_INDEX_IDENTITY_SHA256 = (
    "bbd9b0025636d0a6e95800b42ff56ae1fbd214bf7055cc459832ab630646cbaa"
)

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "parent_artifacts",
    "schedule",
    "splits",
    "identity_sha256",
}
_SCHEDULE_FIELDS = {
    "mode",
    "seed",
    "probe_seed",
    "identity_sha256",
    "probe_manifest",
    "train_batch_sha256",
}
_SPLIT_FIELDS = {"row_count", "rows"}
_ROW_FIELDS = {
    "population_index",
    "sample_id",
    "candidate_sha256",
    "data_source",
    "task_kind",
    "question",
    "ground_truth",
    "image",
    "gt_regions",
}
_IMAGE_FIELDS = {"path", "sha256", "width", "height"}
_HEX = frozenset("0123456789abcdef")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def schedule_index_parent_artifacts() -> Mapping[str, object]:
    """Return the exact immutable parents and derivation contract."""

    return MappingProxyType(
        {
            "t1_final": {
                "root": str(DEEPEYES_T1_ROOT),
                "sample_count": DEEPEYES_T1_SAMPLE_COUNT,
                "manifest_file_sha256": DEEPEYES_T1_MANIFEST_FILE_SHA256,
                "content_sha256": DEEPEYES_T1_CONTENT_SHA256,
                "samples_sha256": DEEPEYES_T1_SAMPLES_SHA256,
                "shuffle_seed": DEEPEYES_T1_SHUFFLE_SEED,
            },
            "candidate_sidecar": {
                "path": str(DEEPEYES_CANDIDATE_SIDECAR),
                "rows": DEEPEYES_CANDIDATE_ROWS,
                "sha256": DEEPEYES_CANDIDATE_SHA256,
            },
            "schedule_derivation": {
                "schema_version": DEEPEYES_T1_SCHEDULE_SCHEMA,
                "mode": "stratified",
                "seed": DEEPEYES_TRAIN_SEED,
                "probe_name": DEEPEYES_PROBE_NAME,
                "probe_seed": DEEPEYES_PROBE_SEED,
                "prompts_per_step": DEEPEYES_PROMPTS_PER_STEP,
                "total_steps": DEEPEYES_TOTAL_STEPS,
                "batch_source_counts": dict(DEEPEYES_BATCH_COUNTS),
                "without_replacement": True,
            },
        }
    )


def _smoke_samples(schedule: DeepEyesSchedule) -> tuple[DeepEyesOfficialSample, ...]:
    excluded = set(schedule.probe_indices)
    excluded.update(index for batch in schedule.batches for index in batch)
    available: dict[str, list[DeepEyesOfficialSample]] = {
        "vstar": [],
        "arxivqa": [],
        "thinklite": [],
    }
    for index, sample in enumerate(schedule.samples):
        if index not in excluded:
            available[sample.data_source].append(sample)
    for values in available.values():
        values.sort(key=lambda sample: sample.sample_id)
    if len(available["vstar"]) < 2 or not all(available.values()):
        raise ValueError("official pool is too small for the PRL13 smoke split")
    return (
        available["vstar"][0],
        available["arxivqa"][0],
        available["thinklite"][0],
        available["vstar"][1],
    )


def _sample_record(sample: DeepEyesOfficialSample) -> dict[str, object]:
    return {
        "population_index": sample.index,
        "sample_id": sample.sample_id,
        "candidate_sha256": sample.candidate_sha256,
        "data_source": sample.data_source,
        "task_kind": sample.task_kind,
        "question": sample.question,
        "ground_truth": sample.ground_truth,
        "image": {
            "path": str(sample.image_path),
            "sha256": sample.image_sha256,
            "width": sample.image_width,
            "height": sample.image_height,
        },
        "gt_regions": (
            [list(region) for region in sample.gt_regions]
            if sample.gt_regions is not None
            else None
        ),
    }


def build_deepeyes_schedule_index_payload(
    samples: Sequence[DeepEyesOfficialSample],
) -> dict[str, object]:
    """Build the canonical index after callers validate the complete parents."""

    schedule = build_deepeyes_schedule(
        samples,
        mode="stratified",
        seed=DEEPEYES_TRAIN_SEED,
        probe_seed=DEEPEYES_PROBE_SEED,
    )
    train = tuple(
        schedule.samples[index] for batch in schedule.batches for index in batch
    )
    probe = schedule.probe
    smoke = _smoke_samples(schedule)
    content: dict[str, object] = {
        "schema_version": DEEPEYES_SCHEDULE_INDEX_SCHEMA,
        "parent_artifacts": dict(schedule_index_parent_artifacts()),
        "schedule": {
            "mode": schedule.mode,
            "seed": schedule.seed,
            "probe_seed": schedule.probe_seed,
            "identity_sha256": schedule.identity_sha256,
            "probe_manifest": dict(schedule.probe_manifest),
            "train_batch_sha256": [
                _sha256_json(
                    [schedule.samples[index].sample_id for index in batch]
                )
                for batch in schedule.batches
            ],
        },
        "splits": {
            "train": {
                "row_count": len(train),
                "rows": [_sample_record(sample) for sample in train],
            },
            "probe": {
                "row_count": len(probe),
                "rows": [_sample_record(sample) for sample in probe],
            },
            "smoke": {
                "row_count": len(smoke),
                "rows": [_sample_record(sample) for sample in smoke],
            },
        },
    }
    return {**content, "identity_sha256": _sha256_json(content)}


@dataclass(frozen=True, slots=True)
class DeepEyesScheduleIndex:
    path: Path
    file_sha256: str
    identity_sha256: str
    schedule_identity_sha256: str
    probe_manifest: Mapping[str, object]
    train: tuple[DeepEyesOfficialSample, ...]
    probe: tuple[DeepEyesOfficialSample, ...]
    smoke: tuple[DeepEyesOfficialSample, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_manifest", MappingProxyType(dict(self.probe_manifest)))


def _parse_sample(record: object) -> DeepEyesOfficialSample:
    if not isinstance(record, Mapping) or set(record) != _ROW_FIELDS:
        raise ValueError("schedule-index row schema differs")
    image = record.get("image")
    if not isinstance(image, Mapping) or set(image) != _IMAGE_FIELDS:
        raise ValueError("schedule-index image schema differs")
    image_path_value = image.get("path")
    if not isinstance(image_path_value, str):
        raise ValueError("schedule-index image path differs")
    image_path = Path(image_path_value)
    if not image_path.is_absolute():
        raise ValueError("schedule-index image path must be absolute")
    regions_value = record.get("gt_regions")
    regions: tuple[tuple[int, int, int, int], ...] | None
    if regions_value is None:
        regions = None
    else:
        if not isinstance(regions_value, list) or not regions_value:
            raise ValueError("schedule-index gt_regions differs")
        parsed_regions: list[tuple[int, int, int, int]] = []
        for region in regions_value:
            if (
                not isinstance(region, list)
                or len(region) != 4
                or any(type(coordinate) is not int for coordinate in region)
            ):
                raise ValueError("schedule-index gt_regions differs")
            parsed_regions.append(tuple(region))  # type: ignore[arg-type]
        regions = tuple(parsed_regions)
    population_index = record.get("population_index")
    width = image.get("width")
    height = image.get("height")
    if type(population_index) is not int or type(width) is not int or type(height) is not int:
        raise ValueError("schedule-index integer field differs")
    string_fields = {
        name: record.get(name)
        for name in (
            "sample_id",
            "candidate_sha256",
            "data_source",
            "task_kind",
            "question",
            "ground_truth",
        )
    }
    if any(not isinstance(value, str) for value in string_fields.values()):
        raise ValueError("schedule-index string field differs")
    sample = DeepEyesOfficialSample(
        index=population_index,
        sample_id=str(string_fields["sample_id"]),
        candidate_sha256=_require_sha256(
            string_fields["candidate_sha256"], "candidate_sha256"
        ),
        data_source=str(string_fields["data_source"]),
        task_kind=str(string_fields["task_kind"]),
        question=str(string_fields["question"]),
        ground_truth=str(string_fields["ground_truth"]),
        image_path=image_path,
        image_sha256=_require_sha256(image.get("sha256"), "image.sha256"),
        image_width=width,
        image_height=height,
        gt_regions=regions,
    )
    assert_verl_route_contract(sample)
    return sample


def _parse_split(
    splits: Mapping[str, Any], name: str, expected_count: int
) -> tuple[DeepEyesOfficialSample, ...]:
    value = splits.get(name)
    if not isinstance(value, Mapping) or set(value) != _SPLIT_FIELDS:
        raise ValueError(f"schedule-index {name} split schema differs")
    rows = value.get("rows")
    if (
        value.get("row_count") != expected_count
        or not isinstance(rows, list)
        or len(rows) != expected_count
    ):
        raise ValueError(f"schedule-index {name} row count differs")
    return tuple(_parse_sample(row) for row in rows)


def _probe_manifest(samples: Sequence[DeepEyesOfficialSample]) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": "tgvf.deepeyes-native-t1-probe.v1",
        "name": DEEPEYES_PROBE_NAME,
        "seed": DEEPEYES_PROBE_SEED,
        "sample_count": len(samples),
        "source_counts": dict(DEEPEYES_BATCH_COUNTS),
        "ordered_sample_ids": [sample.sample_id for sample in samples],
    }
    return {**record, "manifest_sha256": _sha256_json(record)}


def _validate_schedule_semantics(
    *,
    schedule: Mapping[str, Any],
    train: Sequence[DeepEyesOfficialSample],
    probe: Sequence[DeepEyesOfficialSample],
    smoke: Sequence[DeepEyesOfficialSample],
) -> tuple[str, Mapping[str, object]]:
    if set(schedule) != _SCHEDULE_FIELDS:
        raise ValueError("schedule-index schedule schema differs")
    if (
        schedule.get("mode") != "stratified"
        or schedule.get("seed") != DEEPEYES_TRAIN_SEED
        or schedule.get("probe_seed") != DEEPEYES_PROBE_SEED
    ):
        raise ValueError("schedule-index derivation differs")
    all_samples = (*train, *probe, *smoke)
    sample_ids = [sample.sample_id for sample in all_samples]
    population_indices = [sample.index for sample in all_samples]
    if len(sample_ids) != len(set(sample_ids)) or len(population_indices) != len(
        set(population_indices)
    ):
        raise ValueError("schedule-index splits overlap or repeat samples")
    batch_hashes: list[str] = []
    for start in range(0, len(train), DEEPEYES_PROMPTS_PER_STEP):
        batch = train[start : start + DEEPEYES_PROMPTS_PER_STEP]
        if Counter(sample.data_source for sample in batch) != Counter(
            DEEPEYES_BATCH_COUNTS
        ):
            raise ValueError("schedule-index train batch is not exactly 120/77/59")
        batch_hashes.append(_sha256_json([sample.sample_id for sample in batch]))
    if schedule.get("train_batch_sha256") != batch_hashes:
        raise ValueError("schedule-index batch identities differ")
    if Counter(sample.data_source for sample in probe) != Counter(DEEPEYES_BATCH_COUNTS):
        raise ValueError("schedule-index probe is not exactly 120/77/59")
    if [sample.data_source for sample in smoke] != [
        "vstar",
        "arxivqa",
        "thinklite",
        "vstar",
    ]:
        raise ValueError("schedule-index smoke source coverage differs")
    probe_manifest = _probe_manifest(probe)
    if schedule.get("probe_manifest") != probe_manifest:
        raise ValueError("schedule-index probe identity differs")
    schedule_identity = _sha256_json(
        {
            "schema_version": DEEPEYES_T1_SCHEDULE_SCHEMA,
            "mode": "stratified",
            "seed": DEEPEYES_TRAIN_SEED,
            "probe_manifest_sha256": probe_manifest["manifest_sha256"],
            "batch_sha256": batch_hashes,
        }
    )
    if schedule.get("identity_sha256") != schedule_identity:
        raise ValueError("schedule-index schedule identity differs")
    return schedule_identity, MappingProxyType(probe_manifest)


def load_deepeyes_schedule_index(
    path: str | Path = DEEPEYES_SCHEDULE_INDEX_PATH,
    *,
    expected_file_sha256: str = DEEPEYES_SCHEDULE_INDEX_FILE_SHA256,
    expected_identity_sha256: str = DEEPEYES_SCHEDULE_INDEX_IDENTITY_SHA256,
) -> DeepEyesScheduleIndex:
    """Load and fully validate the compact index without touching parent rows."""

    index_path = Path(path)
    expected_file = _require_sha256(expected_file_sha256, "index file SHA-256")
    expected_identity = _require_sha256(
        expected_identity_sha256, "index identity SHA-256"
    )
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("schedule index must be a regular non-symlink file")
    payload = index_path.read_bytes()
    if _sha256_bytes(payload) != expected_file:
        raise ValueError("schedule index file SHA-256 differs")
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("schedule index must be strict UTF-8 JSON") from error
    if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("schedule index top-level schema differs")
    if payload != _canonical_json_bytes(value) + b"\n":
        raise ValueError("schedule index is not canonical JSON")
    identity = _require_sha256(value.get("identity_sha256"), "index identity")
    content = {key: nested for key, nested in value.items() if key != "identity_sha256"}
    if identity != _sha256_json(content) or identity != expected_identity:
        raise ValueError("schedule index identity differs")
    if value.get("schema_version") != DEEPEYES_SCHEDULE_INDEX_SCHEMA:
        raise ValueError("schedule index schema version differs")
    if value.get("parent_artifacts") != dict(schedule_index_parent_artifacts()):
        raise ValueError("schedule index parent-artifact binding differs")
    splits = value.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "probe", "smoke"}:
        raise ValueError("schedule index split set differs")
    train = _parse_split(
        splits, "train", DEEPEYES_TOTAL_STEPS * DEEPEYES_PROMPTS_PER_STEP
    )
    probe = _parse_split(splits, "probe", DEEPEYES_PROMPTS_PER_STEP)
    smoke = _parse_split(splits, "smoke", 4)
    schedule = value.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("schedule index schedule differs")
    schedule_identity, probe_manifest = _validate_schedule_semantics(
        schedule=schedule, train=train, probe=probe, smoke=smoke
    )
    return DeepEyesScheduleIndex(
        path=index_path,
        file_sha256=expected_file,
        identity_sha256=identity,
        schedule_identity_sha256=schedule_identity,
        probe_manifest=probe_manifest,
        train=train,
        probe=probe,
        smoke=smoke,
    )


def write_deepeyes_schedule_index(
    samples: Sequence[DeepEyesOfficialSample], output: str | Path
) -> tuple[str, str, int]:
    """Write one canonical index with create-only semantics.

    Returns ``(file_sha256, identity_sha256, byte_count)``.
    """

    output_path = Path(output)
    payload = build_deepeyes_schedule_index_payload(samples)
    encoded = _canonical_json_bytes(payload) + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
    return _sha256_bytes(encoded), str(payload["identity_sha256"]), len(encoded)


__all__ = [
    "DEEPEYES_SCHEDULE_INDEX_FILE_SHA256",
    "DEEPEYES_SCHEDULE_INDEX_IDENTITY_SHA256",
    "DEEPEYES_SCHEDULE_INDEX_PATH",
    "DEEPEYES_SCHEDULE_INDEX_SCHEMA",
    "DeepEyesScheduleIndex",
    "build_deepeyes_schedule_index_payload",
    "load_deepeyes_schedule_index",
    "schedule_index_parent_artifacts",
    "write_deepeyes_schedule_index",
]
