"""Immutable PRL13/teacher 3:1 policy schedule shared by every tool arm.

The materializer changes only the prompt population.  Crop, TGVF, and atomic
Crop+TGVF consume the same ordered ``samples.jsonl`` and select their renderer
later.  The schedule is deliberately explicit: every aligned group of sixteen
contains twelve PRL13 rows followed by four interleaved teacher rows, and every
256-row macro contains 90 VStar, 58 ArxivQA, 44 ThinkLite, and 64 teacher rows.

Runtime verification hashes the compact manifest and samples file.  Image
bytes are intentionally verified lazily by the veRL Dataset when a row is
first consumed; startup never re-hashes thousands of identical image files.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

from .deepeyes47k import DeepEyesTaskKind

POLICY_TEACHER_QUARTER_MIX_DATASET_KIND = "policy_t1_teacher_quarter_mix"
POLICY_TEACHER_QUARTER_MIX_SAMPLE_SCHEMA = "tgvf.policy-teacher-quarter-mix.sample.v1"
POLICY_TEACHER_QUARTER_MIX_MANIFEST_SCHEMA = (
    "tgvf.policy-teacher-quarter-mix.manifest.v1"
)
POLICY_TEACHER_QUARTER_MIX_RUNTIME_SCHEMA = "tgvf.policy-teacher-quarter-mix.runtime.v1"
POLICY_TEACHER_QUARTER_MIX_MANIFEST_FILE = "manifest.json"
POLICY_TEACHER_QUARTER_MIX_SAMPLES_FILE = "samples.jsonl"
POLICY_TEACHER_QUARTER_MIX_SELECTION_ALGORITHM = (
    "prl13-macro-source-prefix-plus-teacher-source-hash-v1"
)
POLICY_TEACHER_QUARTER_MIX_INTERLEAVE = "old-old-old-teacher-v1"

POLICY_TEACHER_QUARTER_MIX_SEED = 42
POLICY_TEACHER_QUARTER_MIX_STEPS = 80
POLICY_TEACHER_QUARTER_MIX_MACRO_SIZE = 256
POLICY_TEACHER_QUARTER_MIX_MICRO_SIZE = 16
POLICY_TEACHER_QUARTER_MIX_SAMPLE_COUNT = 20_480
POLICY_TEACHER_QUARTER_MIX_TEACHER_COUNT = 5_120
POLICY_TEACHER_QUARTER_MIX_OLD_COUNT = 15_360
POLICY_TEACHER_QUARTER_MIX_OLD_MACRO_COUNTS: Mapping[str, int] = MappingProxyType(
    {"vstar": 90, "arxivqa": 58, "thinklite": 44}
)
POLICY_TEACHER_QUARTER_MIX_MACRO_COUNTS: Mapping[str, int] = MappingProxyType(
    {"vstar": 90, "arxivqa": 58, "thinklite": 44, "teacher": 64}
)
POLICY_TEACHER_QUARTER_MIX_TEACHER_SOURCE_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "chartqa": 455,
        "docvqa": 864,
        "textocr": 816,
        "textvqa": 840,
        "visual_genome": 2_145,
    }
)

POLICY_TEACHER_QUARTER_MIX_DEFAULT_TEACHER_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/"
    "policy_selection/teacher/tgvf-v4-rp66-rp67-train-t1-retained-v1"
)
POLICY_TEACHER_QUARTER_MIX_TEACHER_MANIFEST_FILE_SHA256 = (
    "254e09db547047e3e0788a85e56dc458f2437d87c4a6452771131ad6719a276b"
)
POLICY_TEACHER_QUARTER_MIX_TEACHER_CONTENT_SHA256 = (
    "796f580440662a43b732a8fc33e6380a4ea0c8c240e9023e24954de558607192"
)
POLICY_TEACHER_QUARTER_MIX_TEACHER_SAMPLES_SHA256 = (
    "cf9b86cf11f1ca42c83eb94f1e3b418a193ce0cb1a5d84ec2c8bb5c2e6532880"
)
POLICY_TEACHER_QUARTER_MIX_DEFAULT_SCHEDULE_INDEX = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_rl/"
    "T1-04-INSTRUCT-FULL-MIXED-T1-RETAINED-FINAL-v2/"
    "prl13-stratified-schedule-index-v1.json"
)
POLICY_TEACHER_QUARTER_MIX_SCHEDULE_INDEX_FILE_SHA256 = (
    "52ef77788b822526a3c1becae82bc07cb06915d1a20aaa0b6a8c65360d89617e"
)
POLICY_TEACHER_QUARTER_MIX_SCHEDULE_INDEX_IDENTITY_SHA256 = (
    "bbd9b0025636d0a6e95800b42ff56ae1fbd214bf7055cc459832ab630646cbaa"
)

_TEACHER_PARENT_DATASET_KIND = "policy_teacher_t1_retained"
_TEACHER_PARENT_MANIFEST_SCHEMA = "tgvf.policy-teacher-t1-retained.manifest.v1"
_TEACHER_PARENT_SAMPLE_SCHEMA = "tgvf.policy-teacher-t1-retained.sample.v1"
_TEACHER_PARENT_SAMPLE_COUNT = 24_779
_TEACHER_PARENT_SOURCE_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "chartqa": 2_200,
        "docvqa": 4_182,
        "textocr": 3_951,
        "textvqa": 4_065,
        "visual_genome": 10_381,
    }
)
_BASE_PARENT_KIND = "deepeyes_prl13_schedule_index"
_HEX = frozenset("0123456789abcdef")


class PolicyTeacherQuarterMixMaterializationError(ValueError):
    """A parent or derived schedule violates the PRL22 data contract."""


class PolicyTeacherQuarterMixRuntimeValidationError(ValueError):
    """A materialized schedule differs from its immutable runtime binding."""


@dataclass(frozen=True, slots=True)
class PolicyTeacherQuarterMixRuntimeBinding:
    manifest_file_sha256: str
    content_sha256: str
    schedule_seed: int
    expected_sample_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_file_sha256, "manifest_file_sha256")
        _require_sha256(self.content_sha256, "content_sha256")
        if type(self.schedule_seed) is not int or self.schedule_seed < 0:
            raise ValueError("schedule_seed must be a non-negative integer")
        if (
            type(self.expected_sample_count) is not int
            or self.expected_sample_count <= 0
        ):
            raise ValueError("expected_sample_count must be positive")


@dataclass(frozen=True, slots=True)
class PolicyTeacherQuarterMixMaterializationResult:
    output_root: Path
    sample_count: int
    samples_sha256: str
    content_sha256: str
    manifest_file_sha256: str
    iteration_identity_sha256: str
    schedule_seed: int
    source_counts: Mapping[str, int]
    teacher_source_counts: Mapping[str, int]

    def as_record(self) -> dict[str, object]:
        return {
            "dataset_kind": POLICY_TEACHER_QUARTER_MIX_DATASET_KIND,
            "root": str(self.output_root),
            "decision_stage": "final",
            "sample_count": self.sample_count,
            "manifest_file_sha256": self.manifest_file_sha256,
            "content_sha256": self.content_sha256,
            "samples_sha256": self.samples_sha256,
            "iteration_identity_sha256": self.iteration_identity_sha256,
            "shuffle_seed": self.schedule_seed,
            "source_counts": dict(self.source_counts),
            "teacher_source_counts": dict(self.teacher_source_counts),
        }


@dataclass(frozen=True, slots=True)
class PolicyTeacherQuarterMixRuntimeSample:
    sample_id: str
    candidate_sha256: str
    image_path: Path
    image_sha256: str
    question: str
    ground_truth: str
    data_source: str
    source_dataset: str
    task_kind: DeepEyesTaskKind
    gt_regions: tuple[tuple[int, int, int, int], ...] | None
    tools_kwargs: Mapping[str, object]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            not self.sample_id
            or not self.question.strip()
            or not self.ground_truth.strip()
        ):
            raise ValueError("runtime sample text identity differs")
        _require_sha256(self.candidate_sha256, "candidate_sha256")
        _require_sha256(self.image_sha256, "image_sha256")
        if not self.image_path.is_absolute():
            raise ValueError("runtime sample image path must be absolute")
        if self.data_source not in POLICY_TEACHER_QUARTER_MIX_MACRO_COUNTS:
            raise ValueError("runtime sample source differs")
        if self.data_source == "teacher" and not self.source_dataset:
            raise ValueError("teacher runtime sample requires source_dataset")
        object.__setattr__(
            self, "tools_kwargs", MappingProxyType(dict(self.tools_kwargs))
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PolicyTeacherQuarterMixRuntimeDataset:
    root: Path
    binding: PolicyTeacherQuarterMixRuntimeBinding
    samples_sha256: str
    iteration_identity_sha256: str
    samples: tuple[PolicyTeacherQuarterMixRuntimeSample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PolicyTeacherQuarterMixRuntimeSample:
        return self.samples[index]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher-quarter artifact contains non-canonical JSON data"
        ) from error


def _canonical_json_line(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyTeacherQuarterMixMaterializationError(
                f"duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _parse_json(payload: bytes, *, field: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PolicyTeacherQuarterMixMaterializationError(
                    f"{field} contains non-finite JSON number: {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PolicyTeacherQuarterMixMaterializationError(
            f"{field} is not strict UTF-8 JSON"
        ) from error


def _safe_file(path: Path, *, field: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PolicyTeacherQuarterMixMaterializationError(
            f"{field} must be a regular non-symlink file"
        )
    return path.resolve(strict=True)


def _jsonl_records(path: Path) -> Iterator[tuple[int, dict[str, Any], str]]:
    path = _safe_file(path, field=str(path))
    with path.open("rb") as handle:
        for row_index, raw in enumerate(handle):
            if not raw.strip():
                raise PolicyTeacherQuarterMixMaterializationError(
                    f"{path}:{row_index + 1}: blank rows are forbidden"
                )
            value = _parse_json(raw, field=f"{path}:{row_index + 1}")
            if not isinstance(value, dict) or raw != _canonical_json_line(value):
                raise PolicyTeacherQuarterMixMaterializationError(
                    f"{path}:{row_index + 1}: row is not canonical JSON object"
                )
            yield row_index, value, _sha256_bytes(raw[:-1])


def _stable_teacher_key(record: Mapping[str, Any], seed: int) -> tuple[str, str]:
    source = str(record["extra_info"]["source_dataset"])
    sample_id = str(record["sample_id"])
    payload = (
        f"{POLICY_TEACHER_QUARTER_MIX_SELECTION_ALGORITHM}\0{seed}\0"
        f"{source}\0{sample_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), sample_id


def _smooth_labels(counts: Mapping[str, int]) -> tuple[str, ...]:
    """Integer smooth weighted round-robin with lexical tie breaking."""

    total = sum(counts.values())
    used = {name: 0 for name in counts}
    labels: list[str] = []
    for position in range(total):
        available = [name for name, count in counts.items() if used[name] < count]
        chosen = min(
            available,
            key=lambda name: (
                -((position + 1) * counts[name] - used[name] * total),
                name,
            ),
        )
        labels.append(chosen)
        used[chosen] += 1
    if Counter(labels) != Counter(counts):
        raise AssertionError("smooth interleave lost a source quota")
    return tuple(labels)


def policy_teacher_quarter_mix_iteration_identity_sha256(
    binding: PolicyTeacherQuarterMixRuntimeBinding, *, samples_sha256: str
) -> str:
    if not isinstance(binding, PolicyTeacherQuarterMixRuntimeBinding):
        raise TypeError("binding must be PolicyTeacherQuarterMixRuntimeBinding")
    _require_sha256(samples_sha256, "samples_sha256")
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": POLICY_TEACHER_QUARTER_MIX_RUNTIME_SCHEMA,
                "dataset_kind": POLICY_TEACHER_QUARTER_MIX_DATASET_KIND,
                "sample_count": binding.expected_sample_count,
                "schedule_seed": binding.schedule_seed,
                "selection_algorithm": POLICY_TEACHER_QUARTER_MIX_SELECTION_ALGORITHM,
                "interleave": POLICY_TEACHER_QUARTER_MIX_INTERLEAVE,
                "manifest_file_sha256": binding.manifest_file_sha256,
                "content_sha256": binding.content_sha256,
                "samples_sha256": samples_sha256,
            }
        )
    )


def _required_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyTeacherQuarterMixMaterializationError(
            f"{field} must be a JSON object"
        )
    return value


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyTeacherQuarterMixMaterializationError(
            f"{field} must be non-empty text"
        )
    return value


def _load_teacher_parent(
    root: Path,
) -> tuple[Mapping[str, Any], list[tuple[int, dict[str, Any], str]]]:
    if root.is_symlink() or not root.is_dir():
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher parent must be a regular non-symlink directory"
        )
    manifest_path = _safe_file(root / "manifest.json", field="teacher manifest")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != (
        POLICY_TEACHER_QUARTER_MIX_TEACHER_MANIFEST_FILE_SHA256
    ):
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher manifest file SHA-256 differs"
        )
    manifest = _required_mapping(
        _parse_json(manifest_bytes, field="teacher manifest"),
        field="teacher manifest",
    )
    if (
        manifest.get("schema_version") != _TEACHER_PARENT_MANIFEST_SCHEMA
        or manifest.get("dataset_kind") != _TEACHER_PARENT_DATASET_KIND
        or manifest.get("retained_count") != _TEACHER_PARENT_SAMPLE_COUNT
        or manifest.get("content_sha256")
        != POLICY_TEACHER_QUARTER_MIX_TEACHER_CONTENT_SHA256
        or manifest.get("source_dataset_counts") != dict(_TEACHER_PARENT_SOURCE_COUNTS)
    ):
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher parent manifest semantics differ"
        )
    samples_path = _safe_file(root / "samples.jsonl", field="teacher samples")
    if _sha256_file(samples_path) != POLICY_TEACHER_QUARTER_MIX_TEACHER_SAMPLES_SHA256:
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher parent samples SHA-256 differs"
        )
    rows = list(_jsonl_records(samples_path))
    if len(rows) != _TEACHER_PARENT_SAMPLE_COUNT:
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher parent sample count differs"
        )
    return manifest, rows


def _validated_teacher_row(
    row_index: int, record: Mapping[str, Any], row_sha256: str
) -> dict[str, object]:
    if (
        record.get("schema_version") != _TEACHER_PARENT_SAMPLE_SCHEMA
        or record.get("data_source") != "teacher"
    ):
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher sample schema/source differs"
        )
    sample_id = _required_text(record.get("sample_id"), field="teacher sample_id")
    candidate_sha256 = _require_sha256(
        record.get("candidate_sha256"), "teacher candidate_sha256"
    )
    extra = _required_mapping(record.get("extra_info"), field="teacher extra_info")
    source_dataset = _required_text(
        extra.get("source_dataset"), field="teacher source_dataset"
    )
    if source_dataset not in _TEACHER_PARENT_SOURCE_COUNTS:
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher source_dataset differs"
        )
    answer_format = extra.get("answer_format")
    if answer_format == "multiple_choice":
        task_kind = "mcq"
    elif answer_format == "open":
        task_kind = "open"
    else:
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher answer_format differs"
        )
    question = _required_text(extra.get("question"), field="teacher question")
    reward_model = _required_mapping(
        record.get("reward_model"), field="teacher reward_model"
    )
    ground_truth = _required_text(
        reward_model.get("ground_truth"), field="teacher ground_truth"
    )
    image = _required_mapping(record.get("image"), field="teacher image")
    image_path_text = _required_text(image.get("path"), field="teacher image.path")
    image_path = Path(image_path_text)
    if not image_path.is_absolute():
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher image path must be absolute"
        )
    _safe_file(image_path, field="selected teacher image")
    image_sha256 = _require_sha256(image.get("sha256"), "teacher image.sha256")
    width = image.get("width")
    height = image.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher image dimensions differ"
        )
    return {
        "sample_id": sample_id,
        "candidate_sha256": candidate_sha256,
        "data_source": "teacher",
        "source_dataset": source_dataset,
        "task_kind": task_kind,
        "question": question,
        "ground_truth": ground_truth,
        "image": {
            "path": str(image_path),
            "sha256": image_sha256,
            "width": width,
            "height": height,
        },
        "gt_regions": None,
        "mixture_role": "teacher",
        "parent": {
            "dataset_kind": _TEACHER_PARENT_DATASET_KIND,
            "row_index": row_index,
            "row_sha256": row_sha256,
        },
    }


def _old_record(sample: object, *, parent_row_index: int) -> dict[str, object]:
    gt_regions = getattr(sample, "gt_regions")
    core = {
        "population_index": getattr(sample, "index"),
        "sample_id": getattr(sample, "sample_id"),
        "candidate_sha256": getattr(sample, "candidate_sha256"),
        "data_source": getattr(sample, "data_source"),
        "task_kind": getattr(sample, "task_kind"),
        "question": getattr(sample, "question"),
        "ground_truth": getattr(sample, "ground_truth"),
        "image": {
            "path": str(getattr(sample, "image_path")),
            "sha256": getattr(sample, "image_sha256"),
            "width": getattr(sample, "image_width"),
            "height": getattr(sample, "image_height"),
        },
        "gt_regions": (
            [list(region) for region in gt_regions] if gt_regions is not None else None
        ),
    }
    _safe_file(Path(core["image"]["path"]), field="selected PRL13 image")  # type: ignore[index]
    return {
        "sample_id": core["sample_id"],
        "candidate_sha256": core["candidate_sha256"],
        "data_source": core["data_source"],
        "source_dataset": core["data_source"],
        "task_kind": core["task_kind"],
        "question": core["question"],
        "ground_truth": core["ground_truth"],
        "image": core["image"],
        "gt_regions": core["gt_regions"],
        "mixture_role": "base",
        "parent": {
            "dataset_kind": _BASE_PARENT_KIND,
            "row_index": parent_row_index,
            "row_sha256": _sha256_bytes(_canonical_json_bytes(core)),
        },
    }


def _materialized_record(
    record: Mapping[str, object], *, schedule_index: int
) -> dict[str, object]:
    return {
        "schema_version": POLICY_TEACHER_QUARTER_MIX_SAMPLE_SCHEMA,
        "schedule_index": schedule_index,
        **record,
    }


def _build_schedule_records(
    *, teacher_root: Path, schedule_index_path: Path, schedule_seed: int
) -> tuple[list[dict[str, object]], Mapping[str, object]]:
    # Import lazily: the historical schedule module imports the policy package,
    # whose run-config module in turn imports this dataset binding.
    from .deepeyes_official_schedule_index import load_deepeyes_schedule_index

    if schedule_seed != POLICY_TEACHER_QUARTER_MIX_SEED:
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher-quarter schedule currently binds seed 42"
        )
    old_index = load_deepeyes_schedule_index(schedule_index_path)
    if len(old_index.train) != POLICY_TEACHER_QUARTER_MIX_SAMPLE_COUNT:
        raise PolicyTeacherQuarterMixMaterializationError(
            "PRL13 train schedule length differs"
        )
    _, teacher_rows = _load_teacher_parent(teacher_root)
    grouped: dict[str, list[tuple[int, dict[str, Any], str]]] = defaultdict(list)
    seen_teacher_ids: set[str] = set()
    for row_index, record, row_sha256 in teacher_rows:
        sample_id = _required_text(record.get("sample_id"), field="teacher sample_id")
        if sample_id in seen_teacher_ids:
            raise PolicyTeacherQuarterMixMaterializationError(
                "teacher parent contains duplicate sample_id"
            )
        seen_teacher_ids.add(sample_id)
        extra = _required_mapping(record.get("extra_info"), field="teacher extra_info")
        source_dataset = _required_text(
            extra.get("source_dataset"), field="teacher source_dataset"
        )
        grouped[source_dataset].append((row_index, record, row_sha256))
    if {source: len(rows) for source, rows in grouped.items()} != dict(
        _TEACHER_PARENT_SOURCE_COUNTS
    ):
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher parent source counts differ"
        )
    selected_by_source: dict[str, deque[dict[str, object]]] = {}
    for source, quota in POLICY_TEACHER_QUARTER_MIX_TEACHER_SOURCE_COUNTS.items():
        ranked = sorted(
            grouped[source],
            key=lambda item: _stable_teacher_key(item[1], schedule_seed),
        )
        selected_by_source[source] = deque(
            _validated_teacher_row(*item) for item in ranked[:quota]
        )
    teacher_order = [
        selected_by_source[source].popleft()
        for source in _smooth_labels(POLICY_TEACHER_QUARTER_MIX_TEACHER_SOURCE_COUNTS)
    ]
    if any(selected_by_source.values()):
        raise AssertionError("teacher selection did not consume its exact quotas")

    output: list[dict[str, object]] = []
    selected_old_ids: set[str] = set()
    selected_teacher_ids: set[str] = set()
    for macro_index in range(POLICY_TEACHER_QUARTER_MIX_STEPS):
        begin = macro_index * POLICY_TEACHER_QUARTER_MIX_MACRO_SIZE
        macro = old_index.train[begin : begin + POLICY_TEACHER_QUARTER_MIX_MACRO_SIZE]
        remaining = dict(POLICY_TEACHER_QUARTER_MIX_OLD_MACRO_COUNTS)
        old_records: list[dict[str, object]] = []
        for offset, sample in enumerate(macro):
            source = sample.data_source
            if remaining.get(source, 0) <= 0:
                continue
            old_records.append(_old_record(sample, parent_row_index=begin + offset))
            remaining[source] -= 1
        if remaining != {source: 0 for source in remaining} or len(old_records) != 192:
            raise PolicyTeacherQuarterMixMaterializationError(
                f"PRL13 macro {macro_index} cannot satisfy the 75% source quotas"
            )
        teacher_macro = teacher_order[macro_index * 64 : (macro_index + 1) * 64]
        for micro_index in range(16):
            old_start = micro_index * 12
            teacher_start = micro_index * 4
            micro = (
                old_records[old_start : old_start + 12]
                + teacher_macro[teacher_start : teacher_start + 4]
            )
            # Preserve an exact 3:1 cadence inside the BS16 group as well as
            # its aggregate 12:4 count.
            ordered: list[dict[str, object]] = []
            for group_index in range(4):
                ordered.extend(micro[group_index * 3 : group_index * 3 + 3])
                ordered.append(micro[12 + group_index])
            for record in ordered:
                sample_id = str(record["sample_id"])
                if record["mixture_role"] == "teacher":
                    selected_teacher_ids.add(sample_id)
                else:
                    selected_old_ids.add(sample_id)
                output.append(_materialized_record(record, schedule_index=len(output)))
    if (
        len(output) != POLICY_TEACHER_QUARTER_MIX_SAMPLE_COUNT
        or len(selected_old_ids) != POLICY_TEACHER_QUARTER_MIX_OLD_COUNT
        or len(selected_teacher_ids) != POLICY_TEACHER_QUARTER_MIX_TEACHER_COUNT
        or selected_old_ids.intersection(selected_teacher_ids)
    ):
        raise PolicyTeacherQuarterMixMaterializationError(
            "teacher-quarter selection uniqueness/count differs"
        )
    selected_overlap_images = len(
        {
            str(record["image"]["sha256"])
            for record in output
            if record["mixture_role"] == "base"
        }
        & {
            str(record["image"]["sha256"])
            for record in output
            if record["mixture_role"] == "teacher"
        }
    )
    audit = {
        "selected_exact_image_overlap": selected_overlap_images,
        "parent_population_exact_image_overlap": 587,
        "parent_population_teacher_rows_on_overlapping_images": 1_623,
        "parent_population_exact_image_question_overlap": 0,
    }
    return output, audit


def materialize_policy_teacher_quarter_mix(
    output_root: str | Path,
    *,
    teacher_root: str | Path = POLICY_TEACHER_QUARTER_MIX_DEFAULT_TEACHER_ROOT,
    schedule_index_path: str | Path = POLICY_TEACHER_QUARTER_MIX_DEFAULT_SCHEDULE_INDEX,
    schedule_seed: int = POLICY_TEACHER_QUARTER_MIX_SEED,
) -> PolicyTeacherQuarterMixMaterializationResult:
    """Create the immutable PRL22 schedule without changing either parent."""

    output = Path(output_root)
    if not output.is_absolute():
        raise PolicyTeacherQuarterMixMaterializationError(
            "output_root must be absolute"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"teacher-quarter output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records, overlap_audit = _build_schedule_records(
        teacher_root=Path(teacher_root),
        schedule_index_path=Path(schedule_index_path),
        schedule_seed=schedule_seed,
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    try:
        samples_path = temporary / POLICY_TEACHER_QUARTER_MIX_SAMPLES_FILE
        with samples_path.open("xb") as handle:
            for record in records:
                handle.write(_canonical_json_line(record))
            handle.flush()
            os.fsync(handle.fileno())
        samples_sha256 = _sha256_file(samples_path)
        source_counts = Counter(str(record["data_source"]) for record in records)
        teacher_source_counts = Counter(
            str(record["source_dataset"])
            for record in records
            if record["data_source"] == "teacher"
        )
        content: dict[str, object] = {
            "schema_version": POLICY_TEACHER_QUARTER_MIX_MANIFEST_SCHEMA,
            "dataset_kind": POLICY_TEACHER_QUARTER_MIX_DATASET_KIND,
            "decision_stage": "final",
            "sample_count": len(records),
            "parents": {
                "prl13_schedule_index": {
                    "path": str(Path(schedule_index_path)),
                    "file_sha256": (
                        POLICY_TEACHER_QUARTER_MIX_SCHEDULE_INDEX_FILE_SHA256
                    ),
                    "identity_sha256": (
                        POLICY_TEACHER_QUARTER_MIX_SCHEDULE_INDEX_IDENTITY_SHA256
                    ),
                    "train_rows": POLICY_TEACHER_QUARTER_MIX_SAMPLE_COUNT,
                },
                "teacher_t1_retained": {
                    "root": str(Path(teacher_root)),
                    "manifest_file_sha256": (
                        POLICY_TEACHER_QUARTER_MIX_TEACHER_MANIFEST_FILE_SHA256
                    ),
                    "content_sha256": (
                        POLICY_TEACHER_QUARTER_MIX_TEACHER_CONTENT_SHA256
                    ),
                    "samples_sha256": (
                        POLICY_TEACHER_QUARTER_MIX_TEACHER_SAMPLES_SHA256
                    ),
                    "rows": _TEACHER_PARENT_SAMPLE_COUNT,
                },
            },
            "schedule": {
                "selection_algorithm": (POLICY_TEACHER_QUARTER_MIX_SELECTION_ALGORITHM),
                "interleave": POLICY_TEACHER_QUARTER_MIX_INTERLEAVE,
                "seed": schedule_seed,
                "steps": POLICY_TEACHER_QUARTER_MIX_STEPS,
                "macro_size": POLICY_TEACHER_QUARTER_MIX_MACRO_SIZE,
                "micro_size": POLICY_TEACHER_QUARTER_MIX_MICRO_SIZE,
                "macro_source_counts": dict(POLICY_TEACHER_QUARTER_MIX_MACRO_COUNTS),
                "teacher_per_micro": 4,
                "without_replacement": True,
                "dataloader_shuffle": False,
            },
            "source_counts": dict(sorted(source_counts.items())),
            "teacher_source_counts": dict(sorted(teacher_source_counts.items())),
            "samples": {
                "path": POLICY_TEACHER_QUARTER_MIX_SAMPLES_FILE,
                "rows": len(records),
                "sha256": samples_sha256,
            },
            "images": {
                "address": "absolute-path-plus-sha256",
                "bytes_verified": "lazy-on-first-access",
            },
            "overlap_audit": overlap_audit,
        }
        content_sha256 = _sha256_bytes(_canonical_json_bytes(content))
        manifest = {**content, "content_sha256": content_sha256}
        manifest_bytes = _canonical_json_line(manifest)
        manifest_path = temporary / POLICY_TEACHER_QUARTER_MIX_MANIFEST_FILE
        with manifest_path.open("xb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_file_sha256 = _sha256_bytes(manifest_bytes)
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    binding = PolicyTeacherQuarterMixRuntimeBinding(
        manifest_file_sha256=manifest_file_sha256,
        content_sha256=content_sha256,
        schedule_seed=schedule_seed,
        expected_sample_count=len(records),
    )
    return PolicyTeacherQuarterMixMaterializationResult(
        output_root=output,
        sample_count=len(records),
        samples_sha256=samples_sha256,
        content_sha256=content_sha256,
        manifest_file_sha256=manifest_file_sha256,
        iteration_identity_sha256=(
            policy_teacher_quarter_mix_iteration_identity_sha256(
                binding, samples_sha256=samples_sha256
            )
        ),
        schedule_seed=schedule_seed,
        source_counts=MappingProxyType(dict(source_counts)),
        teacher_source_counts=MappingProxyType(dict(teacher_source_counts)),
    )


def _load_bound_manifest(
    root: Path, binding: PolicyTeacherQuarterMixRuntimeBinding
) -> tuple[Mapping[str, Any], Path, str]:
    if root.is_symlink() or not root.is_dir():
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "teacher-quarter root must be a regular non-symlink directory"
        )
    manifest_path = root / POLICY_TEACHER_QUARTER_MIX_MANIFEST_FILE
    try:
        manifest_bytes = _safe_file(
            manifest_path, field="mixture manifest"
        ).read_bytes()
    except PolicyTeacherQuarterMixMaterializationError as error:
        raise PolicyTeacherQuarterMixRuntimeValidationError(str(error)) from error
    if _sha256_bytes(manifest_bytes) != binding.manifest_file_sha256:
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture manifest file SHA-256 differs"
        )
    try:
        manifest = _required_mapping(
            _parse_json(manifest_bytes, field="mixture manifest"),
            field="mixture manifest",
        )
    except PolicyTeacherQuarterMixMaterializationError as error:
        raise PolicyTeacherQuarterMixRuntimeValidationError(str(error)) from error
    if manifest_bytes != _canonical_json_line(manifest):
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture manifest is not canonical JSON"
        )
    content = dict(manifest)
    observed_content_sha256 = content.pop("content_sha256", None)
    if (
        observed_content_sha256 != binding.content_sha256
        or _sha256_bytes(_canonical_json_bytes(content)) != binding.content_sha256
        or manifest.get("schema_version") != POLICY_TEACHER_QUARTER_MIX_MANIFEST_SCHEMA
        or manifest.get("dataset_kind") != POLICY_TEACHER_QUARTER_MIX_DATASET_KIND
        or manifest.get("sample_count") != binding.expected_sample_count
    ):
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture manifest content binding differs"
        )
    schedule = manifest.get("schedule")
    if (
        not isinstance(schedule, Mapping)
        or schedule.get("seed") != binding.schedule_seed
        or schedule.get("macro_source_counts")
        != dict(POLICY_TEACHER_QUARTER_MIX_MACRO_COUNTS)
        or schedule.get("teacher_per_micro") != 4
    ):
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture schedule contract differs"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, Mapping):
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture samples descriptor differs"
        )
    samples_sha256 = _require_sha256(samples.get("sha256"), "samples.sha256")
    samples_path = root / POLICY_TEACHER_QUARTER_MIX_SAMPLES_FILE
    if (
        samples.get("path") != POLICY_TEACHER_QUARTER_MIX_SAMPLES_FILE
        or samples.get("rows") != binding.expected_sample_count
        or _sha256_file(_safe_file(samples_path, field="mixture samples"))
        != samples_sha256
    ):
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture samples file binding differs"
        )
    return manifest, samples_path, samples_sha256


def verify_policy_teacher_quarter_mix_artifact_binding(
    root: str | Path,
    *,
    binding: PolicyTeacherQuarterMixRuntimeBinding,
    samples_sha256: str,
) -> None:
    """Verify compact artifact identities without eagerly hashing images."""

    if not isinstance(binding, PolicyTeacherQuarterMixRuntimeBinding):
        raise TypeError("binding must be PolicyTeacherQuarterMixRuntimeBinding")
    expected_samples = _require_sha256(samples_sha256, "samples_sha256")
    _, _, observed_samples = _load_bound_manifest(Path(root), binding)
    if observed_samples != expected_samples:
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "configured mixture samples SHA-256 differs"
        )


def _runtime_sample(
    record: Mapping[str, Any],
    *,
    expected_index: int,
    expected_schema: str = POLICY_TEACHER_QUARTER_MIX_SAMPLE_SCHEMA,
):
    # Keep the data contract importable while the policy package initializes.
    from tgvf_rl.policy.deepeyes_official_protocol import tools_kwargs_for_source

    expected_fields = {
        "schema_version",
        "schedule_index",
        "sample_id",
        "candidate_sha256",
        "data_source",
        "source_dataset",
        "task_kind",
        "question",
        "ground_truth",
        "image",
        "gt_regions",
        "mixture_role",
        "parent",
    }
    if set(record) != expected_fields:
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture sample row schema differs"
        )
    if (
        record.get("schema_version") != expected_schema
        or record.get("schedule_index") != expected_index
    ):
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture sample schedule index differs"
        )
    data_source = _required_text(record.get("data_source"), field="data_source")
    if data_source not in POLICY_TEACHER_QUARTER_MIX_MACRO_COUNTS:
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture sample source differs"
        )
    mixture_role = record.get("mixture_role")
    if mixture_role != ("teacher" if data_source == "teacher" else "base"):
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture sample parent role differs"
        )
    task_kind_text = _required_text(record.get("task_kind"), field="task_kind")
    try:
        task_kind = DeepEyesTaskKind(task_kind_text)
    except ValueError as error:
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture task kind differs"
        ) from error
    if data_source == "teacher" and task_kind_text not in {"mcq", "open"}:
        raise PolicyTeacherQuarterMixRuntimeValidationError("teacher task kind differs")
    image = _required_mapping(record.get("image"), field="image")
    if set(image) != {"path", "sha256", "width", "height"}:
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture image schema differs"
        )
    image_path = Path(_required_text(image.get("path"), field="image.path"))
    if not image_path.is_absolute():
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture image path must be absolute"
        )
    gt_regions_value = record.get("gt_regions")
    gt_regions: tuple[tuple[int, int, int, int], ...] | None
    if gt_regions_value is None:
        gt_regions = None
    elif isinstance(gt_regions_value, list) and gt_regions_value:
        parsed: list[tuple[int, int, int, int]] = []
        for region in gt_regions_value:
            if (
                not isinstance(region, list)
                or len(region) != 4
                or any(type(value) is not int for value in region)
            ):
                raise PolicyTeacherQuarterMixRuntimeValidationError(
                    "mixture gt_regions differ"
                )
            parsed.append(tuple(region))  # type: ignore[arg-type]
        gt_regions = tuple(parsed)
    else:
        raise PolicyTeacherQuarterMixRuntimeValidationError("mixture gt_regions differ")
    try:
        tools_kwargs = tools_kwargs_for_source(data_source, gt_regions)
    except (TypeError, ValueError) as error:
        raise PolicyTeacherQuarterMixRuntimeValidationError(str(error)) from error
    parent = _required_mapping(record.get("parent"), field="parent")
    if set(parent) != {"dataset_kind", "row_index", "row_sha256"}:
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture parent schema differs"
        )
    metadata = {
        "candidate_sha256": _require_sha256(
            record.get("candidate_sha256"), "candidate_sha256"
        ),
        "gt_regions": gt_regions,
        "tools_kwargs": tools_kwargs,
        "mixture_role": mixture_role,
        "parent": dict(parent),
        "schedule_index": expected_index,
    }
    return PolicyTeacherQuarterMixRuntimeSample(
        sample_id=_required_text(record.get("sample_id"), field="sample_id"),
        candidate_sha256=str(metadata["candidate_sha256"]),
        image_path=image_path,
        image_sha256=_require_sha256(image.get("sha256"), "image.sha256"),
        question=_required_text(record.get("question"), field="question"),
        ground_truth=_required_text(record.get("ground_truth"), field="ground_truth"),
        data_source=data_source,
        source_dataset=_required_text(
            record.get("source_dataset"), field="source_dataset"
        ),
        task_kind=task_kind,
        gt_regions=gt_regions,
        tools_kwargs=tools_kwargs,
        metadata=metadata,
    )


def load_policy_teacher_quarter_mix_runtime(
    root: str | Path,
    *,
    binding: PolicyTeacherQuarterMixRuntimeBinding,
) -> PolicyTeacherQuarterMixRuntimeDataset:
    """Load and validate the ordered schedule; image bytes remain lazy."""

    manifest, samples_path, samples_sha256 = _load_bound_manifest(Path(root), binding)
    del manifest
    samples: list[PolicyTeacherQuarterMixRuntimeSample] = []
    try:
        for row_index, record, _ in _jsonl_records(samples_path):
            samples.append(_runtime_sample(record, expected_index=row_index))
    except PolicyTeacherQuarterMixMaterializationError as error:
        raise PolicyTeacherQuarterMixRuntimeValidationError(str(error)) from error
    if len(samples) != binding.expected_sample_count:
        raise PolicyTeacherQuarterMixRuntimeValidationError(
            "mixture runtime sample count differs"
        )
    for begin in range(0, len(samples), POLICY_TEACHER_QUARTER_MIX_MICRO_SIZE):
        micro = samples[begin : begin + POLICY_TEACHER_QUARTER_MIX_MICRO_SIZE]
        if Counter(sample.data_source == "teacher" for sample in micro) != Counter(
            {False: 12, True: 4}
        ):
            raise PolicyTeacherQuarterMixRuntimeValidationError(
                "mixture BS16 role counts differ"
            )
        for group in range(4):
            roles = [
                sample.data_source == "teacher"
                for sample in micro[group * 4 : group * 4 + 4]
            ]
            if roles != [False, False, False, True]:
                raise PolicyTeacherQuarterMixRuntimeValidationError(
                    "mixture 3:1 cadence differs"
                )
    for begin in range(0, len(samples), POLICY_TEACHER_QUARTER_MIX_MACRO_SIZE):
        counts = Counter(
            sample.data_source
            for sample in samples[begin : begin + POLICY_TEACHER_QUARTER_MIX_MACRO_SIZE]
        )
        if counts != Counter(POLICY_TEACHER_QUARTER_MIX_MACRO_COUNTS):
            raise PolicyTeacherQuarterMixRuntimeValidationError(
                "mixture macro source counts differ"
            )
    iteration_identity = policy_teacher_quarter_mix_iteration_identity_sha256(
        binding, samples_sha256=samples_sha256
    )
    return PolicyTeacherQuarterMixRuntimeDataset(
        root=Path(root),
        binding=binding,
        samples_sha256=samples_sha256,
        iteration_identity_sha256=iteration_identity,
        samples=tuple(samples),
    )


__all__ = [
    "POLICY_TEACHER_QUARTER_MIX_DATASET_KIND",
    "POLICY_TEACHER_QUARTER_MIX_DEFAULT_TEACHER_ROOT",
    "POLICY_TEACHER_QUARTER_MIX_DEFAULT_SCHEDULE_INDEX",
    "POLICY_TEACHER_QUARTER_MIX_MANIFEST_FILE",
    "POLICY_TEACHER_QUARTER_MIX_SAMPLES_FILE",
    "PolicyTeacherQuarterMixMaterializationError",
    "PolicyTeacherQuarterMixMaterializationResult",
    "PolicyTeacherQuarterMixRuntimeBinding",
    "PolicyTeacherQuarterMixRuntimeDataset",
    "PolicyTeacherQuarterMixRuntimeSample",
    "PolicyTeacherQuarterMixRuntimeValidationError",
    "load_policy_teacher_quarter_mix_runtime",
    "materialize_policy_teacher_quarter_mix",
    "policy_teacher_quarter_mix_iteration_identity_sha256",
    "verify_policy_teacher_quarter_mix_artifact_binding",
]
