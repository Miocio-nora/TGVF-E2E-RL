"""Deterministic Teacher25/50/100 policy schedules for the PRL23 ablation.

Teacher25 is the already-published PRL22 artifact contract.  The public
materializer delegates that profile to the legacy implementation so the
manifest, samples file, and all derived identities stay byte-for-byte stable.
Teacher50 and Teacher100 use the generic v2 manifest below.

All profiles retain the 20,480-prompt horizon, seed 42, no replacement, and
an exact role ratio in every aligned BS16 group.  Source quotas use Hamilton
apportionment.  The teacher selector intentionally retains PRL22's hash
namespace, making the selected populations nested by source:

    Teacher25 subset Teacher50 subset Teacher100.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

from . import policy_teacher_quarter_mix as _legacy


POLICY_TEACHER_RATIO_MIX_DATASET_KIND = "policy_t1_teacher_ratio_mix"
POLICY_TEACHER_RATIO_MIX_SAMPLE_SCHEMA = "tgvf.policy-teacher-ratio-mix.sample.v2"
POLICY_TEACHER_RATIO_MIX_MANIFEST_SCHEMA = (
    "tgvf.policy-teacher-ratio-mix.manifest.v2"
)
POLICY_TEACHER_RATIO_MIX_RUNTIME_SCHEMA = (
    "tgvf.policy-teacher-ratio-mix.runtime.v2"
)
POLICY_TEACHER_RATIO_MIX_MANIFEST_FILE = "manifest.json"
POLICY_TEACHER_RATIO_MIX_SAMPLES_FILE = "samples.jsonl"
POLICY_TEACHER_RATIO_MIX_SELECTION_ALGORITHM = (
    "prl13-source-prefix-plus-nested-teacher-source-hash-ratio-v2"
)
# Do not revise this selector namespace: it is the namespace used by PRL22.
# Keeping it fixed is what makes the teacher populations genuinely nested.
POLICY_TEACHER_RATIO_MIX_SELECTOR_HASH_NAMESPACE = (
    _legacy.POLICY_TEACHER_QUARTER_MIX_SELECTION_ALGORITHM
)

POLICY_TEACHER_RATIO_MIX_SEED = 42
POLICY_TEACHER_RATIO_MIX_STEPS = 80
POLICY_TEACHER_RATIO_MIX_MACRO_SIZE = 256
POLICY_TEACHER_RATIO_MIX_MICRO_SIZE = 16
POLICY_TEACHER_RATIO_MIX_SAMPLE_COUNT = 20_480
POLICY_TEACHER_RATIO_MIX_SUPPORTED_PERCENTAGES = frozenset({25, 50, 100})

POLICY_TEACHER_RATIO_MIX_DEFAULT_TEACHER_ROOT = (
    _legacy.POLICY_TEACHER_QUARTER_MIX_DEFAULT_TEACHER_ROOT
)
POLICY_TEACHER_RATIO_MIX_DEFAULT_SCHEDULE_INDEX = (
    _legacy.POLICY_TEACHER_QUARTER_MIX_DEFAULT_SCHEDULE_INDEX
)

_BASE_MACRO_SOURCE_COUNTS: Mapping[str, int] = MappingProxyType(
    {"vstar": 120, "arxivqa": 77, "thinklite": 59}
)
_TEACHER_PARENT_SOURCE_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "chartqa": 2_200,
        "docvqa": 4_182,
        "textocr": 3_951,
        "textvqa": 4_065,
        "visual_genome": 10_381,
    }
)


class PolicyTeacherRatioMixMaterializationError(ValueError):
    """A parent or derived ratio schedule violates the PRL23 contract."""


class PolicyTeacherRatioMixRuntimeValidationError(ValueError):
    """A ratio schedule differs from its immutable runtime binding."""


def _hamilton_counts(
    population: Mapping[str, int], *, selected_total: int
) -> Mapping[str, int]:
    """Return deterministic largest-remainder quotas in lexical tie order."""

    if not population or any(
        not isinstance(name, str) or type(count) is not int or count < 0
        for name, count in population.items()
    ):
        raise ValueError("population must contain non-negative integer counts")
    population_total = sum(population.values())
    if (
        type(selected_total) is not int
        or selected_total < 0
        or selected_total > population_total
    ):
        raise ValueError("selected_total is outside the population")
    quotas = {
        name: count * selected_total // population_total
        for name, count in population.items()
    }
    remaining = selected_total - sum(quotas.values())
    remainder_order = sorted(
        population,
        key=lambda name: (
            -(population[name] * selected_total % population_total),
            name,
        ),
    )
    for name in remainder_order[:remaining]:
        quotas[name] += 1
    if sum(quotas.values()) != selected_total:
        raise AssertionError("Hamilton apportionment lost rows")
    return MappingProxyType(quotas)


@dataclass(frozen=True, slots=True)
class PolicyTeacherRatioMixProfile:
    teacher_percentage: int
    teacher_per_micro: int
    base_per_micro: int
    teacher_per_macro: int
    base_per_macro: int
    teacher_count: int
    base_count: int
    base_macro_source_counts_cycle: tuple[Mapping[str, int], ...]
    macro_source_counts_cycle: tuple[Mapping[str, int], ...]
    base_source_counts: Mapping[str, int]
    teacher_source_counts: Mapping[str, int]
    role_cadence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.teacher_percentage not in POLICY_TEACHER_RATIO_MIX_SUPPORTED_PERCENTAGES:
            raise ValueError("teacher percentage must be one of 25, 50, or 100")
        if self.teacher_per_micro + self.base_per_micro != (
            POLICY_TEACHER_RATIO_MIX_MICRO_SIZE
        ):
            raise ValueError("micro role counts differ")
        if self.teacher_per_macro + self.base_per_macro != (
            POLICY_TEACHER_RATIO_MIX_MACRO_SIZE
        ):
            raise ValueError("macro role counts differ")
        if self.teacher_count + self.base_count != (
            POLICY_TEACHER_RATIO_MIX_SAMPLE_COUNT
        ):
            raise ValueError("schedule role counts differ")
        if len(self.role_cadence) != POLICY_TEACHER_RATIO_MIX_MICRO_SIZE:
            raise ValueError("role cadence length differs")
        if Counter(self.role_cadence) != Counter(
            {"base": self.base_per_micro, "teacher": self.teacher_per_micro}
        ):
            raise ValueError("role cadence counts differ")
        if not self.base_macro_source_counts_cycle:
            raise ValueError("base macro source-count cycle must not be empty")
        if len(self.base_macro_source_counts_cycle) != len(
            self.macro_source_counts_cycle
        ):
            raise ValueError("base and complete macro cycles differ")
        object.__setattr__(
            self,
            "base_macro_source_counts_cycle",
            tuple(
                MappingProxyType(dict(counts))
                for counts in self.base_macro_source_counts_cycle
            ),
        )
        object.__setattr__(
            self,
            "macro_source_counts_cycle",
            tuple(
                MappingProxyType(dict(counts))
                for counts in self.macro_source_counts_cycle
            ),
        )
        object.__setattr__(
            self, "base_source_counts", MappingProxyType(dict(self.base_source_counts))
        )
        object.__setattr__(
            self,
            "teacher_source_counts",
            MappingProxyType(dict(self.teacher_source_counts)),
        )
        if any(
            sum(counts.values()) != self.base_per_macro
            for counts in self.base_macro_source_counts_cycle
        ):
            raise ValueError("base macro source-count cycle totals differ")
        if any(
            sum(counts.values()) != POLICY_TEACHER_RATIO_MIX_MACRO_SIZE
            for counts in self.macro_source_counts_cycle
        ):
            raise ValueError("complete macro source-count cycle totals differ")
        if sum(self.base_source_counts.values()) != self.base_count:
            raise ValueError("global base source counts differ")

    @property
    def name(self) -> str:
        return f"teacher{self.teacher_percentage}"

    @property
    def interleave_identity(self) -> str:
        compact = "-".join("b" if role == "base" else "t" for role in self.role_cadence)
        return f"fixed-bs16-{compact}-v2"

    def base_macro_source_counts_for(self, macro_index: int) -> Mapping[str, int]:
        if type(macro_index) is not int or macro_index < 0:
            raise ValueError("macro_index must be a non-negative integer")
        return self.base_macro_source_counts_cycle[
            macro_index % len(self.base_macro_source_counts_cycle)
        ]

    def macro_source_counts_for(self, macro_index: int) -> Mapping[str, int]:
        if type(macro_index) is not int or macro_index < 0:
            raise ValueError("macro_index must be a non-negative integer")
        return self.macro_source_counts_cycle[
            macro_index % len(self.macro_source_counts_cycle)
        ]


def policy_teacher_ratio_mix_profile(
    teacher_percentage: int,
) -> PolicyTeacherRatioMixProfile:
    if teacher_percentage not in POLICY_TEACHER_RATIO_MIX_SUPPORTED_PERCENTAGES:
        raise ValueError("teacher percentage must be one of 25, 50, or 100")
    teacher_per_micro = (
        POLICY_TEACHER_RATIO_MIX_MICRO_SIZE * teacher_percentage // 100
    )
    base_per_micro = POLICY_TEACHER_RATIO_MIX_MICRO_SIZE - teacher_per_micro
    teacher_per_macro = (
        POLICY_TEACHER_RATIO_MIX_MACRO_SIZE * teacher_percentage // 100
    )
    base_per_macro = POLICY_TEACHER_RATIO_MIX_MACRO_SIZE - teacher_per_macro
    teacher_count = POLICY_TEACHER_RATIO_MIX_SAMPLE_COUNT * teacher_percentage // 100
    base_count = POLICY_TEACHER_RATIO_MIX_SAMPLE_COUNT - teacher_count
    if teacher_percentage == 25:
        # The published PRL22 artifact deliberately keeps one fixed macro.
        base_macro_cycle: tuple[Mapping[str, int], ...] = (
            MappingProxyType({"vstar": 90, "arxivqa": 58, "thinklite": 44}),
        )
    elif teacher_percentage == 50:
        # Alternate the Hamilton tie so all 80 macros preserve the exact
        # global half of PRL13: V4800, A3080, ThinkLite2360.
        base_macro_cycle = (
            MappingProxyType({"vstar": 60, "arxivqa": 39, "thinklite": 29}),
            MappingProxyType({"vstar": 60, "arxivqa": 38, "thinklite": 30}),
        )
    else:
        base_macro_cycle = (
            MappingProxyType({"vstar": 0, "arxivqa": 0, "thinklite": 0}),
        )
    macro_cycle = tuple(
        MappingProxyType({**dict(counts), "teacher": teacher_per_macro})
        for counts in base_macro_cycle
    )
    cycle_repetitions, cycle_remainder = divmod(
        POLICY_TEACHER_RATIO_MIX_STEPS, len(base_macro_cycle)
    )
    if cycle_remainder:
        raise AssertionError("macro cycle must divide the 80-step horizon")
    base_source_counts = {
        source: cycle_repetitions
        * sum(counts[source] for counts in base_macro_cycle)
        for source in _BASE_MACRO_SOURCE_COUNTS
    }
    teacher_source_counts = _hamilton_counts(
        _TEACHER_PARENT_SOURCE_COUNTS, selected_total=teacher_count
    )
    common_divisor = math.gcd(
        POLICY_TEACHER_RATIO_MIX_MICRO_SIZE, teacher_per_micro
    )
    base_per_period = base_per_micro // common_divisor
    teacher_per_period = teacher_per_micro // common_divisor
    cadence = tuple(
        (["base"] * base_per_period + ["teacher"] * teacher_per_period)
        * common_divisor
    )
    return PolicyTeacherRatioMixProfile(
        teacher_percentage=teacher_percentage,
        teacher_per_micro=teacher_per_micro,
        base_per_micro=base_per_micro,
        teacher_per_macro=teacher_per_macro,
        base_per_macro=base_per_macro,
        teacher_count=teacher_count,
        base_count=base_count,
        base_macro_source_counts_cycle=base_macro_cycle,
        macro_source_counts_cycle=macro_cycle,
        base_source_counts=MappingProxyType(base_source_counts),
        teacher_source_counts=teacher_source_counts,
        role_cadence=cadence,
    )


@dataclass(frozen=True, slots=True)
class PolicyTeacherRatioMixRuntimeBinding:
    manifest_file_sha256: str
    content_sha256: str
    schedule_seed: int
    expected_sample_count: int
    teacher_percentage: int

    def __post_init__(self) -> None:
        _legacy._require_sha256(self.manifest_file_sha256, "manifest_file_sha256")
        _legacy._require_sha256(self.content_sha256, "content_sha256")
        if type(self.schedule_seed) is not int or self.schedule_seed < 0:
            raise ValueError("schedule_seed must be a non-negative integer")
        if self.expected_sample_count != POLICY_TEACHER_RATIO_MIX_SAMPLE_COUNT:
            raise ValueError("ratio schedule sample count must be 20,480")
        policy_teacher_ratio_mix_profile(self.teacher_percentage)

    @property
    def profile(self) -> PolicyTeacherRatioMixProfile:
        return policy_teacher_ratio_mix_profile(self.teacher_percentage)


@dataclass(frozen=True, slots=True)
class PolicyTeacherRatioMixMaterializationResult:
    output_root: Path
    sample_count: int
    samples_sha256: str
    content_sha256: str
    manifest_file_sha256: str
    iteration_identity_sha256: str
    schedule_seed: int
    teacher_percentage: int
    source_counts: Mapping[str, int]
    teacher_source_counts: Mapping[str, int]

    def as_record(self) -> dict[str, object]:
        return {
            "dataset_kind": (
                _legacy.POLICY_TEACHER_QUARTER_MIX_DATASET_KIND
                if self.teacher_percentage == 25
                else POLICY_TEACHER_RATIO_MIX_DATASET_KIND
            ),
            "root": str(self.output_root),
            "decision_stage": "final",
            "sample_count": self.sample_count,
            "manifest_file_sha256": self.manifest_file_sha256,
            "content_sha256": self.content_sha256,
            "samples_sha256": self.samples_sha256,
            "iteration_identity_sha256": self.iteration_identity_sha256,
            "shuffle_seed": self.schedule_seed,
            "teacher_percentage": self.teacher_percentage,
            "source_counts": dict(self.source_counts),
            "teacher_source_counts": dict(self.teacher_source_counts),
        }


@dataclass(frozen=True, slots=True)
class PolicyTeacherRatioMixRuntimeDataset:
    root: Path
    binding: PolicyTeacherRatioMixRuntimeBinding
    samples_sha256: str
    iteration_identity_sha256: str
    samples: tuple[_legacy.PolicyTeacherQuarterMixRuntimeSample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> _legacy.PolicyTeacherQuarterMixRuntimeSample:
        return self.samples[index]


def _legacy_binding(
    binding: PolicyTeacherRatioMixRuntimeBinding,
) -> _legacy.PolicyTeacherQuarterMixRuntimeBinding:
    if binding.teacher_percentage != 25:
        raise ValueError("legacy binding is defined only for Teacher25")
    return _legacy.PolicyTeacherQuarterMixRuntimeBinding(
        manifest_file_sha256=binding.manifest_file_sha256,
        content_sha256=binding.content_sha256,
        schedule_seed=binding.schedule_seed,
        expected_sample_count=binding.expected_sample_count,
    )


def policy_teacher_ratio_mix_iteration_identity_sha256(
    binding: PolicyTeacherRatioMixRuntimeBinding, *, samples_sha256: str
) -> str:
    if not isinstance(binding, PolicyTeacherRatioMixRuntimeBinding):
        raise TypeError("binding must be PolicyTeacherRatioMixRuntimeBinding")
    _legacy._require_sha256(samples_sha256, "samples_sha256")
    if binding.teacher_percentage == 25:
        return _legacy.policy_teacher_quarter_mix_iteration_identity_sha256(
            _legacy_binding(binding), samples_sha256=samples_sha256
        )
    profile = binding.profile
    return _legacy._sha256_bytes(
        _legacy._canonical_json_bytes(
            {
                "schema_version": POLICY_TEACHER_RATIO_MIX_RUNTIME_SCHEMA,
                "dataset_kind": POLICY_TEACHER_RATIO_MIX_DATASET_KIND,
                "sample_count": binding.expected_sample_count,
                "schedule_seed": binding.schedule_seed,
                "teacher_percentage": binding.teacher_percentage,
                "teacher_per_micro": profile.teacher_per_micro,
                "selection_algorithm": POLICY_TEACHER_RATIO_MIX_SELECTION_ALGORITHM,
                "selector_hash_namespace": (
                    POLICY_TEACHER_RATIO_MIX_SELECTOR_HASH_NAMESPACE
                ),
                "interleave": profile.interleave_identity,
                "manifest_file_sha256": binding.manifest_file_sha256,
                "content_sha256": binding.content_sha256,
                "samples_sha256": samples_sha256,
            }
        )
    )


def _stable_teacher_key(record: Mapping[str, Any], seed: int) -> tuple[str, str]:
    """Use the exact PRL22 selector namespace to retain nested prefixes."""

    source = str(record["extra_info"]["source_dataset"])
    sample_id = str(record["sample_id"])
    payload = (
        f"{POLICY_TEACHER_RATIO_MIX_SELECTOR_HASH_NAMESPACE}\0{seed}\0"
        f"{source}\0{sample_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), sample_id


def _build_ratio_schedule_records(
    *,
    teacher_root: Path,
    schedule_index_path: Path,
    schedule_seed: int,
    profile: PolicyTeacherRatioMixProfile,
) -> tuple[list[dict[str, object]], Mapping[str, object]]:
    from .deepeyes_official_schedule_index import load_deepeyes_schedule_index

    if profile.teacher_percentage == 25:
        raise ValueError("Teacher25 must use the byte-compatible legacy builder")
    if schedule_seed != POLICY_TEACHER_RATIO_MIX_SEED:
        raise PolicyTeacherRatioMixMaterializationError(
            "teacher-ratio schedule currently binds seed 42"
        )
    old_index = load_deepeyes_schedule_index(schedule_index_path)
    if len(old_index.train) != POLICY_TEACHER_RATIO_MIX_SAMPLE_COUNT:
        raise PolicyTeacherRatioMixMaterializationError(
            "PRL13 train schedule length differs"
        )
    try:
        _, teacher_rows = _legacy._load_teacher_parent(teacher_root)
    except _legacy.PolicyTeacherQuarterMixMaterializationError as error:
        raise PolicyTeacherRatioMixMaterializationError(str(error)) from error
    grouped: dict[str, list[tuple[int, dict[str, Any], str]]] = defaultdict(list)
    seen_teacher_ids: set[str] = set()
    for row_index, record, row_sha256 in teacher_rows:
        sample_id = _legacy._required_text(
            record.get("sample_id"), field="teacher sample_id"
        )
        if sample_id in seen_teacher_ids:
            raise PolicyTeacherRatioMixMaterializationError(
                "teacher parent contains duplicate sample_id"
            )
        seen_teacher_ids.add(sample_id)
        extra = _legacy._required_mapping(
            record.get("extra_info"), field="teacher extra_info"
        )
        source = _legacy._required_text(
            extra.get("source_dataset"), field="teacher source_dataset"
        )
        grouped[source].append((row_index, record, row_sha256))
    if {source: len(rows) for source, rows in grouped.items()} != dict(
        _TEACHER_PARENT_SOURCE_COUNTS
    ):
        raise PolicyTeacherRatioMixMaterializationError(
            "teacher parent source counts differ"
        )

    selected_by_source: dict[str, deque[dict[str, object]]] = {}
    for source, quota in profile.teacher_source_counts.items():
        ranked = sorted(
            grouped[source], key=lambda item: _stable_teacher_key(item[1], schedule_seed)
        )
        selected_by_source[source] = deque(
            _legacy._validated_teacher_row(*item) for item in ranked[:quota]
        )
    teacher_order = [
        selected_by_source[source].popleft()
        for source in _legacy._smooth_labels(profile.teacher_source_counts)
    ]
    if any(selected_by_source.values()):
        raise AssertionError("teacher selection did not consume its exact quotas")

    output: list[dict[str, object]] = []
    selected_base_ids: set[str] = set()
    selected_teacher_ids: set[str] = set()
    for macro_index in range(POLICY_TEACHER_RATIO_MIX_STEPS):
        begin = macro_index * POLICY_TEACHER_RATIO_MIX_MACRO_SIZE
        macro = old_index.train[begin : begin + POLICY_TEACHER_RATIO_MIX_MACRO_SIZE]
        remaining = dict(profile.base_macro_source_counts_for(macro_index))
        base_records: list[dict[str, object]] = []
        for offset, sample in enumerate(macro):
            source = sample.data_source
            if remaining.get(source, 0) <= 0:
                continue
            base_records.append(
                _legacy._old_record(sample, parent_row_index=begin + offset)
            )
            remaining[source] -= 1
        if (
            remaining != {source: 0 for source in remaining}
            or len(base_records) != profile.base_per_macro
        ):
            raise PolicyTeacherRatioMixMaterializationError(
                f"PRL13 macro {macro_index} cannot satisfy the base source quotas"
            )
        teacher_begin = macro_index * profile.teacher_per_macro
        teacher_macro = teacher_order[
            teacher_begin : teacher_begin + profile.teacher_per_macro
        ]
        if len(teacher_macro) != profile.teacher_per_macro:
            raise AssertionError("teacher macro length differs")
        for micro_index in range(
            POLICY_TEACHER_RATIO_MIX_MACRO_SIZE
            // POLICY_TEACHER_RATIO_MIX_MICRO_SIZE
        ):
            base_begin = micro_index * profile.base_per_micro
            teacher_begin = micro_index * profile.teacher_per_micro
            base_queue = deque(
                base_records[base_begin : base_begin + profile.base_per_micro]
            )
            teacher_queue = deque(
                teacher_macro[
                    teacher_begin : teacher_begin + profile.teacher_per_micro
                ]
            )
            for role in profile.role_cadence:
                record = (
                    base_queue.popleft() if role == "base" else teacher_queue.popleft()
                )
                sample_id = str(record["sample_id"])
                if role == "teacher":
                    selected_teacher_ids.add(sample_id)
                else:
                    selected_base_ids.add(sample_id)
                materialized = _legacy._materialized_record(
                    record, schedule_index=len(output)
                )
                materialized["schema_version"] = POLICY_TEACHER_RATIO_MIX_SAMPLE_SCHEMA
                output.append(materialized)
            if base_queue or teacher_queue:
                raise AssertionError("role cadence did not consume the BS16 group")
    if (
        len(output) != POLICY_TEACHER_RATIO_MIX_SAMPLE_COUNT
        or len(selected_base_ids) != profile.base_count
        or len(selected_teacher_ids) != profile.teacher_count
        or selected_base_ids.intersection(selected_teacher_ids)
    ):
        raise PolicyTeacherRatioMixMaterializationError(
            "teacher-ratio selection uniqueness/count differs"
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
    return output, {
        "selected_exact_image_overlap": selected_overlap_images,
        "parent_population_exact_image_overlap": 587,
        "parent_population_teacher_rows_on_overlapping_images": 1_623,
        "parent_population_exact_image_question_overlap": 0,
    }


def _wrap_legacy_result(
    result: _legacy.PolicyTeacherQuarterMixMaterializationResult,
) -> PolicyTeacherRatioMixMaterializationResult:
    return PolicyTeacherRatioMixMaterializationResult(
        output_root=result.output_root,
        sample_count=result.sample_count,
        samples_sha256=result.samples_sha256,
        content_sha256=result.content_sha256,
        manifest_file_sha256=result.manifest_file_sha256,
        iteration_identity_sha256=result.iteration_identity_sha256,
        schedule_seed=result.schedule_seed,
        teacher_percentage=25,
        source_counts=result.source_counts,
        teacher_source_counts=result.teacher_source_counts,
    )


def materialize_policy_teacher_ratio_mix(
    output_root: str | Path,
    *,
    teacher_percentage: int,
    teacher_root: str | Path = POLICY_TEACHER_RATIO_MIX_DEFAULT_TEACHER_ROOT,
    schedule_index_path: str | Path = (
        POLICY_TEACHER_RATIO_MIX_DEFAULT_SCHEDULE_INDEX
    ),
    schedule_seed: int = POLICY_TEACHER_RATIO_MIX_SEED,
) -> PolicyTeacherRatioMixMaterializationResult:
    """Materialize one ratio profile; Teacher25 delegates to PRL22 v1."""

    profile = policy_teacher_ratio_mix_profile(teacher_percentage)
    if teacher_percentage == 25:
        return _wrap_legacy_result(
            _legacy.materialize_policy_teacher_quarter_mix(
                output_root,
                teacher_root=teacher_root,
                schedule_index_path=schedule_index_path,
                schedule_seed=schedule_seed,
            )
        )
    output = Path(output_root)
    if not output.is_absolute():
        raise PolicyTeacherRatioMixMaterializationError(
            "output_root must be absolute"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"teacher-ratio output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records, overlap_audit = _build_ratio_schedule_records(
        teacher_root=Path(teacher_root),
        schedule_index_path=Path(schedule_index_path),
        schedule_seed=schedule_seed,
        profile=profile,
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    try:
        samples_path = temporary / POLICY_TEACHER_RATIO_MIX_SAMPLES_FILE
        with samples_path.open("xb") as handle:
            for record in records:
                handle.write(_legacy._canonical_json_line(record))
            handle.flush()
            os.fsync(handle.fileno())
        samples_sha256 = _legacy._sha256_file(samples_path)
        source_counts = Counter(str(record["data_source"]) for record in records)
        teacher_source_counts = Counter(
            str(record["source_dataset"])
            for record in records
            if record["data_source"] == "teacher"
        )
        content: dict[str, object] = {
            "schema_version": POLICY_TEACHER_RATIO_MIX_MANIFEST_SCHEMA,
            "dataset_kind": POLICY_TEACHER_RATIO_MIX_DATASET_KIND,
            "decision_stage": "final",
            "sample_count": len(records),
            "parents": {
                "prl13_schedule_index": {
                    "path": str(Path(schedule_index_path)),
                    "file_sha256": (
                        _legacy.POLICY_TEACHER_QUARTER_MIX_SCHEDULE_INDEX_FILE_SHA256
                    ),
                    "identity_sha256": (
                        _legacy.POLICY_TEACHER_QUARTER_MIX_SCHEDULE_INDEX_IDENTITY_SHA256
                    ),
                    "train_rows": POLICY_TEACHER_RATIO_MIX_SAMPLE_COUNT,
                },
                "teacher_t1_retained": {
                    "root": str(Path(teacher_root)),
                    "manifest_file_sha256": (
                        _legacy.POLICY_TEACHER_QUARTER_MIX_TEACHER_MANIFEST_FILE_SHA256
                    ),
                    "content_sha256": (
                        _legacy.POLICY_TEACHER_QUARTER_MIX_TEACHER_CONTENT_SHA256
                    ),
                    "samples_sha256": (
                        _legacy.POLICY_TEACHER_QUARTER_MIX_TEACHER_SAMPLES_SHA256
                    ),
                    "rows": sum(_TEACHER_PARENT_SOURCE_COUNTS.values()),
                },
            },
            "schedule": {
                "selection_algorithm": POLICY_TEACHER_RATIO_MIX_SELECTION_ALGORITHM,
                "selector_hash_namespace": (
                    POLICY_TEACHER_RATIO_MIX_SELECTOR_HASH_NAMESPACE
                ),
                "profile": profile.name,
                "teacher_percentage": teacher_percentage,
                "base_percentage": 100 - teacher_percentage,
                "interleave": profile.interleave_identity,
                "role_cadence": list(profile.role_cadence),
                "seed": schedule_seed,
                "steps": POLICY_TEACHER_RATIO_MIX_STEPS,
                "macro_size": POLICY_TEACHER_RATIO_MIX_MACRO_SIZE,
                "micro_size": POLICY_TEACHER_RATIO_MIX_MICRO_SIZE,
                "macro_source_counts_cycle": [
                    dict(counts) for counts in profile.macro_source_counts_cycle
                ],
                "base_source_counts": dict(profile.base_source_counts),
                "teacher_per_micro": profile.teacher_per_micro,
                "without_replacement": True,
                "dataloader_shuffle": False,
            },
            "source_counts": dict(sorted(source_counts.items())),
            "teacher_source_counts": dict(sorted(teacher_source_counts.items())),
            "samples": {
                "path": POLICY_TEACHER_RATIO_MIX_SAMPLES_FILE,
                "rows": len(records),
                "sha256": samples_sha256,
            },
            "images": {
                "address": "absolute-path-plus-sha256",
                "bytes_verified": "lazy-on-first-access",
            },
            "overlap_audit": overlap_audit,
        }
        content_sha256 = _legacy._sha256_bytes(
            _legacy._canonical_json_bytes(content)
        )
        manifest = {**content, "content_sha256": content_sha256}
        manifest_bytes = _legacy._canonical_json_line(manifest)
        manifest_path = temporary / POLICY_TEACHER_RATIO_MIX_MANIFEST_FILE
        with manifest_path.open("xb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_file_sha256 = _legacy._sha256_bytes(manifest_bytes)
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    binding = PolicyTeacherRatioMixRuntimeBinding(
        manifest_file_sha256=manifest_file_sha256,
        content_sha256=content_sha256,
        schedule_seed=schedule_seed,
        expected_sample_count=len(records),
        teacher_percentage=teacher_percentage,
    )
    return PolicyTeacherRatioMixMaterializationResult(
        output_root=output,
        sample_count=len(records),
        samples_sha256=samples_sha256,
        content_sha256=content_sha256,
        manifest_file_sha256=manifest_file_sha256,
        iteration_identity_sha256=policy_teacher_ratio_mix_iteration_identity_sha256(
            binding, samples_sha256=samples_sha256
        ),
        schedule_seed=schedule_seed,
        teacher_percentage=teacher_percentage,
        source_counts=MappingProxyType(dict(source_counts)),
        teacher_source_counts=MappingProxyType(dict(teacher_source_counts)),
    )


def _load_ratio_manifest(
    root: Path, binding: PolicyTeacherRatioMixRuntimeBinding
) -> tuple[Mapping[str, Any], Path, str]:
    if root.is_symlink() or not root.is_dir():
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "teacher-ratio root must be a regular non-symlink directory"
        )
    manifest_path = root / POLICY_TEACHER_RATIO_MIX_MANIFEST_FILE
    try:
        manifest_bytes = _legacy._safe_file(
            manifest_path, field="mixture manifest"
        ).read_bytes()
    except (
        _legacy.PolicyTeacherQuarterMixMaterializationError,
        _legacy.PolicyTeacherQuarterMixRuntimeValidationError,
    ) as error:
        raise PolicyTeacherRatioMixRuntimeValidationError(str(error)) from error
    if _legacy._sha256_bytes(manifest_bytes) != binding.manifest_file_sha256:
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "mixture manifest file SHA-256 differs"
        )
    try:
        manifest = _legacy._required_mapping(
            _legacy._parse_json(manifest_bytes, field="mixture manifest"),
            field="mixture manifest",
        )
    except _legacy.PolicyTeacherQuarterMixMaterializationError as error:
        raise PolicyTeacherRatioMixRuntimeValidationError(str(error)) from error
    if manifest_bytes != _legacy._canonical_json_line(manifest):
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "mixture manifest is not canonical JSON"
        )
    content = dict(manifest)
    observed_content_sha256 = content.pop("content_sha256", None)
    if (
        observed_content_sha256 != binding.content_sha256
        or _legacy._sha256_bytes(_legacy._canonical_json_bytes(content))
        != binding.content_sha256
        or manifest.get("schema_version") != POLICY_TEACHER_RATIO_MIX_MANIFEST_SCHEMA
        or manifest.get("dataset_kind") != POLICY_TEACHER_RATIO_MIX_DATASET_KIND
        or manifest.get("sample_count") != binding.expected_sample_count
    ):
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "mixture manifest content binding differs"
        )
    profile = binding.profile
    schedule = manifest.get("schedule")
    if not isinstance(schedule, Mapping) or any(
        (
            schedule.get("seed") != binding.schedule_seed,
            schedule.get("teacher_percentage") != binding.teacher_percentage,
            schedule.get("teacher_per_micro") != profile.teacher_per_micro,
            schedule.get("macro_source_counts_cycle")
            != [dict(counts) for counts in profile.macro_source_counts_cycle],
            schedule.get("base_source_counts") != dict(profile.base_source_counts),
            schedule.get("role_cadence") != list(profile.role_cadence),
            schedule.get("selector_hash_namespace")
            != POLICY_TEACHER_RATIO_MIX_SELECTOR_HASH_NAMESPACE,
        )
    ):
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "mixture schedule contract differs"
        )
    expected_source_counts = {
        **{
            source: count
            for source, count in profile.base_source_counts.items()
            if count
        },
        "teacher": profile.teacher_count,
    }
    if (
        manifest.get("source_counts") != expected_source_counts
        or manifest.get("teacher_source_counts")
        != dict(profile.teacher_source_counts)
    ):
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "mixture global source-count contract differs"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, Mapping):
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "mixture samples descriptor differs"
        )
    samples_sha256 = _legacy._require_sha256(
        samples.get("sha256"), "samples.sha256"
    )
    samples_path = root / POLICY_TEACHER_RATIO_MIX_SAMPLES_FILE
    try:
        observed_samples_sha256 = _legacy._sha256_file(
            _legacy._safe_file(samples_path, field="mixture samples")
        )
    except _legacy.PolicyTeacherQuarterMixMaterializationError as error:
        raise PolicyTeacherRatioMixRuntimeValidationError(str(error)) from error
    if (
        samples.get("path") != POLICY_TEACHER_RATIO_MIX_SAMPLES_FILE
        or samples.get("rows") != binding.expected_sample_count
        or observed_samples_sha256 != samples_sha256
    ):
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "mixture samples file binding differs"
        )
    return manifest, samples_path, samples_sha256


def verify_policy_teacher_ratio_mix_artifact_binding(
    root: str | Path,
    *,
    binding: PolicyTeacherRatioMixRuntimeBinding,
    samples_sha256: str,
) -> None:
    if binding.teacher_percentage == 25:
        _legacy.verify_policy_teacher_quarter_mix_artifact_binding(
            root,
            binding=_legacy_binding(binding),
            samples_sha256=samples_sha256,
        )
        return
    expected_samples = _legacy._require_sha256(samples_sha256, "samples_sha256")
    _, _, observed_samples = _load_ratio_manifest(Path(root), binding)
    if observed_samples != expected_samples:
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "configured mixture samples SHA-256 differs"
        )


def load_policy_teacher_ratio_mix_runtime(
    root: str | Path,
    *,
    binding: PolicyTeacherRatioMixRuntimeBinding,
) -> PolicyTeacherRatioMixRuntimeDataset:
    if not isinstance(binding, PolicyTeacherRatioMixRuntimeBinding):
        raise TypeError("binding must be PolicyTeacherRatioMixRuntimeBinding")
    if binding.teacher_percentage == 25:
        legacy_runtime = _legacy.load_policy_teacher_quarter_mix_runtime(
            root, binding=_legacy_binding(binding)
        )
        return PolicyTeacherRatioMixRuntimeDataset(
            root=legacy_runtime.root,
            binding=binding,
            samples_sha256=legacy_runtime.samples_sha256,
            iteration_identity_sha256=legacy_runtime.iteration_identity_sha256,
            samples=legacy_runtime.samples,
        )
    _, samples_path, samples_sha256 = _load_ratio_manifest(Path(root), binding)
    samples: list[_legacy.PolicyTeacherQuarterMixRuntimeSample] = []
    try:
        for row_index, record, _ in _legacy._jsonl_records(samples_path):
            samples.append(
                _legacy._runtime_sample(
                    record,
                    expected_index=row_index,
                    expected_schema=POLICY_TEACHER_RATIO_MIX_SAMPLE_SCHEMA,
                )
            )
    except (
        _legacy.PolicyTeacherQuarterMixMaterializationError,
        _legacy.PolicyTeacherQuarterMixRuntimeValidationError,
    ) as error:
        raise PolicyTeacherRatioMixRuntimeValidationError(str(error)) from error
    if len(samples) != binding.expected_sample_count:
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "mixture runtime sample count differs"
        )
    profile = binding.profile
    for begin in range(0, len(samples), POLICY_TEACHER_RATIO_MIX_MICRO_SIZE):
        micro = samples[begin : begin + POLICY_TEACHER_RATIO_MIX_MICRO_SIZE]
        roles = tuple(
            "teacher" if sample.data_source == "teacher" else "base"
            for sample in micro
        )
        if roles != profile.role_cadence:
            raise PolicyTeacherRatioMixRuntimeValidationError(
                "mixture BS16 role cadence differs"
            )
    for macro_index, begin in enumerate(
        range(0, len(samples), POLICY_TEACHER_RATIO_MIX_MACRO_SIZE)
    ):
        counts = Counter(
            sample.data_source
            for sample in samples[begin : begin + POLICY_TEACHER_RATIO_MIX_MACRO_SIZE]
        )
        if counts != Counter(profile.macro_source_counts_for(macro_index)):
            raise PolicyTeacherRatioMixRuntimeValidationError(
                "mixture macro source counts differ"
            )
    observed_source_counts = Counter(sample.data_source for sample in samples)
    expected_source_counts = Counter(profile.base_source_counts)
    expected_source_counts["teacher"] = profile.teacher_count
    if observed_source_counts != expected_source_counts:
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "mixture runtime global source counts differ"
        )
    observed_teacher_sources = Counter(
        sample.source_dataset for sample in samples if sample.data_source == "teacher"
    )
    if observed_teacher_sources != Counter(profile.teacher_source_counts):
        raise PolicyTeacherRatioMixRuntimeValidationError(
            "mixture runtime teacher-source counts differ"
        )
    iteration_identity = policy_teacher_ratio_mix_iteration_identity_sha256(
        binding, samples_sha256=samples_sha256
    )
    return PolicyTeacherRatioMixRuntimeDataset(
        root=Path(root),
        binding=binding,
        samples_sha256=samples_sha256,
        iteration_identity_sha256=iteration_identity,
        samples=tuple(samples),
    )


__all__ = [
    "POLICY_TEACHER_RATIO_MIX_DATASET_KIND",
    "POLICY_TEACHER_RATIO_MIX_DEFAULT_SCHEDULE_INDEX",
    "POLICY_TEACHER_RATIO_MIX_DEFAULT_TEACHER_ROOT",
    "POLICY_TEACHER_RATIO_MIX_MANIFEST_FILE",
    "POLICY_TEACHER_RATIO_MIX_SAMPLES_FILE",
    "POLICY_TEACHER_RATIO_MIX_SUPPORTED_PERCENTAGES",
    "PolicyTeacherRatioMixMaterializationError",
    "PolicyTeacherRatioMixMaterializationResult",
    "PolicyTeacherRatioMixProfile",
    "PolicyTeacherRatioMixRuntimeBinding",
    "PolicyTeacherRatioMixRuntimeDataset",
    "PolicyTeacherRatioMixRuntimeValidationError",
    "load_policy_teacher_ratio_mix_runtime",
    "materialize_policy_teacher_ratio_mix",
    "policy_teacher_ratio_mix_iteration_identity_sha256",
    "policy_teacher_ratio_mix_profile",
    "verify_policy_teacher_ratio_mix_artifact_binding",
]
