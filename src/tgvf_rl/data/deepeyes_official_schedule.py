"""Strongly joined T1 pool and deterministic DeepEyes training schedules.

The canonical T1 final artifact remains immutable; this module never filters
or rewrites it.  It joins V* grounding boxes back from the candidate sidecar,
reserves one fixed in-distribution probe, and returns index schedules for two
separate arms:

* ``stratified``: exactly 120 V*, 77 ArxivQA, and 59 ThinkLite prompts/step;
* ``natural``: the retained pool's natural mixture under a fixed permutation.

Both arms contain 80 non-repeating batches and exclude every ``T1-PROBE256``
sample.  The two arms are controls and may overlap each other because they are
never part of the same training run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from tgvf_rl.data.policy_selection import SelectionCandidate, SelectionSource
from tgvf_rl.data.policy_t1_mixed_rl_dataset import (
    PolicyT1MixedRuntimeBinding,
    load_policy_t1_mixed_runtime,
)
from tgvf_rl.policy.deepeyes_official_protocol import (
    DEEPEYES_THINKLITE_AGENT_NAME,
    DEEPEYES_VISUAL_AGENT_NAME,
    agent_name_for_source,
    tools_kwargs_for_visual_row,
)


DEEPEYES_T1_SCHEDULE_SCHEMA = "tgvf.deepeyes-native-t1-schedule.v1"
DEEPEYES_T1_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_rl/"
    "T1-04-INSTRUCT-FULL-MIXED-T1-RETAINED-FINAL-v2"
)
DEEPEYES_T1_SAMPLE_COUNT = 77_541
DEEPEYES_T1_MANIFEST_FILE_SHA256 = (
    "752ebe9ea5fced48773b9bc0babfbb6bc57a335dd1b580455f6962053d29fddf"
)
DEEPEYES_T1_CONTENT_SHA256 = (
    "5ab99622a2698a7c52c45795215fa5c467b741c103827a1a7dbe3800ff052934"
)
DEEPEYES_T1_SAMPLES_SHA256 = (
    "06e5b1b9039680111df5ef01f7f969b9cf3d8d0eaefa5774fd8d16169428611a"
)
DEEPEYES_T1_SHUFFLE_SEED = 42
DEEPEYES_CANDIDATE_SIDECAR = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/"
    "policy_selection/full/"
    "qwen3-instruct-t1-vstar170k-arxiv32k-thinklite69842-v1/"
    "candidates.jsonl"
)
DEEPEYES_CANDIDATE_ROWS = 271_842
DEEPEYES_CANDIDATE_SHA256 = (
    "51f4cfeeaa8278c2938f938a2992b30cf91ad9219d5c98d796c8137c60e8b3ec"
)

DEEPEYES_PROMPTS_PER_STEP = 256
DEEPEYES_TOTAL_STEPS = 80
DEEPEYES_PROBE_NAME = "T1-PROBE256"
DEEPEYES_PROBE_SEED = 20_260_807
DEEPEYES_TRAIN_SEED = 42
DEEPEYES_BATCH_COUNTS: Mapping[str, int] = MappingProxyType(
    {"vstar": 120, "arxivqa": 77, "thinklite": 59}
)

ScheduleMode = Literal["stratified", "natural"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _rank(seed: int, namespace: str, sample_id: str) -> tuple[str, str]:
    payload = f"{DEEPEYES_T1_SCHEDULE_SCHEMA}\0{seed}\0{namespace}\0{sample_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), sample_id


@dataclass(frozen=True, slots=True)
class DeepEyesOfficialSample:
    index: int
    sample_id: str
    candidate_sha256: str
    data_source: str
    task_kind: str
    question: str
    ground_truth: str
    image_path: Path
    image_sha256: str
    image_width: int
    image_height: int
    gt_regions: tuple[tuple[int, int, int, int], ...] | None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("sample index must be a non-negative integer")
        if self.data_source not in DEEPEYES_BATCH_COUNTS:
            raise ValueError("sample source is unsupported")
        if not self.sample_id or len(self.candidate_sha256) != 64:
            raise ValueError("sample identity differs")
        if not self.question.strip() or not self.ground_truth.strip():
            raise ValueError("sample question/ground truth must be non-empty")
        if not self.image_path.is_absolute():
            raise ValueError("sample image path must be absolute")
        if len(self.image_sha256) != 64:
            raise ValueError("sample image SHA-256 differs")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("sample image dimensions must be positive")
        if self.data_source == "vstar":
            if self.gt_regions is None or not self.gt_regions:
                raise ValueError("V* rows require candidate-side gt_regions")
        elif self.gt_regions is not None:
            raise ValueError("only V* rows may carry gt_regions")

    @property
    def agent_name(self) -> str:
        return agent_name_for_source(self.data_source)

    @property
    def tools_kwargs(self) -> Mapping[str, object]:
        if self.data_source == "thinklite":
            return {}
        # ArxivQA explicitly supplies the empty tuple; V* supplies GT boxes.
        return tools_kwargs_for_visual_row(self.gt_regions or ())

    def as_verl_route(self) -> dict[str, object]:
        tools_kwargs = dict(self.tools_kwargs)
        return {
            "agent_name": self.agent_name,
            # Pinned veRL RLHFDataset reconstructs the top-level value from
            # ``extra_info.tools_kwargs``.  Keep both so raw/native and
            # post-RLHFDataset rows have the same route identity.
            "tools_kwargs": tools_kwargs,
            "extra_info": {"tools_kwargs": tools_kwargs},
            "data_source": self.data_source,
            "task_kind": self.task_kind,
            "sample_id": self.sample_id,
        }


@dataclass(frozen=True, slots=True)
class DeepEyesSchedule:
    mode: ScheduleMode
    seed: int
    probe_seed: int
    batches: tuple[tuple[int, ...], ...]
    probe_indices: tuple[int, ...]
    samples: tuple[DeepEyesOfficialSample, ...]

    def __post_init__(self) -> None:
        if self.mode not in {"stratified", "natural"}:
            raise ValueError("schedule mode must be stratified or natural")
        if len(self.batches) != DEEPEYES_TOTAL_STEPS:
            raise ValueError("schedule must contain exactly 80 batches")
        if len(self.probe_indices) != DEEPEYES_PROMPTS_PER_STEP:
            raise ValueError("T1-PROBE256 must contain exactly 256 prompts")
        probe = set(self.probe_indices)
        if len(probe) != len(self.probe_indices):
            raise ValueError("T1-PROBE256 contains duplicate samples")
        flat = [index for batch in self.batches for index in batch]
        if any(len(batch) != DEEPEYES_PROMPTS_PER_STEP for batch in self.batches):
            raise ValueError("every schedule batch must contain 256 prompts")
        if len(flat) != len(set(flat)):
            raise ValueError("80-step training schedule repeats a sample")
        if probe.intersection(flat):
            raise ValueError("training schedule overlaps T1-PROBE256")
        if self.mode == "stratified":
            for batch in self.batches:
                counts = Counter(self.samples[index].data_source for index in batch)
                if dict(counts) != dict(DEEPEYES_BATCH_COUNTS):
                    raise ValueError("stratified batch is not exactly 120/77/59")
        probe_counts = Counter(
            self.samples[index].data_source for index in self.probe_indices
        )
        if dict(probe_counts) != dict(DEEPEYES_BATCH_COUNTS):
            raise ValueError("T1-PROBE256 is not exactly 120/77/59")

    def batch(self, optimizer_step: int) -> tuple[DeepEyesOfficialSample, ...]:
        if type(optimizer_step) is not int or not 0 <= optimizer_step < len(
            self.batches
        ):
            raise IndexError(optimizer_step)
        return tuple(self.samples[index] for index in self.batches[optimizer_step])

    @property
    def probe(self) -> tuple[DeepEyesOfficialSample, ...]:
        return tuple(self.samples[index] for index in self.probe_indices)

    @property
    def probe_manifest(self) -> Mapping[str, object]:
        ids = [self.samples[index].sample_id for index in self.probe_indices]
        record = {
            "schema_version": "tgvf.deepeyes-native-t1-probe.v1",
            "name": DEEPEYES_PROBE_NAME,
            "seed": self.probe_seed,
            "sample_count": len(ids),
            "source_counts": dict(DEEPEYES_BATCH_COUNTS),
            "ordered_sample_ids": ids,
        }
        return {**record, "manifest_sha256": _sha256_json(record)}

    @property
    def identity_sha256(self) -> str:
        return _sha256_json(
            {
                "schema_version": DEEPEYES_T1_SCHEDULE_SCHEMA,
                "mode": self.mode,
                "seed": self.seed,
                "probe_manifest_sha256": self.probe_manifest["manifest_sha256"],
                "batch_sha256": [
                    _sha256_json(
                        [self.samples[index].sample_id for index in batch]
                    )
                    for batch in self.batches
                ],
            }
        )


def build_deepeyes_schedule(
    samples: Sequence[DeepEyesOfficialSample],
    *,
    mode: ScheduleMode,
    seed: int = DEEPEYES_TRAIN_SEED,
    probe_seed: int = DEEPEYES_PROBE_SEED,
) -> DeepEyesSchedule:
    """Build one deterministic arm and the shared held-out T1-PROBE256."""

    if type(seed) is not int or seed < 0 or type(probe_seed) is not int or probe_seed < 0:
        raise ValueError("schedule seeds must be non-negative integers")
    if mode not in {"stratified", "natural"}:
        raise ValueError("schedule mode must be stratified or natural")
    sample_tuple = tuple(samples)
    if len({sample.sample_id for sample in sample_tuple}) != len(sample_tuple):
        raise ValueError("schedule population has duplicate sample_id values")
    by_source: dict[str, list[int]] = {source: [] for source in DEEPEYES_BATCH_COUNTS}
    for position, sample in enumerate(sample_tuple):
        if sample.index != position:
            raise ValueError("sample indices must equal their population positions")
        by_source[sample.data_source].append(position)

    probe_indices: list[int] = []
    for source, count in DEEPEYES_BATCH_COUNTS.items():
        ranked = sorted(
            by_source[source],
            key=lambda index: _rank(
                probe_seed, f"{DEEPEYES_PROBE_NAME}/{source}", sample_tuple[index].sample_id
            ),
        )
        if len(ranked) < count:
            raise ValueError(f"source {source} is too small for T1-PROBE256")
        probe_indices.extend(ranked[:count])
    probe_indices.sort(
        key=lambda index: _rank(
            probe_seed, DEEPEYES_PROBE_NAME + "/order", sample_tuple[index].sample_id
        )
    )
    probe_set = set(probe_indices)

    if mode == "stratified":
        selected_by_source: dict[str, list[int]] = {}
        for source, per_step in DEEPEYES_BATCH_COUNTS.items():
            ranked = sorted(
                (index for index in by_source[source] if index not in probe_set),
                key=lambda index: _rank(
                    seed, f"train/stratified/{source}", sample_tuple[index].sample_id
                ),
            )
            required = DEEPEYES_TOTAL_STEPS * per_step
            if len(ranked) < required:
                raise ValueError(f"source {source} is too small for 80 training steps")
            selected_by_source[source] = ranked[:required]
        batches: list[tuple[int, ...]] = []
        for step in range(DEEPEYES_TOTAL_STEPS):
            batch: list[int] = []
            for source, per_step in DEEPEYES_BATCH_COUNTS.items():
                start = step * per_step
                batch.extend(selected_by_source[source][start : start + per_step])
            batch.sort(
                key=lambda index: _rank(
                    seed, f"train/stratified/step/{step}", sample_tuple[index].sample_id
                )
            )
            batches.append(tuple(batch))
    else:
        ranked = sorted(
            (index for index in range(len(sample_tuple)) if index not in probe_set),
            key=lambda index: _rank(
                seed, "train/natural", sample_tuple[index].sample_id
            ),
        )
        required = DEEPEYES_TOTAL_STEPS * DEEPEYES_PROMPTS_PER_STEP
        if len(ranked) < required:
            raise ValueError("population is too small for 80 natural-mixture steps")
        selected = ranked[:required]
        batches = [
            tuple(selected[start : start + DEEPEYES_PROMPTS_PER_STEP])
            for start in range(0, required, DEEPEYES_PROMPTS_PER_STEP)
        ]
    return DeepEyesSchedule(
        mode=mode,
        seed=seed,
        probe_seed=probe_seed,
        batches=tuple(batches),
        probe_indices=tuple(probe_indices),
        samples=sample_tuple,
    )


def _strict_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            try:
                value = json.loads(raw.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield value


def load_deepeyes_official_t1_pool(
    *,
    root: Path = DEEPEYES_T1_ROOT,
    candidate_sidecar: Path = DEEPEYES_CANDIDATE_SIDECAR,
) -> tuple[DeepEyesOfficialSample, ...]:
    """Verify and strongly join all 77,541 final T1 rows to candidate metadata."""

    if root != DEEPEYES_T1_ROOT or candidate_sidecar != DEEPEYES_CANDIDATE_SIDECAR:
        raise ValueError("formal PRL13 pool paths differ from the bound T1 artifacts")
    if candidate_sidecar.is_symlink() or not candidate_sidecar.is_file():
        raise ValueError("candidate sidecar must be a regular non-symlink file")
    if _sha256_file(candidate_sidecar) != DEEPEYES_CANDIDATE_SHA256:
        raise ValueError("candidate sidecar SHA-256 differs")
    runtime = load_policy_t1_mixed_runtime(
        root,
        binding=PolicyT1MixedRuntimeBinding(
            manifest_file_sha256=DEEPEYES_T1_MANIFEST_FILE_SHA256,
            content_sha256=DEEPEYES_T1_CONTENT_SHA256,
            shuffle_seed=DEEPEYES_T1_SHUFFLE_SEED,
            expected_sample_count=DEEPEYES_T1_SAMPLE_COUNT,
        ),
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    candidate_descriptor = manifest.get("inputs", {}).get("candidates")
    if candidate_descriptor != {
        "path": str(candidate_sidecar),
        "rows": DEEPEYES_CANDIDATE_ROWS,
        "sha256": DEEPEYES_CANDIDATE_SHA256,
        "source_counts": {"arxivqa": 32_000, "thinklite": 69_842, "vstar": 170_000},
    }:
        raise ValueError("T1 final manifest candidate-sidecar binding differs")

    retained_by_id = {sample.sample_id: sample for sample in runtime.samples}
    candidates: dict[str, SelectionCandidate] = {}
    seen_ids: set[str] = set()
    observed_rows = 0
    for record in _strict_jsonl(candidate_sidecar):
        observed_rows += 1
        candidate = SelectionCandidate.from_record(record)
        if candidate.sample_id in seen_ids:
            raise ValueError("candidate sidecar contains duplicate sample_id")
        seen_ids.add(candidate.sample_id)
        if candidate.sample_id in retained_by_id:
            candidates[candidate.sample_id] = candidate
    if observed_rows != DEEPEYES_CANDIDATE_ROWS:
        raise ValueError("candidate sidecar row count differs")
    if set(candidates) != set(retained_by_id):
        raise ValueError("candidate sidecar does not cover the complete retained pool")

    final_rows: dict[str, Mapping[str, Any]] = {}
    for record in _strict_jsonl(root / "samples.jsonl"):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in final_rows:
            raise ValueError("T1 final samples contain invalid/duplicate sample_id")
        final_rows[sample_id] = record
    if len(final_rows) != DEEPEYES_T1_SAMPLE_COUNT:
        raise ValueError("T1 final sample row count differs")

    joined: list[DeepEyesOfficialSample] = []
    for index, sample in enumerate(runtime.samples):
        candidate = candidates[sample.sample_id]
        final = final_rows[sample.sample_id]
        image = final.get("image")
        if not isinstance(image, Mapping):
            raise ValueError("T1 final image record differs")
        try:
            source = SelectionSource(sample.data_source)
        except ValueError as error:
            raise ValueError("T1 runtime source differs") from error
        expected_candidate_sha = sample.metadata.get("candidate_sha256")
        if (
            candidate.identity_sha256 != expected_candidate_sha
            or candidate.source is not source
            or candidate.question != sample.question
            or candidate.ground_truth != sample.ground_truth
            or candidate.image.get("path") != str(sample.image_path)
            or candidate.image.get("sha256") != sample.image_sha256
            or candidate.image.get("width") != image.get("width")
            or candidate.image.get("height") != image.get("height")
            or final.get("candidate_sha256") != expected_candidate_sha
            or final.get("data_source") != sample.data_source
            or final.get("extra_info") != {"question": sample.question}
            or final.get("reward_model") != {"ground_truth": sample.ground_truth}
            or image.get("path") != str(sample.image_path)
            or image.get("sha256") != sample.image_sha256
        ):
            raise ValueError(
                f"candidate/final/runtime identity differs for {sample.sample_id}"
            )
        gt_regions = candidate.gt_regions if source is SelectionSource.VSTAR else None
        joined.append(
            DeepEyesOfficialSample(
                index=index,
                sample_id=sample.sample_id,
                candidate_sha256=candidate.identity_sha256,
                data_source=sample.data_source,
                task_kind=sample.task_kind.value,
                question=sample.question,
                ground_truth=sample.ground_truth,
                image_path=sample.image_path,
                image_sha256=sample.image_sha256,
                image_width=int(image["width"]),
                image_height=int(image["height"]),
                gt_regions=gt_regions,
            )
        )
    return tuple(joined)


def assert_verl_route_contract(sample: DeepEyesOfficialSample) -> None:
    """Fail closed on the exact agent/tool kwargs shape expected by PRL13 core."""

    route = sample.as_verl_route()
    if sample.data_source == "thinklite":
        if route["agent_name"] != DEEPEYES_THINKLITE_AGENT_NAME or route[
            "tools_kwargs"
        ] != {} or route["extra_info"] != {"tools_kwargs": {}}:
            raise ValueError("ThinkLite must use single_turn_agent without tools")
        return
    expected_gt = sample.gt_regions if sample.data_source == "vstar" else ()
    if (
        route["agent_name"] != DEEPEYES_VISUAL_AGENT_NAME
        or route["tools_kwargs"] != tools_kwargs_for_visual_row(expected_gt or ())
        or route["extra_info"]
        != {"tools_kwargs": tools_kwargs_for_visual_row(expected_gt or ())}
    ):
        raise ValueError("visual row agent/tools_kwargs route differs")


__all__ = [
    "DEEPEYES_BATCH_COUNTS",
    "DEEPEYES_CANDIDATE_ROWS",
    "DEEPEYES_CANDIDATE_SHA256",
    "DEEPEYES_CANDIDATE_SIDECAR",
    "DEEPEYES_PROBE_NAME",
    "DEEPEYES_PROBE_SEED",
    "DEEPEYES_PROMPTS_PER_STEP",
    "DEEPEYES_T1_SAMPLE_COUNT",
    "DEEPEYES_TOTAL_STEPS",
    "DeepEyesOfficialSample",
    "DeepEyesSchedule",
    "assert_verl_route_contract",
    "build_deepeyes_schedule",
    "load_deepeyes_official_t1_pool",
]
