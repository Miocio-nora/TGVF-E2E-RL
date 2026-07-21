"""Narrow project-owned boundary around a pinned external VLMEvalKit checkout.

This module deliberately does not import VLMEvalKit.  It fixes benchmark and
launch identities while leaving dependency installation and real execution to
an explicitly planned evaluation run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
from typing import Any, Literal


VLMEVALKIT_REVIEW_COMMIT = "7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f"
SHARED_BENCHMARK_ROOT = Path("/nvmesv/dredvpn009/datasets/benchmarks")
COREDEV_2511_MANIFEST_SHA256 = (
    "a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579"
)
COREDEV_2511_SOURCE_FILE_SHA256 = (
    "3a013b2bcc64316054d28239a3cea3f44211cadbfe19787be3b7f285620fa5c1"
)
COREDEV_2511_MANIFEST_ID = "core_balanced_dev_2511_seed20260625"
COREDEV_2511_SEED = 20260625
COREDEV_2511_SAMPLE_COUNT = 2511
_SHA256_CHARS = frozenset("0123456789abcdef")
_TORCHRUN_PROCESS_ENV = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_ERROR_FILE",
)
_VLLM_ENGINE_RUNTIME_FIELDS = (
    "max_model_len",
    "mm_encoder_attn_backend",
)


def isolate_torchrun_environment_for_spawned_factory(factory: Any) -> Any:
    """Hide evaluator-rank identity while a nested vLLM engine is spawned.

    VLMEvalKit first restricts every evaluator rank to one physical GPU.  A
    vLLM V1 engine then spawns its own process.  If that child inherits the
    outer ``LOCAL_WORLD_SIZE``, importing VLMEvalKit's launcher incorrectly
    requires all evaluator GPUs to remain visible inside the one-GPU child.
    The outer environment is restored before dataset inference resumes.
    """

    @wraps(factory)
    def isolated_factory(*args: Any, **kwargs: Any) -> Any:
        saved = {key: os.environ.get(key) for key in _TORCHRUN_PROCESS_ENV}
        for key in _TORCHRUN_PROCESS_ENV:
            os.environ.pop(key, None)
        try:
            return factory(*args, **kwargs)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return isolated_factory


def inject_vllm_engine_options_from_factory_kwargs(factory: Any) -> Any:
    """Forward accepted engine options through VLMEvalKit's Qwen wrapper.

    The pinned Qwen3-VL wrapper accepts arbitrary model configuration fields,
    but constructs :class:`vllm.LLM` with a closed argument list. Keep the
    external checkout immutable while forwarding only the accepted engine
    fields during model construction. The original constructor is restored
    immediately, including when construction fails.
    """

    @wraps(factory)
    def configured_factory(*args: Any, **kwargs: Any) -> Any:
        engine_options = {
            name: kwargs.pop(name)
            for name in _VLLM_ENGINE_RUNTIME_FIELDS
            if name in kwargs
        }
        if not engine_options:
            return factory(*args, **kwargs)

        vllm_module = importlib.import_module("vllm")
        original_llm = vllm_module.LLM

        @wraps(original_llm)
        def configured_llm(*llm_args: Any, **llm_kwargs: Any) -> Any:
            conflicts = {
                name
                for name, value in engine_options.items()
                if name in llm_kwargs and llm_kwargs[name] != value
            }
            if conflicts:
                names = ", ".join(sorted(conflicts))
                raise RuntimeError(f"conflicting vLLM engine options: {names}")
            return original_llm(*llm_args, **llm_kwargs, **engine_options)

        vllm_module.LLM = configured_llm
        try:
            return factory(*args, **kwargs)
        finally:
            vllm_module.LLM = original_llm

    return configured_factory


def materialize_coredev_subset_config(
    *,
    base_config_path: Path,
    output_dir: Path,
    datasets: tuple[str, ...],
) -> Path:
    """Write a content-addressed config for complete CoreDev dataset slices."""

    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("CoreDev subset datasets must be non-empty and unique")
    payload = json.loads(base_config_path.read_text(encoding="utf-8"))
    if set(payload) != {"model", "data"} or not isinstance(payload["data"], dict):
        raise ValueError("CoreDev base config schema drifted")
    available = tuple(payload["data"])
    unknown = tuple(name for name in datasets if name not in payload["data"])
    if unknown:
        raise ValueError(f"unknown CoreDev datasets: {', '.join(unknown)}")
    canonical = tuple(name for name in available if name in datasets)
    if datasets != canonical:
        raise ValueError("CoreDev subset datasets must follow canonical suite order")

    resolved = {
        "model": payload["model"],
        "data": {name: payload["data"][name] for name in datasets},
    }
    content = json.dumps(resolved, indent=2, ensure_ascii=False) + "\n"
    digest = sha256(content.encode("utf-8")).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"coredev-subset-{digest[:16]}.json"
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != content:
            raise RuntimeError("CoreDev resolved-config identity collision")
    else:
        temporary = output_path.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output_path)
    return output_path


def _sha256(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _SHA256_CHARS for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


@dataclass(frozen=True, slots=True)
class CoreDevSliceSpec:
    """One ordered CoreDev slice scored by its native VLMEvalKit scorer."""

    source_id: str
    population_id: str
    vlmeval_dataset: str
    sample_count: int

    def __post_init__(self) -> None:
        for name in ("source_id", "population_id", "vlmeval_dataset"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int
        ):
            raise TypeError("sample_count must be an integer")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")


@dataclass(frozen=True, slots=True)
class CoreDev2511Spec:
    """Exact historical membership, independent of its future TSV encoding."""

    manifest_sha256: str
    seed: int
    slices: tuple[CoreDevSliceSpec, ...]

    def __post_init__(self) -> None:
        _sha256(self.manifest_sha256, name="manifest_sha256")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.slices, tuple) or len(self.slices) != 7:
            raise ValueError("CoreDev-2511 requires exactly seven source slices")
        if any(not isinstance(item, CoreDevSliceSpec) for item in self.slices):
            raise TypeError("CoreDev-2511 contains an invalid slice")
        for field in ("source_id", "population_id", "vlmeval_dataset"):
            values = tuple(getattr(item, field) for item in self.slices)
            if len(set(values)) != len(values):
                raise ValueError(f"CoreDev-2511 {field} values must be unique")
        if self.sample_count != COREDEV_2511_SAMPLE_COUNT:
            raise ValueError("CoreDev-2511 sample count drifted")

    @property
    def sample_count(self) -> int:
        return sum(item.sample_count for item in self.slices)


COREDEV_2511 = CoreDev2511Spec(
    manifest_sha256=COREDEV_2511_MANIFEST_SHA256,
    seed=COREDEV_2511_SEED,
    slices=(
        CoreDevSliceSpec(
            "vstar_bench", "vstar_test_questions_191", "VStarBench", 191
        ),
        CoreDevSliceSpec("hr_bench_4k", "hr_bench_4k_800", "HRBench4K", 200),
        CoreDevSliceSpec("blink", "blink_val_all_subtasks_1901", "BLINK", 420),
        CoreDevSliceSpec(
            "ocrbench_v2", "ocrbench_v2_data_test_10000", "OCRBench_v2", 600
        ),
        CoreDevSliceSpec(
            "mmmu_pro", "mmmu_pro_standard10_test_1730", "MMMU_Pro_10c", 300
        ),
        CoreDevSliceSpec(
            "mathvista", "mathvista_testmini_1000", "MathVista_MINI", 300
        ),
        CoreDevSliceSpec(
            "mathverse", "mathverse_testmini_3940", "MathVerse_MINI", 500
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class CoreDevManifestEntry:
    """One ordered legacy manifest entry, before VLMEvalKit TSV encoding."""

    benchmark: str
    population_id: str
    sample_id: str
    source_file: str
    raw_id: str
    row_index: int

    def __post_init__(self) -> None:
        for name in (
            "benchmark",
            "population_id",
            "sample_id",
            "source_file",
            "raw_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
        ):
            raise ValueError("row_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class CoreDev2511Manifest:
    manifest_id: str
    manifest_sha256: str
    source_file_sha256: str
    seed: int
    entries: tuple[CoreDevManifestEntry, ...]

    def __post_init__(self) -> None:
        if self.manifest_id != COREDEV_2511_MANIFEST_ID:
            raise ValueError("CoreDev-2511 manifest ID drifted")
        if self.manifest_sha256 != COREDEV_2511_MANIFEST_SHA256:
            raise ValueError("CoreDev-2511 logical manifest SHA256 drifted")
        if self.source_file_sha256 != COREDEV_2511_SOURCE_FILE_SHA256:
            raise ValueError("CoreDev-2511 source-file SHA256 drifted")
        if self.seed != COREDEV_2511_SEED:
            raise ValueError("CoreDev-2511 seed drifted")
        if len(self.entries) != COREDEV_2511_SAMPLE_COUNT:
            raise ValueError("CoreDev-2511 entry count drifted")
        sample_ids = tuple(entry.sample_id for entry in self.entries)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("CoreDev-2511 sample IDs must be unique")
        split = self.by_source
        for spec in COREDEV_2511.slices:
            entries = split.get(spec.source_id, ())
            if len(entries) != spec.sample_count:
                raise ValueError(f"CoreDev-2511 {spec.source_id} count drifted")
            if {entry.population_id for entry in entries} != {spec.population_id}:
                raise ValueError(
                    f"CoreDev-2511 {spec.source_id} population identity drifted"
                )
        if set(split) != {spec.source_id for spec in COREDEV_2511.slices}:
            raise ValueError("CoreDev-2511 contains an unknown benchmark source")

    @property
    def by_source(self) -> dict[str, tuple[CoreDevManifestEntry, ...]]:
        return {
            spec.source_id: tuple(
                entry for entry in self.entries if entry.benchmark == spec.source_id
            )
            for spec in COREDEV_2511.slices
        }


def load_coredev_2511_manifest(path: str | Path) -> CoreDev2511Manifest:
    """Load and verify the exact historical manifest before slice conversion."""

    source = Path(path)
    raw = source.read_bytes()
    source_sha256 = sha256(raw).hexdigest()
    if source_sha256 != COREDEV_2511_SOURCE_FILE_SHA256:
        raise ValueError("CoreDev-2511 source-file SHA256 mismatch")
    payload = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "manifest_hash",
        "manifest_id",
        "samples",
        "seed",
        "source_population_ids",
        "stratification",
    }:
        raise ValueError("CoreDev-2511 manifest fields differ from the pinned schema")
    samples = payload["samples"]
    if isinstance(samples, (str, bytes)) or not isinstance(samples, list):
        raise TypeError("CoreDev-2511 samples must be an array")
    entries: list[CoreDevManifestEntry] = []
    for item in samples:
        if not isinstance(item, Mapping) or set(item) != {
            "benchmark",
            "metadata",
            "population_id",
            "sample_id",
            "source_file",
        }:
            raise ValueError("CoreDev-2511 sample fields differ")
        metadata = item["metadata"]
        if not isinstance(metadata, Mapping) or not {"raw_id", "row_index"}.issubset(
            metadata
        ):
            raise ValueError("CoreDev-2511 sample metadata lacks row identity")
        entries.append(
            CoreDevManifestEntry(
                benchmark=item["benchmark"],
                population_id=item["population_id"],
                sample_id=item["sample_id"],
                source_file=item["source_file"],
                raw_id=str(metadata["raw_id"]),
                row_index=metadata["row_index"],
            )
        )
    return CoreDev2511Manifest(
        manifest_id=payload["manifest_id"],
        manifest_sha256=payload["manifest_hash"],
        source_file_sha256=source_sha256,
        seed=payload["seed"],
        entries=tuple(entries),
    )


@dataclass(frozen=True, slots=True)
class VLMEvalKitLaunchPlan:
    """Resolved command/environment for one pinned external evaluation run."""

    checkout: Path
    config_path: Path
    work_dir: Path
    data_root: Path = SHARED_BENCHMARK_ROOT
    python_executable: str = "python"
    mode: Literal["all", "infer", "eval"] = "all"
    prediction_format: Literal["tsv"] = "tsv"
    evaluation_format: Literal["json"] = "json"
    expected_commit: str = VLMEVALKIT_REVIEW_COMMIT

    def __post_init__(self) -> None:
        for name in ("checkout", "config_path", "work_dir", "data_root"):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{name} must be an absolute Path")
        if not isinstance(self.python_executable, str) or not self.python_executable:
            raise ValueError("python_executable must be non-empty")
        if self.mode not in {"all", "infer", "eval"}:
            raise ValueError("unsupported VLMEvalKit mode")
        if self.prediction_format != "tsv" or self.evaluation_format != "json":
            raise ValueError("accepted VLMEvalKit formats are TSV predictions/JSON eval")
        if len(self.expected_commit) != 40 or any(
            char not in _SHA256_CHARS for char in self.expected_commit
        ):
            raise ValueError("expected_commit must be a lowercase git commit")

    @property
    def argv(self) -> tuple[str, ...]:
        return (
            self.python_executable,
            str(self.checkout / "run.py"),
            "--config",
            str(self.config_path),
            "--work-dir",
            str(self.work_dir),
            "--mode",
            self.mode,
        )

    @property
    def environment(self) -> dict[str, str]:
        return {
            "LMUData": str(self.data_root),
            "PRED_FORMAT": self.prediction_format,
            "EVAL_FORMAT": self.evaluation_format,
        }


@dataclass(frozen=True, slots=True)
class TGVFPolicyEvaluationResult:
    """Successful result handed to VLMEvalKit's ``BaseAPI`` boundary."""

    final_answer: str
    extra_records: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.final_answer, str) or not self.final_answer.strip():
            raise ValueError("a successful evaluation result needs a final answer")
        if not isinstance(self.extra_records, Mapping):
            raise TypeError("extra_records must be a mapping")
        try:
            json.dumps(self.extra_records, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("extra_records must be finite JSON data") from error

    def as_generate_inner_result(self) -> tuple[int, str, dict[str, Any]]:
        """Return the official agent-style ``BaseAPI.generate_inner`` shape."""

        return 0, self.final_answer, dict(self.extra_records)
