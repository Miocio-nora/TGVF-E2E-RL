"""Score and aggregate Original Qwen3-VL raw-direct true1M CoreDev inference.

The supervisor waits for the companion inference completion receipt, binds
each raw prediction TSV into an immutable pinned-reuse source view, starts the
accepted Qwen2.5-72B judge only when scoring is still required, and publishes
the frozen seven-component CoreDev Macro* under the Original true1M artifact
root.  ``--validate-only`` performs no GPU query or launch.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping
from urllib import request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.coredev_materialize import (  # noqa: E402
    COREDEV_LLM_JUDGE_MODEL,
)
from tgvf_rl.evaluation.controlled_toolchain import (  # noqa: E402
    build_controlled_toolchain_environment,
    controlled_toolchain_contract,
    controlled_toolchain_verification,
    python312_toolchain_environment,
)
from tgvf_rl.evaluation.coredev_results import (  # noqa: E402
    COREDEV_MACRO_STAR_COMPONENTS,
    extract_coredev_macro_star,
    summarize_coredev_results,
    write_json_atomic,
)
from tgvf_rl.evaluation.final_answer_view import (  # noqa: E402
    COREDEV_REFERENCE_COVERAGE_VIEW_CONTRACT,
    MATHVERSE_METADATA_VIEW_CONTRACT,
    materialize_coredev_reference_coverage_view,
    materialize_mathverse_metadata_view,
)
from tgvf_rl.evaluation.vlmevalkit import (  # noqa: E402
    COREDEV_2511,
    VLMEVALKIT_REVIEW_COMMIT,
)


MAIN_ROOT = Path("/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl")
DEFAULT_OUTPUT_ROOT = (
    MAIN_ROOT
    / "artifacts/evaluation/PRL25-ORIGINAL-QWEN3-INSTRUCT-RAW-DIRECT-TRUE1M-V1"
)
DEFAULT_PYTHON = MAIN_ROOT / ".venv312/bin/python"
INFERENCE_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/evaluation/coredev_2511_qwen3_instruct_direct_prl04_true1m_v1.json"
)
INFERENCE_CONFIG_SHA256 = (
    "7a3f7c55eb3ca8bfd327a0b17d90087628328666b004078188b9b950816b9d13"
)
DEFAULT_TASK_MANIFEST = (
    MAIN_ROOT / "artifacts/evaluation/CoreDev2511-official-visible-v1/tasks.jsonl"
)
DEFAULT_MATHVERSE_SOURCE = Path(
    "/nvmesv/dredvpn009/datasets/benchmarks/mathverse/snapshot/testmini.json"
)
JUDGE_MODEL_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct")
PYTHON_HEADER_ROOT = MAIN_ROOT / ".deps/python312-dev/root/usr/include"
MODEL_NAME = "Qwen3-VL-8B-Instruct"
EVALUATION_CONTRACT = "original-qwen3-instruct-raw-direct-true1m-v1"
INFERENCE_CONTRACT_SCHEMA = "tgvf.original-raw-direct-inference-contract.v2"
INFERENCE_COMPLETION_SCHEMA = "tgvf.original-raw-direct-inference-completion.v2"
RAW_PROMPT_CONTRACT = "official-dataset-raw-prompt-no-system-no-tools"
REQUEST_SEED_NAMESPACE = "coredev-2511-qwen3-instruct-direct-prl04-comparable-v1"
TRUE1M_MAX_PIXELS = 1_003_520
WORKER_ENVIRONMENT_SCHEMA = "tgvf.original-raw-direct-worker-environment.v2"
WORKER_PATH = os.pathsep.join(
    (
        str(MAIN_ROOT / ".venv312/bin"),
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    )
)
_PURGED_TOOLCHAIN_ENVIRONMENT = (
    "ADDR2LINE",
    "AR",
    "AS",
    "BUILD",
    "CC",
    "CC_FOR_BUILD",
    "CFLAGS",
    "CMAKE_ARGS",
    "CMAKE_PREFIX_PATH",
    "COMPILER_PATH",
    "CPP",
    "CPPFLAGS",
    "CPATH",
    "CXX",
    "CXXFILT",
    "CXXFLAGS",
    "CXX_FOR_BUILD",
    "DEBUG_CFLAGS",
    "DEBUG_CPPFLAGS",
    "DEBUG_CXXFLAGS",
    "ELFEDIT",
    "GCC",
    "GCC_AR",
    "GCC_EXEC_PREFIX",
    "GCC_NM",
    "GCC_RANLIB",
    "GPROF",
    "GXX",
    "HOST",
    "LD",
    "LDFLAGS",
    "LD_GOLD",
    "LIBRARY_PATH",
    "NM",
    "NVCC_PREPEND_FLAGS",
    "NVCC_PREPEND_FLAGS_BACKUP",
    "OBJCOPY",
    "OBJDUMP",
    "PATH",
    "RANLIB",
    "READELF",
    "SIZE",
    "STRINGS",
    "STRIP",
    "build_alias",
    "host_alias",
)
_PURGED_ENVIRONMENT_PREFIXES = ("CONDA_", "_CONDA_")
DEFAULT_JUDGE_PORT = 8012
RAW_SOURCE_VIEW_CONTRACT = "tgvf.original-raw-direct-byte-view.v1"
SCORING_CONTRACT = "tgvf.original-raw-direct-true1m-scoring-supervisor.v2"
SCORING_COMPLETION_SCHEMA = "tgvf.original-raw-direct-true1m-scoring-completion.v1"
_SOURCE_VIEW_RUN_DATE = "20260828"
_RUN_ID = re.compile(r"(?:T\d{8}-\d{6}|T\d{8}_G[0-9a-fA-F]+)")
_MATHVERSE_VERSIONS = (
    "Text Dominant",
    "Vision Only",
    "Text Lite",
    "Vision Intensive",
    "Vision Dominant",
)
_HEADLINE_SINGLE_IMAGE_COUNTS = {"BLINK": 180, "MMMU_Pro_10c": 269}
TASK_MANIFEST_SHA256 = (
    "3f69119d24867c3f3210c8b01eb71304247725ddaf9ca983d2b41c2885403cbc"
)
MATHVERSE_SOURCE_SHA256 = (
    "f4ce9b18d111b23d5950dcbc8f377c6a05955a458db6a3103aec706fa63b0e9b"
)
DATASETS = tuple(
    (spec.vlmeval_dataset, spec.sample_count) for spec in COREDEV_2511.slices
)
JUDGE_TOOLCHAIN_ENVIRONMENT = python312_toolchain_environment(
    python_environment_root=MAIN_ROOT / ".venv312",
    python_header_root=PYTHON_HEADER_ROOT,
)


@dataclass(frozen=True, slots=True)
class InferenceSource:
    dataset: str
    expected_rows: int
    source_run_id: str
    status_path: Path
    prediction_path: Path
    prediction_sha256: str


@dataclass(frozen=True, slots=True)
class ScoringSource:
    dataset: str
    expected_rows: int
    source_evaluation_id: str
    source_run_id: str
    work_dir: Path
    cwd: Path
    manifest_path: Path
    prediction_path: Path


@dataclass(frozen=True, slots=True)
class InferenceWorkerContract:
    dataset: str
    expected_rows: int
    gpu_id: int
    resolved_config: Path
    resolved_config_sha256: str
    work_dir: Path


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _absolute_path_without_resolving_symlinks(path: Path) -> Path:
    """Keep virtualenv interpreter identity while making a path absolute."""

    return Path(os.path.abspath(path))


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON artifact is absent or a symlink: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return payload


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _prediction_path(status_path: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"prediction path is missing in {status_path}")
    path = Path(raw_path)
    direct = path if path.is_absolute() else status_path.parent / path
    return direct.resolve()


def _read_tsv_rows(
    path: Path,
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"prediction TSV is absent or a symlink: {path}")
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, path.stat().st_size))
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if (
                reader.fieldnames is None
                or any(not field.strip() for field in reader.fieldnames)
                or len(set(reader.fieldnames)) != len(reader.fieldnames)
            ):
                raise RuntimeError(f"prediction TSV schema differs: {path}")
            fields = tuple(reader.fieldnames)
            rows = []
            for row in reader:
                if None in row or any(row.get(field) is None for field in fields):
                    raise RuntimeError(f"prediction TSV has a ragged row: {path}")
                rows.append({field: str(row[field]) for field in fields})
    finally:
        csv.field_size_limit(previous)
    return fields, tuple(rows)


def _prediction_row_count(path: Path) -> int:
    fields, rows = _read_tsv_rows(path)
    if not {"index", "prediction"}.issubset(fields):
        raise RuntimeError(f"prediction TSV schema differs: {path}")
    indices = [str(row.get("index") or "").strip() for row in rows]
    predictions = [str(row.get("prediction") or "").strip() for row in rows]
    if (
        any(not index for index in indices)
        or len(set(indices)) != len(indices)
        or any(not prediction for prediction in predictions)
    ):
        raise RuntimeError(f"prediction TSV row identity differs: {path}")
    return len(rows)


def _validate_mathverse_metadata_rows(
    source: InferenceSource,
    scoring: ScoringSource,
    *,
    mathverse_source_json: Path,
) -> None:
    source_fields, source_rows = _read_tsv_rows(source.prediction_path)
    derived_fields, derived_rows = _read_tsv_rows(scoring.prediction_path)
    required = {"index", "prediction", "source_row_index", "metadata"}
    if (
        source_fields != derived_fields
        or not required.issubset(source_fields)
        or len(source_rows) != source.expected_rows
        or len(derived_rows) != source.expected_rows
    ):
        raise RuntimeError("MathVerse metadata view table identity differs")
    source_payload = json.loads(mathverse_source_json.read_text(encoding="utf-8"))
    if not isinstance(source_payload, list):
        raise RuntimeError("MathVerse metadata source must contain an array")

    joined_indices: set[int] = set()
    version_counts = {version: 0 for version in _MATHVERSE_VERSIONS}
    for row_number, (raw_row, derived_row) in enumerate(
        zip(source_rows, derived_rows, strict=True), start=1
    ):
        if any(
            raw_row[field] != derived_row[field]
            for field in source_fields
            if field != "metadata"
        ):
            raise RuntimeError(
                f"MathVerse metadata view changed source row {row_number}"
            )
        try:
            source_row_index = int(raw_row["source_row_index"])
        except ValueError as exc:
            raise RuntimeError(
                f"MathVerse row {row_number} source_row_index differs"
            ) from exc
        if (
            source_row_index in joined_indices
            or source_row_index < 0
            or source_row_index >= len(source_payload)
        ):
            raise RuntimeError("MathVerse source-row identity differs")
        joined_indices.add(source_row_index)
        metadata_source_row = source_payload[source_row_index]
        if not isinstance(metadata_source_row, Mapping):
            raise RuntimeError("MathVerse metadata source row is not an object")
        problem_version = metadata_source_row.get("problem_version")
        if problem_version not in version_counts:
            raise RuntimeError("MathVerse problem_version is outside the frozen five")
        try:
            raw_metadata = json.loads(raw_row["metadata"])
            derived_metadata = json.loads(derived_row["metadata"])
        except json.JSONDecodeError as exc:
            raise RuntimeError("MathVerse metadata view contains invalid JSON") from exc
        if not isinstance(raw_metadata, dict) or not isinstance(derived_metadata, dict):
            raise RuntimeError("MathVerse metadata view is not object-valued")
        existing_version = raw_metadata.get("problem_version")
        if existing_version is not None and existing_version != problem_version:
            raise RuntimeError("MathVerse source metadata problem_version differs")
        expected_metadata = dict(raw_metadata)
        expected_metadata["problem_version"] = problem_version
        canonical_metadata = json.dumps(
            expected_metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            derived_metadata != expected_metadata
            or derived_row["metadata"] != canonical_metadata
        ):
            raise RuntimeError("MathVerse metadata-only enrichment differs")
        version_counts[problem_version] += 1

    canonical_mathverse_rows = dict(DATASETS)["MathVerse_MINI"]
    if source.expected_rows == canonical_mathverse_rows and (
        version_counts != {version: 100 for version in _MATHVERSE_VERSIONS}
    ):
        raise RuntimeError("MathVerse five-version coverage differs")


def _one_argv_option(argv: object, name: str) -> str | None:
    if not isinstance(argv, list):
        return None
    values = [str(value) for value in argv]
    positions = [index for index, value in enumerate(values) if value == name]
    if len(positions) != 1:
        return None
    position = positions[0]
    if position + 1 >= len(values) or values[position + 1].startswith("--"):
        return None
    return values[position + 1]


def _expected_controlled_worker_environment(
    worker: InferenceWorkerContract, *, output_root: Path
) -> dict[str, str]:
    cache_root = output_root / "runtime/cache"
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(worker.gpu_id),
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "CPATH": os.pathsep.join(
            (str(PYTHON_HEADER_ROOT), str(PYTHON_HEADER_ROOT / "python3.12"))
        ),
        "LIBRARY_PATH": str(MAIN_ROOT / ".venv312/lib"),
        "PATH": WORKER_PATH,
        "PYTHONHASHSEED": "42",
        "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "TRITON_CACHE_DIR": str(cache_root / "triton" / worker.dataset),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor" / worker.dataset),
        "VLLM_CACHE_ROOT": str(cache_root / "vllm" / worker.dataset),
        "VLLM_PLUGINS": "",
        "VLLM_USE_V1": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def _expected_worker_environment_contract(
    workers: tuple[InferenceWorkerContract, ...], *, output_root: Path
) -> dict[str, Any]:
    required = (
        Path("/usr/bin/gcc"),
        Path("/usr/bin/g++"),
        PYTHON_HEADER_ROOT / "python3.12/Python.h",
        PYTHON_HEADER_ROOT / "python3.12/pyconfig.h",
        PYTHON_HEADER_ROOT / "x86_64-linux-gnu/python3.12/pyconfig.h",
    )
    return {
        "schema_version": WORKER_ENVIRONMENT_SCHEMA,
        "inheritance": "parent-minus-purged-toolchain-then-controlled-overlay",
        "purged_exact": list(_PURGED_TOOLCHAIN_ENVIRONMENT),
        "purged_prefixes": list(_PURGED_ENVIRONMENT_PREFIXES),
        "required_files": [str(path) for path in required],
        "workers": [
            {
                "dataset": worker.dataset,
                "controlled": _expected_controlled_worker_environment(
                    worker, output_root=output_root
                ),
            }
            for worker in workers
        ],
    }


def _load_inference_contract(
    output_root: Path,
    completion: Mapping[str, Any],
) -> dict[str, InferenceWorkerContract]:
    contract_record_path = output_root / "runtime/inference-supervisor/contract.json"
    config_record_path = INFERENCE_CONFIG_PATH
    contract = _load_json(contract_record_path)
    base_config = _load_json(config_record_path)
    contract_path = contract_record_path.resolve()
    config_path = config_record_path.resolve()
    if (
        Path(str(completion.get("inference_contract_path", ""))).resolve()
        != contract_path
        or completion.get("inference_contract_sha256") != _sha256_file(contract_path)
        or Path(str(completion.get("config_path", ""))).resolve() != config_path
        or completion.get("config_sha256") != INFERENCE_CONFIG_SHA256
        or completion.get("request_seed_namespace") != REQUEST_SEED_NAMESPACE
        or completion.get("prompt_contract") != RAW_PROMPT_CONTRACT
        or completion.get("worker_environment_sha256")
        != contract.get("worker_environment_sha256")
        or completion.get("worker_environment") != contract.get("worker_environment")
    ):
        raise RuntimeError("Original inference completion provenance differs")
    if (
        _sha256_file(config_path) != INFERENCE_CONFIG_SHA256
        or contract.get("schema_version") != INFERENCE_CONTRACT_SCHEMA
        or contract.get("evaluation_contract") != EVALUATION_CONTRACT
        or Path(str(contract.get("config_path", ""))).resolve() != config_path
        or contract.get("config_sha256") != INFERENCE_CONFIG_SHA256
        or Path(str(contract.get("output_root", ""))).resolve() != output_root.resolve()
        or contract.get("model") != MODEL_NAME
        or contract.get("prompt_contract") != RAW_PROMPT_CONTRACT
        or contract.get("max_pixels") != TRUE1M_MAX_PIXELS
        or contract.get("request_seed_base") != 42
        or contract.get("request_seed_namespace") != REQUEST_SEED_NAMESPACE
        or contract.get("sample_count") != 2511
        or contract.get("slice_count") != 7
    ):
        raise RuntimeError("Original inference contract identity differs")
    workers = contract.get("workers")
    if (
        not isinstance(workers, list)
        or len(workers) != len(DATASETS)
        or tuple(
            worker.get("dataset") if isinstance(worker, Mapping) else None
            for worker in workers
        )
        != tuple(dataset for dataset, _rows in DATASETS)
    ):
        raise RuntimeError("Original inference worker contract coverage differs")

    worker_contracts: dict[str, InferenceWorkerContract] = {}
    gpu_ids: set[int] = set()
    for worker, (dataset, expected_rows) in zip(workers, DATASETS, strict=True):
        assert isinstance(worker, Mapping)
        raw_resolved_config = Path(str(worker.get("resolved_config", "")))
        resolved_config = raw_resolved_config.resolve()
        work_dir = Path(str(worker.get("work_dir", ""))).resolve()
        gpu_id = worker.get("gpu_id")
        expected_resolved_root = (
            output_root / f"inference/{dataset}/resolved-configs"
        ).resolve()
        expected_work_dir = (output_root / f"inference/{dataset}/work").resolve()
        if (
            worker.get("expected_rows") != expected_rows
            or type(gpu_id) is not int
            or gpu_id < 0
            or gpu_id in gpu_ids
            or raw_resolved_config.is_symlink()
            or not raw_resolved_config.is_file()
            or resolved_config.parent != expected_resolved_root
            or work_dir != expected_work_dir
            or worker.get("resolved_config_sha256") != _sha256_file(resolved_config)
        ):
            raise RuntimeError(f"{dataset} inference worker identity differs")
        resolved_payload = _load_json(resolved_config)
        if resolved_payload != {
            "model": base_config.get("model"),
            "data": {dataset: base_config.get("data", {}).get(dataset)},
        }:
            raise RuntimeError(f"{dataset} resolved inference config differs")
        gpu_ids.add(gpu_id)
        worker_contracts[dataset] = InferenceWorkerContract(
            dataset=dataset,
            expected_rows=expected_rows,
            gpu_id=gpu_id,
            resolved_config=resolved_config,
            resolved_config_sha256=str(worker["resolved_config_sha256"]),
            work_dir=work_dir,
        )
    ordered_worker_contracts = tuple(
        worker_contracts[dataset] for dataset, _expected_rows in DATASETS
    )
    expected_worker_environment = _expected_worker_environment_contract(
        ordered_worker_contracts, output_root=output_root
    )
    if (
        contract.get("worker_environment") != expected_worker_environment
        or contract.get("worker_environment_sha256")
        != _canonical_sha256(expected_worker_environment)
        or completion.get("worker_environment_sha256")
        != _canonical_sha256(expected_worker_environment)
    ):
        raise RuntimeError("Original inference worker environment differs")
    return worker_contracts


def _load_completed_inference(output_root: Path) -> tuple[InferenceSource, ...]:
    completion = (
        output_root
        / "runtime/inference-supervisor/original-true1m-inference-complete.json"
    )
    payload = _load_json(completion)
    if (
        payload.get("schema_version") != INFERENCE_COMPLETION_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("sample_count") != 2511
        or payload.get("slice_count") != 7
        or payload.get("max_pixels") != TRUE1M_MAX_PIXELS
    ):
        raise RuntimeError("Original true1M inference completion contract differs")
    worker_contracts = _load_inference_contract(output_root, payload)
    receipts = payload.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != len(DATASETS):
        raise RuntimeError("Original true1M inference receipt coverage differs")

    sources: list[InferenceSource] = []
    for receipt, (dataset, expected_rows) in zip(receipts, DATASETS, strict=True):
        worker = worker_contracts[dataset]
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("dataset") != dataset
            or receipt.get("rows") != expected_rows
        ):
            raise RuntimeError(f"{dataset} inference receipt identity differs")
        status_path = Path(str(receipt.get("status_path", ""))).resolve()
        prediction_path = Path(str(receipt.get("prediction_path", ""))).resolve()
        expected_model_root = (worker.work_dir / MODEL_NAME).resolve()
        if (
            status_path.name != "status.json"
            or status_path.parent.parent != expected_model_root
            or prediction_path.parent != status_path.parent
            or not _path_within(status_path, output_root)
            or not _path_within(prediction_path, output_root)
        ):
            raise RuntimeError(f"{dataset} inference artifact path differs")
        if _sha256_file(status_path) != receipt.get("status_sha256") or _sha256_file(
            prediction_path
        ) != receipt.get("prediction_sha256"):
            raise RuntimeError(f"{dataset} inference artifact bytes changed")
        status = _load_json(status_path)
        entry = status.get("datasets", {}).get(dataset)
        argv = status.get("argv")
        source_run_id = status_path.parent.name
        if (
            _RUN_ID.fullmatch(source_run_id) is None
            or status.get("eval_id") != source_run_id
            or status.get("mode") != "infer"
            or status.get("model_name") != MODEL_NAME
            or status.get("commit") != VLMEVALKIT_REVIEW_COMMIT[:8]
            or _one_argv_option(argv, "--config") != str(worker.resolved_config)
            or _one_argv_option(argv, "--work-dir") != str(worker.work_dir)
            or _one_argv_option(argv, "--mode") != "infer"
            or not isinstance(entry, Mapping)
            or entry.get("status") != "done"
            or entry.get("skip_reason") != "mode_infer"
            or _prediction_path(status_path, entry.get("prediction_file"))
            != prediction_path
        ):
            raise RuntimeError(f"{dataset} inference status differs")
        if _prediction_row_count(prediction_path) != expected_rows:
            raise RuntimeError(f"{dataset} inference row count differs")
        sources.append(
            InferenceSource(
                dataset=dataset,
                expected_rows=expected_rows,
                source_run_id=source_run_id,
                status_path=status_path,
                prediction_path=prediction_path,
                prediction_sha256=str(receipt["prediction_sha256"]),
            )
        )
    return tuple(sources)


def _source_view_run_id(source: InferenceSource, *, mathverse_source_json: Path) -> str:
    identity = {
        "contract": EVALUATION_CONTRACT,
        "dataset": source.dataset,
        "source_run_id": source.source_run_id,
        "source_prediction_sha256": source.prediction_sha256,
        "mathverse_source_sha256": (
            _sha256_file(mathverse_source_json)
            if source.dataset == "MathVerse_MINI"
            else None
        ),
    }
    return f"T{_SOURCE_VIEW_RUN_DATE}_G{_canonical_sha256(identity)}"


def _raw_source_view_manifest(source: InferenceSource, derived: Path) -> dict[str, Any]:
    if derived.read_bytes() != source.prediction_path.read_bytes():
        raise RuntimeError(f"{source.dataset} raw source view changed prediction bytes")
    return {
        "schema_version": 1,
        "contract": RAW_SOURCE_VIEW_CONTRACT,
        "source": {
            "path": str(source.prediction_path),
            "sha256": source.prediction_sha256,
            "row_count": source.expected_rows,
        },
        "derived": {
            "path": str(derived),
            "sha256": _sha256_file(derived),
            "row_count": source.expected_rows,
        },
        "prediction_values_identical": True,
        "source_bytes_identical": True,
    }


def _validate_source_view(
    source: InferenceSource,
    scoring: ScoringSource,
    *,
    mathverse_source_json: Path,
) -> None:
    manifest = _load_json(scoring.manifest_path)
    status_path = scoring.prediction_path.parent / "status.json"
    status = _load_json(status_path)
    entry = status.get("datasets", {}).get(source.dataset)
    expected_contract = (
        MATHVERSE_METADATA_VIEW_CONTRACT
        if source.dataset == "MathVerse_MINI"
        else RAW_SOURCE_VIEW_CONTRACT
    )
    manifest_source = manifest.get("source")
    manifest_derived = manifest.get("derived")
    if (
        manifest.get("contract") != expected_contract
        or not isinstance(manifest_source, Mapping)
        or not isinstance(manifest_derived, Mapping)
        or Path(str(manifest_source.get("path", ""))).resolve()
        != source.prediction_path
        or manifest_source.get("sha256") != source.prediction_sha256
        or source.prediction_sha256 != _sha256_file(source.prediction_path)
        or manifest_source.get("row_count") != source.expected_rows
        or Path(str(manifest_derived.get("path", ""))).resolve()
        != scoring.prediction_path
        or manifest_derived.get("sha256") != _sha256_file(scoring.prediction_path)
        or manifest_derived.get("row_count") != source.expected_rows
        or manifest.get("prediction_values_identical") is not True
    ):
        raise RuntimeError(f"{source.dataset} scoring source manifest differs")
    if source.dataset == "MathVerse_MINI":
        enrichment = manifest.get("mathverse_metadata_enrichment")
        if (
            not isinstance(enrichment, Mapping)
            or Path(str(enrichment.get("source_json", ""))).resolve()
            != mathverse_source_json.resolve()
            or enrichment.get("source_json_sha256")
            != _sha256_file(mathverse_source_json)
            or enrichment.get("joined_row_count") != source.expected_rows
        ):
            raise RuntimeError("MathVerse metadata source view differs")
        _validate_mathverse_metadata_rows(
            source,
            scoring,
            mathverse_source_json=mathverse_source_json,
        )
    elif (
        manifest.get("source_bytes_identical") is not True
        or scoring.prediction_path.read_bytes() != source.prediction_path.read_bytes()
    ):
        raise RuntimeError(f"{source.dataset} raw source view bytes differ")
    if (
        status.get("eval_id") != scoring.source_run_id
        or status.get("mode") != "infer"
        or status.get("reuse_aux") != "infer"
        or status.get("model_name") != MODEL_NAME
        or status.get("commit") != VLMEVALKIT_REVIEW_COMMIT[:8]
        or not isinstance(entry, Mapping)
        or entry.get("status") != "done"
        or entry.get("source_run") != scoring.source_evaluation_id
        or entry.get("scoring_view_contract") != expected_contract
        or _prediction_path(status_path, entry.get("prediction_file"))
        != scoring.prediction_path
        or _prediction_row_count(scoring.prediction_path) != source.expected_rows
    ):
        raise RuntimeError(f"{source.dataset} scoring source status differs")


def _materialize_source_view(
    source: InferenceSource,
    *,
    output_root: Path,
    mathverse_source_json: Path,
) -> ScoringSource:
    work_dir = (output_root / f"scoring/datasets/{source.dataset}").resolve()
    source_run_id = _source_view_run_id(
        source, mathverse_source_json=mathverse_source_json
    )
    run_dir = work_dir / MODEL_NAME / source_run_id
    prediction = run_dir / f"{MODEL_NAME}_{source.dataset}.tsv"
    manifest_path = run_dir / "final-answer-view-manifest.json"
    scoring = ScoringSource(
        dataset=source.dataset,
        expected_rows=source.expected_rows,
        source_evaluation_id=source.source_run_id,
        source_run_id=source_run_id,
        work_dir=work_dir,
        cwd=(output_root / f"scoring/cwd/{source.dataset}").resolve(),
        manifest_path=manifest_path,
        prediction_path=prediction,
    )
    status_path = run_dir / "status.json"
    if run_dir.exists():
        if not status_path.is_file():
            raise RuntimeError(f"{source.dataset} partial scoring source view exists")
        _validate_source_view(
            source, scoring, mathverse_source_json=mathverse_source_json
        )
        return scoring

    run_dir.mkdir(parents=True, exist_ok=False)
    if source.dataset == "MathVerse_MINI":
        materialize_mathverse_metadata_view(
            source_tsv=source.prediction_path,
            derived_tsv=prediction,
            mathverse_source_json=mathverse_source_json,
            manifest_path=manifest_path,
        )
        view_contract = MATHVERSE_METADATA_VIEW_CONTRACT
    else:
        prediction.write_bytes(source.prediction_path.read_bytes())
        write_json_atomic(
            manifest_path,
            _raw_source_view_manifest(source, prediction),
        )
        view_contract = RAW_SOURCE_VIEW_CONTRACT
    now = datetime.now(timezone.utc).isoformat()
    write_json_atomic(
        status_path,
        {
            "schema_version": "1.0",
            "eval_id": source_run_id,
            "created_at": now,
            "datasets": {
                source.dataset: {
                    "status": "done",
                    "prediction_file": str(prediction),
                    "updated_at": now,
                    "judge_model": COREDEV_LLM_JUDGE_MODEL,
                    "source_run": source.source_run_id,
                    "reuse_aux": "infer",
                    "skip_reason": "mode_infer",
                    "scoring_view_contract": view_contract,
                }
            },
            "model_name": MODEL_NAME,
            "commit": VLMEVALKIT_REVIEW_COMMIT[:8],
            "argv": ["synthetic-original-raw-direct-source-view", source.source_run_id],
            "api_mode": False,
            "world_size": 1,
            "pred_format": "tsv",
            "eval_format": "json",
            "mode": "infer",
            "reuse": False,
            "reuse_aux": "infer",
            "updated_at": now,
        },
    )
    _validate_source_view(source, scoring, mathverse_source_json=mathverse_source_json)
    return scoring


def _score_command(
    source: ScoringSource,
    *,
    python_bin: Path,
    judge_base_url: str,
) -> tuple[str, ...]:
    return (
        str(_absolute_path_without_resolving_symlinks(python_bin)),
        str((REPOSITORY_ROOT / "tools/run_coredev_2511_vlmevalkit.py").resolve()),
        "--data",
        source.dataset,
        "--model",
        MODEL_NAME,
        "--work-dir",
        str(source.work_dir),
        "--mode",
        "eval",
        "--reuse",
        "--reuse-aux",
        "infer",
        "--tgvf-reuse-source-run-id",
        source.source_run_id,
        "--tgvf-reuse-manifest",
        str(source.manifest_path),
        "--judge",
        COREDEV_LLM_JUDGE_MODEL,
        "--judge-base-url",
        judge_base_url,
        "--judge-key",
        "EMPTY",
        "--judge-api-nproc",
        "4",
        "--judge-retry",
        "6",
        "--judge-timeout",
        "600",
    )


def _scoring_environment() -> dict[str, str]:
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH")
    environment.update(
        {
            "OPENAI_API_KEY": "EMPTY",
            "CUDA_VISIBLE_DEVICES": "",
            "VLLM_PLUGINS": "",
            "PYTHONHASHSEED": "42",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(REPOSITORY_ROOT / "src")
            + (f":{prior_pythonpath}" if prior_pythonpath else ""),
        }
    )
    return environment


def _completed_scoring_destination(source: ScoringSource) -> str | None:
    receipt_path = source.work_dir / "pinned-reuse-receipt.json"
    if not receipt_path.exists():
        return None
    receipt = _load_json(receipt_path)
    expected = {
        "schema_version": "tgvf.vlmevalkit-pinned-reuse-receipt.v1",
        "dataset": source.dataset,
        "model": MODEL_NAME,
        "source_evaluation_id": source.source_evaluation_id,
        "source_run_id": source.source_run_id,
        "source_manifest_path": str(source.manifest_path.resolve()),
        "source_prediction_path": str(source.prediction_path.resolve()),
        "source_prediction_sha256": _sha256_file(source.prediction_path),
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise RuntimeError(f"{source.dataset} pinned scoring receipt differs")
    if receipt.get("source_manifest_sha256") != _sha256_file(source.manifest_path):
        raise RuntimeError(f"{source.dataset} source manifest changed after scoring")
    destination_run_id = receipt.get("destination_run_id")
    if (
        not isinstance(destination_run_id, str)
        or _RUN_ID.fullmatch(destination_run_id) is None
    ):
        raise RuntimeError(f"{source.dataset} scorer destination ID differs")
    destination = source.work_dir / MODEL_NAME / destination_run_id
    status_path = destination / "status.json"
    prediction = destination / f"{MODEL_NAME}_{source.dataset}.tsv"
    if (
        Path(str(receipt.get("destination_status_path", ""))).resolve()
        != status_path.resolve()
        or Path(str(receipt.get("destination_prediction_path", ""))).resolve()
        != prediction.resolve()
        or receipt.get("destination_status_sha256") != _sha256_file(status_path)
        or receipt.get("destination_prediction_sha256") != _sha256_file(prediction)
        or _sha256_file(prediction) != _sha256_file(source.prediction_path)
    ):
        raise RuntimeError(f"{source.dataset} scorer destination bytes differ")
    status = _load_json(status_path)
    entry = status.get("datasets", {}).get(source.dataset)
    if (
        status.get("eval_id") != destination_run_id
        or status.get("mode") != "eval"
        or status.get("reuse") is not True
        or status.get("reuse_aux") != "infer"
        or not isinstance(entry, Mapping)
        or entry.get("status") != "done"
        or entry.get("source_run") != source.source_run_id
        or _prediction_path(status_path, entry.get("prediction_file"))
        != prediction.resolve()
    ):
        raise RuntimeError(f"{source.dataset} scorer destination status differs")
    return destination_run_id


def _judge_base_url(port: int) -> str:
    if type(port) is not int or port not in range(8012, 8016):
        raise ValueError("judge port must be one of the pinned ports 8012--8015")
    return f"http://127.0.0.1:{port}/v1"


def _judge_command(*, python_bin: Path, port: int) -> tuple[str, ...]:
    return (
        str(_absolute_path_without_resolving_symlinks(python_bin)),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(JUDGE_MODEL_PATH),
        "--served-model-name",
        COREDEV_LLM_JUDGE_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.85",
        "--max-num-seqs",
        "64",
        "--seed",
        "42",
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
    )


def _judge_environment(gpu_ids: tuple[int, int]) -> dict[str, str]:
    if (
        len(gpu_ids) != 2
        or len(set(gpu_ids)) != 2
        or any(type(gpu) is not int or gpu < 0 for gpu in gpu_ids)
    ):
        raise ValueError("judge requires two unique non-negative GPU IDs")
    return build_controlled_toolchain_environment(
        controlled=JUDGE_TOOLCHAIN_ENVIRONMENT,
        overlay={
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpu_ids),
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_PLUGINS": "",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )


def _busy_gpus(gpu_ids: tuple[int, int]) -> tuple[int, ...]:
    busy = []
    for gpu_id in gpu_ids:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout.strip():
            busy.append(gpu_id)
    return tuple(busy)


def _wait_for_idle_gpus(gpu_ids: tuple[int, int], *, wait: bool) -> None:
    consecutive = 0
    while consecutive < 3:
        busy = _busy_gpus(gpu_ids)
        if busy:
            consecutive = 0
            if not wait:
                raise RuntimeError(f"judge GPUs are busy: {', '.join(map(str, busy))}")
        else:
            consecutive += 1
        if consecutive < 3:
            time.sleep(10)


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_for_judge(process: subprocess.Popen[bytes], *, port: int) -> None:
    health_url = f"http://127.0.0.1:{port}/health"
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError("Qwen2.5-72B judge exited during startup")
        try:
            with request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        time.sleep(5)
    else:
        raise RuntimeError("Qwen2.5-72B judge readiness timed out")
    with request.urlopen(f"{_judge_base_url(port)}/models", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    served = [
        item.get("id") for item in payload.get("data", []) if isinstance(item, Mapping)
    ]
    if COREDEV_LLM_JUDGE_MODEL not in served:
        raise RuntimeError("Qwen2.5-72B served judge identity differs")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def _score_sources(
    sources: tuple[ScoringSource, ...],
    *,
    output_root: Path,
    python_bin: Path,
    judge_base_url: str,
) -> None:
    running: list[tuple[ScoringSource, subprocess.Popen[bytes]]] = []
    try:
        for source in sources:
            source.cwd.mkdir(parents=True, exist_ok=True)
            log_path = output_root / f"logs/score-{source.dataset}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as log_handle:
                process = subprocess.Popen(
                    _score_command(
                        source,
                        python_bin=python_bin,
                        judge_base_url=judge_base_url,
                    ),
                    cwd=source.cwd,
                    env=_scoring_environment(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            running.append((source, process))
        failures = []
        for source, process in running:
            returncode = process.wait()
            if returncode:
                failures.append(f"{source.dataset}={returncode}")
        if failures:
            raise RuntimeError(
                "one or more Original true1M scorers failed: " + ", ".join(failures)
            )
    except BaseException:
        for _source, process in running:
            _terminate_process_group(process)
        raise


def _task_image_counts(task_manifest: Path, *, dataset: str) -> dict[str, int]:
    if task_manifest.is_symlink() or not task_manifest.is_file():
        raise RuntimeError("CoreDev task manifest is absent or a symlink")
    image_counts: dict[str, int] = {}
    with task_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"CoreDev task manifest line {line_number} is invalid"
                ) from exc
            if not isinstance(task, Mapping) or task.get("dataset") != dataset:
                continue
            index = task.get("index")
            image_paths = task.get("image_paths")
            if (
                not isinstance(index, str)
                or not index.strip()
                or index in image_counts
                or not isinstance(image_paths, list)
                or not image_paths
                or any(
                    not isinstance(image_path, str) or not image_path
                    for image_path in image_paths
                )
            ):
                raise RuntimeError(
                    f"CoreDev {dataset} task identity differs at line {line_number}"
                )
            image_counts[index] = len(image_paths)
    return image_counts


def _validate_coverage_view(
    *,
    dataset: str,
    source_result: Path,
    derived_result: Path,
    manifest_path: Path,
    task_manifest: Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    source_manifest = manifest.get("source")
    derived_manifest = manifest.get("derived")
    task_manifest_record = manifest.get("task_manifest")
    expected_rows = dict(DATASETS)[dataset]
    expected_single = _HEADLINE_SINGLE_IMAGE_COUNTS[dataset]
    expected_counts = {
        "single_image_evaluated": expected_single,
        "excluded_multi_image_reference": expected_rows - expected_single,
    }
    source_fields, source_rows = _read_tsv_rows(source_result)
    derived_fields, derived_rows = _read_tsv_rows(derived_result)
    image_counts = _task_image_counts(task_manifest, dataset=dataset)
    if (
        manifest.get("contract") != COREDEV_REFERENCE_COVERAGE_VIEW_CONTRACT
        or manifest.get("dataset") != dataset
        or not isinstance(source_manifest, Mapping)
        or not isinstance(derived_manifest, Mapping)
        or not isinstance(task_manifest_record, Mapping)
        or Path(str(source_manifest.get("path", ""))).resolve() != source_result
        or source_manifest.get("sha256") != _sha256_file(source_result)
        or source_manifest.get("row_count") != expected_rows
        or tuple(source_manifest.get("columns", ())) != source_fields
        or Path(str(derived_manifest.get("path", ""))).resolve()
        != derived_result.resolve()
        or derived_manifest.get("sha256") != _sha256_file(derived_result)
        or derived_manifest.get("row_count") != expected_rows
        or tuple(derived_manifest.get("columns", ())) != derived_fields
        or Path(str(task_manifest_record.get("path", ""))).resolve()
        != task_manifest.resolve()
        or task_manifest_record.get("sha256") != _sha256_file(task_manifest)
        or manifest.get("counts") != expected_counts
        or manifest.get("source_fields_identical") is not True
        or manifest.get("prediction_values_identical") is not True
        or manifest.get("hit_values_identical") is not True
        or derived_fields != (*source_fields, "extra_records")
        or len(source_rows) != expected_rows
        or len(derived_rows) != expected_rows
        or len(image_counts) != expected_rows
    ):
        raise RuntimeError(f"{dataset} coverage view identity differs")

    if (
        not {"index", "prediction", "hit"}.issubset(source_fields)
        or "extra_records" in source_fields
    ):
        raise RuntimeError(f"{dataset} coverage source rows differ")
    source_indices = tuple(row["index"] for row in source_rows)
    if len(set(source_indices)) != expected_rows or set(source_indices) != set(
        image_counts
    ):
        raise RuntimeError(f"{dataset} coverage source identity differs")
    observed_counts = {label: 0 for label in expected_counts}
    for row_number, (source_row, derived_row) in enumerate(
        zip(source_rows, derived_rows, strict=True), start=1
    ):
        if any(source_row[field] != derived_row[field] for field in source_fields):
            raise RuntimeError(
                f"{dataset} coverage view changed source row {row_number}"
            )
        expected_coverage = (
            "single_image_evaluated"
            if image_counts[source_row["index"]] == 1
            else "excluded_multi_image_reference"
        )
        try:
            extra_records = json.loads(derived_row["extra_records"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{dataset} coverage metadata is invalid") from exc
        expected_extra_records = {
            "schema_version": COREDEV_REFERENCE_COVERAGE_VIEW_CONTRACT,
            "dataset": dataset,
            "coverage": expected_coverage,
        }
        if extra_records != expected_extra_records:
            raise RuntimeError(f"{dataset} coverage metadata differs")
        observed_counts[expected_coverage] += 1
    if observed_counts != expected_counts:
        raise RuntimeError(f"{dataset} single-image coverage counts differ")
    return manifest


def _coverage_view(
    item: dict[str, Any],
    *,
    output_root: Path,
    task_manifest: Path,
) -> None:
    dataset = str(item["dataset"])
    if dataset not in _HEADLINE_SINGLE_IMAGE_COUNTS:
        return
    source_status = Path(str(item["status_path"])).resolve()
    if source_status.is_symlink() or not source_status.is_file():
        raise RuntimeError(f"{dataset} scorer status is absent or a symlink")
    results = tuple(source_status.parent.glob(f"{MODEL_NAME}_{dataset}_*_result.tsv"))
    if len(results) != 1:
        raise RuntimeError(f"{dataset} scorer result TSV is not unique")
    source_result = results[0].resolve()
    derived_dir = output_root / f"scoring/headline-coverage-views/{dataset}"
    derived_result = derived_dir / source_result.name
    manifest_path = derived_dir / "coverage-view-manifest.json"
    status_path = derived_dir / "status.json"
    if derived_result.exists() != manifest_path.exists():
        raise RuntimeError(f"{dataset} partial coverage view exists")
    if not derived_result.exists():
        materialize_coredev_reference_coverage_view(
            source_tsv=source_result,
            derived_tsv=derived_result,
            task_manifest_path=task_manifest,
            dataset=dataset,
            manifest_path=manifest_path,
        )
    manifest = _validate_coverage_view(
        dataset=dataset,
        source_result=source_result,
        derived_result=derived_result,
        manifest_path=manifest_path,
        task_manifest=task_manifest,
    )
    status_payload = {
        "schema_version": 1,
        "contract": COREDEV_REFERENCE_COVERAGE_VIEW_CONTRACT,
        "dataset": dataset,
        "source_status_path": str(source_status),
        "source_status_sha256": _sha256_file(source_status),
        "source_result_path": str(source_result),
        "source_result_sha256": _sha256_file(source_result),
        "derived_result_path": str(derived_result.resolve()),
        "derived_result_sha256": _sha256_file(derived_result),
        "coverage_view_manifest": str(manifest_path.resolve()),
        "coverage_view_manifest_sha256": _sha256_file(manifest_path),
        "task_manifest_path": str(task_manifest.resolve()),
        "task_manifest_sha256": _sha256_file(task_manifest),
        "counts": manifest["counts"],
    }
    if status_path.exists():
        if _load_json(status_path) != status_payload:
            raise RuntimeError(f"{dataset} coverage status differs")
    else:
        write_json_atomic(status_path, status_payload)
    item["status_path"] = str(status_path.resolve())


def _revalidate_aggregate_provenance(
    sources: tuple[ScoringSource, ...],
    *,
    output_root: Path,
    mathverse_source_json: Path,
) -> None:
    inference_sources = _load_completed_inference(output_root)
    if tuple(source.dataset for source in sources) != tuple(
        source.dataset for source in inference_sources
    ):
        raise RuntimeError("Original aggregate source order differs")
    for inference_source, scoring_source in zip(
        inference_sources, sources, strict=True
    ):
        if (
            scoring_source.expected_rows != inference_source.expected_rows
            or scoring_source.source_evaluation_id != inference_source.source_run_id
            or scoring_source.source_run_id
            != _source_view_run_id(
                inference_source,
                mathverse_source_json=mathverse_source_json,
            )
        ):
            raise RuntimeError(
                f"{scoring_source.dataset} aggregate source provenance differs"
            )
        _validate_source_view(
            inference_source,
            scoring_source,
            mathverse_source_json=mathverse_source_json,
        )


def _aggregate(
    sources: tuple[ScoringSource, ...],
    *,
    output_root: Path,
    task_manifest: Path,
    mathverse_source_json: Path,
    judge_base_url: str,
) -> dict[str, Any]:
    _validate_frozen_scoring_inputs(
        task_manifest=task_manifest,
        mathverse_source_json=mathverse_source_json,
    )
    _revalidate_aggregate_provenance(
        sources,
        output_root=output_root,
        mathverse_source_json=mathverse_source_json,
    )
    expected_eval_ids = {}
    for source in sources:
        destination = _completed_scoring_destination(source)
        if destination is None:
            raise RuntimeError(f"{source.dataset} pinned scoring receipt is absent")
        expected_eval_ids[source.dataset] = destination
    summary = summarize_coredev_results(
        work_dir=(output_root / "scoring/datasets").resolve(),
        repository_root=REPOSITORY_ROOT.resolve(),
        phase="eval",
        expected_judge_base_url=judge_base_url,
        expected_model=MODEL_NAME,
        expected_eval_ids=expected_eval_ids,
    )
    slices = summary.get("slices")
    if (
        summary.get("sample_count") != 2511
        or summary.get("slice_count") != 7
        or not isinstance(slices, list)
        or len(slices) != 7
    ):
        raise RuntimeError("Original true1M scoring coverage differs")
    for item in slices:
        if not isinstance(item, dict):
            raise RuntimeError("Original true1M scorer slice is malformed")
        _coverage_view(
            item,
            output_root=output_root,
            task_manifest=task_manifest,
        )
    summary.update(
        {
            "evaluation_contract": EVALUATION_CONTRACT,
            "scoring_contract": SCORING_CONTRACT,
            "max_pixels": TRUE1M_MAX_PIXELS,
            "judge_base_url": judge_base_url,
            "judge_model": COREDEV_LLM_JUDGE_MODEL,
            "source_run_ids": {
                source.dataset: source.source_run_id for source in sources
            },
            "raw_inference_run_ids": {
                source.dataset: source.source_evaluation_id for source in sources
            },
        }
    )
    headline = extract_coredev_macro_star(summary)
    if (
        tuple(headline.get("components_percent", {})) != COREDEV_MACRO_STAR_COMPONENTS
        or headline.get("single_image_counts", {}).get("blink", {}).get("count") != 180
        or headline.get("single_image_counts", {}).get("mmmu", {}).get("count") != 269
        or tuple(headline.get("mathverse_version_components_percent", {}))
        != _MATHVERSE_VERSIONS
        or headline.get("aggregation") != "unweighted_mean_of_seven_percent_components"
    ):
        raise RuntimeError("Original true1M Macro* contract differs")
    summary["headline"] = headline
    output = output_root / "scoring/coredev-2511-eval-summary.json"
    write_json_atomic(output, summary)
    return summary


def _static_contract(
    *,
    output_root: Path,
    python_bin: Path,
    task_manifest: Path,
    mathverse_source_json: Path,
    judge_port: int,
    judge_gpu_ids: tuple[int, int],
) -> dict[str, Any]:
    judge_base_url = _judge_base_url(judge_port)
    judge_environment = _judge_environment(judge_gpu_ids)
    return {
        "schema_version": SCORING_CONTRACT,
        "evaluation_contract": EVALUATION_CONTRACT,
        "output_root": str(output_root.resolve()),
        "inference_completion": str(
            (
                output_root / "runtime/inference-supervisor/"
                "original-true1m-inference-complete.json"
            ).resolve()
        ),
        "inference_contract": str(
            (output_root / "runtime/inference-supervisor/contract.json").resolve()
        ),
        "inference_config": str(INFERENCE_CONFIG_PATH.resolve()),
        "inference_config_sha256": INFERENCE_CONFIG_SHA256,
        "inference_contract_schema": INFERENCE_CONTRACT_SCHEMA,
        "inference_completion_schema": INFERENCE_COMPLETION_SCHEMA,
        "worker_environment_schema": WORKER_ENVIRONMENT_SCHEMA,
        "raw_prompt_contract": RAW_PROMPT_CONTRACT,
        "request_seed_namespace": REQUEST_SEED_NAMESPACE,
        "model": MODEL_NAME,
        "max_pixels": TRUE1M_MAX_PIXELS,
        "sample_count": 2511,
        "slice_count": 7,
        "datasets": [
            {"dataset": dataset, "expected_rows": rows} for dataset, rows in DATASETS
        ],
        "judge": {
            "model": COREDEV_LLM_JUDGE_MODEL,
            "model_path": str(JUDGE_MODEL_PATH),
            "base_url": judge_base_url,
            "gpu_ids": list(judge_gpu_ids),
            "tensor_parallel_size": 2,
            "dtype": "bfloat16",
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 64,
            "seed": 42,
            "generation_config": "vllm",
            "prefix_caching": True,
            "attention_backend": judge_environment["VLLM_ATTENTION_BACKEND"],
            "toolchain_environment": controlled_toolchain_contract(
                JUDGE_TOOLCHAIN_ENVIRONMENT
            ),
            "toolchain_verification": controlled_toolchain_verification(
                judge_environment,
                controlled=JUDGE_TOOLCHAIN_ENVIRONMENT,
            ),
        },
        "scorer": {
            "python": str(_absolute_path_without_resolving_symlinks(python_bin)),
            "runner": str(
                (REPOSITORY_ROOT / "tools/run_coredev_2511_vlmevalkit.py").resolve()
            ),
            "reuse": True,
            "reuse_aux": "infer",
            "pinned_source_run": True,
            "judge_api_nproc": 4,
            "judge_retry": 6,
            "judge_timeout": 600,
        },
        "source_views": {
            "raw_prediction_bytes_identical": True,
            "mathverse_metadata_only": True,
            "mathverse_source_json": str(mathverse_source_json.resolve()),
            "mathverse_source_sha256": MATHVERSE_SOURCE_SHA256,
        },
        "headline": {
            "components": list(COREDEV_MACRO_STAR_COMPONENTS),
            "aggregation": "unweighted_mean_of_seven_percent_components",
            "blink_single_image_count": 180,
            "mmmu_single_image_count": 269,
            "mathverse_versions": list(_MATHVERSE_VERSIONS),
            "task_manifest": str(task_manifest.resolve()),
            "task_manifest_sha256": TASK_MANIFEST_SHA256,
        },
    }


def _validate_frozen_scoring_inputs(
    *,
    task_manifest: Path,
    mathverse_source_json: Path,
) -> None:
    if (
        task_manifest.is_symlink()
        or not task_manifest.is_file()
        or _sha256_file(task_manifest) != TASK_MANIFEST_SHA256
    ):
        raise RuntimeError("frozen CoreDev task manifest SHA256 differs")
    if (
        mathverse_source_json.is_symlink()
        or not mathverse_source_json.is_file()
        or _sha256_file(mathverse_source_json) != MATHVERSE_SOURCE_SHA256
    ):
        raise RuntimeError("frozen MathVerse source SHA256 differs")


def _validate_static_dependencies(
    *,
    python_bin: Path,
    task_manifest: Path,
    mathverse_source_json: Path,
) -> None:
    required_files = (
        python_bin,
        INFERENCE_CONFIG_PATH,
        task_manifest,
        mathverse_source_json,
        REPOSITORY_ROOT / "tools/run_coredev_2511_vlmevalkit.py",
        PYTHON_HEADER_ROOT / "python3.12/Python.h",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if not JUDGE_MODEL_PATH.is_dir():
        missing.append(str(JUDGE_MODEL_PATH))
    if missing:
        raise RuntimeError(
            f"Original true1M scoring dependencies are absent: {missing}"
        )
    if _sha256_file(INFERENCE_CONFIG_PATH) != INFERENCE_CONFIG_SHA256:
        raise RuntimeError("frozen Original inference config SHA256 differs")
    _validate_frozen_scoring_inputs(
        task_manifest=task_manifest,
        mathverse_source_json=mathverse_source_json,
    )


def _wait_for_inference(output_root: Path) -> None:
    completion = (
        output_root
        / "runtime/inference-supervisor/original-true1m-inference-complete.json"
    )
    failure = output_root / "runtime/inference-supervisor/failed.json"
    while not completion.is_file():
        if failure.is_file():
            raise RuntimeError("Original true1M inference supervisor reported failure")
        time.sleep(5)


def _run_supervisor(
    *,
    output_root: Path,
    python_bin: Path,
    task_manifest: Path,
    mathverse_source_json: Path,
    judge_port: int,
    judge_gpu_ids: tuple[int, int],
    wait_for_gpus: bool,
) -> None:
    control_root = output_root / "runtime/scoring-supervisor"
    control_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (control_root / "supervisor.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            "Original true1M scoring supervisor is already active"
        ) from exc

    judge: subprocess.Popen[bytes] | None = None
    phase = "initializing"
    failure_path = control_root / "failed.json"
    completion_path = control_root / "original-true1m-scoring-complete.json"
    try:
        _validate_frozen_scoring_inputs(
            task_manifest=task_manifest,
            mathverse_source_json=mathverse_source_json,
        )
        contract = _static_contract(
            output_root=output_root,
            python_bin=python_bin,
            task_manifest=task_manifest,
            mathverse_source_json=mathverse_source_json,
            judge_port=judge_port,
            judge_gpu_ids=judge_gpu_ids,
        )
        contract_path = control_root / "contract.json"
        if contract_path.exists():
            if _load_json(contract_path) != contract:
                raise RuntimeError("existing Original true1M scoring contract differs")
        else:
            write_json_atomic(contract_path, contract)
        completion_path.unlink(missing_ok=True)

        phase = "waiting_for_inference"
        _wait_for_inference(output_root)
        phase = "validating_inference"
        inference_sources = _load_completed_inference(output_root)
        phase = "materializing_source_views"
        sources = tuple(
            _materialize_source_view(
                source,
                output_root=output_root,
                mathverse_source_json=mathverse_source_json,
            )
            for source in inference_sources
        )
        pending = tuple(
            source
            for source in sources
            if _completed_scoring_destination(source) is None
        )
        judge_base_url = _judge_base_url(judge_port)
        if pending:
            phase = "waiting_for_judge_gpus"
            _wait_for_idle_gpus(judge_gpu_ids, wait=wait_for_gpus)
            if not _port_is_available(judge_port):
                raise RuntimeError(f"judge port {judge_port} is already in use")
            phase = "starting_judge"
            judge_log = output_root / "logs/qwen25-72b-judge-original-true1m.log"
            judge_log.parent.mkdir(parents=True, exist_ok=True)
            with judge_log.open("ab") as log_handle:
                judge = subprocess.Popen(
                    _judge_command(python_bin=python_bin, port=judge_port),
                    cwd=MAIN_ROOT,
                    env=_judge_environment(judge_gpu_ids),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            _wait_for_judge(judge, port=judge_port)
            phase = "scoring"
            _score_sources(
                pending,
                output_root=output_root,
                python_bin=python_bin,
                judge_base_url=judge_base_url,
            )
            for source in pending:
                if _completed_scoring_destination(source) is None:
                    raise RuntimeError(f"{source.dataset} scorer receipt is absent")

        phase = "aggregating"
        summary = _aggregate(
            sources,
            output_root=output_root,
            task_manifest=task_manifest,
            mathverse_source_json=mathverse_source_json,
            judge_base_url=judge_base_url,
        )
        summary_path = output_root / "scoring/coredev-2511-eval-summary.json"
        write_json_atomic(
            completion_path,
            {
                "schema_version": SCORING_COMPLETION_SCHEMA,
                "status": "complete",
                "evaluation_contract": EVALUATION_CONTRACT,
                "sample_count": summary["sample_count"],
                "slice_count": summary["slice_count"],
                "max_pixels": TRUE1M_MAX_PIXELS,
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": _sha256_file(summary_path),
                "macro_star_percent": summary["headline"]["macro_star_percent"],
            },
        )
        failure_path.unlink(missing_ok=True)
    except BaseException as exc:
        write_json_atomic(
            failure_path,
            {
                "schema_version": (
                    "tgvf.original-raw-direct-true1m-scoring-failure.v1"
                ),
                "phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        if judge is not None:
            _terminate_process_group(judge)
        lock_handle.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument(
        "--mathverse-source-json", type=Path, default=DEFAULT_MATHVERSE_SOURCE
    )
    parser.add_argument("--judge-port", type=int, default=DEFAULT_JUDGE_PORT)
    parser.add_argument("--judge-gpu-ids", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the static scoring contract without querying or using GPUs.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_root = args.output_root.resolve()
    python_bin = _absolute_path_without_resolving_symlinks(args.python_bin)
    task_manifest = args.task_manifest.resolve()
    mathverse_source_json = args.mathverse_source_json.resolve()
    judge_gpu_ids = tuple(args.judge_gpu_ids)
    _validate_static_dependencies(
        python_bin=python_bin,
        task_manifest=task_manifest,
        mathverse_source_json=mathverse_source_json,
    )
    contract = _static_contract(
        output_root=output_root,
        python_bin=python_bin,
        task_manifest=task_manifest,
        mathverse_source_json=mathverse_source_json,
        judge_port=args.judge_port,
        judge_gpu_ids=judge_gpu_ids,
    )
    if args.validate_only:
        print(json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    _run_supervisor(
        output_root=output_root,
        python_bin=python_bin,
        task_manifest=task_manifest,
        mathverse_source_json=mathverse_source_json,
        judge_port=args.judge_port,
        judge_gpu_ids=judge_gpu_ids,
        wait_for_gpus=args.wait_for_gpus,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
