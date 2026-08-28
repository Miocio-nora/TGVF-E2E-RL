"""Run Original Qwen3-VL raw-direct CoreDev inference, one slice per GPU.

This supervisor deliberately covers inference only.  It writes the same isolated
``inference/<dataset>/work`` layout consumed by the established CoreDev scorer,
coverage-view materializer, and Macro* aggregator.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.vlmevalkit import (  # noqa: E402
    materialize_coredev_subset_config,
)


MAIN_ROOT = Path("/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl")
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/evaluation/coredev_2511_qwen3_instruct_direct_prl04_true1m_v1.json"
)
DEFAULT_CONFIG_SHA256 = (
    "7a3f7c55eb3ca8bfd327a0b17d90087628328666b004078188b9b950816b9d13"
)
DEFAULT_OUTPUT_ROOT = (
    MAIN_ROOT / "artifacts/evaluation/"
    "PRL25-ORIGINAL-QWEN3-INSTRUCT-RAW-DIRECT-TRUE1M-V1"
)
DEFAULT_PYTHON = MAIN_ROOT / ".venv312/bin/python"
MODEL_NAME = "Qwen3-VL-8B-Instruct"
EVALUATION_CONTRACT = "original-qwen3-instruct-raw-direct-true1m-v1"
INFERENCE_CONTRACT_SCHEMA = "tgvf.original-raw-direct-inference-contract.v2"
INFERENCE_COMPLETION_SCHEMA = "tgvf.original-raw-direct-inference-completion.v2"
RAW_PROMPT_CONTRACT = "official-dataset-raw-prompt-no-system-no-tools"
TRUE1M_MAX_PIXELS = 1_003_520
SEED_NAMESPACE = "coredev-2511-qwen3-instruct-direct-prl04-comparable-v1"
_RUN_ID = re.compile(r"(?:T\d{8}-\d{6}|T\d{8}_G[0-9a-fA-F]+)")
PYTHON_HEADER_ROOT = MAIN_ROOT / ".deps/python312-dev/root/usr/include"
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
WORKER_ENVIRONMENT_SCHEMA = "tgvf.original-raw-direct-worker-environment.v2"
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
VLMEVALKIT_QWEN3_SOURCE = Path(
    "/nvmesv/dredvpn009/tools/VLMEvalKit/"
    "7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f/"
    "vlmeval/vlm/qwen3_vl/model.py"
)
VLMEVALKIT_QWEN3_SOURCE_SHA256 = (
    "9cf2a927ae9d0ea39901bff6f702167f70e1af0d4ab35af4656ccf87346b336d"
)


@dataclass(frozen=True, slots=True)
class DatasetContract:
    name: str
    class_name: str
    expected_rows: int


@dataclass(frozen=True, slots=True)
class DatasetLaunch:
    dataset: str
    gpu_id: int
    expected_rows: int
    resolved_config: Path
    work_dir: Path
    cwd: Path
    log_path: Path
    command: tuple[str, ...]


DATASETS = (
    DatasetContract("VStarBench", "CoreDev2511VStarBenchSlice", 191),
    DatasetContract("HRBench4K", "CoreDev2511HRBench4KSlice", 200),
    DatasetContract("BLINK", "CoreDev2511BLINKSlice", 420),
    DatasetContract("OCRBench_v2", "CoreDev2511OCRBenchv2Slice", 600),
    DatasetContract("MMMU_Pro_10c", "CoreDev2511MMMUPro10cSlice", 300),
    DatasetContract("MathVista_MINI", "CoreDev2511MathVistaMINISlice", 300),
    DatasetContract("MathVerse_MINI", "CoreDev2511MathVerseMINISlice", 500),
)

EXPECTED_MODEL_CONTRACT: dict[str, Any] = {
    "model_path": "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct",
    "use_custom_prompt": False,
    "use_vllm": True,
    "min_pixels": None,
    "max_pixels": TRUE1M_MAX_PIXELS,
    "total_pixels": None,
    "max_new_tokens": 8192,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "do_sample": True,
    "post_process": False,
    "system_prompt": None,
    "gpu_utils": 0.9,
    "inference_batch_size": 8,
    "request_seed_base": 42,
    "request_seed_namespace": SEED_NAMESPACE,
    "limit_mm_per_prompt": {"image": 24, "video": 0},
    "max_model_len": 65536,
    "mm_encoder_attn_backend": "TORCH_SDPA",
}


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


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _load_and_validate_base_config(path: Path) -> dict[str, Any]:
    if (
        path.resolve() != DEFAULT_CONFIG.resolve()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise RuntimeError("Original raw-direct config is absent or a symlink")
    if _sha256_file(path) != DEFAULT_CONFIG_SHA256:
        raise RuntimeError("Original raw-direct config SHA256 differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"model", "data"}:
        raise RuntimeError("Original raw-direct config top-level schema differs")
    if tuple(payload["model"]) != (MODEL_NAME,):
        raise RuntimeError("Original raw-direct config model identity differs")
    if payload["model"][MODEL_NAME] != EXPECTED_MODEL_CONTRACT:
        raise RuntimeError(
            "Original raw-direct PRL04-compatible model contract differs"
        )

    expected_data = {
        item.name: {"class": item.class_name, "dataset": item.name} for item in DATASETS
    }
    if payload["data"] != expected_data:
        raise RuntimeError("Original raw-direct CoreDev suite identity/order differs")
    return payload


def _direct_pixel_probe(*, model_path: Path, max_pixels: int) -> dict[str, Any]:
    """Execute the pinned raw-direct image-item -> processor path on CPU."""

    if (
        VLMEVALKIT_QWEN3_SOURCE.is_symlink()
        or not VLMEVALKIT_QWEN3_SOURCE.is_file()
        or _sha256_file(VLMEVALKIT_QWEN3_SOURCE) != VLMEVALKIT_QWEN3_SOURCE_SHA256
    ):
        raise RuntimeError("pinned VLMEvalKit Qwen3 source identity differs")
    if max_pixels != TRUE1M_MAX_PIXELS:
        raise RuntimeError("Original raw-direct processor probe is not true1M")

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    image_processor = getattr(processor, "image_processor", None)
    processor_size = getattr(image_processor, "size", None)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if (
        not isinstance(processor_size, dict)
        or type(patch_size) is not int
        or patch_size != 16
        or type(merge_size) is not int
        or merge_size != 2
    ):
        raise RuntimeError("Original raw-direct Qwen3 processor geometry differs")

    source_width, source_height = 2048, 1536
    source_area = source_width * source_height
    resized_image: Image.Image | None = None
    with tempfile.TemporaryDirectory(prefix="original-true1m-pixel-probe-") as raw:
        image_path = Path(raw) / "probe.png"
        source_image = Image.new("RGB", (source_width, source_height), (10, 20, 30))
        try:
            source_image.save(image_path, format="PNG")
        finally:
            source_image.close()
        image_item = {
            "type": "image",
            "image": f"file://{image_path}",
            "max_pixels": max_pixels,
        }
        messages = [
            {
                "role": "user",
                "content": [
                    image_item,
                    {"type": "text", "text": "STATIC_PIXEL_PROBE"},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if (
            not isinstance(images, list)
            or len(images) != 1
            or not isinstance(images[0], Image.Image)
            or videos is not None
        ):
            raise RuntimeError("Original raw-direct vision-info probe differs")
        resized_image = images[0]
        resized_width, resized_height = resized_image.size
        batch = processor(
            text=[text_prompt], images=[resized_image], return_tensors="pt"
        )
        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is None or tuple(image_grid_thw.shape) != (1, 3):
            raise RuntimeError("Original raw-direct processor grid differs")
        grid = tuple(int(value) for value in image_grid_thw[0].tolist())

    if resized_image is not None:
        resized_image.close()
    temporal, grid_height, grid_width = grid
    represented_area = temporal * grid_height * grid_width * patch_size**2
    premerge_tokens = temporal * grid_height * grid_width
    if (
        source_area <= max_pixels
        or temporal != 1
        or represented_area != resized_width * resized_height
        or represented_area > max_pixels
        or premerge_tokens % (merge_size**2)
    ):
        raise RuntimeError("Original raw-direct represented-pixel proof failed")
    return {
        "schema_version": "tgvf.original-raw-direct-processor-proof.v1",
        "path_semantics": "image-item-max_pixels-then-process_vision_info-v1",
        "vlmevalkit_qwen3_source_path": str(VLMEVALKIT_QWEN3_SOURCE),
        "vlmevalkit_qwen3_source_sha256": VLMEVALKIT_QWEN3_SOURCE_SHA256,
        "configured_max_pixels": max_pixels,
        "image_item_has_max_pixels": image_item["max_pixels"] == max_pixels,
        "processor_default_size": dict(processor_size),
        "processor_patch_size": patch_size,
        "processor_merge_size": merge_size,
        "source_dimensions": [source_width, source_height],
        "source_pixel_area": source_area,
        "process_vision_info_dimensions": [resized_width, resized_height],
        "image_grid_thw": list(grid),
        "represented_pixel_area": represented_area,
        "premerge_patch_count": premerge_tokens,
        "visual_token_count": premerge_tokens // (merge_size**2),
        "video_kwargs": video_kwargs,
    }


def _validate_gpu_ids(
    gpu_ids: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(gpu_ids) not in {7, 8}:
        raise ValueError(
            "Original raw-direct inference requires seven or eight GPU IDs"
        )
    if len(set(gpu_ids)) != len(gpu_ids) or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("GPU IDs must be unique non-negative integers")
    return gpu_ids[: len(DATASETS)], gpu_ids[len(DATASETS) :]


def _materialize_launches(
    *,
    config_path: Path,
    output_root: Path,
    python_bin: Path,
    gpu_ids: tuple[int, ...],
) -> tuple[tuple[DatasetLaunch, ...], tuple[int, ...]]:
    payload = _load_and_validate_base_config(config_path)
    assigned_gpus, spare_gpus = _validate_gpu_ids(gpu_ids)
    launches = []
    for item, gpu_id in zip(DATASETS, assigned_gpus, strict=True):
        dataset_root = output_root / "inference" / item.name
        resolved = materialize_coredev_subset_config(
            base_config_path=config_path,
            output_dir=dataset_root / "resolved-configs",
            datasets=(item.name,),
        ).resolve()
        resolved_payload = json.loads(resolved.read_text(encoding="utf-8"))
        if resolved_payload.get("model") != payload["model"] or resolved_payload.get(
            "data"
        ) != {item.name: payload["data"][item.name]}:
            raise RuntimeError(f"{item.name} isolated config differs")

        work_dir = (dataset_root / "work").resolve()
        cwd = (dataset_root / "cwd").resolve()
        command = (
            str(_absolute_path_without_resolving_symlinks(python_bin)),
            str((REPOSITORY_ROOT / "tools/run_coredev_2511_vlmevalkit.py").resolve()),
            "--config",
            str(resolved),
            "--work-dir",
            str(work_dir),
            "--mode",
            "infer",
        )
        launches.append(
            DatasetLaunch(
                dataset=item.name,
                gpu_id=gpu_id,
                expected_rows=item.expected_rows,
                resolved_config=resolved,
                work_dir=work_dir,
                cwd=cwd,
                log_path=(output_root / "logs" / f"infer-{item.name}.log").resolve(),
                command=command,
            )
        )
    return tuple(launches), spare_gpus


def _controlled_worker_environment(
    launch: DatasetLaunch, *, output_root: Path
) -> dict[str, str]:
    cache_root = output_root / "runtime/cache"
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(launch.gpu_id),
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
        "TRITON_CACHE_DIR": str(cache_root / "triton" / launch.dataset),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor" / launch.dataset),
        "VLLM_CACHE_ROOT": str(cache_root / "vllm" / launch.dataset),
        "VLLM_PLUGINS": "",
        "VLLM_USE_V1": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def _worker_environment_contract(
    *, output_root: Path, launches: tuple[DatasetLaunch, ...]
) -> dict[str, Any]:
    required = (
        Path("/usr/bin/gcc"),
        Path("/usr/bin/g++"),
        PYTHON_HEADER_ROOT / "python3.12/Python.h",
        PYTHON_HEADER_ROOT / "python3.12/pyconfig.h",
        PYTHON_HEADER_ROOT / "x86_64-linux-gnu/python3.12/pyconfig.h",
    )
    if any(path.is_symlink() or not path.is_file() for path in required[2:]) or any(
        not path.is_file() for path in required[:2]
    ):
        raise RuntimeError("Original true1M worker toolchain dependencies differ")
    return {
        "schema_version": WORKER_ENVIRONMENT_SCHEMA,
        "inheritance": "parent-minus-purged-toolchain-then-controlled-overlay",
        "purged_exact": list(_PURGED_TOOLCHAIN_ENVIRONMENT),
        "purged_prefixes": list(_PURGED_ENVIRONMENT_PREFIXES),
        "required_files": [str(path) for path in required],
        "workers": [
            {
                "dataset": launch.dataset,
                "controlled": _controlled_worker_environment(
                    launch, output_root=output_root
                ),
            }
            for launch in launches
        ],
    }


def _contract_receipt(
    *,
    config_path: Path,
    output_root: Path,
    launches: tuple[DatasetLaunch, ...],
    spare_gpus: tuple[int, ...],
) -> dict[str, Any]:
    pixel_probe = _direct_pixel_probe(
        model_path=Path(str(EXPECTED_MODEL_CONTRACT["model_path"])),
        max_pixels=TRUE1M_MAX_PIXELS,
    )
    worker_environment = _worker_environment_contract(
        output_root=output_root, launches=launches
    )
    return {
        "schema_version": INFERENCE_CONTRACT_SCHEMA,
        "evaluation_contract": EVALUATION_CONTRACT,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "output_root": str(output_root.resolve()),
        "model": MODEL_NAME,
        "prompt_contract": RAW_PROMPT_CONTRACT,
        "max_pixels": TRUE1M_MAX_PIXELS,
        "request_seed_base": 42,
        "request_seed_namespace": SEED_NAMESPACE,
        "worker_environment": worker_environment,
        "worker_environment_sha256": _canonical_sha256(worker_environment),
        "pixel_preprocessing_proof": pixel_probe,
        "sample_count": sum(item.expected_rows for item in launches),
        "slice_count": len(launches),
        "workers": [
            {
                "dataset": item.dataset,
                "expected_rows": item.expected_rows,
                "gpu_id": item.gpu_id,
                "resolved_config": str(item.resolved_config),
                "resolved_config_sha256": _sha256_file(item.resolved_config),
                "work_dir": str(item.work_dir),
            }
            for item in launches
        ],
        "spare_gpu_ids": list(spare_gpus),
    }


def _option_value(argv: list[object], name: str) -> str | None:
    values = [str(item) for item in argv]
    positions = [index for index, value in enumerate(values) if value == name]
    if len(positions) != 1:
        return None
    index = positions[0]
    if index + 1 >= len(values) or values[index + 1].startswith("--"):
        return None
    return values[index + 1]


def _prediction_path(status_path: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"prediction path is missing in {status_path}")
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    direct = status_path.parent / path
    return (direct if direct.exists() else Path.cwd() / path).resolve()


def _validate_prediction(path: Path, *, expected_rows: int) -> int:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"prediction is missing or is a symlink: {path}")
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"index", "prediction"}.issubset(
            reader.fieldnames
        ):
            raise RuntimeError(f"prediction TSV schema differs: {path}")
        rows = list(reader)
    indices = [str(row["index"]).strip() for row in rows]
    predictions = [str(row["prediction"]).strip() for row in rows]
    if any(not value for value in indices) or len(set(indices)) != len(indices):
        raise RuntimeError(f"prediction TSV index identity differs: {path}")
    if any(not value for value in predictions):
        raise RuntimeError(f"prediction TSV contains an empty prediction: {path}")
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"prediction row count differs: {len(rows)} != {expected_rows}: {path}"
        )
    return len(rows)


def _completed_receipt(launch: DatasetLaunch) -> dict[str, Any] | None:
    model_root = launch.work_dir / MODEL_NAME
    candidates = []
    for status_path in model_root.glob("T*/status.json"):
        if status_path.is_symlink() or not status_path.is_file():
            continue
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        dataset_statuses = payload.get("datasets")
        if not isinstance(dataset_statuses, dict) or set(dataset_statuses) != {
            launch.dataset
        }:
            continue
        entry = dataset_statuses[launch.dataset]
        argv = payload.get("argv", [])
        run_id = status_path.parent.name
        if (
            _RUN_ID.fullmatch(run_id) is None
            or payload.get("eval_id") != run_id
            or payload.get("mode") != "infer"
            or payload.get("model_name") != MODEL_NAME
            or not isinstance(argv, list)
            or _option_value(argv, "--config") != str(launch.resolved_config)
            or _option_value(argv, "--work-dir") != str(launch.work_dir)
            or _option_value(argv, "--mode") != "infer"
            or not isinstance(entry, dict)
            or entry.get("status") != "done"
            or entry.get("skip_reason") != "mode_infer"
        ):
            continue
        try:
            prediction = _prediction_path(status_path, entry.get("prediction_file"))
        except RuntimeError:
            continue
        if (
            prediction.parent != status_path.parent.resolve()
            or prediction.name != f"{MODEL_NAME}_{launch.dataset}.tsv"
        ):
            continue
        candidates.append(
            (status_path.stat().st_mtime_ns, status_path, entry, prediction)
        )
    if not candidates:
        return None
    _mtime, status_path, _entry, prediction = max(
        candidates, key=lambda candidate: candidate[0]
    )
    rows = _validate_prediction(prediction, expected_rows=launch.expected_rows)
    return {
        "dataset": launch.dataset,
        "rows": rows,
        "status_path": str(status_path.resolve()),
        "status_sha256": _sha256_file(status_path),
        "prediction_path": str(prediction),
        "prediction_sha256": _sha256_file(prediction),
    }


def _busy_gpus(gpu_ids: tuple[int, ...]) -> tuple[int, ...]:
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


def _wait_for_gpus(gpu_ids: tuple[int, ...], *, wait: bool) -> None:
    while busy := _busy_gpus(gpu_ids):
        if not wait:
            raise RuntimeError(f"assigned GPUs are busy: {', '.join(map(str, busy))}")
        print(f"assigned GPUs still busy: {', '.join(map(str, busy))}", flush=True)
        time.sleep(10)


def _launch_environment(launch: DatasetLaunch, *, output_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name in _PURGED_TOOLCHAIN_ENVIRONMENT or any(
            name.startswith(prefix) for prefix in _PURGED_ENVIRONMENT_PREFIXES
        ):
            environment.pop(name)
    controlled = _controlled_worker_environment(launch, output_root=output_root)
    for name in (
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "VLLM_CACHE_ROOT",
    ):
        Path(controlled[name]).mkdir(parents=True, exist_ok=True)
    environment.update(controlled)
    return environment


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return


def _write_launch_table(
    path: Path, launched: list[tuple[DatasetLaunch, subprocess.Popen[bytes]]]
) -> None:
    rows = ["dataset\tgpu_id\tpid\tresolved_config"]
    rows.extend(
        f"{item.dataset}\t{item.gpu_id}\t{process.pid}\t{item.resolved_config}"
        for item, process in launched
    )
    _write_text_atomic(path, "\n".join(rows) + "\n")


def _write_receipt_table(path: Path, receipts: tuple[dict[str, Any], ...]) -> None:
    rows = ["dataset\trows\tstatus_path\tprediction_path\tprediction_sha256"]
    rows.extend(
        "\t".join(
            str(receipt[field])
            for field in (
                "dataset",
                "rows",
                "status_path",
                "prediction_path",
                "prediction_sha256",
            )
        )
        for receipt in receipts
    )
    _write_text_atomic(path, "\n".join(rows) + "\n")


def _inference_completion_receipt(
    *,
    contract_path: Path,
    contract: dict[str, Any],
    launches: tuple[DatasetLaunch, ...],
    receipts: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    persisted_contract = (
        json.loads(contract_path.read_text(encoding="utf-8"))
        if contract_path.is_file() and not contract_path.is_symlink()
        else None
    )
    expected_datasets = tuple(item.name for item in DATASETS)
    receipt_datasets = tuple(receipt.get("dataset") for receipt in receipts)
    expected_workers = [
        {
            "dataset": item.dataset,
            "expected_rows": item.expected_rows,
            "gpu_id": item.gpu_id,
            "resolved_config": str(item.resolved_config),
            "resolved_config_sha256": _sha256_file(item.resolved_config),
            "work_dir": str(item.work_dir),
        }
        for item in launches
    ]
    contract_output_root = Path(str(contract.get("output_root", ""))).resolve()
    expected_worker_environment = _worker_environment_contract(
        output_root=contract_output_root, launches=launches
    )
    revalidated_receipts = tuple(_completed_receipt(item) for item in launches)
    if (
        persisted_contract != contract
        or contract.get("schema_version") != INFERENCE_CONTRACT_SCHEMA
        or contract.get("evaluation_contract") != EVALUATION_CONTRACT
        or Path(str(contract.get("config_path", ""))).resolve()
        != DEFAULT_CONFIG.resolve()
        or contract.get("config_sha256") != DEFAULT_CONFIG_SHA256
        or _sha256_file(DEFAULT_CONFIG) != DEFAULT_CONFIG_SHA256
        or contract.get("prompt_contract") != RAW_PROMPT_CONTRACT
        or contract.get("request_seed_namespace") != SEED_NAMESPACE
        or contract.get("max_pixels") != TRUE1M_MAX_PIXELS
        or contract.get("sample_count") != 2511
        or contract.get("slice_count") != len(DATASETS)
        or tuple(item.dataset for item in launches) != expected_datasets
        or contract.get("workers") != expected_workers
        or contract.get("worker_environment") != expected_worker_environment
        or contract.get("worker_environment_sha256")
        != _canonical_sha256(expected_worker_environment)
        or len(receipts) != len(DATASETS)
        or receipt_datasets != expected_datasets
        or sum(receipt.get("rows", 0) for receipt in receipts) != 2511
        or revalidated_receipts != receipts
    ):
        raise RuntimeError("Original true1M completion inputs differ")
    return {
        "schema_version": INFERENCE_COMPLETION_SCHEMA,
        "status": "complete",
        "sample_count": 2511,
        "slice_count": 7,
        "max_pixels": TRUE1M_MAX_PIXELS,
        "inference_contract_path": str(contract_path.resolve()),
        "inference_contract_sha256": _sha256_file(contract_path),
        "config_path": contract["config_path"],
        "config_sha256": contract["config_sha256"],
        "request_seed_namespace": contract["request_seed_namespace"],
        "prompt_contract": contract["prompt_contract"],
        "worker_environment": contract["worker_environment"],
        "worker_environment_sha256": contract["worker_environment_sha256"],
        "receipts": receipts,
    }


def _run_inference(
    *,
    output_root: Path,
    launches: tuple[DatasetLaunch, ...],
    contract: dict[str, Any],
    wait_for_gpus: bool,
) -> None:
    control_root = output_root / "runtime/inference-supervisor"
    control_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (control_root / "supervisor.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            "Original true1M inference supervisor is already active"
        ) from exc

    running: list[tuple[DatasetLaunch, subprocess.Popen[bytes]]] = []
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def fail_on_sigterm(signum: int, _frame: object) -> None:
        raise RuntimeError(f"Original true1M inference received signal {signum}")

    signal.signal(signal.SIGTERM, fail_on_sigterm)
    try:
        contract_path = control_root / "contract.json"
        if contract_path.exists():
            if contract_path.is_symlink() or not contract_path.is_file():
                raise RuntimeError(
                    "existing Original true1M inference contract is not regular"
                )
            observed = json.loads(contract_path.read_text(encoding="utf-8"))
            if observed != contract:
                raise RuntimeError(
                    "existing Original true1M inference contract differs; "
                    "quarantine the complete failed output root before retry"
                )
        else:
            _write_json_atomic(contract_path, contract)

        completed = {item.dataset: _completed_receipt(item) for item in launches}
        pending = tuple(item for item in launches if completed[item.dataset] is None)
        if pending:
            _wait_for_gpus(tuple(item.gpu_id for item in pending), wait=wait_for_gpus)
            for item in pending:
                item.cwd.mkdir(parents=True, exist_ok=True)
                item.work_dir.mkdir(parents=True, exist_ok=True)
                item.log_path.parent.mkdir(parents=True, exist_ok=True)
                with item.log_path.open("a", encoding="utf-8") as log_handle:
                    process = subprocess.Popen(
                        item.command,
                        cwd=item.cwd,
                        env=_launch_environment(item, output_root=output_root),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                running.append((item, process))
            _write_launch_table(control_root / "launch.tsv", running)

            while running:
                failed = [
                    (item, process.returncode)
                    for item, process in running
                    if process.poll() not in {None, 0}
                ]
                if failed:
                    detail = ", ".join(
                        f"{item.dataset}={returncode}" for item, returncode in failed
                    )
                    raise RuntimeError(
                        f"Original true1M inference worker failed: {detail}"
                    )
                running = [
                    (item, process)
                    for item, process in running
                    if process.returncode is None
                ]
                if running:
                    time.sleep(1)

        receipts = tuple(_completed_receipt(item) for item in launches)
        if any(receipt is None for receipt in receipts):
            raise RuntimeError(
                "one or more Original true1M inference receipts are missing"
            )
        typed_receipts = tuple(receipt for receipt in receipts if receipt is not None)
        if sum(receipt["rows"] for receipt in typed_receipts) != 2511:
            raise RuntimeError("Original true1M aggregate inference coverage differs")
        _write_receipt_table(control_root / "receipts.tsv", typed_receipts)
        _write_json_atomic(
            control_root / "original-true1m-inference-complete.json",
            _inference_completion_receipt(
                contract_path=contract_path,
                contract=contract,
                launches=launches,
                receipts=typed_receipts,
            ),
        )
        (control_root / "failed.json").unlink(missing_ok=True)
    except BaseException as exc:
        for _item, process in running:
            _terminate_process_group(process)
        for _item, process in running:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        _write_json_atomic(
            control_root / "failed.json",
            {
                "schema_version": "tgvf.original-raw-direct-inference-failure.v1",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        lock_handle.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=list(range(7)),
        help="Seven worker GPUs, optionally followed by one explicit spare GPU.",
    )
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the launch contract without querying or using GPUs.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    python_bin = _absolute_path_without_resolving_symlinks(args.python_bin)
    launches, spare_gpus = _materialize_launches(
        config_path=config_path,
        output_root=output_root,
        python_bin=python_bin,
        gpu_ids=tuple(args.gpu_ids),
    )
    contract = _contract_receipt(
        config_path=config_path,
        output_root=output_root,
        launches=launches,
        spare_gpus=spare_gpus,
    )
    if args.validate_only:
        print(json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if not python_bin.is_file():
        raise RuntimeError(f"Python environment is missing: {python_bin}")
    _run_inference(
        output_root=output_root,
        launches=launches,
        contract=contract,
        wait_for_gpus=args.wait_for_gpus,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
