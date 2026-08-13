#!/usr/bin/env python3
"""Validate or launch one arm of the paired texture benchmark matrix."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.texture_bench.io import (  # noqa: E402
    write_json_idempotent,
    write_jsonl_idempotent,
)
from tgvf_rl.evaluation.texture_bench.schema import (  # noqa: E402
    PipelineArm,
    PipelineBackend,
    PipelineKind,
    TextureBenchmarkMatrix,
    canonical_json_sha256,
    load_texture_benchmark_matrix,
)
from tgvf_rl.evaluation.texture_bench.stock_qwen import (  # noqa: E402
    STOCK_QWEN_MM_ENCODER_ATTN_BACKEND,
    STOCK_QWEN_RESULT_SCHEMA,
    STOCK_QWEN_SEED_NAMESPACE,
    STOCK_QWEN_VISION_IDENTITY_SCHEMA,
    StockQwenVLLMRunner,
    stable_stock_qwen_seed,
)
from tgvf_rl.evaluation.texture_bench.task import (  # noqa: E402
    TextureTask,
    load_texture_tasks,
)


TEXTURE_RUN_IDENTITY_SCHEMA = "tgvf-texture-benchmark-run-identity-v1"
TEXTURE_ORIGINAL_EXECUTION_IDENTITY_SCHEMA = (
    "tgvf-texture-original-execution-identity-v1"
)


def _arm(matrix: TextureBenchmarkMatrix, selector: str) -> PipelineArm:
    matches = tuple(
        arm
        for arm in matrix.arms
        if arm.arm_id == selector or arm.kind.value == selector
    )
    if len(matches) != 1:
        raise ValueError(f"pipeline arm selector is absent or ambiguous: {selector}")
    return matches[0]


def _regular_file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"model tree entry is not a regular file: {path}")
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def model_tree_identity(path: str | Path) -> dict[str, object]:
    """Hash every regular model file and reject symlinks/non-files."""

    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("stock model path must be a non-symlink directory")
    records: list[dict[str, object]] = []
    for candidate in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"stock model tree contains a symlink: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"stock model tree contains a non-file: {relative}")
        records.append(
            {
                "path": relative,
                "size_bytes": candidate.stat().st_size,
                "sha256": _regular_file_sha256(candidate),
            }
        )
    if not records:
        raise ValueError("stock model tree is empty")
    content = {"root": str(root), "files": records}
    return {
        "root": str(root),
        "file_count": len(records),
        "logical_bytes": sum(int(item["size_bytes"]) for item in records),
        "tree_sha256": canonical_json_sha256(content),
    }


def _original_gpu_ids(
    matrix: TextureBenchmarkMatrix,
    *,
    world_size: int,
    requested_gpu_ids: Sequence[int] | None,
) -> tuple[int, ...]:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("original world_size must be positive")
    gpu_ids = tuple(matrix.gpu_ids if requested_gpu_ids is None else requested_gpu_ids)
    if (
        len(gpu_ids) != world_size
        or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError(
            "original gpu_ids must contain world_size distinct non-negative IDs"
        )
    return gpu_ids


def _assert_original_worker_cuda_binding(
    *, rank: int, world_size: int, gpu_ids: Sequence[int]
) -> None:
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("original rank must lie in [0, world_size)")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("original worker requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    expected = str(gpu_ids[rank])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise RuntimeError(
            f"original rank {rank} requires CUDA_VISIBLE_DEVICES={expected} exactly"
        )


def _normalized_original_engine_kwargs(
    engine_kwargs: Mapping[str, object] | None,
) -> dict[str, object]:
    options = dict(engine_kwargs or {})
    forbidden = {
        "model",
        "trust_remote_code",
        "limit_mm_per_prompt",
        "mm_encoder_attn_backend",
        "tensor_parallel_size",
    }.intersection(options)
    if forbidden:
        raise ValueError(
            "stock Qwen engine kwargs override owned fields: "
            + ", ".join(sorted(forbidden))
        )
    try:
        json.dumps(options, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "original engine kwargs must be canonical JSON values"
        ) from error
    return options


def _original_execution_identity(
    matrix: TextureBenchmarkMatrix,
    arm: PipelineArm,
    *,
    model_identity: Mapping[str, object],
    batch_size: int,
    max_tokens: int,
    engine_kwargs: Mapping[str, object] | None,
    world_size: int,
    gpu_ids: Sequence[int],
) -> dict[str, object]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("original batch_size must be positive")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("original max_tokens must be positive")
    options = _normalized_original_engine_kwargs(engine_kwargs)
    content: dict[str, object] = {
        "schema_version": TEXTURE_ORIGINAL_EXECUTION_IDENTITY_SCHEMA,
        "matrix_identity_sha256": matrix.identity_sha256,
        "arm": arm.identity_payload(),
        "model_tree": dict(model_identity),
        "vision": asdict(matrix.vision),
        "vision_identity_sha256": matrix.vision.identity_sha256,
        "task_manifest": {
            "path": str(matrix.task_manifest_path),
            "sha256": matrix.task_manifest_sha256,
            "task_count": matrix.task_count,
        },
        "execution": {
            "world_size": world_size,
            "gpu_ids": list(gpu_ids),
            "tensor_parallel_size_per_worker": 1,
        },
        "generation": {
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed_namespace": STOCK_QWEN_SEED_NAMESPACE,
            "seed_base": 0,
            "per_sample_content_addressed_seed": True,
            "mm_encoder_attn_backend": STOCK_QWEN_MM_ENCODER_ATTN_BACKEND,
            "engine_kwargs": options,
        },
    }
    return {**content, "identity_sha256": canonical_json_sha256(content)}


def _original_output_root(matrix: TextureBenchmarkMatrix, arm: PipelineArm) -> Path:
    return matrix.output_root / arm.arm_id


def _execution_identity_path(output_root: Path) -> Path:
    return output_root / "runtime" / "original-execution-identity.json"


def _assert_existing_execution_identity(
    output_root: Path, expected: Mapping[str, object]
) -> bool:
    path = _execution_identity_path(output_root)
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"original execution identity is not a file: {path}")
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("original execution identity is unreadable") from error
    if observed != dict(expected):
        raise RuntimeError("original execution identity differs")
    return True


def _append_durable(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - operating-system failure guard
                raise OSError("short write while appending original result")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _original_rank_lock(output_root: Path, rank: int):
    lock_path = output_root / "runtime" / "locks" / f"rank-{rank}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"original rank {rank} is already active") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _bind_durable_original_result(
    source: Mapping[str, object],
    *,
    task: TextureTask,
    matrix: TextureBenchmarkMatrix,
    arm: PipelineArm,
    execution_identity: Mapping[str, object],
    rank: int,
    world_size: int,
    gpu_ids: Sequence[int],
    wall_seconds: float,
) -> dict[str, object]:
    row = dict(source)
    if "result_identity_sha256" in row:
        raise RuntimeError("stock Qwen runner emitted an owned result digest")
    if row.get("ordinal") != task.ordinal:
        raise RuntimeError("stock Qwen result ordinal differs")
    if row.get("sample_id") != task.bound_sample_id:
        raise RuntimeError("stock Qwen result sample identity differs")
    owned: dict[str, object] = {
        "task_manifest_sha256": matrix.task_manifest_sha256,
        "matrix_identity_sha256": matrix.identity_sha256,
        "arm_id": arm.arm_id,
        "original_execution_identity_sha256": execution_identity["identity_sha256"],
        "rank": rank,
        "world_size": world_size,
        "gpu_ids": list(gpu_ids),
        "wall_seconds": wall_seconds,
    }
    for name, value in owned.items():
        if name in row and row[name] != value:
            raise RuntimeError(f"stock Qwen result {name} differs")
        row[name] = value
    row["result_identity_sha256"] = canonical_json_sha256(row)
    validate_durable_original_result(
        row,
        task=task,
        matrix=matrix,
        arm=arm,
        execution_identity=execution_identity,
        rank=rank,
        world_size=world_size,
        gpu_ids=gpu_ids,
    )
    return row


def validate_durable_original_result(
    payload: Mapping[str, object],
    *,
    task: TextureTask,
    matrix: TextureBenchmarkMatrix,
    arm: PipelineArm,
    execution_identity: Mapping[str, object],
    rank: int,
    world_size: int,
    gpu_ids: Sequence[int],
) -> None:
    """Validate all immutable bindings used to resume one stock-Qwen row."""

    declared_digest = payload.get("result_identity_sha256")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise RuntimeError("original result identity digest is malformed")
    hash_payload = dict(payload)
    hash_payload.pop("result_identity_sha256", None)
    if canonical_json_sha256(hash_payload) != declared_digest:
        raise RuntimeError("original result identity digest differs")
    expected: dict[str, object] = {
        "schema_version": STOCK_QWEN_RESULT_SCHEMA,
        "ordinal": task.ordinal,
        "sample_id": task.bound_sample_id,
        "index": task.index,
        "dataset": task.dataset,
        "task_manifest_sha256": matrix.task_manifest_sha256,
        "matrix_identity_sha256": matrix.identity_sha256,
        "arm_id": arm.arm_id,
        "original_execution_identity_sha256": execution_identity["identity_sha256"],
        "rank": rank,
        "world_size": world_size,
        "gpu_ids": list(gpu_ids),
        "request_seed": stable_stock_qwen_seed(task),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"original result {name} differs")
    if task.ordinal % world_size != rank:
        raise RuntimeError("original result is stored under the wrong rank")
    if not isinstance(payload.get("final_answer"), str):
        raise RuntimeError("original result final_answer is malformed")
    model_response = payload.get("model_response")
    if not isinstance(model_response, Mapping):
        raise RuntimeError("original result model_response is malformed")
    if model_response.get("text") != payload["final_answer"]:
        raise RuntimeError("original result model response text differs")
    wall_seconds = payload.get("wall_seconds")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or wall_seconds < 0
    ):
        raise RuntimeError("original result wall_seconds is malformed")

    vision = payload.get("vision_identity")
    if not isinstance(vision, Mapping):
        raise RuntimeError("original result vision identity is malformed")
    vision_content: dict[str, object] = {
        "schema_version": STOCK_QWEN_VISION_IDENTITY_SCHEMA,
        "source_path": task.image_paths[0],
        "source_image_sha256": task.image_sha256s[0],
        "source_dimensions": list(task.image_dimensions[0]),
        "preprocess": asdict(matrix.vision),
        "preprocess_identity_sha256": matrix.vision.identity_sha256,
    }
    expected_vision = {
        **vision_content,
        "identity_sha256": canonical_json_sha256(vision_content),
    }
    if dict(vision) != expected_vision:
        raise RuntimeError("original result vision identity differs")


def load_durable_original_results(
    output_root: str | Path,
    *,
    tasks: Sequence[TextureTask],
    matrix: TextureBenchmarkMatrix,
    arm: PipelineArm,
    execution_identity: Mapping[str, object],
    require_complete: bool = False,
) -> dict[int, dict[str, object]]:
    """Load rank JSONLs and reject duplicates, corruption, or resume drift."""

    root = Path(output_root)
    execution = execution_identity.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("original execution identity is malformed")
    world_size = execution.get("world_size")
    raw_gpu_ids = execution.get("gpu_ids")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("original execution world_size is malformed")
    if not isinstance(raw_gpu_ids, list):
        raise ValueError("original execution gpu_ids are malformed")
    gpu_ids = tuple(raw_gpu_ids)
    _original_gpu_ids(matrix, world_size=world_size, requested_gpu_ids=gpu_ids)
    task_by_ordinal = {task.ordinal: task for task in tasks}
    records: dict[int, dict[str, object]] = {}
    inference_root = root / "inference"
    expected_paths = {
        inference_root / f"rank-{rank}.jsonl" for rank in range(world_size)
    }
    observed_paths = set(inference_root.glob("rank-*.jsonl"))
    unexpected_paths = observed_paths.difference(expected_paths)
    if unexpected_paths:
        raise RuntimeError(
            "unexpected original rank result: "
            + ", ".join(str(path) for path in sorted(unexpected_paths))
        )
    for rank in range(world_size):
        path = inference_root / f"rank-{rank}.jsonl"
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"original result is not a regular file: {path}")
        try:
            with path.open(encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                try:
                    lines = handle.read().splitlines()
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, UnicodeDecodeError) as error:
            raise RuntimeError(f"cannot read original result: {path}") from error
        for line_number, line in enumerate(lines, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid original result at {path}:{line_number}"
                ) from error
            if not isinstance(raw, dict):
                raise RuntimeError(
                    f"original result is not an object at {path}:{line_number}"
                )
            ordinal = raw.get("ordinal")
            if type(ordinal) is not int or ordinal in records:
                raise RuntimeError(
                    f"duplicate/invalid original ordinal at {path}:{line_number}"
                )
            task = task_by_ordinal.get(ordinal)
            if task is None:
                raise RuntimeError("original result is outside its task manifest")
            validate_durable_original_result(
                raw,
                task=task,
                matrix=matrix,
                arm=arm,
                execution_identity=execution_identity,
                rank=rank,
                world_size=world_size,
                gpu_ids=gpu_ids,
            )
            records[ordinal] = raw
    expected = set(task_by_ordinal)
    if require_complete and set(records) != expected:
        missing = sorted(expected.difference(records))
        extra = sorted(set(records).difference(expected))
        raise RuntimeError(
            "original results are incomplete "
            f"(missing={missing[:8]}, extra={extra[:8]})"
        )
    return records


def validate_matrix(
    matrix_path: str | Path,
    *,
    verify_images: bool = True,
    require_complete_arms: bool = False,
) -> dict[str, object]:
    matrix = load_texture_benchmark_matrix(matrix_path)
    if require_complete_arms:
        matrix.require_complete_arms()
    matrix.validate_files(verify_manifest_hash=True)
    tasks = load_texture_tasks(
        matrix.task_manifest_path,
        expected_count=matrix.task_count,
        expected_sha256=matrix.task_manifest_sha256,
        verify_images=verify_images,
    )
    return {
        "matrix_id": matrix.matrix_id,
        "matrix_identity_sha256": matrix.identity_sha256,
        "task_count": len(tasks),
        "task_manifest_sha256": matrix.task_manifest_sha256,
        "gpu_ids": list(matrix.gpu_ids),
        "complete_four_arm_matrix": matrix.complete_four_arm_matrix,
        "missing_pipeline_kinds": [
            kind.value for kind in matrix.missing_pipeline_kinds
        ],
        "vision": {
            "min_pixels": matrix.vision.min_pixels,
            "max_pixels": matrix.vision.max_pixels,
            "preserve_aspect_ratio": matrix.vision.preserve_aspect_ratio,
            "pre_resize_assets": matrix.vision.pre_resize_assets,
        },
        "arms": [arm.identity_payload() for arm in matrix.arms],
        "resolved_arms": list(matrix.resolved_arm_identities()),
        "images_verified": verify_images,
    }


def run_original(
    matrix_path: str | Path,
    *,
    batch_size: int = 8,
    max_tokens: int = 2048,
    engine_kwargs: Mapping[str, object] | None = None,
    verify_images: bool = True,
    runner_type: type[StockQwenVLLMRunner] = StockQwenVLLMRunner,
) -> dict[str, object]:
    """Run stock Qwen on the exact paired manifest and write immutable results."""

    options = _normalized_original_engine_kwargs(engine_kwargs)
    matrix = load_texture_benchmark_matrix(matrix_path)
    matrix.validate_files(verify_manifest_hash=True)
    arm = _arm(matrix, PipelineKind.ORIGINAL.value)
    if arm.backend is not PipelineBackend.STOCK_QWEN_VLLM:
        raise ValueError("original arm does not use the stock Qwen backend")
    assert arm.model_path is not None
    tasks = load_texture_tasks(
        matrix.task_manifest_path,
        expected_count=matrix.task_count,
        expected_sha256=matrix.task_manifest_sha256,
        verify_images=verify_images,
    )
    model_identity = model_tree_identity(arm.model_path)
    runner = runner_type(
        model_path=arm.model_path,
        vision=matrix.vision,
        batch_size=batch_size,
        max_tokens=max_tokens,
        engine_kwargs=options,
    )
    raw_rows = runner.run(tasks)
    task_by_ordinal = {task.ordinal: task for task in tasks}
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for source in raw_rows:
        row = dict(source)
        ordinal = row.get("ordinal")
        if (
            type(ordinal) is not int
            or ordinal in seen
            or ordinal not in task_by_ordinal
        ):
            raise RuntimeError("stock Qwen result ordinal is invalid or duplicated")
        seen.add(ordinal)
        task = task_by_ordinal[ordinal]
        if row.get("sample_id") != task.bound_sample_id:
            raise RuntimeError("stock Qwen result sample identity differs")
        owned = {
            "task_manifest_sha256": matrix.task_manifest_sha256,
            "matrix_identity_sha256": matrix.identity_sha256,
            "arm_id": arm.arm_id,
        }
        for name, value in owned.items():
            if name in row and row[name] != value:
                raise RuntimeError(f"stock Qwen result {name} differs")
            row[name] = value
        rows.append(row)
    if seen != set(task_by_ordinal):
        raise RuntimeError("stock Qwen result set is incomplete")
    output_root = matrix.output_root / arm.arm_id
    result_artifact = write_jsonl_idempotent(output_root / "results.jsonl", rows)
    content: dict[str, object] = {
        "schema_version": TEXTURE_RUN_IDENTITY_SCHEMA,
        "matrix_identity_sha256": matrix.identity_sha256,
        "arm": arm.identity_payload(),
        "model_tree": model_identity,
        "vision_identity_sha256": matrix.vision.identity_sha256,
        "task_manifest": {
            "path": str(matrix.task_manifest_path),
            "sha256": matrix.task_manifest_sha256,
            "task_count": matrix.task_count,
        },
        "results": result_artifact,
        "generation": {
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "per_sample_content_addressed_seed": True,
            "mm_encoder_attn_backend": STOCK_QWEN_MM_ENCODER_ATTN_BACKEND,
            "engine_kwargs": options,
        },
    }
    identity = {**content, "identity_sha256": canonical_json_sha256(content)}
    write_json_idempotent(output_root / "run-identity.json", identity)
    return identity


def _durable_original_context(
    matrix_path: str | Path,
    *,
    batch_size: int,
    max_tokens: int,
    engine_kwargs: Mapping[str, object] | None,
    world_size: int,
    requested_gpu_ids: Sequence[int] | None,
    verify_images: bool,
) -> tuple[
    TextureBenchmarkMatrix,
    PipelineArm,
    tuple[TextureTask, ...],
    dict[str, object],
    tuple[int, ...],
]:
    options = _normalized_original_engine_kwargs(engine_kwargs)
    matrix = load_texture_benchmark_matrix(matrix_path)
    matrix.validate_files(verify_manifest_hash=True)
    arm = _arm(matrix, PipelineKind.ORIGINAL.value)
    if arm.backend is not PipelineBackend.STOCK_QWEN_VLLM:
        raise ValueError("original arm does not use the stock Qwen backend")
    assert arm.model_path is not None
    gpu_ids = _original_gpu_ids(
        matrix,
        world_size=world_size,
        requested_gpu_ids=requested_gpu_ids,
    )
    tasks = load_texture_tasks(
        matrix.task_manifest_path,
        expected_count=matrix.task_count,
        expected_sha256=matrix.task_manifest_sha256,
        verify_images=verify_images,
    )
    model_identity = model_tree_identity(arm.model_path)
    execution_identity = _original_execution_identity(
        matrix,
        arm,
        model_identity=model_identity,
        batch_size=batch_size,
        max_tokens=max_tokens,
        engine_kwargs=options,
        world_size=world_size,
        gpu_ids=gpu_ids,
    )
    return matrix, arm, tasks, execution_identity, gpu_ids


def run_original_worker(
    matrix_path: str | Path,
    *,
    rank: int,
    world_size: int,
    gpu_ids: Sequence[int] | None = None,
    batch_size: int = 8,
    max_tokens: int = 2048,
    max_tasks: int = -1,
    engine_kwargs: Mapping[str, object] | None = None,
    verify_images: bool = True,
    runner_type: type[StockQwenVLLMRunner] = StockQwenVLLMRunner,
) -> dict[str, object]:
    """Run one durable stock-Qwen rank and resume verified prior rows."""

    if type(max_tasks) is not int or max_tasks < -1:
        raise ValueError("original max_tasks must be -1 or non-negative")
    matrix = load_texture_benchmark_matrix(matrix_path)
    resolved_gpu_ids = _original_gpu_ids(
        matrix, world_size=world_size, requested_gpu_ids=gpu_ids
    )
    _assert_original_worker_cuda_binding(
        rank=rank, world_size=world_size, gpu_ids=resolved_gpu_ids
    )
    arm = _arm(matrix, PipelineKind.ORIGINAL.value)
    output_root = _original_output_root(matrix, arm)
    with _original_rank_lock(output_root, rank):
        (
            matrix,
            arm,
            tasks,
            execution_identity,
            resolved_gpu_ids,
        ) = _durable_original_context(
            matrix_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            engine_kwargs=engine_kwargs,
            world_size=world_size,
            requested_gpu_ids=resolved_gpu_ids,
            verify_images=verify_images,
        )
        output_root = _original_output_root(matrix, arm)
        write_json_idempotent(_execution_identity_path(output_root), execution_identity)
        records = load_durable_original_results(
            output_root,
            tasks=tasks,
            matrix=matrix,
            arm=arm,
            execution_identity=execution_identity,
        )
        selected = [
            task
            for task in tasks
            if task.ordinal % world_size == rank and task.ordinal not in records
        ]
        if max_tasks >= 0:
            selected = selected[:max_tasks]
        result_path = output_root / "inference" / f"rank-{rank}.jsonl"
        if not selected:
            return {
                "matrix_identity_sha256": matrix.identity_sha256,
                "arm_id": arm.arm_id,
                "original_execution_identity_sha256": execution_identity[
                    "identity_sha256"
                ],
                "rank": rank,
                "world_size": world_size,
                "gpu_ids": list(resolved_gpu_ids),
                "result_path": str(result_path.resolve()),
                "completed_this_run": 0,
                "completed": len(records),
                "total": len(tasks),
                "remaining": len(tasks) - len(records),
            }

        assert arm.model_path is not None
        options = _normalized_original_engine_kwargs(engine_kwargs)
        runner = runner_type(
            model_path=arm.model_path,
            vision=matrix.vision,
            batch_size=batch_size,
            max_tokens=max_tokens,
            engine_kwargs=options,
        )
        completed_this_run = 0
        for batch_start in range(0, len(selected), batch_size):
            batch = selected[batch_start : batch_start + batch_size]
            batch_started = time.time()
            raw_rows = runner.run(batch)
            raw_by_ordinal: dict[int, Mapping[str, object]] = {}
            for raw in raw_rows:
                if not isinstance(raw, Mapping):
                    raise RuntimeError("stock Qwen runner returned a non-object")
                ordinal = raw.get("ordinal")
                if type(ordinal) is not int or ordinal in raw_by_ordinal:
                    raise RuntimeError(
                        "stock Qwen batch result ordinal is invalid or duplicated"
                    )
                raw_by_ordinal[ordinal] = raw
            expected_ordinals = {task.ordinal for task in batch}
            if set(raw_by_ordinal) != expected_ordinals:
                raise RuntimeError("stock Qwen batch result set is incomplete")
            wall_seconds = time.time() - batch_started
            durable_rows = [
                _bind_durable_original_result(
                    raw_by_ordinal[task.ordinal],
                    task=task,
                    matrix=matrix,
                    arm=arm,
                    execution_identity=execution_identity,
                    rank=rank,
                    world_size=world_size,
                    gpu_ids=resolved_gpu_ids,
                    wall_seconds=wall_seconds,
                )
                for task in batch
            ]
            for row in durable_rows:
                _append_durable(result_path, row)
                completed_this_run += 1
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "done": completed_this_run,
                            "selected": len(selected),
                            "ordinal": row["ordinal"],
                            "dataset": row["dataset"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        completed_records = load_durable_original_results(
            output_root,
            tasks=tasks,
            matrix=matrix,
            arm=arm,
            execution_identity=execution_identity,
        )
        completed = len(completed_records)
        return {
            "matrix_identity_sha256": matrix.identity_sha256,
            "arm_id": arm.arm_id,
            "original_execution_identity_sha256": execution_identity["identity_sha256"],
            "rank": rank,
            "world_size": world_size,
            "gpu_ids": list(resolved_gpu_ids),
            "result_path": str(result_path.resolve()),
            "completed_this_run": completed_this_run,
            "completed": completed,
            "total": len(tasks),
            "remaining": len(tasks) - completed,
        }


def original_status(
    matrix_path: str | Path,
    *,
    world_size: int,
    gpu_ids: Sequence[int] | None = None,
    batch_size: int = 8,
    max_tokens: int = 2048,
    engine_kwargs: Mapping[str, object] | None = None,
    verify_images: bool = True,
) -> dict[str, object]:
    """Validate every durable row and report exact per-rank coverage."""

    matrix, arm, tasks, execution_identity, resolved_gpu_ids = (
        _durable_original_context(
            matrix_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            engine_kwargs=engine_kwargs,
            world_size=world_size,
            requested_gpu_ids=gpu_ids,
            verify_images=verify_images,
        )
    )
    output_root = _original_output_root(matrix, arm)
    identity_present = _assert_existing_execution_identity(
        output_root, execution_identity
    )
    records = load_durable_original_results(
        output_root,
        tasks=tasks,
        matrix=matrix,
        arm=arm,
        execution_identity=execution_identity,
    )
    if records and not identity_present:
        raise RuntimeError("original rows exist without their execution identity")
    per_rank = [
        sum(ordinal % world_size == rank for ordinal in records)
        for rank in range(world_size)
    ]
    completed = len(records)
    return {
        "matrix_identity_sha256": matrix.identity_sha256,
        "arm_id": arm.arm_id,
        "original_execution_identity_sha256": execution_identity["identity_sha256"],
        "execution_identity_present": identity_present,
        "world_size": world_size,
        "gpu_ids": list(resolved_gpu_ids),
        "per_rank_completed": per_rank,
        "completed": completed,
        "total": len(tasks),
        "remaining": len(tasks) - completed,
        "complete": completed == len(tasks),
    }


def finalize_original(
    matrix_path: str | Path,
    *,
    world_size: int,
    gpu_ids: Sequence[int] | None = None,
    batch_size: int = 8,
    max_tokens: int = 2048,
    engine_kwargs: Mapping[str, object] | None = None,
    verify_images: bool = True,
) -> dict[str, object]:
    """Validate complete rank coverage and publish immutable merged artifacts."""

    matrix, arm, tasks, execution_identity, resolved_gpu_ids = (
        _durable_original_context(
            matrix_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            engine_kwargs=engine_kwargs,
            world_size=world_size,
            requested_gpu_ids=gpu_ids,
            verify_images=verify_images,
        )
    )
    output_root = _original_output_root(matrix, arm)
    if not _assert_existing_execution_identity(output_root, execution_identity):
        raise RuntimeError("original execution identity does not exist")
    records = load_durable_original_results(
        output_root,
        tasks=tasks,
        matrix=matrix,
        arm=arm,
        execution_identity=execution_identity,
        require_complete=True,
    )
    ordered_rows = [records[task.ordinal] for task in tasks]
    result_artifact = write_jsonl_idempotent(
        output_root / "results.jsonl", ordered_rows
    )
    content: dict[str, object] = {
        "schema_version": TEXTURE_RUN_IDENTITY_SCHEMA,
        "matrix_identity_sha256": matrix.identity_sha256,
        "arm": arm.identity_payload(),
        "model_tree": execution_identity["model_tree"],
        "vision_identity_sha256": matrix.vision.identity_sha256,
        "task_manifest": {
            "path": str(matrix.task_manifest_path),
            "sha256": matrix.task_manifest_sha256,
            "task_count": matrix.task_count,
        },
        "results": result_artifact,
        "generation": execution_identity["generation"],
        "durable_execution": {
            "identity_sha256": execution_identity["identity_sha256"],
            "identity_path": str(_execution_identity_path(output_root).resolve()),
            "world_size": world_size,
            "gpu_ids": list(resolved_gpu_ids),
            "assignment": "ordinal_mod_world_size",
        },
    }
    identity = {**content, "identity_sha256": canonical_json_sha256(content)}
    write_json_idempotent(output_root / "run-identity.json", identity)
    return identity


def policy_commands(matrix_path: str | Path, *, arm_selector: str) -> dict[str, object]:
    """Return argv-safe prepare/validate/worker commands for one policy arm."""

    matrix = load_texture_benchmark_matrix(matrix_path)
    matrix.validate_files(verify_manifest_hash=True)
    arm = _arm(matrix, arm_selector)
    if arm.backend is not PipelineBackend.POLICY_BENCHMARK:
        raise ValueError("policy-command requires a policy benchmark arm")
    assert arm.policy_config_path is not None
    assert arm.expected_optimizer_step is not None
    output_root = matrix.output_root / arm.arm_id
    evaluation_root = output_root / "evaluation"
    config_path = output_root / "policy-benchmark-config.json"
    common = [
        "--evaluation-id",
        f"{matrix.matrix_id}-{arm.arm_id}",
        "--policy-config",
        str(arm.policy_config_path),
        "--expected-optimizer-step",
        str(arm.expected_optimizer_step),
        "--tasks",
        str(matrix.task_manifest_path),
        "--expected-task-count",
        str(matrix.task_count),
        "--expected-single-image-count",
        str(matrix.task_count),
        "--output-root",
        str(evaluation_root),
        "--config-output",
        str(config_path),
        "--image-max-pixels",
        str(matrix.vision.max_pixels),
        "--gpu-ids",
        *(str(gpu_id) for gpu_id in matrix.gpu_ids),
    ]
    if arm.paired_qwen_model_path is not None:
        assert arm.paired_snapshot_receipt_path is not None
        paired = [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "tools/materialize_paired_tgvf_policy_benchmark_config.py"
            ),
            "--evaluation-id",
            f"{matrix.matrix_id}-{arm.arm_id}",
            "--policy-config",
            str(arm.policy_config_path),
            "--optimizer-step",
            str(arm.expected_optimizer_step),
            "--qwen-model",
            str(arm.paired_qwen_model_path),
            "--snapshot-receipt",
            str(arm.paired_snapshot_receipt_path),
            "--task-manifest",
            str(matrix.task_manifest_path),
            "--expected-task-count",
            str(matrix.task_count),
            "--expected-single-image-count",
            str(matrix.task_count),
            "--output-root",
            str(evaluation_root),
            "--config",
            str(config_path),
            "--image-max-pixels",
            str(matrix.vision.max_pixels),
            "--gpu-ids",
            *(str(gpu_id) for gpu_id in matrix.gpu_ids),
            "--paired-seed-namespace",
            matrix.task_manifest_sha256,
        ]
        if arm.paired_rp66_pointer_path is not None:
            paired.extend(("--rp66-pointer", str(arm.paired_rp66_pointer_path)))
        materialize = paired
    elif arm.lora_pointer_path is not None:
        materialize = [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/materialize_policy_benchmark_config.py"),
            *common,
            "--lora-pointer",
            str(arm.lora_pointer_path),
            "--evaluation-protocol",
            str(arm.evaluation_protocol),
        ]
    else:
        assert arm.full_model_snapshot_manifest_path is not None
        assert arm.full_model_materialization_receipt_path is not None
        materialize = [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "tools/materialize_full_model_policy_benchmark_config.py"
            ),
            *common,
            "--snapshot-manifest",
            str(arm.full_model_snapshot_manifest_path),
            "--materialization-receipt",
            str(arm.full_model_materialization_receipt_path),
        ]
    runner = str(REPOSITORY_ROOT / "tools/run_policy_benchmark.py")
    prepare = [
        sys.executable,
        runner,
        "--config",
        str(config_path),
        "--mode",
        "prepare",
    ]
    validate = [
        sys.executable,
        runner,
        "--config",
        str(config_path),
        "--mode",
        "validate",
        "--world-size",
        "4",
    ]
    workers = [
        {
            "environment": {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(matrix.gpu_ids[rank]),
            },
            "argv": [
                sys.executable,
                runner,
                "--config",
                str(config_path),
                "--mode",
                "worker",
                "--rank",
                str(rank),
                "--world-size",
                "4",
            ],
        }
        for rank in range(4)
    ]
    return {
        "matrix_identity_sha256": matrix.identity_sha256,
        "arm": arm.identity_payload(),
        "config_path": str(config_path),
        "materialize": materialize,
        "prepare": prepare,
        "validate": validate,
        "workers_run_concurrently": workers,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--matrix", type=Path, required=True)
    validate.add_argument(
        "--verify-images", action=argparse.BooleanOptionalAction, default=True
    )
    validate.add_argument(
        "--require-complete-arms",
        action="store_true",
        help="Fail unless original, crop, tgvf, and tgvf_crop are all present.",
    )
    original = subparsers.add_parser("original")
    original.add_argument("--matrix", type=Path, required=True)
    original.add_argument("--batch-size", type=int, default=8)
    original.add_argument("--max-tokens", type=int, default=2048)
    original.add_argument("--engine-kwargs-json", default="{}")
    original.add_argument("--rank", type=int)
    original.add_argument("--world-size", type=int)
    original.add_argument("--gpu-ids", type=int, nargs="+")
    original.add_argument("--max-tasks", type=int, default=-1)
    original.add_argument(
        "--verify-images", action=argparse.BooleanOptionalAction, default=True
    )
    for command in ("original-status", "original-finalize"):
        durable = subparsers.add_parser(command)
        durable.add_argument("--matrix", type=Path, required=True)
        durable.add_argument("--world-size", type=int, required=True)
        durable.add_argument("--gpu-ids", type=int, nargs="+")
        durable.add_argument("--batch-size", type=int, default=8)
        durable.add_argument("--max-tokens", type=int, default=2048)
        durable.add_argument("--engine-kwargs-json", default="{}")
        durable.add_argument(
            "--verify-images", action=argparse.BooleanOptionalAction, default=True
        )
    policy = subparsers.add_parser("policy-command")
    policy.add_argument("--matrix", type=Path, required=True)
    policy.add_argument("--arm", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        output = validate_matrix(
            args.matrix,
            verify_images=args.verify_images,
            require_complete_arms=args.require_complete_arms,
        )
    elif args.command == "original":
        kwargs = json.loads(args.engine_kwargs_json)
        if not isinstance(kwargs, dict):
            raise ValueError("engine kwargs JSON must be an object")
        if (args.rank is None) != (args.world_size is None):
            raise ValueError("original --rank and --world-size must be used together")
        if args.rank is None:
            if args.gpu_ids is not None or args.max_tasks != -1:
                raise ValueError(
                    "original --gpu-ids/--max-tasks require sharded rank mode"
                )
            output = run_original(
                args.matrix,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                engine_kwargs=kwargs,
                verify_images=args.verify_images,
            )
        else:
            output = run_original_worker(
                args.matrix,
                rank=args.rank,
                world_size=args.world_size,
                gpu_ids=args.gpu_ids,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                max_tasks=args.max_tasks,
                engine_kwargs=kwargs,
                verify_images=args.verify_images,
            )
    elif args.command in {"original-status", "original-finalize"}:
        kwargs = json.loads(args.engine_kwargs_json)
        if not isinstance(kwargs, dict):
            raise ValueError("engine kwargs JSON must be an object")
        function = (
            original_status if args.command == "original-status" else finalize_original
        )
        output = function(
            args.matrix,
            world_size=args.world_size,
            gpu_ids=args.gpu_ids,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            engine_kwargs=kwargs,
            verify_images=args.verify_images,
        )
    else:
        output = policy_commands(args.matrix, arm_selector=args.arm)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
