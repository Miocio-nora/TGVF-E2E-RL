"""Immutable task and result artifacts for standalone policy benchmarks.

This leaf owns both sides of the durable benchmark protocol: bound task/image
manifests entering evaluation and identity-checked trajectory/result records
leaving it. It deliberately has no dependency on evaluator or policy-backend
implementations.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
import csv
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from types import FunctionType
from typing import Mapping, Sequence
from uuid import uuid4

from PIL import Image
import torch

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.trajectories.schema import (
    CropTGVFToolCallRecord,
    CropToolCallRecord,
    ToolCallRecord,
    TrajectoryIdentity,
    TrajectoryRecord,
    trajectory_checksum,
)

from .policy_evaluation_config import (
    FULL_MODEL_EVALUATION_BACKEND,
    LORA_ADAPTER_EVALUATION_BACKEND,
    POLICY_COREDEV_SCHEMA,
    PolicyCoreDevConfig,
    _LEGACY_COREDEV_TASK_COUNT,
    _require_sha256,
)
from .policy_evaluation_identity import (
    canonical_json_sha256 as _canonical_json_sha256,
    policy_benchmark_task_path,
    sha256_file as _sha256_file,
)


POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA = (
    "tgvf-policy-coredev-trajectory-audit-v1"
)


@dataclass(frozen=True, slots=True)
class CoreDevTask:
    ordinal: int
    dataset: str
    row_number: int
    index: str
    question: str
    image_paths: tuple[str, ...]
    sample_id: str | None = None
    answer: str | None = None
    options: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    image_sha256s: tuple[str, ...] = ()
    image_dimensions: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_paths", tuple(self.image_paths))
        raw_options = self.options
        if isinstance(raw_options, Mapping):
            normalized_options = tuple(
                (str(key), str(value)) for key, value in raw_options.items()
            )
        else:
            normalized_options = tuple(tuple(item) for item in raw_options)
        object.__setattr__(self, "options", normalized_options)
        raw_metadata = self.metadata
        if isinstance(raw_metadata, Mapping):
            normalized_metadata = tuple(
                (str(key), str(value)) for key, value in raw_metadata.items()
            )
        else:
            normalized_metadata = tuple(tuple(item) for item in raw_metadata)
        object.__setattr__(self, "metadata", normalized_metadata)
        object.__setattr__(self, "image_sha256s", tuple(self.image_sha256s))
        object.__setattr__(
            self,
            "image_dimensions",
            tuple(tuple(item) for item in self.image_dimensions),
        )
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("policy benchmark task ordinal must be non-negative")
        if type(self.row_number) is not int or self.row_number < 0:
            raise ValueError("policy benchmark row_number must be non-negative")
        for name, value in (
            ("dataset", self.dataset),
            ("index", self.index),
            ("question", self.question),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"policy benchmark task {name} must be non-empty")
        if not self.image_paths or any(
            not isinstance(path, str) or not path for path in self.image_paths
        ):
            raise ValueError("policy benchmark task must carry image_paths")
        if self.sample_id is not None and (
            not isinstance(self.sample_id, str) or not self.sample_id.strip()
        ):
            raise ValueError("policy benchmark task sample_id must be non-empty")
        if self.answer is not None and (
            not isinstance(self.answer, str) or not self.answer.strip()
        ):
            raise ValueError("policy benchmark task answer must be non-empty")
        option_names: set[str] = set()
        for item in self.options:
            if (
                len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or not isinstance(item[1], str)
                or not item[1]
                or item[0] in option_names
            ):
                raise ValueError("policy benchmark task options are malformed")
            option_names.add(item[0])
        if self.answer is not None and self.options and self.answer not in option_names:
            raise ValueError("policy benchmark task answer is absent from its options")
        metadata_names: set[str] = set()
        for item in self.metadata:
            if (
                len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or not isinstance(item[1], str)
                or item[0] in metadata_names
            ):
                raise ValueError("policy benchmark task metadata is malformed")
            metadata_names.add(item[0])
        image_identity_counts = (
            len(self.image_sha256s),
            len(self.image_dimensions),
        )
        if any(image_identity_counts) and image_identity_counts != (
            len(self.image_paths),
            len(self.image_paths),
        ):
            raise ValueError("policy benchmark task image identity counts differ")
        for digest in self.image_sha256s:
            _require_sha256(digest, name="task image SHA256")
        for dimensions in self.image_dimensions:
            if len(dimensions) != 2 or any(
                type(value) is not int or value <= 0 for value in dimensions
            ):
                raise ValueError(
                    "task image dimensions must be positive [width,height]"
                )

    @property
    def single_image(self) -> bool:
        return len(self.image_paths) == 1

    @property
    def bound_sample_id(self) -> str:
        """Return explicit generic identity or the legacy CoreDev fallback."""

        return self.sample_id or f"{self.dataset}:{self.index}"

    @property
    def has_bound_images(self) -> bool:
        return len(self.image_sha256s) == len(self.image_paths)


def _read_regular_file_bytes(path: Path, *, owner: str) -> bytes:
    if not path.is_absolute():
        raise ValueError(f"{owner} path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{owner} is missing or unreadable: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{owner} is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _read_bound_image_bytes(path: Path) -> bytes:
    return _read_regular_file_bytes(path, owner="benchmark image")


def _decode_rgb_bytes(
    payload: bytes, *, path: Path
) -> tuple[torch.Tensor, tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            dimensions = opened.size
            rgb = opened.convert("RGB")
            import numpy as np

            array = np.asarray(rgb, dtype=np.uint8).copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"benchmark image cannot be decoded: {path}") from error
    return torch.from_numpy(array), (int(dimensions[0]), int(dimensions[1]))


def image_file_identity(path: str | Path) -> tuple[str, tuple[int, int]]:
    """Hash and decode the same open-file byte snapshot."""

    resolved = Path(path)
    payload = _read_bound_image_bytes(resolved)
    _rgb, dimensions = _decode_rgb_bytes(payload, path=resolved)
    return hashlib.sha256(payload).hexdigest(), dimensions


def load_verified_task_image(task: CoreDevTask, image_index: int = 0) -> torch.Tensor:
    """Load one task image from bytes that match its bound hash and dimensions."""

    if not task.has_bound_images:
        raise ValueError("policy benchmark task has no bound image identities")
    if not 0 <= image_index < len(task.image_paths):
        raise IndexError("task image index is out of range")
    path = Path(task.image_paths[image_index])
    payload = _read_bound_image_bytes(path)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    expected_sha256 = task.image_sha256s[image_index]
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"benchmark image SHA256 changed for {task.bound_sample_id}: "
            f"expected {expected_sha256}, observed {actual_sha256}"
        )
    rgb, dimensions = _decode_rgb_bytes(payload, path=path)
    if dimensions != task.image_dimensions[image_index]:
        raise ValueError(
            f"benchmark image dimensions changed for {task.bound_sample_id}: "
            f"expected {task.image_dimensions[image_index]}, observed {dimensions}"
        )
    return rgb


def write_official_coredev_tasks(output_path: str | Path) -> dict[str, int]:
    """Materialize pinned TSV contents with their official dataset prompt text."""

    repository_root = Path(__file__).resolve().parents[3]
    pinned = json.loads(
        (
            repository_root / "configs/evaluation/coredev_2511_vlmevalkit_v1.json"
        ).read_text()
    )
    artifact_root = Path(pinned["artifact_root"])
    rows: list[dict[str, object]] = []
    sample_ids: set[str] = set()
    ordinal = 0
    counts = {"total": 0, "single_image": 0, "multi_image": 0}
    for slice_spec in pinned["slices"]:
        dataset_name = slice_spec["dataset"]
        tsv = artifact_root / f"{dataset_name}.tsv"
        if _sha256_file(tsv) != slice_spec["tsv_sha256"]:
            raise ValueError(f"pinned CoreDev TSV changed: {dataset_name}")
        with tsv.open(encoding="utf-8", newline="") as handle:
            source_rows = tuple(csv.DictReader(handle, delimiter="\t"))
        if len(source_rows) != slice_spec["sample_count"]:
            raise ValueError(f"pinned CoreDev row count changed: {dataset_name}")
        for row_number, source in enumerate(source_rows):
            images = _tsv_image_paths(source["image_path"])
            image_identities = tuple(image_file_identity(path) for path in images)
            text = _official_prompt_text(dataset_name, source)
            index = source["index"]
            if not text.strip() or not images:
                raise ValueError(
                    f"official prompt is incomplete: {dataset_name}/{index}"
                )
            if index in sample_ids:
                raise ValueError(
                    f"CoreDev sample index is not globally unique: {index}"
                )
            sample_ids.add(index)
            rows.append(
                {
                    "ordinal": ordinal,
                    "dataset": dataset_name,
                    "row_number": row_number,
                    "index": index,
                    # Generic benchmark manifests require an explicit stable
                    # identity.  The pinned CoreDev source indices are globally
                    # unique, so preserve the official index verbatim instead
                    # of inventing a second namespace.
                    "sample_id": index,
                    "question": text,
                    "image_paths": list(images),
                    "image_sha256s": [identity[0] for identity in image_identities],
                    "image_dimensions": [
                        list(identity[1]) for identity in image_identities
                    ],
                }
            )
            ordinal += 1
            counts["single_image" if len(images) == 1 else "multi_image"] += 1
    counts["total"] = ordinal
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return counts


def _tsv_image_paths(value: str) -> tuple[str, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = value
    paths = (
        tuple(str(item) for item in parsed)
        if isinstance(parsed, list)
        else (str(parsed),)
    )
    if not paths or any(not Path(path).is_file() for path in paths):
        raise ValueError("CoreDev TSV contains a missing image path")
    return paths


def _option_lines(source: Mapping[str, str]) -> str:
    rows = []
    for letter in "ABCDEFGHIJ":
        value = source.get(letter, "")
        if not value:
            break
        rows.append(f"{letter}. {value}")
    if len(rows) < 2:
        raise ValueError("CoreDev MCQ row has fewer than two contiguous choices")
    return "\n".join(rows)


def _official_prompt_text(dataset: str, source: Mapping[str, str]) -> str:
    question = source["question"]
    if dataset in {"VStarBench", "HRBench4K", "HRBench8K", "BLINK"}:
        return (
            f"Question: {question}\nOptions:\n{_option_lines(source)}\n"
            "Please select the correct answer from the options above. \n"
        )
    if dataset == "MMMU_Pro_10c":
        # VLMEvalKit interleaves image items at these markers.  The accepted
        # visual-tool prompt owns its one image placeholder separately.
        question = re.sub(r"<image\s+\d+>", "", question)
        return (
            f"Question: {question}\nOptions:\n{_option_lines(source)}\n"
            "Answer directly with the option letter from the given choices. "
        )
    if dataset in {"OCRBench_v2", "MathVista_MINI", "MathVerse_MINI"}:
        return question
    raise ValueError(f"unsupported CoreDev dataset: {dataset}")


def load_benchmark_tasks(
    path: str | Path,
    *,
    expected_task_count: int,
    expected_single_image_count: int | None,
    expected_sha256: str | None = None,
    verify_image_paths: bool = True,
    verify_image_contents: bool = True,
    require_explicit_sample_ids: bool = True,
    require_image_identities: bool = True,
) -> tuple[CoreDevTask, ...]:
    """Load an ordered task manifest and enforce its complete bound identity."""

    manifest_path = Path(path)
    manifest_bytes = _read_regular_file_bytes(
        manifest_path, owner="policy benchmark task manifest"
    )
    if (
        expected_sha256 is not None
        and hashlib.sha256(manifest_bytes).hexdigest() != expected_sha256
    ):
        raise ValueError("policy benchmark task manifest SHA256 differs")
    try:
        manifest_text = manifest_bytes.decode("utf-8")
        tasks = tuple(
            CoreDevTask(**json.loads(line))
            for line in manifest_text.splitlines()
            if line
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("policy benchmark task manifest is unreadable") from error
    if len(tasks) != expected_task_count:
        raise ValueError("policy benchmark task manifest count differs")
    if tuple(item.ordinal for item in tasks) != tuple(range(expected_task_count)):
        raise ValueError("policy benchmark task manifest order differs")
    if require_explicit_sample_ids:
        if any(task.sample_id is None for task in tasks):
            raise ValueError(
                "generic policy benchmark task manifest requires explicit sample_id"
            )
        if any(task.sample_id != task.index for task in tasks):
            raise ValueError(
                "generic policy benchmark task sample_id must equal its index"
            )
    if require_image_identities and any(not task.has_bound_images for task in tasks):
        raise ValueError(
            "policy benchmark task manifest requires image SHA256 and dimensions"
        )
    sample_ids = tuple(task.bound_sample_id for task in tasks)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("policy benchmark task manifest contains duplicate sample_id")
    if verify_image_paths:
        missing_or_relative = [
            image_path
            for task in tasks
            for image_path in task.image_paths
            if not Path(image_path).is_absolute() or not Path(image_path).is_file()
        ]
        if missing_or_relative:
            raise ValueError(
                "policy benchmark task manifest contains a relative or missing image_path"
            )
    if verify_image_contents:
        for task in tasks:
            if task.has_bound_images:
                for image_index in range(len(task.image_paths)):
                    load_verified_task_image(task, image_index)
    single_image_count = sum(task.single_image for task in tasks)
    if (
        expected_single_image_count is not None
        and single_image_count != expected_single_image_count
    ):
        raise ValueError("policy benchmark single-image task count differs")
    return tasks


def load_coredev_tasks(path: str | Path) -> tuple[CoreDevTask, ...]:
    """Backward-compatible loader for the historical 2,511-row suite."""

    return load_benchmark_tasks(
        path,
        expected_task_count=_LEGACY_COREDEV_TASK_COUNT,
        expected_single_image_count=None,
        verify_image_paths=False,
        verify_image_contents=False,
        require_explicit_sample_ids=False,
        require_image_identities=False,
    )


def prepare_policy_benchmark_tasks(config: PolicyCoreDevConfig) -> dict[str, int]:
    """Materialize the legacy suite or bind an immutable supplied task manifest."""

    target = policy_benchmark_task_path(config)
    if config.uses_legacy_coredev_manifest:
        counts = write_official_coredev_tasks(target)
    else:
        assert config.task_manifest_path is not None
        assert config.task_manifest_sha256 is not None
        # Validate the source before copying so a partial/corrupt manifest is
        # never admitted into a resumable evaluation directory.
        tasks = load_benchmark_tasks(
            config.task_manifest_path,
            expected_task_count=config.expected_task_count,
            expected_single_image_count=config.expected_single_image_count,
            expected_sha256=config.task_manifest_sha256,
            verify_image_contents=True,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            load_benchmark_tasks(
                target,
                expected_task_count=config.expected_task_count,
                expected_single_image_count=config.expected_single_image_count,
                expected_sha256=config.task_manifest_sha256,
                verify_image_contents=True,
            )
        else:
            source_bytes = _read_regular_file_bytes(
                config.task_manifest_path,
                owner="policy benchmark task manifest",
            )
            if hashlib.sha256(source_bytes).hexdigest() != config.task_manifest_sha256:
                raise ValueError("policy benchmark task manifest SHA256 changed")
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            try:
                with temporary.open("xb") as handle:
                    handle.write(source_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        counts = {
            "total": len(tasks),
            "single_image": sum(task.single_image for task in tasks),
            "multi_image": sum(not task.single_image for task in tasks),
        }
    load_benchmark_tasks(
        target,
        expected_task_count=config.expected_task_count,
        expected_single_image_count=config.expected_single_image_count,
        expected_sha256=(
            config.task_manifest_sha256
            if not config.uses_legacy_coredev_manifest
            else None
        ),
        require_explicit_sample_ids=not config.uses_legacy_coredev_manifest,
        require_image_identities=True,
        verify_image_contents=True,
    )
    return counts


def load_bound_policy_benchmark_tasks(
    config: PolicyCoreDevConfig,
) -> tuple[CoreDevTask, ...]:
    return load_benchmark_tasks(
        policy_benchmark_task_path(config),
        expected_task_count=config.expected_task_count,
        expected_single_image_count=config.expected_single_image_count,
        expected_sha256=(
            config.task_manifest_sha256
            if not config.uses_legacy_coredev_manifest
            else None
        ),
        require_explicit_sample_ids=not config.uses_legacy_coredev_manifest,
        require_image_identities=True,
        # Each task is rehashed from one open-file byte snapshot immediately
        # before inference. Avoid a redundant full-suite image read here.
        verify_image_contents=False,
    )


def trajectory_audit_payload(
    task: CoreDevTask,
    trajectory: TrajectoryRecord,
    *,
    evaluation_identity: Mapping[str, object],
    rank: int,
    world_size: int,
) -> dict[str, object]:
    def call_payload(call: object) -> dict[str, object]:
        common: dict[str, object]
        if isinstance(call, ToolCallRecord):
            common = {"target": call.target}
        elif isinstance(call, CropToolCallRecord):
            common = {"bbox_2d": list(call.bbox_2d), "label": call.label}
        elif isinstance(call, CropTGVFToolCallRecord):
            common = {"bbox_2d": list(call.bbox_2d), "target": call.target}
        else:
            raise TypeError("unsupported tool call record")
        return {
            "call_index": call.call_index,
            "assistant_turn_index": call.assistant_turn_index,
            "function_name": call.function_name,
            "raw_call_text": call.raw_call_text,
            **common,
        }

    identity_sha256 = evaluation_identity.get("identity_sha256")
    if not isinstance(identity_sha256, str):
        raise ValueError("evaluation identity SHA256 is missing")
    _require_sha256(identity_sha256, name="evaluation identity SHA256")
    execution = evaluation_identity.get("execution")
    policy_snapshot = evaluation_identity.get("policy_snapshot")
    task_manifest = evaluation_identity.get("task_manifest")
    model_identity = evaluation_identity.get("model_identity")
    if not all(
        isinstance(value, Mapping)
        for value in (execution, policy_snapshot, task_manifest, model_identity)
    ):
        raise ValueError("evaluation identity sub-bindings are malformed")
    assert isinstance(execution, Mapping)
    assert isinstance(policy_snapshot, Mapping)
    assert isinstance(task_manifest, Mapping)
    assert isinstance(model_identity, Mapping)
    if type(rank) is not int or type(world_size) is not int or world_size <= 0:
        raise ValueError("result rank/world_size identity is invalid")
    if execution.get("world_size") != world_size or not 0 <= rank < world_size:
        raise ValueError("result rank/world_size differs from evaluation identity")
    if task.ordinal % world_size != rank:
        raise ValueError("task ordinal is assigned to another evaluator rank")
    if asdict(trajectory.model) != dict(model_identity):
        raise ValueError("trajectory model differs from evaluation identity")
    if trajectory.behavior_policy != PolicyVersion(
        run_id=str(policy_snapshot.get("run_id")),
        optimizer_step=policy_snapshot.get("optimizer_step"),
        weights_sha256=str(policy_snapshot.get("weights_sha256")),
    ):
        raise ValueError("trajectory policy differs from evaluation identity")
    payload = {
        "schema_version": POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA,
        "selection_reasons": ["representative_rollout_zero"],
        "evaluation_identity_sha256": identity_sha256,
        "policy_run_identity_sha256": policy_snapshot["run_identity_sha256"],
        "policy_pointer_file_sha256": policy_snapshot["pointer_file_sha256"],
        "policy_manifest_file_sha256": policy_snapshot["manifest_file_sha256"],
        "policy_tensor_file_sha256": policy_snapshot["tensor_file_sha256"],
        "policy_config_identity_sha256": evaluation_identity[
            "policy_run_config_identity_sha256"
        ],
        "task_manifest_sha256": task_manifest["sha256"],
        "model_identity": dict(model_identity),
        "rank": rank,
        "world_size": world_size,
        "evaluation_id": trajectory.identity.run_id,
        "sample_id": trajectory.identity.sample_id,
        "group_uid": trajectory.identity.group_id,
        "rollout_index": trajectory.identity.rollout_index,
        "ordinal": task.ordinal,
        "dataset": task.dataset,
        "row_number": task.row_number,
        "index": task.index,
        "question": task.question,
        "image_paths": list(task.image_paths),
        "image_sha256s": list(task.image_sha256s),
        "image_dimensions": [list(item) for item in task.image_dimensions],
        "trajectory_id": trajectory.identity.canonical_id,
        "trajectory_sha256": trajectory_checksum(trajectory),
        "policy_run_id": trajectory.behavior_policy.run_id,
        "optimizer_step": trajectory.behavior_policy.optimizer_step,
        "policy_weights_sha256": trajectory.behavior_policy.weights_sha256,
        "stop": trajectory.stop.value,
        "final_answer": trajectory.final_answer,
        "assistant_turns": [
            {
                "turn_index": turn.turn_index,
                "raw_text": turn.raw_text,
                "sampled_token_count": len(turn.tokens.token_ids),
                "is_tool_call": turn.is_tool_call,
                "stop_reason": turn.stop_reason,
            }
            for turn in trajectory.assistant_turns
        ],
        "tool_calls": [call_payload(call) for call in trajectory.tool_calls],
        "tool_errors": [
            {
                "attempt_index": error.attempt_index,
                "assistant_turn_index": error.assistant_turn_index,
                "function_name": error.function_name,
                "code": error.code,
                "payload_json": error.payload_json,
                "recoverable": error.recoverable,
            }
            for error in trajectory.tool_errors
        ],
        "successful_observation_count": len(trajectory.observations),
    }
    payload["result_identity_sha256"] = _canonical_json_sha256(payload)
    return payload


def validate_policy_benchmark_result(
    payload: Mapping[str, object],
    *,
    task: CoreDevTask,
    evaluation_identity: Mapping[str, object],
    rank: int,
    world_size: int,
) -> None:
    """Validate the complete resume identity of one durable result row."""

    expected_hash = payload.get("result_identity_sha256")
    _require_sha256(expected_hash, name="result identity SHA256")
    hash_payload = dict(payload)
    hash_payload.pop("result_identity_sha256", None)
    hash_payload.pop("wall_seconds", None)
    if _canonical_json_sha256(hash_payload) != expected_hash:
        raise RuntimeError("policy benchmark result identity digest differs")
    policy_snapshot = evaluation_identity["policy_snapshot"]
    task_manifest = evaluation_identity["task_manifest"]
    snapshot_backend = policy_snapshot.get(
        "snapshot_backend", LORA_ADAPTER_EVALUATION_BACKEND
    )
    if snapshot_backend == FULL_MODEL_EVALUATION_BACKEND:
        snapshot_expected = {
            "policy_snapshot_backend": FULL_MODEL_EVALUATION_BACKEND,
            "policy_full_snapshot_identity_sha256": policy_snapshot[
                "snapshot_identity_sha256"
            ],
            "policy_checkpoint_sha256": policy_snapshot["checkpoint_sha256"],
            "policy_source_tree_sha256": policy_snapshot["source_tree_sha256"],
            "policy_materialization_identity_sha256": policy_snapshot[
                "materialization_identity_sha256"
            ],
            "policy_materialized_model_tree_sha256": policy_snapshot[
                "materialized_model_tree_sha256"
            ],
        }
    elif snapshot_backend in {LORA_ADAPTER_EVALUATION_BACKEND, "lora"}:
        snapshot_expected = {
            "policy_pointer_file_sha256": policy_snapshot["pointer_file_sha256"],
            "policy_manifest_file_sha256": policy_snapshot["manifest_file_sha256"],
            "policy_tensor_file_sha256": policy_snapshot["tensor_file_sha256"],
        }
        # The legacy trajectory writer predates an explicit backend field;
        # the official-visible writer emits its public audit spelling instead
        # of the internal config spelling ``lora_adapter``.
        if "policy_snapshot_backend" in payload:
            snapshot_expected["policy_snapshot_backend"] = "lora"
    else:
        raise RuntimeError("policy benchmark snapshot backend differs")
    expected = {
        "schema_version": POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA,
        "evaluation_identity_sha256": evaluation_identity["identity_sha256"],
        "policy_run_identity_sha256": policy_snapshot["run_identity_sha256"],
        **snapshot_expected,
        "policy_config_identity_sha256": evaluation_identity[
            "policy_run_config_identity_sha256"
        ],
        "task_manifest_sha256": task_manifest["sha256"],
        "model_identity": evaluation_identity["model_identity"],
        "rank": rank,
        "world_size": world_size,
        "evaluation_id": evaluation_identity["evaluation_id"],
        "sample_id": task.bound_sample_id,
        "group_uid": (
            f"coredev:{task.ordinal}"
            if evaluation_identity["evaluation_schema_version"] == POLICY_COREDEV_SCHEMA
            else f"benchmark:{task.ordinal}"
        ),
        "rollout_index": 0,
        "ordinal": task.ordinal,
        "dataset": task.dataset,
        "row_number": task.row_number,
        "index": task.index,
        "question": task.question,
        "image_paths": list(task.image_paths),
        "image_sha256s": list(task.image_sha256s),
        "image_dimensions": [list(item) for item in task.image_dimensions],
        "policy_run_id": policy_snapshot["run_id"],
        "optimizer_step": policy_snapshot["optimizer_step"],
        "policy_weights_sha256": policy_snapshot["weights_sha256"],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"policy benchmark result {field} differs")
    expected_trajectory_id = TrajectoryIdentity(
        str(evaluation_identity["evaluation_id"]),
        task.bound_sample_id,
        0,
        str(expected["group_uid"]),
    ).canonical_id
    if payload.get("trajectory_id") != expected_trajectory_id:
        raise RuntimeError("policy benchmark result trajectory_id differs")
    if task.ordinal % world_size != rank:
        raise RuntimeError("policy benchmark result is stored under the wrong rank")


def load_policy_benchmark_results(
    inference_root: str | Path,
    *,
    tasks: Sequence[CoreDevTask],
    evaluation_identity: Mapping[str, object],
    require_complete: bool = False,
) -> dict[int, dict[str, object]]:
    """Load all rank JSONLs, rejecting duplicates and any resume drift."""

    root = Path(inference_root)
    task_by_ordinal = {task.ordinal: task for task in tasks if task.single_image}
    world_size = evaluation_identity.get("execution", {}).get("world_size")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("evaluation identity world_size is invalid")
    records: dict[int, dict[str, object]] = {}
    for rank in range(world_size):
        path = root / f"rank-{rank}.jsonl"
        if not path.exists():
            if require_complete:
                raise FileNotFoundError(f"missing policy benchmark rank result: {path}")
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                try:
                    lines = handle.read().splitlines()
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, UnicodeDecodeError) as error:
            raise RuntimeError(
                f"cannot read policy benchmark result: {path}"
            ) from error
        for line_number, line in enumerate(lines, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid policy benchmark result at {path}:{line_number}"
                ) from error
            if not isinstance(raw, dict):
                raise RuntimeError(
                    f"policy benchmark result is not an object at {path}:{line_number}"
                )
            ordinal = raw.get("ordinal")
            if type(ordinal) is not int or ordinal in records:
                raise RuntimeError(
                    f"duplicate/invalid policy benchmark ordinal at {path}:{line_number}"
                )
            task = task_by_ordinal.get(ordinal)
            if task is None:
                raise RuntimeError(
                    "policy benchmark result is outside its task tranche"
                )
            validate_policy_benchmark_result(
                raw,
                task=task,
                evaluation_identity=evaluation_identity,
                rank=rank,
                world_size=world_size,
            )
            records[ordinal] = raw
    if require_complete and set(records) != set(task_by_ordinal):
        missing = sorted(set(task_by_ordinal).difference(records))
        raise RuntimeError(f"policy benchmark results are incomplete: {missing[:5]}")
    return records


_LEGACY_MODULE = "tgvf_rl.evaluation.policy_coredev"


def _bind_legacy_function(value: Callable[..., object], *, name: str) -> None:
    value.__module__ = _LEGACY_MODULE
    value.__name__ = name
    value.__qualname__ = name


# These objects historically lived in policy_coredev. The facade re-exports
# the exact objects so old imports and pickle payloads continue to resolve.
CoreDevTask.__module__ = _LEGACY_MODULE
for _member in CoreDevTask.__dict__.values():
    if isinstance(_member, FunctionType) and _member.__module__ == __name__:
        _member.__module__ = _LEGACY_MODULE
    elif isinstance(_member, property):
        for _accessor in (_member.fget, _member.fset, _member.fdel):
            if (
                isinstance(_accessor, FunctionType)
                and _accessor.__module__ == __name__
            ):
                _accessor.__module__ = _LEGACY_MODULE
for _function, _legacy_name in (
    (_read_regular_file_bytes, "_read_regular_file_bytes"),
    (_read_bound_image_bytes, "_read_bound_image_bytes"),
    (_decode_rgb_bytes, "_decode_rgb_bytes"),
    (image_file_identity, "image_file_identity"),
    (load_verified_task_image, "load_verified_task_image"),
    (write_official_coredev_tasks, "write_official_coredev_tasks"),
    (_tsv_image_paths, "_tsv_image_paths"),
    (_option_lines, "_option_lines"),
    (_official_prompt_text, "_official_prompt_text"),
    (load_benchmark_tasks, "load_benchmark_tasks"),
    (load_coredev_tasks, "load_coredev_tasks"),
    (prepare_policy_benchmark_tasks, "prepare_policy_benchmark_tasks"),
    (load_bound_policy_benchmark_tasks, "load_bound_policy_benchmark_tasks"),
    (trajectory_audit_payload, "trajectory_audit_payload"),
    (validate_policy_benchmark_result, "validate_policy_benchmark_result"),
    (load_policy_benchmark_results, "load_policy_benchmark_results"),
):
    _bind_legacy_function(_function, name=_legacy_name)


__all__ = [
    "CoreDevTask",
    "POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA",
    "image_file_identity",
    "load_benchmark_tasks",
    "load_bound_policy_benchmark_tasks",
    "load_coredev_tasks",
    "load_policy_benchmark_results",
    "load_verified_task_image",
    "prepare_policy_benchmark_tasks",
    "trajectory_audit_payload",
    "validate_policy_benchmark_result",
    "write_official_coredev_tasks",
]
