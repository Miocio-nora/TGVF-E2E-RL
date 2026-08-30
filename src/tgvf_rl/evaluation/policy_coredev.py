"""Standalone benchmark inference for completed visual-tool policy artifacts.

The training runtime already owns the native multi-turn protocol and the
colocated vLLM visual-tool implementation.  This module supplies only the
post-training boundary: an immutable LoRA closure or immutable full-model
identity records whose external weight closure is re-hashed before use, one
vLLM replica, and the official CoreDev prompt rows. It deliberately performs
no reward or update.
"""

from __future__ import annotations

import asyncio
import ast
import csv
from dataclasses import asdict
import fcntl
import hashlib
import io
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4
import re

import torch
from PIL import Image

from tgvf_rl.conditioning import TargetConditioningProviderKind
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.environment import (
    CropExecutionLedger,
    FocusExecutionLedger,
    FrameworkNeutralAgentLoop,
    ImageZoomInToolRuntime,
    Qwen3NativeToolLayoutBuilder,
    QwenNativeToolObservationAppender,
    record_trajectory_source_visual,
)
from tgvf_rl.environment.native_appender import NativeSuccessObservationContract
from tgvf_rl.environment.qwen3_crop_materializer import preprocess_qwen3_rgb
from tgvf_rl.framework.verl.native_agent_loop import VerlAsyncServerPolicyTurnClient
from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyLoRASnapshot as PolicyLoRASnapshot,
    PolicyWeightSyncState,
    load_lora_snapshot_pointer,
)
from tgvf_rl.framework.verl.policy_live_runtime import (
    _BRANCH_LAYERS,
    _RemoteCropVisualMaterializer,
    _RemoteTGVFFocusToolRuntime,
    _VisualTokenCountResolver,
    _artifact_identity,
    _initial_vllm_inputs,
    _source_visual_positions,
)
from tgvf_rl.framework.vllm import (
    ContentAddressedVLLMTurnRNG,
    FastTokenizerTokenByteSpanDecoder,
    LiveVLLMTurnContextRegistry,
    Qwen3VLLMObservationPayloadResolver,
    VLLMPolicySampler,
)
from tgvf_rl.observations.store import ObservationStore, tensor_checksum
from tgvf_rl.qwen import Qwen3VLAdapter
from tgvf_rl.policy.run_config import (
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)
from tgvf_rl.protocol import (
    native_assistant_dialect_for_model,
    NativeProtocolRenderer,
    NativeToolCapabilityProfile,
    StrictToolCallParser,
    build_native_tool_schemas,
    build_visual_tool_prompt_messages,
)
from tgvf_rl.protocol.state_machine import CapErrorBehavior
from tgvf_rl.trajectories.behavior import BehaviorTraceStore, VLLMBehaviorRecorder
from tgvf_rl.trajectories.schema import (
    CropTGVFToolCallRecord,
    CropToolCallRecord,
    ToolCallRecord,
    TrajectoryIdentity,
    TrajectoryRecord,
    trajectory_checksum,
)

from .policy_full_model_snapshot import (
    FULL_MODEL_EVALUATION_BACKEND,
    FullModelEvaluationSnapshot,
    _base_equivalent_step_zero_full_model,
    build_full_model_standalone_manager,
    full_model_snapshot_identity_record,
    load_full_model_evaluation_snapshot,
)
from .policy_evaluation_config import (
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    LORA_ADAPTER_EVALUATION_BACKEND,
    POLICY_BENCHMARK_LEGACY_SCHEMA_V1,
    POLICY_BENCHMARK_SCHEMA,
    POLICY_COREDEV_LEGACY_SCHEMA_V1,
    POLICY_COREDEV_SCHEMA,
    TRAINING_RUN_EVALUATION_PROTOCOL,
    PolicyCoreDevConfig,
    _LEGACY_COREDEV_TASK_COUNT,
    _require_sha256,
    load_policy_coredev_config,
)
from .policy_evaluation_identity import (
    POLICY_EVAL_CONTRACT_SCHEMA,
    POLICY_EVALUATION_IDENTITY_SCHEMA,
    PolicyEvalContract,
    _decoding_contract,
    _termination_contract,
    build_policy_eval_contract,
    build_policy_evaluation_identity as _build_policy_evaluation_identity,
    canonical_json_sha256 as _canonical_json_sha256,
    effective_evaluation_image_max_pixels,
    evaluation_protocol_identity as _shared_evaluation_protocol_identity,
    policy_benchmark_task_path,
    policy_eval_action_boundary_identity as _policy_eval_action_boundary_identity,  # noqa: F401
    policy_eval_observation_identity as _policy_eval_observation_identity,  # noqa: F401
    policy_eval_parser_identity as _policy_eval_parser_identity,  # noqa: F401
)
from .policy_lora_snapshot import (
    VLLM_LORA_ADAPTER_CONFIG_FILENAME as VLLM_LORA_ADAPTER_CONFIG_FILENAME,
    VLLM_LORA_ADAPTER_IDENTITY_FILENAME as VLLM_LORA_ADAPTER_IDENTITY_FILENAME,
    VLLM_LORA_ADAPTER_MODEL_FILENAME as VLLM_LORA_ADAPTER_MODEL_FILENAME,
    VLLM_LORA_ADAPTER_SCHEMA as VLLM_LORA_ADAPTER_SCHEMA,
    VLLM_LORA_ENGINE_ATTESTATION as VLLM_LORA_ENGINE_ATTESTATION,
    VLLM_LORA_RESIDUAL_RACE as VLLM_LORA_RESIDUAL_RACE,
    PolicyEvaluationSnapshot,
    VLLMLoRAAdapterIntegrityVerifier,
    _assert_private_vllm_lora_file_equals_at as _assert_private_vllm_lora_file_equals_at,
    _base_equivalent_step_zero_lora,
    _open_absolute_directory_nofollow as _open_absolute_directory_nofollow,
    _open_or_create_private_directory_at as _open_or_create_private_directory_at,
    _open_vllm_lora_adapter_root as _open_vllm_lora_adapter_root,
    _publish_private_vllm_lora_file_at as _publish_private_vllm_lora_file_at,
    _read_private_vllm_lora_file_at as _read_private_vllm_lora_file_at,
    _standalone_engine_kwargs,
    _vllm_lora_adapter_payloads as _vllm_lora_adapter_payloads,
    _write_private_vllm_lora_file_at as _write_private_vllm_lora_file_at,
    build_vllm_lora_adapter_integrity_verifier,
    materialize_vllm_lora_adapter,
    policy_lora_request_name,
)
from .policy_vllm_manager import (
    StandaloneTGVFVLLMManager,
    _single_collective as _single_collective,
    _TurnRoute as _TurnRoute,
)


POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA = "tgvf-policy-coredev-trajectory-audit-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


PolicyEvaluationSubject = PolicyEvaluationSnapshot | FullModelEvaluationSnapshot


def _assert_policy_snapshot_binding(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
    *,
    owner: str,
) -> None:
    run_identity_sha256 = (
        snapshot.lora.run_identity_sha256
        if isinstance(snapshot, PolicyEvaluationSnapshot)
        else snapshot.run_identity_sha256
    )
    if (
        config.expected_policy_run_id is not None
        and snapshot.policy_version.run_id != config.expected_policy_run_id
    ):
        raise ValueError(f"{owner} run_id differs from evaluation binding")
    if (
        config.expected_policy_run_identity_sha256 is not None
        and run_identity_sha256 != config.expected_policy_run_identity_sha256
    ):
        raise ValueError(f"{owner} run identity differs from evaluation binding")
    if (
        config.expected_optimizer_step is not None
        and snapshot.policy_version.optimizer_step != config.expected_optimizer_step
    ):
        raise ValueError(f"{owner} optimizer step differs from evaluation binding")
    if (
        config.expected_policy_weights_sha256 is not None
        and snapshot.policy_version.weights_sha256
        != config.expected_policy_weights_sha256
    ):
        raise ValueError(f"{owner} weights differ from evaluation binding")
    if isinstance(snapshot, FullModelEvaluationSnapshot):
        if config.required_snapshot_identity_sha256 is None:
            raise ValueError(
                "full-model evaluation lacks its required snapshot identity"
            )
        if (
            snapshot.manifest.identity_sha256
            != config.required_snapshot_identity_sha256
        ):
            raise ValueError(
                f"{owner} full-model identity differs from evaluation binding"
            )
        if _sha256_file(config.policy_config_path) != (
            snapshot.manifest.run_contract_file_sha256
        ):
            raise ValueError(f"{owner} full-model run contract bytes differ")


def _load_full_model_from_paths(
    config: PolicyCoreDevConfig,
    *,
    manifest_path: Path,
    receipt_path: Path,
) -> FullModelEvaluationSnapshot:
    if (
        config.full_model_snapshot_manifest_sha256 is None
        or config.full_model_materialization_receipt_sha256 is None
    ):
        raise ValueError("full-model evaluation file hashes are absent")
    if _sha256_file(manifest_path) != config.full_model_snapshot_manifest_sha256:
        raise ValueError("full-model snapshot manifest file SHA256 differs")
    if _sha256_file(receipt_path) != config.full_model_materialization_receipt_sha256:
        raise ValueError("full-model materialization receipt file SHA256 differs")
    snapshot = load_full_model_evaluation_snapshot(
        manifest_path, receipt_path, runtime_lightweight=False
    )
    _assert_policy_snapshot_binding(config, snapshot, owner="full-model snapshot")
    return snapshot


def load_policy_evaluation_snapshot(
    config: PolicyCoreDevConfig,
) -> PolicyEvaluationSubject:
    """Strictly load the configured LoRA or full-model snapshot exactly once."""

    if config.snapshot_backend == FULL_MODEL_EVALUATION_BACKEND:
        assert config.full_model_snapshot_manifest_path is not None
        assert config.full_model_materialization_receipt_path is not None
        return _load_full_model_from_paths(
            config,
            manifest_path=config.full_model_snapshot_manifest_path,
            receipt_path=config.full_model_materialization_receipt_path,
        )

    run = load_policy_e2e_smoke_run_config(
        config.policy_config_path,
        allow_external_agent_loop_config=True,
        allow_historical_reward_contract=True,
    )
    assert config.lora_pointer_path is not None
    state = PolicyWeightSyncState(
        directory=config.lora_pointer_path.parent,
        run_id=run.run_id,
        run_identity_sha256=run.identity_sha256,
    )
    snapshot = load_lora_snapshot_pointer(
        state,
        pointer_path=config.lora_pointer_path,
        expected_pointer_file_sha256=config.lora_pointer_sha256,
        expected_optimizer_step=config.expected_optimizer_step,
    )
    loaded = PolicyEvaluationSnapshot(run=run, lora=snapshot)
    _assert_policy_snapshot_binding(config, loaded, owner="policy snapshot")
    return loaded


def frozen_policy_state_root(config: PolicyCoreDevConfig) -> Path:
    return config.output_root / "runtime" / "frozen-policy-state"


def frozen_full_model_state_root(config: PolicyCoreDevConfig) -> Path:
    """Return the immutable record root; full-model payload bytes stay external."""

    return config.output_root / "runtime" / "frozen-full-model-state"


def _write_immutable_snapshot_file(path: Path, payload: bytes) -> None:
    """Legacy freeze writer for an evaluation-controlled output hierarchy.

    This helper uses path-based parent creation and therefore does not claim a
    TOCTOU-safe write boundary when the output ancestry is concurrently mutable.
    LoRA callers subsequently re-open the frozen closure through the strict
    snapshot reader, but that does not upgrade this writer's path boundary.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"frozen policy snapshot is not regular: {path}")
        if path.read_bytes() != payload:
            raise RuntimeError(f"frozen policy snapshot collision: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(f"frozen policy snapshot collision: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def freeze_policy_evaluation_snapshot(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
) -> PolicyEvaluationSubject:
    """Copy verified identity records into trusted evaluation-private storage.

    The legacy writer assumes non-adversarial output ancestry; this function is
    not an end-to-end TOCTOU-free materializer. The LoRA branch copies and
    reloads its complete closure through the descriptor-relative, no-symlink
    reader. The full-model branch copies only manifest/receipt records because
    duplicating the 100+ GiB checkpoint is out of scope; every reload therefore
    streams and verifies the external source and materialized model bytes.
    """

    if isinstance(snapshot, FullModelEvaluationSnapshot):
        assert config.full_model_snapshot_manifest_path is not None
        assert config.full_model_materialization_receipt_path is not None
        frozen_root = frozen_full_model_state_root(config)
        _write_immutable_snapshot_file(
            frozen_root / "snapshot-manifest.json",
            _read_regular_file_bytes(
                config.full_model_snapshot_manifest_path,
                owner="full-model snapshot manifest",
            ),
        )
        _write_immutable_snapshot_file(
            frozen_root / "materialization-receipt.json",
            _read_regular_file_bytes(
                config.full_model_materialization_receipt_path,
                owner="full-model materialization receipt",
            ),
        )
        return load_frozen_policy_evaluation_snapshot(config)

    if snapshot.run.run_id != snapshot.policy_version.run_id:
        raise ValueError("cannot freeze a policy snapshot from another run")
    source_root = snapshot.lora.pointer_file.parent
    try:
        manifest_relative = snapshot.lora.manifest_file.relative_to(source_root)
        tensor_relative = snapshot.lora.tensor_file.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            "policy snapshot closure escapes its state directory"
        ) from error
    frozen_root = frozen_policy_state_root(config)
    pointer_path = frozen_root / "latest-lora-snapshot.json"

    # Pointer publication is last so its existence proves the complete closure
    # has already been materialized.
    _write_immutable_snapshot_file(
        frozen_root / tensor_relative, snapshot.lora.tensor_bytes
    )
    _write_immutable_snapshot_file(
        frozen_root / manifest_relative, snapshot.lora.manifest_bytes
    )
    _write_immutable_snapshot_file(pointer_path, snapshot.lora.pointer_bytes)
    return load_frozen_policy_evaluation_snapshot(config)


def load_frozen_policy_evaluation_snapshot(
    config: PolicyCoreDevConfig,
) -> PolicyEvaluationSubject:
    """Load private records and strongly verify every externally bound payload."""

    if config.snapshot_backend == FULL_MODEL_EVALUATION_BACKEND:
        frozen_root = frozen_full_model_state_root(config)
        return _load_full_model_from_paths(
            config,
            manifest_path=frozen_root / "snapshot-manifest.json",
            receipt_path=frozen_root / "materialization-receipt.json",
        )

    run = load_policy_e2e_smoke_run_config(
        config.policy_config_path,
        allow_external_agent_loop_config=True,
        allow_historical_reward_contract=True,
    )
    state = PolicyWeightSyncState(
        directory=frozen_policy_state_root(config),
        run_id=run.run_id,
        run_identity_sha256=run.identity_sha256,
    )
    snapshot = load_lora_snapshot_pointer(
        state,
        pointer_path=state.latest_path,
        expected_pointer_file_sha256=config.lora_pointer_sha256,
        expected_optimizer_step=config.expected_optimizer_step,
    )
    frozen = PolicyEvaluationSnapshot(run=run, lora=snapshot)
    _assert_policy_snapshot_binding(config, frozen, owner="frozen policy snapshot")
    return frozen


def policy_version_from_pointer(config: PolicyCoreDevConfig) -> PolicyVersion:
    """Backward-compatible strict one-shot policy identity loader."""

    return load_policy_evaluation_snapshot(config).policy_version


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


def _evaluation_protocol_identity(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
) -> dict[str, object]:
    """Compatibility wrapper around the backend-neutral protocol identity."""

    is_lora = isinstance(snapshot, PolicyEvaluationSnapshot)
    if not is_lora and not isinstance(snapshot, FullModelEvaluationSnapshot):
        raise TypeError("snapshot must be a policy evaluation snapshot")
    return _shared_evaluation_protocol_identity(
        config,
        snapshot,
        is_lora_snapshot=is_lora,
        step_zero_equivalence=lambda: (
            _base_equivalent_step_zero_lora(snapshot)
            if is_lora
            else _base_equivalent_step_zero_full_model(snapshot)
        ),
    )


def policy_evaluation_identity(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
) -> dict[str, object]:
    """Bind experiment, model, task population, and exact policy bytes."""

    is_lora = isinstance(snapshot, PolicyEvaluationSnapshot)
    if not is_lora and not isinstance(snapshot, FullModelEvaluationSnapshot):
        raise TypeError("snapshot must be a policy evaluation snapshot")
    policy_snapshot = (
        {
            "run_id": snapshot.policy_version.run_id,
            "run_identity_sha256": snapshot.lora.run_identity_sha256,
            "optimizer_step": snapshot.policy_version.optimizer_step,
            "weights_sha256": snapshot.policy_version.weights_sha256,
            "pointer_file_sha256": snapshot.lora.pointer_file_sha256,
            "manifest_file_sha256": snapshot.lora.manifest_file_sha256,
            "tensor_file_sha256": snapshot.lora.tensor_file_sha256,
            "request_sha256": snapshot.lora.request_sha256,
        }
        if is_lora
        else full_model_snapshot_identity_record(snapshot)
    )
    policy_config_path = Path(
        config.policy_config_path
        if is_lora
        else getattr(config, "policy_config_path", snapshot.contract.source_path)
    )
    return _build_policy_evaluation_identity(
        config,
        snapshot,
        is_lora_snapshot=is_lora,
        policy_snapshot=policy_snapshot,
        policy_config_path=policy_config_path,
        step_zero_equivalence=lambda: (
            _base_equivalent_step_zero_lora(snapshot)
            if is_lora
            else _base_equivalent_step_zero_full_model(snapshot)
        ),
        contract_type=PolicyEvalContract,
    )


def write_policy_evaluation_identity(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
) -> dict[str, object]:
    """Create one immutable run identity shared by all evaluator ranks."""

    identity = policy_evaluation_identity(config, snapshot)
    encoded = (
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path = config.output_root / "runtime" / "evaluation-identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("policy evaluation identity is not a regular file")
        if path.read_bytes() != encoded:
            raise RuntimeError("policy evaluation identity collision")
        return identity
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(encoded)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError("policy evaluation identity collision")
    finally:
        temporary.unlink(missing_ok=True)
    return identity


async def build_standalone_manager(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
) -> tuple[StandaloneTGVFVLLMManager, object, object]:
    """Construct one single-GPU post-training vLLM replica."""

    if isinstance(snapshot, FullModelEvaluationSnapshot):
        return await build_full_model_standalone_manager(config, snapshot)

    from vllm import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.v1.engine.async_llm import AsyncLLM

    if not isinstance(snapshot, PolicyEvaluationSnapshot):
        raise TypeError("snapshot must be a PolicyEvaluationSnapshot")
    run = snapshot.run
    adapter_root = materialize_vllm_lora_adapter(config, snapshot)
    adapter_integrity_verifier = build_vllm_lora_adapter_integrity_verifier(
        config,
        snapshot,
        adapter_root,
    )
    adapter_integrity_verifier.verify(phase="before engine construction")
    engine_args = AsyncEngineArgs(**_standalone_engine_kwargs(config, run))
    engine = AsyncLLM.from_engine_args(engine_args)
    adapter_integrity_verifier.verify(phase="after engine construction")
    lora = LoRARequest(policy_lora_request_name(snapshot), 1, str(adapter_root))
    manager = StandaloneTGVFVLLMManager(
        engine,
        lora,
        capture_hidden=(
            config.evaluation_protocol == TRAINING_RUN_EVALUATION_PROTOCOL
            and run.protocol.tool_profile is NativeToolCapabilityProfile.TGVF_ONLY
        ),
        native_pixels=(
            config.evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
        ),
        adapter_integrity_verifier=adapter_integrity_verifier,
    )
    return manager, engine, run


class PolicyCoreDevEvaluator:
    def __init__(
        self,
        *,
        config: PolicyCoreDevConfig,
        run: PolicyE2ESmokeRunConfig,
        manager: StandaloneTGVFVLLMManager,
        processor: object,
        snapshot: PolicyEvaluationSnapshot,
        evaluation_identity: Mapping[str, object],
    ) -> None:
        self.config = config
        self.run = run
        self.manager = manager
        self.processor = processor
        if snapshot.run != run:
            raise ValueError("evaluator run differs from its frozen policy snapshot")
        expected_identity = policy_evaluation_identity(config, snapshot)
        if dict(evaluation_identity) != expected_identity:
            raise ValueError("evaluator identity differs from its bound experiment")
        self.snapshot = snapshot
        self.evaluation_identity = expected_identity
        self.policy_version = snapshot.policy_version
        self.store = ObservationStore()
        self.behavior_store = BehaviorTraceStore()
        self.focus_ledger = FocusExecutionLedger()
        self.crop_ledger = CropExecutionLedger()
        self.layout_builder = Qwen3NativeToolLayoutBuilder.from_processor_config(
            processor=processor,
            model_identity=run.model,
            observation_store=self.store,
        )
        schemas = tuple(build_native_tool_schemas(run.protocol.enabled_tool_names))
        self.assistant_dialect = native_assistant_dialect_for_model(
            run.model.model_name
        )
        self.eval_contract = build_policy_eval_contract(config, snapshot)
        self.observation_contract = NativeSuccessObservationContract(
            protocol_id=self.eval_contract.success_observation_protocol_id,
            tool_profile=run.protocol.tool_profile,
            assistant_dialect=self.assistant_dialect,
        )
        self.renderer = NativeProtocolRenderer(
            processor,
            expected_tokenizer_length=run.model.tokenizer_length,
            tool_names=run.protocol.enabled_tool_names,
            tool_schemas=schemas,
            assistant_dialect=self.assistant_dialect,
        )
        conditioning = run.representation.conditioning
        self.contextual_identity = (
            _artifact_identity(
                "policy-evaluation",
                "qwen3-contextual-behavior-forward",
                config.schema_version,
                {
                    "evaluation_id": config.evaluation_id,
                    "policy": self.policy_version.weights_sha256,
                    "provider": conditioning.provider.value,
                    "hidden_layer": conditioning.hidden_layer,
                },
            )
            if conditioning.provider
            is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
            else None
        )
        export = load_rank_zero_adapter_owned_state_export(
            run.representation.artifact_path
        )
        adapter_contract = export.manifest.run_identity.adapter_contract
        self.branch_identities = tuple(
            _artifact_identity(
                "qwen3-vl",
                f"deepstack-merger-{layer}",
                "frozen-base-model-v1",
                {
                    "model": run.model.revision_or_path,
                    "projection_identity": projection_identity,
                    "layer": layer,
                },
            )
            for layer, projection_identity in zip(
                _BRANCH_LAYERS,
                adapter_contract.deepstack_projection_identities,
                strict=True,
            )
        )

    def render_initial_prompt(
        self, task: CoreDevTask, *, source_rgb: torch.Tensor
    ) -> tuple[int, ...]:
        if not task.single_image:
            raise ValueError("current visual-tool protocol has no multi-image selector")
        messages = build_visual_tool_prompt_messages(
            task.question,
            tool_profile=self.run.protocol.tool_profile,
            assistant_dialect=self.assistant_dialect,
        )
        rendered = self.renderer.render(messages, add_generation_prompt=True)
        self.renderer.assert_generation_prefill(rendered, self.renderer.tokenizer)
        from tgvf_rl.framework.verl.smoke_dataset import (
            _materialize_source_image_prompt_token_ids,
        )

        return _materialize_source_image_prompt_token_ids(
            processor=self.processor,
            canonical_token_ids=rendered.token_ids,
            prompt_text=rendered.text,
            image_max_pixels=self.eval_contract.effective_image_max_pixels,
            source_rgb=source_rgb,
        )

    async def evaluate(self, task: CoreDevTask) -> TrajectoryRecord:
        # This is deliberately the last filesystem read before model input
        # construction. Prompt expansion and visual encoding share these exact
        # verified bytes, so a mutable source/hardlink cannot create a TOCTOU.
        source_rgb = load_verified_task_image(task)
        prompt_ids = self.render_initial_prompt(task, source_rgb=source_rgb)
        identity = TrajectoryIdentity(
            self.config.evaluation_id,
            task.bound_sample_id,
            0,
            (
                f"coredev:{task.ordinal}"
                if self.config.uses_legacy_coredev_manifest
                else f"benchmark:{task.ordinal}"
            ),
        )
        trajectory_id = identity.canonical_id
        pixel_values, image_grid_thw = preprocess_qwen3_rgb(
            processor=self.processor,
            rgb=source_rgb,
            image_max_pixels=self.eval_contract.effective_image_max_pixels,
        )
        source = await self.manager.materialize_source(
            request_id=trajectory_id,
            expected_step=self.policy_version.optimizer_step,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_sha256=tensor_checksum(source_rgb),
        )
        positions = _source_visual_positions(
            prompt_ids,
            image_token_id=self.layout_builder.image_pad_id,
            expected_count=int(source.merged_main.shape[-2]),
        )
        trajectory_source = record_trajectory_source_visual(
            trajectory_id=trajectory_id,
            source_visual=source,
            source_positions=positions,
            deepstack_branch_layers=_BRANCH_LAYERS,
            deepstack_injection_positions=tuple(positions for _ in _BRANCH_LAYERS),
            observation_store=self.store,
            source_rgb=source_rgb,
        )
        registry = LiveVLLMTurnContextRegistry(
            observation_resolver=Qwen3VLLMObservationPayloadResolver(
                store=self.store, include_multi_modal_uuid=False
            )
        )
        registry.register_initial_prompt(
            prompt_ids,
            _initial_vllm_inputs(
                initial_prompt_token_ids=prompt_ids,
                image_token_id=self.layout_builder.image_pad_id,
                source=source,
                image_max_pixels=self.eval_contract.effective_image_max_pixels,
            ),
        )
        appender = QwenNativeToolObservationAppender(
            tokenizer=self.layout_builder.tokenizer,
            registrar=registry,
            visual_token_count_resolver=_VisualTokenCountResolver(self.store),
            observation_contract=self.observation_contract,
        )
        if self.run.protocol.tool_profile is NativeToolCapabilityProfile.TGVF_ONLY:
            tool_runtime = _RemoteTGVFFocusToolRuntime(
                event_loop=asyncio.get_running_loop(),
                server_client=self.manager,
                config=self.run,
                source_visual=source,
                layout_builder=self.layout_builder,
                observation_store=self.store,
                execution_ledger=self.focus_ledger,
                contextual_forward_identity=self.contextual_identity,
                branch_merger_identities=self.branch_identities,
                observation_contract=self.observation_contract,
            )
        elif self.run.protocol.tool_profile is NativeToolCapabilityProfile.CROP_ONLY:
            processor_identity = _artifact_identity(
                "policy-evaluation",
                "qwen3-shared-vllm-crop-processor",
                self.config.schema_version,
                {
                    "model": self.run.model.revision_or_path,
                    "max_pixels": self.eval_contract.effective_image_max_pixels,
                },
            )
            layout_identity = _artifact_identity(
                "policy-evaluation",
                "qwen3-native-crop-layout",
                self.config.schema_version,
                {
                    "model": self.run.model.revision_or_path,
                    "success_observation_protocol_id": (
                        self.observation_contract.protocol_id.value
                    ),
                },
            )
            materializer = _RemoteCropVisualMaterializer(
                event_loop=asyncio.get_running_loop(),
                server_client=self.manager,
                processor=self.processor,
                model_identity=self.run.model,
                image_max_pixels=self.eval_contract.effective_image_max_pixels,
                trajectory_id=trajectory_id,
                behavior_policy=self.policy_version,
            )
            tool_runtime = ImageZoomInToolRuntime(
                model=self.run.model,
                materializer=materializer,
                layout_builder=self.layout_builder,
                observation_store=self.store,
                crop_processor_identity=processor_identity,
                crop_layout_identity=layout_identity,
                execution_ledger=self.crop_ledger,
                coordinate_mapper=Qwen3VLAdapter(),
                observation_contract=self.observation_contract,
            )
        else:
            raise RuntimeError("policy CoreDev supports the two trained atomic arms")

        decoding = _decoding_contract()
        client = VerlAsyncServerPolicyTurnClient(
            server_manager=self.manager,
            event_loop=asyncio.get_running_loop(),
            tokenizer=self.renderer.tokenizer,
            prompt_inputs=registry,
            token_byte_span_decoder=FastTokenizerTokenByteSpanDecoder(
                self.renderer.tokenizer
            ),
            sticky_request_id=trajectory_id,
            max_model_len=self.config.max_model_len,
            server_timeout_seconds=2400.0,
            logprobs_mode="processed_logprobs",
        )
        sampler = VLLMPolicySampler(
            client=client,
            behavior_policy=self.policy_version,
            rng=ContentAddressedVLLMTurnRNG(
                master_seed=self.run.rollout_rng.master_seed,
                stream_identity=trajectory_id,
            ),
            request_context=registry,
            decoding=decoding,
            termination=_termination_contract(self.run),
            assistant_dialect=self.assistant_dialect,
        )
        loop = FrameworkNeutralAgentLoop(
            sampler=sampler,
            tool_runtime=tool_runtime,
            appender=appender,
            parser=StrictToolCallParser(
                enabled_tool_names=self.run.protocol.enabled_tool_names
            ),
            behavior_recorder=VLLMBehaviorRecorder(self.behavior_store),
            max_tool_calls=self.run.protocol.maximum_tool_calls,
            enabled_tool_names=self.run.protocol.enabled_tool_names,
            cap_error_behavior=CapErrorBehavior.ONE_FINAL_ANSWER_TURN,
            assistant_dialect=self.assistant_dialect,
        )
        from tgvf_rl.environment.agent_loop import RolloutRequest

        request = RolloutRequest(
            "trajectory-v1",
            identity,
            self.run.model,
            self.policy_version,
            trajectory_source,
            prompt_ids,
            {},
            self.run.policy.sampling,
        )
        try:
            return await asyncio.to_thread(loop.run, request)
        finally:
            try:
                await self.manager.release_trajectory(trajectory_id)
            finally:
                trajectory_ids = (trajectory_id,)
                self.focus_ledger.release_trajectories(trajectory_ids)
                self.crop_ledger.release_trajectories(trajectory_ids)
                self.behavior_store.release_trajectories(trajectory_ids)
                self.store.release_trajectories(trajectory_ids)


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


__all__ = [
    "CoreDevTask",
    "FULL_MODEL_EVALUATION_BACKEND",
    "LORA_ADAPTER_EVALUATION_BACKEND",
    "POLICY_BENCHMARK_LEGACY_SCHEMA_V1",
    "POLICY_BENCHMARK_SCHEMA",
    "POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA",
    "POLICY_COREDEV_LEGACY_SCHEMA_V1",
    "POLICY_COREDEV_SCHEMA",
    "POLICY_EVAL_CONTRACT_SCHEMA",
    "POLICY_EVALUATION_IDENTITY_SCHEMA",
    "PolicyCoreDevConfig",
    "PolicyCoreDevEvaluator",
    "PolicyEvalContract",
    "PolicyEvaluationSnapshot",
    "PolicyEvaluationSubject",
    "StandaloneTGVFVLLMManager",
    "VLLMLoRAAdapterIntegrityVerifier",
    "build_vllm_lora_adapter_integrity_verifier",
    "build_standalone_manager",
    "build_policy_eval_contract",
    "effective_evaluation_image_max_pixels",
    "freeze_policy_evaluation_snapshot",
    "frozen_full_model_state_root",
    "frozen_policy_state_root",
    "image_file_identity",
    "load_benchmark_tasks",
    "load_bound_policy_benchmark_tasks",
    "load_coredev_tasks",
    "load_frozen_policy_evaluation_snapshot",
    "load_policy_benchmark_results",
    "load_policy_coredev_config",
    "load_policy_evaluation_snapshot",
    "load_verified_task_image",
    "materialize_vllm_lora_adapter",
    "policy_benchmark_task_path",
    "policy_evaluation_identity",
    "policy_lora_request_name",
    "policy_version_from_pointer",
    "prepare_policy_benchmark_tasks",
    "trajectory_audit_payload",
    "validate_policy_benchmark_result",
    "write_policy_evaluation_identity",
    "write_official_coredev_tasks",
]
