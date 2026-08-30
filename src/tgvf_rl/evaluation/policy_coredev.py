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
import hmac
import io
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from uuid import uuid4
import re

import torch
from PIL import Image

from tgvf_rl.conditioning import TargetConditioningProviderKind
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
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
    PolicyLoRASnapshot,
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
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVF_VLLM_WORKER_EXTENSION_FQN,
    TGVFFocusMaterializationResult,
    _focus_from_utility_wire,
    _source_from_utility_wire,
    _tensor_to_utility_wire,
)
from tgvf_rl.framework.vllm import (
    ContentAddressedVLLMTurnRNG,
    FastTokenizerTokenByteSpanDecoder,
    LiveVLLMTurnContextRegistry,
    Qwen3VLLMObservationPayloadResolver,
    VLLMOutputDecodingContract,
    VLLMPolicySampler,
    VLLMTerminationOutcome,
    VLLMTurnTerminationContract,
)
from tgvf_rl.framework.vllm.registration import (
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
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
    NativeActionBoundaryProtocolId,
    NativeAssistantDialect,
    NativeProtocolRenderer,
    NativeSuccessObservationProtocolId,
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
    FullModelSourceKind,
    build_full_model_standalone_manager,
    full_model_snapshot_identity_record,
    load_full_model_evaluation_snapshot,
)


POLICY_COREDEV_LEGACY_SCHEMA_V1 = "tgvf-policy-coredev-evaluation-v1"
POLICY_BENCHMARK_LEGACY_SCHEMA_V1 = "tgvf-policy-benchmark-evaluation-v1"
POLICY_COREDEV_SCHEMA = "tgvf-policy-coredev-evaluation-v2"
POLICY_BENCHMARK_SCHEMA = "tgvf-policy-benchmark-evaluation-v2"
POLICY_EVALUATION_IDENTITY_SCHEMA = "tgvf-policy-evaluation-identity-v1"
POLICY_EVAL_CONTRACT_SCHEMA = "tgvf-policy-eval-contract-v1"
POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA = "tgvf-policy-coredev-trajectory-audit-v1"
TRAINING_RUN_EVALUATION_PROTOCOL = "training_run"
DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL = (
    "deepeyes_official_visible_native_crop_v1"
)
POLICY_EVALUATION_PROTOCOLS = frozenset(
    {
        TRAINING_RUN_EVALUATION_PROTOCOL,
        DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    }
)
_LEGACY_COREDEV_TASK_COUNT = 2511
_LEGACY_COREDEV_SINGLE_IMAGE_COUNT = 2240
_SHA256_LENGTH = 64
LORA_ADAPTER_EVALUATION_BACKEND = "lora_adapter"
VLLM_LORA_ADAPTER_SCHEMA = "tgvf-policy-vllm-lora-adapter-v2"
VLLM_LORA_ADAPTER_MODEL_FILENAME = "adapter_model.safetensors"
VLLM_LORA_ADAPTER_CONFIG_FILENAME = "adapter_config.json"
VLLM_LORA_ADAPTER_IDENTITY_FILENAME = "identity.json"
VLLM_LORA_ENGINE_ATTESTATION = "unavailable-in-vllm-0.12-public-api"
VLLM_LORA_RESIDUAL_RACE = (
    "same-UID mutation between the final pre-generate verification and "
    "vLLM's lazy adapter file read remains outside the public API boundary"
)
POLICY_EVALUATION_BACKENDS = frozenset(
    {LORA_ADAPTER_EVALUATION_BACKEND, FULL_MODEL_EVALUATION_BACKEND}
)


@dataclass(frozen=True, slots=True)
class PolicyCoreDevConfig:
    evaluation_id: str
    policy_config_path: Path
    lora_pointer_path: Path | None
    output_root: Path
    gpu_ids: tuple[int, ...]
    declared_image_max_pixels: int
    success_observation_protocol_id: NativeSuccessObservationProtocolId
    action_boundary_protocol_id: NativeActionBoundaryProtocolId
    inference_concurrency_per_gpu: int = 8
    max_model_len: int = 16384
    max_num_batched_tokens: int = 16384
    enable_chunked_prefill: bool = True
    gpu_memory_utilization: float = 0.90
    task_manifest_path: Path | None = None
    task_manifest_sha256: str | None = None
    expected_task_count: int = _LEGACY_COREDEV_TASK_COUNT
    expected_single_image_count: int = _LEGACY_COREDEV_SINGLE_IMAGE_COUNT
    lora_pointer_sha256: str | None = None
    expected_policy_run_id: str | None = None
    expected_policy_run_identity_sha256: str | None = None
    expected_optimizer_step: int | None = None
    expected_policy_weights_sha256: str | None = None
    snapshot_backend: str = LORA_ADAPTER_EVALUATION_BACKEND
    full_model_snapshot_manifest_path: Path | None = None
    full_model_snapshot_manifest_sha256: str | None = None
    full_model_materialization_receipt_path: Path | None = None
    full_model_materialization_receipt_sha256: str | None = None
    required_snapshot_identity_sha256: str | None = None
    evaluation_protocol: str = TRAINING_RUN_EVALUATION_PROTOCOL
    schema_version: str = POLICY_COREDEV_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_config_path", Path(self.policy_config_path))
        if self.lora_pointer_path is not None:
            object.__setattr__(self, "lora_pointer_path", Path(self.lora_pointer_path))
        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.task_manifest_path is not None:
            object.__setattr__(
                self, "task_manifest_path", Path(self.task_manifest_path)
            )
        for name in (
            "full_model_snapshot_manifest_path",
            "full_model_materialization_receipt_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        object.__setattr__(self, "gpu_ids", tuple(self.gpu_ids))
        if (
            type(self.declared_image_max_pixels) is not int
            or self.declared_image_max_pixels <= 0
        ):
            raise ValueError("declared_image_max_pixels must be a positive integer")
        try:
            observation_protocol_id = NativeSuccessObservationProtocolId(
                self.success_observation_protocol_id
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "success_observation_protocol_id must be an explicit known protocol"
            ) from error
        object.__setattr__(
            self, "success_observation_protocol_id", observation_protocol_id
        )
        try:
            action_boundary_protocol_id = NativeActionBoundaryProtocolId(
                self.action_boundary_protocol_id
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "action_boundary_protocol_id must be an explicit known protocol"
            ) from error
        object.__setattr__(
            self, "action_boundary_protocol_id", action_boundary_protocol_id
        )
        if self.schema_version in {
            POLICY_COREDEV_LEGACY_SCHEMA_V1,
            POLICY_BENCHMARK_LEGACY_SCHEMA_V1,
        }:
            raise ValueError(
                "legacy v1 policy evaluation configs are immutable evidence only; "
                "materialize an explicit v2 config before evaluation"
            )
        if self.schema_version not in {POLICY_COREDEV_SCHEMA, POLICY_BENCHMARK_SCHEMA}:
            raise ValueError("policy benchmark config schema differs")
        if self.evaluation_protocol not in POLICY_EVALUATION_PROTOCOLS:
            raise ValueError("policy benchmark evaluation protocol differs")
        if self.snapshot_backend not in POLICY_EVALUATION_BACKENDS:
            raise ValueError("policy benchmark snapshot backend differs")
        if not self.evaluation_id:
            raise ValueError("evaluation_id must be non-empty")
        if (
            len(self.gpu_ids) != 4
            or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in self.gpu_ids)
            or len(set(self.gpu_ids)) != 4
        ):
            raise ValueError(
                "formal policy benchmark evaluation requires four distinct "
                "non-negative GPU IDs"
            )
        if not 1 <= self.inference_concurrency_per_gpu <= 8:
            raise ValueError("inference_concurrency_per_gpu must be in [1,8]")
        if type(self.max_model_len) is not int or self.max_model_len <= 0:
            raise ValueError("max_model_len must be a positive integer")
        if (
            type(self.max_num_batched_tokens) is not int
            or self.max_num_batched_tokens <= 0
        ):
            raise ValueError("max_num_batched_tokens must be a positive integer")
        if type(self.enable_chunked_prefill) is not bool:
            raise ValueError("enable_chunked_prefill must be boolean")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0,1]")
        if type(self.expected_task_count) is not int or self.expected_task_count <= 0:
            raise ValueError("expected_task_count must be a positive integer")
        if (
            type(self.expected_single_image_count) is not int
            or not 0 <= self.expected_single_image_count <= self.expected_task_count
        ):
            raise ValueError(
                "expected_single_image_count must be in [0, expected_task_count]"
            )
        manifest_fields = (self.task_manifest_path, self.task_manifest_sha256)
        if (manifest_fields[0] is None) != (manifest_fields[1] is None):
            raise ValueError(
                "task manifest path and SHA256 must be configured together"
            )
        if self.task_manifest_sha256 is not None:
            digest = self.task_manifest_sha256
            if (
                len(digest) != _SHA256_LENGTH
                or digest != digest.lower()
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("task_manifest_sha256 must be lowercase SHA256")
        elif (
            self.expected_task_count != _LEGACY_COREDEV_TASK_COUNT
            or self.expected_single_image_count != _LEGACY_COREDEV_SINGLE_IMAGE_COUNT
        ):
            raise ValueError(
                "non-legacy task counts require an explicitly hashed task manifest"
            )
        if (
            self.schema_version == POLICY_BENCHMARK_SCHEMA
            and self.task_manifest_path is None
        ):
            raise ValueError(
                "generic policy benchmark schema requires an explicitly hashed task manifest"
            )
        common_snapshot_binding = (
            self.expected_policy_run_id,
            self.expected_policy_run_identity_sha256,
            self.expected_optimizer_step,
            self.expected_policy_weights_sha256,
        )
        if any(value is not None for value in common_snapshot_binding) and not all(
            value is not None for value in common_snapshot_binding
        ):
            raise ValueError("explicit policy snapshot binding fields are all-or-none")
        if not all(value is not None for value in common_snapshot_binding):
            raise ValueError(
                "v2 policy evaluation requires an explicit policy snapshot binding"
            )
        full_model_binding = (
            self.full_model_snapshot_manifest_path,
            self.full_model_snapshot_manifest_sha256,
            self.full_model_materialization_receipt_path,
            self.full_model_materialization_receipt_sha256,
            self.required_snapshot_identity_sha256,
        )
        if self.snapshot_backend == LORA_ADAPTER_EVALUATION_BACKEND:
            if self.lora_pointer_path is None:
                raise ValueError("LoRA snapshot backend requires a pointer path")
            if self.lora_pointer_sha256 is None:
                raise ValueError("v2 LoRA evaluation requires a pointer SHA256")
            if any(value is not None for value in full_model_binding):
                raise ValueError("LoRA snapshot backend forbids full-model bindings")
        else:
            if (
                self.lora_pointer_path is not None
                or self.lora_pointer_sha256 is not None
            ):
                raise ValueError("full-model snapshot backend forbids LoRA bindings")
            if not all(value is not None for value in full_model_binding):
                raise ValueError(
                    "full-model snapshot backend requires manifest, receipt, and identity bindings"
                )
            if (
                self.evaluation_protocol
                != DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
            ):
                raise ValueError(
                    "full-model snapshots are supported only by official-visible evaluation"
                )
        for name, digest in (
            ("lora_pointer_sha256", self.lora_pointer_sha256),
            (
                "expected_policy_run_identity_sha256",
                self.expected_policy_run_identity_sha256,
            ),
            ("expected_policy_weights_sha256", self.expected_policy_weights_sha256),
            (
                "full_model_snapshot_manifest_sha256",
                self.full_model_snapshot_manifest_sha256,
            ),
            (
                "full_model_materialization_receipt_sha256",
                self.full_model_materialization_receipt_sha256,
            ),
            (
                "required_snapshot_identity_sha256",
                self.required_snapshot_identity_sha256,
            ),
        ):
            if digest is not None:
                _require_sha256(digest, name=name)
        if self.expected_policy_run_id is not None and not self.expected_policy_run_id:
            raise ValueError("expected_policy_run_id must be non-empty")
        if self.expected_optimizer_step is not None and (
            type(self.expected_optimizer_step) is not int
            or self.expected_optimizer_step < 0
        ):
            raise ValueError("expected_optimizer_step must be a non-negative integer")
        if self.evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
            if (
                self.success_observation_protocol_id
                is not NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1
            ):
                raise ValueError(
                    "official-visible Crop requires the explicit matched60 "
                    "success observation protocol"
                )
            if self.action_boundary_protocol_id not in {
                NativeActionBoundaryProtocolId.LEGACY_ANSWER_OVER_ACTION_V1,
                NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2,
            }:
                raise ValueError(
                    "official-visible evaluation requires an explicit supported "
                    "action-boundary protocol"
                )
            if self.schema_version != POLICY_BENCHMARK_SCHEMA:
                raise ValueError(
                    "official-visible DeepEyes evaluation requires generic benchmark schema"
                )
            if (
                self.expected_optimizer_step != 0
                and self.snapshot_backend != FULL_MODEL_EVALUATION_BACKEND
            ):
                raise ValueError(
                    "official-visible nonzero evaluation requires a full-model snapshot"
                )
        elif (
            self.action_boundary_protocol_id
            is not NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
        ):
            raise ValueError(
                "training-run evaluation requires the explicit strict action boundary"
            )

    @property
    def uses_legacy_coredev_manifest(self) -> bool:
        return self.task_manifest_path is None


def load_policy_coredev_config(path: str | Path) -> PolicyCoreDevConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("policy benchmark config must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version in {
        POLICY_COREDEV_LEGACY_SCHEMA_V1,
        POLICY_BENCHMARK_LEGACY_SCHEMA_V1,
    }:
        raise ValueError(
            "legacy v1 policy evaluation config is immutable evidence only; "
            "materialize an explicit v2 config before evaluation"
        )
    if schema_version not in {POLICY_COREDEV_SCHEMA, POLICY_BENCHMARK_SCHEMA}:
        raise ValueError("policy benchmark config schema differs")
    required = {
        "schema_version",
        "evaluation_id",
        "policy_config_path",
        "lora_pointer_path",
        "output_root",
        "gpu_ids",
        "declared_image_max_pixels",
        "success_observation_protocol_id",
        "action_boundary_protocol_id",
        "evaluation_protocol",
        "inference_concurrency_per_gpu",
        "max_model_len",
        "gpu_memory_utilization",
    }
    optional = {
        "task_manifest_path",
        "task_manifest_sha256",
        "expected_task_count",
        "expected_single_image_count",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "lora_pointer_sha256",
        "expected_policy_run_id",
        "expected_policy_run_identity_sha256",
        "expected_optimizer_step",
        "expected_policy_weights_sha256",
        "snapshot_backend",
        "full_model_snapshot_manifest_path",
        "full_model_snapshot_manifest_sha256",
        "full_model_materialization_receipt_path",
        "full_model_materialization_receipt_sha256",
        "required_snapshot_identity_sha256",
    }
    if not required <= set(payload) or not set(payload) <= required | optional:
        raise ValueError("policy benchmark config fields differ")
    return PolicyCoreDevConfig(**payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


@dataclass(frozen=True, slots=True)
class PolicyEvaluationSnapshot:
    """One strictly loaded policy snapshot reused for the whole process."""

    run: PolicyE2ESmokeRunConfig
    lora: PolicyLoRASnapshot

    def __post_init__(self) -> None:
        if self.lora.policy_version.run_id != self.run.run_id:
            raise ValueError("policy snapshot run_id differs from policy config")
        if self.lora.run_identity_sha256 != self.run.identity_sha256:
            raise ValueError("policy snapshot run identity differs from policy config")

    @property
    def policy_version(self) -> PolicyVersion:
        return self.lora.policy_version


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


@dataclass(frozen=True, slots=True)
class VLLMLoRAAdapterIntegrityVerifier:
    """Re-read the exact private PEFT closure before vLLM can consume it.

    vLLM 0.12 does not expose a digest receipt for the adapter bytes loaded by
    the engine.  The verifier therefore binds the request path and checks all
    three files immediately before every ``generate``.  A same-UID writer can
    still race the small interval between that check and vLLM's lazy file read;
    ``residual_race`` records that limitation instead of claiming a sealed
    engine-side identity.
    """

    adapter_root: Path
    materialization_identity_sha256: str
    root_device: int
    root_inode: int
    adapter_model_bytes: bytes
    adapter_model_sha256: str
    adapter_config_bytes: bytes
    adapter_config_sha256: str
    identity_bytes: bytes
    identity_sha256: str
    engine_loaded_identity_attestation: str = VLLM_LORA_ENGINE_ATTESTATION
    residual_race: str = VLLM_LORA_RESIDUAL_RACE

    def __post_init__(self) -> None:
        root = Path(self.adapter_root)
        if not root.is_absolute():
            raise ValueError("vLLM LoRA adapter root must be absolute")
        object.__setattr__(
            self,
            "adapter_root",
            Path(os.path.abspath(os.fspath(root))),
        )
        _require_sha256(
            self.materialization_identity_sha256,
            name="vLLM LoRA materialization identity",
        )
        for name, payload, digest in self._expected_files():
            if not isinstance(payload, bytes) or not payload:
                raise ValueError(f"vLLM LoRA {name} bytes must be non-empty")
            _require_sha256(digest, name=f"vLLM LoRA {name} digest")
            if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), digest):
                raise ValueError(f"vLLM LoRA {name} digest differs from bytes")
        if self.root_device < 0 or self.root_inode <= 0:
            raise ValueError("vLLM LoRA root filesystem identity is invalid")
        if self.engine_loaded_identity_attestation != VLLM_LORA_ENGINE_ATTESTATION:
            raise ValueError("vLLM LoRA engine attestation statement changed")
        if self.residual_race != VLLM_LORA_RESIDUAL_RACE:
            raise ValueError("vLLM LoRA residual-race statement changed")

    def _expected_files(self) -> tuple[tuple[str, bytes, str], ...]:
        return (
            (
                VLLM_LORA_ADAPTER_MODEL_FILENAME,
                self.adapter_model_bytes,
                self.adapter_model_sha256,
            ),
            (
                VLLM_LORA_ADAPTER_CONFIG_FILENAME,
                self.adapter_config_bytes,
                self.adapter_config_sha256,
            ),
            (
                VLLM_LORA_ADAPTER_IDENTITY_FILENAME,
                self.identity_bytes,
                self.identity_sha256,
            ),
        )

    def verify(self, *, phase: str) -> None:
        if not isinstance(phase, str) or not phase:
            raise ValueError("vLLM LoRA verification phase must be non-empty")
        descriptor = _open_vllm_lora_adapter_root(
            self.adapter_root,
            owner=f"vLLM LoRA adapter during {phase}",
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                metadata.st_dev != self.root_device
                or metadata.st_ino != self.root_inode
            ):
                raise ReplayMismatchError(
                    f"vLLM LoRA adapter root changed during {phase}"
                )
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise ReplayMismatchError(
                    f"vLLM LoRA adapter root is not private during {phase}"
                )
            names = set(os.listdir(descriptor))
            expected_names = {
                name for name, _payload, _digest in self._expected_files()
            }
            if names != expected_names:
                raise ReplayMismatchError(
                    f"vLLM LoRA adapter files changed during {phase}"
                )
            for name, expected_bytes, expected_digest in self._expected_files():
                observed = _read_private_vllm_lora_file_at(
                    descriptor,
                    name,
                    phase=phase,
                )
                observed_digest = hashlib.sha256(observed).hexdigest()
                if not hmac.compare_digest(observed_digest, expected_digest):
                    raise ReplayMismatchError(
                        f"vLLM LoRA {name} digest changed during {phase}"
                    )
                if observed != expected_bytes:
                    raise ReplayMismatchError(
                        f"vLLM LoRA {name} bytes changed during {phase}"
                    )
        finally:
            os.close(descriptor)

    def assert_lora_request_binding(self, lora_request: object) -> None:
        request_path = getattr(lora_request, "lora_path", None)
        if not isinstance(request_path, str) or not request_path:
            raise TypeError("vLLM LoRARequest must expose its lora_path")
        normalized = Path(os.path.abspath(request_path))
        if normalized != self.adapter_root:
            raise IdentityMismatchError(
                "vLLM LoRARequest path differs from verified adapter root"
            )


def _vllm_lora_adapter_payloads(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSnapshot,
) -> tuple[str, bytes, bytes]:
    adapter_config = {
        "base_model_name_or_path": str(snapshot.run.model.revision_or_path),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 64,
        "revision": None,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    config_bytes = (json.dumps(adapter_config, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    materialization_content = {
        "schema_version": VLLM_LORA_ADAPTER_SCHEMA,
        "evaluation_id": config.evaluation_id,
        "optimizer_step": snapshot.policy_version.optimizer_step,
        "policy_run_id": snapshot.policy_version.run_id,
        "policy_run_identity_sha256": snapshot.lora.run_identity_sha256,
        "weights_sha256": snapshot.policy_version.weights_sha256,
        "pointer_file_sha256": snapshot.lora.pointer_file_sha256,
        "manifest_file_sha256": snapshot.lora.manifest_file_sha256,
        "adapter_model_file_sha256": snapshot.lora.tensor_file_sha256,
        "adapter_config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "base_model_name_or_path": str(snapshot.run.model.revision_or_path),
    }
    materialization_identity_sha256 = _canonical_json_sha256(materialization_content)
    identity = {
        **materialization_content,
        "materialization_identity_sha256": materialization_identity_sha256,
        "adapter_model_file": VLLM_LORA_ADAPTER_MODEL_FILENAME,
        "adapter_config_file": VLLM_LORA_ADAPTER_CONFIG_FILENAME,
        "engine_loaded_identity_attestation": VLLM_LORA_ENGINE_ATTESTATION,
        "residual_race": VLLM_LORA_RESIDUAL_RACE,
    }
    identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return materialization_identity_sha256, config_bytes, identity_bytes


def _open_absolute_directory_nofollow(
    path: Path,
    *,
    owner: str,
    create_missing: bool = False,
) -> int:
    """Traverse an absolute directory from ``/`` without following symlinks."""

    normalized = Path(os.path.abspath(os.fspath(path)))
    if not normalized.is_absolute():
        raise ValueError(f"{owner} path must be absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(os.sep, flags)
    completed = False
    try:
        for part in normalized.parts[1:]:
            next_descriptor: int | None = None
            try:
                try:
                    next_descriptor = os.open(part, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create_missing:
                        raise
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    next_descriptor = os.open(part, flags, dir_fd=descriptor)
                if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                    raise ReplayMismatchError(f"{owner} path contains a non-directory")
            except BaseException:
                if next_descriptor is not None:
                    os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        completed = True
        return descriptor
    except OSError as error:
        raise ReplayMismatchError(
            f"{owner} path is missing, unreadable, or contains a symlink"
        ) from error
    finally:
        if not completed:
            os.close(descriptor)


def _open_vllm_lora_adapter_root(path: Path, *, owner: str) -> int:
    return _open_absolute_directory_nofollow(path, owner=owner)


def _open_or_create_private_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    owner: str,
) -> int:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"{owner} name must be one safe path component")
    descriptor: int | None = None
    completed = False
    try:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            pass
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ReplayMismatchError(f"{owner} is not a current-user directory")
        os.fchmod(descriptor, 0o700)
        completed = True
        return descriptor
    except OSError as error:
        raise ReplayMismatchError(
            f"{owner} is unreadable, non-directory, or symlinked"
        ) from error
    finally:
        if descriptor is not None and not completed:
            os.close(descriptor)


def _read_private_vllm_lora_file_at(
    root_descriptor: int,
    name: str,
    *,
    phase: str,
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplayMismatchError(f"vLLM LoRA {name} is not regular during {phase}")
        if (
            metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
        ):
            raise ReplayMismatchError(f"vLLM LoRA {name} is not private during {phase}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    except OSError as error:
        raise ReplayMismatchError(
            f"vLLM LoRA {name} is missing, unreadable, or symlinked during {phase}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_private_vllm_lora_file_at(
    root_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ValueError("vLLM LoRA output name must be one safe path component")
    lock_acquired = False
    try:
        fcntl.flock(root_descriptor, fcntl.LOCK_EX)
        lock_acquired = True
        try:
            _assert_private_vllm_lora_file_equals_at(
                root_descriptor,
                name,
                payload,
            )
        except FileNotFoundError:
            pass
        else:
            return

        _publish_private_vllm_lora_file_at(root_descriptor, name, payload)
    finally:
        if lock_acquired:
            fcntl.flock(root_descriptor, fcntl.LOCK_UN)


def _assert_private_vllm_lora_file_equals_at(
    root_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplayMismatchError(f"vLLM LoRA output {name} is not a regular file")
        if metadata.st_uid != os.geteuid():
            raise ReplayMismatchError(
                f"vLLM LoRA output {name} is not owned by the current user"
            )
        if metadata.st_nlink != 1:
            raise ReplayMismatchError(
                f"vLLM LoRA output {name} has an unexpected hardlink"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        if b"".join(chunks) != payload:
            raise ReplayMismatchError(
                f"content-addressed vLLM LoRA output {name} differs"
            )
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ReplayMismatchError(
            f"vLLM LoRA output {name} is unreadable or symlinked"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_private_vllm_lora_file_at(
    root_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    """Publish exact bytes without replacing a concurrent immutable winner."""

    temporary_name = f".{name}.{uuid4().hex}.tmp"
    temporary_descriptor: int | None = None
    temporary_exists = False
    try:
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_descriptor,
        )
        temporary_exists = True
        view = memoryview(payload)
        while view:
            written = os.write(temporary_descriptor, view)
            if written <= 0:
                raise OSError("short write while materializing vLLM LoRA")
            view = view[written:]
        os.fchmod(temporary_descriptor, 0o600)
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=root_descriptor,
                dst_dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            _assert_private_vllm_lora_file_equals_at(
                root_descriptor,
                name,
                payload,
            )
        os.unlink(temporary_name, dir_fd=root_descriptor)
        temporary_exists = False
        os.fsync(root_descriptor)
    except OSError as error:
        raise ReplayMismatchError(
            f"could not publish private vLLM LoRA output {name}"
        ) from error
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=root_descriptor)
            except FileNotFoundError:
                pass


def build_vllm_lora_adapter_integrity_verifier(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSnapshot,
    adapter_root: Path,
) -> VLLMLoRAAdapterIntegrityVerifier:
    if not isinstance(snapshot, PolicyEvaluationSnapshot):
        raise TypeError("snapshot must be a PolicyEvaluationSnapshot")
    identity_sha256, config_bytes, identity_bytes = _vllm_lora_adapter_payloads(
        config,
        snapshot,
    )
    expected_root = config.output_root / "runtime" / "lora-adapters" / identity_sha256
    expected_root = Path(os.path.abspath(os.fspath(expected_root)))
    normalized_root = Path(os.path.abspath(os.fspath(adapter_root)))
    if normalized_root != expected_root:
        raise IdentityMismatchError(
            "vLLM LoRA adapter root differs from content identity"
        )
    descriptor = _open_vllm_lora_adapter_root(
        normalized_root,
        owner="vLLM LoRA adapter",
    )
    try:
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    verifier = VLLMLoRAAdapterIntegrityVerifier(
        adapter_root=normalized_root,
        materialization_identity_sha256=identity_sha256,
        root_device=metadata.st_dev,
        root_inode=metadata.st_ino,
        adapter_model_bytes=snapshot.lora.tensor_bytes,
        adapter_model_sha256=snapshot.lora.tensor_file_sha256,
        adapter_config_bytes=config_bytes,
        adapter_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        identity_bytes=identity_bytes,
        identity_sha256=hashlib.sha256(identity_bytes).hexdigest(),
    )
    verifier.verify(phase="verifier construction")
    return verifier


def materialize_vllm_lora_adapter(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSnapshot,
) -> Path:
    """Expose an exact runtime snapshot through vLLM's PEFT directory ABI.

    Creation is relative to progressively opened, no-follow directory fds and
    the lexical path is re-opened and verified before use.  A same-UID writer
    can still rename ancestors after the final verification; that remaining
    pre-consumption race is covered by ``VLLM_LORA_RESIDUAL_RACE``.
    """

    if not isinstance(snapshot, PolicyEvaluationSnapshot):
        raise TypeError("snapshot must be a PolicyEvaluationSnapshot")
    identity_sha256, config_bytes, identity_bytes = _vllm_lora_adapter_payloads(
        config,
        snapshot,
    )
    adapter_parent = Path(
        os.path.abspath(os.fspath(config.output_root / "runtime" / "lora-adapters"))
    )
    adapter_root = adapter_parent / identity_sha256
    parent_descriptor = _open_absolute_directory_nofollow(
        adapter_parent,
        owner="vLLM LoRA adapter parent",
        create_missing=True,
    )
    descriptor: int | None = None
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if parent_metadata.st_uid != os.geteuid():
            raise ReplayMismatchError(
                "vLLM LoRA adapter parent is not owned by the current user"
            )
        os.fchmod(parent_descriptor, 0o700)
        descriptor = _open_or_create_private_directory_at(
            parent_descriptor,
            identity_sha256,
            owner="content-addressed vLLM LoRA adapter",
        )
        _write_private_vllm_lora_file_at(
            descriptor,
            VLLM_LORA_ADAPTER_MODEL_FILENAME,
            snapshot.lora.tensor_bytes,
        )
        _write_private_vllm_lora_file_at(
            descriptor,
            VLLM_LORA_ADAPTER_CONFIG_FILENAME,
            config_bytes,
        )
        _write_private_vllm_lora_file_at(
            descriptor,
            VLLM_LORA_ADAPTER_IDENTITY_FILENAME,
            identity_bytes,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    build_vllm_lora_adapter_integrity_verifier(
        config,
        snapshot,
        adapter_root,
    ).verify(phase="post-materialization")
    return adapter_root


def policy_version_from_pointer(config: PolicyCoreDevConfig) -> PolicyVersion:
    """Backward-compatible strict one-shot policy identity loader."""

    return load_policy_evaluation_snapshot(config).policy_version


def policy_lora_request_name(snapshot: PolicyEvaluationSnapshot) -> str:
    """Name the vLLM adapter after the already frozen optimizer step."""

    step = snapshot.policy_version.optimizer_step
    if type(step) is not int or step < 0:
        raise ValueError("LoRA pointer optimizer_step must be a non-negative integer")
    return f"policy-step{step}"


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


def policy_benchmark_task_path(config: PolicyCoreDevConfig) -> Path:
    filename = (
        "coredev-official-tasks.jsonl"
        if config.uses_legacy_coredev_manifest
        else "policy-benchmark-tasks.jsonl"
    )
    return config.output_root / "runtime" / filename


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


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PolicyEvalContract:
    """Immutable identity for every behavior-relevant evaluation boundary."""

    evaluation_protocol: str
    training_image_max_pixels: int
    declared_image_max_pixels: int
    effective_image_max_pixels: int
    prompt_protocol_id: str
    prompt_identity_sha256: str
    parser_protocol_id: str
    parser_identity_sha256: str
    success_observation_protocol_id: NativeSuccessObservationProtocolId
    success_observation_identity_sha256: str
    action_boundary_protocol_id: NativeActionBoundaryProtocolId
    action_boundary_identity_sha256: str
    schema_version: str = POLICY_EVAL_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.evaluation_protocol not in POLICY_EVALUATION_PROTOCOLS:
            raise ValueError("evaluation contract protocol differs")
        for name in (
            "training_image_max_pixels",
            "declared_image_max_pixels",
            "effective_image_max_pixels",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "prompt_protocol_id",
            "parser_protocol_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "prompt_identity_sha256",
            "parser_identity_sha256",
            "success_observation_identity_sha256",
            "action_boundary_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if not isinstance(
            self.success_observation_protocol_id,
            NativeSuccessObservationProtocolId,
        ):
            raise TypeError(
                "success_observation_protocol_id must be an explicit protocol ID"
            )
        if not isinstance(
            self.action_boundary_protocol_id,
            NativeActionBoundaryProtocolId,
        ):
            raise TypeError(
                "action_boundary_protocol_id must be an explicit protocol ID"
            )

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_protocol": self.evaluation_protocol,
            "pixels": {
                "training_image_max_pixels": self.training_image_max_pixels,
                "declared_image_max_pixels": self.declared_image_max_pixels,
                "effective_image_max_pixels": self.effective_image_max_pixels,
            },
            "prompt": {
                "protocol_id": self.prompt_protocol_id,
                "identity_sha256": self.prompt_identity_sha256,
            },
            "parser": {
                "protocol_id": self.parser_protocol_id,
                "identity_sha256": self.parser_identity_sha256,
            },
            "success_observation": {
                "protocol_id": self.success_observation_protocol_id.value,
                "identity_sha256": self.success_observation_identity_sha256,
            },
            "action_boundary": {
                "protocol_id": self.action_boundary_protocol_id.value,
                "identity_sha256": self.action_boundary_identity_sha256,
            },
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_json_sha256(self.canonical_payload)


def _policy_eval_parser_identity(
    *,
    evaluation_protocol: str,
    run: PolicyE2ESmokeRunConfig,
) -> tuple[str, str]:
    if evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        from tgvf_rl.policy.deepeyes_official_protocol import (
            DEEPEYES_TOOL_NAME,
            DEEPEYES_TOOL_PARSER,
        )

        protocol_id = "deepeyes-hermes-last-complete-crop-call-v1"
        payload = {
            "implementation": (
                "tgvf_rl.policy.deepeyes_official_protocol.parse_hermes_crop_call"
            ),
            "upstream_parser": DEEPEYES_TOOL_PARSER,
            "enabled_tool_names": [DEEPEYES_TOOL_NAME],
            "multiple_complete_calls": "select_last",
        }
    else:
        protocol_id = "strict-native-single-tool-call-v1"
        payload = {
            "implementation": "tgvf_rl.protocol.parser.StrictToolCallParser",
            "enabled_tool_names": list(run.protocol.enabled_tool_names),
            "tool_schema_sha256": run.protocol.tool_schema_sha256,
            "complete_call_count": 1,
            "trailing_assistant_text": "reject",
        }
    return protocol_id, _canonical_json_sha256(payload)


def _policy_eval_action_boundary_identity(
    *,
    evaluation_protocol: str,
    run: PolicyE2ESmokeRunConfig,
    action_boundary_protocol_id: NativeActionBoundaryProtocolId,
) -> tuple[NativeActionBoundaryProtocolId, str]:
    sampling = run.policy.sampling
    if evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        payload: dict[str, object] = {
            "dispatcher": (
                "tgvf_rl.evaluation.policy_official_visible."
                "OfficialVisiblePolicyEvaluator"
            ),
            "boundary_classifier": (
                "tgvf_rl.protocol.action_boundary.classify_assistant_action_boundary"
            ),
            "tool_marker": "<tool_call>...</tool_call>",
            "required_request_stop_strings": list(
                getattr(sampling, "stop_strings", ()) or ()
            ),
            "required_request_stop_token_ids": list(
                getattr(sampling, "stop_token_ids", ()) or ()
            ),
            "include_stop_str_in_output": bool(
                getattr(sampling, "include_stop_str_in_output", False)
            ),
        }
        if (
            action_boundary_protocol_id
            is NativeActionBoundaryProtocolId.LEGACY_ANSWER_OVER_ACTION_V1
        ):
            payload.update(
                {
                    "trailing_final_answer_precedence": "final_answer",
                    "multiple_complete_calls": "execute_last",
                    "malformed_tool_call_tags": "reject",
                }
            )
        elif (
            action_boundary_protocol_id
            is NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
        ):
            payload.update(
                {
                    "complete_call_count": 1,
                    "terminal_tool_call_required": True,
                    "trailing_assistant_text": "reject",
                    "multiple_complete_calls": "reject",
                    "malformed_tool_call_tags": "reject",
                }
            )
        else:  # pragma: no cover - enum expansion requires an explicit contract
            raise ValueError("official-visible action-boundary protocol is unsupported")
    else:
        if (
            action_boundary_protocol_id
            is not NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
        ):
            raise ValueError("training-run evaluator requires strict boundary v2")
        payload = {
            "dispatcher": "tgvf_rl.environment.agent_loop.FrameworkNeutralAgentLoop",
            "tool_marker_precedence": "any_marker_routes_strict_parser",
            "multiple_complete_calls": "reject",
            "trailing_assistant_text": "reject",
            "cap_error_behavior": CapErrorBehavior.ONE_FINAL_ANSWER_TURN.value,
            "decoding": _decoding_contract().canonical_payload,
            "termination": _termination_contract(run).canonical_payload,
        }
    payload["protocol_id"] = action_boundary_protocol_id.value
    return action_boundary_protocol_id, _canonical_json_sha256(payload)


def _policy_eval_observation_identity(
    contract: NativeSuccessObservationContract,
) -> str:
    from tgvf_rl.environment.native_appender import (
        QWEN_NATIVE_IMAGE_PLACEHOLDER,
        QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT_SHA256,
        QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256,
        QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX,
        qwen_native_response_suffix,
    )
    from tgvf_rl.protocol.tool_prompts import (
        IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT_SHA256,
        QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER_SHA256,
        TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE_SHA256,
        TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE_SHA256,
    )

    if (
        contract.protocol_id
        is NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1
    ):
        return QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256
    if (
        contract.protocol_id
        is NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1
    ):
        return QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT_SHA256
    response_template_sha256 = {
        NativeToolCapabilityProfile.TGVF_ONLY: (
            TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE_SHA256
        ),
        NativeToolCapabilityProfile.CROP_ONLY: (
            IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT_SHA256
        ),
        NativeToolCapabilityProfile.CROP_TGVF: (
            TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE_SHA256
        ),
    }[contract.tool_profile]
    payload = {
        "protocol_id": contract.protocol_id.value,
        "tool_profile": contract.tool_profile.value,
        "assistant_dialect": contract.assistant_dialect.value,
        "prefix_sha256": hashlib.sha256(
            QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX.encode("utf-8")
        ).hexdigest(),
        "response_template_sha256": response_template_sha256,
        "image_placeholder_sha256": hashlib.sha256(
            QWEN_NATIVE_IMAGE_PLACEHOLDER.encode("utf-8")
        ).hexdigest(),
        "reasoning_reminder_sha256": (
            QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER_SHA256
            if contract.assistant_dialect is NativeAssistantDialect.QWEN3_VL_INSTRUCT
            else None
        ),
        "suffix_sha256": hashlib.sha256(
            qwen_native_response_suffix(contract.assistant_dialect).encode("utf-8")
        ).hexdigest(),
    }
    return _canonical_json_sha256(payload)


def effective_evaluation_image_max_pixels(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
) -> int:
    """Return the pixel cap actually consumed by the selected evaluator."""

    if config.evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        effective = snapshot.run.policy.image_max_pixels
        if config.declared_image_max_pixels != effective:
            raise ValueError(
                "official-visible declared pixels differ from its effective runtime"
            )
        return effective
    return config.declared_image_max_pixels


def build_policy_eval_contract(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
) -> PolicyEvalContract:
    """Bind pixels, prompt, parser, observation bytes, and action boundary.

    No observation renderer is inferred from a historical run schema.  The
    evaluation config must name it, and validation against the actual tool
    profile happens before any model or GPU runtime is constructed.
    """

    run = snapshot.run
    dialect = native_assistant_dialect_for_model(run.model.model_name)
    if config.evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        from tgvf_rl.policy.deepeyes_official_protocol import (
            DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA,
            VISUAL_PROMPT_IDENTITY,
        )

        observation_profile = NativeToolCapabilityProfile.CROP_ONLY
        prompt_protocol_id = (
            f"{DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA}:{VISUAL_PROMPT_IDENTITY.version}"
        )
        prompt_identity_sha256 = VISUAL_PROMPT_IDENTITY.bundle_sha256
    else:
        observation_profile = run.protocol.tool_profile
        prompt_protocol_id = "tgvf-native-run-prompt-v1"
        prompt_identity_sha256 = run.protocol.prompt_sha256

    observation_contract = NativeSuccessObservationContract(
        protocol_id=config.success_observation_protocol_id,
        tool_profile=observation_profile,
        assistant_dialect=dialect,
    )
    parser_protocol_id, parser_identity_sha256 = _policy_eval_parser_identity(
        evaluation_protocol=config.evaluation_protocol,
        run=run,
    )
    action_protocol_id, action_identity_sha256 = _policy_eval_action_boundary_identity(
        evaluation_protocol=config.evaluation_protocol,
        run=run,
        action_boundary_protocol_id=config.action_boundary_protocol_id,
    )
    effective_image_max_pixels = effective_evaluation_image_max_pixels(config, snapshot)
    return PolicyEvalContract(
        evaluation_protocol=config.evaluation_protocol,
        training_image_max_pixels=run.policy.image_max_pixels,
        declared_image_max_pixels=config.declared_image_max_pixels,
        effective_image_max_pixels=effective_image_max_pixels,
        prompt_protocol_id=prompt_protocol_id,
        prompt_identity_sha256=prompt_identity_sha256,
        parser_protocol_id=parser_protocol_id,
        parser_identity_sha256=parser_identity_sha256,
        success_observation_protocol_id=observation_contract.protocol_id,
        success_observation_identity_sha256=(
            _policy_eval_observation_identity(observation_contract)
        ),
        action_boundary_protocol_id=action_protocol_id,
        action_boundary_identity_sha256=action_identity_sha256,
    )


def _base_equivalent_step_zero_lora(
    snapshot: PolicyEvaluationSnapshot,
) -> dict[str, object]:
    """Prove that the step-zero adapter has an exactly zero LoRA delta."""

    if snapshot.policy_version.optimizer_step != 0:
        raise ValueError("base-equivalent LoRA proof requires optimizer step zero")
    tensors = snapshot.lora.tensors
    if not tensors:
        raise ValueError("step-zero LoRA snapshot contains no tensors")
    a_by_stem: dict[str, torch.Tensor] = {}
    b_by_stem: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if name.endswith(".lora_A.weight"):
            a_by_stem[name.removesuffix(".lora_A.weight")] = tensor
        elif name.endswith(".lora_B.weight"):
            b_by_stem[name.removesuffix(".lora_B.weight")] = tensor
        else:
            raise ValueError("step-zero snapshot contains a non-LoRA tensor")
    if not a_by_stem or set(a_by_stem) != set(b_by_stem):
        raise ValueError("step-zero LoRA A/B tensor names differ")
    b_evidence: list[dict[str, object]] = []
    for stem in sorted(b_by_stem):
        tensor = b_by_stem[stem]
        if torch.count_nonzero(tensor).item() != 0:
            raise ValueError("step-zero LoRA B tensor is not exactly zero")
        b_evidence.append(
            {
                "name": f"{stem}.lora_B.weight",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "tensor_sha256": tensor_checksum(tensor),
            }
        )
    proof_content = {
        "schema_version": "tgvf-base-equivalent-step-zero-lora-v1",
        "optimizer_step": 0,
        "weights_sha256": snapshot.policy_version.weights_sha256,
        "tensor_file_sha256": snapshot.lora.tensor_file_sha256,
        "lora_pair_count": len(a_by_stem),
        "only_lora_a_and_b": True,
        "all_lora_b_exactly_zero": True,
        "lora_b_tensors": b_evidence,
    }
    return {
        **proof_content,
        "proof_sha256": _canonical_json_sha256(proof_content),
    }


def _base_equivalent_step_zero_full_model(
    snapshot: FullModelEvaluationSnapshot,
) -> dict[str, object]:
    """Bind step zero to the run contract's exact immutable base-HF tree."""

    if (
        snapshot.policy_version.optimizer_step != 0
        or snapshot.manifest.source_kind is not FullModelSourceKind.BASE_HF
    ):
        raise ValueError("base-equivalent full-model proof requires base-HF step zero")
    content = {
        "schema_version": "tgvf-base-equivalent-step-zero-full-model-v1",
        "optimizer_step": 0,
        "source_kind": snapshot.manifest.source_kind.value,
        "source_is_bound_run_base_model": True,
        "snapshot_identity_sha256": snapshot.manifest.identity_sha256,
        "checkpoint_sha256": snapshot.manifest.checkpoint_sha256,
        "source_tree_sha256": snapshot.manifest.source_tree_sha256,
        "weights_sha256": snapshot.policy_version.weights_sha256,
        "materialized_model_tree_sha256": snapshot.receipt.model_tree_sha256,
    }
    return {**content, "proof_sha256": _canonical_json_sha256(content)}


def _evaluation_protocol_identity(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
) -> dict[str, object]:
    if config.evaluation_protocol == TRAINING_RUN_EVALUATION_PROTOCOL:
        if not isinstance(snapshot, PolicyEvaluationSnapshot):
            raise ValueError("training-run evaluation requires a LoRA snapshot")
        protocol = snapshot.run.protocol
        return {
            "profile": TRAINING_RUN_EVALUATION_PROTOCOL,
            "prompt_sha256": protocol.prompt_sha256,
            "tool_schema_sha256": protocol.tool_schema_sha256,
            "tool_profile": protocol.tool_profile.value,
            "enabled_tool_names": list(protocol.enabled_tool_names),
            "maximum_tool_calls": protocol.maximum_tool_calls,
            "native_pixels": False,
        }
    if config.evaluation_protocol != DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        raise ValueError("unsupported policy evaluation protocol")
    from tgvf_rl.policy.deepeyes_official_protocol import (
        DEEPEYES_MAX_ACTIVE_PERCEPTION,
        DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA,
        DEEPEYES_TOOL_NAME,
        DEEPEYES_TOOL_PARSER,
        SYSTEM_PROMPT_V2_SHA256,
        USER_PROMPT_V2_SHA256,
        VISUAL_PROMPT_IDENTITY,
    )
    from tgvf_rl.qwen.crop_coordinates import (
        QWEN3_CROP_CONVERSION_VERSION,
        QWEN3_CROP_COORDINATE_SPACE,
    )

    if snapshot.run.model.model_name != "Qwen3-VL-8B-Instruct":
        raise ValueError(
            "official-visible base evaluation requires Qwen3-VL-8B-Instruct"
        )
    identity: dict[str, object] = {
        "profile": DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
        "protocol_schema_version": DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA,
        "source_repository": "https://github.com/Visual-Agent/DeepEyes",
        "source_commit": "11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1",
        "prompt_source_path": ("verl/workers/agent/envs/mm_process_engine/prompt.py"),
        "prompt_source_file_sha256": (
            "35ef1bae8da550827bc53e23751e64d4c8eecc76d9170ea5673aa2493628cc23"
        ),
        "crop_source_path": (
            "verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py"
        ),
        "crop_source_file_sha256": (
            "0d56b2ff584fe56e68f20bbb4d25a9774ecbab605ad02cdaf1dac7cd6fa8bc60"
        ),
        "system_prompt_sha256": SYSTEM_PROMPT_V2_SHA256,
        "user_prompt_sha256": USER_PROMPT_V2_SHA256,
        "prompt_bundle_sha256": VISUAL_PROMPT_IDENTITY.bundle_sha256,
        "visible_system_tool_schema": True,
        "template_tools_argument": [],
        "tool_parser": DEEPEYES_TOOL_PARSER,
        "enabled_tool_names": [DEEPEYES_TOOL_NAME],
        "maximum_tool_calls": DEEPEYES_MAX_ACTIVE_PERCEPTION,
        "coordinate_mapper": "qwen_0_1000_to_source_v1",
        "crop_coordinate_space": QWEN3_CROP_COORDINATE_SPACE,
        "crop_coordinate_conversion_version": QWEN3_CROP_CONVERSION_VERSION,
        "crop_coordinate_reference_size": [1000, 1000],
        "crop_source": "immutable_original_image",
        "native_pixels": True,
        "precomputed_image_embeds": False,
        "image_max_pixels": effective_evaluation_image_max_pixels(config, snapshot),
        "native_image_limit_per_prompt": DEEPEYES_MAX_ACTIVE_PERCEPTION + 1,
        "observation_role": "user",
        "observation_envelope": (
            "<tool_response><image>USER_PROMPT_V2</tool_response>"
        ),
    }
    if snapshot.policy_version.optimizer_step == 0:
        identity["base_equivalence"] = (
            _base_equivalent_step_zero_lora(snapshot)
            if isinstance(snapshot, PolicyEvaluationSnapshot)
            else _base_equivalent_step_zero_full_model(snapshot)
        )
    return identity


def policy_evaluation_identity(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
) -> dict[str, object]:
    """Bind experiment, model, task population, and exact policy bytes."""

    task_path = policy_benchmark_task_path(config).resolve()
    task_sha256 = _sha256_file(task_path)
    if (
        config.task_manifest_sha256 is not None
        and task_sha256 != config.task_manifest_sha256
    ):
        raise ValueError("bound policy benchmark task manifest SHA256 changed")
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
        if isinstance(snapshot, PolicyEvaluationSnapshot)
        else full_model_snapshot_identity_record(snapshot)
    )
    policy_config_path = Path(
        getattr(config, "policy_config_path", snapshot.contract.source_path)
        if isinstance(snapshot, FullModelEvaluationSnapshot)
        else config.policy_config_path
    )
    eval_contract = build_policy_eval_contract(config, snapshot)
    content: dict[str, object] = {
        "schema_version": POLICY_EVALUATION_IDENTITY_SCHEMA,
        "evaluation_id": config.evaluation_id,
        "evaluation_schema_version": config.schema_version,
        "policy_config_path": str(policy_config_path.resolve()),
        "policy_config_file_sha256": _sha256_file(policy_config_path),
        "policy_run_config_identity_sha256": snapshot.run.identity_sha256,
        "model_identity": asdict(snapshot.run.model),
        "policy_snapshot": policy_snapshot,
        "task_manifest": {
            "path": str(task_path),
            "sha256": task_sha256,
            "task_count": config.expected_task_count,
            "single_image_count": config.expected_single_image_count,
        },
        "eval_contract": {
            **eval_contract.canonical_payload,
            "identity_sha256": eval_contract.identity_sha256,
        },
        "execution": {
            "world_size": len(config.gpu_ids),
            "gpu_ids": list(config.gpu_ids),
            "max_model_len": config.max_model_len,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "enable_chunked_prefill": config.enable_chunked_prefill,
            "inference_concurrency_per_gpu": config.inference_concurrency_per_gpu,
        },
    }
    if config.evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        content["protocol"] = _evaluation_protocol_identity(config, snapshot)
    return {**content, "identity_sha256": _canonical_json_sha256(content)}


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


def _single_collective(value: object, *, operation: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 1
    ):
        raise RuntimeError(f"{operation} requires one vLLM worker result")
    result = value[0]
    if not isinstance(result, Mapping):
        raise TypeError(f"{operation} returned a non-mapping utility result")
    return result


@dataclass(frozen=True, slots=True)
class _TurnRoute:
    backend_request_id: str
    output_ids: tuple[int, ...]
    optimizer_step: int


class StandaloneTGVFVLLMManager:
    """Small AsyncLLM adapter matching the already-audited training client ABI."""

    def __init__(
        self,
        engine: object,
        lora_request: object,
        *,
        capture_hidden: bool,
        native_pixels: bool = False,
        adapter_integrity_verifier: VLLMLoRAAdapterIntegrityVerifier | None = None,
    ) -> None:
        if lora_request is None:
            if adapter_integrity_verifier is not None:
                raise ValueError(
                    "full-model manager cannot receive a LoRA integrity verifier"
                )
        else:
            if not isinstance(
                adapter_integrity_verifier,
                VLLMLoRAAdapterIntegrityVerifier,
            ):
                raise TypeError(
                    "LoRA manager requires VLLMLoRAAdapterIntegrityVerifier"
                )
            adapter_integrity_verifier.assert_lora_request_binding(lora_request)
            adapter_integrity_verifier.verify(phase="manager construction")
        self.engine = engine
        self.lora_request = lora_request
        self.adapter_integrity_verifier = adapter_integrity_verifier
        self.capture_hidden = capture_hidden
        self.native_pixels = native_pixels
        self.turns: dict[str, _TurnRoute] = {}
        self.backend_ids: dict[str, list[str]] = {}

    async def materialize_source(
        self,
        *,
        request_id: str,
        expected_step: int,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_sha256: str,
    ) -> object:
        del expected_step
        result = await self.engine.collective_rpc(
            "tgvf_materialize_source",
            kwargs={
                "trajectory_id": request_id,
                "pixel_values_wire": _tensor_to_utility_wire(pixel_values),
                "image_grid_thw": tuple(int(v) for v in image_grid_thw[0].tolist()),
                "image_sha256": image_sha256,
            },
        )
        return _source_from_utility_wire(
            _single_collective(result, operation="source materialization")
        )

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        audio_data: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> object:
        del video_data, audio_data
        from vllm import SamplingParams

        step = int(kwargs.pop("tgvf_expected_step"))
        if kwargs:
            raise TypeError(f"unsupported standalone vLLM arguments: {sorted(kwargs)}")
        if self.adapter_integrity_verifier is not None:
            self.adapter_integrity_verifier.assert_lora_request_binding(
                self.lora_request
            )
            self.adapter_integrity_verifier.verify(phase="before engine.generate")
        backend_id = f"eval-{uuid4().hex}"
        maximum = sampling_params.get("max_tokens")
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("generation requires positive max_tokens")
        if self.capture_hidden:
            await self.engine.collective_rpc(
                "tgvf_register_behavior_trace",
                kwargs={
                    "request_id": backend_id,
                    "prompt_length": len(prompt_ids),
                    "maximum_output_tokens": maximum,
                },
            )
            self.backend_ids.setdefault(request_id, []).append(backend_id)
        prompt = {
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": {"image": image_data},
            "mm_processor_kwargs": mm_processor_kwargs,
        }
        parameters = dict(sampling_params)
        if parameters.get("logprobs") is True:
            parameters["logprobs"] = 0
        final = None
        adapter_arguments = (
            {} if self.lora_request is None else {"lora_request": self.lora_request}
        )
        async for output in self.engine.generate(
            prompt,
            SamplingParams(**parameters),
            backend_id,
            **adapter_arguments,
        ):
            final = output
        if self.adapter_integrity_verifier is not None:
            self.adapter_integrity_verifier.verify(phase="after engine.generate")
        if final is None or not final.finished or len(final.outputs) != 1:
            raise RuntimeError("standalone vLLM generation did not finish exactly once")
        completion = final.outputs[0]
        token_ids = tuple(int(value) for value in completion.token_ids)
        logprobs = []
        for token_id, position in zip(token_ids, completion.logprobs, strict=True):
            entry = position.get(token_id)
            if entry is None:
                raise RuntimeError("sampled token is absent from vLLM logprobs")
            logprobs.append(float(entry.logprob))
        self.turns[request_id] = _TurnRoute(backend_id, token_ids, step)
        return SimpleNamespace(
            token_ids=list(token_ids),
            log_probs=logprobs,
            stop_reason="completed",
            extra_fields={
                "global_steps": step,
                "min_global_steps": step,
                "max_global_steps": step,
                "logprobs_mode": "processed_logprobs",
                "tgvf_vllm_finish_reason": completion.finish_reason,
                "tgvf_vllm_stop_reason": completion.stop_reason,
            },
        )

    async def materialize_focus(
        self,
        *,
        request_id: str,
        expected_step: int,
        sampled_output_ids: tuple[int, ...],
        target_start: int,
        target_end: int,
        expected_target_token_ids: tuple[int, ...],
        provider: str,
    ) -> tuple[torch.Tensor, object]:
        turn = self._validated_turn(request_id, expected_step, sampled_output_ids)
        if turn.output_ids[target_start:target_end] != expected_target_token_ids:
            raise RuntimeError("focus target differs from sampled output")
        result = await self.engine.collective_rpc(
            "tgvf_materialize_focus",
            kwargs={
                "trajectory_id": request_id,
                "backend_request_id": turn.backend_request_id,
                "target_start": target_start,
                "target_end": target_end,
                "expected_target_token_ids": expected_target_token_ids,
                "provider": provider,
            },
        )
        typed = _focus_from_utility_wire(
            _single_collective(result, operation="focus materialization")
        )
        if not isinstance(typed, TGVFFocusMaterializationResult):
            raise TypeError("focus RPC returned an invalid result")
        return typed.hq, typed.observation

    async def materialize_crop(
        self,
        *,
        request_id: str,
        expected_step: int,
        sampled_output_ids: tuple[int, ...],
        call_index: int,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        crop_sha256: str,
    ) -> object:
        self._validated_turn(request_id, expected_step, sampled_output_ids)
        result = await self.engine.collective_rpc(
            "tgvf_materialize_crop",
            kwargs={
                "trajectory_id": request_id,
                "call_index": call_index,
                "pixel_values_wire": _tensor_to_utility_wire(pixel_values),
                "image_grid_thw": tuple(int(v) for v in image_grid_thw[0].tolist()),
                "crop_sha256": crop_sha256,
            },
        )
        return _source_from_utility_wire(
            _single_collective(result, operation="crop materialization")
        )

    def _validated_turn(
        self, request_id: str, expected_step: int, output_ids: tuple[int, ...]
    ) -> _TurnRoute:
        turn = self.turns.get(request_id)
        if turn is None or turn.output_ids != tuple(output_ids):
            raise RuntimeError("tool call differs from the last vLLM turn")
        if turn.optimizer_step != expected_step:
            raise RuntimeError("tool call policy step changed")
        return turn

    async def release_trajectory(self, request_id: str) -> None:
        backend_ids = tuple(self.backend_ids.pop(request_id, ()))
        self.turns.pop(request_id, None)
        if self.native_pixels:
            if backend_ids:
                raise RuntimeError(
                    "native-pixel evaluator unexpectedly registered hidden traces"
                )
            return
        await self.engine.collective_rpc(
            "tgvf_release_trajectory", args=(request_id, backend_ids)
        )


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


def _standalone_engine_kwargs(
    config: PolicyCoreDevConfig, run: PolicyE2ESmokeRunConfig
) -> dict[str, object]:
    """Build explicit vLLM arguments, including suite-specific context limits."""

    common: dict[str, object] = dict(
        model=run.model.revision_or_path,
        dtype="bfloat16",
        trust_remote_code=True,
        distributed_executor_backend="mp",
        max_model_len=config.max_model_len,
        max_num_seqs=config.inference_concurrency_per_gpu,
        max_num_batched_tokens=config.max_num_batched_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        enable_prefix_caching=False,
        gpu_memory_utilization=config.gpu_memory_utilization,
        logprobs_mode="processed_logprobs",
        enforce_eager=False,
        seed=run.rollout_rng.master_seed,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={
            "image": 1
            + (
                6
                if config.evaluation_protocol
                == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
                else run.protocol.maximum_tool_calls
            ),
            "video": 0,
        },
    )
    if config.evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        # This is the PRL13 comparison path: source/crop PIL images are handled
        # by stock Qwen3-VL/vLLM.  Loading the recorded-feature architecture or
        # worker extension here would silently turn the control into PRL11.
        return common
    return {
        **common,
        "worker_extension_cls": TGVF_VLLM_WORKER_EXTENSION_FQN,
        "enable_mm_embeds": True,
        "mm_encoder_attn_backend": TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
        "hf_overrides": {"architectures": [TGVF_QWEN3_VLLM_ARCHITECTURE]},
    }


def _decoding_contract() -> VLLMOutputDecodingContract:
    return VLLMOutputDecodingContract(
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        output_kind="final_only",
    )


def _termination_contract(run: PolicyE2ESmokeRunConfig) -> VLLMTurnTerminationContract:
    sampling = run.policy.sampling
    return VLLMTurnTerminationContract(
        required_request_stop_strings=tuple(sampling.stop_strings or ()),
        required_request_stop_token_ids=tuple(sampling.stop_token_ids or ()),
        include_stop_str_in_output=bool(sampling.include_stop_str_in_output),
        tool_call_terminal_suffixes=("",),
        tool_call_outcomes=(VLLMTerminationOutcome("stop", "</tool_call>"),),
        final_turn_outcomes=tuple(
            VLLMTerminationOutcome("stop", token_id)
            for token_id in tuple(sampling.stop_token_ids or ())
        )
        + (
            # vLLM 0.12 reports native EOS as finish_reason="stop" with no
            # separate stop_reason; this remains distinct from a length stop.
            VLLMTerminationOutcome("stop", None),
            VLLMTerminationOutcome("length", None),
        ),
    )


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
