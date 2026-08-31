"""Validated configuration schema for standalone policy evaluation.

This module owns the immutable policy-evaluation configuration contract.  It
is intentionally independent of the evaluator, task materializer, and vLLM
manager implementations so configuration parsing can evolve and be tested at
a narrow boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from tgvf_rl.protocol.action_boundary import NativeActionBoundaryProtocolId
from tgvf_rl.protocol.observation_contract import (
    NativeSuccessObservationProtocolId,
)


POLICY_COREDEV_LEGACY_SCHEMA_V1 = "tgvf-policy-coredev-evaluation-v1"
POLICY_BENCHMARK_LEGACY_SCHEMA_V1 = "tgvf-policy-benchmark-evaluation-v1"
POLICY_COREDEV_SCHEMA = "tgvf-policy-coredev-evaluation-v2"
POLICY_BENCHMARK_SCHEMA = "tgvf-policy-benchmark-evaluation-v2"
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
LORA_ADAPTER_EVALUATION_BACKEND = "lora_adapter"
FULL_MODEL_EVALUATION_BACKEND = "full_model"
POLICY_EVALUATION_BACKENDS = frozenset(
    {LORA_ADAPTER_EVALUATION_BACKEND, FULL_MODEL_EVALUATION_BACKEND}
)

_LEGACY_COREDEV_TASK_COUNT = 2511
_LEGACY_COREDEV_SINGLE_IMAGE_COUNT = 2240
_SHA256_LENGTH = 64


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
            not self.gpu_ids
            or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in self.gpu_ids)
            or len(set(self.gpu_ids)) != len(self.gpu_ids)
        ):
            raise ValueError(
                "policy benchmark evaluation requires at least one distinct "
                "non-negative GPU ID"
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


__all__ = [
    "DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL",
    "FULL_MODEL_EVALUATION_BACKEND",
    "LORA_ADAPTER_EVALUATION_BACKEND",
    "POLICY_BENCHMARK_LEGACY_SCHEMA_V1",
    "POLICY_BENCHMARK_SCHEMA",
    "POLICY_COREDEV_LEGACY_SCHEMA_V1",
    "POLICY_COREDEV_SCHEMA",
    "POLICY_EVALUATION_BACKENDS",
    "POLICY_EVALUATION_PROTOCOLS",
    "PolicyCoreDevConfig",
    "TRAINING_RUN_EVALUATION_PROTOCOL",
    "load_policy_coredev_config",
]
