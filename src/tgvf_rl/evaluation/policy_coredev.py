"""Standalone benchmark inference for completed visual-tool policy artifacts.

The training runtime already owns the native multi-turn protocol and the
colocated vLLM visual-tool implementation.  This module supplies only the
post-training boundary: an immutable LoRA or full-model snapshot, one vLLM
replica, and the official CoreDev prompt rows.  It deliberately performs no
reward or update.
"""

from __future__ import annotations

import asyncio
import ast
import csv
from dataclasses import asdict, replace
import fcntl
import hashlib
import io
import inspect
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
from tgvf_rl.contracts.errors import IdentityMismatchError, PolicyOutputContractError
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
from tgvf_rl.environment.native_appender import (
    render_qwen_native_matched_crop_tgvf_success_environment_text,
    render_qwen_native_matched_tgvf_success_environment_text,
    render_qwen_native_success_environment_text,
)
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
from tgvf_rl.environment.qwen3_crop_materializer import preprocess_qwen3_rgb
from tgvf_rl.framework.verl.native_agent_loop import VerlAsyncServerPolicyTurnClient
from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyLoRASnapshot,
    PolicyWeightSyncState,
    load_lora_snapshot_pointer,
)
from tgvf_rl.framework.verl.policy_live_runtime import (
    _BRANCH_LAYERS,
    _RemoteAtomicCropTGVFToolRuntime,
    _RemoteCropVisualMaterializer,
    _RemoteTGVFFocusToolRuntime,
    _VisualTokenCountResolver,
    _DisabledNoToolRuntime,
    _artifact_identity,
    _initial_vllm_inputs,
    _source_visual_positions,
)
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVF_VLLM_WORKER_EXTENSION_FQN,
    TGVFFocusMaterializationResult,
    TGVFCropMaterializationResult,
    _adapter_owned_state_to_utility_wire,
    _focus_from_utility_wire,
    _crop_tgvf_from_utility_wire,
    _source_from_utility_wire,
    _tensor_to_utility_wire,
    _validate_adapter_update_ack,
    adapter_owned_state_sha256,
)
from tgvf_rl.framework.vllm import (
    ContentAddressedVLLMTurnRNG,
    FastTokenizerTokenByteSpanDecoder,
    LiveVLLMTurnContextRegistry,
    Qwen3VLLMObservationPayloadResolver,
    VLLMOutputDecodingContract,
    VLLMPolicySampler,
    VLLMTerminationOutcome,
    VLLMTurnRNGIdentity,
    VLLMTurnTerminationContract,
    qwen3_vl_final_turn_outcomes,
)
from tgvf_rl.framework.vllm.registration import (
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
)
from tgvf_rl.observations.store import ObservationStore, tensor_checksum
from tgvf_rl.qwen import Qwen3VLAdapter
from tgvf_rl.policy.run_config import (
    POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    POLICY_E2E_RP66_MATCHED_RUN_CONFIG_SCHEMAS,
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.policy.no_tool_rl_protocol import (
    NO_TOOL_RL_PROMPT_IDENTITY,
    NO_TOOL_RL_PROMPT_VERSION,
    build_no_tool_visual_messages,
)
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_VERSION,
    build_crop_tgvf_visual_messages,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    TGVF_DEEPEYES_MATCHED_PROMPT_VERSION,
    build_tgvf_visual_messages,
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
    FullModelSourceKind,
    build_full_model_standalone_manager,
    full_model_snapshot_identity_record,
    load_full_model_evaluation_snapshot,
)
from .policy_paired_tgvf_snapshot import (
    PAIRED_TGVF_EVALUATION_BACKEND,
    PairedTGVFEvaluationSnapshot,
    load_paired_tgvf_snapshot,
    paired_tgvf_snapshot_identity_record,
)


POLICY_COREDEV_SCHEMA = "tgvf-policy-coredev-evaluation-v1"
POLICY_BENCHMARK_SCHEMA = "tgvf-policy-benchmark-evaluation-v1"
POLICY_EVALUATION_IDENTITY_SCHEMA = "tgvf-policy-evaluation-identity-v1"
POLICY_MATCHED_PROMPT_MATERIALIZER_SCHEMA = (
    "tgvf-policy-evaluation-matched-prompt-materializer-v1"
)
POLICY_MATCHED_PROMPT_MATERIALIZER_VERSION = (
    "qwen3-native-chat-template-explicit-empty-tools-v1"
)
POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA = "tgvf-policy-coredev-trajectory-audit-v1"
POLICY_OUTPUT_CONTRACT_FAILURE_SCHEMA = "tgvf-policy-output-contract-failure-v1"
PAIRED_POLICY_EVALUATION_RNG_SCHEMA = "tgvf-policy-paired-evaluation-rng-v1"
RESOLUTION_PAIRED_POLICY_EVALUATION_RNG_SCHEMA = "tgvf-policy-paired-evaluation-rng-v2"
IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION = "image_max_pixels_resolution_pair_v1"
IMAGE_MAX_PIXELS_RESOLUTION_PAIR_VALUES = (262144, 1003520)
IMAGE_MAX_PIXELS_RESOLUTION_PROJECTED_OPTIMIZER_STEPS = (32, 80)
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
_VLLM_SEED_MODULUS = 2**31 - 1
LORA_ADAPTER_EVALUATION_BACKEND = "lora_adapter"
POLICY_EVALUATION_BACKENDS = frozenset(
    {
        LORA_ADAPTER_EVALUATION_BACKEND,
        FULL_MODEL_EVALUATION_BACKEND,
        PAIRED_TGVF_EVALUATION_BACKEND,
    }
)


def _success_environment_text_renderer(run: PolicyE2ESmokeRunConfig):
    if run.schema_version == POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA:
        return render_qwen_native_matched_crop_tgvf_success_environment_text
    return (
        render_qwen_native_matched_tgvf_success_environment_text
        if run.schema_version in POLICY_E2E_RP66_MATCHED_RUN_CONFIG_SCHEMAS
        else render_qwen_native_success_environment_text
    )


def _matched_prompt_materializer_identity(
    run: PolicyE2ESmokeRunConfig,
) -> dict[str, object] | None:
    """Bind the corrected training-matched prompt path without legacy drift."""

    if run.schema_version in POLICY_E2E_RP66_MATCHED_RUN_CONFIG_SCHEMAS:
        if run.protocol.tool_profile is not NativeToolCapabilityProfile.TGVF_ONLY:
            raise ValueError("matched TGVF run has a non-TGVF tool profile")
        builder = "build_tgvf_visual_messages"
        prompt_version = TGVF_DEEPEYES_MATCHED_PROMPT_VERSION
        prompt_sha256 = TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    elif run.schema_version == POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA:
        if run.protocol.tool_profile is not NativeToolCapabilityProfile.CROP_TGVF:
            raise ValueError("matched Crop+TGVF run has a non-combined tool profile")
        builder = "build_crop_tgvf_visual_messages"
        prompt_version = CROP_TGVF_DEEPEYES_MATCHED_PROMPT_VERSION
        prompt_sha256 = CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    elif run.schema_version == POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMA:
        if run.protocol.tool_profile is not NativeToolCapabilityProfile.NO_TOOL:
            raise ValueError("matched no-tool run has a non-empty tool profile")
        if run.protocol.enabled_tool_names:
            raise ValueError("matched no-tool run exposes tool names")
        builder = "build_no_tool_visual_messages"
        prompt_version = NO_TOOL_RL_PROMPT_VERSION
        prompt_sha256 = NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256
    else:
        # Existing generic native and official-visible evaluations retain their
        # historical rendering and identity bytes.
        return None
    return {
        "schema_version": POLICY_MATCHED_PROMPT_MATERIALIZER_SCHEMA,
        "version": POLICY_MATCHED_PROMPT_MATERIALIZER_VERSION,
        "message_builder": builder,
        "prompt_version": prompt_version,
        "prompt_bundle_sha256": prompt_sha256,
        "template_tools_argument": [],
    }


def _evaluation_prompt_materializer_identity(
    config: PolicyCoreDevConfig,
    run: PolicyE2ESmokeRunConfig,
) -> dict[str, object] | None:
    """Keep official-visible and legacy identity envelopes byte-compatible."""

    if config.evaluation_protocol != TRAINING_RUN_EVALUATION_PROTOCOL:
        return None
    return _matched_prompt_materializer_identity(run)


def _render_training_run_visual_prompt(
    *,
    run: PolicyE2ESmokeRunConfig,
    processor: object,
    renderer: NativeProtocolRenderer,
    question: str,
) -> tuple[str, tuple[int, ...]]:
    """Render one training-run prompt, preserving all historical routes."""

    materializer = _matched_prompt_materializer_identity(run)
    if materializer is None:
        messages = build_visual_tool_prompt_messages(
            question,
            tool_profile=run.protocol.tool_profile,
            assistant_dialect=renderer.assistant_dialect,
        )
        rendered = renderer.render(messages, add_generation_prompt=True)
        renderer.assert_generation_prefill(rendered, renderer.tokenizer)
        return rendered.text, rendered.token_ids

    if run.schema_version == POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA:
        messages = build_crop_tgvf_visual_messages(question)
    elif run.schema_version == POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMA:
        messages = build_no_tool_visual_messages(question)
    else:
        messages = build_tgvf_visual_messages(question)
    renderer.assert_tokenizer_length()
    renderer.assert_chat_template_identity()
    renderer.assert_tool_schema_identity()
    text = processor.apply_chat_template(
        list(messages),
        tools=[],
        tokenize=False,
        add_generation_prompt=True,
    )
    renderer.assert_tokenizer_length()
    renderer.assert_chat_template_identity()
    renderer.assert_tool_schema_identity()
    if not isinstance(text, str):
        raise TypeError("matched evaluation chat template did not return text")
    if "<answer>" in text or "</answer>" in text:
        raise ValueError("matched evaluation prompt contains an answer wrapper")
    token_ids = tuple(renderer.tokenizer.encode(text, add_special_tokens=False))
    if not token_ids or any(
        type(token_id) is not int or token_id < 0 for token_id in token_ids
    ):
        raise TypeError("matched evaluation tokenizer returned invalid prompt IDs")
    renderer.assert_tokenizer_length()
    renderer.assert_chat_template_identity()
    renderer.assert_tool_schema_identity()
    renderer.assert_generation_prefill(
        SimpleNamespace(text=text, token_ids=token_ids), renderer.tokenizer
    )
    return text, token_ids


def _build_remote_tgvf_focus_tool_runtime(
    *,
    event_loop: asyncio.AbstractEventLoop,
    server_client: object,
    config: object,
    source_visual: object,
    layout_builder: object,
    observation_store: ObservationStore,
    execution_ledger: FocusExecutionLedger,
    contextual_forward_identity: object | None,
    branch_merger_identities: tuple[object, ...],
    success_environment_text_renderer: object,
    assistant_dialect: object,
) -> _RemoteTGVFFocusToolRuntime:
    """Single construction boundary shared by preflight and live inference."""

    return _RemoteTGVFFocusToolRuntime(
        event_loop=event_loop,
        server_client=server_client,
        config=config,
        source_visual=source_visual,
        layout_builder=layout_builder,
        observation_store=observation_store,
        execution_ledger=execution_ledger,
        contextual_forward_identity=contextual_forward_identity,
        branch_merger_identities=branch_merger_identities,
        success_environment_text_renderer=success_environment_text_renderer,
        assistant_dialect=assistant_dialect,
    )


def validate_policy_benchmark_runtime_interfaces(
    run: PolicyE2ESmokeRunConfig,
    *,
    image_max_pixels: int | None = None,
) -> dict[str, object]:
    """Exercise CPU-only evaluator interfaces before any vLLM construction."""

    effective_image_max_pixels = (
        run.policy.image_max_pixels if image_max_pixels is None else image_max_pixels
    )
    if type(effective_image_max_pixels) is not int or effective_image_max_pixels <= 0:
        raise ValueError("policy benchmark image_max_pixels must be positive")

    from tgvf_rl.framework.verl.smoke_dataset import (
        _materialize_source_image_prompt_token_ids,
    )

    inspect.signature(_materialize_source_image_prompt_token_ids).bind(
        processor=object(),
        canonical_token_ids=(),
        prompt_text="",
        source_rgb=torch.zeros((1, 1, 3), dtype=torch.uint8),
        image_max_pixels=1,
    )
    dialect = native_assistant_dialect_for_model(run.model.model_name)
    renderer = _success_environment_text_renderer(run)
    profile = run.protocol.tool_profile
    manager_method = {
        NativeToolCapabilityProfile.NO_TOOL: None,
        NativeToolCapabilityProfile.TGVF_ONLY: "materialize_focus",
        NativeToolCapabilityProfile.CROP_ONLY: "materialize_crop",
        NativeToolCapabilityProfile.CROP_TGVF: "materialize_crop_tgvf",
    }.get(profile)
    if profile is not NativeToolCapabilityProfile.NO_TOOL and manager_method is None:
        raise ValueError("policy benchmark has an unsupported tool profile")
    if manager_method is not None and not callable(
        getattr(StandaloneTGVFVLLMManager, manager_method, None)
    ):
        raise TypeError(f"standalone evaluator lacks {manager_method}()")

    event_loop = asyncio.new_event_loop()
    try:
        if profile is NativeToolCapabilityProfile.NO_TOOL:
            runtime = _DisabledNoToolRuntime()
        elif profile is NativeToolCapabilityProfile.TGVF_ONLY:
            runtime = _build_remote_tgvf_focus_tool_runtime(
                event_loop=event_loop,
                server_client=object(),
                config=run,
                source_visual=object(),
                layout_builder=object(),
                observation_store=ObservationStore(),
                execution_ledger=FocusExecutionLedger(),
                contextual_forward_identity=None,
                branch_merger_identities=(),
                success_environment_text_renderer=renderer,
                assistant_dialect=dialect,
            )
        elif profile is NativeToolCapabilityProfile.CROP_ONLY:
            runtime = _RemoteCropVisualMaterializer(
                event_loop=event_loop,
                server_client=object(),
                processor=object(),
                model_identity=run.model,
                image_max_pixels=effective_image_max_pixels,
                trajectory_id="cpu-preflight",
                behavior_policy=PolicyVersion("cpu-preflight", 0, "0" * 64),
            )
        else:
            zero = torch.zeros((1, 1), dtype=torch.bfloat16)
            source = SourceVisualTensorBundle(
                image_sha256="0" * 64,
                premerge_main=zero,
                premerge_deepstack=(),
                merged_main=zero,
                merged_deepstack=(),
                image_grid_thw=(1, 1, 1),
                spatial_merge_size=1,
                decoded_rgb_sha256="0" * 64,
            )
            runtime = _RemoteAtomicCropTGVFToolRuntime(
                event_loop=event_loop,
                server_client=object(),
                config=run,
                source_visual=source,
                layout_builder=object(),
                observation_store=ObservationStore(),
                execution_ledger=FocusExecutionLedger(),
                contextual_forward_identity=None,
                branch_merger_identities=(),
                crop_processor_identity=_artifact_identity(
                    "policy-evaluation-preflight",
                    "crop-processor",
                    "v1",
                    {"profile": profile.value},
                ),
                crop_layout_identity=_artifact_identity(
                    "policy-evaluation-preflight",
                    "crop-layout",
                    "v1",
                    {"profile": profile.value},
                ),
                processor=object(),
                image_max_pixels=effective_image_max_pixels,
                success_environment_text_renderer=renderer,
                assistant_dialect=dialect,
            )
    finally:
        event_loop.close()
    result = {
        "source_rgb_prompt_materializer": True,
        "tool_profile": profile.value,
        "standalone_manager_method": manager_method,
        "remote_tool_runtime": type(runtime).__name__,
        "success_environment_renderer": renderer.__name__,
        "assistant_dialect": dialect.value,
    }
    if profile is NativeToolCapabilityProfile.TGVF_ONLY:
        result["remote_tgvf_focus_runtime"] = type(runtime).__name__
    return result


@dataclass(frozen=True, slots=True)
class PolicyCoreDevConfig:
    evaluation_id: str
    policy_config_path: Path
    lora_pointer_path: Path | None
    output_root: Path
    gpu_ids: tuple[int, ...]
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
    paired_snapshot_receipt_path: Path | None = None
    paired_snapshot_receipt_sha256: str | None = None
    evaluation_protocol: str = TRAINING_RUN_EVALUATION_PROTOCOL
    paired_seed_namespace: str | None = None
    paired_rng_protocol_projection: str | None = None
    evaluation_image_max_pixels: int | None = None
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
            "paired_snapshot_receipt_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        object.__setattr__(self, "gpu_ids", tuple(self.gpu_ids))
        if self.schema_version not in {POLICY_COREDEV_SCHEMA, POLICY_BENCHMARK_SCHEMA}:
            raise ValueError("policy benchmark config schema differs")
        if self.evaluation_protocol not in POLICY_EVALUATION_PROTOCOLS:
            raise ValueError("policy benchmark evaluation protocol differs")
        if self.snapshot_backend not in POLICY_EVALUATION_BACKENDS:
            raise ValueError("policy benchmark snapshot backend differs")
        if not self.evaluation_id:
            raise ValueError("evaluation_id must be non-empty")
        if self.paired_seed_namespace is not None:
            namespace = self.paired_seed_namespace
            if (
                not isinstance(namespace, str)
                or not namespace
                or namespace.strip() != namespace
                or any(character.isspace() for character in namespace)
            ):
                raise ValueError(
                    "paired_seed_namespace must be a non-empty canonical string"
                )
            if self.schema_version != POLICY_BENCHMARK_SCHEMA:
                raise ValueError(
                    "paired evaluation RNG requires the generic benchmark schema"
                )
            if self.evaluation_protocol == TRAINING_RUN_EVALUATION_PROTOCOL:
                pass
            elif not (
                self.evaluation_protocol
                == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
                and self.snapshot_backend == FULL_MODEL_EVALUATION_BACKEND
            ):
                raise ValueError(
                    "paired evaluation RNG requires training-run evaluation or "
                    "official-visible full-model evaluation"
                )
            if self.task_manifest_sha256 is None:
                raise ValueError(
                    "paired evaluation RNG requires an explicitly hashed task manifest"
                )
        if self.evaluation_image_max_pixels is not None and (
            type(self.evaluation_image_max_pixels) is not int
            or self.evaluation_image_max_pixels <= 0
        ):
            raise ValueError("evaluation_image_max_pixels must be a positive integer")
        if self.paired_rng_protocol_projection is not None:
            if (
                self.paired_rng_protocol_projection
                != IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION
            ):
                raise ValueError("paired RNG protocol projection differs")
            if (
                self.schema_version != POLICY_BENCHMARK_SCHEMA
                or self.paired_seed_namespace is None
                or self.evaluation_protocol
                != DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
                or self.snapshot_backend != FULL_MODEL_EVALUATION_BACKEND
                or self.expected_optimizer_step
                not in IMAGE_MAX_PIXELS_RESOLUTION_PROJECTED_OPTIMIZER_STEPS
                or self.evaluation_image_max_pixels
                not in IMAGE_MAX_PIXELS_RESOLUTION_PAIR_VALUES
                or (
                    self.expected_optimizer_step == 32
                    and self.evaluation_image_max_pixels != 1003520
                )
            ):
                raise ValueError(
                    "image_max_pixels resolution projection requires the exact "
                    "official-visible full-model step80 pair or step32 true1M "
                    "extension contract"
                )
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
        if self.schema_version == POLICY_BENCHMARK_SCHEMA and not all(
            value is not None for value in common_snapshot_binding
        ):
            raise ValueError(
                "generic policy benchmark requires an explicit policy snapshot binding"
            )
        full_model_binding = (
            self.full_model_snapshot_manifest_path,
            self.full_model_snapshot_manifest_sha256,
            self.full_model_materialization_receipt_path,
            self.full_model_materialization_receipt_sha256,
            self.required_snapshot_identity_sha256,
        )
        paired_binding = (
            self.paired_snapshot_receipt_path,
            self.paired_snapshot_receipt_sha256,
        )
        if self.snapshot_backend == LORA_ADAPTER_EVALUATION_BACKEND:
            if self.lora_pointer_path is None:
                raise ValueError("LoRA snapshot backend requires a pointer path")
            if self.schema_version == POLICY_BENCHMARK_SCHEMA and (
                self.lora_pointer_sha256 is None
            ):
                raise ValueError("generic LoRA benchmark requires a pointer SHA256")
            if any(value is not None for value in full_model_binding):
                raise ValueError("LoRA snapshot backend forbids full-model bindings")
            if any(value is not None for value in paired_binding):
                raise ValueError("LoRA snapshot backend forbids paired bindings")
        elif self.snapshot_backend == FULL_MODEL_EVALUATION_BACKEND:
            if (
                self.lora_pointer_path is not None
                or self.lora_pointer_sha256 is not None
            ):
                raise ValueError("full-model snapshot backend forbids LoRA bindings")
            if not all(value is not None for value in full_model_binding):
                raise ValueError(
                    "full-model snapshot backend requires manifest, receipt, and identity bindings"
                )
            if self.evaluation_protocol not in {
                DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
                TRAINING_RUN_EVALUATION_PROTOCOL,
            }:
                raise ValueError(
                    "full-model snapshots require official-visible or training-run evaluation"
                )
            if any(value is not None for value in paired_binding):
                raise ValueError("full-model snapshot backend forbids paired bindings")
        else:
            if (
                self.lora_pointer_path is not None
                or self.lora_pointer_sha256 is not None
            ):
                raise ValueError("paired snapshot backend forbids policy-LoRA bindings")
            if any(value is not None for value in full_model_binding):
                raise ValueError(
                    "paired snapshot backend forbids Crop full-model bindings"
                )
            if not all(value is not None for value in paired_binding):
                raise ValueError("paired snapshot backend requires its receipt binding")
            if self.evaluation_protocol != TRAINING_RUN_EVALUATION_PROTOCOL:
                raise ValueError(
                    "paired TGVF snapshots require training-run evaluation"
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
            ("paired_snapshot_receipt_sha256", self.paired_snapshot_receipt_sha256),
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

    @property
    def uses_legacy_coredev_manifest(self) -> bool:
        return self.task_manifest_path is None


def evaluation_image_max_pixels(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
) -> int:
    """Resolve a benchmark-only pixel cap without mutating checkpoint identity."""

    frozen_value = snapshot.run.policy.image_max_pixels
    override = getattr(config, "evaluation_image_max_pixels", None)
    value = frozen_value if override is None else override
    if type(value) is not int or value <= 0:
        raise ValueError("resolved evaluation image max pixels must be positive")
    return value


def load_policy_coredev_config(path: str | Path) -> PolicyCoreDevConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "evaluation_id",
        "policy_config_path",
        "lora_pointer_path",
        "output_root",
        "gpu_ids",
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
        "paired_snapshot_receipt_path",
        "paired_snapshot_receipt_sha256",
        "evaluation_protocol",
        "paired_seed_namespace",
        "paired_rng_protocol_projection",
        "evaluation_image_max_pixels",
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


PolicyEvaluationSubject = (
    PolicyEvaluationSnapshot
    | FullModelEvaluationSnapshot
    | PairedTGVFEvaluationSnapshot
)


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
        expected_config_sha256 = (
            snapshot.manifest.run_contract_file_sha256
            if snapshot.manifest.checkpoint_owner is None
            else snapshot.manifest.checkpoint_owner.config_file_sha256
        )
        if _sha256_file(config.policy_config_path) != expected_config_sha256:
            raise ValueError(f"{owner} full-model owner config bytes differ")
    if isinstance(snapshot, PairedTGVFEvaluationSnapshot):
        if config.required_snapshot_identity_sha256 is not None:
            raise ValueError("paired snapshot must not use Crop snapshot identity")
        if config.paired_snapshot_receipt_sha256 is None:
            raise ValueError("paired snapshot receipt binding is absent")
        if _sha256_file(config.policy_config_path) != (
            snapshot.receipt.policy_config_sha256
        ):
            raise ValueError(f"{owner} paired run config bytes differ")


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
        manifest_path, receipt_path, runtime_lightweight=True
    )
    _assert_policy_snapshot_binding(config, snapshot, owner="full-model snapshot")
    if config.evaluation_protocol == TRAINING_RUN_EVALUATION_PROTOCOL:
        if snapshot.manifest.checkpoint_owner is None:
            raise ValueError(
                "training-run full-model evaluation requires a checkpoint owner"
            )
        run = load_policy_e2e_smoke_run_config(
            config.policy_config_path, allow_external_agent_loop_config=True
        )
        if (
            run.schema_version != POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMA
            or run.protocol.tool_profile is not NativeToolCapabilityProfile.NO_TOOL
            or run.protocol.enabled_tool_names
        ):
            raise ValueError(
                "training-run full-model evaluation is restricted to matched no-tool runs"
            )
        if (
            run.run_id != snapshot.policy_version.run_id
            or run.identity_sha256 != snapshot.run_identity_sha256
        ):
            raise ValueError("full-model checkpoint owner run identity differs")
        snapshot = replace(snapshot, run=run)
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
    if config.snapshot_backend == PAIRED_TGVF_EVALUATION_BACKEND:
        assert config.paired_snapshot_receipt_path is not None
        assert config.paired_snapshot_receipt_sha256 is not None
        if _sha256_file(config.paired_snapshot_receipt_path) != (
            config.paired_snapshot_receipt_sha256
        ):
            raise ValueError("paired TGVF snapshot receipt file SHA256 differs")
        loaded = load_paired_tgvf_snapshot(
            config.paired_snapshot_receipt_path, runtime_lightweight=True
        )
        _assert_policy_snapshot_binding(config, loaded, owner="paired TGVF snapshot")
        return loaded

    run = load_policy_e2e_smoke_run_config(
        config.policy_config_path, allow_external_agent_loop_config=True
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
    return config.output_root / "runtime" / "frozen-full-model-state"


def frozen_paired_tgvf_state_root(config: PolicyCoreDevConfig) -> Path:
    return config.output_root / "runtime" / "frozen-paired-tgvf-state"


def _write_immutable_snapshot_file(path: Path, payload: bytes) -> None:
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
    """Copy verified identity records into evaluation-private storage."""

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
    if isinstance(snapshot, PairedTGVFEvaluationSnapshot):
        assert config.paired_snapshot_receipt_path is not None
        _write_immutable_snapshot_file(
            frozen_paired_tgvf_state_root(config) / "snapshot-receipt.json",
            _read_regular_file_bytes(
                config.paired_snapshot_receipt_path,
                owner="paired TGVF snapshot receipt",
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
    """Load only the evaluation-private identity records, never mutable latest."""

    if config.snapshot_backend == FULL_MODEL_EVALUATION_BACKEND:
        frozen_root = frozen_full_model_state_root(config)
        return _load_full_model_from_paths(
            config,
            manifest_path=frozen_root / "snapshot-manifest.json",
            receipt_path=frozen_root / "materialization-receipt.json",
        )
    if config.snapshot_backend == PAIRED_TGVF_EVALUATION_BACKEND:
        receipt = frozen_paired_tgvf_state_root(config) / "snapshot-receipt.json"
        assert config.paired_snapshot_receipt_sha256 is not None
        if _sha256_file(receipt) != config.paired_snapshot_receipt_sha256:
            raise ValueError("frozen paired TGVF receipt SHA256 differs")
        loaded = load_paired_tgvf_snapshot(receipt, runtime_lightweight=True)
        _assert_policy_snapshot_binding(
            config, loaded, owner="frozen paired TGVF snapshot"
        )
        return loaded

    run = load_policy_e2e_smoke_run_config(
        config.policy_config_path, allow_external_agent_loop_config=True
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


def _write_private_file_exact(path: Path, payload: bytes) -> None:
    """Write exact bytes without retaining a mutable hardlink to the source."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"evaluation output is not a regular file: {path}")
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable evaluation output differs: {path}")
        if path.stat().st_nlink == 1:
            return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_vllm_lora_adapter(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSnapshot,
) -> Path:
    """Expose an exact runtime snapshot through vLLM's PEFT directory ABI."""

    if not isinstance(snapshot, PolicyEvaluationSnapshot):
        raise TypeError("snapshot must be a PolicyEvaluationSnapshot")
    weights_sha256 = snapshot.policy_version.weights_sha256
    adapter_root = config.output_root / "runtime" / "lora-adapter"
    adapter_root.mkdir(parents=True, exist_ok=True)
    model_file = adapter_root / "adapter_model.safetensors"
    _write_private_file_exact(model_file, snapshot.lora.tensor_bytes)
    if _sha256_file(model_file) != snapshot.lora.tensor_file_sha256:
        raise RuntimeError("materialized vLLM LoRA differs from snapshot")
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
    adapter_config_path = adapter_root / "adapter_config.json"
    _write_private_file_exact(adapter_config_path, config_bytes)
    identity = {
        "schema_version": "tgvf-policy-vllm-lora-adapter-v1",
        "evaluation_id": config.evaluation_id,
        "optimizer_step": snapshot.policy_version.optimizer_step,
        "policy_run_id": snapshot.policy_version.run_id,
        "policy_run_identity_sha256": snapshot.lora.run_identity_sha256,
        "weights_sha256": weights_sha256,
        "pointer_file_sha256": snapshot.lora.pointer_file_sha256,
        "manifest_file_sha256": snapshot.lora.manifest_file_sha256,
        "tensor_file_sha256": snapshot.lora.tensor_file_sha256,
        "snapshot": str(snapshot.lora.tensor_file),
    }
    identity_path = adapter_root / "identity.json"
    identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_private_file_exact(identity_path, identity_bytes)
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
class PairedEvaluationVLLMTurnRNG:
    """Common-random-number stream used only by benchmark evaluation.

    Unlike the training RNG, this evaluator-owned stream deliberately excludes
    the evaluation ID, arm name, optimizer step, checkpoint hash, behavior
    policy identity, and evolving prompt tokens.  Consequently the same task
    and assistant-turn index receive the same vLLM seed at every compared
    checkpoint, while the ordinary rollout RNG remains completely unchanged.
    """

    master_seed: int
    seed_namespace: str
    task_manifest_sha256: str
    protocol_sha256: str
    sample_id: str
    rollout_index: int
    schema_version: str = PAIRED_POLICY_EVALUATION_RNG_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.master_seed) is not int
            or self.master_seed < 0
            or self.master_seed >= 2**63
        ):
            raise ValueError("paired evaluation master_seed must be in [0, 2**63)")
        if (
            not isinstance(self.seed_namespace, str)
            or not self.seed_namespace
            or self.seed_namespace.strip() != self.seed_namespace
            or any(character.isspace() for character in self.seed_namespace)
        ):
            raise ValueError("paired evaluation seed namespace is not canonical")
        _require_sha256(
            self.task_manifest_sha256, name="paired RNG task manifest SHA256"
        )
        _require_sha256(self.protocol_sha256, name="paired RNG protocol SHA256")
        if self.schema_version not in {
            PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
            RESOLUTION_PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
        }:
            raise ValueError("paired evaluation RNG schema differs")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("paired evaluation sample_id must be non-empty")
        if type(self.rollout_index) is not int or self.rollout_index < 0:
            raise ValueError("paired evaluation rollout_index must be non-negative")

    @property
    def stream_identity(self) -> dict[str, object]:
        identity: dict[str, object] = {
            "schema_version": self.schema_version,
            "seed_namespace": self.seed_namespace,
            "master_seed": self.master_seed,
            "task_manifest_sha256": self.task_manifest_sha256,
            "sample_id": self.sample_id,
            "rollout_index": self.rollout_index,
        }
        protocol_field = (
            "protocol_sha256"
            if self.schema_version == PAIRED_POLICY_EVALUATION_RNG_SCHEMA
            else "seed_protocol_sha256"
        )
        identity[protocol_field] = self.protocol_sha256
        return identity

    @property
    def stream_identity_sha256(self) -> str:
        return _canonical_json_sha256(self.stream_identity)

    def for_turn(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        turn_index: int,
        behavior_policy: PolicyVersion,
    ) -> VLLMTurnRNGIdentity:
        # Validate the sampler port inputs even though prompt/policy identity is
        # intentionally not part of the paired probability stream.
        if not prompt_token_ids or any(
            type(token_id) is not int or token_id < 0 for token_id in prompt_token_ids
        ):
            raise ValueError("paired evaluation RNG prompt token IDs are invalid")
        if type(turn_index) is not int or turn_index < 0:
            raise ValueError("paired evaluation turn_index must be non-negative")
        if not isinstance(behavior_policy, PolicyVersion):
            raise TypeError("paired evaluation behavior_policy must be PolicyVersion")
        state = {
            **self.stream_identity,
            "assistant_turn_index": turn_index,
        }
        rng_state_sha256 = _canonical_json_sha256(state)
        seed_domain = (
            b"tgvf-policy-paired-evaluation-seed-v1\0"
            if self.schema_version == PAIRED_POLICY_EVALUATION_RNG_SCHEMA
            else b"tgvf-policy-paired-evaluation-seed-v2\0"
        )
        seed_digest = hashlib.sha256(
            seed_domain + bytes.fromhex(rng_state_sha256)
        ).digest()
        return VLLMTurnRNGIdentity(
            seed=int.from_bytes(seed_digest[:8], "big") % _VLLM_SEED_MODULUS,
            rng_state_sha256=rng_state_sha256,
        )


def _paired_evaluation_rng_contract(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
    *,
    task_manifest_sha256: str,
) -> dict[str, object] | None:
    namespace = getattr(config, "paired_seed_namespace", None)
    if namespace is None:
        return None
    # Step-zero base equivalence is a checkpoint proof, not a sampling-protocol
    # component.  Keeping it out of this hash lets Step0 and trained checkpoints
    # share common random numbers while the full evaluation identity still binds
    # and audits the proof itself.
    rng_protocol = dict(_evaluation_protocol_identity(config, snapshot))
    rng_protocol.pop("base_equivalence", None)
    protocol_sha256 = _canonical_json_sha256(rng_protocol)
    projection_kind = getattr(config, "paired_rng_protocol_projection", None)
    if projection_kind is not None:
        if projection_kind != IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION:
            raise ValueError("paired RNG protocol projection differs")
        image_max_pixels = rng_protocol.pop("image_max_pixels", None)
        if image_max_pixels not in IMAGE_MAX_PIXELS_RESOLUTION_PAIR_VALUES:
            raise ValueError("resolution-pair protocol image_max_pixels differs")
        projection = {
            "kind": IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION,
            "excluded_protocol_field": "image_max_pixels",
            "axis_values": list(IMAGE_MAX_PIXELS_RESOLUTION_PAIR_VALUES),
        }
        seed_protocol_sha256 = _canonical_json_sha256(
            {
                "projection": projection,
                "projected_protocol": rng_protocol,
            }
        )
        return {
            "schema_version": RESOLUTION_PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
            "mode": "common_random_numbers_per_task_turn",
            "seed_namespace": namespace,
            "master_seed": snapshot.run.rollout_rng.master_seed,
            "task_manifest_sha256": task_manifest_sha256,
            "arm_protocol_sha256": protocol_sha256,
            "seed_protocol_sha256": seed_protocol_sha256,
            "protocol_projection": projection,
            "seed_components": [
                "master_seed",
                "seed_namespace",
                "task_manifest_sha256",
                "seed_protocol_sha256",
                "sample_id",
                "rollout_index",
                "assistant_turn_index",
            ],
            "excluded_arm_components": [
                "evaluation_id",
                "arm_name",
                "optimizer_step",
                "checkpoint_hash",
                "policy_weights_sha256",
                "prompt_token_ids_sha256",
            ],
        }
    return {
        "schema_version": PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
        "mode": "common_random_numbers_per_task_turn",
        "seed_namespace": namespace,
        "master_seed": snapshot.run.rollout_rng.master_seed,
        "task_manifest_sha256": task_manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "seed_components": [
            "master_seed",
            "seed_namespace",
            "task_manifest_sha256",
            "protocol_sha256",
            "sample_id",
            "rollout_index",
            "assistant_turn_index",
        ],
        "excluded_arm_components": [
            "evaluation_id",
            "arm_name",
            "optimizer_step",
            "checkpoint_hash",
            "policy_weights_sha256",
            "prompt_token_ids_sha256",
        ],
    }


def paired_evaluation_rng_contract(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSubject,
    *,
    task_manifest_sha256: str,
) -> dict[str, object] | None:
    """Expose the exact immutable RNG contract used by evaluation identities."""

    return _paired_evaluation_rng_contract(
        config,
        snapshot,
        task_manifest_sha256=task_manifest_sha256,
    )


def paired_evaluation_rng_for_task(
    evaluation_identity: Mapping[str, object],
    *,
    sample_id: str,
    rollout_index: int,
) -> PairedEvaluationVLLMTurnRNG:
    """Construct the evaluator RNG from an already immutable identity."""

    contract = evaluation_identity.get("sampling_rng")
    if not isinstance(contract, Mapping):
        raise ValueError("evaluation identity has no paired sampling RNG contract")
    schema_version = contract.get("schema_version")
    if schema_version not in {
        PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
        RESOLUTION_PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
    }:
        raise ValueError("paired sampling RNG schema differs")
    protocol_sha256 = (
        contract.get("protocol_sha256")
        if schema_version == PAIRED_POLICY_EVALUATION_RNG_SCHEMA
        else contract.get("seed_protocol_sha256")
    )
    return PairedEvaluationVLLMTurnRNG(
        master_seed=contract.get("master_seed"),
        seed_namespace=contract.get("seed_namespace"),
        task_manifest_sha256=contract.get("task_manifest_sha256"),
        protocol_sha256=protocol_sha256,
        sample_id=sample_id,
        rollout_index=rollout_index,
        schema_version=schema_version,
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
        if not isinstance(
            snapshot,
            (
                PolicyEvaluationSnapshot,
                PairedTGVFEvaluationSnapshot,
                FullModelEvaluationSnapshot,
            ),
        ):
            raise ValueError("training-run evaluation requires a bound policy snapshot")
        protocol = snapshot.run.protocol
        return {
            "profile": TRAINING_RUN_EVALUATION_PROTOCOL,
            "prompt_sha256": protocol.prompt_sha256,
            "tool_schema_sha256": protocol.tool_schema_sha256,
            "tool_profile": protocol.tool_profile.value,
            "enabled_tool_names": list(protocol.enabled_tool_names),
            "maximum_tool_calls": protocol.maximum_tool_calls,
            "native_pixels": isinstance(snapshot, FullModelEvaluationSnapshot),
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
        "image_max_pixels": evaluation_image_max_pixels(config, snapshot),
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
    if isinstance(snapshot, PolicyEvaluationSnapshot):
        policy_snapshot = {
            "run_id": snapshot.policy_version.run_id,
            "run_identity_sha256": snapshot.lora.run_identity_sha256,
            "optimizer_step": snapshot.policy_version.optimizer_step,
            "weights_sha256": snapshot.policy_version.weights_sha256,
            "pointer_file_sha256": snapshot.lora.pointer_file_sha256,
            "manifest_file_sha256": snapshot.lora.manifest_file_sha256,
            "tensor_file_sha256": snapshot.lora.tensor_file_sha256,
            "request_sha256": snapshot.lora.request_sha256,
        }
    elif isinstance(snapshot, PairedTGVFEvaluationSnapshot):
        policy_snapshot = paired_tgvf_snapshot_identity_record(snapshot)
    else:
        policy_snapshot = full_model_snapshot_identity_record(snapshot)
    policy_config_path = Path(
        getattr(config, "policy_config_path", snapshot.contract.source_path)
        if isinstance(snapshot, FullModelEvaluationSnapshot)
        else config.policy_config_path
    )
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
        "execution": {
            "world_size": len(config.gpu_ids),
            "gpu_ids": list(config.gpu_ids),
            "max_model_len": config.max_model_len,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "enable_chunked_prefill": config.enable_chunked_prefill,
            "inference_concurrency_per_gpu": config.inference_concurrency_per_gpu,
        },
    }
    if getattr(config, "evaluation_image_max_pixels", None) is not None:
        content["image_preprocessing"] = {
            "max_pixels": evaluation_image_max_pixels(config, snapshot),
            "source": "evaluation_override",
            "frozen_policy_max_pixels": snapshot.run.policy.image_max_pixels,
        }
    if (
        isinstance(snapshot, FullModelEvaluationSnapshot)
        and snapshot.manifest.checkpoint_owner is not None
    ):
        content["checkpoint_owner"] = {
            "run_id": snapshot.policy_version.run_id,
            "run_identity_sha256": snapshot.run_identity_sha256,
            "config_path": (
                snapshot.manifest.checkpoint_owner.config_path
                if snapshot.manifest.checkpoint_owner is not None
                else snapshot.manifest.run_contract_path
            ),
            "config_file_sha256": (
                snapshot.manifest.checkpoint_owner.config_file_sha256
                if snapshot.manifest.checkpoint_owner is not None
                else snapshot.manifest.run_contract_file_sha256
            ),
            "completion_path": (
                snapshot.manifest.checkpoint_owner.completion_path
                if snapshot.manifest.checkpoint_owner is not None
                else None
            ),
            "completion_file_sha256": (
                snapshot.manifest.checkpoint_owner.completion_file_sha256
                if snapshot.manifest.checkpoint_owner is not None
                else None
            ),
        }
        content["protocol_contract"] = {
            "run_id": snapshot.manifest.run_id,
            "run_identity_sha256": snapshot.manifest.run_identity_sha256,
            "path": snapshot.manifest.run_contract_path,
            "file_sha256": snapshot.manifest.run_contract_file_sha256,
        }
    prompt_materializer = _evaluation_prompt_materializer_identity(config, snapshot.run)
    if prompt_materializer is not None:
        # This top-level binding intentionally does not enter
        # _evaluation_protocol_identity(): paired common-random-number
        # streams keep their established protocol partition while resume
        # identity rejects outputs from the formerly mismatched renderer.
        content["prompt_materializer"] = prompt_materializer
    if (
        config.evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
        or isinstance(snapshot, PairedTGVFEvaluationSnapshot)
        or (
            isinstance(snapshot, FullModelEvaluationSnapshot)
            and config.evaluation_protocol == TRAINING_RUN_EVALUATION_PROTOCOL
        )
    ):
        content["protocol"] = _evaluation_protocol_identity(config, snapshot)
    paired_rng = _paired_evaluation_rng_contract(
        config,
        snapshot,
        task_manifest_sha256=task_sha256,
    )
    if paired_rng is not None:
        content["sampling_rng"] = paired_rng
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
    ) -> None:
        self.engine = engine
        self.lora_request = lora_request
        self.capture_hidden = capture_hidden
        self.native_pixels = native_pixels
        self.turns: dict[str, _TurnRoute] = {}
        self.backend_ids: dict[str, list[str]] = {}

    async def update_adapter_owned_state(
        self,
        *,
        optimizer_step: int,
        state: Mapping[str, torch.Tensor],
    ) -> dict[str, object]:
        """Install the RP66 member of a paired evaluation snapshot."""

        state_sha256 = adapter_owned_state_sha256(state)
        result = await self.engine.collective_rpc(
            "tgvf_update_adapter_owned_state",
            kwargs={
                "optimizer_step": optimizer_step,
                "state_sha256": state_sha256,
                "state_wire": _adapter_owned_state_to_utility_wire(state),
            },
        )
        return _validate_adapter_update_ack(
            _single_collective(result, operation="Adapter-owned state update"),
            expected_optimizer_step=optimizer_step,
            expected_state_sha256=state_sha256,
            expected_tensor_count=len(state),
        )

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

    async def materialize_crop_tgvf(
        self,
        *,
        request_id: str,
        expected_step: int,
        sampled_output_ids: tuple[int, ...],
        call_index: int,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        source_image_sha256: str,
        crop_sha256: str,
        preprocessed_visual_sha256: str,
        model_bbox_2d: tuple[int, int, int, int],
        target_start: int,
        target_end: int,
        expected_target_token_ids: tuple[int, ...],
        provider: str,
    ) -> TGVFCropMaterializationResult:
        """Materialize one crop-conditioned D bound to the sampled target."""

        turn = self._validated_turn(request_id, expected_step, sampled_output_ids)
        target_token_ids = tuple(expected_target_token_ids)
        if (
            type(target_start) is not int
            or type(target_end) is not int
            or target_start < 0
            or target_end <= target_start
            or turn.output_ids[target_start:target_end] != target_token_ids
        ):
            raise RuntimeError("crop+TGVF target differs from sampled output")
        if not isinstance(image_grid_thw, torch.Tensor) or image_grid_thw.shape != (
            1,
            3,
        ):
            raise TypeError("crop+TGVF image_grid_thw must have shape [1,3]")
        image_grid = tuple(int(value) for value in image_grid_thw[0].tolist())
        bbox = tuple(model_bbox_2d)
        result = await self.engine.collective_rpc(
            "tgvf_materialize_crop_tgvf",
            kwargs={
                "trajectory_id": request_id,
                "backend_request_id": turn.backend_request_id,
                "call_index": call_index,
                "pixel_values_wire": _tensor_to_utility_wire(pixel_values),
                "image_grid_thw": image_grid,
                "source_image_sha256": source_image_sha256,
                "crop_sha256": crop_sha256,
                "preprocessed_visual_sha256": preprocessed_visual_sha256,
                "model_bbox_2d": bbox,
                "target_start": target_start,
                "target_end": target_end,
                "expected_target_token_ids": target_token_ids,
                "provider": provider,
            },
        )
        typed = _crop_tgvf_from_utility_wire(
            _single_collective(result, operation="crop+TGVF materialization")
        )
        if not isinstance(typed, TGVFCropMaterializationResult):
            raise TypeError("crop+TGVF RPC returned an invalid result")
        expected_binding = (
            source_image_sha256,
            crop_sha256,
            preprocessed_visual_sha256,
            image_grid,
            call_index,
            bbox,
            target_start,
            target_end,
            target_token_ids,
            provider,
        )
        actual_binding = (
            typed.source_image_sha256,
            typed.crop_sha256,
            typed.preprocessed_visual_sha256,
            typed.image_grid_thw,
            typed.call_index,
            typed.model_bbox_2d,
            typed.target_start,
            typed.target_end,
            typed.target_token_ids,
            typed.provider,
        )
        if actual_binding != expected_binding:
            raise IdentityMismatchError(
                "crop+TGVF RPC result differs from requested binding"
            )
        return typed

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

    if isinstance(snapshot, PairedTGVFEvaluationSnapshot):
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        os.environ["TGVF_POLICY_RUN_CONFIG_PATH"] = str(
            Path(snapshot.receipt.policy_config_path).resolve()
        )
        engine_args = AsyncEngineArgs(**_paired_tgvf_engine_kwargs(config, snapshot))
        engine = AsyncLLM.from_engine_args(engine_args)
        manager = StandaloneTGVFVLLMManager(
            engine, None, capture_hidden=True, native_pixels=False
        )
        acknowledgement = await manager.update_adapter_owned_state(
            optimizer_step=snapshot.policy_version.optimizer_step,
            state=snapshot.rp66_tensors,
        )
        if acknowledgement["state_sha256"] != snapshot.receipt.rp66_state_sha256:
            engine.shutdown()
            raise RuntimeError("paired evaluator loaded a different RP66 state")
        return manager, engine, snapshot.run

    from vllm import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.v1.engine.async_llm import AsyncLLM

    if not isinstance(snapshot, PolicyEvaluationSnapshot):
        raise TypeError("snapshot must be a PolicyEvaluationSnapshot")
    run = snapshot.run
    adapter_root = materialize_vllm_lora_adapter(config, snapshot)
    engine_args = AsyncEngineArgs(**_standalone_engine_kwargs(config, run))
    engine = AsyncLLM.from_engine_args(engine_args)
    lora = LoRARequest(policy_lora_request_name(snapshot), 1, str(adapter_root))
    manager = StandaloneTGVFVLLMManager(
        engine,
        lora,
        capture_hidden=_evaluation_requires_hidden_capture(config, run),
        native_pixels=(
            config.evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
        ),
    )
    return manager, engine, run


def _evaluation_requires_hidden_capture(
    config: PolicyCoreDevConfig,
    run: PolicyE2ESmokeRunConfig,
) -> bool:
    return (
        config.evaluation_protocol == TRAINING_RUN_EVALUATION_PROTOCOL
        and run.protocol.tool_profile
        in {
            NativeToolCapabilityProfile.TGVF_ONLY,
            NativeToolCapabilityProfile.CROP_TGVF,
        }
    )


def _paired_tgvf_engine_kwargs(
    config: PolicyCoreDevConfig, snapshot: PairedTGVFEvaluationSnapshot
) -> dict[str, object]:
    run = snapshot.run
    return {
        "model": str(snapshot.model_path),
        "tokenizer": run.model.revision_or_path,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "distributed_executor_backend": "mp",
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.inference_concurrency_per_gpu,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "enable_chunked_prefill": config.enable_chunked_prefill,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "logprobs_mode": "processed_logprobs",
        "enforce_eager": False,
        "seed": run.rollout_rng.master_seed,
        "enable_lora": False,
        "worker_extension_cls": TGVF_VLLM_WORKER_EXTENSION_FQN,
        "enable_mm_embeds": True,
        "mm_encoder_attn_backend": TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
        "hf_overrides": {"architectures": [TGVF_QWEN3_VLLM_ARCHITECTURE]},
        "mm_processor_cache_gb": 0,
        "limit_mm_per_prompt": {
            "image": 1 + run.protocol.maximum_tool_calls,
            "video": 0,
        },
    }


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
    if run.schema_version == POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMA:
        return VLLMTurnTerminationContract(
            required_request_stop_strings=tuple(sampling.stop_strings or ()),
            required_request_stop_token_ids=tuple(sampling.stop_token_ids or ()),
            include_stop_str_in_output=bool(sampling.include_stop_str_in_output),
            tool_call_terminal_suffixes=(),
            tool_call_outcomes=(),
            final_turn_outcomes=qwen3_vl_final_turn_outcomes(
                tuple(sampling.stop_token_ids or ())
            ),
            tool_calls_enabled=False,
        )
    return VLLMTurnTerminationContract(
        required_request_stop_strings=tuple(sampling.stop_strings or ()),
        required_request_stop_token_ids=tuple(sampling.stop_token_ids or ()),
        include_stop_str_in_output=bool(sampling.include_stop_str_in_output),
        tool_call_terminal_suffixes=("",),
        tool_call_outcomes=(VLLMTerminationOutcome("stop", "</tool_call>"),),
        final_turn_outcomes=qwen3_vl_final_turn_outcomes(
            tuple(sampling.stop_token_ids or ())
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
        snapshot: PolicyEvaluationSubject,
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
        self.image_max_pixels = evaluation_image_max_pixels(config, snapshot)
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
        self.success_environment_text_renderer = _success_environment_text_renderer(run)
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
        prompt_text, canonical_token_ids = _render_training_run_visual_prompt(
            run=self.run,
            processor=self.processor,
            renderer=self.renderer,
            question=task.question,
        )
        from tgvf_rl.framework.verl.smoke_dataset import (
            _materialize_source_image_prompt_token_ids,
        )

        return _materialize_source_image_prompt_token_ids(
            processor=self.processor,
            canonical_token_ids=canonical_token_ids,
            prompt_text=prompt_text,
            image_max_pixels=self.image_max_pixels,
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
            image_max_pixels=self.image_max_pixels,
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
            preprocessed_pixel_values=pixel_values,
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
                image_max_pixels=self.image_max_pixels,
            ),
        )
        appender = QwenNativeToolObservationAppender(
            tokenizer=self.layout_builder.tokenizer,
            registrar=registry,
            visual_token_count_resolver=_VisualTokenCountResolver(self.store),
            success_environment_text_renderer=(self.success_environment_text_renderer),
            assistant_dialect=self.assistant_dialect,
        )
        if self.run.protocol.tool_profile is NativeToolCapabilityProfile.TGVF_ONLY:
            tool_runtime = _build_remote_tgvf_focus_tool_runtime(
                event_loop=asyncio.get_running_loop(),
                server_client=self.manager,
                config=self.run,
                source_visual=source,
                layout_builder=self.layout_builder,
                observation_store=self.store,
                execution_ledger=self.focus_ledger,
                contextual_forward_identity=self.contextual_identity,
                branch_merger_identities=self.branch_identities,
                success_environment_text_renderer=(
                    self.success_environment_text_renderer
                ),
                assistant_dialect=self.assistant_dialect,
            )
        elif self.run.protocol.tool_profile is NativeToolCapabilityProfile.CROP_ONLY:
            processor_identity = _artifact_identity(
                "policy-evaluation",
                "qwen3-shared-vllm-crop-processor",
                self.config.schema_version,
                {
                    "model": self.run.model.revision_or_path,
                    "max_pixels": self.image_max_pixels,
                },
            )
            layout_identity = _artifact_identity(
                "policy-evaluation",
                "qwen3-native-crop-layout",
                self.config.schema_version,
                {"model": self.run.model.revision_or_path},
            )
            materializer = _RemoteCropVisualMaterializer(
                event_loop=asyncio.get_running_loop(),
                server_client=self.manager,
                processor=self.processor,
                model_identity=self.run.model,
                image_max_pixels=self.image_max_pixels,
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
            )
        elif self.run.protocol.tool_profile is NativeToolCapabilityProfile.CROP_TGVF:
            processor_identity = _artifact_identity(
                "policy-evaluation",
                "qwen3-shared-vllm-crop-tgvf-processor",
                self.config.schema_version,
                {
                    "model": self.run.model.revision_or_path,
                    "max_pixels": self.image_max_pixels,
                },
            )
            layout_identity = _artifact_identity(
                "policy-evaluation",
                "qwen3-native-crop-tgvf-layout",
                self.config.schema_version,
                {"model": self.run.model.revision_or_path},
            )
            tool_runtime = _RemoteAtomicCropTGVFToolRuntime(
                event_loop=asyncio.get_running_loop(),
                server_client=self.manager,
                config=self.run,
                source_visual=source,
                layout_builder=self.layout_builder,
                observation_store=self.store,
                execution_ledger=self.focus_ledger,
                contextual_forward_identity=self.contextual_identity,
                branch_merger_identities=self.branch_identities,
                crop_processor_identity=processor_identity,
                crop_layout_identity=layout_identity,
                processor=self.processor,
                image_max_pixels=self.image_max_pixels,
                success_environment_text_renderer=(
                    self.success_environment_text_renderer
                ),
                assistant_dialect=self.assistant_dialect,
            )
        else:
            raise RuntimeError("policy CoreDev has an unsupported visual-tool profile")

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
            rng=(
                paired_evaluation_rng_for_task(
                    self.evaluation_identity,
                    sample_id=identity.sample_id,
                    rollout_index=identity.rollout_index,
                )
                if self.config.paired_seed_namespace is not None
                else ContentAddressedVLLMTurnRNG(
                    master_seed=self.run.rollout_rng.master_seed,
                    stream_identity=trajectory_id,
                )
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
            forbidden_policy_token_ids=(
                self.layout_builder.forbidden_policy_visual_token_ids
            ),
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


def full_model_protocol_audit_fields(
    evaluation_identity: Mapping[str, object],
    policy_snapshot: Mapping[str, object],
) -> dict[str, object]:
    """Return v2 owner/protocol result fields while keeping v1 field-free.

    The schema generation is selected by the independent top-level owner and
    protocol bindings, never by the optional fields being validated.
    """

    has_owner = "checkpoint_owner" in evaluation_identity
    has_protocol = "protocol_contract" in evaluation_identity
    snapshot_protocol_fields = {
        "protocol_run_id",
        "protocol_run_identity_sha256",
    }.intersection(policy_snapshot)
    if has_owner != has_protocol:
        raise ValueError("full-model v2 owner/protocol bindings are incomplete")
    if not has_owner:
        if snapshot_protocol_fields:
            raise ValueError("full-model v1 snapshot has unexpected protocol fields")
        return {}
    owner = evaluation_identity["checkpoint_owner"]
    protocol = evaluation_identity["protocol_contract"]
    if not isinstance(owner, Mapping) or not isinstance(protocol, Mapping):
        raise ValueError("full-model v2 owner/protocol bindings are malformed")
    protocol_run_id = protocol.get("run_id")
    protocol_run_identity_sha256 = protocol.get("run_identity_sha256")
    if not isinstance(protocol_run_id, str) or not protocol_run_id:
        raise ValueError("full-model v2 protocol run ID is malformed")
    _require_sha256(
        protocol_run_identity_sha256,
        name="full-model v2 protocol run identity SHA256",
    )
    if policy_snapshot.get("run_id") != owner.get("run_id") or policy_snapshot.get(
        "run_identity_sha256"
    ) != owner.get("run_identity_sha256"):
        raise ValueError("full-model v2 checkpoint owner binding differs")
    expected = {
        "protocol_run_id": protocol_run_id,
        "protocol_run_identity_sha256": protocol_run_identity_sha256,
    }
    if any(policy_snapshot.get(field) != value for field, value in expected.items()):
        raise ValueError("full-model v2 protocol snapshot binding differs")
    return expected


def _policy_result_identity_fields(
    task: CoreDevTask,
    *,
    evaluation_identity: Mapping[str, object],
    rank: int,
    world_size: int,
) -> dict[str, object]:
    """Build the common immutable identity for trajectory and failure rows."""

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
    snapshot_backend = policy_snapshot.get(
        "snapshot_backend", LORA_ADAPTER_EVALUATION_BACKEND
    )
    if snapshot_backend == PAIRED_TGVF_EVALUATION_BACKEND:
        snapshot_fields = {
            "policy_snapshot_backend": PAIRED_TGVF_EVALUATION_BACKEND,
            "policy_paired_snapshot_identity_sha256": policy_snapshot[
                "snapshot_identity_sha256"
            ],
            "policy_qwen_tree_sha256": policy_snapshot["qwen_tree_sha256"],
            "policy_rp66_state_sha256": policy_snapshot["rp66_state_sha256"],
            "policy_rp66_storage_sha256": policy_snapshot["rp66_storage_sha256"],
        }
    elif snapshot_backend == FULL_MODEL_EVALUATION_BACKEND:
        snapshot_fields = {
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
        snapshot_fields = {
            "policy_pointer_file_sha256": policy_snapshot["pointer_file_sha256"],
            "policy_manifest_file_sha256": policy_snapshot["manifest_file_sha256"],
            "policy_tensor_file_sha256": policy_snapshot["tensor_file_sha256"],
        }
    else:
        raise ValueError("training-run result snapshot backend differs")
    group_uid = (
        f"coredev:{task.ordinal}"
        if evaluation_identity["evaluation_schema_version"] == POLICY_COREDEV_SCHEMA
        else f"benchmark:{task.ordinal}"
    )
    trajectory_identity = TrajectoryIdentity(
        str(evaluation_identity["evaluation_id"]),
        task.bound_sample_id,
        0,
        group_uid,
    )
    payload: dict[str, object] = {
        "schema_version": POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA,
        "selection_reasons": ["representative_rollout_zero"],
        "evaluation_identity_sha256": identity_sha256,
        "policy_run_identity_sha256": policy_snapshot["run_identity_sha256"],
        **snapshot_fields,
        "policy_config_identity_sha256": evaluation_identity[
            "policy_run_config_identity_sha256"
        ],
        "task_manifest_sha256": task_manifest["sha256"],
        "model_identity": dict(model_identity),
        "rank": rank,
        "world_size": world_size,
        "evaluation_id": trajectory_identity.run_id,
        "sample_id": trajectory_identity.sample_id,
        "group_uid": trajectory_identity.group_id,
        "rollout_index": trajectory_identity.rollout_index,
        "ordinal": task.ordinal,
        "dataset": task.dataset,
        "row_number": task.row_number,
        "index": task.index,
        "question": task.question,
        "image_paths": list(task.image_paths),
        "image_sha256s": list(task.image_sha256s),
        "image_dimensions": [list(item) for item in task.image_dimensions],
        "trajectory_id": trajectory_identity.canonical_id,
        "policy_run_id": policy_snapshot["run_id"],
        "optimizer_step": policy_snapshot["optimizer_step"],
        "policy_weights_sha256": policy_snapshot["weights_sha256"],
    }
    if snapshot_backend == FULL_MODEL_EVALUATION_BACKEND:
        payload.update(
            full_model_protocol_audit_fields(evaluation_identity, policy_snapshot)
        )
    if "sampling_rng" in evaluation_identity:
        rng = paired_evaluation_rng_for_task(
            evaluation_identity,
            sample_id=trajectory_identity.sample_id,
            rollout_index=trajectory_identity.rollout_index,
        )
        payload["sampling_rng"] = dict(evaluation_identity["sampling_rng"])
        payload["paired_rng_stream_identity_sha256"] = rng.stream_identity_sha256
    return payload


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

    policy_snapshot = evaluation_identity.get("policy_snapshot")
    model_identity = evaluation_identity.get("model_identity")
    if not isinstance(policy_snapshot, Mapping) or not isinstance(
        model_identity, Mapping
    ):
        raise ValueError("evaluation identity sub-bindings are malformed")
    if asdict(trajectory.model) != dict(model_identity):
        raise ValueError("trajectory model differs from evaluation identity")
    if trajectory.behavior_policy != PolicyVersion(
        run_id=str(policy_snapshot.get("run_id")),
        optimizer_step=policy_snapshot.get("optimizer_step"),
        weights_sha256=str(policy_snapshot.get("weights_sha256")),
    ):
        raise ValueError("trajectory policy differs from evaluation identity")
    payload = _policy_result_identity_fields(
        task,
        evaluation_identity=evaluation_identity,
        rank=rank,
        world_size=world_size,
    )
    if payload["trajectory_id"] != trajectory.identity.canonical_id:
        raise ValueError("trajectory identity differs from evaluation task identity")
    payload.update(
        {
            "trajectory_sha256": trajectory_checksum(trajectory),
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
    )
    payload["result_identity_sha256"] = _canonical_json_sha256(payload)
    return payload


def policy_output_contract_failure_audit_payload(
    task: CoreDevTask,
    error: PolicyOutputContractError,
    *,
    evaluation_identity: Mapping[str, object],
    rank: int,
    world_size: int,
) -> dict[str, object]:
    """Materialize one scored, identity-bound sample-local output failure.

    The row is intentionally not a fabricated :class:`TrajectoryRecord`: the
    sampler rejected the response before a legal assistant turn existed.  It
    remains a completed benchmark row with a null answer, so scoring keeps the
    task in the denominator and deterministically marks it wrong.
    """

    if not isinstance(error, PolicyOutputContractError):
        raise TypeError("sample-local failure requires PolicyOutputContractError")
    if error.code != "tool_call_terminal_suffix":
        raise ValueError("unsupported sample-local policy-output failure code")
    diagnostic = dict(error.diagnostic)
    expected_diagnostic_fields = {
        "response_text_sha256",
        "suffix_sha256",
        "suffix_char_count",
        "suffix_utf8_byte_count",
        "finish_reason",
        "stop_reason",
        "backend_request_sha256",
        "backend_response_sha256",
    }
    if set(diagnostic) != expected_diagnostic_fields:
        raise ValueError("policy-output failure diagnostic fields differ")
    for field in (
        "response_text_sha256",
        "suffix_sha256",
        "backend_request_sha256",
        "backend_response_sha256",
    ):
        _require_sha256(diagnostic[field], name=f"policy-output {field}")
    for field in ("suffix_char_count", "suffix_utf8_byte_count"):
        value = diagnostic[field]
        if type(value) is not int or value <= 0:
            raise ValueError(f"policy-output {field} must be a positive integer")
    if (
        not isinstance(diagnostic["finish_reason"], str)
        or not diagnostic["finish_reason"]
    ):
        raise ValueError("policy-output finish_reason must be non-empty")
    if diagnostic["stop_reason"] is not None and (
        isinstance(diagnostic["stop_reason"], bool)
        or not isinstance(diagnostic["stop_reason"], (int, str))
    ):
        raise TypeError("policy-output stop_reason must be int, str, or null")
    failure = {
        "schema_version": POLICY_OUTPUT_CONTRACT_FAILURE_SCHEMA,
        "kind": "policy_output_contract",
        "code": error.code,
        "exception_type": type(error).__name__,
        "message": str(error),
        "diagnostic": diagnostic,
        "diagnostic_sha256": _canonical_json_sha256(diagnostic),
    }
    payload = _policy_result_identity_fields(
        task,
        evaluation_identity=evaluation_identity,
        rank=rank,
        world_size=world_size,
    )
    failure_record = {
        "schema_version": POLICY_OUTPUT_CONTRACT_FAILURE_SCHEMA,
        "trajectory_id": payload["trajectory_id"],
        "failure": failure,
    }
    payload.update(
        {
            "result_kind": "sample_local_failure",
            "trajectory_available": False,
            "stop": "invalid_format",
            "final_answer": None,
            "assistant_turns": [],
            "tool_calls": [],
            "tool_errors": [],
            "successful_observation_count": 0,
            "failure": failure,
            "failure_record_sha256": _canonical_json_sha256(failure_record),
        }
    )
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
    elif snapshot_backend == PAIRED_TGVF_EVALUATION_BACKEND:
        snapshot_expected = {
            "policy_snapshot_backend": PAIRED_TGVF_EVALUATION_BACKEND,
            "policy_paired_snapshot_identity_sha256": policy_snapshot[
                "snapshot_identity_sha256"
            ],
            "policy_qwen_tree_sha256": policy_snapshot["qwen_tree_sha256"],
            "policy_rp66_state_sha256": policy_snapshot["rp66_state_sha256"],
            "policy_rp66_storage_sha256": policy_snapshot["rp66_storage_sha256"],
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
    if snapshot_backend == FULL_MODEL_EVALUATION_BACKEND:
        expected.update(
            full_model_protocol_audit_fields(evaluation_identity, policy_snapshot)
        )
    if "sampling_rng" in evaluation_identity:
        rng = paired_evaluation_rng_for_task(
            evaluation_identity,
            sample_id=task.bound_sample_id,
            rollout_index=0,
        )
        expected["sampling_rng"] = evaluation_identity["sampling_rng"]
        expected["paired_rng_stream_identity_sha256"] = rng.stream_identity_sha256
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
    result_kind = payload.get("result_kind", "trajectory")
    if result_kind == "trajectory":
        _require_sha256(
            payload.get("trajectory_sha256"), name="trajectory result SHA256"
        )
        return
    if result_kind != "sample_local_failure":
        raise RuntimeError("policy benchmark result kind is unsupported")
    if payload.get("trajectory_available") is not False:
        raise RuntimeError("sample-local failure claims a trajectory")
    if "trajectory_sha256" in payload:
        raise RuntimeError("sample-local failure must not claim a trajectory SHA256")
    expected_failure_fields = {
        "schema_version",
        "kind",
        "code",
        "exception_type",
        "message",
        "diagnostic",
        "diagnostic_sha256",
    }
    failure = payload.get("failure")
    if not isinstance(failure, Mapping) or set(failure) != expected_failure_fields:
        raise RuntimeError("sample-local failure envelope is malformed")
    if (
        failure.get("schema_version") != POLICY_OUTPUT_CONTRACT_FAILURE_SCHEMA
        or failure.get("kind") != "policy_output_contract"
        or failure.get("code") != "tool_call_terminal_suffix"
        or failure.get("exception_type") != "PolicyOutputContractError"
        or failure.get("message")
        != "vLLM emitted a tool-call suffix outside the run-bound contract"
    ):
        raise RuntimeError("sample-local policy-output failure identity differs")
    diagnostic = failure.get("diagnostic")
    expected_diagnostic_fields = {
        "response_text_sha256",
        "suffix_sha256",
        "suffix_char_count",
        "suffix_utf8_byte_count",
        "finish_reason",
        "stop_reason",
        "backend_request_sha256",
        "backend_response_sha256",
    }
    if not isinstance(diagnostic, Mapping) or set(diagnostic) != (
        expected_diagnostic_fields
    ):
        raise RuntimeError("sample-local failure diagnostic fields differ")
    if failure.get("diagnostic_sha256") != _canonical_json_sha256(diagnostic):
        raise RuntimeError("sample-local failure diagnostic digest differs")
    for field in (
        "response_text_sha256",
        "suffix_sha256",
        "backend_request_sha256",
        "backend_response_sha256",
    ):
        _require_sha256(diagnostic.get(field), name=f"policy-output {field}")
    if any(
        type(diagnostic.get(field)) is not int or diagnostic[field] <= 0
        for field in ("suffix_char_count", "suffix_utf8_byte_count")
    ):
        raise RuntimeError("sample-local suffix lengths are malformed")
    if not isinstance(diagnostic.get("finish_reason"), str) or not diagnostic.get(
        "finish_reason"
    ):
        raise RuntimeError("sample-local finish reason is malformed")
    if diagnostic.get("stop_reason") is not None and (
        isinstance(diagnostic.get("stop_reason"), bool)
        or not isinstance(diagnostic.get("stop_reason"), (int, str))
    ):
        raise RuntimeError("sample-local stop reason is malformed")
    if (
        payload.get("stop") != "invalid_format"
        or payload.get("final_answer") is not None
        or payload.get("assistant_turns") != []
        or payload.get("tool_calls") != []
        or payload.get("tool_errors") != []
        or payload.get("successful_observation_count") != 0
    ):
        raise RuntimeError("sample-local failure scoring fields differ")
    failure_record = {
        "schema_version": POLICY_OUTPUT_CONTRACT_FAILURE_SCHEMA,
        "trajectory_id": expected_trajectory_id,
        "failure": dict(failure),
    }
    expected_failure_sha256 = payload.get("failure_record_sha256")
    _require_sha256(expected_failure_sha256, name="failure record SHA256")
    if expected_failure_sha256 != _canonical_json_sha256(failure_record):
        raise RuntimeError("sample-local failure record digest differs")


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
    "POLICY_BENCHMARK_SCHEMA",
    "POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA",
    "POLICY_COREDEV_SCHEMA",
    "POLICY_EVALUATION_IDENTITY_SCHEMA",
    "POLICY_MATCHED_PROMPT_MATERIALIZER_SCHEMA",
    "POLICY_MATCHED_PROMPT_MATERIALIZER_VERSION",
    "POLICY_OUTPUT_CONTRACT_FAILURE_SCHEMA",
    "PAIRED_POLICY_EVALUATION_RNG_SCHEMA",
    "PairedEvaluationVLLMTurnRNG",
    "PolicyCoreDevConfig",
    "PolicyCoreDevEvaluator",
    "PolicyEvaluationSnapshot",
    "PolicyEvaluationSubject",
    "StandaloneTGVFVLLMManager",
    "build_standalone_manager",
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
    "policy_output_contract_failure_audit_payload",
    "policy_version_from_pointer",
    "paired_evaluation_rng_for_task",
    "paired_evaluation_rng_contract",
    "prepare_policy_benchmark_tasks",
    "trajectory_audit_payload",
    "validate_policy_benchmark_runtime_interfaces",
    "validate_policy_benchmark_result",
    "write_policy_evaluation_identity",
    "write_official_coredev_tasks",
]
