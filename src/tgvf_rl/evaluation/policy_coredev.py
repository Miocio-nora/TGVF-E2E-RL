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
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import torch

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
    TrajectoryIdentity,
    TrajectoryRecord,
)

from .policy_benchmark_artifacts import (
    POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA as POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA,
    CoreDevTask,
    _decode_rgb_bytes as _decode_rgb_bytes,
    _official_prompt_text as _official_prompt_text,
    _option_lines as _option_lines,
    _read_bound_image_bytes as _read_bound_image_bytes,
    _read_regular_file_bytes,
    _tsv_image_paths as _tsv_image_paths,
    image_file_identity,
    load_benchmark_tasks,
    load_bound_policy_benchmark_tasks,
    load_coredev_tasks,
    load_policy_benchmark_results,
    load_verified_task_image,
    prepare_policy_benchmark_tasks,
    trajectory_audit_payload,
    validate_policy_benchmark_result,
    write_official_coredev_tasks,
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
    canonical_json_sha256 as _canonical_json_sha256,  # noqa: F401
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
