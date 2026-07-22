"""Model-free AgentLoop composition for the two-model Policy RL runtime.

The only full model roles are the upstream FSDP2 actor and its colocated vLLM
rollout replica.  AgentLoop workers own CPU protocol/replay state only.  Source
vision, sampled target Hq, and the frozen TGVF Adapter execute through the
sticky vLLM client and the resulting D tensors are recorded exactly once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.conditioning import (
    bind_preselected_target_conditioning,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.tokens import TokenSpan
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.environment import (
    CropExecutionLedger,
    CropVisualTensorBundle,
    FocusExecutionLedger,
    FrameworkNeutralAgentLoop,
    ImageZoomInToolRuntime,
    Qwen3NativeToolLayoutBuilder,
    QwenNativeToolObservationAppender,
    record_trajectory_source_visual,
)
from tgvf_rl.environment.focus_runtime import _call_fingerprint
from tgvf_rl.environment.focus_tool import (
    SourceVisualTensorBundle,
    TGVFFocusTool,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from tgvf_rl.environment.qwen3_crop_materializer import preprocess_qwen3_rgb
from tgvf_rl.protocol.state_machine import CapErrorBehavior
from tgvf_rl.observations.finalizer import (
    MaterializedTrajectoryReplayTensors,
    TrajectoryReplayFinalizationRequest,
    finalize_trajectory_replay,
)
from tgvf_rl.observations.schema import FocusedObservationRecord, TrajectorySourceVisual
from tgvf_rl.observations.schema import CropObservationRecord
from tgvf_rl.observations.store import (
    ObservationHandle,
    ObservationStore,
    tensor_checksum,
)
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import (
    NativeToolCapabilityProfile,
    ParsedImageZoomInCall,
    ParsedToolCall,
)
from tgvf_rl.policy.trajectory_audit import PolicyTrajectoryAuditWriter
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)
from tgvf_rl.rewards import (
    AnswerTaskKind,
    PilotRewardPipeline,
    PilotRewardSpec,
    RuleFirstAnswerVerifier,
    reward_context_from_trajectory,
)
from tgvf_rl.rewards.schema import NormalizationSpec
from tgvf_rl.judges import DisabledJudgeProvider, load_openai_compatible_judge
from tgvf_rl.trajectories.behavior import BehaviorTraceStore, VLLMBehaviorRecorder
from tgvf_rl.trajectories.schema import TrajectoryRecord, trajectory_checksum
from tgvf_rl.framework.vllm import (
    LiveVLLMTurnContextRegistry,
    Qwen3VLLMObservationPayloadResolver,
    VLLMLivePromptInputs,
    bind_preexpanded_prompt_contract,
)

from .native_agent_loop import VerlNativeTrajectoryComponents
from .objective_bridge import make_objective_sentinels
from .reward_bridge import VerlRewardedAgentLoopOutputBuilder
from .rollout_bridge import RolloutBridgeRecord
from .policy_runtime import (
    PolicyE2ERuntimeBuildContext,
    PolicyE2ERuntimeProduct,
)
from .reward_bridge import RewardedTrajectoryFinalizerPort
from tgvf_rl.rewards.verl_adapter import (
    PilotRewardContextProvider,
    PilotVerlTrajectoryReward,
    PilotVerlTrajectoryRewardScorer,
)


QWEN3_POLICY_E2E_LIVE_RUNTIME_SCHEMA = "tgvf-qwen3-policy-e2e-live-runtime-v1"
_BRANCH_LAYERS = (8, 16, 24)


class Qwen3PolicyE2ELiveRuntimeBuilder:
    """Build CPU AgentLoop state bound to the existing vLLM rollout client."""

    singleton_identity = QWEN3_POLICY_E2E_LIVE_RUNTIME_SCHEMA

    def __init__(
        self,
        *,
        agent_loop_output_cls: type[Any] | None = None,
        metrics_factory: Callable[[TrajectoryRecord, PilotVerlTrajectoryReward], object]
        | None = None,
    ) -> None:
        if metrics_factory is not None and not callable(metrics_factory):
            raise TypeError("metrics_factory must be callable")
        self.agent_loop_output_cls = agent_loop_output_cls
        self.metrics_factory = metrics_factory or _default_metrics_factory

    def build(self, context: PolicyE2ERuntimeBuildContext, /) -> PolicyE2ERuntimeProduct:
        if not isinstance(context, PolicyE2ERuntimeBuildContext):
            raise TypeError("live runtime builder requires PolicyE2ERuntimeBuildContext")
        config = context.config
        if config.model.family != "qwen3_vl":
            raise ValueError("Policy Pilot live runtime requires qwen3_vl")
        server_client = context.server_manager
        required_methods = ["materialize_source", "generate"]
        if config.protocol.tool_profile is NativeToolCapabilityProfile.TGVF_ONLY:
            required_methods.append("materialize_focus")
        elif config.protocol.tool_profile is NativeToolCapabilityProfile.CROP_ONLY:
            required_methods.append("materialize_crop")
        else:
            raise ValueError("atomic crop+TGVF is not wired into this live runtime")
        for method in required_methods:
            if not callable(getattr(server_client, method, None)):
                raise TypeError(
                    "Policy Pilot requires the TGVF two-model vLLM client; "
                    f"missing {method}()"
                )

        store = ObservationStore()
        behavior_store = BehaviorTraceStore()
        focus_execution_ledger = FocusExecutionLedger()
        crop_execution_ledger = CropExecutionLedger()
        layout_builder = Qwen3NativeToolLayoutBuilder.from_processor_config(
            processor=context.processor,
            model_identity=config.model,
            observation_store=store,
        )
        if config.protocol.tool_profile is NativeToolCapabilityProfile.TGVF_ONLY:
            export = load_rank_zero_adapter_owned_state_export(
                config.representation.artifact_path
            )
            run_identity = export.manifest.run_identity
            if state_digest(export.manifest) != config.representation.artifact.sha256:
                raise IdentityMismatchError(
                    "representation export manifest identity differs"
                )
            if (
                run_identity.run_id != config.representation.expected_run_id
                or export.manifest.run_identity_sha256
                != config.representation.expected_run_identity_sha256
                or run_identity.identity_sha256
                != config.representation.expected_run_identity_sha256
            ):
                raise IdentityMismatchError(
                    "representation export run identity differs"
                )
            if run_identity.model != config.model:
                raise IdentityMismatchError(
                    "representation export and Policy model identities differ"
                )
            if run_identity.provider != config.representation.conditioning:
                raise IdentityMismatchError(
                    "representation export and selected conditioning differ"
                )
        else:
            run_identity = None

        conditioning = config.representation.conditioning
        if run_identity is not None and (
            conditioning.provider
            is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
        ):
            if conditioning.hidden_layer != -1:
                raise ValueError(
                    "colocated vLLM contextual conditioning requires hidden layer -1"
                )
            contextual_forward_identity = _artifact_identity(
                "policy-runtime",
                "qwen3-contextual-behavior-forward",
                QWEN3_POLICY_E2E_LIVE_RUNTIME_SCHEMA,
                {
                    "run_identity": config.identity_sha256,
                    "model": config.model.revision_or_path,
                    "provider": conditioning.provider.value,
                    "hidden_layer": conditioning.hidden_layer,
                    "deterministic": True,
                    "policy_adapter_dropout": 0.0,
                },
            )
        else:
            contextual_forward_identity = None
        branch_mergers = tuple(
            _artifact_identity(
                "qwen3-vl",
                f"deepstack-merger-{layer}",
                "frozen-base-model-v1",
                {
                    "model": config.model.revision_or_path,
                    "projection_identity": projection_identity,
                    "layer": layer,
                },
            )
            for layer, projection_identity in zip(
                _BRANCH_LAYERS,
                run_identity.adapter_contract.deepstack_projection_identities,
                strict=True,
            )
        ) if run_identity is not None else ()
        components = _Qwen3PolicyTrajectoryComponents(
            context=context,
            layout_builder=layout_builder,
            server_client=server_client,
            contextual_forward_identity=contextual_forward_identity,
            branch_merger_identities=branch_mergers,
            observation_store=store,
            behavior_store=behavior_store,
            focus_execution_ledger=focus_execution_ledger,
            crop_execution_ledger=crop_execution_ledger,
            metrics_factory=self.metrics_factory,
            agent_loop_output_cls=self.agent_loop_output_cls,
            sample_index=_load_bound_sample_index(config),
        )
        return PolicyE2ERuntimeProduct(
            trajectory_components=components,
            snapshot_consumer=_IdentityOnlyLoRASnapshotConsumer(),
        )


class _Qwen3PolicyTrajectoryComponents:
    """Materialize one exact source binding and trajectory-owned collaborators."""

    def __init__(
        self,
        *,
        context: PolicyE2ERuntimeBuildContext,
        layout_builder: Qwen3NativeToolLayoutBuilder,
        server_client: object,
        contextual_forward_identity: ArtifactIdentity | None,
        branch_merger_identities: tuple[ArtifactIdentity, ...],
        observation_store: ObservationStore,
        behavior_store: BehaviorTraceStore,
        focus_execution_ledger: FocusExecutionLedger,
        crop_execution_ledger: CropExecutionLedger,
        metrics_factory: Callable[[TrajectoryRecord, PilotVerlTrajectoryReward], object],
        agent_loop_output_cls: type[Any] | None,
        sample_index: Mapping[str, Mapping[str, object]],
    ) -> None:
        self.context = context
        self.config = context.config
        self.layout_builder = layout_builder
        self.server_client = server_client
        self.contextual_forward_identity = contextual_forward_identity
        self.branch_merger_identities = branch_merger_identities
        self.store = observation_store
        self.behavior_store = behavior_store
        self.focus_execution_ledger = focus_execution_ledger
        self.crop_execution_ledger = crop_execution_ledger
        self.metrics_factory = metrics_factory
        self.agent_loop_output_cls = agent_loop_output_cls
        self.sample_index = sample_index
        self.reward_pipeline = _build_reward_pipeline(self.config)

    async def build_trajectory_components_async(
        self,
        *,
        identity: object,
        model: ModelIdentity,
        behavior_policy: PolicyVersion,
        initial_prompt_token_ids: tuple[int, ...],
        sample_fields: Mapping[str, object],
    ) -> VerlNativeTrajectoryComponents:
        from tgvf_rl.trajectories.schema import TrajectoryIdentity

        if not isinstance(identity, TrajectoryIdentity):
            raise TypeError("trajectory identity has the wrong type")
        if model != self.config.model:
            raise IdentityMismatchError("trajectory model differs from live runtime")
        _validate_sample_fields(
            self.config,
            identity.sample_id,
            sample_fields,
            sample_index=self.sample_index,
        )
        source_rgb = _load_bound_rgb(Path(_scalar(sample_fields["source_image_path"])))
        pixel_values, image_grid_thw = preprocess_qwen3_rgb(
            processor=self.context.processor,
            rgb=source_rgb,
            image_max_pixels=self.config.policy.image_max_pixels,
        )
        source = await self.server_client.materialize_source(
            request_id=identity.canonical_id,
            expected_step=behavior_policy.optimizer_step,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_sha256=tensor_checksum(source_rgb),
        )
        if not isinstance(source, SourceVisualTensorBundle):
            raise TypeError("vLLM source RPC returned an invalid visual bundle")
        source_positions = _source_visual_positions(
            initial_prompt_token_ids,
            image_token_id=self.layout_builder.image_pad_id,
            expected_count=int(source.merged_main.shape[-2]),
        )
        trajectory_source = record_trajectory_source_visual(
            trajectory_id=identity.canonical_id,
            source_visual=source,
            source_positions=source_positions,
            deepstack_branch_layers=_BRANCH_LAYERS,
            deepstack_injection_positions=tuple(
                source_positions for _ in _BRANCH_LAYERS
            ),
            observation_store=self.store,
            source_rgb=source_rgb,
        )

        registry = LiveVLLMTurnContextRegistry(
            observation_resolver=Qwen3VLLMObservationPayloadResolver(
                store=self.store,
                include_multi_modal_uuid=False,
            )
        )
        initial_inputs = _initial_vllm_inputs(
            initial_prompt_token_ids=initial_prompt_token_ids,
            image_token_id=self.layout_builder.image_pad_id,
            source=source,
            image_max_pixels=self.config.policy.image_max_pixels,
        )
        registry.register_initial_prompt(initial_prompt_token_ids, initial_inputs)
        appender = QwenNativeToolObservationAppender(
            tokenizer=self.layout_builder.tokenizer,
            registrar=registry,
            visual_token_count_resolver=_VisualTokenCountResolver(self.store),
        )
        parser = StrictToolCallParser(
            enabled_tool_names=self.config.protocol.enabled_tool_names
        )
        if (
            self.config.protocol.tool_profile
            is NativeToolCapabilityProfile.TGVF_ONLY
        ):
            tool_runtime = _RemoteTGVFFocusToolRuntime(
                event_loop=asyncio.get_running_loop(),
                server_client=self.server_client,
                config=self.config,
                source_visual=source,
                layout_builder=self.layout_builder,
                observation_store=self.store,
                execution_ledger=self.focus_execution_ledger,
                contextual_forward_identity=self.contextual_forward_identity,
                branch_merger_identities=self.branch_merger_identities,
            )
        elif (
            self.config.protocol.tool_profile
            is NativeToolCapabilityProfile.CROP_ONLY
        ):
            crop_processor_identity = _artifact_identity(
                "policy-runtime",
                "qwen3-shared-vllm-crop-processor",
                QWEN3_POLICY_E2E_LIVE_RUNTIME_SCHEMA,
                {
                    "model": self.config.model.revision_or_path,
                    "max_pixels": self.config.policy.image_max_pixels,
                },
            )
            crop_layout_identity = _artifact_identity(
                "policy-runtime",
                "qwen3-native-crop-layout",
                QWEN3_POLICY_E2E_LIVE_RUNTIME_SCHEMA,
                {"model": self.config.model.revision_or_path},
            )
            crop_materializer = _RemoteCropVisualMaterializer(
                event_loop=asyncio.get_running_loop(),
                server_client=self.server_client,
                processor=self.context.processor,
                model_identity=model,
                image_max_pixels=self.config.policy.image_max_pixels,
                trajectory_id=identity.canonical_id,
                behavior_policy=behavior_policy,
            )
            tool_runtime = ImageZoomInToolRuntime(
                model=model,
                materializer=crop_materializer,
                layout_builder=self.layout_builder,
                observation_store=self.store,
                crop_processor_identity=crop_processor_identity,
                crop_layout_identity=crop_layout_identity,
                execution_ledger=self.crop_execution_ledger,
            )
        else:  # guarded by the builder
            raise RuntimeError("unsupported live visual-tool profile")

        def native_loop_factory(sampler: object) -> FrameworkNeutralAgentLoop:
            return FrameworkNeutralAgentLoop(
                sampler=sampler,  # type: ignore[arg-type]
                tool_runtime=tool_runtime,
                appender=appender,
                parser=parser,
                behavior_recorder=VLLMBehaviorRecorder(self.behavior_store),
                max_tool_calls=self.config.protocol.maximum_tool_calls,
                enabled_tool_names=self.config.protocol.enabled_tool_names,
                cap_error_behavior=CapErrorBehavior.ONE_FINAL_ANSWER_TURN,
            )

        reward_context = _BoundRewardContextProvider(
            **_reward_source_from_sample_fields(sample_fields),
        )
        scorer = PilotVerlTrajectoryRewardScorer(
            pipeline=self.reward_pipeline,
            context_provider=reward_context,
            audit_sink=PolicyTrajectoryAuditWriter(
                Path(self.config.output.root) / "trajectory_audit"
            ).record,
        )
        finalizer = _ExactQwen3RewardedTrajectoryFinalizer(
            request_identity=identity,
            model=model,
            behavior_policy=behavior_policy,
            initial_prompt_token_ids=initial_prompt_token_ids,
            source_visual=trajectory_source,
            layout_builder=self.layout_builder,
            observation_store=self.store,
            behavior_store=self.behavior_store,
        )
        request_proxy = _RewardRequestProxy(identity)
        output_builder = VerlRewardedAgentLoopOutputBuilder(
            request=request_proxy,
            scorer=scorer,
            finalizer=finalizer,
            metrics_factory=self.metrics_factory,
            agent_loop_output_cls=self.agent_loop_output_cls,
        )
        return VerlNativeTrajectoryComponents(
            source_visual=trajectory_source,
            native_loop_factory=native_loop_factory,
            prompt_context=registry,
            output_builder=output_builder,
        )


class _IdentityOnlyLoRASnapshotConsumer:
    """Validate snapshot identity without constructing a local policy model."""

    def apply_policy_lora_snapshot(self, snapshot: object, /) -> PolicyVersion:
        from .policy_weight_sync import (
            PolicyLoRASnapshot,
            lora_parameter_mapping_sha256,
        )

        if not isinstance(snapshot, PolicyLoRASnapshot):
            raise TypeError("Policy version consumer requires PolicyLoRASnapshot")
        digest = lora_parameter_mapping_sha256(snapshot.tensors)
        if digest != snapshot.policy_version.weights_sha256:
            raise ReplayMismatchError("LoRA snapshot tensor identity differs")
        return snapshot.policy_version


class _RemoteTGVFFocusToolRuntime:
    """Bridge a synchronous native tool call to the sticky vLLM worker."""

    def __init__(
        self,
        *,
        event_loop: asyncio.AbstractEventLoop,
        server_client: object,
        config: object,
        source_visual: SourceVisualTensorBundle,
        layout_builder: Qwen3NativeToolLayoutBuilder,
        observation_store: ObservationStore,
        execution_ledger: FocusExecutionLedger,
        contextual_forward_identity: ArtifactIdentity | None,
        branch_merger_identities: tuple[ArtifactIdentity, ...],
    ) -> None:
        self.event_loop = event_loop
        self.server_client = server_client
        self.config = config
        self.source_visual = source_visual
        self.layout_builder = layout_builder
        self.focus_tool = TGVFFocusTool(None, observation_store)
        self.execution_ledger = execution_ledger
        self.contextual_forward_identity = contextual_forward_identity
        self.branch_merger_identities = tuple(branch_merger_identities)

    def execute(self, parsed_call: object, context: object) -> ObservationHandle:
        from tgvf_rl.environment.agent_loop import ToolExecutionContext

        if not isinstance(parsed_call, ParsedToolCall):
            raise TypeError("remote focus runtime requires ParsedToolCall")
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("remote focus runtime requires ToolExecutionContext")
        if (
            parsed_call.sampled_text != context.sampled_turn.text
            or parsed_call.sampled_token_ids != context.sampled_turn.token_ids
            or parsed_call.sampled_token_byte_spans
            != context.sampled_turn.token_byte_spans
        ):
            raise ReplayMismatchError("parsed TGVF call differs from sampled turn")
        conditioning = self.config.representation.conditioning
        fingerprint = _call_fingerprint(
            parsed_call=parsed_call,
            context=context,
            provider_name=conditioning.provider.value,
            hidden_layer=conditioning.hidden_layer,
            contextual_forward_identity=self.contextual_forward_identity,
            representation=self.config.representation.artifact,
            branch_mergers=self.branch_merger_identities,
        )
        return self.execution_ledger.execute_once(
            key=(context.trajectory_identity.canonical_id, context.call_index),
            fingerprint=fingerprint,
            operation=lambda: self._execute_once(parsed_call, context),
        )

    def _execute_once(self, parsed_call: ParsedToolCall, context: object) -> ObservationHandle:
        from tgvf_rl.environment.agent_loop import ToolExecutionContext

        assert isinstance(context, ToolExecutionContext)
        trajectory_id = context.trajectory_identity.canonical_id
        target = parsed_call.target_span
        future = asyncio.run_coroutine_threadsafe(
            self.server_client.materialize_focus(
                request_id=trajectory_id,
                expected_step=context.behavior_policy.optimizer_step,
                sampled_output_ids=context.sampled_turn.token_ids,
                target_start=target.token_start,
                target_end=target.token_end,
                expected_target_token_ids=target.token_ids,
                provider=self.config.representation.conditioning.provider.value,
            ),
            self.event_loop,
        )
        hq, adapter_output = future.result(timeout=300.0)
        prefix_length = len(context.prompt_token_ids_before_turn)
        global_span = TokenSpan(
            prefix_length + target.token_start,
            prefix_length + target.token_end,
        )
        input_ids = torch.tensor(context.conditioning_input_ids, dtype=torch.long)
        conditioning = self.config.representation.conditioning
        condition = bind_preselected_target_conditioning(
            values=hq,
            input_ids=input_ids,
            target_span=global_span,
            expected_target_token_ids=target.token_ids,
            trajectory_id=trajectory_id,
            call_index=context.call_index,
            model_identity=context.model,
            provider=conditioning.provider,
            hidden_layer=conditioning.hidden_layer,
            embedding_identity=conditioning.embedding_identity,
        )
        layout = self.layout_builder.build_focus_from_recorded_prefix(
            conditioning_input_ids=context.conditioning_input_ids,
            parsed_call=parsed_call,
            trajectory_source_visual=context.trajectory_source_visual,
            prior_observation_handles=context.prior_observation_handles,
            source_visual=self.source_visual,
        )
        result = self.focus_tool.record_precomputed(
            ToolExecutionRequest(
                trajectory_id=trajectory_id,
                call_index=context.call_index,
                parsed_call=parsed_call,
                condition=condition,
                source_visual=self.source_visual,
                layout=layout,
                model=context.model,
                policy_version=context.behavior_policy,
                contextual_forward_identity=self.contextual_forward_identity,
                representation=self.config.representation.artifact,
                branch_merger_identities=self.branch_merger_identities,
            ),
            adapter_output,
        )
        if not isinstance(result, ToolExecutionResult):
            raise TypeError("remote TGVF materialization returned an invalid result")
        return result.handle


class _RemoteCropVisualMaterializer:
    """Run crop vision on the same sticky vLLM replica as rollout sampling."""

    def __init__(
        self,
        *,
        event_loop: asyncio.AbstractEventLoop,
        server_client: object,
        processor: object,
        model_identity: ModelIdentity,
        image_max_pixels: int,
        trajectory_id: str,
        behavior_policy: PolicyVersion,
    ) -> None:
        self.event_loop = event_loop
        self.server_client = server_client
        self.processor = processor
        self.model_identity = model_identity
        self.image_max_pixels = image_max_pixels
        self.trajectory_id = trajectory_id
        self.behavior_policy = behavior_policy

    def materialize(
        self,
        crop_rgb: torch.Tensor,
        *,
        parsed_call: object,
        call_index: int,
    ) -> CropVisualTensorBundle:
        if not isinstance(parsed_call, ParsedImageZoomInCall):
            raise TypeError("remote crop materializer requires ParsedImageZoomInCall")
        pixel_values, image_grid_thw = preprocess_qwen3_rgb(
            processor=self.processor,
            rgb=crop_rgb,
            image_max_pixels=self.image_max_pixels,
        )
        future = asyncio.run_coroutine_threadsafe(
            self.server_client.materialize_crop(
                request_id=self.trajectory_id,
                expected_step=self.behavior_policy.optimizer_step,
                sampled_output_ids=parsed_call.sampled_token_ids,
                call_index=call_index,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                crop_sha256=tensor_checksum(crop_rgb),
            ),
            self.event_loop,
        )
        source = future.result(timeout=300.0)
        if not isinstance(source, SourceVisualTensorBundle):
            raise TypeError("remote crop RPC returned an invalid visual bundle")
        return CropVisualTensorBundle(
            merged_main=source.merged_main,
            merged_deepstack=source.merged_deepstack,
            image_grid_thw=source.image_grid_thw,
            spatial_merge_size=source.spatial_merge_size,
            deepstack_branch_layers=_BRANCH_LAYERS,
        )


class _ExactQwen3RewardedTrajectoryFinalizer(RewardedTrajectoryFinalizerPort):
    """Freeze exact final IDs/layout and export one role-shared replay bundle."""

    def __init__(
        self,
        *,
        request_identity: object,
        model: ModelIdentity,
        behavior_policy: PolicyVersion,
        initial_prompt_token_ids: tuple[int, ...],
        source_visual: TrajectorySourceVisual,
        layout_builder: Qwen3NativeToolLayoutBuilder,
        observation_store: ObservationStore,
        behavior_store: BehaviorTraceStore,
    ) -> None:
        self.request_identity = request_identity
        self.model = model
        self.behavior_policy = behavior_policy
        self.initial_prompt_token_ids = initial_prompt_token_ids
        self.source_visual = source_visual
        self.layout_builder = layout_builder
        self.store = observation_store
        self.behavior_store = behavior_store

    def finalize(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
        reward: object,
    ) -> RolloutBridgeRecord:
        from tgvf_rl.rewards.schema import RewardResult

        if getattr(request, "identity", None) != self.request_identity:
            raise IdentityMismatchError("reward finalization request identity changed")
        if not isinstance(reward, RewardResult):
            raise TypeError("reward finalizer requires RewardResult")
        if trajectory.identity != self.request_identity:
            raise IdentityMismatchError("trajectory identity changed before finalization")
        if trajectory.model != self.model or trajectory.behavior_policy != self.behavior_policy:
            raise IdentityMismatchError("trajectory model/policy changed before replay")

        final_ids, native_rows = _final_token_materialization(
            self.initial_prompt_token_ids,
            trajectory,
        )
        handles = tuple(item.handle for item in trajectory.observations)
        records = tuple(self.store.resolve_record(handle) for handle in handles)
        crop_vision_replay_mode = (
            "shared_frozen_recorded_features"
            if any(isinstance(record, CropObservationRecord) for record in records)
            else "no_crop"
        )
        expanded = self.layout_builder.expand_recorded_visual_sequence(
            final_ids,
            trajectory_source_visual=self.source_visual,
            observation_handles=handles,
        )
        if tuple(int(value) for value in expanded.input_ids[0].tolist()) != final_ids:
            raise ReplayMismatchError(
                "final replay attempted to grow already-expanded visual token runs"
            )
        base_mask = expanded.attention_mask.to(dtype=torch.bool)
        replay_id = "policy-replay:" + _canonical_sha256(
            {
                "trajectory_sha256": trajectory_checksum(trajectory),
                "final_ids": final_ids,
                "reward": reward.total,
            }
        )
        replay_request = TrajectoryReplayFinalizationRequest(
            trajectory=trajectory,
            source_visual=self.source_visual,
            tensors=MaterializedTrajectoryReplayTensors(
                input_ids=expanded.input_ids,
                position_ids=expanded.position_ids,
                base_attention_mask=base_mask,
                policy_attention_mask=base_mask.clone(),
                reference_attention_mask=base_mask.clone(),
                teacher_attention_mask=base_mask.clone(),
            ),
            replay_schema_version="trajectory-replay-v1",
            replay_id=replay_id,
            trajectory_id=trajectory.identity.canonical_id,
            model=self.model,
            behavior_policy=self.behavior_policy,
            crop_vision_replay_mode=crop_vision_replay_mode,
            cache_mode="no_cache",
            cache_prefix_length=0,
            deterministic_forward=True,
            adapter_dropout=0.0,
            maximum_policy_staleness=0,
            initial_prompt_token_ids=self.initial_prompt_token_ids,
            native_tool_appended_token_ids=native_rows,
            sentinel_fields=make_objective_sentinels(
                prefix="policy-e2e:" + _canonical_sha256(trajectory.identity.canonical_id)[:16]
            ),
            extra_fields={
                "tgvf_exact_replay_roles": ("current", "reference"),
                "tgvf_exact_replay_observation_count": len(handles),
            },
            reward_score=float(reward.total),
        )
        return finalize_trajectory_replay(
            replay_request,
            observation_store=self.store,
            behavior_store=self.behavior_store,
        )


@dataclass(frozen=True, slots=True)
class _RewardRequestProxy:
    identity: object


@dataclass(frozen=True, slots=True)
class _SourceMaterializationRequest:
    trajectory_id: str


@dataclass(frozen=True, slots=True)
class _BoundRewardContextProvider(PilotRewardContextProvider):
    question: str
    expected_answer: str
    task_kind: AnswerTaskKind
    data_source: str

    def build(self, *, request: object, trajectory: TrajectoryRecord):
        if getattr(request, "identity", None) != trajectory.identity:
            raise IdentityMismatchError("reward context request/trajectory differ")
        return reward_context_from_trajectory(
            trajectory,
            question=self.question,
            expected_answer=self.expected_answer,
            task_kind=self.task_kind,
            data_source=self.data_source,
        )


class _VisualTokenCountResolver:
    def __init__(self, store: ObservationStore) -> None:
        self.store = store

    def resolve_visual_token_count(self, observation: ObservationHandle) -> int:
        record = self.store.resolve_record(observation)
        if isinstance(record, FocusedObservationRecord):
            return len(record.layout.d_positions)
        if isinstance(record, CropObservationRecord):
            return len(record.crop_visual.positions)
        raise TypeError("live runtime received an unsupported visual observation")


def _initial_vllm_inputs(
    *,
    initial_prompt_token_ids: tuple[int, ...],
    image_token_id: int,
    source: object,
    image_max_pixels: int,
) -> VLLMLivePromptInputs:
    merged_main = source.merged_main
    merged_deepstack = tuple(source.merged_deepstack)
    if len(merged_deepstack) != 3 or any(
        item.shape != merged_main.shape
        or item.dtype != merged_main.dtype
        or item.device != merged_main.device
        for item in merged_deepstack
    ):
        raise ValueError("source main/DeepStack features do not align")
    packed = torch.cat((merged_main, *merged_deepstack), dim=-1).detach().cpu()
    grid = torch.tensor((source.image_grid_thw,), dtype=torch.long)
    base_kwargs = {"max_pixels": image_max_pixels}
    mm_kwargs = bind_preexpanded_prompt_contract(
        base_kwargs,
        prompt_token_ids=initial_prompt_token_ids,
        image_token_id=image_token_id,
        expected_image_items=1,
    )
    payload_sha = _canonical_sha256(
        {
            "schema": "tgvf-qwen3-source-vllm-payload-v1",
            "prompt": initial_prompt_token_ids,
            "image_embeds_sha256": tensor_checksum(packed),
            "image_grid_thw": source.image_grid_thw,
            "max_pixels": image_max_pixels,
        }
    )
    return VLLMLivePromptInputs(
        backend_prompt_payload_sha256=payload_sha,
        multi_modal_data={
            "image": [
                {
                    "image_embeds": packed,
                    "image_grid_thw": grid,
                }
            ]
        },
        mm_processor_kwargs=mm_kwargs,
        multi_modal_uuids=None,
    )


def _source_visual_positions(
    token_ids: tuple[int, ...], *, image_token_id: int, expected_count: int
) -> tuple[int, ...]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(token_ids):
        if token_ids[index] != image_token_id:
            index += 1
            continue
        start = index
        while index < len(token_ids) and token_ids[index] == image_token_id:
            index += 1
        runs.append((start, index))
    if len(runs) != 1:
        raise ReplayMismatchError("initial prompt must contain one source visual run")
    start, end = runs[0]
    if end - start != expected_count:
        raise ReplayMismatchError(
            "source visual feature count differs from expanded prompt tokens"
        )
    return tuple(range(start, end))


def _final_token_materialization(
    initial_prompt_token_ids: tuple[int, ...], trajectory: TrajectoryRecord
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    observation_rows = {
        call.assistant_turn_index: observation.template_token_ids
        for call, observation in zip(
            trajectory.tool_calls, trajectory.observations, strict=True
        )
    }
    error_rows = {
        error.assistant_turn_index: error.template_token_ids
        for error in trajectory.tool_errors
    }
    final = list(initial_prompt_token_ids)
    native_rows: list[tuple[int, ...]] = []
    for expected_turn, turn in enumerate(trajectory.assistant_turns):
        if turn.turn_index != expected_turn:
            raise ReplayMismatchError("assistant turns are not contiguous")
        final.extend(turn.tokens.token_ids)
        row = observation_rows.get(turn.turn_index, error_rows.get(turn.turn_index))
        if row is not None:
            native_rows.append(tuple(row))
            final.extend(row)
    return tuple(final), tuple(native_rows)


def _build_reward_pipeline(config: object) -> PilotRewardPipeline:
    reward = config.reward
    answer_identity = ArtifactIdentity(
        "policy-reward",
        reward.answer_verifier,
        "pilot-v1",
        reward.answer_verifier_sha256,
    )
    format_identity = _artifact_identity(
        "policy-reward", "native-format", "pilot-v1", {"run": config.identity_sha256}
    )
    tool_identity = _artifact_identity(
        "policy-reward",
        "conditional-visual-tool",
        "pilot-v1",
        {
            "run": config.identity_sha256,
            "tools": config.protocol.enabled_tool_names,
        },
    )
    pipeline_identity = _artifact_identity(
        "policy-reward",
        "pilot-reward-equation",
        "0.8-0.2-1.2",
        {
            "run": config.identity_sha256,
            "answer": answer_identity.sha256,
            "format": format_identity.sha256,
            "tool": tool_identity.sha256,
            "judge_config": reward.judge_config_sha256,
        },
    )
    if reward.judge_config_path is None:
        disabled = _artifact_identity(
            "judge", "disabled-qwen2.5-72b", "pilot-smoke", {"disabled": True}
        )
        judge = DisabledJudgeProvider()
        judge_prompt_identity = disabled
        judge_model_identity = disabled
        judge_service_identity = disabled
        judge_sampling_identity = disabled
        judge_calibration_identity = disabled
    else:
        if reward.judge_config_sha256 is None:
            raise ValueError("enabled RL judge lacks its config identity")
        bound_judge = load_openai_compatible_judge(
            reward.judge_config_path,
            expected_file_sha256=reward.judge_config_sha256,
        )
        bound_judge.provider.validate_credentials()
        judge = bound_judge.provider
        judge_prompt_identity = bound_judge.prompt_identity
        judge_model_identity = bound_judge.model_identity
        judge_service_identity = bound_judge.service_identity
        judge_sampling_identity = bound_judge.sampling_identity
        judge_calibration_identity = bound_judge.calibration_identity
    verifier = RuleFirstAnswerVerifier(
        rule_identity=answer_identity,
        normalization=NormalizationSpec(True, True, True),
        judge=judge,
        judge_prompt_identity=judge_prompt_identity,
        judge_model_identity=judge_model_identity,
        judge_service_identity=judge_service_identity,
        judge_sampling_identity=judge_sampling_identity,
        judge_calibration_identity=judge_calibration_identity,
    )
    return PilotRewardPipeline(
        PilotRewardSpec(
            pipeline_identity=pipeline_identity,
            answer_verifier_identity=answer_identity,
            format_verifier_identity=format_identity,
            tool_verifier_identity=tool_identity,
        ),
        verifier,
    )


def _validate_sample_fields(
    config: object,
    sample_id: str,
    fields: Mapping[str, object],
    *,
    sample_index: Mapping[str, Mapping[str, object]],
) -> None:
    sample = config.dataset.selected_sample
    if sample is None:
        try:
            record = sample_index[sample_id]
        except KeyError as error:
            raise IdentityMismatchError("upstream sample_id is absent from DeepEyes") from error
        image = record.get("image")
        extra_info = record.get("extra_info")
        bound_reward = record.get("reward_model")
        if (
            not isinstance(image, Mapping)
            or not isinstance(extra_info, Mapping)
            or not isinstance(bound_reward, Mapping)
        ):
            raise IdentityMismatchError("bound DeepEyes sample schema differs")
        relative_image = image.get("path")
        if not isinstance(relative_image, str):
            raise IdentityMismatchError("bound DeepEyes image path differs")
        source_image_path = (config.dataset.root / relative_image).resolve()
        expected = {
            "sample_id": sample_id,
            "dataset_iteration_identity_sha256": (
                config.dataset.iteration_identity_sha256
            ),
            "prompt_bundle_sha256": config.protocol.prompt_sha256,
            "source_image_path": str(source_image_path),
            "source_image_sha256": image.get("sha256"),
            "question": extra_info.get("question"),
            "data_source": record.get("data_source"),
            "task_kind": record.get("task_kind"),
        }
        expected_ground_truth = bound_reward.get("ground_truth")
    else:
        expected = {
            "sample_id": sample_id,
            "source_image_path": str(sample.image_path),
            "source_image_sha256": sample.image_sha256,
            "question": sample.question,
            "data_source": sample.data_source,
            "task_kind": sample.task_kind,
        }
        expected_ground_truth = sample.ground_truth
    for key, value in expected.items():
        if key not in fields or _scalar(fields[key]) != value:
            raise IdentityMismatchError(f"upstream sample field {key!r} changed")
    reward_model = _scalar(fields.get("reward_model"))
    if (
        not isinstance(reward_model, Mapping)
        or reward_model.get("ground_truth") != expected_ground_truth
    ):
        raise IdentityMismatchError("upstream ground truth changed")


def _load_bound_sample_index(
    config: object,
) -> Mapping[str, Mapping[str, object]]:
    if config.dataset.selected_sample is not None:
        return {}
    samples_path = config.dataset.root / "samples.jsonl"
    if samples_path.is_symlink() or not samples_path.is_file():
        raise ValueError("bound DeepEyes samples file is missing or unsafe")
    if _sha256_file(samples_path) != config.dataset.samples_sha256:
        raise IdentityMismatchError("bound DeepEyes samples file changed")
    records: dict[str, Mapping[str, object]] = {}
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError("bound DeepEyes row must be an object")
            row_sample_id = record.get("sample_id")
            if not isinstance(row_sample_id, str) or row_sample_id in records:
                raise ValueError("bound DeepEyes sample identity differs")
            records[row_sample_id] = record
    expected_count = config.dataset.runtime_binding.expected_sample_count
    if len(records) != expected_count:
        raise IdentityMismatchError("bound DeepEyes sample count changed")
    return records


def _reward_source_from_sample_fields(
    fields: Mapping[str, object],
) -> dict[str, object]:
    question = _scalar(fields.get("question"))
    data_source = _scalar(fields.get("data_source"))
    task_kind = _scalar(fields.get("task_kind"))
    reward_model = _scalar(fields.get("reward_model"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("sample question must be non-empty")
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("sample data_source must be non-empty")
    if not isinstance(reward_model, Mapping):
        raise ValueError("sample reward_model must be a mapping")
    expected_answer = reward_model.get("ground_truth")
    if not isinstance(expected_answer, str) or not expected_answer.strip():
        raise ValueError("sample ground truth must be non-empty text")
    task_kinds = {
        "mcq": AnswerTaskKind.MULTIPLE_CHOICE,
        "math": AnswerTaskKind.MATH,
        "open": AnswerTaskKind.OPEN_VQA,
    }
    try:
        resolved_kind = task_kinds[task_kind]
    except (KeyError, TypeError) as error:
        raise ValueError("sample task_kind is not mcq, math, or open") from error
    return {
        "question": question,
        "expected_answer": expected_answer,
        "task_kind": resolved_kind,
        "data_source": data_source,
    }


def _load_bound_rgb(path: Path) -> torch.Tensor:
    if path.is_symlink() or not path.is_file():
        raise ValueError("source image must be a regular file")
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - accepted runtime owns numpy
            raise RuntimeError("source materialization requires numpy") from error
        array = np.asarray(rgb, dtype=np.uint8).copy()
    return torch.from_numpy(array)


def _scalar(value: object) -> object:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _artifact_identity(
    namespace: str, name: str, version: str, payload: object
) -> ArtifactIdentity:
    return ArtifactIdentity(namespace, name, version, _canonical_sha256(payload))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_metrics_factory(
    trajectory: TrajectoryRecord, reward: PilotVerlTrajectoryReward
) -> object:
    del trajectory, reward
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
    except ImportError as error:  # pragma: no cover - accepted live env owns veRL
        raise RuntimeError("live AgentLoop metrics require the pinned veRL") from error
    return AgentLoopMetrics()


__all__ = [
    "QWEN3_POLICY_E2E_LIVE_RUNTIME_SCHEMA",
    "Qwen3PolicyE2ELiveRuntimeBuilder",
]
