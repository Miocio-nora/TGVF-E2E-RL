"""Concrete Qwen3 composition for the Policy E2E native agent loop.

This is the production default behind :class:`PolicyE2ERuntimeInvocationFactory`.
It owns one frozen local Qwen3 visual/readout model per AgentLoop worker, loads
the selected representation export, installs the exact served decoder LoRA
snapshot, materializes source/D tensors once, and exports one self-contained
replay bundle before a trajectory crosses the veRL boundary.

The current and reference actors consume the same exported replay tensors.
This module never asks either role to regenerate a TGVF observation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any

import torch
from PIL import Image
from torch import nn

from tgvf_rl.conditioning import (
    TargetConditioningDependencies,
    TargetConditioningProviderKind,
    TargetTokenEmbeddingConditionProvider,
)
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.environment import (
    BehaviorHiddenStateMaterialization,
    BranchMergerRuntimeBinding,
    FocusExecutionLedger,
    FrameworkNeutralAgentLoop,
    Qwen3CropVisualMaterializer,
    Qwen3NativeToolLayoutBuilder,
    QwenNativeToolObservationAppender,
    RepresentationArtifactRuntimeBinding,
    build_policy_pilot_focus_runtime,
    record_trajectory_source_visual,
)
from tgvf_rl.environment.adapter_runtime import (
    BehaviorHiddenStateDependency,
    BoundSourceVisual,
)
from tgvf_rl.environment.focus_runtime import BehaviorHiddenStateCaptureRequest
from tgvf_rl.environment.focus_tool import ReplayLayoutTensors
from tgvf_rl.protocol.state_machine import CapErrorBehavior
from tgvf_rl.observations.finalizer import (
    MaterializedTrajectoryReplayTensors,
    TrajectoryReplayFinalizationRequest,
    finalize_trajectory_replay,
)
from tgvf_rl.observations.schema import FocusedObservationRecord, TrajectorySourceVisual
from tgvf_rl.observations.store import (
    ObservationHandle,
    ObservationStore,
    tensor_checksum,
)
from tgvf_rl.policy.qwen_replay import build_qwen3_decoder_lora_policy
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.qwen import InjectedForwardRequest, InjectedVisualBlock, Qwen3VLAdapter
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)
from tgvf_rl.representation.training.runtime import create_qwen3_representation_runtime
from tgvf_rl.rewards import (
    AnswerTaskKind,
    PilotRewardPipeline,
    PilotRewardSpec,
    RuleFirstAnswerVerifier,
    reward_context_from_trajectory,
)
from tgvf_rl.rewards.schema import NormalizationSpec
from tgvf_rl.judges import DisabledJudgeProvider
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
    PeftPolicyLoRASnapshotConsumer,
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
    """Build the real process-local Policy E2E runtime on the assigned GPU."""

    singleton_identity = QWEN3_POLICY_E2E_LIVE_RUNTIME_SCHEMA

    def __init__(
        self,
        *,
        model_loader: Callable[[PolicyE2ERuntimeBuildContext], nn.Module] | None = None,
        agent_loop_output_cls: type[Any] | None = None,
        metrics_factory: Callable[[TrajectoryRecord, PilotVerlTrajectoryReward], object]
        | None = None,
    ) -> None:
        if model_loader is not None and not callable(model_loader):
            raise TypeError("model_loader must be callable")
        if metrics_factory is not None and not callable(metrics_factory):
            raise TypeError("metrics_factory must be callable")
        self.model_loader = model_loader or _load_local_qwen3_model
        self.agent_loop_output_cls = agent_loop_output_cls
        self.metrics_factory = metrics_factory or _default_metrics_factory

    def build(self, context: PolicyE2ERuntimeBuildContext, /) -> PolicyE2ERuntimeProduct:
        if not isinstance(context, PolicyE2ERuntimeBuildContext):
            raise TypeError("live runtime builder requires PolicyE2ERuntimeBuildContext")
        config = context.config
        if config.model.family != "qwen3_vl":
            raise ValueError("Policy Pilot live runtime requires qwen3_vl")
        device = context.placement.torch_device
        model = self.model_loader(context)
        if not isinstance(model, nn.Module):
            raise TypeError("model_loader must return an nn.Module")
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()

        export = load_rank_zero_adapter_owned_state_export(
            config.representation.artifact_path
        )
        run_identity = export.manifest.run_identity
        if run_identity.model != config.model:
            raise IdentityMismatchError(
                "representation export and Policy model identities differ"
            )
        if run_identity.provider != config.representation.conditioning:
            raise IdentityMismatchError(
                "representation export and selected conditioning differ"
            )

        # Construct the Adapter against the exact live Qwen modules before PEFT
        # wraps decoder layers.  The representation loader later freezes and
        # state-loads only Adapter-owned tensors.
        representation_runtime = create_qwen3_representation_runtime(
            model=model,
            processor=context.processor,
            model_identity=config.model,
            conditioning_config=config.representation.conditioning,
            adapter_dtype=torch.bfloat16,
        )
        lora_build = build_qwen3_decoder_lora_policy(
            model,
            config=config.policy.lora,
        )
        policy_model = lora_build.model
        policy_model.eval()
        forward_model = policy_model.get_base_model()
        model_lock = RLock()
        snapshot_consumer = PeftPolicyLoRASnapshotConsumer(
            policy_model,
            model_lock=model_lock,
        )

        store = ObservationStore()
        behavior_store = BehaviorTraceStore()
        execution_ledger = FocusExecutionLedger()
        layout_builder = Qwen3NativeToolLayoutBuilder.from_model(
            model=forward_model,
            processor=context.processor,
            model_identity=config.model,
            observation_store=store,
        )
        source_materializer = Qwen3CropVisualMaterializer.from_model(
            model=forward_model,
            processor=context.processor,
            model_identity=config.model,
            image_max_pixels=config.policy.image_max_pixels,
        )
        conditioning = config.representation.conditioning
        if (
            conditioning.provider
            is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
        ):
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
            hidden_dependency = _Qwen3BehaviorHiddenStateDependency(
                model=forward_model,
                family_adapter=Qwen3VLAdapter(),
                layout_builder=layout_builder,
                store=store,
                device=device,
                forward_identity=contextual_forward_identity,
                model_lock=model_lock,
            )
            conditioning_dependencies = TargetConditioningDependencies()
        else:
            provider = representation_runtime.conditioning_provider
            if not isinstance(provider, TargetTokenEmbeddingConditionProvider):
                raise TypeError(
                    "representation runtime lost the target-token embedding provider"
                )
            contextual_forward_identity = None
            hidden_dependency = None
            conditioning_dependencies = TargetConditioningDependencies(
                base_embedding=provider.borrowed_embedding
            )
        layout_dependency = _Qwen3FocusReplayLayoutDependency(layout_builder)
        artifact_binding = RepresentationArtifactRuntimeBinding(
            artifact_path=config.representation.artifact_path,
            artifact=config.representation.artifact,
            expected_run_id=config.representation.expected_run_id,
            expected_run_identity_sha256=(
                config.representation.expected_run_identity_sha256
            ),
            model=config.model,
            conditioning=config.representation.conditioning,
            adapter_contract=run_identity.adapter_contract,
        )
        branch_mergers = tuple(
            BranchMergerRuntimeBinding(
                projection_identity=projection_identity,
                artifact=_artifact_identity(
                    "qwen3-vl",
                    f"deepstack-merger-{layer}",
                    "frozen-base-model-v1",
                    {
                        "model": config.model.revision_or_path,
                        "projection_identity": projection_identity,
                        "layer": layer,
                    },
                ),
            )
            for layer, projection_identity in zip(
                _BRANCH_LAYERS,
                run_identity.adapter_contract.deepstack_projection_identities,
                strict=True,
            )
        )
        focus_bridge = build_policy_pilot_focus_runtime(
            artifact_binding=artifact_binding,
            adapter=representation_runtime.adapter,
            conditioning_dependencies=conditioning_dependencies,
            contextual_hidden_state_dependency=hidden_dependency,
            contextual_forward_identity=contextual_forward_identity,
            replay_layout_dependency=layout_dependency,
            branch_mergers=branch_mergers,
            observation_store=store,
            execution_ledger=execution_ledger,
            runtime_device=device,
        )
        components = _Qwen3PolicyTrajectoryComponents(
            context=context,
            policy_model=policy_model,
            forward_model=forward_model,
            source_materializer=source_materializer,
            layout_builder=layout_builder,
            tool_runtime=focus_bridge.runtime,
            observation_store=store,
            behavior_store=behavior_store,
            execution_ledger=execution_ledger,
            metrics_factory=self.metrics_factory,
            agent_loop_output_cls=self.agent_loop_output_cls,
        )
        return PolicyE2ERuntimeProduct(
            trajectory_components=components,
            snapshot_consumer=snapshot_consumer,
        )


class _Qwen3PolicyTrajectoryComponents:
    """Materialize one exact source binding and trajectory-owned collaborators."""

    def __init__(
        self,
        *,
        context: PolicyE2ERuntimeBuildContext,
        policy_model: nn.Module,
        forward_model: nn.Module,
        source_materializer: Qwen3CropVisualMaterializer,
        layout_builder: Qwen3NativeToolLayoutBuilder,
        tool_runtime: object,
        observation_store: ObservationStore,
        behavior_store: BehaviorTraceStore,
        execution_ledger: FocusExecutionLedger,
        metrics_factory: Callable[[TrajectoryRecord, PilotVerlTrajectoryReward], object],
        agent_loop_output_cls: type[Any] | None,
    ) -> None:
        self.context = context
        self.config = context.config
        self.policy_model = policy_model
        self.forward_model = forward_model
        self.source_materializer = source_materializer
        self.layout_builder = layout_builder
        self.tool_runtime = tool_runtime
        self.store = observation_store
        self.behavior_store = behavior_store
        self.execution_ledger = execution_ledger
        self.metrics_factory = metrics_factory
        self.agent_loop_output_cls = agent_loop_output_cls
        self.reward_pipeline = _build_reward_pipeline(self.config)

    def build_trajectory_components(
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
        _validate_sample_fields(self.config, identity.sample_id, sample_fields)
        source_rgb = _load_bound_rgb(Path(_scalar(sample_fields["source_image_path"])))
        source = self.source_materializer.materialize_source_visual(
            source_rgb,
            parsed_call=_SourceMaterializationRequest(identity.canonical_id),
            call_index=0,
        )
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
            visual_token_count_resolver=_FocusedVisualTokenCountResolver(self.store),
        )
        parser = StrictToolCallParser(
            enabled_tool_names=self.config.protocol.enabled_tool_names
        )

        def native_loop_factory(sampler: object) -> FrameworkNeutralAgentLoop:
            return FrameworkNeutralAgentLoop(
                sampler=sampler,  # type: ignore[arg-type]
                tool_runtime=self.tool_runtime,  # type: ignore[arg-type]
                appender=appender,
                parser=parser,
                behavior_recorder=VLLMBehaviorRecorder(self.behavior_store),
                max_tool_calls=self.config.protocol.maximum_tool_calls,
                enabled_tool_names=self.config.protocol.enabled_tool_names,
                cap_error_behavior=CapErrorBehavior.ONE_FINAL_ANSWER_TURN,
            )

        reward_context = _BoundRewardContextProvider(
            question=self.config.dataset.selected_sample.question,
            expected_answer=self.config.dataset.selected_sample.ground_truth,
            data_source=self.config.dataset.selected_sample.data_source,
        )
        scorer = PilotVerlTrajectoryRewardScorer(
            pipeline=self.reward_pipeline,
            context_provider=reward_context,
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


class _Qwen3BehaviorHiddenStateDependency(BehaviorHiddenStateDependency):
    """Run one deterministic behavior-policy forward over recorded visuals."""

    def __init__(
        self,
        *,
        model: nn.Module,
        family_adapter: Qwen3VLAdapter,
        layout_builder: Qwen3NativeToolLayoutBuilder,
        store: ObservationStore,
        device: torch.device,
        forward_identity: ArtifactIdentity,
        model_lock: RLock,
    ) -> None:
        self.model = model
        self.family_adapter = family_adapter
        self.layout_builder = layout_builder
        self.store = store
        self.device = device
        self.forward_identity = forward_identity
        self.model_lock = model_lock

    def capture_hidden_states(
        self, request: BehaviorHiddenStateCaptureRequest, /
    ) -> BehaviorHiddenStateMaterialization:
        if request.hidden_layer != -1:
            raise ValueError("Qwen3 live contextual capture currently requires layer -1")
        call = request.call
        handles = call.identity.prior_observation_handles
        expanded = self.layout_builder.expand_recorded_visual_sequence(
            call.conditioning_input_ids,
            trajectory_source_visual=call.trajectory_source_visual,
            observation_handles=handles,
        )
        expanded_ids = tuple(int(value) for value in expanded.input_ids[0].tolist())
        if expanded_ids != call.conditioning_input_ids:
            raise ReplayMismatchError(
                "contextual forward attempted to expand an already-expanded prefix"
            )
        blocks = _injected_visual_blocks(
            store=self.store,
            trajectory_id=call.identity.trajectory_id,
            source=call.trajectory_source_visual,
            observation_handles=handles,
            device=self.device,
        )
        forward_request = InjectedForwardRequest(
            input_ids=expanded.input_ids.to(self.device),
            attention_mask=expanded.attention_mask.to(self.device),
            position_ids=expanded.position_ids.to(self.device),
            visual_blocks=blocks,
            use_cache=False,
        )
        with self.model_lock, torch.no_grad():
            self.model.eval()
            result = self.family_adapter.forward_injected(self.model, forward_request)
        hidden = result.hidden_states
        if hidden.shape[0] != 1:
            raise ValueError("contextual behavior forward must contain one sequence")
        return BehaviorHiddenStateMaterialization(
            policy_version=call.identity.behavior_policy,
            forward_identity=self.forward_identity,
            hidden_layer=-1,
            hidden_states=hidden[0].detach(),
            deterministic_forward=True,
            policy_adapter_dropout=0.0,
        )


class _Qwen3FocusReplayLayoutDependency:
    def __init__(self, layout_builder: Qwen3NativeToolLayoutBuilder) -> None:
        self.layout_builder = layout_builder

    def build_replay_layout(
        self,
        request: object,
        source_visual: BoundSourceVisual,
        /,
    ) -> ReplayLayoutTensors:
        from tgvf_rl.environment.focus_runtime import FocusRuntimeCallRequest

        if not isinstance(request, FocusRuntimeCallRequest):
            raise TypeError("focus layout request has the wrong type")
        return self.layout_builder.build_focus_from_recorded_prefix(
            conditioning_input_ids=request.conditioning_input_ids,
            parsed_call=request.parsed_call,
            trajectory_source_visual=request.trajectory_source_visual,
            prior_observation_handles=request.identity.prior_observation_handles,
            source_visual=source_visual.tensors,
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
            crop_vision_replay_mode="no_crop",
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
    data_source: str

    def build(self, *, request: object, trajectory: TrajectoryRecord):
        if getattr(request, "identity", None) != trajectory.identity:
            raise IdentityMismatchError("reward context request/trajectory differ")
        return reward_context_from_trajectory(
            trajectory,
            question=self.question,
            expected_answer=self.expected_answer,
            task_kind=AnswerTaskKind.MULTIPLE_CHOICE,
            data_source=self.data_source,
        )


class _FocusedVisualTokenCountResolver:
    def __init__(self, store: ObservationStore) -> None:
        self.store = store

    def resolve_visual_token_count(self, observation: ObservationHandle) -> int:
        record = self.store.resolve_record(observation)
        if not isinstance(record, FocusedObservationRecord):
            raise TypeError("TGVF-only runtime requires a focused-D observation")
        return len(record.layout.d_positions)


def _load_local_qwen3_model(context: PolicyE2ERuntimeBuildContext) -> nn.Module:
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as error:  # pragma: no cover - accepted live env owns HF
        raise RuntimeError("Policy E2E live runtime requires transformers") from error
    torch.cuda.set_device(context.placement.logical_gpu_id)
    return AutoModelForImageTextToText.from_pretrained(
        context.config.model.revision_or_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )


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


def _injected_visual_blocks(
    *,
    store: ObservationStore,
    trajectory_id: str,
    source: TrajectorySourceVisual,
    observation_handles: tuple[ObservationHandle, ...],
    device: torch.device,
) -> tuple[InjectedVisualBlock, ...]:
    def resolve(ref: object, *, count: int, name: str) -> torch.Tensor:
        tensor = store.resolve_verified_for_trajectory(
            ref, trajectory_id=trajectory_id
        ).to(device=device)
        return _single_sequence_visual_features(tensor, count=count, name=name)

    blocks = [
        InjectedVisualBlock(
            kind="source_image",
            positions=source.positions,
            embeddings=resolve(
                source.state.merged_main,
                count=len(source.positions),
                name="source visual embeddings",
            ),
            deepstack=tuple(
                resolve(
                    ref,
                    count=len(positions),
                    name=f"source DeepStack branch {index}",
                )
                for index, (ref, positions) in enumerate(
                    zip(
                        source.state.merged_deepstack,
                        source.deepstack_injection_positions,
                        strict=True,
                    )
                )
            ),
            deepstack_positions=source.deepstack_injection_positions,
        )
    ]
    for expected_call, handle in enumerate(observation_handles):
        record = store.resolve_record(handle)
        if not isinstance(record, FocusedObservationRecord):
            raise TypeError("contextual TGVF-only forward requires focused-D records")
        if record.call_index != expected_call:
            raise ReplayMismatchError("contextual observations are out of order")
        blocks.append(
            InjectedVisualBlock(
                kind="focused_d",
                positions=record.layout.d_positions,
                embeddings=resolve(
                    record.payload.main_d,
                    count=len(record.layout.d_positions),
                    name=f"call {record.call_index} main D",
                ),
                deepstack=tuple(
                    resolve(
                        branch.d_tensor,
                        count=len(branch.injection_positions),
                        name=(
                            f"call {record.call_index} D-DeepStack branch {index}"
                        ),
                    )
                    for index, branch in enumerate(record.branches)
                ),
                deepstack_positions=record.layout.deepstack_injection_positions,
            )
        )
    return tuple(blocks)


def _single_sequence_visual_features(
    tensor: torch.Tensor, *, count: int, name: str
) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != count:
        raise ValueError(
            f"{name} must resolve to shape [1,{count},H], got {tuple(tensor.shape)}"
        )
    return tensor


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
        "multiple-choice-rule",
        "pilot-v1",
        reward.answer_verifier_sha256,
    )
    format_identity = _artifact_identity(
        "policy-reward", "native-format", "pilot-v1", {"run": config.identity_sha256}
    )
    tool_identity = _artifact_identity(
        "policy-reward",
        "conditional-tgvf-tool",
        "pilot-v1",
        {"run": config.identity_sha256, "tool": "tgvf_focus_tool"},
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
        },
    )
    disabled = _artifact_identity(
        "judge", "disabled-qwen2.5-72b", "pilot-smoke", {"disabled": True}
    )
    verifier = RuleFirstAnswerVerifier(
        rule_identity=answer_identity,
        normalization=NormalizationSpec(True, True, True),
        judge=DisabledJudgeProvider(),
        judge_prompt_identity=disabled,
        judge_model_identity=disabled,
        judge_service_identity=disabled,
        judge_sampling_identity=disabled,
        judge_calibration_identity=disabled,
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


def _validate_sample_fields(config: object, sample_id: str, fields: Mapping[str, object]) -> None:
    sample = config.dataset.selected_sample
    expected = {
        "sample_id": sample_id,
        "source_image_path": str(sample.image_path),
        "source_image_sha256": sample.image_sha256,
        "data_source": sample.data_source,
    }
    for key, value in expected.items():
        if key not in fields or _scalar(fields[key]) != value:
            raise IdentityMismatchError(f"upstream sample field {key!r} changed")
    reward_model = _scalar(fields.get("reward_model"))
    if not isinstance(reward_model, Mapping) or reward_model.get("ground_truth") != sample.ground_truth:
        raise IdentityMismatchError("upstream ground truth changed")


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


def _default_metrics_factory(
    trajectory: TrajectoryRecord, reward: PilotVerlTrajectoryReward
) -> object:
    del trajectory, reward
    try:
        from verl.experimental.agent_loop import AgentLoopMetrics
    except ImportError as error:  # pragma: no cover - accepted live env owns veRL
        raise RuntimeError("live AgentLoop metrics require the pinned veRL") from error
    return AgentLoopMetrics()


__all__ = [
    "QWEN3_POLICY_E2E_LIVE_RUNTIME_SCHEMA",
    "Qwen3PolicyE2ELiveRuntimeBuilder",
]
