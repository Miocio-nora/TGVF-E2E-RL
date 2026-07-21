"""ToolRuntimePort-compatible live runtime for atomic ``tgvf_crop_tool``."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import torch

from tgvf_rl.conditioning.base import (
    CONTEXTUAL_HIDDEN_STATE,
    TARGET_TOKEN_EMBEDDING,
    TargetConditionProvider,
    TargetConditioningConfig,
    TargetConditioningOutput,
    TargetConditioningRequest,
    _bind_canonical_input_ids,
)
from tgvf_rl.conditioning.providers import (
    ContextualHiddenStateConditionProvider,
    TargetTokenEmbeddingConditionProvider,
)
from tgvf_rl.contracts.errors import (
    IdentityMismatchError,
    RecoverableToolExecutionError,
)
from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.contracts.tokens import TokenSpan
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol.schema import (
    TGVF_CROP_TOOL_NAME,
    TGVF_FOCUS_TOOL_NAME,
    ParsedCropTGVFCall,
    ParsedToolCall,
)

from .agent_loop import ToolExecutionContext
from .adapter_runtime import (
    BranchMergerRuntimeBinding,
    LoadedFrozenTGVFAdapter,
)
from .crop_tgvf_tool import (
    AtomicCropTGVFTool,
    CropTGVFToolExecutionRequest,
    CropTGVFToolExecutionResult,
)
from .focus_runtime import (
    BehaviorHiddenStateCapture,
    BehaviorHiddenStateCapturePort,
    BehaviorHiddenStateCaptureRequest,
    FocusExecutionLedger,
    FocusRuntimeCallIdentity,
    FocusRuntimeCallRequest,
    _source_binding_sha256,
)
from .qwen3_tool_layout import Qwen3NativeToolLayoutBuilder


class AtomicCropTGVFToolRuntime:
    """Build target conditioning and execute one indivisible crop+TGVF call."""

    def __init__(
        self,
        *,
        conditioning_provider: TargetConditionProvider,
        hidden_state_capture: BehaviorHiddenStateCapturePort | None,
        atomic_tool: AtomicCropTGVFTool,
        layout_builder: Qwen3NativeToolLayoutBuilder,
        loaded_adapter: LoadedFrozenTGVFAdapter,
        branch_mergers: tuple[BranchMergerRuntimeBinding, ...],
        crop_processor_identity: ArtifactIdentity,
        crop_layout_identity: ArtifactIdentity,
        conditioning_input_device: torch.device,
        contextual_forward_identity: ArtifactIdentity | None,
        execution_ledger: FocusExecutionLedger,
    ) -> None:
        if not isinstance(conditioning_provider, TargetConditionProvider):
            raise TypeError("conditioning_provider must implement its typed protocol")
        provider = conditioning_provider.provider_name
        if provider not in {CONTEXTUAL_HIDDEN_STATE, TARGET_TOKEN_EMBEDDING}:
            raise ValueError("unsupported atomic target-conditioning provider")
        if not isinstance(atomic_tool, AtomicCropTGVFTool):
            raise TypeError("atomic_tool must be AtomicCropTGVFTool")
        if not isinstance(layout_builder, Qwen3NativeToolLayoutBuilder):
            raise TypeError("atomic runtime requires Qwen3NativeToolLayoutBuilder")
        if not isinstance(loaded_adapter, LoadedFrozenTGVFAdapter):
            raise TypeError("atomic runtime requires a loaded representation Adapter")
        merger_bindings = tuple(branch_mergers)
        if any(
            not isinstance(value, BranchMergerRuntimeBinding)
            for value in merger_bindings
        ):
            raise TypeError(
                "atomic runtime branch_mergers require typed runtime bindings"
            )
        identities = (
            crop_processor_identity,
            crop_layout_identity,
            *(value.artifact for value in merger_bindings),
        )
        if any(not isinstance(value, ArtifactIdentity) for value in identities):
            raise TypeError("atomic runtime artifact identities must be explicit")
        if not isinstance(conditioning_input_device, torch.device):
            raise TypeError("conditioning_input_device must be explicit")
        if not isinstance(execution_ledger, FocusExecutionLedger):
            raise TypeError("execution_ledger must be FocusExecutionLedger")
        conditioning = loaded_adapter.binding.conditioning
        _assert_provider_matches_artifact(conditioning_provider, conditioning)
        contextual_hidden_layer = conditioning.hidden_layer
        if provider == CONTEXTUAL_HIDDEN_STATE:
            if not callable(getattr(hidden_state_capture, "capture", None)):
                raise TypeError("contextual atomic runtime requires hidden capture")
            if type(contextual_hidden_layer) is not int:
                raise ValueError("contextual atomic runtime requires a hidden layer")
            if not isinstance(contextual_forward_identity, ArtifactIdentity):
                raise ValueError("contextual atomic runtime requires forward identity")
        elif (
            hidden_state_capture is not None
            or contextual_hidden_layer is not None
            or contextual_forward_identity is not None
        ):
            raise ValueError("embedding atomic runtime cannot name contextual state")

        self.conditioning_provider = conditioning_provider
        self.hidden_state_capture = hidden_state_capture
        self.atomic_tool = atomic_tool
        self.layout_builder = layout_builder
        self.loaded_adapter = loaded_adapter
        self.branch_mergers = merger_bindings
        self.branch_merger_identities = tuple(
            value.artifact for value in merger_bindings
        )
        self.crop_processor_identity = crop_processor_identity
        self.crop_layout_identity = crop_layout_identity
        self.conditioning_input_device = conditioning_input_device
        self.contextual_hidden_layer = contextual_hidden_layer
        self.contextual_forward_identity = contextual_forward_identity
        self.execution_ledger = execution_ledger
        self._assert_representation_binding()

    def execute(
        self,
        parsed_call: object,
        context: ToolExecutionContext,
    ) -> ObservationHandle:
        if not isinstance(parsed_call, ParsedCropTGVFCall):
            raise TypeError("atomic runtime requires ParsedCropTGVFCall")
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("atomic runtime requires ToolExecutionContext")
        if parsed_call.name != TGVF_CROP_TOOL_NAME:
            raise ValueError("atomic runtime received another tool")
        self._assert_representation_binding()
        _validate_sampled_turn(parsed_call, context)
        if context.model != self.conditioning_provider.model_identity:
            raise ValueError("atomic runtime model differs from conditioning provider")
        if context.model != self.layout_builder.model_identity:
            raise ValueError("atomic runtime model differs from layout builder")
        fingerprint = _call_fingerprint(
            parsed_call=parsed_call,
            context=context,
            provider_name=self.conditioning_provider.provider_name,
            hidden_layer=self.contextual_hidden_layer,
            contextual_forward_identity=self.contextual_forward_identity,
            representation=self.loaded_adapter.binding.artifact,
            branch_mergers=self.branch_merger_identities,
            crop_processor_identity=self.crop_processor_identity,
            crop_layout_identity=self.crop_layout_identity,
        )
        return self.execution_ledger.execute_once(
            key=(context.trajectory_identity.canonical_id, context.call_index),
            fingerprint=fingerprint,
            operation=lambda: self._execute_once(parsed_call, context),
        )

    def _execute_once(
        self,
        parsed_call: ParsedCropTGVFCall,
        context: ToolExecutionContext,
    ) -> ObservationHandle:
        canonical_ids = context.conditioning_input_ids
        prefix_length = len(context.prompt_token_ids_before_turn)
        target_span = TokenSpan(
            prefix_length + parsed_call.target_span.token_start,
            prefix_length + parsed_call.target_span.token_end,
        )
        input_ids = torch.tensor(
            canonical_ids,
            dtype=torch.long,
            device=self.conditioning_input_device,
        )
        proof = _bind_canonical_input_ids(input_ids, canonical_ids)
        identity = FocusRuntimeCallIdentity(
            trajectory_id=context.trajectory_identity.canonical_id,
            assistant_turn_index=context.assistant_turn_index,
            attempt_index=context.attempt_index,
            call_index=context.call_index,
            model=context.model,
            behavior_policy=context.behavior_policy,
            contextual_forward_identity=self.contextual_forward_identity,
            source_input_ids_sha256=proof.digest,
            source_binding_sha256=_source_binding_sha256(
                context.trajectory_source_visual
            ),
            prior_observation_handles=context.prior_observation_handles,
        )
        focus_proxy = _focus_proxy(parsed_call)
        call_request = FocusRuntimeCallRequest(
            identity=identity,
            parsed_call=focus_proxy,
            conditioning_input_ids=canonical_ids,
            target_span=target_span,
            trajectory_source_visual=context.trajectory_source_visual,
        )
        capture = self._capture_contextual(call_request, input_ids)
        condition = self.conditioning_provider.build(
            TargetConditioningRequest(
                input_ids=input_ids,
                target_span=target_span,
                expected_target_token_ids=parsed_call.target_span.token_ids,
                trajectory_id=identity.trajectory_id,
                call_index=identity.call_index,
                model_identity=identity.model,
                contextual_hidden_states=(
                    None if capture is None else capture.hidden_states
                ),
                canonical_input_ids_proof=proof,
            )
        )
        self._validate_condition(condition, call_request)
        try:
            result = self.atomic_tool.execute(
                CropTGVFToolExecutionRequest(
                    trajectory_id=identity.trajectory_id,
                    call_index=identity.call_index,
                    parsed_call=parsed_call,
                    condition=condition,
                    trajectory_source_visual=context.trajectory_source_visual,
                    layout_builder=self.layout_builder.bind_crop_tgvf(context),
                    model=identity.model,
                    policy_version=identity.behavior_policy,
                    contextual_forward_identity=(
                        None if capture is None else capture.forward_identity
                    ),
                    representation=self.loaded_adapter.binding.artifact,
                    branch_merger_identities=self.branch_merger_identities,
                    crop_processor_identity=self.crop_processor_identity,
                    crop_layout_identity=self.crop_layout_identity,
                )
            )
        except ValueError as error:
            if "bbox is empty after clamping" in str(error):
                raise RecoverableToolExecutionError(str(error)) from error
            raise
        if not isinstance(result, CropTGVFToolExecutionResult):
            raise TypeError("atomic tool returned an invalid result")
        return result.handle

    def _assert_representation_binding(self) -> None:
        loaded = self.loaded_adapter
        loaded.assert_bound_invariants()
        binding = loaded.binding
        config = binding.conditioning
        provider = self.conditioning_provider
        if self.atomic_tool.adapter is not loaded.adapter:
            raise IdentityMismatchError(
                "atomic tool Adapter differs from the loaded representation artifact"
            )
        if provider.model_identity != binding.model:
            raise IdentityMismatchError(
                "atomic conditioning model differs from representation artifact"
            )
        if self.layout_builder.model_identity != binding.model:
            raise IdentityMismatchError(
                "atomic layout model differs from representation artifact"
            )
        _assert_provider_matches_artifact(provider, config)
        projection_identities = tuple(
            value.projection_identity for value in self.branch_mergers
        )
        if projection_identities != (
            binding.adapter_contract.deepstack_projection_identities
        ):
            raise IdentityMismatchError(
                "atomic branch merger bindings differ from representation architecture"
            )

    def _capture_contextual(
        self,
        request: FocusRuntimeCallRequest,
        input_ids: torch.Tensor,
    ) -> BehaviorHiddenStateCapture | None:
        if self.conditioning_provider.provider_name == TARGET_TOKEN_EMBEDDING:
            return None
        assert self.hidden_state_capture is not None
        assert self.contextual_hidden_layer is not None
        capture = self.hidden_state_capture.capture(
            BehaviorHiddenStateCaptureRequest(
                request,
                input_ids,
                self.contextual_hidden_layer,
            )
        )
        if not isinstance(capture, BehaviorHiddenStateCapture):
            raise TypeError("atomic hidden capture returned an invalid result")
        if capture.identity != request.identity or capture.input_ids is not input_ids:
            raise ValueError("atomic hidden capture changed call/input identity")
        if capture.hidden_layer != self.contextual_hidden_layer or (
            capture.forward_identity != self.contextual_forward_identity
        ):
            raise ValueError("atomic hidden capture changed forward identity")
        if capture.hidden_states.shape[0] != input_ids.shape[0] or (
            capture.hidden_states.device != input_ids.device
        ):
            raise ValueError("atomic hidden states do not align with input IDs")
        return capture

    def _validate_condition(
        self,
        condition: TargetConditioningOutput,
        request: FocusRuntimeCallRequest,
    ) -> None:
        if not isinstance(condition, TargetConditioningOutput):
            raise TypeError("atomic conditioning provider returned an invalid output")
        provenance = condition.provenance
        identity = request.identity
        expected = (
            provenance.model == identity.model
            and provenance.provider == self.conditioning_provider.provider_name
            and provenance.target_span == request.target_span
            and provenance.target_token_ids
            == (request.parsed_call.target_span.token_ids,)
            and provenance.trajectory_ids == (identity.trajectory_id,)
            and provenance.call_indices == (identity.call_index,)
            and provenance.source_input_ids_sha256
            == identity.source_input_ids_sha256
        )
        if not expected:
            raise ValueError("atomic conditioning provenance differs from sampled call")
        if provenance.provider == CONTEXTUAL_HIDDEN_STATE:
            if provenance.hidden_layer != self.contextual_hidden_layer:
                raise ValueError("atomic contextual hidden layer changed")
        elif provenance.hidden_layer is not None:
            raise ValueError("atomic embedding conditioning named a hidden layer")


def _assert_provider_matches_artifact(
    provider: TargetConditionProvider,
    config: TargetConditioningConfig,
) -> None:
    if not isinstance(config, TargetConditioningConfig):
        raise TypeError("representation artifact lost its conditioning contract")
    if config.provider.value != provider.provider_name:
        raise IdentityMismatchError(
            "atomic conditioning provider differs from representation artifact"
        )
    if config.provider.value == CONTEXTUAL_HIDDEN_STATE:
        if not isinstance(provider, ContextualHiddenStateConditionProvider) or (
            provider.hidden_layer != config.hidden_layer
        ):
            raise IdentityMismatchError(
                "atomic contextual provider contract differs from representation artifact"
            )
    elif not isinstance(provider, TargetTokenEmbeddingConditionProvider) or (
        provider.embedding_identity != config.embedding_identity
    ):
        raise IdentityMismatchError(
            "atomic embedding provider contract differs from representation artifact"
        )


def _focus_proxy(parsed: ParsedCropTGVFCall) -> ParsedToolCall:
    return ParsedToolCall(
        name=TGVF_FOCUS_TOOL_NAME,
        target=parsed.target,
        sampled_text=parsed.sampled_text,
        sampled_token_ids=parsed.sampled_token_ids,
        sampled_token_byte_spans=parsed.sampled_token_byte_spans,
        raw_tool_call=parsed.raw_tool_call,
        raw_json=parsed.raw_json,
        call_offsets=parsed.call_offsets,
        json_offsets=parsed.json_offsets,
        target_span=parsed.target_span,
    )


def _validate_sampled_turn(
    parsed: ParsedCropTGVFCall,
    context: ToolExecutionContext,
) -> None:
    sampled = context.sampled_turn
    if (
        parsed.sampled_text != sampled.text
        or parsed.sampled_token_ids != sampled.token_ids
        or parsed.sampled_token_byte_spans != sampled.token_byte_spans
    ):
        raise ValueError("atomic parsed call differs from sampled assistant turn")


def _call_fingerprint(
    *,
    parsed_call: ParsedCropTGVFCall,
    context: ToolExecutionContext,
    provider_name: str,
    hidden_layer: int | None,
    contextual_forward_identity: ArtifactIdentity | None,
    representation: ArtifactIdentity,
    branch_mergers: tuple[ArtifactIdentity, ...],
    crop_processor_identity: ArtifactIdentity,
    crop_layout_identity: ArtifactIdentity,
) -> str:
    payload = {
        "schema": "atomic-crop-tgvf-runtime-call-v1",
        "trajectory_id": context.trajectory_identity.canonical_id,
        "assistant_turn_index": context.assistant_turn_index,
        "attempt_index": context.attempt_index,
        "call_index": context.call_index,
        "model": asdict(context.model),
        "behavior_policy": asdict(context.behavior_policy),
        "prompt_token_ids_before_turn": context.prompt_token_ids_before_turn,
        "conditioning_input_ids": context.conditioning_input_ids,
        "sampled_text": parsed_call.sampled_text,
        "sampled_token_ids": parsed_call.sampled_token_ids,
        "sampled_token_byte_spans": tuple(
            asdict(span) for span in parsed_call.sampled_token_byte_spans
        ),
        "raw_tool_call": parsed_call.raw_tool_call,
        "raw_json": parsed_call.raw_json,
        "target_span": asdict(parsed_call.target_span),
        "bbox_2d": parsed_call.bbox_2d,
        "source_binding_sha256": _source_binding_sha256(
            context.trajectory_source_visual
        ),
        "prior_observation_handles": tuple(
            asdict(handle) for handle in context.prior_observation_handles
        ),
        "provider": provider_name,
        "hidden_layer": hidden_layer,
        "contextual_forward_identity": (
            None
            if contextual_forward_identity is None
            else asdict(contextual_forward_identity)
        ),
        "representation": asdict(representation),
        "branch_mergers": tuple(asdict(value) for value in branch_mergers),
        "crop_processor_identity": asdict(crop_processor_identity),
        "crop_layout_identity": asdict(crop_layout_identity),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = ["AtomicCropTGVFToolRuntime"]
