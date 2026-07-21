"""Live composition boundary for one native ``tgvf_focus_tool`` call.

The runtime only wires already-selected model components together.  Source
visual tensors and replay layout are resolved through injected ports, while a
contextual provider additionally receives one exact behavior-forward capture.
It never renders a prompt, appends a tool response, or regenerates an existing
observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from threading import Condition
from typing import Callable, Protocol

import torch

from tgvf_rl.conditioning.base import (
    CONTEXTUAL_HIDDEN_STATE,
    TARGET_TOKEN_EMBEDDING,
    TargetConditionProvider,
    TargetConditioningOutput,
    TargetConditioningRequest,
    _bind_canonical_input_ids,
)
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import TokenSpan
from tgvf_rl.observations.schema import TrajectorySourceVisual
from tgvf_rl.observations.store import ObservationHandle, tensor_checksum
from tgvf_rl.protocol.schema import ParsedToolCall, TGVF_FOCUS_TOOL_NAME

from .agent_loop import ToolExecutionContext
from .focus_tool import (
    ReplayLayoutTensors,
    SourceVisualTensorBundle,
    TGVFFocusTool,
    ToolExecutionRequest,
    ToolExecutionResult,
)


@dataclass(frozen=True, slots=True)
class FocusRuntimeCallIdentity:
    """Identity shared by every injected artifact for one admitted call."""

    trajectory_id: str
    assistant_turn_index: int
    attempt_index: int
    call_index: int
    model: ModelIdentity
    behavior_policy: PolicyVersion
    contextual_forward_identity: ArtifactIdentity | None
    source_input_ids_sha256: str
    source_binding_sha256: str
    prior_observation_handles: tuple[ObservationHandle, ...]

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("focus runtime trajectory identity must be non-empty")
        if min(self.assistant_turn_index, self.attempt_index, self.call_index) < 0:
            raise ValueError("focus runtime call indices must be non-negative")
        if not isinstance(self.model, ModelIdentity):
            raise TypeError("focus runtime model identity must be a ModelIdentity")
        if not isinstance(self.behavior_policy, PolicyVersion):
            raise TypeError("focus runtime behavior policy must be a PolicyVersion")
        if self.contextual_forward_identity is not None and not isinstance(
            self.contextual_forward_identity, ArtifactIdentity
        ):
            raise TypeError(
                "focus runtime contextual forward identity must be an ArtifactIdentity"
            )
        if len(self.source_input_ids_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_input_ids_sha256
        ):
            raise ValueError("focus runtime source input identity must be a SHA256")
        if len(self.source_binding_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_binding_sha256
        ):
            raise ValueError("focus runtime source binding must be a SHA256")
        object.__setattr__(
            self, "prior_observation_handles", tuple(self.prior_observation_handles)
        )
        if any(
            not isinstance(handle, ObservationHandle)
            for handle in self.prior_observation_handles
        ):
            raise TypeError("focus runtime prior observations must be handles")
        if len(self.prior_observation_handles) != self.call_index:
            raise ValueError("focus runtime prior observations differ from call index")


@dataclass(frozen=True, slots=True)
class FocusRuntimeCallRequest:
    """Exact token and call identity supplied to source/layout ports."""

    identity: FocusRuntimeCallIdentity
    parsed_call: ParsedToolCall
    conditioning_input_ids: tuple[int, ...]
    target_span: TokenSpan
    trajectory_source_visual: TrajectorySourceVisual

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FocusRuntimeCallIdentity):
            raise TypeError("focus runtime request requires a call identity")
        if not isinstance(self.parsed_call, ParsedToolCall):
            raise TypeError("focus runtime request requires a parsed TGVF call")
        object.__setattr__(
            self, "conditioning_input_ids", tuple(self.conditioning_input_ids)
        )
        if not self.conditioning_input_ids or any(
            type(token_id) is not int or token_id < 0
            for token_id in self.conditioning_input_ids
        ):
            raise ValueError("conditioning input IDs must be non-negative integers")
        if not isinstance(self.target_span, TokenSpan):
            raise TypeError("focus runtime target span must be a TokenSpan")
        if self.target_span.end > len(self.conditioning_input_ids):
            raise ValueError("focus runtime target span lies outside exact input IDs")
        realized = self.conditioning_input_ids[
            self.target_span.start : self.target_span.end
        ]
        if realized != self.parsed_call.target_span.token_ids:
            raise ValueError("full-sequence target span differs from parsed target IDs")
        if not isinstance(self.trajectory_source_visual, TrajectorySourceVisual):
            raise TypeError("focus runtime request requires its source binding")
        if _source_binding_sha256(self.trajectory_source_visual) != (
            self.identity.source_binding_sha256
        ):
            raise ValueError("focus runtime source binding identity changed")


@dataclass(frozen=True, slots=True)
class BehaviorHiddenStateCaptureRequest:
    """One exact behavior forward requested at an explicit hidden layer."""

    call: FocusRuntimeCallRequest
    input_ids: torch.Tensor
    hidden_layer: int

    def __post_init__(self) -> None:
        if not isinstance(self.call, FocusRuntimeCallRequest):
            raise TypeError("hidden-state capture requires a focus call request")
        if not isinstance(self.input_ids, torch.Tensor):
            raise TypeError("hidden-state capture input_ids must be a tensor")
        if self.input_ids.ndim != 1 or self.input_ids.shape[0] != len(
            self.call.conditioning_input_ids
        ):
            raise ValueError("hidden-state capture IDs must match the exact sequence")
        if not isinstance(self.hidden_layer, int) or isinstance(
            self.hidden_layer, bool
        ):
            raise TypeError("hidden-state capture layer must be an explicit integer")


@dataclass(frozen=True, slots=True)
class BehaviorHiddenStateCapture:
    """Aligned full-sequence state returned by the behavior policy forward."""

    identity: FocusRuntimeCallIdentity
    input_ids: torch.Tensor
    hidden_layer: int
    forward_identity: ArtifactIdentity
    hidden_states: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FocusRuntimeCallIdentity):
            raise TypeError("hidden-state capture requires a call identity")
        if not isinstance(self.input_ids, torch.Tensor):
            raise TypeError("hidden-state capture input_ids must be a tensor")
        if not isinstance(self.hidden_layer, int) or isinstance(
            self.hidden_layer, bool
        ):
            raise TypeError("hidden-state capture layer must be an integer")
        if not isinstance(self.forward_identity, ArtifactIdentity):
            raise TypeError("hidden-state capture requires a forward identity")
        if not isinstance(self.hidden_states, torch.Tensor):
            raise TypeError("hidden-state capture must return a tensor")
        if self.hidden_states.ndim != 2 or self.hidden_states.shape[-1] <= 0:
            raise ValueError(
                "captured hidden states must have shape [sequence, hidden]"
            )
        if not self.hidden_states.is_floating_point():
            raise TypeError("captured hidden states must use a floating-point dtype")


class BehaviorHiddenStateCapturePort(Protocol):
    def capture(
        self, request: BehaviorHiddenStateCaptureRequest, /
    ) -> BehaviorHiddenStateCapture: ...


@dataclass(frozen=True, slots=True)
class BoundSourceVisual:
    identity: FocusRuntimeCallIdentity
    tensors: SourceVisualTensorBundle

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FocusRuntimeCallIdentity):
            raise TypeError("source visual requires a focus call identity")
        if not isinstance(self.tensors, SourceVisualTensorBundle):
            raise TypeError("source visual port must return SourceVisualTensorBundle")


class SourceVisualPort(Protocol):
    def resolve(self, request: FocusRuntimeCallRequest, /) -> BoundSourceVisual: ...


@dataclass(frozen=True, slots=True)
class BoundReplayLayout:
    identity: FocusRuntimeCallIdentity
    tensors: ReplayLayoutTensors

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FocusRuntimeCallIdentity):
            raise TypeError("replay layout requires a focus call identity")
        if not isinstance(self.tensors, ReplayLayoutTensors):
            raise TypeError("replay layout port must return ReplayLayoutTensors")


class ReplayLayoutPort(Protocol):
    def resolve(
        self,
        request: FocusRuntimeCallRequest,
        source_visual: BoundSourceVisual,
        /,
    ) -> BoundReplayLayout: ...


@dataclass(slots=True)
class _LedgerEntry:
    fingerprint: str
    handle: ObservationHandle | None = None
    running: bool = True


class FocusExecutionLedger:
    """Thread-safe execute-once ledger keyed by trajectory and call index."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._entries: dict[tuple[str, int], _LedgerEntry] = {}
        self._released_trajectory_ids: set[str] = set()

    def execute_once(
        self,
        *,
        key: tuple[str, int],
        fingerprint: str,
        operation: Callable[[], ObservationHandle],
    ) -> ObservationHandle:
        with self._condition:
            while True:
                if key[0] in self._released_trajectory_ids:
                    raise RuntimeError(
                        "focus execution batch has already been released"
                    )
                entry = self._entries.get(key)
                if entry is None:
                    self._entries[key] = _LedgerEntry(fingerprint)
                    break
                if entry.fingerprint != fingerprint:
                    raise ValueError("focus call key was reused with different content")
                if not entry.running:
                    assert entry.handle is not None
                    return entry.handle
                self._condition.wait()
        try:
            handle = operation()
            if not isinstance(handle, ObservationHandle):
                raise TypeError("focus execution must return an ObservationHandle")
        except BaseException:
            with self._condition:
                self._entries.pop(key, None)
                self._condition.notify_all()
            raise
        with self._condition:
            entry = self._entries[key]
            entry.handle = handle
            entry.running = False
            self._condition.notify_all()
        return handle

    def assert_releasable(self, trajectory_ids: tuple[str, ...]) -> None:
        identities = _trajectory_id_set(trajectory_ids)
        with self._condition:
            running = tuple(
                key
                for key, entry in self._entries.items()
                if key[0] in identities and entry.running
            )
            if running:
                raise RuntimeError(
                    "cannot release a batch while focus execution is active"
                )

    def release_trajectories(self, trajectory_ids: tuple[str, ...]) -> int:
        identities = _trajectory_id_set(trajectory_ids)
        with self._condition:
            self.assert_releasable(identities)
            keys = tuple(key for key in self._entries if key[0] in identities)
            for key in keys:
                del self._entries[key]
            self._released_trajectory_ids.update(identities)
            self._condition.notify_all()
            return len(keys)

    def entry_count(self) -> int:
        with self._condition:
            return len(self._entries)


class TGVFFocusToolRuntime:
    """Concrete ``ToolRuntimePort`` adapter for a live TGVF focus call."""

    def __init__(
        self,
        *,
        conditioning_provider: TargetConditionProvider,
        hidden_state_capture: BehaviorHiddenStateCapturePort,
        source_visual: SourceVisualPort,
        replay_layout: ReplayLayoutPort,
        focus_tool: TGVFFocusTool,
        representation: ArtifactIdentity,
        branch_merger_identities: tuple[ArtifactIdentity, ...],
        conditioning_input_device: torch.device,
        contextual_hidden_layer: int | None,
        contextual_forward_identity: ArtifactIdentity | None,
        execution_ledger: FocusExecutionLedger,
    ) -> None:
        if not isinstance(conditioning_provider, TargetConditionProvider):
            raise TypeError(
                "conditioning_provider must implement TargetConditionProvider"
            )
        provider_name = conditioning_provider.provider_name
        if provider_name not in {CONTEXTUAL_HIDDEN_STATE, TARGET_TOKEN_EMBEDDING}:
            raise ValueError(
                f"unsupported target-conditioning provider {provider_name!r}"
            )
        for name, value, method in (
            ("hidden_state_capture", hidden_state_capture, "capture"),
            ("source_visual", source_visual, "resolve"),
            ("replay_layout", replay_layout, "resolve"),
        ):
            if not callable(getattr(value, method, None)):
                raise TypeError(f"{name} must implement {method}()")
        if not isinstance(focus_tool, TGVFFocusTool):
            raise TypeError("focus_tool must be a TGVFFocusTool")
        if not isinstance(representation, ArtifactIdentity):
            raise TypeError("representation must be an ArtifactIdentity")
        branch_identities = tuple(branch_merger_identities)
        if any(not isinstance(value, ArtifactIdentity) for value in branch_identities):
            raise TypeError("branch merger identities must be ArtifactIdentity values")
        if not isinstance(conditioning_input_device, torch.device):
            raise TypeError(
                "conditioning_input_device must be an explicit torch.device"
            )
        if not isinstance(execution_ledger, FocusExecutionLedger):
            raise TypeError("execution_ledger must be a FocusExecutionLedger")
        if provider_name == CONTEXTUAL_HIDDEN_STATE:
            if not isinstance(contextual_hidden_layer, int) or isinstance(
                contextual_hidden_layer, bool
            ):
                raise ValueError(
                    "contextual_hidden_state requires an explicit hidden layer"
                )
            if not isinstance(contextual_forward_identity, ArtifactIdentity):
                raise ValueError(
                    "contextual_hidden_state requires an explicit forward identity"
                )
        elif (
            contextual_hidden_layer is not None
            or contextual_forward_identity is not None
        ):
            raise ValueError(
                "target_token_embedding cannot configure contextual forward state"
            )

        self.conditioning_provider = conditioning_provider
        self.hidden_state_capture = hidden_state_capture
        self.source_visual = source_visual
        self.replay_layout = replay_layout
        self.focus_tool = focus_tool
        self.representation = representation
        self.branch_merger_identities = branch_identities
        self.conditioning_input_device = conditioning_input_device
        self.contextual_hidden_layer = contextual_hidden_layer
        self.contextual_forward_identity = contextual_forward_identity
        self.execution_ledger = execution_ledger

    def execute(
        self, parsed_call: object, context: ToolExecutionContext
    ) -> ObservationHandle:
        if not isinstance(parsed_call, ParsedToolCall):
            raise TypeError("focus runtime requires a ParsedToolCall")
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("focus runtime context must be ToolExecutionContext")
        if parsed_call.name != TGVF_FOCUS_TOOL_NAME:
            raise ValueError("focus runtime received a non-TGVF tool call")
        self._validate_sampled_turn(parsed_call, context)
        if context.model != self.conditioning_provider.model_identity:
            raise ValueError("runtime model differs from target-conditioning provider")

        fingerprint = _call_fingerprint(
            parsed_call=parsed_call,
            context=context,
            provider_name=self.conditioning_provider.provider_name,
            hidden_layer=self.contextual_hidden_layer,
            contextual_forward_identity=self.contextual_forward_identity,
            representation=self.representation,
            branch_mergers=self.branch_merger_identities,
        )
        return self.execution_ledger.execute_once(
            key=(context.trajectory_identity.canonical_id, context.call_index),
            fingerprint=fingerprint,
            operation=lambda: self._execute_once(parsed_call, context),
        )

    def _execute_once(
        self, parsed_call: ParsedToolCall, context: ToolExecutionContext
    ) -> ObservationHandle:

        canonical_input_ids = context.conditioning_input_ids
        prefix_length = len(context.prompt_token_ids_before_turn)
        target_span = TokenSpan(
            prefix_length + parsed_call.target_span.token_start,
            prefix_length + parsed_call.target_span.token_end,
        )
        input_ids = torch.tensor(
            canonical_input_ids,
            dtype=torch.long,
            device=self.conditioning_input_device,
        )
        proof = _bind_canonical_input_ids(input_ids, canonical_input_ids)
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
        call_request = FocusRuntimeCallRequest(
            identity=identity,
            parsed_call=parsed_call,
            conditioning_input_ids=canonical_input_ids,
            target_span=target_span,
            trajectory_source_visual=context.trajectory_source_visual,
        )

        contextual_capture = self._capture_contextual_states(call_request, input_ids)
        condition = self.conditioning_provider.build(
            TargetConditioningRequest(
                input_ids=input_ids,
                target_span=target_span,
                expected_target_token_ids=parsed_call.target_span.token_ids,
                trajectory_id=identity.trajectory_id,
                call_index=identity.call_index,
                model_identity=identity.model,
                contextual_hidden_states=(
                    None
                    if contextual_capture is None
                    else contextual_capture.hidden_states
                ),
                canonical_input_ids_proof=proof,
            )
        )
        self._validate_condition(condition, call_request)

        source = self.source_visual.resolve(call_request)
        self._validate_bound_identity(source, identity, name="source visual")
        _validate_source_visual_binding(
            source.tensors, call_request.trajectory_source_visual
        )
        layout = self.replay_layout.resolve(call_request, source)
        self._validate_bound_identity(layout, identity, name="replay layout")

        result = self.focus_tool.execute(
            ToolExecutionRequest(
                trajectory_id=identity.trajectory_id,
                call_index=identity.call_index,
                parsed_call=parsed_call,
                condition=condition,
                source_visual=source.tensors,
                layout=layout.tensors,
                model=identity.model,
                policy_version=identity.behavior_policy,
                contextual_forward_identity=(
                    None
                    if contextual_capture is None
                    else contextual_capture.forward_identity
                ),
                representation=self.representation,
                branch_merger_identities=self.branch_merger_identities,
            )
        )
        if not isinstance(result, ToolExecutionResult):
            raise TypeError("TGVFFocusTool must return ToolExecutionResult")
        if not isinstance(result.handle, ObservationHandle):
            raise TypeError("TGVFFocusTool result must contain an ObservationHandle")
        return result.handle

    @staticmethod
    def _validate_sampled_turn(
        parsed_call: ParsedToolCall, context: ToolExecutionContext
    ) -> None:
        sampled = context.sampled_turn
        if (
            parsed_call.sampled_text != sampled.text
            or parsed_call.sampled_token_ids != sampled.token_ids
            or parsed_call.sampled_token_byte_spans != sampled.token_byte_spans
        ):
            raise ValueError(
                "parsed call differs from the exact sampled assistant turn"
            )

    def _capture_contextual_states(
        self,
        request: FocusRuntimeCallRequest,
        input_ids: torch.Tensor,
    ) -> BehaviorHiddenStateCapture | None:
        if self.conditioning_provider.provider_name == TARGET_TOKEN_EMBEDDING:
            return None
        hidden_layer = self.contextual_hidden_layer
        if hidden_layer is None:
            raise RuntimeError("contextual runtime lost its explicit hidden layer")
        capture = self.hidden_state_capture.capture(
            BehaviorHiddenStateCaptureRequest(request, input_ids, hidden_layer)
        )
        if not isinstance(capture, BehaviorHiddenStateCapture):
            raise TypeError(
                "hidden-state capture port must return BehaviorHiddenStateCapture"
            )
        if capture.identity != request.identity:
            raise ValueError("hidden-state capture identity differs from focus call")
        if capture.input_ids is not input_ids:
            raise ValueError("hidden-state capture replaced the exact input-ID tensor")
        if capture.hidden_layer != hidden_layer:
            raise ValueError("hidden-state capture returned a different layer")
        if capture.forward_identity != self.contextual_forward_identity:
            raise ValueError(
                "hidden-state capture returned a different forward identity"
            )
        if capture.hidden_states.shape[0] != input_ids.shape[0]:
            raise ValueError("captured hidden states do not align with exact input IDs")
        if capture.hidden_states.device != input_ids.device:
            raise ValueError("captured hidden states and input IDs must share a device")
        return capture

    def _validate_condition(
        self,
        condition: TargetConditioningOutput,
        request: FocusRuntimeCallRequest,
    ) -> None:
        if not isinstance(condition, TargetConditioningOutput):
            raise TypeError(
                "conditioning provider must return TargetConditioningOutput"
            )
        provenance = condition.provenance
        identity = request.identity
        if provenance.model != identity.model:
            raise ValueError("conditioning provenance model differs from focus call")
        if provenance.provider != self.conditioning_provider.provider_name:
            raise ValueError("conditioning provenance provider differs from binding")
        if provenance.target_span != request.target_span:
            raise ValueError(
                "conditioning provenance target span differs from focus call"
            )
        if provenance.target_token_ids != (request.parsed_call.target_span.token_ids,):
            raise ValueError(
                "conditioning provenance target tokens differ from focus call"
            )
        if provenance.trajectory_ids != (identity.trajectory_id,):
            raise ValueError(
                "conditioning provenance trajectory differs from focus call"
            )
        if provenance.call_indices != (identity.call_index,):
            raise ValueError(
                "conditioning provenance call index differs from focus call"
            )
        if provenance.source_input_ids_sha256 != identity.source_input_ids_sha256:
            raise ValueError("conditioning provenance input IDs differ from focus call")
        if provenance.provider == CONTEXTUAL_HIDDEN_STATE:
            if provenance.hidden_layer != self.contextual_hidden_layer:
                raise ValueError(
                    "conditioning provenance hidden layer differs from binding"
                )
        elif provenance.hidden_layer is not None:
            raise ValueError("token embedding conditioning named a hidden layer")

    @staticmethod
    def _validate_bound_identity(
        value: BoundSourceVisual | BoundReplayLayout,
        expected: FocusRuntimeCallIdentity,
        *,
        name: str,
    ) -> None:
        expected_type = (
            BoundSourceVisual if name == "source visual" else BoundReplayLayout
        )
        if not isinstance(value, expected_type):
            raise TypeError(f"{name} port returned an invalid bound artifact")
        if value.identity != expected:
            raise ValueError(f"{name} identity differs from focus call")


def _trajectory_id_set(trajectory_ids: tuple[str, ...]) -> tuple[str, ...]:
    identities = tuple(trajectory_ids)
    if not identities or any(
        not isinstance(identity, str) or not identity for identity in identities
    ):
        raise ValueError("trajectory_ids must contain non-empty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("trajectory_ids must be unique")
    return identities


__all__ = [
    "BehaviorHiddenStateCapture",
    "BehaviorHiddenStateCapturePort",
    "BehaviorHiddenStateCaptureRequest",
    "BoundReplayLayout",
    "BoundSourceVisual",
    "FocusRuntimeCallIdentity",
    "FocusRuntimeCallRequest",
    "FocusExecutionLedger",
    "ReplayLayoutPort",
    "SourceVisualPort",
    "TGVFFocusToolRuntime",
]


def _source_binding_sha256(source: TrajectorySourceVisual) -> str:
    payload = json.dumps(asdict(source), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _call_fingerprint(
    *,
    parsed_call: ParsedToolCall,
    context: ToolExecutionContext,
    provider_name: str,
    hidden_layer: int | None,
    contextual_forward_identity: ArtifactIdentity | None,
    representation: ArtifactIdentity,
    branch_mergers: tuple[ArtifactIdentity, ...],
) -> str:
    conditioning_target_start = (
        len(context.prompt_token_ids_before_turn) + parsed_call.target_span.token_start
    )
    conditioning_target_end = (
        len(context.prompt_token_ids_before_turn) + parsed_call.target_span.token_end
    )
    payload = {
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
        "target_span": (
            parsed_call.target_span.token_start,
            parsed_call.target_span.token_end,
        ),
        "target_span_identity": asdict(parsed_call.target_span),
        "conditioning_target_span": (
            conditioning_target_start,
            conditioning_target_end,
        ),
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
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_visual_binding(
    tensors: SourceVisualTensorBundle,
    binding: TrajectorySourceVisual,
) -> None:
    state = binding.state
    if tensors.image_sha256 != state.image_sha256:
        raise ValueError("resolved source image differs from immutable binding")
    if tensors.decoded_rgb_sha256 != state.decoded_rgb_sha256:
        raise ValueError("resolved decoded RGB identity differs from immutable binding")
    tensor_rows = (
        (tensors.premerge_main, state.premerge_main),
        *zip(tensors.premerge_deepstack, state.premerge_deepstack, strict=False),
        (tensors.merged_main, state.merged_main),
        *zip(tensors.merged_deepstack, state.merged_deepstack, strict=False),
    )
    if len(tensors.premerge_deepstack) != len(state.premerge_deepstack) or len(
        tensors.merged_deepstack
    ) != len(state.merged_deepstack):
        raise ValueError("resolved source branches differ from immutable binding")
    if any(
        tensor_checksum(tensor) != ref.address.digest for tensor, ref in tensor_rows
    ):
        raise ValueError("resolved source tensors differ from immutable binding")
    if (
        tensors.image_grid_thw != state.image_grid_thw
        or tensors.spatial_merge_size != state.spatial_merge_size
    ):
        raise ValueError("resolved source geometry differs from immutable binding")
