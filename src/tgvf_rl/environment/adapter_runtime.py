"""Contract-scaffold injection boundary for a frozen TGVF Adapter.

This module validates caller-supplied representation and replay contracts; it
does not assemble a family adapter against a live Qwen model or prove numerical
parity.  In particular, borrowed merger modules, target-token embedding origin,
and source-visual origin still need a family adapter to bind them to the selected
live model.  Consequently this scaffold does not close RO-S03 or F01.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningDependencies,
    TargetConditioningProviderKind,
    create_target_condition_provider,
)
from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tensors import TensorArtifactRef
from tgvf_rl.observations.store import ObservationStore
from tgvf_rl.representation import TGVFAdapter
from tgvf_rl.representation.training.checkpoint import (
    RepresentationAdapterContractIdentity,
    RepresentationRunIdentity,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    RankZeroAdapterOwnedStateManifest,
    load_rank_zero_adapter_owned_state_export,
)

from .focus_runtime import (
    BehaviorHiddenStateCapture,
    BehaviorHiddenStateCaptureRequest,
    BoundReplayLayout,
    BoundSourceVisual,
    FocusExecutionLedger,
    FocusRuntimeCallRequest,
    TGVFFocusToolRuntime,
)
from .focus_tool import (
    ReplayLayoutTensors,
    SourceVisualTensorBundle,
    TGVFFocusTool,
    ToolExecutionRequest,
    ToolExecutionResult,
)


@dataclass(frozen=True, slots=True)
class RepresentationArtifactRuntimeBinding:
    """Caller-declared identities used to select one representation export.

    ``artifact.sha256`` is the canonical digest of the rank-zero export
    manifest, not an unchecked filename or a legacy checkpoint label.
    """

    artifact_path: Path
    artifact: ArtifactIdentity
    expected_run_id: str
    expected_run_identity_sha256: str
    model: ModelIdentity
    conditioning: TargetConditioningConfig
    adapter_contract: RepresentationAdapterContractIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_path, Path):
            raise TypeError("representation artifact_path must be an explicit Path")
        if not isinstance(self.artifact, ArtifactIdentity):
            raise TypeError("representation artifact must be an ArtifactIdentity")
        if (
            not isinstance(self.expected_run_id, str)
            or not self.expected_run_id.strip()
        ):
            raise ValueError("representation expected_run_id must be non-empty")
        _require_sha256(
            self.expected_run_identity_sha256,
            name="expected_run_identity_sha256",
        )
        if not isinstance(self.model, ModelIdentity):
            raise TypeError("representation model must be a ModelIdentity")
        if not isinstance(self.conditioning, TargetConditioningConfig):
            raise TypeError(
                "representation conditioning must be a TargetConditioningConfig"
            )
        if not isinstance(self.adapter_contract, RepresentationAdapterContractIdentity):
            raise TypeError(
                "representation adapter_contract must be explicit architecture identity"
            )


@dataclass(frozen=True, slots=True)
class BranchMergerRuntimeBinding:
    """Record a caller-declared projection/artifact association.

    The scaffold does not inspect a borrowed merger module.  A family adapter
    must still prove that the association came from the selected live model.
    """

    projection_identity: str
    artifact: ArtifactIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.projection_identity, str) or not (
            self.projection_identity.strip()
        ):
            raise ValueError("branch projection_identity must be non-empty")
        if not isinstance(self.artifact, ArtifactIdentity):
            raise TypeError("branch merger artifact must be an ArtifactIdentity")


@dataclass(frozen=True, slots=True)
class LoadedFrozenTGVFAdapter:
    """A state-loaded Adapter and the manifest checked by this scaffold."""

    adapter: TGVFAdapter
    binding: RepresentationArtifactRuntimeBinding
    manifest: RankZeroAdapterOwnedStateManifest
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, TGVFAdapter):
            raise TypeError("loaded Adapter must be a TGVFAdapter")
        if not isinstance(self.binding, RepresentationArtifactRuntimeBinding):
            raise TypeError("loaded Adapter requires its runtime binding")
        if not isinstance(self.manifest, RankZeroAdapterOwnedStateManifest):
            raise TypeError("loaded Adapter requires a rank-zero export manifest")
        _require_sha256(self.manifest_sha256, name="manifest_sha256")
        if self.manifest_sha256 != self.binding.artifact.sha256:
            raise IdentityMismatchError(
                "loaded Adapter manifest differs from selected artifact identity"
            )
        _assert_frozen_eval_adapter(self.adapter)

    @property
    def run_identity(self) -> RepresentationRunIdentity:
        return self.manifest.run_identity


def load_frozen_tgvf_adapter(
    *,
    binding: RepresentationArtifactRuntimeBinding,
    adapter: TGVFAdapter,
) -> LoadedFrozenTGVFAdapter:
    """Mutate one caller Adapter after export contract checks succeed.

    This low-level loader intentionally changes ``adapter`` in place.  Manifest,
    run, model, provider, and architecture checks happen before that mutation;
    the higher-level composer preflights its injected dependencies before it
    invokes this loader.  These checks do not establish live-model parity.
    """

    if not isinstance(binding, RepresentationArtifactRuntimeBinding):
        raise TypeError("binding must be a RepresentationArtifactRuntimeBinding")
    if not isinstance(adapter, TGVFAdapter):
        raise TypeError("adapter must be a TGVFAdapter")
    if not binding.artifact_path.is_file():
        raise FileNotFoundError(
            f"representation artifact does not exist: {binding.artifact_path}"
        )

    export = load_rank_zero_adapter_owned_state_export(binding.artifact_path)
    manifest = export.manifest
    run_identity = manifest.run_identity
    manifest_sha256 = state_digest(manifest)

    if manifest_sha256 != binding.artifact.sha256:
        raise IdentityMismatchError(
            "representation artifact manifest digest differs from Pilot binding"
        )
    if run_identity.run_id != binding.expected_run_id:
        raise IdentityMismatchError("representation artifact run ID mismatch")
    if (
        manifest.run_identity_sha256 != binding.expected_run_identity_sha256
        or run_identity.identity_sha256 != binding.expected_run_identity_sha256
    ):
        raise IdentityMismatchError("representation artifact run identity mismatch")
    if run_identity.model != binding.model:
        raise IdentityMismatchError(
            "representation artifact model identity mismatch; cross-family "
            "checkpoint reuse is forbidden"
        )
    if run_identity.provider != binding.conditioning:
        raise IdentityMismatchError(
            "representation artifact target-conditioning provider mismatch"
        )
    if run_identity.adapter_contract != binding.adapter_contract:
        raise IdentityMismatchError(
            "representation artifact architecture identity mismatch"
        )
    binding.adapter_contract.assert_matches(adapter)

    state = export.state
    if state is None:  # The strict loader already rejects this; keep the type narrow.
        raise RuntimeError("representation Adapter export has no tensor state")
    adapter.load_artifact_state_dict(state)
    adapter.requires_grad_(False)
    adapter.eval()
    _assert_frozen_eval_adapter(adapter)
    return LoadedFrozenTGVFAdapter(
        adapter=adapter,
        binding=binding,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )


@dataclass(frozen=True, slots=True)
class BehaviorHiddenStateMaterialization:
    """Typed result claimed by an injected behavior-forward dependency."""

    policy_version: PolicyVersion
    forward_identity: ArtifactIdentity
    hidden_layer: int
    hidden_states: torch.Tensor
    deterministic_forward: bool
    policy_adapter_dropout: float

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, PolicyVersion):
            raise TypeError("behavior hidden state requires a PolicyVersion")
        if not isinstance(self.forward_identity, ArtifactIdentity):
            raise TypeError("behavior hidden state requires a forward identity")
        if not isinstance(self.hidden_layer, int) or isinstance(
            self.hidden_layer, bool
        ):
            raise TypeError("behavior hidden state requires an integer layer")
        if not isinstance(self.hidden_states, torch.Tensor):
            raise TypeError("behavior hidden state payload must be a tensor")
        if self.deterministic_forward is not True:
            raise ValueError("behavior hidden-state forward must be deterministic")
        if self.policy_adapter_dropout != 0.0:
            raise ValueError(
                "behavior hidden-state forward requires zero policy dropout"
            )


class BehaviorHiddenStateDependency(Protocol):
    """Injected forward port used only by contextual conditioning."""

    def capture_hidden_states(
        self, request: BehaviorHiddenStateCaptureRequest, /
    ) -> BehaviorHiddenStateMaterialization: ...


class FrozenBehaviorHiddenStateCapturePort:
    """Check declared forward metadata and detach an injected hidden state.

    The family adapter remains responsible for proving that the dependency ran
    the selected behavior policy rather than merely returning matching labels.
    """

    def __init__(
        self,
        *,
        model: ModelIdentity,
        forward_identity: ArtifactIdentity,
        dependency: BehaviorHiddenStateDependency,
    ) -> None:
        if not isinstance(model, ModelIdentity):
            raise TypeError("hidden-state capture model must be a ModelIdentity")
        if not callable(getattr(dependency, "capture_hidden_states", None)):
            raise TypeError(
                "hidden-state dependency must implement capture_hidden_states()"
            )
        if not isinstance(forward_identity, ArtifactIdentity):
            raise TypeError("hidden-state capture requires a forward identity")
        self.model = model
        self.forward_identity = forward_identity
        self.dependency = dependency

    def capture(
        self, request: BehaviorHiddenStateCaptureRequest, /
    ) -> BehaviorHiddenStateCapture:
        if not isinstance(request, BehaviorHiddenStateCaptureRequest):
            raise TypeError("hidden-state request has the wrong type")
        if request.call.identity.model != self.model:
            raise IdentityMismatchError("hidden-state capture model identity mismatch")
        with torch.no_grad():
            materialized = self.dependency.capture_hidden_states(request)
        if not isinstance(materialized, BehaviorHiddenStateMaterialization):
            raise TypeError(
                "hidden-state dependency must return BehaviorHiddenStateMaterialization"
            )
        if materialized.policy_version != request.call.identity.behavior_policy:
            raise IdentityMismatchError(
                "hidden-state capture used a different behavior policy"
            )
        if materialized.forward_identity != self.forward_identity:
            raise IdentityMismatchError("hidden-state forward identity mismatch")
        if materialized.hidden_layer != request.hidden_layer:
            raise IdentityMismatchError("hidden-state capture used a different layer")
        # Re-run these invariants even if a caller bypassed dataclass construction.
        if materialized.deterministic_forward is not True:
            raise IdentityMismatchError("hidden-state forward was not deterministic")
        if materialized.policy_adapter_dropout != 0.0:
            raise IdentityMismatchError("hidden-state forward used nonzero dropout")
        hidden_states = materialized.hidden_states
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[0] != request.input_ids.shape[0]
        ):
            raise ValueError(
                "hidden-state dependency must return aligned [sequence, hidden] state"
            )
        if hidden_states.shape[-1] <= 0 or not hidden_states.is_floating_point():
            raise TypeError("hidden-state dependency must return floating hidden state")
        if hidden_states.device != request.input_ids.device:
            raise ValueError("hidden-state dependency returned state on another device")
        return BehaviorHiddenStateCapture(
            identity=request.call.identity,
            input_ids=request.input_ids,
            hidden_layer=request.hidden_layer,
            forward_identity=materialized.forward_identity,
            hidden_states=hidden_states.detach(),
        )


class StoredSourceVisualPort:
    """Resolve stored source tensors and validate their structural contract.

    Content addressing proves store integrity, not that a selected live family
    model produced the tensors.  That provenance remains a family-adapter gate.
    """

    def __init__(
        self,
        *,
        store: ObservationStore,
        model: ModelIdentity,
        adapter_contract: RepresentationAdapterContractIdentity,
        device: torch.device,
    ) -> None:
        if not isinstance(store, ObservationStore):
            raise TypeError("source visual store must be an ObservationStore")
        if not isinstance(model, ModelIdentity):
            raise TypeError("source visual model must be a ModelIdentity")
        if not isinstance(adapter_contract, RepresentationAdapterContractIdentity):
            raise TypeError("source visual port requires an Adapter contract")
        if not isinstance(device, torch.device):
            raise TypeError("source visual device must be explicit")
        self.store = store
        self.model = model
        self.adapter_contract = adapter_contract
        self.device = device

    def resolve(self, request: FocusRuntimeCallRequest, /) -> BoundSourceVisual:
        if not isinstance(request, FocusRuntimeCallRequest):
            raise TypeError("source visual request has the wrong type")
        if request.identity.model != self.model:
            raise IdentityMismatchError("source visual model identity mismatch")
        binding = request.trajectory_source_visual
        state = binding.state
        contract = self.adapter_contract
        if binding.deepstack_branch_layers != contract.deepstack_branch_layers:
            raise IdentityMismatchError(
                "source visual branch layers differ from Adapter architecture"
            )
        if state.spatial_merge_size != contract.spatial_merge_size:
            raise IdentityMismatchError(
                "source visual merge size differs from Adapter architecture"
            )
        tensors = SourceVisualTensorBundle(
            image_sha256=state.image_sha256,
            premerge_main=self._resolve(
                state.premerge_main, trajectory_id=request.identity.trajectory_id
            ),
            premerge_deepstack=tuple(
                self._resolve(ref, trajectory_id=request.identity.trajectory_id)
                for ref in state.premerge_deepstack
            ),
            merged_main=self._resolve(
                state.merged_main, trajectory_id=request.identity.trajectory_id
            ),
            merged_deepstack=tuple(
                self._resolve(ref, trajectory_id=request.identity.trajectory_id)
                for ref in state.merged_deepstack
            ),
            image_grid_thw=state.image_grid_thw,
            spatial_merge_size=state.spatial_merge_size,
        )
        _validate_source_visual_shape(tensors, contract)
        return BoundSourceVisual(request.identity, tensors)

    def _resolve(self, ref: TensorArtifactRef, *, trajectory_id: str) -> torch.Tensor:
        tensor = self.store.resolve_verified_for_trajectory(
            ref, trajectory_id=trajectory_id
        )
        return tensor.to(device=self.device)


class ReplayLayoutBuilderDependency(Protocol):
    """Injected family-layout builder for an already-rendered call.

    Native Qwen serialization/layout parity is outside this scaffold.
    """

    def build_replay_layout(
        self,
        request: FocusRuntimeCallRequest,
        source_visual: BoundSourceVisual,
        /,
    ) -> ReplayLayoutTensors: ...


class VerifiedReplayLayoutPort:
    """Validate shape and declared identity of an injected replay layout."""

    def __init__(
        self,
        *,
        model: ModelIdentity,
        adapter_contract: RepresentationAdapterContractIdentity,
        dependency: ReplayLayoutBuilderDependency,
    ) -> None:
        if not isinstance(model, ModelIdentity):
            raise TypeError("replay-layout model must be a ModelIdentity")
        if not isinstance(adapter_contract, RepresentationAdapterContractIdentity):
            raise TypeError("replay-layout port requires an Adapter contract")
        if not callable(getattr(dependency, "build_replay_layout", None)):
            raise TypeError(
                "replay-layout dependency must implement build_replay_layout()"
            )
        self.model = model
        self.adapter_contract = adapter_contract
        self.dependency = dependency

    def resolve(
        self,
        request: FocusRuntimeCallRequest,
        source_visual: BoundSourceVisual,
        /,
    ) -> BoundReplayLayout:
        if not isinstance(request, FocusRuntimeCallRequest):
            raise TypeError("replay-layout request has the wrong type")
        if not isinstance(source_visual, BoundSourceVisual):
            raise TypeError("replay layout requires a bound source visual")
        if request.identity.model != self.model:
            raise IdentityMismatchError("replay-layout model identity mismatch")
        if source_visual.identity != request.identity:
            raise IdentityMismatchError("replay-layout source identity mismatch")
        with torch.no_grad():
            layout = self.dependency.build_replay_layout(request, source_visual)
        if not isinstance(layout, ReplayLayoutTensors):
            raise TypeError("replay-layout dependency returned the wrong type")
        _validate_replay_layout(
            layout,
            request=request,
            source=source_visual.tensors,
            contract=self.adapter_contract,
        )
        return BoundReplayLayout(request.identity, layout)


class FrozenTGVFFocusTool(TGVFFocusTool):
    """Contract executor refusing train-mode/gradient-capable Adapter state."""

    def __init__(
        self,
        *,
        loaded_adapter: LoadedFrozenTGVFAdapter,
        store: ObservationStore,
        branch_merger_artifacts: tuple[ArtifactIdentity, ...],
    ) -> None:
        if not isinstance(loaded_adapter, LoadedFrozenTGVFAdapter):
            raise TypeError("focus tool requires a loaded frozen Adapter")
        artifacts = tuple(branch_merger_artifacts)
        if len(artifacts) != len(
            loaded_adapter.binding.adapter_contract.deepstack_branch_layers
        ) or any(not isinstance(item, ArtifactIdentity) for item in artifacts):
            raise ValueError(
                "focus tool requires one explicit artifact per D-DeepStack branch"
            )
        super().__init__(loaded_adapter.adapter, store)
        self.loaded_adapter = loaded_adapter
        self.branch_merger_artifacts = artifacts

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        binding = self.loaded_adapter.binding
        if request.model != binding.model:
            raise IdentityMismatchError("focus execution model identity mismatch")
        if request.representation != binding.artifact:
            raise IdentityMismatchError(
                "focus execution representation artifact mismatch"
            )
        if request.branch_merger_identities != self.branch_merger_artifacts:
            raise IdentityMismatchError("focus execution branch merger mismatch")
        _validate_conditioning_provenance(request, binding.conditioning)
        _assert_frozen_eval_adapter(self.adapter)
        with torch.no_grad():
            result = super().execute(request)
        tensors = (
            result.adapter_output.main_d,
            *result.adapter_output.deepstack_visual_embeds,
            result.adapter_output.conditioned_pre_merge_visual_tokens,
            *result.adapter_output.conditioned_deepstack_pre_merge_visual_tokens,
        )
        if any(
            tensor.requires_grad or tensor.grad_fn is not None for tensor in tensors
        ):
            raise RuntimeError(
                "frozen Adapter execution unexpectedly built an autograd graph"
            )
        return result


@dataclass(frozen=True, slots=True)
class PolicyPilotFocusRuntimeBridge:
    """Retained components of the contract-scaffold focus composition."""

    runtime: TGVFFocusToolRuntime
    loaded_adapter: LoadedFrozenTGVFAdapter
    conditioning_dependencies: TargetConditioningDependencies
    source_visual_port: StoredSourceVisualPort
    replay_layout_port: VerifiedReplayLayoutPort
    focus_tool: FrozenTGVFFocusTool


def build_policy_pilot_focus_runtime(
    *,
    artifact_binding: RepresentationArtifactRuntimeBinding,
    adapter: TGVFAdapter,
    conditioning_dependencies: TargetConditioningDependencies,
    contextual_hidden_state_dependency: BehaviorHiddenStateDependency | None,
    contextual_forward_identity: ArtifactIdentity | None,
    replay_layout_dependency: ReplayLayoutBuilderDependency,
    branch_mergers: tuple[BranchMergerRuntimeBinding, ...],
    observation_store: ObservationStore,
    execution_ledger: FocusExecutionLedger,
    runtime_device: torch.device,
) -> PolicyPilotFocusRuntimeBridge:
    """Compose the injection scaffold without implicit artifact/provider choices.

    Declared dependency checks are performed before the in-place Adapter loader.
    Successful construction still requires live family-adapter provenance and
    parity work before it can be treated as a production focus runtime.
    """

    if not isinstance(artifact_binding, RepresentationArtifactRuntimeBinding):
        raise TypeError("artifact_binding must be explicitly supplied")
    if not isinstance(adapter, TGVFAdapter):
        raise TypeError("adapter must be a TGVFAdapter")
    if not isinstance(conditioning_dependencies, TargetConditioningDependencies):
        raise TypeError("conditioning_dependencies must be explicitly supplied")
    if not isinstance(observation_store, ObservationStore):
        raise TypeError("observation_store must be an ObservationStore")
    if not isinstance(execution_ledger, FocusExecutionLedger):
        raise TypeError("execution_ledger must be a FocusExecutionLedger")
    if not isinstance(runtime_device, torch.device):
        raise TypeError("runtime_device must be explicit")

    # Preflight every declared non-mutating composition contract before calling
    # the intentionally in-place representation loader below.
    artifact_binding.adapter_contract.assert_matches(adapter)
    _assert_module_device(adapter, runtime_device, name="TGVF Adapter")

    merger_bindings = tuple(branch_mergers)
    if any(
        not isinstance(item, BranchMergerRuntimeBinding) for item in merger_bindings
    ):
        raise TypeError("branch_mergers must contain explicit runtime bindings")
    expected_projection_ids = (
        artifact_binding.adapter_contract.deepstack_projection_identities
    )
    if tuple(item.projection_identity for item in merger_bindings) != (
        expected_projection_ids
    ):
        raise IdentityMismatchError(
            "branch merger bindings differ from representation architecture"
        )
    branch_artifacts = tuple(item.artifact for item in merger_bindings)

    provider_kind = artifact_binding.conditioning.provider
    if provider_kind is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE:
        if conditioning_dependencies.base_embedding is not None:
            raise ValueError(
                "contextual_hidden_state cannot carry a base-embedding dependency"
            )
        if contextual_hidden_state_dependency is None:
            raise ValueError(
                "contextual_hidden_state requires an explicit behavior-forward dependency"
            )
        if not isinstance(contextual_forward_identity, ArtifactIdentity):
            raise ValueError(
                "contextual_hidden_state requires an explicit forward identity"
            )
        hidden_capture = FrozenBehaviorHiddenStateCapturePort(
            model=artifact_binding.model,
            forward_identity=contextual_forward_identity,
            dependency=contextual_hidden_state_dependency,
        )
    else:
        if contextual_hidden_state_dependency is not None:
            raise ValueError(
                "target_token_embedding cannot carry a contextual hidden-state dependency"
            )
        if contextual_forward_identity is not None:
            raise ValueError(
                "target_token_embedding cannot carry a contextual forward identity"
            )
        embedding = conditioning_dependencies.base_embedding
        if embedding is None:
            raise ValueError(
                "target_token_embedding requires an explicit base embedding dependency"
            )
        _assert_frozen_parameters(embedding, name="base input embedding")
        _assert_module_device(embedding, runtime_device, name="base input embedding")
        hidden_capture = _ForbiddenHiddenStateCapturePort()

    provider = create_target_condition_provider(
        config=artifact_binding.conditioning,
        model_identity=artifact_binding.model,
        dependencies=conditioning_dependencies,
    )
    source_port = StoredSourceVisualPort(
        store=observation_store,
        model=artifact_binding.model,
        adapter_contract=artifact_binding.adapter_contract,
        device=runtime_device,
    )
    layout_port = VerifiedReplayLayoutPort(
        model=artifact_binding.model,
        adapter_contract=artifact_binding.adapter_contract,
        dependency=replay_layout_dependency,
    )

    # This is the first intended mutation of the caller-owned Adapter.
    loaded = load_frozen_tgvf_adapter(binding=artifact_binding, adapter=adapter)
    focus_tool = FrozenTGVFFocusTool(
        loaded_adapter=loaded,
        store=observation_store,
        branch_merger_artifacts=branch_artifacts,
    )
    runtime = TGVFFocusToolRuntime(
        conditioning_provider=provider,
        hidden_state_capture=hidden_capture,
        source_visual=source_port,
        replay_layout=layout_port,
        focus_tool=focus_tool,
        representation=artifact_binding.artifact,
        branch_merger_identities=branch_artifacts,
        conditioning_input_device=runtime_device,
        contextual_hidden_layer=artifact_binding.conditioning.hidden_layer,
        contextual_forward_identity=contextual_forward_identity,
        execution_ledger=execution_ledger,
    )
    return PolicyPilotFocusRuntimeBridge(
        runtime=runtime,
        loaded_adapter=loaded,
        conditioning_dependencies=conditioning_dependencies,
        source_visual_port=source_port,
        replay_layout_port=layout_port,
        focus_tool=focus_tool,
    )


class _ForbiddenHiddenStateCapturePort:
    def capture(self, request: BehaviorHiddenStateCaptureRequest, /) -> object:
        del request
        raise RuntimeError(
            "target_token_embedding runtime must never request contextual hidden state"
        )


def _validate_conditioning_provenance(
    request: ToolExecutionRequest,
    config: TargetConditioningConfig,
) -> None:
    provenance = request.condition.provenance
    if provenance.provider != config.provider.value:
        raise IdentityMismatchError("focus conditioning provider identity mismatch")
    if config.provider is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE:
        if provenance.hidden_layer != config.hidden_layer or (
            provenance.embedding_identity is not None
        ):
            raise IdentityMismatchError("focus contextual hidden-layer mismatch")
    elif provenance.hidden_layer is not None or (
        provenance.embedding_identity != config.embedding_identity
    ):
        raise IdentityMismatchError("focus target-embedding identity mismatch")


def _validate_source_visual_shape(
    source: SourceVisualTensorBundle,
    contract: RepresentationAdapterContractIdentity,
) -> None:
    premerge = (source.premerge_main, *source.premerge_deepstack)
    merged = (source.merged_main, *source.merged_deepstack)
    expected_branches = len(contract.deepstack_branch_layers)
    if (
        len(source.premerge_deepstack) != expected_branches
        or len(source.merged_deepstack) != expected_branches
    ):
        raise ValueError("source visual is missing a model-supported DeepStack branch")
    for name, tensor, feature_dim in (
        *(("pre-merge", tensor, contract.d_v) for tensor in premerge),
        *(("merged", tensor, contract.d_lm) for tensor in merged),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"{name} source visual must have shape [tokens, hidden]")
        if not tensor.is_floating_point() or tensor.shape[-1] != feature_dim:
            raise ValueError(f"{name} source visual feature dimension mismatch")
        if tensor.requires_grad or tensor.grad_fn is not None:
            raise RuntimeError("stored source visual must be detached")
    if any(tensor.shape != source.premerge_main.shape for tensor in premerge[1:]):
        raise ValueError("source pre-merge DeepStack branch shape mismatch")
    if any(tensor.shape != source.merged_main.shape for tensor in merged[1:]):
        raise ValueError("source merged DeepStack branch shape mismatch")
    premerge_tokens = int(source.premerge_main.shape[0])
    merge_group_size = contract.spatial_merge_size**2
    if premerge_tokens % merge_group_size:
        raise ValueError("source pre-merge token count is not merge-compatible")
    if source.merged_main.shape[0] != premerge_tokens // merge_group_size:
        raise ValueError("source merged/pre-merge token counts differ")
    if source.image_grid_thw[0] * source.image_grid_thw[1] * source.image_grid_thw[
        2
    ] != (premerge_tokens):
        raise ValueError("source image grid differs from pre-merge token count")


def _validate_replay_layout(
    layout: ReplayLayoutTensors,
    *,
    request: FocusRuntimeCallRequest,
    source: SourceVisualTensorBundle,
    contract: RepresentationAdapterContractIdentity,
) -> None:
    visual = layout.visual_layout
    if visual.original_image_positions != request.trajectory_source_visual.positions:
        raise IdentityMismatchError(
            "replay layout original-image positions differ from trajectory source"
        )
    if visual.deepstack_branch_layers != contract.deepstack_branch_layers:
        raise IdentityMismatchError(
            "replay layout branch layers differ from Adapter architecture"
        )
    d_token_count = int(source.premerge_main.shape[0]) // (
        contract.spatial_merge_size**2
    )
    if len(visual.d_positions) != d_token_count or any(
        len(positions) != d_token_count
        for positions in visual.deepstack_injection_positions
    ):
        raise ValueError("replay layout does not place complete main/DeepStack D")
    if len(visual.original_image_positions) != source.merged_main.shape[0]:
        raise ValueError("replay layout source positions differ from source features")
    sequence = visual.sequence_length
    if layout.attention_mask.shape != (1, sequence) or (
        layout.attention_mask.dtype != torch.bool
    ):
        raise ValueError("replay attention mask must be bool [1,S]")
    for name, mask in (
        ("policy", layout.policy_visible_mask),
        ("reference", layout.reference_visible_mask),
        ("teacher", layout.teacher_visible_mask),
    ):
        if mask.shape != (1, sequence) or mask.dtype != torch.bool:
            raise ValueError(f"replay {name} visibility mask must be bool [1,S]")
    position_shape = tuple(layout.position_ids.shape)
    if position_shape != (1, sequence) and not (
        len(position_shape) == 3 and position_shape[-2:] == (1, sequence)
    ):
        raise ValueError("replay position IDs have the wrong shape")
    if layout.token_type_ids is not None and layout.token_type_ids.shape != (
        1,
        sequence,
    ):
        raise ValueError("replay token type IDs have the wrong shape")


def _assert_frozen_eval_adapter(adapter: TGVFAdapter) -> None:
    _assert_frozen_parameters(adapter, name="TGVF Adapter")
    if any(module.training for module in adapter.modules()):
        raise RuntimeError("TGVF Adapter and every submodule must remain in eval mode")


def _assert_frozen_parameters(module: nn.Module, *, name: str) -> None:
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError(f"{name} parameters must be frozen")


def _assert_module_device(
    module: nn.Module,
    expected: torch.device,
    *,
    name: str,
) -> None:
    owners = (*tuple(module.parameters()), *tuple(module.buffers()))
    if not owners:
        raise ValueError(f"{name} exposes no tensor state to bind its device")
    devices = {tensor.device for tensor in owners}
    if devices != {expected}:
        raise IdentityMismatchError(
            f"{name} device mismatch: expected {expected}, got {sorted(map(str, devices))}"
        )


def _require_sha256(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


__all__ = [
    "BehaviorHiddenStateDependency",
    "BehaviorHiddenStateMaterialization",
    "BranchMergerRuntimeBinding",
    "FrozenBehaviorHiddenStateCapturePort",
    "FrozenTGVFFocusTool",
    "LoadedFrozenTGVFAdapter",
    "PolicyPilotFocusRuntimeBridge",
    "ReplayLayoutBuilderDependency",
    "RepresentationArtifactRuntimeBinding",
    "StoredSourceVisualPort",
    "VerifiedReplayLayoutPort",
    "build_policy_pilot_focus_runtime",
    "load_frozen_tgvf_adapter",
]
