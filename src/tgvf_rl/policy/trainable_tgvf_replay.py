"""Differentiable current-policy replay for joint Qwen + RP66 training.

The rollout bundle remains the behavior-policy record: its sampled token IDs,
ownership mask, behavior log-probabilities, Hq and D are immutable.  For the
current-policy numerator only, this module reruns the current Qwen vision tower
from the recorded preprocessed pixels and reruns the current RP66 Adapter from
the recorded tool-call Hq.  The resulting live source/D tensors are injected
at the rollout-recorded positions, preserving the exact PPO action sequence.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import ComponentRole
from tgvf_rl.contracts.tensors import TensorArtifactRef
from tgvf_rl.contracts.tokens import (
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
)
from tgvf_rl.observations.schema import (
    CropTGVFObservationRecord,
    FocusedObservationRecord,
)
from tgvf_rl.observations.store import (
    ObservationStore,
    TrajectoryReplayBundle,
    TrajectoryReplayRecord,
    validate_replay_bundle,
)
from tgvf_rl.qwen.base import (
    InjectedForwardRequest,
    InjectedVisualBlock,
    RecordedVisualBlock,
    ReplayConsumer,
    gather_behavior_measure_logprobs,
    resolve_lm_head,
    resolve_replay_request,
    validate_injected_request,
)
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.adapter import TGVFAdapter, TGVFAdapterInput
from tgvf_rl.tensor_device import tensor_compute_device

from .logprob_materializer import SelectedTokenLogprobMaterializer


TRAINABLE_TGVF_ADAPTER_ATTRIBUTE = "tgvf_adapter"


@dataclass(frozen=True, slots=True)
class LiveQwen3VisionFeatures:
    premerge_main: torch.Tensor
    premerge_deepstack: tuple[torch.Tensor, ...]
    merged_main: torch.Tensor
    merged_deepstack: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        premerge = (self.premerge_main, *self.premerge_deepstack)
        merged = (self.merged_main, *self.merged_deepstack)
        if len(premerge) != 4 or len(merged) != 4:
            raise ValueError("Qwen3 live vision requires main plus three branches")
        for name, values in (("premerge", premerge), ("merged", merged)):
            first = values[0]
            if first.ndim != 2 or not first.is_floating_point():
                raise ValueError(f"live {name} features must be floating [N,H]")
            if any(
                value.shape != first.shape
                or value.dtype != first.dtype
                or value.device != first.device
                for value in values[1:]
            ):
                raise ValueError(f"live {name} feature branches differ")


@dataclass(frozen=True, slots=True)
class LiveQwen3VisionImageSpec:
    """One content-addressed image in a trajectory-local vision replay plan."""

    kind: str
    pixel_values: TensorArtifactRef
    image_grid_thw: tuple[int, int, int]
    observation_index: int | None
    call_index: int | None

    def __post_init__(self) -> None:
        if self.kind not in {"source_image", "crop_image"}:
            raise ValueError("live vision image kind must be source_image/crop_image")
        if not isinstance(self.pixel_values, TensorArtifactRef):
            raise TypeError("live vision image requires a tensor artifact reference")
        if len(self.image_grid_thw) != 3 or any(
            type(value) is not int or value <= 0 for value in self.image_grid_thw
        ):
            raise ValueError("live vision image grid must contain three positive ints")
        descriptor = self.pixel_values.descriptor
        expected_rows = (
            self.image_grid_thw[0] * self.image_grid_thw[1] * self.image_grid_thw[2]
        )
        if (
            len(descriptor.shape) != 2
            or descriptor.shape[0] != expected_rows
            or descriptor.shape[1] <= 0
        ):
            raise ValueError("live vision pixel artifact rows differ from its grid")
        descriptor_dtype = getattr(torch, descriptor.dtype, None)
        if (
            not isinstance(descriptor_dtype, torch.dtype)
            or not torch.empty((), dtype=descriptor_dtype).is_floating_point()
        ):
            raise TypeError("live vision pixel artifact must use a floating dtype")
        if self.kind == "source_image":
            if self.observation_index is not None or self.call_index is not None:
                raise ValueError("source image cannot carry a tool-call identity")
        elif (
            type(self.observation_index) is not int
            or self.observation_index < 0
            or type(self.call_index) is not int
            or self.call_index < 0
        ):
            raise ValueError("crop image requires observation and call identities")


@dataclass(frozen=True, slots=True)
class LiveQwen3VisionReplayPlan:
    """Immutable source/crop packing and observation placement for one row.

    A trajectory is the bounded distributed execution unit: source plus every
    crop is one native Qwen multi-image call, while a plain TGVF observation
    points back to the source image.  Keeping this plan trajectory-local bounds
    activation memory independently of the actor micro-batch size.
    """

    replay_id: str
    trajectory_id: str
    images: tuple[LiveQwen3VisionImageSpec, ...]
    observation_image_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(
            self,
            "observation_image_indices",
            tuple(self.observation_image_indices),
        )
        if not self.replay_id or not self.trajectory_id:
            raise ValueError("live vision replay identities must be non-empty")
        if not self.images or self.images[0].kind != "source_image":
            raise ValueError("live vision replay must begin with one source image")
        if any(image.kind != "crop_image" for image in self.images[1:]):
            raise ValueError("only crop images may follow the source image")
        if any(
            type(index) is not int or index < 0 or index >= len(self.images)
            for index in self.observation_image_indices
        ):
            raise ValueError("observation image placement lies outside the plan")
        crop_observations = tuple(image.observation_index for image in self.images[1:])
        if len(set(crop_observations)) != len(crop_observations):
            raise ValueError("crop observation placements must be unique")
        for observation_index, packed_index in enumerate(
            self.observation_image_indices
        ):
            if packed_index and (
                self.images[packed_index].observation_index != observation_index
            ):
                raise ValueError("observation placement aliases another crop image")
        for packed_index, image in enumerate(self.images[1:], start=1):
            assert image.observation_index is not None
            if (
                image.observation_index >= len(self.observation_image_indices)
                or self.observation_image_indices[image.observation_index]
                != packed_index
            ):
                raise ValueError("crop image and observation placement differ")


@dataclass(frozen=True, slots=True)
class LiveQwen3VisionReplayResult:
    """Differentiable features split according to one verified replay plan."""

    plan: LiveQwen3VisionReplayPlan
    features: tuple[LiveQwen3VisionFeatures, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        if len(self.features) != len(self.plan.images):
            raise ValueError("live vision plan/result image counts differ")

    @property
    def source(self) -> LiveQwen3VisionFeatures:
        return self.features[0]

    def for_observation(self, index: int) -> LiveQwen3VisionFeatures:
        if type(index) is not int or not 0 <= index < len(
            self.plan.observation_image_indices
        ):
            raise IndexError("live vision observation index lies outside the plan")
        return self.features[self.plan.observation_image_indices[index]]


@dataclass(frozen=True, slots=True)
class TrainableTGVFRoleReplay:
    role: ComponentRole
    bundle_sha256: str
    response_token_ids: tuple[int, ...]
    response_ownership: tuple[TokenOwnership, ...]
    policy_sampled_mask: torch.Tensor
    logprobs: torch.Tensor


class TrainableTGVFCurrentReplayPort:
    """Exact-response replay whose current visual state is differentiable."""

    def __init__(
        self,
        *,
        engine: Any,
        model: nn.Module,
        selected_logprob_materializer: SelectedTokenLogprobMaterializer | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("trainable TGVF replay model must be a torch module")
        adapter = getattr(model, TRAINABLE_TGVF_ADAPTER_ATTRIBUTE, None)
        if not isinstance(adapter, TGVFAdapter):
            raise TypeError(
                "current Qwen module has no attached trainable RP66 Adapter"
            )
        artifact_names = set(adapter.artifact_state_dict(keep_vars=True))
        adapter_parameters = tuple(
            parameter
            for name, parameter in adapter.named_parameters()
            if name in artifact_names
        )
        if not adapter_parameters:
            raise RuntimeError("attached RP66 Adapter has no owned parameters")
        trainable_flags = {
            bool(parameter.requires_grad) for parameter in adapter_parameters
        }
        if len(trainable_flags) != 1:
            raise RuntimeError("RP66 Adapter mixes frozen and trainable parameters")
        if selected_logprob_materializer is not None and not callable(
            selected_logprob_materializer
        ):
            raise TypeError("selected_logprob_materializer must be callable")
        self.engine = engine
        self.model = model
        self.adapter = adapter
        self.adapter_trainable = trainable_flags == {True}
        self.family_adapter = Qwen3VLAdapter()
        self.selected_logprob_materializer = selected_logprob_materializer
        self.materializes_fused_kernels = selected_logprob_materializer is not None
        self.binding = SimpleNamespace(role=ComponentRole.CURRENT)

    def replay_response_logprobs(
        self,
        *,
        bundle: TrajectoryReplayBundle,
        prompt_token_ids: tuple[int, ...],
        response: OwnedTokenSequence,
        sampling: SamplingIdentity,
    ) -> TrainableTGVFRoleReplay:
        if not isinstance(bundle, TrajectoryReplayBundle):
            raise TypeError("current replay requires TrajectoryReplayBundle")
        if not isinstance(response, OwnedTokenSequence):
            raise TypeError("response must be OwnedTokenSequence")
        if not isinstance(sampling, SamplingIdentity):
            raise TypeError("sampling must be SamplingIdentity")
        if sampling.policy_version != bundle.replay_record.behavior_policy:
            raise IdentityMismatchError(
                "sampling version differs from replay behavior policy"
            )
        validate_replay_bundle(bundle)
        store, replay_handle = ObservationStore.from_replay_bundle(bundle)
        recorded = resolve_replay_request(store, replay_handle, ReplayConsumer.POLICY)
        exact_ids = tuple(int(value) for value in recorded.input_ids[0].tolist())
        if exact_ids != tuple(prompt_token_ids) + response.token_ids:
            raise ReplayMismatchError(
                "current replay prompt/response differs from rollout token IDs"
            )
        if not response.policy_indices:
            raise ValueError("current replay requires policy-owned response tokens")

        sampled_positions = torch.tensor(
            [[len(prompt_token_ids) + index for index in response.policy_indices]],
            dtype=torch.long,
        )
        with torch.enable_grad(), self._autocast_context():
            # The current-policy replay is one differentiable visual-language
            # forward, so its mixed-precision boundary must begin before the
            # live Qwen vision tower and RP66 Adapter are recomputed.  In
            # particular, Qwen's vision blocks use activation checkpointing;
            # creating those checkpoints outside autocast lets their backward
            # recomputation see FP32 FSDP master parameters with BF16 hidden
            # states.  Crop's ordinary top-level forward already scopes the
            # complete vision-to-decoder path this way.
            request = build_trainable_tgvf_current_request(
                model=self.model,
                adapter=self.adapter,
                store=store,
                replay_handle=replay_handle,
            )
            if self.selected_logprob_materializer is None:
                output = self.family_adapter.forward_injected(self.model, request)
                device = output.logits.device
                selected = gather_behavior_measure_logprobs(
                    output.logits,
                    request.input_ids.to(device=device),
                    sampled_positions.to(device=device),
                    sampling,
                ).squeeze(0)
            else:
                hidden = self.family_adapter.forward_injected_hidden(
                    self.model, request
                )
                selected = self.selected_logprob_materializer(
                    hidden_states=hidden.hidden_states,
                    lm_head=resolve_lm_head(self.model),
                    token_ids=request.input_ids,
                    sampled_positions=sampled_positions,
                    sampling=sampling,
                ).squeeze(0)
                device = selected.device
            if self.adapter_trainable:
                gradient_coverage_anchor = trainable_parameter_zero_anchor(self.adapter)
                selected = selected + gradient_coverage_anchor.to(dtype=selected.dtype)
        if not selected.requires_grad:
            raise RuntimeError("current TGVF log-probabilities lost autograd")
        scatter = torch.tensor(response.policy_indices, dtype=torch.long, device=device)
        response_logprobs = torch.zeros(
            len(response.token_ids), dtype=selected.dtype, device=device
        ).scatter(0, scatter, selected)
        mask = torch.tensor(
            tuple(
                owner is TokenOwnership.POLICY_SAMPLED for owner in response.ownership
            ),
            dtype=torch.bool,
            device=device,
        )
        return TrainableTGVFRoleReplay(
            role=ComponentRole.CURRENT,
            bundle_sha256=bundle.bundle_sha256,
            response_token_ids=response.token_ids,
            response_ownership=response.ownership,
            policy_sampled_mask=mask,
            logprobs=response_logprobs,
        )

    def _autocast_context(self):
        dtype = getattr(self.engine, "_autocast_dtype", torch.float32)
        if dtype == torch.float32:
            return nullcontext()
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("FSDP actor autocast dtype must be floating")
        owner = next(self.model.parameters())
        return torch.autocast(
            device_type=tensor_compute_device(owner).type,
            dtype=dtype,
        )


def build_trainable_tgvf_current_request(
    *,
    model: nn.Module,
    adapter: TGVFAdapter,
    store: ObservationStore,
    replay_handle: object,
) -> InjectedForwardRequest:
    """Replace recorded source/D tensors with current differentiable tensors."""

    recorded = resolve_replay_request(
        store,
        replay_handle,
        ReplayConsumer.POLICY,  # type: ignore[arg-type]
    )
    replay = store.resolve_replay(replay_handle)  # type: ignore[arg-type]
    source = replay.source_visual
    pixel_ref = source.preprocessed_pixel_values
    if pixel_ref is None:
        raise ReplayMismatchError(
            "joint RP66 replay requires trajectory-source-visual-v2 pixel_values"
        )
    recorded_blocks = recorded.visual_blocks
    if not recorded_blocks or recorded_blocks[0].kind != "source_image":
        raise ReplayMismatchError("recorded replay lost its leading source image")
    observations = tuple(
        store.resolve_record(handle) for handle in replay.observation_handles
    )
    if len(observations) != len(recorded_blocks) - 1:
        raise ReplayMismatchError("recorded observation/block counts differ")
    plan = build_live_qwen3_vision_replay_plan(
        replay=replay,
        observations=observations,
        recorded_blocks=recorded_blocks,
    )
    live_vision = execute_live_qwen3_vision_replay_plan(
        model,
        store=store,
        plan=plan,
    )
    vision = live_vision.source
    source_block = InjectedVisualBlock(
        kind="source_image",
        positions=recorded_blocks[0].positions,
        embeddings=_batched(vision.merged_main),
        deepstack=tuple(_batched(value) for value in vision.merged_deepstack),
        deepstack_positions=recorded_blocks[0].deepstack_positions,
    )

    live_blocks: list[InjectedVisualBlock] = [source_block]
    for index, (record, block) in enumerate(
        zip(observations, recorded_blocks[1:], strict=True)
    ):
        if isinstance(record, CropTGVFObservationRecord):
            observation_vision = live_vision.for_observation(index)
            output_kind = "crop_focused_d"
        elif isinstance(record, FocusedObservationRecord):
            observation_vision = live_vision.for_observation(index)
            output_kind = "focused_d"
        else:
            raise AssertionError("observation type changed after replay validation")
        hq = store.resolve_verified(record.condition_hq)
        owner = next(adapter.parameters())
        owner_device = tensor_compute_device(owner)
        output = adapter(
            TGVFAdapterInput(
                target_hidden_states=hq.to(device=owner_device, dtype=owner.dtype),
                pre_merge_visual_tokens=observation_vision.premerge_main,
                deepstack_pre_merge_visual_tokens=(
                    observation_vision.premerge_deepstack
                ),
            )
        )
        if tuple(output.metadata.branch_layers) != tuple(
            record.layout.deepstack_branch_layers
        ):
            raise ReplayMismatchError("current RP66 branch layout changed")
        live_blocks.append(
            InjectedVisualBlock(
                kind=output_kind,
                positions=block.positions,
                embeddings=_batched(output.main_d),
                deepstack=tuple(
                    _batched(value) for value in output.deepstack_visual_embeds
                ),
                deepstack_positions=block.deepstack_positions,
            )
        )

    request = InjectedForwardRequest(
        input_ids=recorded.input_ids,
        attention_mask=recorded.attention_mask,
        position_ids=recorded.position_ids,
        visual_blocks=tuple(live_blocks),
        use_cache=False,
    )
    validate_injected_request(request)
    return request


def build_live_qwen3_vision_replay_plan(
    *,
    replay: TrajectoryReplayRecord,
    observations: tuple[CropTGVFObservationRecord | FocusedObservationRecord, ...],
    recorded_blocks: tuple[RecordedVisualBlock, ...],
) -> LiveQwen3VisionReplayPlan:
    """Build a trajectory-local, identity-preserving source/crop replay plan."""

    if not isinstance(replay, TrajectoryReplayRecord):
        raise TypeError("live vision replay requires a TrajectoryReplayRecord")
    source_ref = replay.source_visual.preprocessed_pixel_values
    if source_ref is None:
        raise ReplayMismatchError(
            "joint RP66 replay requires trajectory-source-visual-v2 pixel_values"
        )
    if not recorded_blocks or recorded_blocks[0].kind != "source_image":
        raise ReplayMismatchError("recorded replay lost its leading source image")
    if len(observations) != len(recorded_blocks) - 1:
        raise ReplayMismatchError("recorded observation/block counts differ")

    images = [
        LiveQwen3VisionImageSpec(
            kind="source_image",
            pixel_values=source_ref,
            image_grid_thw=replay.source_visual.state.image_grid_thw,
            observation_index=None,
            call_index=None,
        )
    ]
    placements: list[int] = []
    for observation_index, (record, block) in enumerate(
        zip(observations, recorded_blocks[1:], strict=True)
    ):
        if not isinstance(
            record, (CropTGVFObservationRecord, FocusedObservationRecord)
        ):
            raise ValueError(
                "trainable RP66 replay accepts TGVF or atomic Crop+TGVF "
                "observations only"
            )
        if block.call_index != record.call_index:
            raise ReplayMismatchError("TGVF observation and replay call index differ")
        if isinstance(record, CropTGVFObservationRecord):
            if block.kind != "crop_focused_d":
                raise ReplayMismatchError(
                    "atomic Crop+TGVF observation and replay block differ"
                )
            placements.append(len(images))
            images.append(
                LiveQwen3VisionImageSpec(
                    kind="crop_image",
                    pixel_values=record.crop_visual.preprocessed_pixel_values,
                    image_grid_thw=record.crop_visual.source.image_grid_thw,
                    observation_index=observation_index,
                    call_index=record.call_index,
                )
            )
        else:
            if block.kind != "focused_d":
                raise ReplayMismatchError("focused observation and replay block differ")
            if record.condition_hq is None:
                raise ReplayMismatchError(
                    "joint RP66 replay requires focused-observation-v2 condition Hq"
                )
            placements.append(0)
    return LiveQwen3VisionReplayPlan(
        replay_id=replay.replay_id,
        trajectory_id=replay.trajectory_id,
        images=tuple(images),
        observation_image_indices=tuple(placements),
    )


def execute_live_qwen3_vision_replay_plan(
    model: nn.Module,
    *,
    store: ObservationStore,
    plan: LiveQwen3VisionReplayPlan,
) -> LiveQwen3VisionReplayResult:
    """Execute one bounded multi-image call for one verified trajectory plan."""

    if not isinstance(store, ObservationStore):
        raise TypeError("live vision replay requires an ObservationStore")
    if not isinstance(plan, LiveQwen3VisionReplayPlan):
        raise TypeError("plan must be LiveQwen3VisionReplayPlan")
    features = extract_live_qwen3_vision_feature_batch(
        model,
        pixel_values=tuple(
            store.resolve_verified_for_trajectory(
                image.pixel_values,
                trajectory_id=plan.trajectory_id,
            )
            for image in plan.images
        ),
        image_grid_thw=tuple(image.image_grid_thw for image in plan.images),
    )
    return LiveQwen3VisionReplayResult(plan=plan, features=features)


def extract_live_qwen3_vision_features(
    model: nn.Module,
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: tuple[int, int, int],
) -> LiveQwen3VisionFeatures:
    """Run current Qwen vision once and retain every merger autograd edge."""

    return extract_live_qwen3_vision_feature_batch(
        model,
        pixel_values=(pixel_values,),
        image_grid_thw=(image_grid_thw,),
    )[0]


def extract_live_qwen3_vision_feature_batch(
    model: nn.Module,
    *,
    pixel_values: tuple[torch.Tensor, ...],
    image_grid_thw: tuple[tuple[int, int, int], ...],
) -> tuple[LiveQwen3VisionFeatures, ...]:
    """Run one native Qwen multi-image vision pass and split exact features.

    Every image remains an independent attention sequence because Qwen derives
    its cumulative sequence boundaries from the rows of ``grid_thw``.  The
    packed call is thus mathematically equivalent to one vision call per image,
    while keeping child-FSDP collective counts identical across actor ranks.
    """

    if not pixel_values or len(pixel_values) != len(image_grid_thw):
        raise ValueError("live Qwen vision inputs must contain matching images/grids")

    visual = _resolve_visual(model)
    spatial_merge_size = getattr(visual, "spatial_merge_size", None)
    if type(spatial_merge_size) is not int or spatial_merge_size <= 0:
        raise ValueError("Qwen3 visual must expose a positive spatial_merge_size")
    merge_group_size = spatial_merge_size**2
    premerge_counts: list[int] = []
    for index, (pixels, grid) in enumerate(
        zip(pixel_values, image_grid_thw, strict=True)
    ):
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 2:
            raise ValueError(f"live Qwen image {index} pixels must have shape [N,D]")
        if not pixels.is_floating_point():
            raise TypeError(f"live Qwen image {index} pixels must be floating")
        if len(grid) != 3 or any(
            type(value) is not int or value <= 0 for value in grid
        ):
            raise ValueError(
                f"live Qwen image {index} grid must be three positive ints"
            )
        if grid[1] % spatial_merge_size or grid[2] % spatial_merge_size:
            raise ValueError(
                f"live Qwen image {index} grid is not spatial-merge aligned"
            )
        expected_rows = grid[0] * grid[1] * grid[2]
        if int(pixels.shape[0]) != expected_rows:
            raise ValueError(
                f"live Qwen image {index} pixel rows differ from grid: "
                f"{pixels.shape[0]} vs {expected_rows}"
            )
        premerge_counts.append(expected_rows)
    pixel_widths = {int(value.shape[1]) for value in pixel_values}
    if len(pixel_widths) != 1:
        raise ValueError("packed live Qwen images must share a patch feature width")

    mergers = (
        visual.merger,
        *tuple(visual.deepstack_merger_list),
    )
    if len(mergers) != 4 or len({id(value) for value in mergers}) != 4:
        raise ValueError("Qwen3 vision must expose four distinct mergers")
    captures: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in mergers]
    handles = tuple(
        merger.register_forward_hook(
            _capture_live_merger(captures[index]), with_kwargs=True
        )
        for index, merger in enumerate(mergers)
    )
    owner = next(visual.parameters())
    owner_device = tensor_compute_device(owner)
    grid = torch.tensor(image_grid_thw, dtype=torch.long, device=owner_device)
    packed_pixels = torch.cat(
        tuple(
            value.to(device=owner_device, dtype=owner.dtype) for value in pixel_values
        ),
        dim=0,
    )
    try:
        visual(packed_pixels, grid_thw=grid)
    finally:
        for handle in handles:
            handle.remove()
    if any(len(rows) != 1 for rows in captures):
        raise RuntimeError("Qwen3 live vision did not execute every merger once")
    packed_premerge = tuple(rows[0][0] for rows in captures)
    packed_merged = tuple(rows[0][1] for rows in captures)
    merged_counts = tuple(value // merge_group_size for value in premerge_counts)
    premerge_splits = tuple(
        torch.split(value, tuple(premerge_counts), dim=0) for value in packed_premerge
    )
    merged_splits = tuple(
        torch.split(value, merged_counts, dim=0) for value in packed_merged
    )
    return tuple(
        LiveQwen3VisionFeatures(
            premerge_main=premerge_splits[0][index],
            premerge_deepstack=tuple(branch[index] for branch in premerge_splits[1:]),
            merged_main=merged_splits[0][index],
            merged_deepstack=tuple(branch[index] for branch in merged_splits[1:]),
        )
        for index in range(len(pixel_values))
    )


def _capture_live_merger(destination: list[tuple[torch.Tensor, torch.Tensor]]):
    def hook(
        _module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        output: object,
    ) -> None:
        source = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if source is None:
            source = kwargs.get("hidden_states")  # type: ignore[assignment]
        if not isinstance(source, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise TypeError("Qwen3 merger must expose tensor input and output")
        if source.ndim == 3 and source.shape[1] == 1:
            source = source[:, 0]
        if source.ndim != 2 or output.ndim != 2:
            raise ValueError("Qwen3 merger boundaries must be rank two")
        # Deliberately do not detach or clone: these are the actor autograd edges.
        destination.append((source, output))

    return hook


def trainable_parameter_zero_anchor(module: nn.Module) -> torch.Tensor:
    """Connect every trainable parameter to a scalar, numerically-zero loss edge.

    FSDP2 reduces only parameters whose unsharded gradient is not ``None``.
    Joint RP66 replay legitimately includes both tool-using rows and direct
    rows.  Without this edge, ranks processing direct rows omit every
    Adapter-owned parameter from the root reduce-scatter while peer ranks
    include them, producing different collective payloads.

    Touching one scalar from *each* parameter is sufficient to materialize its
    zero gradient.  It avoids a synthetic Adapter forward and leaves logits,
    rewards, and all numerical gradients unchanged.
    """

    parameters = tuple(
        parameter for parameter in module.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise RuntimeError("gradient coverage requires trainable parameters")
    if any(parameter.numel() == 0 for parameter in parameters):
        raise ValueError("gradient coverage does not accept empty parameters")
    first = parameters[0]
    if any(
        parameter.device != first.device or parameter.dtype != first.dtype
        for parameter in parameters[1:]
    ):
        raise ValueError("gradient-covered parameters must share device and dtype")

    anchor = first.reshape(-1)[0] * 0.0
    for parameter in parameters[1:]:
        anchor = anchor + parameter.reshape(-1)[0] * 0.0
    if anchor.ndim != 0 or not anchor.requires_grad:
        raise RuntimeError("gradient coverage anchor lost its autograd graph")
    return anchor


def _resolve_visual(model: nn.Module) -> nn.Module:
    container = getattr(model, "model", model)
    visual = getattr(container, "visual", None)
    if not isinstance(visual, nn.Module):
        raise TypeError("Qwen3 model does not expose model.visual")
    return visual


def _batched(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(0)
    if value.ndim == 3 and value.shape[0] == 1:
        return value
    raise ValueError("live visual feature must have shape [N,H] or [1,N,H]")


__all__ = [
    "TRAINABLE_TGVF_ADAPTER_ATTRIBUTE",
    "LiveQwen3VisionFeatures",
    "LiveQwen3VisionImageSpec",
    "LiveQwen3VisionReplayPlan",
    "LiveQwen3VisionReplayResult",
    "TrainableTGVFCurrentReplayPort",
    "TrainableTGVFRoleReplay",
    "build_live_qwen3_vision_replay_plan",
    "build_trainable_tgvf_current_request",
    "execute_live_qwen3_vision_replay_plan",
    "extract_live_qwen3_vision_feature_batch",
    "extract_live_qwen3_vision_features",
    "trainable_parameter_zero_anchor",
]
