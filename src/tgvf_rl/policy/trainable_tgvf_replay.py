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
from tgvf_rl.contracts.tokens import (
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
)
from tgvf_rl.observations.schema import FocusedObservationRecord
from tgvf_rl.observations.store import (
    ObservationStore,
    TrajectoryReplayBundle,
    validate_replay_bundle,
)
from tgvf_rl.qwen.base import (
    InjectedForwardRequest,
    InjectedVisualBlock,
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
            raise TypeError("current Qwen module has no attached trainable RP66 Adapter")
        if not any(parameter.requires_grad for parameter in adapter.parameters()):
            raise RuntimeError("attached RP66 Adapter has no trainable parameters")
        if selected_logprob_materializer is not None and not callable(
            selected_logprob_materializer
        ):
            raise TypeError("selected_logprob_materializer must be callable")
        self.engine = engine
        self.model = model
        self.adapter = adapter
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
        recorded = resolve_replay_request(
            store, replay_handle, ReplayConsumer.POLICY
        )
        exact_ids = tuple(int(value) for value in recorded.input_ids[0].tolist())
        if exact_ids != tuple(prompt_token_ids) + response.token_ids:
            raise ReplayMismatchError(
                "current replay prompt/response differs from rollout token IDs"
            )
        if not response.policy_indices:
            raise ValueError("current replay requires policy-owned response tokens")

        request = build_trainable_tgvf_current_request(
            model=self.model,
            adapter=self.adapter,
            store=store,
            replay_handle=replay_handle,
        )
        sampled_positions = torch.tensor(
            [
                [
                    len(prompt_token_ids) + index
                    for index in response.policy_indices
                ]
            ],
            dtype=torch.long,
        )
        with torch.enable_grad(), self._autocast_context():
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
            gradient_coverage_anchor = trainable_parameter_zero_anchor(self.adapter)
            selected = selected + gradient_coverage_anchor.to(dtype=selected.dtype)
        if not selected.requires_grad:
            raise RuntimeError("current TGVF log-probabilities lost autograd")
        scatter = torch.tensor(
            response.policy_indices, dtype=torch.long, device=device
        )
        response_logprobs = torch.zeros(
            len(response.token_ids), dtype=selected.dtype, device=device
        ).scatter(0, scatter, selected)
        mask = torch.tensor(
            tuple(
                owner is TokenOwnership.POLICY_SAMPLED
                for owner in response.ownership
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
        store, replay_handle, ReplayConsumer.POLICY  # type: ignore[arg-type]
    )
    replay = store.resolve_replay(replay_handle)  # type: ignore[arg-type]
    source = replay.source_visual
    pixel_ref = source.preprocessed_pixel_values
    if pixel_ref is None:
        raise ReplayMismatchError(
            "joint RP66 replay requires trajectory-source-visual-v2 pixel_values"
        )
    pixel_values = store.resolve_verified(pixel_ref)
    vision = extract_live_qwen3_vision_features(
        model,
        pixel_values=pixel_values,
        image_grid_thw=source.state.image_grid_thw,
    )

    recorded_blocks = recorded.visual_blocks
    if not recorded_blocks or recorded_blocks[0].kind != "source_image":
        raise ReplayMismatchError("recorded replay lost its leading source image")
    source_block = InjectedVisualBlock(
        kind="source_image",
        positions=recorded_blocks[0].positions,
        embeddings=_batched(vision.merged_main),
        deepstack=tuple(_batched(value) for value in vision.merged_deepstack),
        deepstack_positions=recorded_blocks[0].deepstack_positions,
    )

    observations = tuple(
        store.resolve_record(handle) for handle in replay.observation_handles
    )
    if len(observations) != len(recorded_blocks) - 1:
        raise ReplayMismatchError("recorded observation/block counts differ")
    live_blocks: list[InjectedVisualBlock] = [source_block]
    for record, block in zip(observations, recorded_blocks[1:], strict=True):
        if not isinstance(record, FocusedObservationRecord):
            raise ValueError(
                "trainable RP66 pilot accepts original-image TGVF observations only"
            )
        if block.kind != "focused_d" or block.call_index != record.call_index:
            raise ReplayMismatchError("focused observation and replay block differ")
        if record.condition_hq is None:
            raise ReplayMismatchError(
                "joint RP66 replay requires focused-observation-v2 condition Hq"
            )
        hq = store.resolve_verified(record.condition_hq)
        owner = next(adapter.parameters())
        owner_device = tensor_compute_device(owner)
        output = adapter(
            TGVFAdapterInput(
                target_hidden_states=hq.to(
                    device=owner_device, dtype=owner.dtype
                ),
                pre_merge_visual_tokens=vision.premerge_main,
                deepstack_pre_merge_visual_tokens=vision.premerge_deepstack,
            )
        )
        if tuple(output.metadata.branch_layers) != tuple(
            record.layout.deepstack_branch_layers
        ):
            raise ReplayMismatchError("current RP66 branch layout changed")
        live_blocks.append(
            InjectedVisualBlock(
                kind="focused_d",
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


def extract_live_qwen3_vision_features(
    model: nn.Module,
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: tuple[int, int, int],
) -> LiveQwen3VisionFeatures:
    """Run current Qwen vision once and retain every merger autograd edge."""

    visual = _resolve_visual(model)
    mergers = (
        visual.merger,
        *tuple(visual.deepstack_merger_list),
    )
    if len(mergers) != 4 or len({id(value) for value in mergers}) != 4:
        raise ValueError("Qwen3 vision must expose four distinct mergers")
    captures: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
        [] for _ in mergers
    ]
    handles = tuple(
        merger.register_forward_hook(
            _capture_live_merger(captures[index]), with_kwargs=True
        )
        for index, merger in enumerate(mergers)
    )
    owner = next(visual.parameters())
    owner_device = tensor_compute_device(owner)
    grid = torch.tensor(
        (image_grid_thw,), dtype=torch.long, device=owner_device
    )
    try:
        visual(
            pixel_values.to(device=owner_device, dtype=owner.dtype),
            grid_thw=grid,
        )
    finally:
        for handle in handles:
            handle.remove()
    if any(len(rows) != 1 for rows in captures):
        raise RuntimeError("Qwen3 live vision did not execute every merger once")
    premerge = tuple(rows[0][0] for rows in captures)
    merged = tuple(rows[0][1] for rows in captures)
    return LiveQwen3VisionFeatures(
        premerge_main=premerge[0],
        premerge_deepstack=premerge[1:],
        merged_main=merged[0],
        merged_deepstack=merged[1:],
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
        if not isinstance(source, torch.Tensor) or not isinstance(
            output, torch.Tensor
        ):
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
    "TrainableTGVFCurrentReplayPort",
    "TrainableTGVFRoleReplay",
    "build_trainable_tgvf_current_request",
    "extract_live_qwen3_vision_features",
    "trainable_parameter_zero_anchor",
]
