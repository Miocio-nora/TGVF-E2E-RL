"""Differentiable exact replay for a full-Qwen plain-Crop actor.

The rollout bundle remains immutable behavior evidence.  Current-policy replay
keeps its exact token sequence and visual placement, but recomputes source and
crop features from the rollout-recorded processor pixels through the live Qwen
vision tower.  Consequently the PPO numerator updates the same full Qwen
parameter scope as the matched Atomic Crop+TGVF arm.
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
from tgvf_rl.observations.schema import CropObservationRecord
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
from tgvf_rl.tensor_device import tensor_compute_device

from .logprob_materializer import SelectedTokenLogprobMaterializer
from .trainable_tgvf_replay import (
    build_live_qwen3_vision_replay_plan,
    execute_live_qwen3_vision_replay_plan,
)


@dataclass(frozen=True, slots=True)
class TrainableCropRoleReplay:
    role: ComponentRole
    bundle_sha256: str
    response_token_ids: tuple[int, ...]
    response_ownership: tuple[TokenOwnership, ...]
    policy_sampled_mask: torch.Tensor
    logprobs: torch.Tensor


class TrainableCropCurrentReplayPort:
    """Exact-response replay with live, differentiable source/crop vision."""

    def __init__(
        self,
        *,
        engine: Any,
        model: nn.Module,
        selected_logprob_materializer: SelectedTokenLogprobMaterializer | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("trainable Crop replay model must be a torch module")
        if selected_logprob_materializer is not None and not callable(
            selected_logprob_materializer
        ):
            raise TypeError("selected_logprob_materializer must be callable")
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("current Crop actor exposes no trainable parameters")
        visual = getattr(getattr(model, "model", model), "visual", None)
        if not isinstance(visual, nn.Module) or not any(
            parameter.requires_grad for parameter in visual.parameters()
        ):
            raise RuntimeError(
                "current Crop actor requires a trainable Qwen vision tower"
            )
        self.engine = engine
        self.model = model
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
    ) -> TrainableCropRoleReplay:
        if not isinstance(bundle, TrajectoryReplayBundle):
            raise TypeError("current Crop replay requires TrajectoryReplayBundle")
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
                "current Crop replay prompt/response differs from rollout token IDs"
            )
        if not response.policy_indices:
            raise ValueError(
                "current Crop replay requires policy-owned response tokens"
            )

        sampled_positions = torch.tensor(
            [[len(prompt_token_ids) + index for index in response.policy_indices]],
            dtype=torch.long,
        )
        with torch.enable_grad(), self._autocast_context():
            request = build_trainable_crop_current_request(
                model=self.model,
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
        if not selected.requires_grad:
            raise RuntimeError("current Crop log-probabilities lost autograd")
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
        return TrainableCropRoleReplay(
            role=ComponentRole.CURRENT,
            bundle_sha256=bundle.bundle_sha256,
            response_token_ids=response.token_ids,
            response_ownership=response.ownership,
            policy_sampled_mask=mask,
            logprobs=response_logprobs,
        )

    def replay_response_logprobs_batch(
        self,
        *,
        rows: tuple[Any, ...],
    ) -> tuple[TrainableCropRoleReplay, ...]:
        """Keep live vision rowwise but execute all decoder rows together."""

        if not rows:
            raise ValueError("current Crop replay batch cannot be empty")
        prepared: list[
            tuple[
                TrajectoryReplayBundle,
                tuple[int, ...],
                OwnedTokenSequence,
                SamplingIdentity,
                ObservationStore,
                object,
            ]
        ] = []
        for row in rows:
            bundle = getattr(row, "bundle", None)
            prompt = tuple(getattr(row, "prompt_token_ids", ()))
            response = getattr(row, "response", None)
            sampling = getattr(row, "sampling", None)
            if not isinstance(bundle, TrajectoryReplayBundle):
                raise TypeError("current Crop replay batch row lost its bundle")
            if not isinstance(response, OwnedTokenSequence):
                raise TypeError("current Crop replay batch row lost its response")
            if not isinstance(sampling, SamplingIdentity):
                raise TypeError(
                    "current Crop replay batch row lost its sampling identity"
                )
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
            if exact_ids != prompt + response.token_ids:
                raise ReplayMismatchError(
                    "current Crop replay prompt/response differs from rollout token IDs"
                )
            if not response.policy_indices:
                raise ValueError(
                    "current Crop replay requires policy-owned response tokens"
                )
            prepared.append((bundle, prompt, response, sampling, store, replay_handle))

        with torch.enable_grad(), self._autocast_context():
            requests = tuple(
                build_trainable_crop_current_request(
                    model=self.model,
                    store=item[4],
                    replay_handle=item[5],
                )
                for item in prepared
            )
            hidden_rows = self.family_adapter.forward_injected_hidden_batch(
                self.model, requests
            )
            results: list[TrainableCropRoleReplay] = []
            for item, request, hidden in zip(
                prepared, requests, hidden_rows, strict=True
            ):
                bundle, prompt, response, sampling, _store, _handle = item
                response_indices = response.policy_indices
                device = hidden.hidden_states.device
                sampled_positions = torch.tensor(
                    [[len(prompt) + index for index in response_indices]],
                    dtype=torch.long,
                    device=device,
                )
                if self.selected_logprob_materializer is None:
                    logits = resolve_lm_head(self.model)(hidden.hidden_states)
                    selected = gather_behavior_measure_logprobs(
                        logits,
                        request.input_ids.to(device=device),
                        sampled_positions,
                        sampling,
                    ).squeeze(0)
                else:
                    selected = self.selected_logprob_materializer(
                        hidden_states=hidden.hidden_states,
                        lm_head=resolve_lm_head(self.model),
                        token_ids=request.input_ids,
                        sampled_positions=sampled_positions,
                        sampling=sampling,
                    ).squeeze(0)
                if not selected.requires_grad:
                    raise RuntimeError("current Crop log-probabilities lost autograd")
                scatter = torch.tensor(
                    response_indices, dtype=torch.long, device=device
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
                results.append(
                    TrainableCropRoleReplay(
                        role=ComponentRole.CURRENT,
                        bundle_sha256=bundle.bundle_sha256,
                        response_token_ids=response.token_ids,
                        response_ownership=response.ownership,
                        policy_sampled_mask=mask,
                        logprobs=response_logprobs,
                    )
                )
        return tuple(results)

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


def build_trainable_crop_current_request(
    *,
    model: nn.Module,
    store: ObservationStore,
    replay_handle: object,
) -> InjectedForwardRequest:
    """Replace every recorded RGB visual block with current Qwen features."""

    recorded = resolve_replay_request(
        store,
        replay_handle,
        ReplayConsumer.POLICY,  # type: ignore[arg-type]
    )
    replay = store.resolve_replay(replay_handle)  # type: ignore[arg-type]
    if replay.source_visual.preprocessed_pixel_values is None:
        raise ReplayMismatchError(
            "differentiable Crop replay requires recorded source pixel_values"
        )
    recorded_blocks = recorded.visual_blocks
    if not recorded_blocks or recorded_blocks[0].kind != "source_image":
        raise ReplayMismatchError("recorded replay lost its leading source image")
    observations = tuple(
        store.resolve_record(handle) for handle in replay.observation_handles
    )
    if any(not isinstance(record, CropObservationRecord) for record in observations):
        raise ReplayMismatchError(
            "plain Crop replay received a non-Crop observation record"
        )
    plan = build_live_qwen3_vision_replay_plan(
        replay=replay,
        observations=observations,  # type: ignore[arg-type]
        recorded_blocks=recorded_blocks,
    )
    live_vision = execute_live_qwen3_vision_replay_plan(
        model,
        store=store,
        plan=plan,
    )
    source = live_vision.source
    live_blocks: list[InjectedVisualBlock] = [
        InjectedVisualBlock(
            kind="source_image",
            positions=recorded_blocks[0].positions,
            embeddings=_batched(source.merged_main),
            deepstack=tuple(_batched(value) for value in source.merged_deepstack),
            deepstack_positions=recorded_blocks[0].deepstack_positions,
        )
    ]
    for index, (record, block) in enumerate(
        zip(observations, recorded_blocks[1:], strict=True)
    ):
        assert isinstance(record, CropObservationRecord)
        if block.kind != "crop_image" or block.call_index != record.call_index:
            raise ReplayMismatchError("Crop observation and replay block differ")
        crop = live_vision.for_observation(index)
        live_blocks.append(
            InjectedVisualBlock(
                kind="crop_image",
                positions=block.positions,
                embeddings=_batched(crop.merged_main),
                deepstack=tuple(_batched(value) for value in crop.merged_deepstack),
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


def _batched(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(0)
    if value.ndim == 3 and value.shape[0] == 1:
        return value
    raise ValueError("live visual feature must have shape [N,H] or [1,N,H]")


__all__ = [
    "TrainableCropCurrentReplayPort",
    "TrainableCropRoleReplay",
    "build_trainable_crop_current_request",
]
