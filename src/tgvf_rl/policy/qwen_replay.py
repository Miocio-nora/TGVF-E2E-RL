"""Concrete Qwen3 recorded-observation replay and decoder-LoRA construction.

This module is deliberately below any optimizer or veRL worker.  It binds an
already constructed model to the framework-neutral replay port, consumes only
rollout-exported :class:`TrajectoryReplayBundle` values, and exposes the exact
response-token log probabilities needed by a future worker integration.
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
from tgvf_rl.observations.store import (
    ObservationStore,
    TrajectoryReplayBundle,
    validate_replay_bundle,
)
from tgvf_rl.qwen.base import (
    RecordedReplayRequest,
    ReplayConsumer,
    gather_behavior_measure_logprobs,
    injected_request_from_recorded,
    resolve_language_model,
    resolve_lm_head,
    resolve_replay_request,
    validate_replay_request,
)
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter

from .config import POLICY_PILOT_V1_MODEL_FAMILY, DecoderLoRAConfig
from .exact_replay import (
    RecordedPolicyForwardBinding,
    RecordedPolicyForwardOutput,
    RecordedPolicyForwardStateProof,
)
from .model_scope import (
    ModelScopeAudit,
    audit_policy_model_scope,
    audit_reference_model_scope,
    resolve_qwen3_decoder_lora_targets,
)


@dataclass(frozen=True, slots=True)
class Qwen3DecoderLoRABuild:
    """A built PEFT policy plus the positive-whitelist ownership proof."""

    model: nn.Module
    config: DecoderLoRAConfig
    target_modules: tuple[str, ...]
    scope_audit: ModelScopeAudit

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError("built Qwen3 LoRA policy must be a torch module")
        if not isinstance(self.config, DecoderLoRAConfig):
            raise TypeError("built Qwen3 LoRA policy requires DecoderLoRAConfig")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if self.target_modules != self.scope_audit.decoder_target_modules:
            raise IdentityMismatchError(
                "built Qwen3 LoRA targets differ from the startup scope audit"
            )


def build_qwen3_decoder_lora_policy(
    model: nn.Module,
    *,
    config: DecoderLoRAConfig | None = None,
) -> Qwen3DecoderLoRABuild:
    """Freeze Qwen3, attach only the accepted decoder LoRA, and audit it.

    PEFT is imported lazily because the contract-only package remains usable
    without the accepted Qwen/veRL environment.  The returned PEFT model owns
    exactly one adapter named ``default``.  This function constructs no
    optimizer and selects no precision or distributed execution policy.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("Qwen3 LoRA construction requires a torch module")
    selected = config or DecoderLoRAConfig()
    if not isinstance(selected, DecoderLoRAConfig):
        raise TypeError("config must be DecoderLoRAConfig")
    if getattr(model, "peft_config", None):
        raise ValueError("Qwen3 policy already contains a PEFT adapter")
    if any(".lora_" in name for name, _ in model.named_parameters()):
        raise ValueError("Qwen3 policy already contains LoRA parameters")

    targets = resolve_qwen3_decoder_lora_targets(model, selected)
    original_training = model.training
    model.requires_grad_(False)
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:  # pragma: no cover - accepted runtime has PEFT
        raise RuntimeError(
            "Qwen3 decoder-LoRA construction requires the accepted PEFT package"
        ) from error

    peft_config = LoraConfig(
        task_type=None,
        inference_mode=False,
        r=selected.rank,
        target_modules=selected.target_modules,
        exclude_modules=selected.exclude_modules,
        lora_alpha=selected.alpha,
        lora_dropout=selected.dropout,
        fan_in_fan_out=False,
        bias="none",
        use_rslora=False,
        modules_to_save=None,
        init_lora_weights=True,
        use_dora=False,
    )
    policy = get_peft_model(
        model,
        peft_config,
        adapter_name="default",
        # The accepted veRL FSDP2 actor casts trainable LoRA tensors to the
        # base dtype before publishing them to vLLM.  Preserve that dtype here
        # so rollout workers can install and prove the exact BF16 snapshot.
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=False,
    )
    policy.train(original_training)
    audit = audit_policy_model_scope(policy, config=selected)
    return Qwen3DecoderLoRABuild(
        model=policy,
        config=selected,
        target_modules=targets,
        scope_audit=audit,
    )


def freeze_qwen3_reference_model(model: nn.Module) -> ModelScopeAudit:
    """Put an unadapted Qwen3 reference in frozen eval state and audit it."""

    if not isinstance(model, nn.Module):
        raise TypeError("Qwen3 reference must be a torch module")
    if getattr(model, "peft_config", None):
        raise ValueError("the frozen Qwen3 reference must not contain PEFT state")
    model.requires_grad_(False)
    model.eval()
    return audit_reference_model_scope(model)


@dataclass(frozen=True, slots=True)
class Qwen3RoleReplay:
    """One role's exact response-aligned replay result."""

    role: ComponentRole
    bundle_sha256: str
    response_token_ids: tuple[int, ...]
    response_ownership: tuple[TokenOwnership, ...]
    policy_sampled_mask: torch.Tensor
    logprobs: torch.Tensor
    state_proof: RecordedPolicyForwardStateProof

    def __post_init__(self) -> None:
        if self.role not in {ComponentRole.CURRENT, ComponentRole.REFERENCE}:
            raise ValueError("Qwen3 role replay supports current/reference only")
        object.__setattr__(self, "response_token_ids", tuple(self.response_token_ids))
        object.__setattr__(self, "response_ownership", tuple(self.response_ownership))
        width = len(self.response_token_ids)
        if len(self.response_ownership) != width:
            raise ValueError("response IDs and ownership must align")
        if self.policy_sampled_mask.shape != (width,) or (
            self.policy_sampled_mask.dtype is not torch.bool
        ):
            raise ValueError("policy-sampled mask must be bool [response]")
        if self.logprobs.shape != (width,) or not self.logprobs.dtype.is_floating_point:
            raise ValueError("role logprobs must be floating [response]")
        if self.logprobs.device != self.policy_sampled_mask.device:
            raise ValueError("role logprobs and ownership mask must share a device")
        expected_mask = torch.tensor(
            tuple(
                owner is TokenOwnership.POLICY_SAMPLED
                for owner in self.response_ownership
            ),
            dtype=torch.bool,
            device=self.policy_sampled_mask.device,
        )
        if not torch.equal(self.policy_sampled_mask, expected_mask):
            raise ReplayMismatchError("role replay changed the exact ownership mask")
        if not bool(self.policy_sampled_mask.any().item()):
            raise ValueError("role replay requires at least one policy-sampled token")
        if not bool(torch.isfinite(self.logprobs.detach()).all().item()):
            raise ValueError("role replay logprobs must be finite")
        if bool(torch.count_nonzero(self.logprobs[~self.policy_sampled_mask]).item()):
            raise ReplayMismatchError(
                "non-policy response tokens carry replay logprobs"
            )
        if self.role is ComponentRole.CURRENT and not self.logprobs.requires_grad:
            raise ValueError(
                "current Qwen3 response logprobs lost their autograd graph"
            )
        if self.role is ComponentRole.REFERENCE and self.logprobs.requires_grad:
            raise ValueError("frozen Qwen3 reference logprobs carry gradients")


@dataclass(frozen=True, slots=True)
class Qwen3CurrentReferenceReplay:
    """Current/reference proof over one bundle and one ownership mask."""

    current: Qwen3RoleReplay
    reference: Qwen3RoleReplay

    def __post_init__(self) -> None:
        if self.current.role is not ComponentRole.CURRENT:
            raise ValueError("current replay result has the wrong role")
        if self.reference.role is not ComponentRole.REFERENCE:
            raise ValueError("reference replay result has the wrong role")
        if self.current.bundle_sha256 != self.reference.bundle_sha256:
            raise ReplayMismatchError("current/reference used different replay bundles")
        if (
            self.current.response_token_ids != self.reference.response_token_ids
            or self.current.response_ownership != self.reference.response_ownership
            or not torch.equal(
                self.current.policy_sampled_mask.to(device="cpu"),
                self.reference.policy_sampled_mask.to(device="cpu"),
            )
        ):
            raise ReplayMismatchError(
                "current/reference response tokens or ownership masks differ"
            )


class Qwen3RecordedPolicyForwardPort:
    """Role-bound concrete ``RecordedPolicyForwardPort`` for Qwen3-VL.

    ``forward_recorded`` preserves compatibility with the neutral materializer.
    ``replay_response_logprobs`` is the bundle-native boundary intended for a
    future worker: it calls :meth:`Qwen3VLAdapter.forward_replay_bundle` so every
    tensor is rehydrated and checksum-verified at the consumer boundary.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        binding: RecordedPolicyForwardBinding,
        lora_config: DecoderLoRAConfig | None = None,
        family_adapter: Qwen3VLAdapter | None = None,
        selected_logprob_materializer: Any | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("Qwen3 replay model must be a torch module")
        if not isinstance(binding, RecordedPolicyForwardBinding):
            raise TypeError("Qwen3 replay requires RecordedPolicyForwardBinding")
        if binding.role not in {ComponentRole.CURRENT, ComponentRole.REFERENCE}:
            raise ValueError("concrete Qwen3 replay supports current/reference roles")
        if binding.model.family != POLICY_PILOT_V1_MODEL_FAMILY:
            raise IdentityMismatchError("Qwen3 replay binding has another model family")
        selected = lora_config or DecoderLoRAConfig()
        if not isinstance(selected, DecoderLoRAConfig):
            raise TypeError("lora_config must be DecoderLoRAConfig")
        adapter = family_adapter or Qwen3VLAdapter()
        if not isinstance(adapter, Qwen3VLAdapter):
            raise TypeError("family_adapter must be Qwen3VLAdapter")
        if selected_logprob_materializer is not None and not callable(
            selected_logprob_materializer
        ):
            raise TypeError("selected_logprob_materializer must be callable")

        if binding.role is ComponentRole.CURRENT:
            scope_audit = audit_policy_model_scope(model, config=selected)
        else:
            scope_audit = audit_reference_model_scope(model)
        self.model = model
        self.binding = binding
        self.lora_config = selected
        self.family_adapter = adapter
        self.scope_audit = scope_audit
        self.selected_logprob_materializer = selected_logprob_materializer
        self.materializes_fused_kernels = selected_logprob_materializer is not None
        self._forward_model = _unwrap_peft_model(model)
        self._expected_trainable_names = tuple(
            sorted(
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        )
        self.capture_state_proof()

    def capture_state_proof(self) -> RecordedPolicyForwardStateProof:
        actual_training = bool(self.model.training)
        if actual_training is not self.binding.model_training:
            raise IdentityMismatchError(
                f"{self.binding.role.value} Qwen3 train/eval state differs from binding"
            )
        actual_trainable = tuple(
            sorted(
                name
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            )
        )
        if actual_trainable != self._expected_trainable_names:
            raise IdentityMismatchError(
                f"{self.binding.role.value} Qwen3 trainable scope changed after audit"
            )
        compute_dtype = _embedding_compute_dtype(self._forward_model)
        if compute_dtype != self.binding.compute_dtype:
            raise IdentityMismatchError(
                f"{self.binding.role.value} Qwen3 compute dtype differs from binding"
            )
        return RecordedPolicyForwardStateProof(
            role=self.binding.role,
            model=self.binding.model,
            policy_version=self.binding.policy_version,
            base_weights_sha256=self.binding.base_weights_sha256,
            lora_state_sha256=self.binding.lora_state_sha256,
            model_training=actual_training,
            compute_dtype=compute_dtype,
            autocast_enabled=self.binding.autocast_enabled,
            autocast_dtype=self.binding.autocast_dtype,
            attention_backend=self.binding.attention_backend,
            forward_implementation_sha256=self.binding.forward_implementation_sha256,
        )

    def forward_recorded(
        self, request: RecordedReplayRequest
    ) -> RecordedPolicyForwardOutput:
        """Forward one already rehydrated request without raw images or D rebuild."""

        if not isinstance(request, RecordedReplayRequest):
            raise TypeError("Qwen3 recorded replay requires RecordedReplayRequest")
        expected_consumer = self._consumer
        if request.consumer is not expected_consumer:
            raise ReplayMismatchError(
                f"{self.binding.role.value} received another role's attention contract"
            )
        validate_replay_request(request)
        before = self.capture_state_proof()
        with self._gradient_context(), self._autocast_context():
            result = self.family_adapter.forward_injected(
                self._forward_model,
                injected_request_from_recorded(request),
            )
        self._validate_state_after_forward(before)
        return self._coerce_output(result.logits, request.input_ids)

    def forward_replay_bundle(
        self, bundle: TrajectoryReplayBundle
    ) -> RecordedPolicyForwardOutput:
        """Rehydrate one exact bundle at this role's model boundary and forward it."""

        self._validate_bundle_identity(bundle)
        validate_replay_bundle(bundle)
        before = self.capture_state_proof()
        with self._gradient_context(), self._autocast_context():
            result = self.family_adapter.forward_replay_bundle(
                self._forward_model,
                bundle,
                self._consumer,
            )
        self._validate_state_after_forward(before)
        validate_replay_bundle(bundle)
        store, handle = ObservationStore.from_replay_bundle(bundle)
        request = resolve_replay_request(store, handle, self._consumer)
        return self._coerce_output(result.logits, request.input_ids)

    def replay_response_logprobs(
        self,
        *,
        bundle: TrajectoryReplayBundle,
        prompt_token_ids: tuple[int, ...],
        response: OwnedTokenSequence,
        sampling: SamplingIdentity,
    ) -> Qwen3RoleReplay:
        """Select only exact policy-owned response positions from one role forward."""

        if self.selected_logprob_materializer is not None:
            return self.replay_response_logprobs_batch(
                rows=(
                    SimpleNamespace(
                        bundle=bundle,
                        prompt_token_ids=prompt_token_ids,
                        response=response,
                        sampling=sampling,
                    ),
                )
            )[0]

        if not isinstance(response, OwnedTokenSequence):
            raise TypeError("response must be an OwnedTokenSequence")
        prompt = tuple(prompt_token_ids)
        if not prompt or any(
            type(token_id) is not int or token_id < 0 for token_id in prompt
        ):
            raise ValueError(
                "exact prompt token IDs must be non-empty and non-negative"
            )
        if TokenOwnership.PADDING in response.ownership:
            raise ValueError(
                "the exact unpadded response cannot contain padding ownership"
            )
        if not isinstance(sampling, SamplingIdentity):
            raise TypeError("sampling must be SamplingIdentity")
        if sampling.policy_version != bundle.replay_record.behavior_policy:
            raise IdentityMismatchError(
                "sampling policy version differs from the replay behavior policy"
            )

        store, handle = ObservationStore.from_replay_bundle(bundle)
        request = resolve_replay_request(store, handle, self._consumer)
        validate_replay_request(request)
        exact_ids = tuple(int(token_id) for token_id in request.input_ids[0].tolist())
        expected_ids = prompt + response.token_ids
        if exact_ids != expected_ids:
            raise ReplayMismatchError(
                "prompt/response token IDs differ from the recorded final sequence"
            )
        response_indices = response.policy_indices
        if not response_indices:
            raise ValueError("Qwen3 role replay requires a policy-owned response token")

        output = self.forward_replay_bundle(bundle)
        device = output.logits.device
        sampled_positions = torch.tensor(
            [[len(prompt) + index for index in response_indices]],
            dtype=torch.int64,
            device=device,
        )
        selected = gather_behavior_measure_logprobs(
            output.logits,
            request.input_ids.to(device=device),
            sampled_positions,
            sampling,
        ).squeeze(0)
        scatter_indices = torch.tensor(
            response_indices, dtype=torch.int64, device=device
        )
        response_logprobs = torch.zeros(
            len(response.token_ids), dtype=selected.dtype, device=device
        ).scatter(0, scatter_indices, selected)
        mask = torch.tensor(
            tuple(
                owner is TokenOwnership.POLICY_SAMPLED for owner in response.ownership
            ),
            dtype=torch.bool,
            device=device,
        )
        return Qwen3RoleReplay(
            role=self.binding.role,
            bundle_sha256=bundle.bundle_sha256,
            response_token_ids=response.token_ids,
            response_ownership=response.ownership,
            policy_sampled_mask=mask,
            logprobs=response_logprobs,
            state_proof=self.capture_state_proof(),
        )

    def replay_response_logprobs_batch(
        self,
        *,
        rows: tuple[Any, ...],
    ) -> tuple[Qwen3RoleReplay, ...]:
        """Replay independently verified rows in one Qwen decoder forward."""

        if not rows:
            raise ValueError("Qwen3 replay batch cannot be empty")
        prepared: list[
            tuple[
                TrajectoryReplayBundle,
                tuple[int, ...],
                OwnedTokenSequence,
                SamplingIdentity,
                Any,
                tuple[int, ...],
            ]
        ] = []
        for row in rows:
            bundle = getattr(row, "bundle", None)
            prompt = tuple(getattr(row, "prompt_token_ids", ()))
            response = getattr(row, "response", None)
            sampling = getattr(row, "sampling", None)
            if not isinstance(bundle, TrajectoryReplayBundle):
                raise TypeError("Qwen3 replay batch row lost its bundle")
            if not isinstance(response, OwnedTokenSequence):
                raise TypeError("Qwen3 replay batch row lost its response")
            if not prompt or any(
                type(token_id) is not int or token_id < 0 for token_id in prompt
            ):
                raise ValueError(
                    "exact prompt token IDs must be non-empty and non-negative"
                )
            if TokenOwnership.PADDING in response.ownership:
                raise ValueError(
                    "the exact unpadded response cannot contain padding ownership"
                )
            if not isinstance(sampling, SamplingIdentity):
                raise TypeError("sampling must be SamplingIdentity")
            if sampling.policy_version != bundle.replay_record.behavior_policy:
                raise IdentityMismatchError(
                    "sampling policy version differs from replay behavior policy"
                )
            self._validate_bundle_identity(bundle)
            validate_replay_bundle(bundle)
            store, handle = ObservationStore.from_replay_bundle(bundle)
            request = resolve_replay_request(store, handle, self._consumer)
            validate_replay_request(request)
            exact_ids = tuple(
                int(token_id) for token_id in request.input_ids[0].tolist()
            )
            if exact_ids != prompt + response.token_ids:
                raise ReplayMismatchError(
                    "prompt/response token IDs differ from the recorded final sequence"
                )
            response_indices = response.policy_indices
            if not response_indices:
                raise ValueError(
                    "Qwen3 role replay requires a policy-owned response token"
                )
            prepared.append(
                (bundle, prompt, response, sampling, request, response_indices)
            )

        before = self.capture_state_proof()
        with self._gradient_context(), self._autocast_context():
            hidden_rows = self.family_adapter.forward_injected_hidden_batch(
                self._forward_model,
                tuple(injected_request_from_recorded(item[4]) for item in prepared),
            )
            results: list[Qwen3RoleReplay] = []
            for item, hidden_result in zip(prepared, hidden_rows, strict=True):
                bundle, prompt, response, sampling, request, response_indices = item
                device = hidden_result.hidden_states.device
                sampled_positions = torch.tensor(
                    [[len(prompt) + index for index in response_indices]],
                    dtype=torch.int64,
                    device=device,
                )
                if self.selected_logprob_materializer is None:
                    logits = resolve_lm_head(self._forward_model)(
                        hidden_result.hidden_states
                    )
                    selected = gather_behavior_measure_logprobs(
                        logits,
                        request.input_ids.to(device=device),
                        sampled_positions,
                        sampling,
                    ).squeeze(0)
                else:
                    selected = self.selected_logprob_materializer(
                        hidden_states=hidden_result.hidden_states,
                        lm_head=resolve_lm_head(self._forward_model),
                        token_ids=request.input_ids,
                        sampled_positions=sampled_positions,
                        sampling=sampling,
                    ).squeeze(0)
                scatter_indices = torch.tensor(
                    response_indices, dtype=torch.int64, device=device
                )
                response_logprobs = torch.zeros(
                    len(response.token_ids), dtype=selected.dtype, device=device
                ).scatter(0, scatter_indices, selected)
                mask = torch.tensor(
                    tuple(
                        owner is TokenOwnership.POLICY_SAMPLED
                        for owner in response.ownership
                    ),
                    dtype=torch.bool,
                    device=device,
                )
                results.append(
                    Qwen3RoleReplay(
                        role=self.binding.role,
                        bundle_sha256=bundle.bundle_sha256,
                        response_token_ids=response.token_ids,
                        response_ownership=response.ownership,
                        policy_sampled_mask=mask,
                        logprobs=response_logprobs,
                        state_proof=before,
                    )
                )
        self._validate_state_after_forward(before)
        for bundle, *_ in prepared:
            validate_replay_bundle(bundle)
        return tuple(results)

    @property
    def _consumer(self) -> ReplayConsumer:
        return (
            ReplayConsumer.POLICY
            if self.binding.role is ComponentRole.CURRENT
            else ReplayConsumer.REFERENCE
        )

    def _validate_bundle_identity(self, bundle: TrajectoryReplayBundle) -> None:
        if not isinstance(bundle, TrajectoryReplayBundle):
            raise TypeError("Qwen3 replay requires TrajectoryReplayBundle")
        replay = bundle.replay_record
        if replay.model != self.binding.model:
            raise IdentityMismatchError(
                "Qwen3 replay bundle changed the model identity"
            )
        if not replay.deterministic_forward or replay.adapter_dropout != 0.0:
            raise ReplayMismatchError(
                "Qwen3 replay bundle is not deterministic with adapter dropout zero"
            )
        if self.binding.role is ComponentRole.CURRENT and (
            replay.behavior_policy != self.binding.policy_version
        ):
            raise IdentityMismatchError(
                "zero-staleness current replay differs from the behavior policy"
            )

    def _gradient_context(self):
        return (
            torch.enable_grad()
            if self.binding.role is ComponentRole.CURRENT
            else torch.no_grad()
        )

    def _autocast_context(self):
        if not self.binding.autocast_enabled:
            return nullcontext()
        dtype = _torch_dtype(self.binding.autocast_dtype)
        device_type = _module_device_type(self._forward_model)
        return torch.autocast(device_type=device_type, dtype=dtype)

    def _validate_state_after_forward(
        self, before: RecordedPolicyForwardStateProof
    ) -> None:
        after = self.capture_state_proof()
        if after != before:
            raise IdentityMismatchError(
                f"{self.binding.role.value} Qwen3 execution state drifted during replay"
            )

    def _coerce_output(
        self, logits: torch.Tensor, input_ids: torch.Tensor
    ) -> RecordedPolicyForwardOutput:
        output = RecordedPolicyForwardOutput(logits)
        if output.logits.shape[:2] != input_ids.shape:
            raise ReplayMismatchError(
                f"{self.binding.role.value} Qwen3 logits differ from replay sequence shape"
            )
        if not bool(torch.isfinite(output.logits.detach()).all().item()):
            raise ValueError(f"{self.binding.role.value} Qwen3 logits must be finite")
        if (
            self.binding.role is ComponentRole.CURRENT
            and not output.logits.requires_grad
        ):
            raise ValueError("current Qwen3 logits lost their autograd graph")
        if self.binding.role is ComponentRole.REFERENCE and output.logits.requires_grad:
            raise ValueError("frozen Qwen3 reference logits carry gradients")
        return output


def replay_qwen3_current_reference(
    *,
    current: Qwen3RecordedPolicyForwardPort,
    reference: Qwen3RecordedPolicyForwardPort,
    bundle: TrajectoryReplayBundle,
    prompt_token_ids: tuple[int, ...],
    response: OwnedTokenSequence,
    sampling: SamplingIdentity,
) -> Qwen3CurrentReferenceReplay:
    """Replay current/reference on the same exact bundle and ownership mask."""

    if not isinstance(current, Qwen3RecordedPolicyForwardPort) or (
        current.binding.role is not ComponentRole.CURRENT
    ):
        raise TypeError("current must be a current-role Qwen3 replay port")
    if not isinstance(reference, Qwen3RecordedPolicyForwardPort) or (
        reference.binding.role is not ComponentRole.REFERENCE
    ):
        raise TypeError("reference must be a reference-role Qwen3 replay port")
    if current.binding.model != reference.binding.model:
        raise IdentityMismatchError("current/reference Qwen3 model identities differ")
    if current.binding.base_weights_sha256 != reference.binding.base_weights_sha256:
        raise IdentityMismatchError("current/reference Qwen3 base weights differ")
    validate_replay_bundle(bundle)
    current_result = current.replay_response_logprobs(
        bundle=bundle,
        prompt_token_ids=prompt_token_ids,
        response=response,
        sampling=sampling,
    )
    reference_result = reference.replay_response_logprobs(
        bundle=bundle,
        prompt_token_ids=prompt_token_ids,
        response=response,
        sampling=sampling,
    )
    validate_replay_bundle(bundle)
    return Qwen3CurrentReferenceReplay(current_result, reference_result)


def _unwrap_peft_model(model: nn.Module) -> nn.Module:
    getter = getattr(model, "get_base_model", None)
    unwrapped = getter() if callable(getter) else model
    if not isinstance(unwrapped, nn.Module):
        raise TypeError("PEFT get_base_model() did not return a torch module")
    return unwrapped


def _embedding_compute_dtype(model: nn.Module) -> str:
    embedding = resolve_language_model(model).get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor) or not weight.dtype.is_floating_point:
        raise TypeError("Qwen3 input embedding must expose floating-point weights")
    return str(weight.dtype).removeprefix("torch.")


def _module_device_type(model: nn.Module) -> str:
    try:
        return next(model.parameters()).device.type
    except StopIteration as error:
        raise ValueError("Qwen3 replay model has no parameters") from error


def _torch_dtype(name: str | None) -> torch.dtype:
    if name is None:
        raise ValueError("enabled autocast requires an explicit dtype")
    value: Any = getattr(torch, name, None)
    if not isinstance(value, torch.dtype) or not value.is_floating_point:
        raise ValueError(f"unsupported Qwen3 autocast dtype {name!r}")
    return value


__all__ = [
    "Qwen3CurrentReferenceReplay",
    "Qwen3DecoderLoRABuild",
    "Qwen3RecordedPolicyForwardPort",
    "Qwen3RoleReplay",
    "build_qwen3_decoder_lora_policy",
    "freeze_qwen3_reference_model",
    "replay_qwen3_current_reference",
]
