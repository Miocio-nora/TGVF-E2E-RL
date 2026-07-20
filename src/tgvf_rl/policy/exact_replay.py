"""Exact policy-role replay from rollout-materialized evidence only.

This is a framework-neutral CPU contract boundary.  It accepts the replay
bundles already minted by rollout, rehydrates their content-addressed tensors,
and gives three role-specific forward ports only verified recorded tensors.  It
has no source-image processor, vision encoder, target-conditioning provider, or
TGVF Adapter dependency, so replay cannot regenerate ``D`` or an observation.

The concrete Qwen/FSDP2 worker assembly remains a separate integration.  A
worker adapter can implement :class:`RecordedPolicyForwardPort` around the
family adapter's recorded-input forward without changing this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from types import SimpleNamespace
from typing import Protocol

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import (
    ComponentRole,
    ModelIdentity,
    PolicyVersion,
    _validate_sha256,
)
from tgvf_rl.contracts.tokens import SamplingIdentity, TokenOwnership
from tgvf_rl.framework.verl.data_bridge import (
    DataProtoPayload,
    validate_data_proto_integrity,
)
from tgvf_rl.framework.verl.rollout_bridge import (
    EXACT_PROMPT_IDS_FIELD,
    EXACT_RESPONSE_IDS_FIELD,
)
from tgvf_rl.objectives import (
    LogProbSource,
    PolicyLogProbSet,
    RoleLogProbs,
)
from tgvf_rl.observations.store import (
    ObservationStore,
    TrajectoryReplayBundle,
    record_checksum,
    replay_checksum,
    tensor_checksum,
)
from tgvf_rl.qwen.base import (
    RecordedReplayRequest,
    ReplayConsumer,
    gather_behavior_measure_logprobs,
    resolve_replay_request,
    validate_replay_request,
)

from .runtime import PolicyReplayMaterialization


class ReplayParameterization(str, Enum):
    """The only two policy parameterizations used by Policy Pilot replay."""

    BASE_PLUS_LORA = "base_plus_lora"
    FROZEN_BASE = "frozen_base"


@dataclass(frozen=True, slots=True)
class RecordedPolicyForwardStateProof:
    """State captured from the actual forward executor immediately around a call."""

    role: ComponentRole
    model: ModelIdentity
    policy_version: PolicyVersion
    base_weights_sha256: str
    lora_state_sha256: str | None
    model_training: bool
    compute_dtype: str
    autocast_enabled: bool
    autocast_dtype: str | None
    attention_backend: str
    forward_implementation_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {
            ComponentRole.PROXIMAL_OLD,
            ComponentRole.CURRENT,
            ComponentRole.REFERENCE,
        }:
            raise ValueError("forward state proof has an unsupported policy role")
        if not isinstance(self.model, ModelIdentity):
            raise TypeError("forward state proof requires a ModelIdentity")
        if not isinstance(self.policy_version, PolicyVersion):
            raise TypeError("forward state proof requires a PolicyVersion")
        _validate_sha256(self.base_weights_sha256)
        if self.lora_state_sha256 is not None:
            _validate_sha256(self.lora_state_sha256)
        if not isinstance(self.model_training, bool):
            raise TypeError("forward state proof model_training must be bool")
        _validate_runtime_name(self.compute_dtype, "compute dtype")
        if not isinstance(self.autocast_enabled, bool):
            raise TypeError("forward state proof autocast_enabled must be bool")
        if self.autocast_enabled:
            _validate_runtime_name(self.autocast_dtype, "autocast dtype")
        elif self.autocast_dtype is not None:
            raise ValueError("disabled autocast cannot carry an autocast dtype")
        _validate_runtime_name(self.attention_backend, "attention backend")
        _validate_sha256(self.forward_implementation_sha256)

    @property
    def identity_sha256(self) -> str:
        return _json_sha256(
            {
                "schema": "recorded-policy-forward-state-v1",
                "role": self.role.value,
                "model": {
                    "family": self.model.family,
                    "model_name": self.model.model_name,
                    "revision_or_path": self.model.revision_or_path,
                    "tokenizer_length": self.model.tokenizer_length,
                    "chat_template_sha256": self.model.chat_template_sha256,
                },
                "policy_version": {
                    "run_id": self.policy_version.run_id,
                    "optimizer_step": self.policy_version.optimizer_step,
                    "weights_sha256": self.policy_version.weights_sha256,
                },
                "base_weights_sha256": self.base_weights_sha256,
                "lora_state_sha256": self.lora_state_sha256,
                "model_training": self.model_training,
                "compute_dtype": self.compute_dtype,
                "autocast_enabled": self.autocast_enabled,
                "autocast_dtype": self.autocast_dtype,
                "attention_backend": self.attention_backend,
                "forward_implementation_sha256": (self.forward_implementation_sha256),
            }
        )


@dataclass(frozen=True, slots=True)
class RecordedPolicyForwardBinding:
    """Static identity and determinism proof for one injected forward port."""

    role: ComponentRole
    model: ModelIdentity
    policy_version: PolicyVersion
    parameterization: ReplayParameterization
    base_weights_sha256: str
    lora_state_sha256: str | None
    parameters_frozen: bool
    deterministic_forward: bool
    lora_dropout: float
    model_training: bool
    compute_dtype: str
    autocast_enabled: bool
    autocast_dtype: str | None
    attention_backend: str
    forward_implementation_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {
            ComponentRole.PROXIMAL_OLD,
            ComponentRole.CURRENT,
            ComponentRole.REFERENCE,
        }:
            raise ValueError(
                "recorded replay binding must be proximal_old, current, or reference"
            )
        if not isinstance(self.model, ModelIdentity):
            raise TypeError("recorded replay binding requires a ModelIdentity")
        if not isinstance(self.policy_version, PolicyVersion):
            raise TypeError("recorded replay binding requires a PolicyVersion")
        if not isinstance(self.parameterization, ReplayParameterization):
            raise TypeError("parameterization must be ReplayParameterization")
        _validate_sha256(self.base_weights_sha256)
        if not isinstance(self.parameters_frozen, bool):
            raise TypeError("parameters_frozen must be bool")
        if not isinstance(self.deterministic_forward, bool):
            raise TypeError("deterministic_forward must be bool")
        if not isinstance(self.lora_dropout, (int, float)) or isinstance(
            self.lora_dropout, bool
        ):
            raise TypeError("lora_dropout must be a real number")
        if not self.deterministic_forward or float(self.lora_dropout) != 0.0:
            raise ValueError(
                "Policy Pilot replay requires deterministic forward and LoRA dropout 0"
            )
        if not isinstance(self.model_training, bool):
            raise TypeError("model_training must be bool")
        _validate_runtime_name(self.compute_dtype, "compute dtype")
        if not isinstance(self.autocast_enabled, bool):
            raise TypeError("autocast_enabled must be bool")
        if self.autocast_enabled:
            _validate_runtime_name(self.autocast_dtype, "autocast dtype")
        elif self.autocast_dtype is not None:
            raise ValueError("disabled autocast cannot carry an autocast dtype")
        _validate_runtime_name(self.attention_backend, "attention backend")
        _validate_sha256(self.forward_implementation_sha256)

        if self.role in {ComponentRole.CURRENT, ComponentRole.PROXIMAL_OLD}:
            if self.parameterization is not ReplayParameterization.BASE_PLUS_LORA:
                raise ValueError("current/proximal replay must use base_plus_lora")
            if self.role is ComponentRole.CURRENT and self.parameters_frozen:
                raise ValueError("current LoRA replay cannot freeze every parameter")
            if self.role is ComponentRole.PROXIMAL_OLD and not self.parameters_frozen:
                raise ValueError(
                    "proximal-old behavior snapshot must be frozen during replay"
                )
            if self.lora_state_sha256 is None:
                raise ValueError(
                    "current/proximal replay requires an exact LoRA state identity"
                )
            _validate_sha256(self.lora_state_sha256)
        else:
            if self.parameterization is not ReplayParameterization.FROZEN_BASE:
                raise ValueError("reference replay must use the frozen base")
            if not self.parameters_frozen:
                raise ValueError("reference replay parameters must be frozen")
            if self.lora_state_sha256 is not None:
                raise ValueError("reference replay must not load LoRA")
            if self.policy_version.weights_sha256 != self.base_weights_sha256:
                raise IdentityMismatchError(
                    "reference policy weights must identify the frozen base weights"
                )

    def expected_state_proof(self) -> RecordedPolicyForwardStateProof:
        return RecordedPolicyForwardStateProof(
            role=self.role,
            model=self.model,
            policy_version=self.policy_version,
            base_weights_sha256=self.base_weights_sha256,
            lora_state_sha256=self.lora_state_sha256,
            model_training=self.model_training,
            compute_dtype=self.compute_dtype,
            autocast_enabled=self.autocast_enabled,
            autocast_dtype=self.autocast_dtype,
            attention_backend=self.attention_backend,
            forward_implementation_sha256=self.forward_implementation_sha256,
        )


@dataclass(frozen=True, slots=True)
class RecordedPolicyForwardOutput:
    """Logits produced from one exact :class:`RecordedReplayRequest`."""

    logits: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor):
            raise TypeError("recorded policy forward logits must be a tensor")
        if self.logits.ndim != 3:
            raise ValueError("recorded policy forward logits must have shape [B,S,V]")
        if not self.logits.dtype.is_floating_point:
            raise TypeError("recorded policy forward logits must be floating point")


class RecordedPolicyForwardPort(Protocol):
    """A role-bound forward which receives recorded tensors, never raw images."""

    binding: RecordedPolicyForwardBinding

    def capture_state_proof(self) -> RecordedPolicyForwardStateProof: ...

    def forward_recorded(
        self, request: RecordedReplayRequest
    ) -> RecordedPolicyForwardOutput: ...


@dataclass(frozen=True, slots=True)
class ExactPolicyReplayMaterialization(PolicyReplayMaterialization):
    """Runtime-compatible result plus per-trajectory evidence identities."""

    exact_replay_evidence_sha256s: tuple[str, ...]
    proximal_old_replay_bundle_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        super(ExactPolicyReplayMaterialization, self).__post_init__()
        object.__setattr__(
            self,
            "exact_replay_evidence_sha256s",
            tuple(self.exact_replay_evidence_sha256s),
        )
        object.__setattr__(
            self,
            "proximal_old_replay_bundle_sha256s",
            tuple(self.proximal_old_replay_bundle_sha256s),
        )
        if len(self.exact_replay_evidence_sha256s) != len(
            self.policy_replay_bundle_sha256s
        ):
            raise ValueError("replay evidence identities must align with bundles")
        if self.proximal_old_replay_bundle_sha256s != (
            self.policy_replay_bundle_sha256s
        ):
            raise ReplayMismatchError(
                "proximal-old replay did not consume the exact rollout bundles"
            )
        for digest in self.exact_replay_evidence_sha256s:
            _validate_sha256(digest)


class ExactPolicyReplayMaterializer:
    """Materialize independent policy-role logprobs without recomputing D."""

    def __init__(
        self,
        *,
        current_forward: RecordedPolicyForwardPort,
        proximal_old_forward: RecordedPolicyForwardPort,
        reference_forward: RecordedPolicyForwardPort,
    ) -> None:
        self.current_forward = _require_forward_port(
            current_forward, ComponentRole.CURRENT
        )
        self.proximal_old_forward = _require_forward_port(
            proximal_old_forward, ComponentRole.PROXIMAL_OLD
        )
        self.reference_forward = _require_forward_port(
            reference_forward, ComponentRole.REFERENCE
        )
        current = self.current_forward.binding
        proximal_old = self.proximal_old_forward.binding
        reference = self.reference_forward.binding
        if len({current.model, proximal_old.model, reference.model}) != 1:
            raise IdentityMismatchError(
                "current/proximal/reference replay must bind one base model identity"
            )
        if (
            len(
                {
                    current.base_weights_sha256,
                    proximal_old.base_weights_sha256,
                    reference.base_weights_sha256,
                }
            )
            != 1
        ):
            raise IdentityMismatchError(
                "current/proximal/reference replay must share exact base weights"
            )
        if (
            current.policy_version != proximal_old.policy_version
            or current.lora_state_sha256 != proximal_old.lora_state_sha256
        ):
            raise IdentityMismatchError(
                "current and proximal-old must bind the same behavior LoRA snapshot"
            )

    def materialize(
        self, payload: DataProtoPayload
    ) -> ExactPolicyReplayMaterialization:
        """Replay one transported batch and return the neutral GRPO role set."""

        if not isinstance(payload, DataProtoPayload):
            raise TypeError("exact replay requires a DataProtoPayload")
        payload.assert_sidecars_available()
        if any(tensor.device.type != "cpu" for tensor in payload.tensor_batch.values()):
            raise ValueError("the contract-level exact replay materializer is CPU-only")

        integrity = validate_data_proto_integrity(
            SimpleNamespace(
                batch=payload.tensor_batch,
                non_tensor_batch=payload.non_tensor_batch,
            )
        )
        bundles = integrity.replay_bundles
        if not bundles:
            raise ValueError("exact replay requires at least one rollout bundle")

        behavior_version = _single_behavior_version(bundles)
        current_binding = self.current_forward.binding
        proximal_old_binding = self.proximal_old_forward.binding
        reference_binding = self.reference_forward.binding
        if current_binding.policy_version != behavior_version:
            raise IdentityMismatchError(
                "zero-staleness current replay differs from the behavior policy version"
            )
        if current_binding.model != bundles[0].replay_record.model:
            raise IdentityMismatchError(
                "replay forward model identity differs from rollout bundles"
            )

        sampling = _single_sampling_identity(integrity.behavior_trace_records)
        transform_sha256 = sampling.transform_identity_sha256
        response_width = int(payload.tensor_batch["responses"].shape[1])
        current_rows: list[torch.Tensor] = []
        proximal_old_rows: list[torch.Tensor] = []
        reference_rows: list[torch.Tensor] = []
        evidence_sha256s: list[str] = []

        prompt_rows = tuple(payload.non_tensor_batch[EXACT_PROMPT_IDS_FIELD])
        response_rows = tuple(payload.non_tensor_batch[EXACT_RESPONSE_IDS_FIELD])
        for row_index, bundle in enumerate(bundles):
            require_single_sequence_replay_bundle(bundle)
            prompt_ids = tuple(int(value) for value in prompt_rows[row_index])
            response_ids = tuple(int(value) for value in response_rows[row_index])
            policy_response_indices = tuple(
                index
                for index, owner in enumerate(
                    integrity.response_token_ownership[row_index][: len(response_ids)]
                )
                if owner is TokenOwnership.POLICY_SAMPLED
            )
            if not policy_response_indices:
                raise ReplayMismatchError(
                    "every exact replay row must contain a policy-sampled token"
                )

            store, handle = ObservationStore.from_replay_bundle(bundle)
            replay = store.resolve_replay(handle)
            if handle != integrity.replay_handles[row_index]:
                raise ReplayMismatchError(
                    "DataProto replay handle differs from transported bundle"
                )
            if replay.trajectory_id != integrity.trajectory_ids[row_index]:
                raise IdentityMismatchError(
                    "replay trajectory identity differs from rollout provenance"
                )
            if replay.model != current_binding.model:
                raise IdentityMismatchError(
                    "one replay bundle changed the bound model identity"
                )
            if replay.behavior_policy != behavior_version:
                raise IdentityMismatchError(
                    "one replay bundle changed the behavior policy version"
                )
            if not replay.deterministic_forward or replay.adapter_dropout != 0.0:
                raise ReplayMismatchError(
                    "rollout replay state is not deterministic with dropout zero"
                )

            current_request = resolve_replay_request(
                store, handle, ReplayConsumer.POLICY
            )
            proximal_old_request = resolve_replay_request(
                store, handle, ReplayConsumer.POLICY
            )
            reference_request = resolve_replay_request(
                store, handle, ReplayConsumer.REFERENCE
            )
            validate_replay_request(current_request)
            validate_replay_request(proximal_old_request)
            validate_replay_request(reference_request)
            _validate_final_tokens(
                current_request,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
            )
            _validate_final_tokens(
                proximal_old_request,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
            )
            _validate_final_tokens(
                reference_request,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
            )

            current_state_sha256 = _request_state_sha256(current_request)
            proximal_old_state_sha256 = _request_state_sha256(proximal_old_request)
            reference_state_sha256 = _request_state_sha256(reference_request)
            if (
                len(
                    {
                        current_state_sha256,
                        proximal_old_state_sha256,
                        reference_state_sha256,
                    }
                )
                != 1
            ):
                raise ReplayMismatchError(
                    "current/proximal/reference replay state differs in tokens, masks, "
                    "positions, source visual, D, DeepStack, layout, or cache"
                )

            current_output = _execute_verified_forward(
                self.current_forward,
                current_request,
                request_state_sha256=current_state_sha256,
                gradients_enabled=True,
            )
            proximal_old_output = _execute_verified_forward(
                self.proximal_old_forward,
                proximal_old_request,
                request_state_sha256=proximal_old_state_sha256,
                gradients_enabled=False,
            )
            reference_output = _execute_verified_forward(
                self.reference_forward,
                reference_request,
                request_state_sha256=reference_state_sha256,
                gradients_enabled=False,
            )

            _validate_forward_logits(
                current_output.logits,
                current_request,
                role=ComponentRole.CURRENT,
            )
            _validate_forward_logits(
                proximal_old_output.logits,
                proximal_old_request,
                role=ComponentRole.PROXIMAL_OLD,
            )
            _validate_forward_logits(
                reference_output.logits,
                reference_request,
                role=ComponentRole.REFERENCE,
            )
            if (
                len(
                    {
                        current_output.logits.shape[-1],
                        proximal_old_output.logits.shape[-1],
                        reference_output.logits.shape[-1],
                    }
                )
                != 1
            ):
                raise ReplayMismatchError(
                    "current/proximal/reference replay vocabulary dimensions differ"
                )

            sampled_positions = torch.tensor(
                [[len(prompt_ids) + index for index in policy_response_indices]],
                dtype=torch.int64,
            )
            current_selected = gather_behavior_measure_logprobs(
                current_output.logits,
                current_request.input_ids,
                sampled_positions,
                sampling,
            ).squeeze(0)
            proximal_old_selected = gather_behavior_measure_logprobs(
                proximal_old_output.logits,
                proximal_old_request.input_ids,
                sampled_positions,
                sampling,
            ).squeeze(0)
            if not torch.equal(current_selected, proximal_old_selected):
                raise ReplayMismatchError(
                    "zero-staleness current and proximal-old replay logprobs differ "
                    "on policy-sampled tokens"
                )
            reference_selected = gather_behavior_measure_logprobs(
                reference_output.logits,
                reference_request.input_ids,
                sampled_positions,
                sampling,
            ).squeeze(0)
            scatter_indices = torch.tensor(policy_response_indices, dtype=torch.int64)
            current_rows.append(
                torch.zeros(response_width, dtype=current_selected.dtype).scatter(
                    0, scatter_indices, current_selected
                )
            )
            proximal_old_rows.append(
                torch.zeros(response_width, dtype=proximal_old_selected.dtype).scatter(
                    0, scatter_indices, proximal_old_selected
                )
            )
            reference_rows.append(
                torch.zeros(response_width, dtype=reference_selected.dtype).scatter(
                    0, scatter_indices, reference_selected
                )
            )
            evidence_sha256s.append(_exact_replay_evidence_sha256(bundle))

        current_values = torch.stack(current_rows)
        proximal_old_values = torch.stack(proximal_old_rows)
        reference_values = torch.stack(reference_rows)
        if not current_values.requires_grad:
            raise ValueError(
                "current replay log probabilities lost their autograd graph"
            )
        if reference_values.requires_grad:
            raise ValueError(
                "frozen reference replay log probabilities carry gradients"
            )
        if proximal_old_values.requires_grad:
            raise ValueError("proximal-old replay log probabilities carry gradients")

        behavior_values = (
            payload.tensor_batch["rollout_log_probs"]
            .to(dtype=current_values.dtype)
            .detach()
            .clone()
        )
        policy_mask = payload.tensor_batch["response_mask"].to(dtype=torch.bool)
        bundle_sha256s = tuple(bundle.bundle_sha256 for bundle in bundles)
        return ExactPolicyReplayMaterialization(
            logprobs=PolicyLogProbSet(
                behavior=RoleLogProbs(
                    role=ComponentRole.BEHAVIOR,
                    values=behavior_values,
                    policy_version=behavior_version,
                    source=LogProbSource.ROLLOUT_RECORDED,
                    sampling_transform_sha256=transform_sha256,
                ),
                proximal_old=RoleLogProbs(
                    role=ComponentRole.PROXIMAL_OLD,
                    values=proximal_old_values,
                    policy_version=proximal_old_binding.policy_version,
                    source=LogProbSource.DETERMINISTIC_REPLAY,
                    sampling_transform_sha256=transform_sha256,
                ),
                current=RoleLogProbs(
                    role=ComponentRole.CURRENT,
                    values=current_values,
                    policy_version=current_binding.policy_version,
                    source=LogProbSource.DETERMINISTIC_REPLAY,
                    sampling_transform_sha256=transform_sha256,
                ),
                reference=RoleLogProbs(
                    role=ComponentRole.REFERENCE,
                    values=reference_values,
                    policy_version=reference_binding.policy_version,
                    source=LogProbSource.DETERMINISTIC_REPLAY,
                    sampling_transform_sha256=transform_sha256,
                ),
                policy_sampled_mask=policy_mask,
            ),
            policy_replay_bundle_sha256s=bundle_sha256s,
            reference_replay_bundle_sha256s=bundle_sha256s,
            exact_replay_evidence_sha256s=tuple(evidence_sha256s),
            proximal_old_replay_bundle_sha256s=bundle_sha256s,
        )


def require_single_sequence_replay_bundle(
    bundle: TrajectoryReplayBundle,
) -> None:
    """Reject batched replay evidence: one bundle is exactly one trajectory."""

    if not isinstance(bundle, TrajectoryReplayBundle):
        raise TypeError("replay evidence must be a TrajectoryReplayBundle")
    input_shape = bundle.replay_record.tensors.input_ids.descriptor.shape
    if len(input_shape) != 2 or input_shape[0] != 1:
        raise ReplayMismatchError(
            "one trajectory replay bundle must contain exactly one sequence (B=1)"
        )


def _execute_verified_forward(
    port: RecordedPolicyForwardPort,
    request: RecordedReplayRequest,
    *,
    request_state_sha256: str,
    gradients_enabled: bool,
) -> RecordedPolicyForwardOutput:
    binding = port.binding
    before = port.capture_state_proof()
    _validate_forward_state_proof(binding, before)
    with torch.set_grad_enabled(gradients_enabled):
        output = port.forward_recorded(request)
    if not isinstance(output, RecordedPolicyForwardOutput):
        raise TypeError(
            f"{binding.role.value} forward must return RecordedPolicyForwardOutput"
        )
    after = port.capture_state_proof()
    if after != before:
        raise IdentityMismatchError(
            f"{binding.role.value} forward execution state drifted during replay"
        )
    _validate_forward_state_proof(binding, after)
    if _request_state_sha256(request) != request_state_sha256:
        raise ReplayMismatchError(
            f"{binding.role.value} forward modified exact rollout-recorded replay tensors"
        )
    return output


def _validate_forward_state_proof(
    binding: RecordedPolicyForwardBinding,
    proof: RecordedPolicyForwardStateProof,
) -> None:
    if not isinstance(proof, RecordedPolicyForwardStateProof):
        raise TypeError("forward port must return RecordedPolicyForwardStateProof")
    expected = binding.expected_state_proof()
    if proof != expected:
        raise IdentityMismatchError(
            f"{binding.role.value} actual forward state differs from its binding"
        )


def _require_forward_port(
    port: RecordedPolicyForwardPort, role: ComponentRole
) -> RecordedPolicyForwardPort:
    binding = getattr(port, "binding", None)
    if not isinstance(binding, RecordedPolicyForwardBinding):
        raise TypeError("recorded policy forward port must expose an exact binding")
    if binding.role is not role:
        raise ValueError(f"expected {role.value} replay forward binding")
    if not callable(getattr(port, "capture_state_proof", None)):
        raise TypeError(
            "recorded policy forward port must implement capture_state_proof()"
        )
    if not callable(getattr(port, "forward_recorded", None)):
        raise TypeError(
            "recorded policy forward port must implement forward_recorded()"
        )
    return port


def _validate_runtime_name(value: object, owner: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{owner} must be an explicit canonical string")


def _single_behavior_version(
    bundles: tuple[TrajectoryReplayBundle, ...],
) -> PolicyVersion:
    versions = {bundle.replay_record.behavior_policy for bundle in bundles}
    if len(versions) != 1:
        raise IdentityMismatchError(
            "one replay batch must come from exactly one behavior policy snapshot"
        )
    models = {bundle.replay_record.model for bundle in bundles}
    if len(models) != 1:
        raise IdentityMismatchError("one replay batch cannot mix model identities")
    return next(iter(versions))


def _single_sampling_identity(
    rows: tuple[tuple[object, ...], ...],
) -> SamplingIdentity:
    samplings: list[SamplingIdentity] = []
    for row in rows:
        for record in row:
            sampling = getattr(getattr(record, "behavior", None), "sampling", None)
            if not isinstance(sampling, SamplingIdentity):
                raise TypeError("behavior replay is missing SamplingIdentity")
            if sampling.asynchronous_staleness_steps != 0:
                raise ReplayMismatchError(
                    "Policy Pilot requires rollout staleness zero"
                )
            samplings.append(sampling)
    if not samplings:
        raise ValueError("exact replay requires behavior sampling evidence")
    transform_sha256s = {sampling.transform_identity_sha256 for sampling in samplings}
    if len(transform_sha256s) != 1:
        raise ReplayMismatchError(
            "one exact replay batch cannot mix sampling probability measures"
        )
    if any(not sampling.has_identity_sampling_transforms for sampling in samplings):
        raise ValueError(
            "policy-role processed-logprob replay is fail-closed until "
            "non-identity vLLM transform parity is accepted"
        )
    return samplings[0]


def _validate_final_tokens(
    request: RecordedReplayRequest,
    *,
    prompt_ids: tuple[int, ...],
    response_ids: tuple[int, ...],
) -> None:
    expected = prompt_ids + response_ids
    actual = tuple(int(value) for value in request.input_ids[0].tolist())
    if actual != expected:
        raise ReplayMismatchError(
            "replay input IDs differ from exact prompt/response rollout bytes"
        )


def _validate_forward_logits(
    logits: torch.Tensor,
    request: RecordedReplayRequest,
    *,
    role: ComponentRole,
) -> None:
    if logits.device.type != "cpu":
        raise ValueError("the contract-level recorded forward must return CPU logits")
    expected_prefix = tuple(request.input_ids.shape)
    if logits.shape[:2] != expected_prefix or logits.shape[-1] <= 0:
        raise ValueError("recorded forward logits differ from exact replay shape")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise ValueError("recorded forward logits must be finite")
    if int(request.input_ids.max().item()) >= logits.shape[-1]:
        raise ValueError("recorded token ID lies outside replay logits vocabulary")
    if role is ComponentRole.CURRENT and not logits.requires_grad:
        raise ValueError("current replay logits must retain gradients")
    if role in {ComponentRole.PROXIMAL_OLD, ComponentRole.REFERENCE} and (
        logits.requires_grad
    ):
        raise ValueError(
            f"frozen {role.value} replay logits must not require gradients"
        )


def _request_state_sha256(request: RecordedReplayRequest) -> str:
    """Hash every exact forward input except the intentional consumer label."""

    payload = {
        "schema": "policy-pilot-recorded-forward-state-v1",
        "replay": {
            "id": request.replay_handle.replay_id,
            "sha256": request.replay_handle.record_sha256,
        },
        "input_ids": _tensor_state_identity(request.input_ids),
        "attention_mask": _tensor_state_identity(request.attention_mask),
        "position_ids": _tensor_state_identity(request.position_ids),
        "token_type_ids": (
            None
            if request.token_type_ids is None
            else _tensor_state_identity(request.token_type_ids)
        ),
        "cache_position": (
            None
            if request.cache_position is None
            else _tensor_state_identity(request.cache_position)
        ),
        "rope_delta": (
            None
            if request.rope_delta is None
            else _tensor_state_identity(request.rope_delta)
        ),
        "use_cache": request.use_cache,
        "visual_blocks": [
            {
                "kind": block.kind,
                "observation": (
                    None
                    if block.observation_handle is None
                    else {
                        "id": block.observation_handle.observation_id,
                        "sha256": block.observation_handle.record_sha256,
                    }
                ),
                "call_index": block.call_index,
                "positions": block.positions,
                "embeddings": _tensor_state_identity(block.embeddings),
                "deepstack_positions": block.deepstack_positions,
                "deepstack": [
                    _tensor_state_identity(value) for value in block.deepstack
                ],
            }
            for block in request.visual_blocks
        ],
    }
    return _json_sha256(payload)


def _tensor_state_identity(tensor: torch.Tensor) -> dict[str, object]:
    """Bind tensor semantics as well as bytes inside one replay-state proof."""

    return {
        "sha256": tensor_checksum(tensor),
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "stride": tuple(tensor.stride()),
    }


def _exact_replay_evidence_sha256(bundle: TrajectoryReplayBundle) -> str:
    """Bind source, every observation, layouts/masks/cache, and tensor bytes."""

    return _json_sha256(
        {
            "schema": "policy-pilot-exact-replay-evidence-v1",
            "bundle_sha256": bundle.bundle_sha256,
            "replay_sha256": replay_checksum(bundle.replay_record),
            "observation_sha256s": [
                record_checksum(record) for record in bundle.observation_records
            ],
            "tensor_sha256s": [item.sha256 for item in bundle.tensor_payloads],
        }
    )


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ExactPolicyReplayMaterialization",
    "ExactPolicyReplayMaterializer",
    "RecordedPolicyForwardBinding",
    "RecordedPolicyForwardOutput",
    "RecordedPolicyForwardPort",
    "RecordedPolicyForwardStateProof",
    "ReplayParameterization",
    "require_single_sequence_replay_bundle",
]
