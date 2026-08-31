"""Executable internal evaluation for a representation-phase artifact.

The runner consumes the same family adapter, frozen Qwen model, TGVF Adapter,
and group-builder interface as representation training.  It retains full
lower-is-better NLL matrices and evaluates every latent observation as one
atomic main-``D`` plus ordered D-DeepStack bundle.

Native counterfactual generation remains family/runtime owned.  It enters this
module through mandatory typed callbacks, which makes the seam executable in
CPU fixtures without pretending that a tiny fixture is a real generation
backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Literal, Protocol

import torch
from torch import nn

from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.qwen.base import (
    CachedTokenForwardRequest,
    InjectedForwardRequest,
    InjectedVisualBlock,
    QwenVLMFamilyAdapter,
)
from tgvf_rl.representation.adapter import TGVFAdapter
from tgvf_rl.representation.deepstack import build_original_image_key_block_mask

from .losses import causal_evidence_losses
from .metrics import (
    AttentionDiagnostics,
    AttentionDiagnosticsSummary,
    NormComparisonDiagnostics,
    QueryMetrics,
    QueryScoreMatrixMetrics,
    ReadoutMetrics,
    ReadoutNLLs,
    ReadoutSampleMetrics,
    RepresentationHealthSummary,
    attention_diagnostics,
    grouped_query_row_metrics,
    grouped_readout_metrics,
    norm_comparison_diagnostics,
    query_score_matrix_metrics,
    readout_sample_metrics,
    summarize_attention_diagnostics,
    summarize_query_score_matrices,
    summarize_readout,
    summarize_representation_health,
)
from .readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
    assert_frozen_deterministic_readout_model,
)
from .schema import RepresentationTrainingSample
from .streaming import StreamingGroupScores, score_streaming_same_image_group
from .trainer import RepresentationGroupBuilder


REPRESENTATION_INTERNAL_EVALUATION_SCHEMA_VERSION = (
    "representation_internal_evaluation_v1"
)
REPRESENTATION_INTERNAL_EVALUATION_ARTIFACT_SCHEMA_VERSION = (
    "representation_internal_evaluation_artifact_v1"
)
DETERMINISTIC_RANDOM_D_ALGORITHM = "shared_feature_permutation_v1"
ATTENTION_TOPK = 5
READOUT_FORWARD_PATH = "family_injected_language_model_no_vision_rerun_v1"
READOUT_MASK_MODE = "causal_post_evidence_original_image_key_block_v1"
READOUT_POSITION_SOURCE = "family_native_group_builder_v1"
READOUT_VISUAL_SWAP_UNIT = "atomic_main_and_all_deepstack_v1"
CAUSAL_VALUE_FLIP_LOG_ODDS_CONTRACT = "summed_teacher_forced_token_logprob_v1"
TARGET_PRESENCE_LOG_ODDS_CONTRACT = "mean_teacher_forced_token_logprob_v1"
_UNSPECIFIED_GROUP_LABEL = "__unspecified__"
_GROUPING_DIMENSIONS = (
    "evidence_type",
    "source_profile",
    "answer_type",
    "visual_difficulty",
)

ContinuationVariant = Literal["value_a", "value_b"]
TargetPresenceVariant = Literal["positive_target", "negative_target"]
ContinuationStopReason = Literal[
    "natural_stop", "length_cap", "malformed", "runtime_error"
]


@dataclass(frozen=True, slots=True)
class RepresentationInternalEvaluationIdentity:
    """Exact artifact, data, prompt, provider, and random-control identity."""

    evaluation_id: str
    model_identity: str
    checkpoint_identity: str
    data_manifest_sha256: str
    prompt_identity: str
    target_conditioning_provider: TargetConditioningProviderKind
    random_seed: int
    schema_version: str = REPRESENTATION_INTERNAL_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "evaluation_id",
            "model_identity",
            "checkpoint_identity",
            "prompt_identity",
        ):
            _non_empty_text(getattr(self, name), name=name)
        _lowercase_sha256(self.data_manifest_sha256, name="data_manifest_sha256")
        if not isinstance(
            self.target_conditioning_provider, TargetConditioningProviderKind
        ):
            raise TypeError("target_conditioning_provider must be an explicit kind")
        _non_negative_int(self.random_seed, name="random_seed")
        if self.schema_version != REPRESENTATION_INTERNAL_EVALUATION_SCHEMA_VERSION:
            raise ValueError("internal-evaluation identity schema mismatch")


@dataclass(frozen=True, slots=True)
class NativeDOnlyContext:
    """Fresh native context containing one focused-D visual block and no image.

    ``source_image_positions`` and ``pre_d_text_kv_reused`` are explicit
    negative assertions.  A callback therefore cannot silently receive the
    original-image block or a cache whose text states already attended to it.
    """

    context_id: str
    transcript_identity: str
    family: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    d_positions: tuple[int, ...]
    image_grid_thw: tuple[tuple[int, int, int], ...]
    source_image_positions: tuple[int, ...] = ()
    pre_d_text_kv_reused: bool = False
    cache_mode: str = "no_cache"

    def __post_init__(self) -> None:
        for name in ("context_id", "transcript_identity", "family"):
            _non_empty_text(getattr(self, name), name=name)
        if not isinstance(self.input_ids, torch.Tensor):
            raise TypeError("native context input_ids must be a tensor")
        if self.input_ids.dtype != torch.long or self.input_ids.ndim != 2:
            raise ValueError("native context input_ids must be long [B,S]")
        if self.input_ids.shape[0] != 1 or self.input_ids.shape[1] == 0:
            raise ValueError("native context currently requires one non-empty row")
        batch, sequence = self.input_ids.shape
        if not isinstance(self.attention_mask, torch.Tensor):
            raise TypeError("native context attention_mask must be a tensor")
        if self.attention_mask.shape not in {
            (batch, sequence),
            (batch, 1, sequence, sequence),
        }:
            raise ValueError("native context attention_mask has an invalid shape")
        if not isinstance(self.position_ids, torch.Tensor) or (
            self.position_ids.ndim not in {2, 3}
            or self.position_ids.shape[-2:] != (batch, sequence)
        ):
            raise ValueError("native context position_ids must end in [B,S]")
        if (
            not isinstance(self.d_positions, tuple)
            or not self.d_positions
            or len(set(self.d_positions)) != len(self.d_positions)
            or any(
                position < 0 or position >= sequence for position in self.d_positions
            )
        ):
            raise ValueError("native context D positions must be unique and in range")
        if (
            not isinstance(self.image_grid_thw, tuple)
            or len(self.image_grid_thw) != 1
            or not isinstance(self.image_grid_thw[0], tuple)
            or len(self.image_grid_thw[0]) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.image_grid_thw[0]
            )
        ):
            raise ValueError(
                "native D-only context requires one positive image_grid_thw row"
            )
        if self.family == "qwen3_vl":
            if self.attention_mask.shape != (batch, sequence) or not bool(
                self.attention_mask.bool().all().item()
            ):
                raise ValueError("Qwen3 D-only context requires an unpadded [1,S] mask")
            if self.position_ids.shape != (3, batch, sequence):
                raise ValueError(
                    "Qwen3 D-only context requires three-axis M-RoPE [3,1,S]"
                )
        if self.source_image_positions:
            raise ValueError("native counterfactual context must remove source images")
        if self.pre_d_text_kv_reused:
            raise ValueError("native counterfactual context cannot reuse pre-D text KV")
        if self.cache_mode != "no_cache":
            raise ValueError("native counterfactual context must use no_cache")


@dataclass(frozen=True, slots=True)
class NativeCounterfactualCase:
    """A local-value pair evaluated by swapping the complete D observation."""

    case_id: str
    pair_identity: str
    sample_a_id: str
    sample_b_id: str
    expected_value_a: str
    expected_value_b: str
    observation_a_identity: str
    observation_b_identity: str
    context: NativeDOnlyContext
    observation_a: RepresentationVisualTensorBundle
    observation_b: RepresentationVisualTensorBundle

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "pair_identity",
            "sample_a_id",
            "sample_b_id",
            "expected_value_a",
            "expected_value_b",
            "observation_a_identity",
            "observation_b_identity",
        ):
            _non_empty_text(getattr(self, name), name=name)
        if self.sample_a_id == self.sample_b_id:
            raise ValueError("counterfactual samples must be distinct")
        if self.expected_value_a == self.expected_value_b:
            raise ValueError("counterfactual expected values must be distinct")
        if not isinstance(self.context, NativeDOnlyContext):
            raise TypeError("counterfactual context must be NativeDOnlyContext")
        if not isinstance(self.observation_a, RepresentationVisualTensorBundle) or not (
            isinstance(self.observation_b, RepresentationVisualTensorBundle)
        ):
            raise TypeError("counterfactual observations must be visual bundles")
        _assert_visual_bundle_contract(
            self.observation_b,
            self.observation_a,
            name="counterfactual observation B",
        )
        if len(self.context.d_positions) != self.observation_a.main.shape[1]:
            raise ValueError("counterfactual D positions differ from D token count")


@dataclass(frozen=True, slots=True)
class NativeCausalValueFlipRequest:
    case: NativeCounterfactualCase

    def __post_init__(self) -> None:
        if not isinstance(self.case, NativeCounterfactualCase):
            raise TypeError("causal value-flip request requires a typed case")


@dataclass(frozen=True, slots=True)
class NativeCausalValueFlipOutput:
    """Callback output: log p(value A) - log p(value B) for both D values."""

    case_id: str
    observation_a_log_odds_a_over_b: float
    observation_b_log_odds_a_over_b: float

    def __post_init__(self) -> None:
        _non_empty_text(self.case_id, name="case_id")
        for name in (
            "observation_a_log_odds_a_over_b",
            "observation_b_log_odds_a_over_b",
        ):
            _finite_float(getattr(self, name), name=name)


class NativeCausalValueFlipEvaluator(Protocol):
    def __call__(
        self, request: NativeCausalValueFlipRequest
    ) -> NativeCausalValueFlipOutput: ...


@dataclass(frozen=True, slots=True)
class NativeBinaryLogOddsRequest:
    """Score two declared continuations under one exact native-D context."""

    case_id: str
    context: NativeDOnlyContext
    observation: RepresentationVisualTensorBundle
    positive_value: str
    negative_value: str

    def __post_init__(self) -> None:
        for name in ("case_id", "positive_value", "negative_value"):
            _non_empty_text(getattr(self, name), name=name)
        if self.positive_value == self.negative_value:
            raise ValueError("binary log-odds values must be distinct")
        if not isinstance(self.context, NativeDOnlyContext):
            raise TypeError("binary log-odds context must be NativeDOnlyContext")
        if not isinstance(self.observation, RepresentationVisualTensorBundle):
            raise TypeError("binary log-odds observation must be a visual bundle")
        if len(self.context.d_positions) != self.observation.main.shape[1]:
            raise ValueError("binary log-odds D positions differ from D token count")


@dataclass(frozen=True, slots=True)
class NativeBinaryLogOddsOutput:
    case_id: str
    log_odds_positive_over_negative: float

    def __post_init__(self) -> None:
        _non_empty_text(self.case_id, name="case_id")
        _finite_float(
            self.log_odds_positive_over_negative,
            name="log_odds_positive_over_negative",
        )


class NativeBinaryLogOddsEvaluator(Protocol):
    def __call__(
        self, request: NativeBinaryLogOddsRequest
    ) -> NativeBinaryLogOddsOutput: ...


@dataclass(frozen=True, slots=True)
class NativeTeacherForcedForward:
    """One fully materialized family-native value-scoring request."""

    request: InjectedForwardRequest
    continuation_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, InjectedForwardRequest):
            raise TypeError("teacher-forced forward requires InjectedForwardRequest")
        sequence = int(self.request.input_ids.shape[1])
        if (
            not isinstance(self.continuation_positions, tuple)
            or not self.continuation_positions
            or tuple(sorted(self.continuation_positions)) != self.continuation_positions
            or len(set(self.continuation_positions)) != len(self.continuation_positions)
            or self.continuation_positions[0] <= 0
            or self.continuation_positions[-1] >= sequence
        ):
            raise ValueError(
                "teacher-forced continuation positions must be unique, ordered, "
                "non-initial, and in range"
            )


@dataclass(frozen=True, slots=True)
class NativeGenerationForward:
    """One fully materialized family-native no-cache generation step."""

    request: InjectedForwardRequest
    next_token_logit_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.request, InjectedForwardRequest):
            raise TypeError("generation forward requires InjectedForwardRequest")
        if (
            isinstance(self.next_token_logit_position, bool)
            or not isinstance(self.next_token_logit_position, int)
            or self.next_token_logit_position < 0
            or self.next_token_logit_position >= self.request.input_ids.shape[1]
        ):
            raise ValueError("next-token logit position lies outside the request")


class NativeInjectedRequestMaterializer(Protocol):
    """Family-native serialization/M-RoPE boundary missing from the base adapter.

    Implementations are expected to live beside the selected native pipeline.
    They own tokenizer IDs and exact position extension; the concrete evaluator
    below owns model execution, scoring, and autoregressive token selection.
    """

    def value_token_ids(
        self, context: NativeDOnlyContext, value: str
    ) -> tuple[int, ...]: ...

    def teacher_forced(
        self,
        *,
        context: NativeDOnlyContext,
        observation: RepresentationVisualTensorBundle,
        continuation_token_ids: tuple[int, ...],
    ) -> NativeTeacherForcedForward: ...

    def generation_step(
        self,
        *,
        context: NativeDOnlyContext,
        observation: RepresentationVisualTensorBundle,
        generated_token_ids: tuple[int, ...],
    ) -> NativeGenerationForward: ...

    def decode_generated(self, token_ids: tuple[int, ...]) -> str: ...

    def extract_expected_value(
        self, generated_text: str, expected_value: str
    ) -> str | None: ...


@dataclass(frozen=True, slots=True)
class NativeFreeContinuationRequest:
    case_id: str
    variant: ContinuationVariant
    expected_value: str
    context: NativeDOnlyContext
    observation_identity: str
    observation: RepresentationVisualTensorBundle

    def __post_init__(self) -> None:
        for name in ("case_id", "expected_value", "observation_identity"):
            _non_empty_text(getattr(self, name), name=name)
        if self.variant not in {"value_a", "value_b"}:
            raise ValueError("unknown continuation variant")
        if not isinstance(self.context, NativeDOnlyContext):
            raise TypeError("free-continuation context must be NativeDOnlyContext")
        if not isinstance(self.observation, RepresentationVisualTensorBundle):
            raise TypeError("free-continuation observation must be a visual bundle")
        if len(self.context.d_positions) != self.observation.main.shape[1]:
            raise ValueError("continuation D positions differ from D token count")


@dataclass(frozen=True, slots=True)
class NativeFreeContinuationOutput:
    case_id: str
    variant: ContinuationVariant
    generated_token_ids: tuple[int, ...]
    generated_text: str
    extracted_value: str | None
    stop_reason: ContinuationStopReason

    def __post_init__(self) -> None:
        _non_empty_text(self.case_id, name="case_id")
        if self.variant not in {"value_a", "value_b"}:
            raise ValueError("unknown continuation variant")
        if (
            not isinstance(self.generated_token_ids, tuple)
            or not self.generated_token_ids
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in self.generated_token_ids
            )
        ):
            raise ValueError("generated_token_ids must be non-empty non-negative IDs")
        _non_empty_text(self.generated_text, name="generated_text")
        if self.extracted_value is not None:
            _non_empty_text(self.extracted_value, name="extracted_value")
        if self.stop_reason not in {
            "natural_stop",
            "length_cap",
            "malformed",
            "runtime_error",
        }:
            raise ValueError("unknown free-continuation stop reason")


@dataclass(frozen=True, slots=True)
class NativeContinuationCacheParity:
    output: NativeFreeContinuationOutput
    compared_logit_steps: int
    max_abs_logit_difference: float
    mean_abs_logit_difference: float
    max_selected_token_logit_difference: float
    min_cached_top1_margin: float
    min_oracle_top1_margin: float
    logits_within_tolerance: bool
    atol: float
    rtol: float

    def __post_init__(self) -> None:
        if not isinstance(self.output, NativeFreeContinuationOutput):
            raise TypeError("cache parity output must be a continuation output")
        if self.compared_logit_steps != len(self.output.generated_token_ids):
            raise ValueError("cache parity logit count differs from generated tokens")
        if not isinstance(self.logits_within_tolerance, bool):
            raise TypeError("logits_within_tolerance must be boolean")
        for name in (
            "max_abs_logit_difference",
            "mean_abs_logit_difference",
            "max_selected_token_logit_difference",
            "min_cached_top1_margin",
            "min_oracle_top1_margin",
            "atol",
            "rtol",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


class NativeFreeContinuationEvaluator(Protocol):
    def __call__(
        self, request: NativeFreeContinuationRequest
    ) -> NativeFreeContinuationOutput: ...


@dataclass(frozen=True, slots=True)
class NativeTargetPresenceCase:
    """One audited same-image supported/unsupported target pair."""

    case_id: str
    pair_identity: str
    source_sample_id: str
    source_image_sha256: str
    positive_target: str
    negative_target: str
    positive_context: NativeDOnlyContext
    negative_context: NativeDOnlyContext
    positive_observation_identity: str
    negative_observation_identity: str
    positive_observation: RepresentationVisualTensorBundle
    negative_observation: RepresentationVisualTensorBundle
    present_value: str = "PRESENT"
    not_present_value: str = "NOT_PRESENT"

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "pair_identity",
            "source_sample_id",
            "positive_target",
            "negative_target",
            "positive_observation_identity",
            "negative_observation_identity",
        ):
            _non_empty_text(getattr(self, name), name=name)
        _lowercase_sha256(self.source_image_sha256, name="source_image_sha256")
        if self.positive_target == self.negative_target:
            raise ValueError("target-presence targets must be distinct")
        if self.present_value != "PRESENT" or self.not_present_value != "NOT_PRESENT":
            raise ValueError("target-presence labels are fixed")
        for context in (self.positive_context, self.negative_context):
            if not isinstance(context, NativeDOnlyContext):
                raise TypeError("target-presence contexts must be NativeDOnlyContext")
        for observation in (
            self.positive_observation,
            self.negative_observation,
        ):
            if not isinstance(observation, RepresentationVisualTensorBundle):
                raise TypeError("target-presence observations must be visual bundles")
        _assert_visual_bundle_contract(
            self.negative_observation,
            self.positive_observation,
            name="negative-target observation",
        )
        if (
            len(self.positive_context.d_positions)
            != self.positive_observation.main.shape[1]
        ):
            raise ValueError("positive target context and D token count differ")
        if (
            len(self.negative_context.d_positions)
            != self.negative_observation.main.shape[1]
        ):
            raise ValueError("negative target context and D token count differ")


class InjectedNativeCounterfactualEvaluator:
    """Concrete exact-D scoring and cached greedy-continuation evaluator.

    This class performs real :meth:`QwenVLMFamilyAdapter.forward_injected`
    calls.  A family-native materializer supplies complete requests because the
    common Qwen adapter deliberately does not guess tokenizer serialization or
    M-RoPE position extension.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        family_adapter: QwenVLMFamilyAdapter,
        materializer: NativeInjectedRequestMaterializer,
        eos_token_ids: tuple[int, ...],
        max_new_tokens: int,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("native counterfactual model must be an nn.Module")
        if not isinstance(family_adapter, QwenVLMFamilyAdapter):
            raise TypeError("family_adapter must be QwenVLMFamilyAdapter")
        if not family_adapter.capabilities.native_injected_kv_cache:
            raise ValueError("family adapter has no accepted injected KV-cache path")
        required_methods = (
            "value_token_ids",
            "teacher_forced",
            "generation_step",
            "decode_generated",
            "extract_expected_value",
        )
        if any(
            not callable(getattr(materializer, name, None)) for name in required_methods
        ):
            raise TypeError("native request materializer is incomplete")
        if (
            not isinstance(eos_token_ids, tuple)
            or not eos_token_ids
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in eos_token_ids
            )
            or len(set(eos_token_ids)) != len(eos_token_ids)
        ):
            raise ValueError("eos_token_ids must be unique non-negative token IDs")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        assert_frozen_deterministic_readout_model(model)
        self.model = model
        self.family_adapter = family_adapter
        self.materializer = materializer
        self.eos_token_ids = eos_token_ids
        self.max_new_tokens = max_new_tokens

    def causal_value_flip(
        self, request: NativeCausalValueFlipRequest
    ) -> NativeCausalValueFlipOutput:
        if not isinstance(request, NativeCausalValueFlipRequest):
            raise TypeError("causal value-flip input must be a typed request")
        case = request.case
        self._validate_case_family(case)
        value_a_ids = _validated_generated_ids(
            self.materializer.value_token_ids(case.context, case.expected_value_a),
            name="value A token IDs",
        )
        value_b_ids = _validated_generated_ids(
            self.materializer.value_token_ids(case.context, case.expected_value_b),
            name="value B token IDs",
        )
        if value_a_ids == value_b_ids:
            raise ValueError("counterfactual values cannot share token IDs")
        observation_a_log_odds = self._value_log_odds(
            context=case.context,
            observation=case.observation_a,
            value_a_ids=value_a_ids,
            value_b_ids=value_b_ids,
        )
        observation_b_log_odds = self._value_log_odds(
            context=case.context,
            observation=case.observation_b,
            value_a_ids=value_a_ids,
            value_b_ids=value_b_ids,
        )
        return NativeCausalValueFlipOutput(
            case_id=case.case_id,
            observation_a_log_odds_a_over_b=observation_a_log_odds,
            observation_b_log_odds_a_over_b=observation_b_log_odds,
        )

    def binary_log_odds(
        self, request: NativeBinaryLogOddsRequest
    ) -> NativeBinaryLogOddsOutput:
        if not isinstance(request, NativeBinaryLogOddsRequest):
            raise TypeError("binary log-odds input must be a typed request")
        if request.context.family != self.family_adapter.capabilities.family:
            raise ValueError("binary log-odds context differs from Qwen family")
        if len(request.observation.deepstack) != (
            self.family_adapter.capabilities.deepstack_branch_count
        ):
            raise ValueError("binary log-odds branch count differs from Qwen family")
        positive_ids = _validated_generated_ids(
            self.materializer.value_token_ids(request.context, request.positive_value),
            name="positive continuation token IDs",
        )
        negative_ids = _validated_generated_ids(
            self.materializer.value_token_ids(request.context, request.negative_value),
            name="negative continuation token IDs",
        )
        if positive_ids == negative_ids:
            raise ValueError("binary continuation values cannot share token IDs")
        return NativeBinaryLogOddsOutput(
            case_id=request.case_id,
            log_odds_positive_over_negative=(
                self._value_logprob(
                    context=request.context,
                    observation=request.observation,
                    token_ids=positive_ids,
                )
                / len(positive_ids)
                - self._value_logprob(
                    context=request.context,
                    observation=request.observation,
                    token_ids=negative_ids,
                )
                / len(negative_ids)
            ),
        )

    def free_continuation(
        self, request: NativeFreeContinuationRequest
    ) -> NativeFreeContinuationOutput:
        """Decode with one injected-D prefill followed by cached text tokens."""

        return self._free_continuation_cached(request, captured_logits=None)

    def _free_continuation_cached(
        self,
        request: NativeFreeContinuationRequest,
        *,
        captured_logits: list[torch.Tensor] | None,
    ) -> NativeFreeContinuationOutput:

        self._validate_free_continuation_request(request)
        generated: list[int] = []
        stop_reason: ContinuationStopReason = "length_cap"
        materialized = self.materializer.generation_step(
            context=request.context,
            observation=request.observation,
            generated_token_ids=(),
        )
        self._validate_generation_materialization(
            request=request,
            materialized=materialized,
            generated=(),
        )
        with torch.no_grad():
            result = self.family_adapter.prefill_injected_cache(
                self.model,
                materialized.request,
            )
        past_key_values = result.past_key_values
        next_logits = result.logits[0, materialized.next_token_logit_position].float()

        for token_index in range(self.max_new_tokens):
            if captured_logits is not None:
                captured_logits.append(next_logits.detach().cpu())
            token_id = _greedy_token_id(next_logits)
            generated.append(token_id)
            if token_id in self.eos_token_ids:
                stop_reason = "natural_stop"
                break
            if token_index + 1 == self.max_new_tokens:
                break
            materialized = self.materializer.generation_step(
                context=request.context,
                observation=request.observation,
                generated_token_ids=tuple(generated),
            )
            self._validate_generation_materialization(
                request=request,
                materialized=materialized,
                generated=tuple(generated),
            )
            full_request = materialized.request
            cache_position = torch.tensor(
                (full_request.input_ids.shape[1] - 1,),
                dtype=torch.long,
                device=full_request.input_ids.device,
            )
            with torch.no_grad():
                result = self.family_adapter.forward_cached_token(
                    self.model,
                    CachedTokenForwardRequest(
                        input_ids=full_request.input_ids[:, -1:],
                        attention_mask=full_request.attention_mask,
                        position_ids=full_request.position_ids[..., -1:],
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                    ),
                )
            past_key_values = result.past_key_values
            next_logits = result.logits[0, -1].float()
        return self._finalize_continuation(
            request=request,
            generated=generated,
            stop_reason=stop_reason,
        )

    def free_continuation_no_cache(
        self, request: NativeFreeContinuationRequest
    ) -> NativeFreeContinuationOutput:
        """Bounded full-prefix oracle retained only for cache parity tests."""

        return self._free_continuation_no_cache(request, captured_logits=None)

    def _free_continuation_no_cache(
        self,
        request: NativeFreeContinuationRequest,
        *,
        captured_logits: list[torch.Tensor] | None,
    ) -> NativeFreeContinuationOutput:

        self._validate_free_continuation_request(request)
        generated: list[int] = []
        stop_reason: ContinuationStopReason = "length_cap"
        for _ in range(self.max_new_tokens):
            materialized = self.materializer.generation_step(
                context=request.context,
                observation=request.observation,
                generated_token_ids=tuple(generated),
            )
            self._validate_generation_materialization(
                request=request,
                materialized=materialized,
                generated=tuple(generated),
            )
            with torch.no_grad():
                result = self.family_adapter.forward_injected(
                    self.model, materialized.request
                )
            next_logits = result.logits[
                0, materialized.next_token_logit_position
            ].float()
            if captured_logits is not None:
                captured_logits.append(next_logits.detach().cpu())
            token_id = _greedy_token_id(next_logits)
            generated.append(token_id)
            if token_id in self.eos_token_ids:
                stop_reason = "natural_stop"
                break
        return self._finalize_continuation(
            request=request,
            generated=generated,
            stop_reason=stop_reason,
        )

    def continuation_cache_parity(
        self,
        request: NativeFreeContinuationRequest,
        *,
        atol: float,
        rtol: float,
        require_logits_within_tolerance: bool = True,
    ) -> NativeContinuationCacheParity:
        """Compare every cached next-token logit with the full-prefix oracle."""

        for name, value in (("atol", atol), ("rtol", rtol)):
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative float")
        cached_logits: list[torch.Tensor] = []
        oracle_logits: list[torch.Tensor] = []
        cached = self._free_continuation_cached(
            request,
            captured_logits=cached_logits,
        )
        oracle = self._free_continuation_no_cache(
            request,
            captured_logits=oracle_logits,
        )
        if cached != oracle:
            raise RuntimeError("cached continuation differs from no-cache oracle")
        if len(cached_logits) != len(oracle_logits) or not cached_logits:
            raise RuntimeError("cached and no-cache logit traces differ in length")
        max_abs_difference = 0.0
        sum_abs_difference = 0.0
        logit_count = 0
        max_selected_difference = 0.0
        cached_margins: list[float] = []
        oracle_margins: list[float] = []
        logits_within_tolerance = True
        for index, (cached_step, oracle_step) in enumerate(
            zip(cached_logits, oracle_logits, strict=True)
        ):
            absolute = (cached_step - oracle_step).abs()
            difference = float(absolute.max().item())
            max_abs_difference = max(max_abs_difference, difference)
            sum_abs_difference += float(absolute.sum().item())
            logit_count += absolute.numel()
            selected_token_id = cached.generated_token_ids[index]
            max_selected_difference = max(
                max_selected_difference,
                float(absolute[selected_token_id].item()),
            )
            cached_top2 = torch.topk(cached_step, k=2).values
            oracle_top2 = torch.topk(oracle_step, k=2).values
            cached_margins.append(float((cached_top2[0] - cached_top2[1]).item()))
            oracle_margins.append(float((oracle_top2[0] - oracle_top2[1]).item()))
            step_within_tolerance = torch.allclose(
                cached_step,
                oracle_step,
                atol=atol,
                rtol=rtol,
            )
            logits_within_tolerance = logits_within_tolerance and step_within_tolerance
            if require_logits_within_tolerance and not step_within_tolerance:
                raise RuntimeError(
                    "cached continuation logit parity failed at step "
                    f"{index}: max_abs_difference={difference}"
                )
        return NativeContinuationCacheParity(
            output=cached,
            compared_logit_steps=len(cached_logits),
            max_abs_logit_difference=max_abs_difference,
            mean_abs_logit_difference=sum_abs_difference / logit_count,
            max_selected_token_logit_difference=max_selected_difference,
            min_cached_top1_margin=min(cached_margins),
            min_oracle_top1_margin=min(oracle_margins),
            logits_within_tolerance=logits_within_tolerance,
            atol=atol,
            rtol=rtol,
        )

    def _validate_free_continuation_request(
        self, request: NativeFreeContinuationRequest
    ) -> None:
        if not isinstance(request, NativeFreeContinuationRequest):
            raise TypeError("free continuation input must be a typed request")
        if request.context.family != self.family_adapter.capabilities.family:
            raise ValueError("free-continuation context differs from Qwen family")
        if len(request.observation.deepstack) != (
            self.family_adapter.capabilities.deepstack_branch_count
        ):
            raise ValueError("free-continuation branch count differs from Qwen family")

    def _validate_generation_materialization(
        self,
        *,
        request: NativeFreeContinuationRequest,
        materialized: NativeGenerationForward,
        generated: tuple[int, ...],
    ) -> None:
        if not isinstance(materialized, NativeGenerationForward):
            raise TypeError("generation materializer returned an invalid forward")
        _validate_materialized_request(
            materialized.request,
            context=request.context,
            observation=request.observation,
            generated_prefix=generated,
        )

    def _finalize_continuation(
        self,
        *,
        request: NativeFreeContinuationRequest,
        generated: list[int],
        stop_reason: ContinuationStopReason,
    ) -> NativeFreeContinuationOutput:
        generated_ids = tuple(generated)
        generated_text = self.materializer.decode_generated(generated_ids)
        _non_empty_text(generated_text, name="decoded free continuation")
        extracted = self.materializer.extract_expected_value(
            generated_text, request.expected_value
        )
        if extracted is not None:
            _non_empty_text(extracted, name="extracted expected value")
        return NativeFreeContinuationOutput(
            case_id=request.case_id,
            variant=request.variant,
            generated_token_ids=generated_ids,
            generated_text=generated_text,
            extracted_value=extracted,
            stop_reason=stop_reason,
        )

    def _validate_case_family(self, case: NativeCounterfactualCase) -> None:
        if case.context.family != self.family_adapter.capabilities.family:
            raise ValueError("causal value-flip context differs from Qwen family")
        if any(
            len(observation.deepstack)
            != self.family_adapter.capabilities.deepstack_branch_count
            for observation in (case.observation_a, case.observation_b)
        ):
            raise ValueError("causal value-flip branch count differs from Qwen family")

    def _value_log_odds(
        self,
        *,
        context: NativeDOnlyContext,
        observation: RepresentationVisualTensorBundle,
        value_a_ids: tuple[int, ...],
        value_b_ids: tuple[int, ...],
    ) -> float:
        logprob_a = self._value_logprob(
            context=context,
            observation=observation,
            token_ids=value_a_ids,
        )
        logprob_b = self._value_logprob(
            context=context,
            observation=observation,
            token_ids=value_b_ids,
        )
        return logprob_a - logprob_b

    def _value_logprob(
        self,
        *,
        context: NativeDOnlyContext,
        observation: RepresentationVisualTensorBundle,
        token_ids: tuple[int, ...],
    ) -> float:
        materialized = self.materializer.teacher_forced(
            context=context,
            observation=observation,
            continuation_token_ids=token_ids,
        )
        if not isinstance(materialized, NativeTeacherForcedForward):
            raise TypeError("teacher-forced materializer returned an invalid forward")
        _validate_materialized_request(
            materialized.request,
            context=context,
            observation=observation,
            generated_prefix=token_ids,
            teacher_forced_positions=materialized.continuation_positions,
        )
        if len(materialized.continuation_positions) != len(token_ids):
            raise ValueError("teacher-forced token IDs and positions must align")
        with torch.no_grad():
            result = self.family_adapter.forward_injected(
                self.model, materialized.request
            )
        log_probs = torch.log_softmax(result.logits.float(), dim=-1)
        terms = tuple(
            log_probs[0, position - 1, token_id]
            for position, token_id in zip(
                materialized.continuation_positions,
                token_ids,
                strict=True,
            )
        )
        value = torch.stack(terms).sum()
        if not bool(torch.isfinite(value).item()):
            raise RuntimeError("teacher-forced value log probability is non-finite")
        return float(value.item())


def _greedy_token_id(logits: torch.Tensor) -> int:
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("native generation produced invalid next-token logits")
    return int(torch.argmax(logits).item())


def create_injected_native_counterfactual_evaluator(
    *,
    model: nn.Module,
    family_adapter: QwenVLMFamilyAdapter,
    materializer: NativeInjectedRequestMaterializer,
    eos_token_ids: tuple[int, ...],
    max_new_tokens: int,
) -> InjectedNativeCounterfactualEvaluator:
    """Construct the concrete evaluator used as both native runner callbacks."""

    return InjectedNativeCounterfactualEvaluator(
        model=model,
        family_adapter=family_adapter,
        materializer=materializer,
        eos_token_ids=eos_token_ids,
        max_new_tokens=max_new_tokens,
    )


@dataclass(frozen=True, slots=True)
class RepresentationReadoutControlIdentity:
    correct_candidate_sample_id: str
    wrong_same_image_candidate_sample_id: str
    wrong_different_image_candidate_sample_id: str | None
    wrong_different_image_group_key: str | None
    random_d_seed: int
    random_d_algorithm: str = DETERMINISTIC_RANDOM_D_ALGORITHM


@dataclass(frozen=True, slots=True)
class RepresentationBranchHealthRecord:
    branch_layer: int
    norm: NormComparisonDiagnostics
    attention: AttentionDiagnostics


@dataclass(frozen=True, slots=True)
class RepresentationSampleHealthRecord:
    main: NormComparisonDiagnostics
    main_attention: AttentionDiagnostics
    branches: tuple[RepresentationBranchHealthRecord, ...]


@dataclass(frozen=True, slots=True)
class RepresentationInternalEvaluationSampleRecord:
    sample_id: str
    sample_content_sha256: str
    image_group_key: str
    source_visual_identity: str
    target_conditioning_provider: str
    projection_identities: tuple[str, ...]
    evidence_type: str | None
    source_profile: str | None
    answer_type: str | None
    visual_difficulty: str | None
    controls: RepresentationReadoutControlIdentity
    nlls: ReadoutNLLs
    readout: ReadoutSampleMetrics
    health: RepresentationSampleHealthRecord


@dataclass(frozen=True, slots=True)
class RepresentationInternalEvaluationGroupRecord:
    image_group_key: str
    source_visual_identity: str
    sample_ids: tuple[str, ...]
    wrong_different_image_group_key: str | None
    query: QueryScoreMatrixMetrics


@dataclass(frozen=True, slots=True)
class RepresentationGroupedInternalMetrics:
    dimension: str
    label: str
    readout: ReadoutMetrics
    query: QueryMetrics


@dataclass(frozen=True, slots=True)
class RepresentationBranchHealthSummary:
    branch_layer: int
    representation: RepresentationHealthSummary
    attention: AttentionDiagnosticsSummary


@dataclass(frozen=True, slots=True)
class RepresentationInternalHealthSummary:
    main: RepresentationHealthSummary
    main_attention: AttentionDiagnosticsSummary
    branches: tuple[RepresentationBranchHealthSummary, ...]


@dataclass(frozen=True, slots=True)
class RepresentationReadoutExecutionContract:
    family: str
    qwen_frozen: bool = True
    second_full_qwen_forward_used: bool = False
    forward_path: str = READOUT_FORWARD_PATH
    mask_mode: str = READOUT_MASK_MODE
    position_source: str = READOUT_POSITION_SOURCE
    visual_swap_unit: str = READOUT_VISUAL_SWAP_UNIT

    def __post_init__(self) -> None:
        _non_empty_text(self.family, name="family")
        if not self.qwen_frozen or self.second_full_qwen_forward_used:
            raise ValueError("internal readout must use one frozen injected LM path")
        if self.forward_path != READOUT_FORWARD_PATH:
            raise ValueError("internal readout forward path drifted")
        if self.mask_mode != READOUT_MASK_MODE:
            raise ValueError("internal readout mask mode drifted")
        if self.position_source != READOUT_POSITION_SOURCE:
            raise ValueError("internal readout position source drifted")
        if self.visual_swap_unit != READOUT_VISUAL_SWAP_UNIT:
            raise ValueError("internal readout visual swap unit drifted")


@dataclass(frozen=True, slots=True)
class NativeFreeContinuationRecord:
    variant: ContinuationVariant
    expected_value: str
    generated_token_ids: tuple[int, ...]
    generated_text: str
    extracted_value: str | None
    stop_reason: ContinuationStopReason
    value_consistent: bool
    healthy_termination: bool


@dataclass(frozen=True, slots=True)
class NativeCounterfactualEvaluationRecord:
    case_id: str
    pair_identity: str
    sample_a_id: str
    sample_b_id: str
    context_id: str
    transcript_identity: str
    observation_a_identity: str
    observation_b_identity: str
    observation_a_log_odds_a_over_b: float
    observation_b_log_odds_a_over_b: float
    expected_direction_flip: bool
    log_odds_separation: float
    continuations: tuple[NativeFreeContinuationRecord, ...]


@dataclass(frozen=True, slots=True)
class NativeCounterfactualSummary:
    pair_count: int
    expected_direction_flip_rate: float
    continuation_accuracy: float
    healthy_termination_rate: float

    def __post_init__(self) -> None:
        if isinstance(self.pair_count, bool) or self.pair_count <= 0:
            raise ValueError("counterfactual summary requires positive pair_count")
        for name in (
            "expected_direction_flip_rate",
            "continuation_accuracy",
            "healthy_termination_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0 or value > 1:
                raise ValueError(f"{name} must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class NativeTargetPresenceContinuationRecord:
    variant: TargetPresenceVariant
    expected_value: str
    generated_token_ids: tuple[int, ...]
    generated_text: str
    extracted_value: str | None
    stop_reason: ContinuationStopReason
    value_consistent: bool
    healthy_termination: bool


@dataclass(frozen=True, slots=True)
class NativeTargetPresenceEvaluationRecord:
    case_id: str
    pair_identity: str
    source_sample_id: str
    source_image_sha256: str
    positive_target: str
    negative_target: str
    positive_context_id: str
    negative_context_id: str
    positive_observation_identity: str
    negative_observation_identity: str
    actual_positive_log_odds_present: float
    actual_negative_log_odds_present: float
    zero_d_positive_log_odds_present: float
    zero_d_negative_log_odds_present: float
    actual_separation: float
    positive_d_contribution: float
    negative_d_false_positive_amplification: float
    actual_direction_correct: bool
    negative_false_positive: bool
    continuations: tuple[NativeTargetPresenceContinuationRecord, ...]


@dataclass(frozen=True, slots=True)
class NativeTargetPresenceSummary:
    pair_count: int
    actual_direction_accuracy: float
    negative_false_positive_rate: float
    mean_actual_separation: float
    mean_positive_d_contribution: float
    mean_negative_d_false_positive_amplification: float
    continuation_accuracy: float
    healthy_termination_rate: float
    score_contract: str = TARGET_PRESENCE_LOG_ODDS_CONTRACT

    def __post_init__(self) -> None:
        if isinstance(self.pair_count, bool) or self.pair_count <= 0:
            raise ValueError("target-presence summary requires positive pair_count")
        for name in (
            "actual_direction_accuracy",
            "negative_false_positive_rate",
            "mean_actual_separation",
            "mean_positive_d_contribution",
            "mean_negative_d_false_positive_amplification",
            "continuation_accuracy",
            "healthy_termination_rate",
        ):
            _finite_float(getattr(self, name), name=name)
        if self.score_contract != TARGET_PRESENCE_LOG_ODDS_CONTRACT:
            raise ValueError("target-presence score contract drifted")


@dataclass(frozen=True, slots=True)
class RepresentationInternalEvaluationReport:
    identity: RepresentationInternalEvaluationIdentity
    execution: RepresentationReadoutExecutionContract
    groups: tuple[RepresentationInternalEvaluationGroupRecord, ...]
    samples: tuple[RepresentationInternalEvaluationSampleRecord, ...]
    readout: ReadoutMetrics
    query: QueryMetrics
    grouped_metrics: tuple[RepresentationGroupedInternalMetrics, ...]
    health: RepresentationInternalHealthSummary
    native_counterfactuals: tuple[NativeCounterfactualEvaluationRecord, ...]
    native_counterfactual_summary: NativeCounterfactualSummary
    native_target_presence: tuple[NativeTargetPresenceEvaluationRecord, ...] = ()
    native_target_presence_summary: NativeTargetPresenceSummary | None = None
    random_d_algorithm: str = DETERMINISTIC_RANDOM_D_ALGORITHM
    attention_topk: int = ATTENTION_TOPK

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RepresentationInternalEvaluationIdentity):
            raise TypeError("internal-evaluation report requires a typed identity")
        if not isinstance(self.execution, RepresentationReadoutExecutionContract):
            raise TypeError("internal-evaluation report requires an execution contract")
        if len(self.groups) < 2:
            raise ValueError("wrong-different-image evaluation requires two groups")
        if not self.samples:
            raise ValueError("internal-evaluation report requires sample records")
        if self.readout.sample_count != len(self.samples):
            raise ValueError("readout summary count differs from sample records")
        if self.query.group_count != len(self.groups):
            raise ValueError("query summary count differs from group records")
        if not self.native_counterfactuals:
            raise ValueError("native counterfactual outputs are required")
        if self.native_counterfactual_summary.pair_count != len(
            self.native_counterfactuals
        ):
            raise ValueError("counterfactual summary count differs from records")
        if bool(self.native_target_presence) != (
            self.native_target_presence_summary is not None
        ):
            raise ValueError("target-presence records and summary must co-occur")
        if self.native_target_presence_summary is not None and (
            self.native_target_presence_summary.pair_count
            != len(self.native_target_presence)
        ):
            raise ValueError("target-presence summary count differs from records")
        if self.random_d_algorithm != DETERMINISTIC_RANDOM_D_ALGORITHM:
            raise ValueError("internal-evaluation random-D algorithm drifted")
        if self.attention_topk != ATTENTION_TOPK:
            raise ValueError("internal-evaluation attention top-k drifted")


@dataclass(frozen=True, slots=True)
class RepresentationInternalEvaluationArtifact:
    path: str
    payload_sha256: str
    byte_count: int
    schema_version: str = REPRESENTATION_INTERNAL_EVALUATION_ARTIFACT_SCHEMA_VERSION


def run_representation_internal_evaluation(
    *,
    identity: RepresentationInternalEvaluationIdentity,
    adapter: TGVFAdapter,
    qwen_model: nn.Module,
    family_adapter: QwenVLMFamilyAdapter,
    sample_groups: Sequence[Sequence[RepresentationTrainingSample]],
    group_builder: RepresentationGroupBuilder,
    native_counterfactual_cases: Sequence[NativeCounterfactualCase],
    causal_value_flip_evaluator: NativeCausalValueFlipEvaluator,
    free_continuation_evaluator: NativeFreeContinuationEvaluator,
    native_target_presence_cases: Sequence[NativeTargetPresenceCase] = (),
    target_presence_log_odds_evaluator: NativeBinaryLogOddsEvaluator | None = None,
    target_presence_free_continuation_evaluator: (
        NativeFreeContinuationEvaluator | None
    ) = None,
) -> RepresentationInternalEvaluationReport:
    """Run all accepted historical controls and native causal evaluations."""

    if not isinstance(identity, RepresentationInternalEvaluationIdentity):
        raise TypeError("identity must be RepresentationInternalEvaluationIdentity")
    if not isinstance(adapter, TGVFAdapter):
        raise TypeError("adapter must be a TGVFAdapter")
    if not isinstance(qwen_model, nn.Module):
        raise TypeError("qwen_model must be an nn.Module")
    if not isinstance(family_adapter, QwenVLMFamilyAdapter):
        raise TypeError("family_adapter must be a QwenVLMFamilyAdapter")
    if not callable(group_builder):
        raise TypeError("group_builder must be callable")
    if not callable(causal_value_flip_evaluator) or not callable(
        free_continuation_evaluator
    ):
        raise TypeError("native internal evaluators must be callable")
    groups_of_samples = _validated_sample_groups(sample_groups)
    native_cases = _validated_native_cases(native_counterfactual_cases)
    target_presence_cases = _validated_target_presence_cases(
        native_target_presence_cases
    )
    if target_presence_cases and (
        target_presence_log_odds_evaluator is None
        or target_presence_free_continuation_evaluator is None
    ):
        raise ValueError("target-presence cases require both native evaluators")
    if not target_presence_cases and (
        target_presence_log_odds_evaluator is not None
        or target_presence_free_continuation_evaluator is not None
    ):
        raise ValueError("target-presence evaluators require target-presence cases")
    if family_adapter.capabilities.family != native_cases[0].context.family:
        raise ValueError("native counterfactual family differs from Qwen adapter")
    if any(
        case.context.family != family_adapter.capabilities.family
        for case in native_cases
    ):
        raise ValueError("native counterfactual cases cannot mix Qwen families")
    if any(
        len(observation.deepstack) != family_adapter.capabilities.deepstack_branch_count
        for case in native_cases
        for observation in (case.observation_a, case.observation_b)
    ):
        raise ValueError("native counterfactual branch count differs from Qwen family")
    if any(
        context.family != family_adapter.capabilities.family
        for case in target_presence_cases
        for context in (case.positive_context, case.negative_context)
    ):
        raise ValueError("target-presence cases differ from Qwen family")
    if any(
        len(observation.deepstack) != family_adapter.capabilities.deepstack_branch_count
        for case in target_presence_cases
        for observation in (
            case.positive_observation,
            case.negative_observation,
        )
    ):
        raise ValueError("target-presence branch count differs from Qwen family")

    assert_frozen_deterministic_readout_model(qwen_model)
    adapter_modes = tuple((module, module.training) for module in adapter.modules())
    adapter_parameter_state = tuple(
        (parameter, parameter.requires_grad, parameter.grad)
        for parameter in adapter.parameters()
    )
    built_groups: list[SameImageReadoutGroup] = []
    try:
        adapter.eval()
        with torch.no_grad():
            for samples in groups_of_samples:
                group = group_builder(
                    samples,
                    adapter,
                    collective_candidate_count=len(samples),
                )
                _validate_built_group(group, samples, identity=identity)
                built_groups.append(group)
    finally:
        for module, was_training in adapter_modes:
            module.training = was_training
    _assert_adapter_state_unchanged(adapter_parameter_state)

    groups = tuple(built_groups)
    _validate_cross_group_contract(groups)
    samples_by_id = {
        sample.sample_id: sample
        for group_samples in groups_of_samples
        for sample in group_samples
    }

    score_records = tuple(
        score_streaming_same_image_group(family_adapter, qwen_model, group)
        for group in groups
    )
    nll_matrices = tuple(_mean_nll_matrix(record) for record in score_records)
    query_records = tuple(query_score_matrix_metrics(matrix) for matrix in nll_matrices)

    group_records: list[RepresentationInternalEvaluationGroupRecord] = []
    sample_records: list[RepresentationInternalEvaluationSampleRecord] = []
    for group_index, (group, scores, query) in enumerate(
        zip(groups, score_records, query_records, strict=True)
    ):
        wrong_different_pairs = tuple(
            _find_wrong_different_candidate(
                groups,
                group_index=group_index,
                row_index=row_index,
                expected=candidate.visual,
            )
            for row_index, candidate in enumerate(group.candidates)
        )
        paired_group_keys = {
            pair[0].image_group_key
            for pair in wrong_different_pairs
            if pair is not None
        }
        group_records.append(
            RepresentationInternalEvaluationGroupRecord(
                image_group_key=group.image_group_key,
                source_visual_identity=group.source_visual_identity,
                sample_ids=scores.sample_ids,
                wrong_different_image_group_key=(
                    next(iter(paired_group_keys))
                    if len(paired_group_keys) == 1
                    else None
                ),
                query=query,
            )
        )
        for row_index, (row, candidate) in enumerate(
            zip(group.rows, group.candidates, strict=True)
        ):
            same_index = (row_index + 1) % len(group.candidates)
            wrong_same = group.candidates[same_index]
            wrong_different_pair = wrong_different_pairs[row_index]
            different_group, wrong_different = (
                wrong_different_pair
                if wrong_different_pair is not None
                else (None, None)
            )
            random_seed = _random_control_seed(identity, row.sample_id)
            random_visual = _deterministic_random_visual_bundle(
                candidate.visual, seed=random_seed
            )
            token_count = int(scores.evidence_token_counts[row_index].item())
            if token_count <= 0:
                raise RuntimeError("internal readout has no evidence token")
            correct_nll = _token_mean_nll_from_cell_score(
                scores.score_matrix[row_index, row_index], token_count
            )
            diagonal_l_gen = float(scores.diagonal_l_gen[row_index].float().item())
            if not math.isclose(
                correct_nll, diagonal_l_gen, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise RuntimeError("query diagonal and readability NLL diverged")
            wrong_same_nll = _token_mean_nll_from_cell_score(
                scores.score_matrix[row_index, same_index], token_count
            )
            target_only_nll, target_only_count = _score_readout_control(
                family_adapter,
                qwen_model,
                source=group.source_visual,
                row=row,
                candidate=None,
            )
            random_nll, random_count = _score_readout_control(
                family_adapter,
                qwen_model,
                source=group.source_visual,
                row=row,
                candidate=random_visual,
            )
            different_nll: float | None = None
            different_count: int | None = None
            if wrong_different is not None:
                different_nll, different_count = _score_readout_control(
                    family_adapter,
                    qwen_model,
                    source=group.source_visual,
                    row=row,
                    candidate=wrong_different.visual,
                )
            available_counts = {target_only_count, random_count}
            if different_count is not None:
                available_counts.add(different_count)
            if available_counts != {token_count}:
                raise RuntimeError(
                    "control evidence-token counts differ from correct D"
                )
            nlls = ReadoutNLLs(
                correct_d=correct_nll,
                target_only=target_only_nll,
                random_d=random_nll,
                wrong_same_image_d=wrong_same_nll,
                wrong_different_image_d=different_nll,
            )
            sample = samples_by_id[row.sample_id]
            sample_records.append(
                RepresentationInternalEvaluationSampleRecord(
                    sample_id=sample.sample_id,
                    sample_content_sha256=sample.content_sha256,
                    image_group_key=group.image_group_key,
                    source_visual_identity=group.source_visual_identity,
                    target_conditioning_provider=(
                        candidate.target_conditioning_provider.value
                    ),
                    projection_identities=candidate.projection_identities,
                    evidence_type=sample.evidence_type,
                    source_profile=sample.source_profile,
                    answer_type=sample.answer_type,
                    visual_difficulty=sample.visual_difficulty,
                    controls=RepresentationReadoutControlIdentity(
                        correct_candidate_sample_id=candidate.sample_id,
                        wrong_same_image_candidate_sample_id=wrong_same.sample_id,
                        wrong_different_image_candidate_sample_id=(
                            wrong_different.sample_id
                            if wrong_different is not None
                            else None
                        ),
                        wrong_different_image_group_key=(
                            different_group.image_group_key
                            if different_group is not None
                            else None
                        ),
                        random_d_seed=random_seed,
                    ),
                    nlls=nlls,
                    readout=readout_sample_metrics(nlls),
                    health=_sample_health(group, candidate),
                )
            )
    sample_record_tuple = tuple(sample_records)
    readout_rows = tuple(record.nlls for record in sample_record_tuple)
    readout_summary = summarize_readout(readout_rows)
    query_summary = summarize_query_score_matrices(nll_matrices)
    grouped_metrics = _grouped_metrics(
        groups_of_samples,
        sample_record_tuple,
        nll_matrices,
    )
    health = _summarize_health(sample_record_tuple)
    native_records = tuple(
        _evaluate_native_case(
            case,
            causal_value_flip_evaluator=causal_value_flip_evaluator,
            free_continuation_evaluator=free_continuation_evaluator,
        )
        for case in native_cases
    )
    target_presence_records: tuple[NativeTargetPresenceEvaluationRecord, ...] = ()
    target_presence_summary: NativeTargetPresenceSummary | None = None
    if target_presence_cases:
        assert target_presence_log_odds_evaluator is not None
        assert target_presence_free_continuation_evaluator is not None
        target_presence_records = tuple(
            _evaluate_target_presence_case(
                case,
                log_odds_evaluator=target_presence_log_odds_evaluator,
                free_continuation_evaluator=(
                    target_presence_free_continuation_evaluator
                ),
            )
            for case in target_presence_cases
        )
        target_presence_summary = _summarize_target_presence(target_presence_records)
    return RepresentationInternalEvaluationReport(
        identity=identity,
        execution=RepresentationReadoutExecutionContract(
            family=family_adapter.capabilities.family
        ),
        groups=tuple(group_records),
        samples=sample_record_tuple,
        readout=readout_summary,
        query=query_summary,
        grouped_metrics=grouped_metrics,
        health=health,
        native_counterfactuals=native_records,
        native_counterfactual_summary=_summarize_counterfactuals(native_records),
        native_target_presence=target_presence_records,
        native_target_presence_summary=target_presence_summary,
    )


def _token_mean_nll_from_cell_score(
    summed_log_likelihood: torch.Tensor,
    valid_token_count: int,
) -> float:
    """Preserve the score tensor's reduction dtype before exporting FP32."""

    if (
        summed_log_likelihood.ndim != 0
        or not summed_log_likelihood.dtype.is_floating_point
    ):
        raise TypeError("Matrix-CE cell score must be one floating scalar")
    if isinstance(valid_token_count, bool) or not isinstance(valid_token_count, int):
        raise TypeError("valid evidence-token count must be an integer")
    if valid_token_count <= 0:
        raise ValueError("valid evidence-token count must be positive")
    return float((-summed_log_likelihood / valid_token_count).float().item())


def _mean_nll_matrix(scores: StreamingGroupScores) -> torch.Tensor:
    """Convert row-local summed log likelihoods to Golden-compatible mean NLL."""

    if scores.score_matrix.ndim != 2 or scores.score_matrix.shape[0] != (
        scores.evidence_token_counts.numel()
    ):
        raise ValueError("score matrix and evidence-token counts do not align")
    if bool((scores.evidence_token_counts <= 0).any().item()):
        raise ValueError("query matrix contains an empty evidence row")
    rows = tuple(
        (-scores.score_matrix[row_index] / int(token_count.item())).float().cpu()
        for row_index, token_count in enumerate(scores.evidence_token_counts)
    )
    return torch.stack(rows)


def save_representation_internal_evaluation_report_atomic(
    report: RepresentationInternalEvaluationReport,
    path: str | Path,
) -> RepresentationInternalEvaluationArtifact:
    """Create one immutable canonical JSON artifact without overwriting a file."""

    if not isinstance(report, RepresentationInternalEvaluationReport):
        raise TypeError("report must be RepresentationInternalEvaluationReport")
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("internal-evaluation artifact path must be absolute")
    if not destination.parent.is_dir():
        raise ValueError("internal-evaluation artifact parent must already exist")
    payload = (
        json.dumps(
            _json_value(report),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return RepresentationInternalEvaluationArtifact(
        path=str(destination),
        payload_sha256=sha256(payload).hexdigest(),
        byte_count=len(payload),
    )


def _validated_sample_groups(
    sample_groups: Sequence[Sequence[RepresentationTrainingSample]],
) -> tuple[tuple[RepresentationTrainingSample, ...], ...]:
    if isinstance(sample_groups, (str, bytes)) or not isinstance(
        sample_groups, Sequence
    ):
        raise TypeError("sample_groups must be a nested sequence")
    materialized = tuple(tuple(group) for group in sample_groups)
    if len(materialized) < 2:
        raise ValueError(
            "wrong-different-image evaluation requires at least two groups"
        )
    all_ids: list[str] = []
    group_keys: list[str] = []
    for group in materialized:
        if len(group) < 2 or any(
            not isinstance(sample, RepresentationTrainingSample) for sample in group
        ):
            raise TypeError("every evaluation group must contain K>=2 typed samples")
        keys = {sample.image_group_key for sample in group}
        if len(keys) != 1:
            raise ValueError("one internal-evaluation group must share one image key")
        group_keys.append(next(iter(keys)))
        all_ids.extend(sample.sample_id for sample in group)
    if len(set(group_keys)) != len(group_keys):
        raise ValueError("internal-evaluation image groups must be distinct")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("internal-evaluation sample IDs must be globally unique")
    return materialized


def _validated_native_cases(
    cases: Sequence[NativeCounterfactualCase],
) -> tuple[NativeCounterfactualCase, ...]:
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise TypeError("native_counterfactual_cases must be a sequence")
    materialized = tuple(cases)
    if not materialized or any(
        not isinstance(case, NativeCounterfactualCase) for case in materialized
    ):
        raise TypeError("at least one typed native counterfactual case is required")
    if len({case.case_id for case in materialized}) != len(materialized):
        raise ValueError("native counterfactual case IDs must be unique")
    return materialized


def _validated_target_presence_cases(
    cases: Sequence[NativeTargetPresenceCase],
) -> tuple[NativeTargetPresenceCase, ...]:
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise TypeError("native_target_presence_cases must be a sequence")
    materialized = tuple(cases)
    if any(not isinstance(case, NativeTargetPresenceCase) for case in materialized):
        raise TypeError("target-presence cases must be typed")
    if len({case.case_id for case in materialized}) != len(materialized):
        raise ValueError("target-presence case IDs must be unique")
    return materialized


def _validate_built_group(
    group: object,
    samples: tuple[RepresentationTrainingSample, ...],
    *,
    identity: RepresentationInternalEvaluationIdentity,
) -> None:
    if not isinstance(group, SameImageReadoutGroup):
        raise TypeError("group_builder must return SameImageReadoutGroup")
    expected_ids = tuple(sample.sample_id for sample in samples)
    if tuple(row.sample_id for row in group.rows) != expected_ids:
        raise ValueError("group_builder changed internal-evaluation sample order")
    if group.image_group_key != samples[0].image_group_key:
        raise ValueError("group_builder changed internal-evaluation image identity")
    if group.collective_padding:
        raise ValueError("internal evaluation must not materialize collective padding")
    for candidate in group.candidates:
        if candidate.target_conditioning_provider is not (
            identity.target_conditioning_provider
        ):
            raise ValueError(
                "group candidate provider differs from evaluation identity"
            )
        if candidate.attention is None:
            raise ValueError(
                "internal evaluation requires retained Adapter attention diagnostics"
            )
        if any(
            tensor.requires_grad or tensor.grad_fn is not None
            for tensor in (candidate.visual.main, *candidate.visual.deepstack)
        ):
            raise RuntimeError("internal-evaluation D retained an autograd graph")


def _validate_cross_group_contract(groups: tuple[SameImageReadoutGroup, ...]) -> None:
    first = groups[0]
    expected_projection = first.candidates[0].projection_identities
    expected_provider = first.candidates[0].target_conditioning_provider
    source_identities = tuple(group.source_visual_identity for group in groups)
    if len(set(source_identities)) != len(source_identities):
        raise ValueError("internal-evaluation source visual IDs must be distinct")
    for group in groups:
        for candidate in group.candidates:
            if candidate.projection_identities != expected_projection:
                raise ValueError("internal-evaluation groups mix projection identities")
            if candidate.target_conditioning_provider is not expected_provider:
                raise ValueError("internal-evaluation groups mix providers")


def _find_wrong_different_candidate(
    groups: tuple[SameImageReadoutGroup, ...],
    *,
    group_index: int,
    row_index: int,
    expected: RepresentationVisualTensorBundle,
) -> tuple[SameImageReadoutGroup, RepresentationCandidateObservation] | None:
    """Select a deterministic compatible control without constraining the suite."""

    for offset in range(1, len(groups)):
        group = groups[(group_index + offset) % len(groups)]
        preferred = row_index % len(group.candidates)
        candidate_indices = (
            preferred,
            *(index for index in range(len(group.candidates)) if index != preferred),
        )
        for candidate_index in candidate_indices:
            candidate = group.candidates[candidate_index]
            if _visual_bundle_contract_matches(candidate.visual, expected):
                return group, candidate
    return None


def _score_readout_control(
    family_adapter: QwenVLMFamilyAdapter,
    model: nn.Module,
    *,
    source: RepresentationVisualTensorBundle,
    row: RepresentationReadoutRow,
    candidate: RepresentationVisualTensorBundle | None,
) -> tuple[float, int]:
    blocked_mask = build_original_image_key_block_mask(
        attention_mask=row.attention_mask,
        original_image_token_indices=torch.tensor(
            row.source_positions,
            dtype=torch.long,
            device=row.attention_mask.device,
        ),
        block_query_start=row.source_image_block_query_start,
        block_query_end=row.source_image_block_query_end,
        dtype=source.main.dtype,
    )
    blocks = [_injected_block("source_image", row.source_positions, source)]
    if candidate is not None:
        blocks.append(_injected_block("focused_d", row.d_positions, candidate))
    with torch.no_grad():
        result = family_adapter.forward_injected(
            model,
            InjectedForwardRequest(
                input_ids=row.input_ids,
                attention_mask=blocked_mask,
                position_ids=row.position_ids,
                visual_blocks=tuple(blocks),
                use_cache=False,
            ),
        )
        labels = torch.tensor(
            row.loss_labels,
            dtype=torch.long,
            device=result.logits.device,
        ).unsqueeze(0)
        losses = causal_evidence_losses(result.logits, labels)
    return (
        float(losses.per_sample_token_mean_nll[0].float().item()),
        int(losses.valid_token_counts[0].item()),
    )


def _injected_block(
    kind: str,
    positions: tuple[int, ...],
    visual: RepresentationVisualTensorBundle,
) -> InjectedVisualBlock:
    return InjectedVisualBlock(
        kind=kind,
        positions=positions,
        embeddings=visual.main,
        deepstack=visual.deepstack,
        deepstack_positions=tuple(positions for _ in visual.deepstack),
    )


def _validate_materialized_request(
    request: InjectedForwardRequest,
    *,
    context: NativeDOnlyContext,
    observation: RepresentationVisualTensorBundle,
    generated_prefix: tuple[int, ...],
    teacher_forced_positions: tuple[int, ...] | None = None,
) -> None:
    if request.input_ids.shape[0] != 1:
        raise ValueError("native materialized request must have batch size one")
    context_length = int(context.input_ids.shape[1])
    if request.input_ids.shape[1] < context_length or not torch.equal(
        request.input_ids[:, :context_length],
        context.input_ids.to(device=request.input_ids.device),
    ):
        raise ValueError("native materialized request changed the fresh context prefix")
    context_positions = context.position_ids.to(device=request.position_ids.device)
    if not torch.equal(request.position_ids[..., :context_length], context_positions):
        raise ValueError("native materialized request changed context positions")
    if len(request.visual_blocks) != 1:
        raise ValueError("native materialized request must contain only focused D")
    block = request.visual_blocks[0]
    if block.kind != "focused_d" or block.positions != context.d_positions:
        raise ValueError("native materialized request has a non-D visual block")
    if block.deepstack_positions != tuple(
        context.d_positions for _ in observation.deepstack
    ):
        raise ValueError("native materialized request changed D-DeepStack positions")
    for actual, expected in zip(
        (block.embeddings, *block.deepstack),
        (observation.main, *observation.deepstack),
        strict=True,
    ):
        if actual is not expected and not torch.equal(actual, expected):
            raise ValueError("native materializer changed the atomic D observation")
    if teacher_forced_positions is None:
        if (
            generated_prefix
            and tuple(
                int(token_id)
                for token_id in request.input_ids[0, -len(generated_prefix) :].tolist()
            )
            != generated_prefix
        ):
            raise ValueError("generation materializer changed generated prefix tokens")
    else:
        actual_ids = tuple(
            int(request.input_ids[0, position].item())
            for position in teacher_forced_positions
        )
        if actual_ids != generated_prefix:
            raise ValueError("teacher-forced materializer changed value token IDs")


def _validated_generated_ids(value: object, *, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in value
        )
    ):
        raise ValueError(f"{name} must be non-empty non-negative token IDs")
    return value


def _deterministic_random_visual_bundle(
    visual: RepresentationVisualTensorBundle,
    *,
    seed: int,
) -> RepresentationVisualTensorBundle:
    hidden = int(visual.main.shape[-1])
    if hidden < 2:
        raise ValueError("deterministic random-D requires hidden size at least two")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(hidden, generator=generator, device="cpu")
    if torch.equal(permutation, torch.arange(hidden)):
        permutation = permutation.roll(1)
    permutation = permutation.to(device=visual.main.device)
    tensors = tuple(
        tensor.index_select(-1, permutation).detach()
        for tensor in (visual.main, *visual.deepstack)
    )
    return RepresentationVisualTensorBundle(
        main=tensors[0],
        deepstack=tensors[1:],
        branch_layers=visual.branch_layers,
        d_deepstack_active=visual.d_deepstack_active,
    )


def _random_control_seed(
    identity: RepresentationInternalEvaluationIdentity,
    sample_id: str,
) -> int:
    payload = (
        f"{DETERMINISTIC_RANDOM_D_ALGORITHM}\0{identity.evaluation_id}\0"
        f"{identity.random_seed}\0{sample_id}"
    ).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _sample_health(
    group: SameImageReadoutGroup,
    candidate: RepresentationCandidateObservation,
) -> RepresentationSampleHealthRecord:
    attention = candidate.attention
    if attention is None:  # guarded while validating the built group
        raise RuntimeError("candidate attention diagnostics disappeared")
    main_attention = attention_diagnostics(
        sub_slot_attention_weights=attention.main,
        topk=ATTENTION_TOPK,
    )
    if main_attention is None:
        raise RuntimeError("main Adapter attention is not evaluable")
    branches: list[RepresentationBranchHealthRecord] = []
    if candidate.visual.d_deepstack_active:
        for layer, d, source, weights in zip(
            candidate.visual.branch_layers,
            candidate.visual.deepstack,
            group.source_visual.deepstack,
            attention.deepstack,
            strict=True,
        ):
            branch_attention = attention_diagnostics(
                sub_slot_attention_weights=weights,
                topk=ATTENTION_TOPK,
            )
            if branch_attention is None:
                raise RuntimeError("D-DeepStack Adapter attention is not evaluable")
            branches.append(
                RepresentationBranchHealthRecord(
                    branch_layer=layer,
                    norm=norm_comparison_diagnostics(d, source),
                    attention=branch_attention,
                )
            )
    return RepresentationSampleHealthRecord(
        main=norm_comparison_diagnostics(
            candidate.visual.main, group.source_visual.main
        ),
        main_attention=main_attention,
        branches=tuple(branches),
    )


def _summarize_health(
    records: tuple[RepresentationInternalEvaluationSampleRecord, ...],
) -> RepresentationInternalHealthSummary:
    branch_layers = records[0].health.branches
    expected_layers = tuple(branch.branch_layer for branch in branch_layers)
    if any(
        tuple(branch.branch_layer for branch in record.health.branches)
        != expected_layers
        for record in records
    ):
        raise ValueError("sample health records mix DeepStack branch order")
    branches = tuple(
        RepresentationBranchHealthSummary(
            branch_layer=layer,
            representation=summarize_representation_health(
                tuple(record.health.branches[index].norm for record in records)
            ),
            attention=summarize_attention_diagnostics(
                tuple(record.health.branches[index].attention for record in records)
            ),
        )
        for index, layer in enumerate(expected_layers)
    )
    return RepresentationInternalHealthSummary(
        main=summarize_representation_health(
            tuple(record.health.main for record in records)
        ),
        main_attention=summarize_attention_diagnostics(
            tuple(record.health.main_attention for record in records)
        ),
        branches=branches,
    )


def _grouped_metrics(
    sample_groups: tuple[tuple[RepresentationTrainingSample, ...], ...],
    records: tuple[RepresentationInternalEvaluationSampleRecord, ...],
    nll_matrices: tuple[torch.Tensor, ...],
) -> tuple[RepresentationGroupedInternalMetrics, ...]:
    records_by_id = {record.sample_id: record for record in records}
    result: list[RepresentationGroupedInternalMetrics] = []
    for dimension in _GROUPING_DIMENSIONS:
        flat_labels = tuple(
            _metadata_label(getattr(records_by_id[sample.sample_id], dimension))
            for group in sample_groups
            for sample in group
        )
        nested_labels = tuple(
            tuple(
                _metadata_label(getattr(records_by_id[sample.sample_id], dimension))
                for sample in group
            )
            for group in sample_groups
        )
        readout_by_label = grouped_readout_metrics(
            tuple(record.nlls for record in records),
            flat_labels,
        )
        query_by_label = grouped_query_row_metrics(nll_matrices, nested_labels)
        if set(readout_by_label) != set(query_by_label):
            raise RuntimeError("grouped readout/query labels diverged")
        result.extend(
            RepresentationGroupedInternalMetrics(
                dimension=dimension,
                label=label,
                readout=readout_by_label[label],
                query=query_by_label[label],
            )
            for label in sorted(readout_by_label)
        )
    return tuple(result)


def _evaluate_native_case(
    case: NativeCounterfactualCase,
    *,
    causal_value_flip_evaluator: NativeCausalValueFlipEvaluator,
    free_continuation_evaluator: NativeFreeContinuationEvaluator,
) -> NativeCounterfactualEvaluationRecord:
    causal = causal_value_flip_evaluator(NativeCausalValueFlipRequest(case))
    if not isinstance(causal, NativeCausalValueFlipOutput):
        raise TypeError("causal value-flip callback returned an invalid output")
    if causal.case_id != case.case_id:
        raise ValueError("causal value-flip callback changed case identity")
    continuation_records: list[NativeFreeContinuationRecord] = []
    requests = (
        NativeFreeContinuationRequest(
            case_id=case.case_id,
            variant="value_a",
            expected_value=case.expected_value_a,
            context=case.context,
            observation_identity=case.observation_a_identity,
            observation=case.observation_a,
        ),
        NativeFreeContinuationRequest(
            case_id=case.case_id,
            variant="value_b",
            expected_value=case.expected_value_b,
            context=case.context,
            observation_identity=case.observation_b_identity,
            observation=case.observation_b,
        ),
    )
    for request in requests:
        output = free_continuation_evaluator(request)
        if not isinstance(output, NativeFreeContinuationOutput):
            raise TypeError("free-continuation callback returned an invalid output")
        if output.case_id != request.case_id or output.variant != request.variant:
            raise ValueError("free-continuation callback changed request identity")
        continuation_records.append(
            NativeFreeContinuationRecord(
                variant=output.variant,
                expected_value=request.expected_value,
                generated_token_ids=output.generated_token_ids,
                generated_text=output.generated_text,
                extracted_value=output.extracted_value,
                stop_reason=output.stop_reason,
                value_consistent=output.extracted_value == request.expected_value,
                healthy_termination=output.stop_reason == "natural_stop",
            )
        )
    return NativeCounterfactualEvaluationRecord(
        case_id=case.case_id,
        pair_identity=case.pair_identity,
        sample_a_id=case.sample_a_id,
        sample_b_id=case.sample_b_id,
        context_id=case.context.context_id,
        transcript_identity=case.context.transcript_identity,
        observation_a_identity=case.observation_a_identity,
        observation_b_identity=case.observation_b_identity,
        observation_a_log_odds_a_over_b=(causal.observation_a_log_odds_a_over_b),
        observation_b_log_odds_a_over_b=(causal.observation_b_log_odds_a_over_b),
        expected_direction_flip=(
            causal.observation_a_log_odds_a_over_b > 0
            and causal.observation_b_log_odds_a_over_b < 0
        ),
        log_odds_separation=(
            causal.observation_a_log_odds_a_over_b
            - causal.observation_b_log_odds_a_over_b
        ),
        continuations=tuple(continuation_records),
    )


def _evaluate_target_presence_case(
    case: NativeTargetPresenceCase,
    *,
    log_odds_evaluator: NativeBinaryLogOddsEvaluator,
    free_continuation_evaluator: NativeFreeContinuationEvaluator,
) -> NativeTargetPresenceEvaluationRecord:
    zero_positive = _zero_visual_bundle(case.positive_observation)
    zero_negative = _zero_visual_bundle(case.negative_observation)

    def score(
        *, context: NativeDOnlyContext, observation: RepresentationVisualTensorBundle
    ) -> float:
        output = log_odds_evaluator(
            NativeBinaryLogOddsRequest(
                case_id=case.case_id,
                context=context,
                observation=observation,
                positive_value=case.present_value,
                negative_value=case.not_present_value,
            )
        )
        if not isinstance(output, NativeBinaryLogOddsOutput):
            raise TypeError("target-presence log-odds callback returned another type")
        if output.case_id != case.case_id:
            raise ValueError("target-presence log-odds callback changed case identity")
        return output.log_odds_positive_over_negative

    actual_positive = score(
        context=case.positive_context,
        observation=case.positive_observation,
    )
    actual_negative = score(
        context=case.negative_context,
        observation=case.negative_observation,
    )
    zero_positive_score = score(
        context=case.positive_context,
        observation=zero_positive,
    )
    zero_negative_score = score(
        context=case.negative_context,
        observation=zero_negative,
    )

    continuation_records: list[NativeTargetPresenceContinuationRecord] = []
    continuation_inputs = (
        (
            "positive_target",
            case.present_value,
            case.positive_context,
            case.positive_observation_identity,
            case.positive_observation,
            "value_a",
        ),
        (
            "negative_target",
            case.not_present_value,
            case.negative_context,
            case.negative_observation_identity,
            case.negative_observation,
            "value_b",
        ),
    )
    for (
        variant,
        expected,
        context,
        observation_identity,
        observation,
        callback_variant,
    ) in continuation_inputs:
        output = free_continuation_evaluator(
            NativeFreeContinuationRequest(
                case_id=case.case_id,
                variant=callback_variant,
                expected_value=expected,
                context=context,
                observation_identity=observation_identity,
                observation=observation,
            )
        )
        if not isinstance(output, NativeFreeContinuationOutput):
            raise TypeError(
                "target-presence continuation callback returned another type"
            )
        if output.case_id != case.case_id or output.variant != callback_variant:
            raise ValueError("target-presence continuation callback changed identity")
        continuation_records.append(
            NativeTargetPresenceContinuationRecord(
                variant=variant,
                expected_value=expected,
                generated_token_ids=output.generated_token_ids,
                generated_text=output.generated_text,
                extracted_value=output.extracted_value,
                stop_reason=output.stop_reason,
                value_consistent=output.extracted_value == expected,
                healthy_termination=output.stop_reason == "natural_stop",
            )
        )

    return NativeTargetPresenceEvaluationRecord(
        case_id=case.case_id,
        pair_identity=case.pair_identity,
        source_sample_id=case.source_sample_id,
        source_image_sha256=case.source_image_sha256,
        positive_target=case.positive_target,
        negative_target=case.negative_target,
        positive_context_id=case.positive_context.context_id,
        negative_context_id=case.negative_context.context_id,
        positive_observation_identity=case.positive_observation_identity,
        negative_observation_identity=case.negative_observation_identity,
        actual_positive_log_odds_present=actual_positive,
        actual_negative_log_odds_present=actual_negative,
        zero_d_positive_log_odds_present=zero_positive_score,
        zero_d_negative_log_odds_present=zero_negative_score,
        actual_separation=actual_positive - actual_negative,
        positive_d_contribution=actual_positive - zero_positive_score,
        negative_d_false_positive_amplification=(actual_negative - zero_negative_score),
        actual_direction_correct=(actual_positive > 0 and actual_negative < 0),
        negative_false_positive=actual_negative > 0,
        continuations=tuple(continuation_records),
    )


def _summarize_counterfactuals(
    records: tuple[NativeCounterfactualEvaluationRecord, ...],
) -> NativeCounterfactualSummary:
    if not records:
        raise ValueError("counterfactual summary requires records")
    continuations = tuple(
        continuation for record in records for continuation in record.continuations
    )
    return NativeCounterfactualSummary(
        pair_count=len(records),
        expected_direction_flip_rate=sum(
            record.expected_direction_flip for record in records
        )
        / len(records),
        continuation_accuracy=sum(
            continuation.value_consistent for continuation in continuations
        )
        / len(continuations),
        healthy_termination_rate=sum(
            continuation.healthy_termination for continuation in continuations
        )
        / len(continuations),
    )


def _zero_visual_bundle(
    observation: RepresentationVisualTensorBundle,
) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.zeros_like(observation.main),
        deepstack=tuple(torch.zeros_like(branch) for branch in observation.deepstack),
        branch_layers=observation.branch_layers,
        d_deepstack_active=observation.d_deepstack_active,
    )


def _summarize_target_presence(
    records: tuple[NativeTargetPresenceEvaluationRecord, ...],
) -> NativeTargetPresenceSummary:
    if not records:
        raise ValueError("target-presence summary requires records")
    continuations = tuple(
        continuation for record in records for continuation in record.continuations
    )
    return NativeTargetPresenceSummary(
        pair_count=len(records),
        actual_direction_accuracy=sum(
            record.actual_direction_correct for record in records
        )
        / len(records),
        negative_false_positive_rate=sum(
            record.negative_false_positive for record in records
        )
        / len(records),
        mean_actual_separation=sum(record.actual_separation for record in records)
        / len(records),
        mean_positive_d_contribution=sum(
            record.positive_d_contribution for record in records
        )
        / len(records),
        mean_negative_d_false_positive_amplification=sum(
            record.negative_d_false_positive_amplification for record in records
        )
        / len(records),
        continuation_accuracy=sum(
            continuation.value_consistent for continuation in continuations
        )
        / len(continuations),
        healthy_termination_rate=sum(
            continuation.healthy_termination for continuation in continuations
        )
        / len(continuations),
    )


def _assert_visual_bundle_contract(
    actual: RepresentationVisualTensorBundle,
    expected: RepresentationVisualTensorBundle,
    *,
    name: str,
) -> None:
    if actual.d_deepstack_active != expected.d_deepstack_active:
        raise ValueError(f"{name} D-DeepStack activity differs")
    if actual.branch_layers != expected.branch_layers:
        raise ValueError(f"{name} branch layer order differs")
    for actual_tensor, expected_tensor in zip(
        (actual.main, *actual.deepstack),
        (expected.main, *expected.deepstack),
        strict=True,
    ):
        if (
            actual_tensor.shape != expected_tensor.shape
            or actual_tensor.dtype != expected_tensor.dtype
            or actual_tensor.device != expected_tensor.device
        ):
            raise ValueError(f"{name} shape/dtype/device contract differs")


def _visual_bundle_contract_matches(
    actual: RepresentationVisualTensorBundle,
    expected: RepresentationVisualTensorBundle,
) -> bool:
    if actual.d_deepstack_active != expected.d_deepstack_active:
        return False
    if actual.branch_layers != expected.branch_layers:
        return False
    return all(
        actual_tensor.shape == expected_tensor.shape
        and actual_tensor.dtype == expected_tensor.dtype
        and actual_tensor.device == expected_tensor.device
        for actual_tensor, expected_tensor in zip(
            (actual.main, *actual.deepstack),
            (expected.main, *expected.deepstack),
            strict=True,
        )
    )


def _assert_adapter_state_unchanged(
    state: tuple[tuple[nn.Parameter, bool, torch.Tensor | None], ...],
) -> None:
    for parameter, requires_grad, gradient in state:
        if parameter.requires_grad != requires_grad:
            raise RuntimeError("internal evaluation changed Adapter trainability")
        if parameter.grad is not gradient:
            raise RuntimeError("internal evaluation replaced an Adapter gradient")


def _metadata_label(value: str | None) -> str:
    return _UNSPECIFIED_GROUP_LABEL if value is None else value


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"internal-evaluation report contains {type(value).__name__}")


def _non_empty_text(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _lowercase_sha256(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _non_negative_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _finite_float(value: object, *, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not (math.isfinite(float(value)))
    ):
        raise ValueError(f"{name} must be finite")


__all__ = [
    "ATTENTION_TOPK",
    "CAUSAL_VALUE_FLIP_LOG_ODDS_CONTRACT",
    "DETERMINISTIC_RANDOM_D_ALGORITHM",
    "READOUT_FORWARD_PATH",
    "READOUT_MASK_MODE",
    "READOUT_POSITION_SOURCE",
    "READOUT_VISUAL_SWAP_UNIT",
    "TARGET_PRESENCE_LOG_ODDS_CONTRACT",
    "REPRESENTATION_INTERNAL_EVALUATION_ARTIFACT_SCHEMA_VERSION",
    "REPRESENTATION_INTERNAL_EVALUATION_SCHEMA_VERSION",
    "NativeCausalValueFlipEvaluator",
    "NativeCausalValueFlipOutput",
    "NativeCausalValueFlipRequest",
    "NativeBinaryLogOddsEvaluator",
    "NativeBinaryLogOddsOutput",
    "NativeBinaryLogOddsRequest",
    "NativeCounterfactualCase",
    "NativeCounterfactualEvaluationRecord",
    "NativeCounterfactualSummary",
    "NativeContinuationCacheParity",
    "NativeDOnlyContext",
    "NativeGenerationForward",
    "NativeInjectedRequestMaterializer",
    "NativeFreeContinuationEvaluator",
    "NativeFreeContinuationOutput",
    "NativeFreeContinuationRecord",
    "NativeFreeContinuationRequest",
    "NativeTeacherForcedForward",
    "NativeTargetPresenceCase",
    "NativeTargetPresenceContinuationRecord",
    "NativeTargetPresenceEvaluationRecord",
    "NativeTargetPresenceSummary",
    "InjectedNativeCounterfactualEvaluator",
    "RepresentationBranchHealthRecord",
    "RepresentationBranchHealthSummary",
    "RepresentationGroupedInternalMetrics",
    "RepresentationInternalEvaluationArtifact",
    "RepresentationInternalEvaluationGroupRecord",
    "RepresentationInternalEvaluationIdentity",
    "RepresentationInternalEvaluationReport",
    "RepresentationInternalEvaluationSampleRecord",
    "RepresentationInternalHealthSummary",
    "RepresentationReadoutControlIdentity",
    "RepresentationReadoutExecutionContract",
    "RepresentationSampleHealthRecord",
    "run_representation_internal_evaluation",
    "create_injected_native_counterfactual_evaluator",
    "save_representation_internal_evaluation_report_atomic",
]
