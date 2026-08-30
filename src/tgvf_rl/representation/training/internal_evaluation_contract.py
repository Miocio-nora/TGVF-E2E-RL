"""Stable declarative contracts for representation internal evaluation.

This dependency-light module owns only validated identities, native callback
contracts, and report value objects. Model execution, metric computation, and
artifact publication remain outside this module. The public evaluator module
re-exports these exact objects to preserve established import and pickle
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import FunctionType
from typing import Literal, Protocol

import torch

from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.qwen.base import InjectedForwardRequest

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
)
from .readout import RepresentationVisualTensorBundle


REPRESENTATION_INTERNAL_EVALUATION_SCHEMA_VERSION = (
    "representation_internal_evaluation_v1"
)
DETERMINISTIC_RANDOM_D_ALGORITHM = "shared_feature_permutation_v1"
ATTENTION_TOPK = 5
READOUT_FORWARD_PATH = "family_injected_language_model_no_vision_rerun_v1"
READOUT_MASK_MODE = "causal_post_evidence_original_image_key_block_v1"
READOUT_POSITION_SOURCE = "family_native_group_builder_v1"
READOUT_VISUAL_SWAP_UNIT = "atomic_main_and_all_deepstack_v1"
CAUSAL_VALUE_FLIP_LOG_ODDS_CONTRACT = "summed_teacher_forced_token_logprob_v1"
TARGET_PRESENCE_LOG_ODDS_CONTRACT = "mean_teacher_forced_token_logprob_v1"

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

_PUBLIC_INTERNAL_EVALUATION_MODULE = (
    "tgvf_rl.representation.training.internal_evaluation"
)
_INTERNAL_EVALUATION_CONTRACT_TYPES = (
    RepresentationInternalEvaluationIdentity,
    NativeDOnlyContext,
    NativeCounterfactualCase,
    NativeCausalValueFlipRequest,
    NativeCausalValueFlipOutput,
    NativeCausalValueFlipEvaluator,
    NativeBinaryLogOddsRequest,
    NativeBinaryLogOddsOutput,
    NativeBinaryLogOddsEvaluator,
    NativeTeacherForcedForward,
    NativeGenerationForward,
    NativeInjectedRequestMaterializer,
    NativeFreeContinuationRequest,
    NativeFreeContinuationOutput,
    NativeContinuationCacheParity,
    NativeFreeContinuationEvaluator,
    NativeTargetPresenceCase,
    RepresentationReadoutControlIdentity,
    RepresentationBranchHealthRecord,
    RepresentationSampleHealthRecord,
    RepresentationInternalEvaluationSampleRecord,
    RepresentationInternalEvaluationGroupRecord,
    RepresentationGroupedInternalMetrics,
    RepresentationBranchHealthSummary,
    RepresentationInternalHealthSummary,
    RepresentationReadoutExecutionContract,
    NativeFreeContinuationRecord,
    NativeCounterfactualEvaluationRecord,
    NativeCounterfactualSummary,
    NativeTargetPresenceContinuationRecord,
    NativeTargetPresenceEvaluationRecord,
    NativeTargetPresenceSummary,
    RepresentationInternalEvaluationReport,
)

# These types historically lived in internal_evaluation. Preserve their public
# and pickle coordinates while ownership moves behind that facade.
for _contract_type in _INTERNAL_EVALUATION_CONTRACT_TYPES:
    _contract_type.__module__ = _PUBLIC_INTERNAL_EVALUATION_MODULE
    for _member in vars(_contract_type).values():
        _functions = (
            (_member.fget, _member.fset, _member.fdel)
            if isinstance(_member, property)
            else (_member,)
        )
        for _function in _functions:
            if (
                isinstance(_function, FunctionType)
                and _function.__module__ == __name__
            ):
                _function.__module__ = _PUBLIC_INTERNAL_EVALUATION_MODULE
for _helper in (
    _assert_visual_bundle_contract,
    _non_empty_text,
    _lowercase_sha256,
    _non_negative_int,
    _finite_float,
):
    _helper.__module__ = _PUBLIC_INTERNAL_EVALUATION_MODULE
del _contract_type, _function, _functions, _helper, _member


__all__ = [
    "ATTENTION_TOPK",
    "CAUSAL_VALUE_FLIP_LOG_ODDS_CONTRACT",
    "ContinuationStopReason",
    "ContinuationVariant",
    "DETERMINISTIC_RANDOM_D_ALGORITHM",
    "NativeBinaryLogOddsEvaluator",
    "NativeBinaryLogOddsOutput",
    "NativeBinaryLogOddsRequest",
    "NativeCausalValueFlipEvaluator",
    "NativeCausalValueFlipOutput",
    "NativeCausalValueFlipRequest",
    "NativeContinuationCacheParity",
    "NativeCounterfactualCase",
    "NativeCounterfactualEvaluationRecord",
    "NativeCounterfactualSummary",
    "NativeDOnlyContext",
    "NativeFreeContinuationEvaluator",
    "NativeFreeContinuationOutput",
    "NativeFreeContinuationRecord",
    "NativeFreeContinuationRequest",
    "NativeGenerationForward",
    "NativeInjectedRequestMaterializer",
    "NativeTargetPresenceCase",
    "NativeTargetPresenceContinuationRecord",
    "NativeTargetPresenceEvaluationRecord",
    "NativeTargetPresenceSummary",
    "NativeTeacherForcedForward",
    "READOUT_FORWARD_PATH",
    "READOUT_MASK_MODE",
    "READOUT_POSITION_SOURCE",
    "READOUT_VISUAL_SWAP_UNIT",
    "REPRESENTATION_INTERNAL_EVALUATION_SCHEMA_VERSION",
    "RepresentationBranchHealthRecord",
    "RepresentationBranchHealthSummary",
    "RepresentationGroupedInternalMetrics",
    "RepresentationInternalEvaluationGroupRecord",
    "RepresentationInternalEvaluationIdentity",
    "RepresentationInternalEvaluationReport",
    "RepresentationInternalEvaluationSampleRecord",
    "RepresentationInternalHealthSummary",
    "RepresentationReadoutControlIdentity",
    "RepresentationReadoutExecutionContract",
    "RepresentationSampleHealthRecord",
    "TARGET_PRESENCE_LOG_ODDS_CONTRACT",
    "TargetPresenceVariant",
]
