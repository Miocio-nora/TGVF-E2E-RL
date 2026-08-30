"""Native-D execution and diagnostic materialization.

This module owns exact injected-Qwen execution, cache-parity evaluation, and
the native counterfactual/target-presence records derived from those callbacks.
Generic readout metrics, report orchestration, and artifact publication remain
in the public evaluator facade.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tgvf_rl.public_api_compat import (
    rebind_public_class,
    rebind_public_function,
)
from tgvf_rl.qwen.base import (
    CachedTokenForwardRequest,
    InjectedForwardRequest,
    QwenVLMFamilyAdapter,
)

from .internal_evaluation_contract import (
    ContinuationStopReason,
    NativeBinaryLogOddsEvaluator,
    NativeBinaryLogOddsOutput,
    NativeBinaryLogOddsRequest,
    NativeCausalValueFlipEvaluator,
    NativeCausalValueFlipOutput,
    NativeCausalValueFlipRequest,
    NativeContinuationCacheParity,
    NativeCounterfactualCase,
    NativeCounterfactualEvaluationRecord,
    NativeCounterfactualSummary,
    NativeDOnlyContext,
    NativeFreeContinuationEvaluator,
    NativeFreeContinuationOutput,
    NativeFreeContinuationRecord,
    NativeFreeContinuationRequest,
    NativeGenerationForward,
    NativeInjectedRequestMaterializer,
    NativeTargetPresenceCase,
    NativeTargetPresenceContinuationRecord,
    NativeTargetPresenceEvaluationRecord,
    NativeTargetPresenceSummary,
    NativeTeacherForcedForward,
    _non_empty_text,
)
from .readout import (
    RepresentationVisualTensorBundle,
    assert_frozen_deterministic_readout_model,
)


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

_PUBLIC_INTERNAL_EVALUATION_MODULE = (
    "tgvf_rl.representation.training.internal_evaluation"
)
_NATIVE_RUNTIME_FUNCTIONS = (
    _greedy_token_id,
    create_injected_native_counterfactual_evaluator,
    _validate_materialized_request,
    _validated_generated_ids,
    _evaluate_native_case,
    _evaluate_target_presence_case,
    _summarize_counterfactuals,
    _zero_visual_bundle,
    _summarize_target_presence,
)

# Preserve the established facade and pickle coordinates without importing the
# facade back into this runtime leaf.
rebind_public_class(
    InjectedNativeCounterfactualEvaluator,
    implementation_module=__name__,
    public_module=_PUBLIC_INTERNAL_EVALUATION_MODULE,
)
for _function in _NATIVE_RUNTIME_FUNCTIONS:
    rebind_public_function(
        _function,
        implementation_module=__name__,
        public_module=_PUBLIC_INTERNAL_EVALUATION_MODULE,
    )
del _function


__all__ = [
    "InjectedNativeCounterfactualEvaluator",
    "create_injected_native_counterfactual_evaluator",
]
