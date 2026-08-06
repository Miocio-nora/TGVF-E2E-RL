"""Deterministic reward plumbing with explicitly injected components."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from tgvf_rl.contracts.errors import ContractUnsetError, IdentityMismatchError

from .schema import (
    AnswerVerifier,
    NormalizationSpec,
    PILOT_REWARD_EQUATION_DEEPEYES_MATH,
    PilotRewardSpec,
    RewardComponent,
    RewardComponentResult,
    RewardContext,
    RewardPipelineSpec,
    RewardResult,
)


@dataclass(frozen=True, slots=True)
class ExactTextVerifier:
    normalization: NormalizationSpec

    def score(self, context: RewardContext) -> tuple[float, str]:
        if context.expected_answer is None:
            raise ContractUnsetError("exact-text reward requires an expected answer")
        candidate = self._normalize(context.candidate_answer)
        expected = self._normalize(context.expected_answer)
        matched = candidate == expected
        return float(matched), f"normalized_exact_match={matched}"

    def _normalize(self, text: str) -> str:
        value = text.strip() if self.normalization.strip else text
        value = value.casefold() if self.normalization.casefold else value
        if self.normalization.collapse_whitespace:
            value = re.sub(r"\s+", " ", value)
        return value


class RewardPipeline:
    def __init__(
        self, spec: RewardPipelineSpec, components: Mapping[str, RewardComponent]
    ) -> None:
        expected = {component.name for component in spec.components}
        if set(components) != expected:
            raise IdentityMismatchError(
                f"reward component bindings differ: expected={sorted(expected)} actual={sorted(components)}"
            )
        self.spec = spec
        self.components = dict(components)

    def score(self, context: RewardContext) -> RewardResult:
        results: list[RewardComponentResult] = []
        for component_spec in self.spec.components:
            raw, evidence = self.components[component_spec.name].score(context)
            if not component_spec.minimum_score <= raw <= component_spec.maximum_score:
                raise ValueError(
                    f"reward component {component_spec.name!r} returned score outside "
                    f"[{component_spec.minimum_score},{component_spec.maximum_score}]"
                )
            results.append(
                RewardComponentResult(
                    name=component_spec.name,
                    raw_score=raw,
                    weighted_score=raw * component_spec.weight,
                    verifier_identity=component_spec.verifier_identity,
                    evidence=evidence,
                )
            )
        return RewardResult(
            total=sum(item.weighted_score for item in results),
            components=tuple(results),
            pipeline_identity=self.spec.identity,
        )


class PilotRewardPipeline:
    """Exact `0.8 answer + 0.2 format + 1.2 conditional-tool` reward."""

    def __init__(self, spec: PilotRewardSpec, answer_verifier: AnswerVerifier) -> None:
        if not isinstance(spec, PilotRewardSpec):
            raise TypeError("spec must be PilotRewardSpec")
        if not hasattr(answer_verifier, "verify"):
            raise TypeError("answer_verifier must implement verify")
        self.spec = spec
        self.answer_verifier = answer_verifier

    def score(self, context: RewardContext) -> RewardResult:
        verification = self.answer_verifier.verify(context)
        answer_score = float(verification.correct)
        format_valid = (
            context.protocol_valid
            and context.has_valid_final_answer
            and bool(context.candidate_answer.strip())
        )
        format_score = 0.0 if format_valid else -1.0
        equation_route, weights = self.spec.equation_for_context(context)
        answer_weight, format_weight, conditional_tool_weight = weights
        deepeyes_math_route = equation_route == PILOT_REWARD_EQUATION_DEEPEYES_MATH
        if deepeyes_math_route:
            conditional_tool_score = 0.0
        else:
            conditional_tool_score = float(
                verification.correct and context.successful_tgvf_observation_count >= 1
            )
        error_summary = ",".join(context.tool_error_codes) or "none"
        components = (
            RewardComponentResult(
                name="answer_reward",
                raw_score=answer_score,
                weighted_score=answer_weight * answer_score,
                verifier_identity=verification.verifier_identity,
                evidence=(
                    f"route={verification.route}; {verification.evidence}; "
                    f"equation={equation_route}"
                ),
            ),
            RewardComponentResult(
                name="format_reward",
                raw_score=format_score,
                weighted_score=format_weight * format_score,
                verifier_identity=self.spec.format_verifier_identity,
                evidence=(
                    f"protocol_valid={context.protocol_valid}; "
                    f"has_valid_final_answer={context.has_valid_final_answer}"
                ),
            ),
            RewardComponentResult(
                name="conditional_tool_reward",
                raw_score=conditional_tool_score,
                weighted_score=(conditional_tool_weight * conditional_tool_score),
                verifier_identity=self.spec.tool_verifier_identity,
                evidence=(
                    f"answer_correct={verification.correct}; "
                    "successful_tgvf_observations="
                    f"{context.successful_tgvf_observation_count}; "
                    f"tool_errors={error_summary}"
                ),
            ),
        )
        return RewardResult(
            total=sum(component.weighted_score for component in components),
            components=components,
            pipeline_identity=self.spec.pipeline_identity,
            answer_verification=verification,
        )
