"""Deterministic reward plumbing with explicitly injected components."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from tgvf_rl.contracts.errors import ContractUnsetError, IdentityMismatchError

from .schema import (
    NormalizationSpec,
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
            if not 0.0 <= raw <= 1.0:
                raise ValueError(
                    f"reward component {component_spec.name!r} returned score outside [0,1]"
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
