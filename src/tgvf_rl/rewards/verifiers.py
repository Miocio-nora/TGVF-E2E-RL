"""Rule-first answer verification with an explicit 72B fallback boundary."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import re

from tgvf_rl.contracts.errors import ContractUnsetError, IdentityMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.judges import JudgeProvider, JudgeRequest

from .schema import (
    AnswerTaskKind,
    AnswerVerificationResult,
    NormalizationSpec,
    RewardContext,
)


_MCQ_CANONICAL_LETTER = re.compile(
    r"^\s*(?:[\(\[]\s*([A-H])\s*[\)\]]|([A-H])\s*[.:]|([A-H])\s*$)",
    re.IGNORECASE,
)
_MCQ_ANSWER_MARKER = re.compile(
    r"\b(?:final\s+answer|answer)\s*(?:is|:|=|-)\s*"
    r"(?:(?:option|choice)\s*)?[\(\[]?\s*([A-H])\s*[\)\]]?"
    r"(?=\s|[.,:;!?]|$)",
    re.IGNORECASE,
)
_MCQ_NAMED_OPTION = re.compile(
    r"\b(?:option|choice|range)\s*(?:is\s*)?[\(\[]?\s*([A-H])\s*[\)\]]?"
    r"(?=\s|[.,:;!?]|$)",
    re.IGNORECASE,
)
_MCQ_MARKDOWN = re.compile(r"[*_`]")
_QWEN_TERMINAL_TEXT = re.compile(r"(?:(?:<\|im_end\|>|<\|endoftext\|>)\s*)+$")
_LATEX_FRACTION = re.compile(
    r"^\s*\\frac\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}\s*$"
)


@dataclass(frozen=True, slots=True)
class RuleFirstAnswerVerifier:
    """Apply deterministic rules first and invoke the judge only when allowed."""

    rule_identity: ArtifactIdentity
    normalization: NormalizationSpec
    judge: JudgeProvider
    judge_prompt_identity: ArtifactIdentity
    judge_model_identity: ArtifactIdentity
    judge_service_identity: ArtifactIdentity
    judge_sampling_identity: ArtifactIdentity
    judge_calibration_identity: ArtifactIdentity

    def verify(self, context: RewardContext) -> AnswerVerificationResult:
        if context.expected_answer is None:
            raise ContractUnsetError("Pilot answer verification requires ground truth")
        if not context.has_valid_final_answer:
            return AnswerVerificationResult(
                False,
                "missing_final_answer",
                "trajectory has no valid final answer",
                self.rule_identity,
            )

        if context.task_kind is AnswerTaskKind.MULTIPLE_CHOICE:
            return self._verify_multiple_choice(context)

        candidate = _normalize_answer(context.candidate_answer, self.normalization)
        expected = _normalize_answer(context.expected_answer, self.normalization)
        if candidate == expected:
            return AnswerVerificationResult(
                True, "normalized_exact", "normalized answers match", self.rule_identity
            )

        if context.task_kind is AnswerTaskKind.MATH:
            numeric = _numeric_equivalence(candidate, expected)
            if numeric is not None:
                return AnswerVerificationResult(
                    numeric,
                    "math_numeric_rule",
                    f"numeric_equivalence={numeric}",
                    self.rule_identity,
                )

        # Math expressions not decided by the deterministic verifier and open
        # VQA exact mismatches use the separately identified formal-Pilot judge.
        return self._judge_fallback(context)

    def _verify_multiple_choice(
        self, context: RewardContext
    ) -> AnswerVerificationResult:
        assert context.expected_answer is not None
        candidate = _multiple_choice_letter(context.candidate_answer)
        expected = _multiple_choice_letter(context.expected_answer)
        if candidate is None or expected is None:
            correct = _normalize_answer(
                context.candidate_answer, self.normalization
            ) == _normalize_answer(context.expected_answer, self.normalization)
            evidence = "letter_parse_failed; normalized_exact=" + str(correct)
        else:
            correct = candidate == expected
            evidence = f"candidate={candidate}; expected={expected}"
        return AnswerVerificationResult(
            correct,
            "multiple_choice_rule",
            evidence,
            self.rule_identity,
        )

    def _judge_fallback(self, context: RewardContext) -> AnswerVerificationResult:
        assert context.expected_answer is not None
        request_payload = {
            "schema": "policy-pilot-v1-answer-judge-request-v1",
            "sample_id": context.sample_id,
            "question": context.question,
            "candidate": context.candidate_answer,
            "reference": context.expected_answer,
            "task_kind": context.task_kind.value,
            "prompt_sha256": self.judge_prompt_identity.sha256,
        }
        request_id = hashlib.sha256(
            json.dumps(
                request_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        result = self.judge.judge(
            JudgeRequest(
                request_id=request_id,
                task_kind=context.task_kind.value,
                question=context.question,
                candidate_answer=context.candidate_answer,
                reference_answer=context.expected_answer,
                prompt_identity=self.judge_prompt_identity,
            )
        )
        expected_identities = (
            ("model", result.model_identity, self.judge_model_identity),
            ("service", result.service_identity, self.judge_service_identity),
            ("sampling", result.sampling_identity, self.judge_sampling_identity),
            ("calibration", result.calibration_identity, self.judge_calibration_identity),
        )
        for name, actual, expected in expected_identities:
            if actual != expected:
                raise IdentityMismatchError(
                    f"answer judge {name} identity differs from Pilot binding"
                )
        if result.score not in {0.0, 1.0}:
            raise ValueError(
                "formal Pilot judge must return a calibrated binary 0/1 verdict"
            )
        return AnswerVerificationResult(
            bool(result.score),
            "qwen2.5_72b_semantic_fallback",
            result.rationale,
            result.model_identity,
            judge_usage=result.usage,
        )


def _multiple_choice_letter(text: str) -> str | None:
    value = _QWEN_TERMINAL_TEXT.sub("", text.strip()).strip()
    value = _unwrap_answer(value)
    value = _QWEN_TERMINAL_TEXT.sub("", value).strip()
    value = _MCQ_MARKDOWN.sub("", value)

    # A bare choice is decisive only when it starts the final non-empty line.
    # Looking at the entire response would mistake an option-like line in the
    # reasoning prefix for the model's final decision.
    decision_line = next(
        (line.strip() for line in reversed(value.splitlines()) if line.strip()),
        "",
    )
    canonical = _MCQ_CANONICAL_LETTER.match(decision_line)
    if canonical is not None:
        return next(group.upper() for group in canonical.groups() if group is not None)

    answer_markers = tuple(_MCQ_ANSWER_MARKER.finditer(value))
    if answer_markers:
        return answer_markers[-1].group(1).upper()

    named_options = tuple(_MCQ_NAMED_OPTION.finditer(value))
    if named_options:
        return named_options[-1].group(1).upper()
    return None


def _normalize_answer(text: str, spec: NormalizationSpec) -> str:
    value = _unwrap_answer(text)
    value = value.strip() if spec.strip else value
    value = value.casefold() if spec.casefold else value
    if spec.collapse_whitespace:
        value = re.sub(r"\s+", " ", value)
    return value


def _unwrap_answer(text: str) -> str:
    value = text.strip()
    answer_match = re.fullmatch(r"<answer>\s*(.*?)\s*</answer>", value, re.DOTALL)
    if answer_match is not None:
        value = answer_match.group(1).strip()
    boxed_match = re.fullmatch(r"\\boxed\s*\{(.*)\}", value, re.DOTALL)
    if boxed_match is not None:
        value = boxed_match.group(1).strip()
    return value


def _numeric_equivalence(candidate: str, expected: str) -> bool | None:
    left = _parse_number(candidate)
    right = _parse_number(expected)
    if left is None or right is None:
        return None
    return left == right


def _parse_number(value: str) -> Fraction | None:
    compact = value.strip().replace(",", "")
    latex = _LATEX_FRACTION.fullmatch(compact)
    if latex is not None:
        denominator = int(latex.group(2))
        if denominator == 0:
            return None
        return Fraction(int(latex.group(1)), denominator)
    try:
        if "/" in compact:
            numerator, denominator = compact.split("/", 1)
            return Fraction(int(numerator.strip()), int(denominator.strip()))
        return Fraction(Decimal(compact))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


__all__ = ["RuleFirstAnswerVerifier"]
