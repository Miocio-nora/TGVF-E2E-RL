"""Deterministic answer parsing and verification for T1 policy selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
import re
from typing import Any

from tgvf_rl.public_api_compat import rebind_public_class, rebind_public_function

from .policy_selection import SelectionSource
from .policy_selection_config_schema import (
    T1_INSTRUCT_ANSWER_PARSER,
    T1_THINKING_ANSWER_PARSER,
)
from .policy_selection_config_values import _required_int

_QWEN_TERMINAL_TEXT = re.compile(r"(?:<\|(?:im_end|endoftext)\|>\s*)+$")
_ANSWER_TAG = re.compile(r"^<answer>\s*(.*?)\s*</answer>$", re.DOTALL)
_BOXED = re.compile(r"^\\boxed\s*\{(.*)\}$", re.DOTALL)
_LATEX_FRACTION = re.compile(r"^\\frac\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}$")
_ARXIV_CANONICAL = re.compile(
    r"^\s*(?:[\(\[]\s*([A-Z])\s*[\)\]]|([A-Z])\s*[\).:]?|"
    r"(?:option|choice)\s*[\(\[]?\s*([A-Z])\s*[\)\]]?)\s*$",
    re.IGNORECASE,
)
_ARXIV_ANSWER_MARKER = re.compile(
    r"\b(?:final\s+answer|answer)\s*(?:is|:|=|-)\s*"
    r"(?:(?:option|choice)\s*)?[\(\[]?\s*([A-Z])\s*[\)\]]?"
    r"(?=\s|[.,:;!?]|$)",
    re.IGNORECASE,
)
_ARXIV_NAMED_OPTION = re.compile(
    r"\b(?:option|choice)\s*(?:is\s*)?[\(\[]?\s*([A-Z])\s*[\)\]]?"
    r"(?=\s|[.,:;!?]|$)",
    re.IGNORECASE,
)


def extract_final_answer(raw_text: str) -> str | None:
    """Return the non-empty suffix after the last sampled ``</think>``."""

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    marker = "</think>"
    index = raw_text.rfind(marker)
    if index < 0:
        return None
    suffix = raw_text[index + len(marker) :].strip()
    suffix = _QWEN_TERMINAL_TEXT.sub("", suffix).strip()
    return suffix or None


def extract_direct_completion(raw_text: str) -> str | None:
    """Return an Instruct completion without inventing a reasoning boundary."""

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    answer = _QWEN_TERMINAL_TEXT.sub("", raw_text.strip()).strip()
    return answer or None


def parse_t1_answer(raw_text: str, *, answer_parser: str) -> str | None:
    """Dispatch final-answer extraction by the run-bound native dialect."""

    if answer_parser == T1_THINKING_ANSWER_PARSER:
        return extract_final_answer(raw_text)
    if answer_parser == T1_INSTRUCT_ANSWER_PARSER:
        return extract_direct_completion(raw_text)
    raise ValueError(f"unsupported T1 answer parser: {answer_parser!r}")


class VerificationOutcome(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    SEMANTIC_REQUIRED = "semantic_required"


@dataclass(frozen=True, slots=True)
class DeterministicVerification:
    outcome: VerificationOutcome
    route: str
    evidence: str

    @property
    def correct(self) -> bool | None:
        if self.outcome is VerificationOutcome.CORRECT:
            return True
        if self.outcome is VerificationOutcome.INCORRECT:
            return False
        return None


def _unwrap_answer(text: str) -> str:
    value = _QWEN_TERMINAL_TEXT.sub("", text.strip()).strip()
    answer = _ANSWER_TAG.fullmatch(value)
    if answer is not None:
        value = answer.group(1).strip()
    boxed = _BOXED.fullmatch(value)
    if boxed is not None:
        value = boxed.group(1).strip()
    return value


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", _unwrap_answer(text).casefold()).strip()


def _reference_answers(expected_answer: Any) -> tuple[str, ...]:
    if isinstance(expected_answer, str) and expected_answer.strip():
        return (expected_answer,)
    if isinstance(expected_answer, Sequence) and not isinstance(
        expected_answer, (str, bytes)
    ):
        values = tuple(expected_answer)
        if values and all(isinstance(value, str) and value.strip() for value in values):
            return values
    raise ValueError("expected_answer must be a non-empty string or string list")


def _parse_number(value: str) -> Fraction | None:
    compact = _unwrap_answer(value).strip().replace(",", "")
    percent = compact.endswith("%")
    if percent:
        compact = compact[:-1].strip()
    latex = _LATEX_FRACTION.fullmatch(compact)
    try:
        if latex is not None:
            denominator = int(latex.group(2))
            if denominator == 0:
                return None
            result = Fraction(int(latex.group(1)), denominator)
        elif "/" in compact and compact.count("/") == 1:
            numerator, denominator = compact.split("/", 1)
            result = Fraction(int(numerator.strip()), int(denominator.strip()))
        else:
            result = Fraction(Decimal(compact))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return result / 100 if percent else result


def _arxiv_letter(value: str) -> str | None:
    text = re.sub(r"[*_`]", "", _unwrap_answer(value))
    canonical = _ARXIV_CANONICAL.fullmatch(text)
    if canonical is not None:
        return next(group.upper() for group in canonical.groups() if group is not None)
    markers = tuple(_ARXIV_ANSWER_MARKER.finditer(text))
    if markers:
        return markers[-1].group(1).upper()
    named = tuple(_ARXIV_NAMED_OPTION.finditer(text))
    if named:
        return named[-1].group(1).upper()
    return None


def verify_arxivqa_answer(
    candidate_answer: str | None, expected_answer: Any, *, option_count: int
) -> DeterministicVerification:
    """Verify one ArxivQA answer inside that row's canonical A--Z range."""

    count = _required_int(
        option_count, field_name="option_count", minimum=1, maximum=26
    )
    references = _reference_answers(expected_answer)
    if len(references) != 1:
        raise ValueError("ArxivQA requires one canonical ground-truth label")
    expected = _arxiv_letter(references[0])
    upper_bound = chr(ord("A") + count - 1)
    if expected is None or not "A" <= expected <= upper_bound:
        raise ValueError("ArxivQA ground truth is outside the row option range")
    if candidate_answer is None:
        return DeterministicVerification(
            VerificationOutcome.INCORRECT,
            "arxivqa_missing_final_answer",
            f"expected={expected}; range=A-{upper_bound}",
        )
    candidate = _arxiv_letter(candidate_answer)
    if candidate is None or not "A" <= candidate <= upper_bound:
        return DeterministicVerification(
            VerificationOutcome.INCORRECT,
            "arxivqa_rule",
            f"candidate=unparsed_or_out_of_range; expected={expected}; range=A-{upper_bound}",
        )
    correct = candidate == expected
    return DeterministicVerification(
        VerificationOutcome.CORRECT if correct else VerificationOutcome.INCORRECT,
        "arxivqa_rule",
        f"candidate={candidate}; expected={expected}; range=A-{upper_bound}",
    )


def verify_thinklite_answer(
    candidate_answer: str | None, expected_answer: Any
) -> DeterministicVerification:
    references = _reference_answers(expected_answer)
    if candidate_answer is None:
        return DeterministicVerification(
            VerificationOutcome.INCORRECT,
            "thinklite_missing_final_answer",
            "no non-empty suffix follows the last </think>",
        )
    normalized_candidate = _normalize_answer(candidate_answer)
    normalized_references = tuple(_normalize_answer(value) for value in references)
    if normalized_candidate in normalized_references:
        return DeterministicVerification(
            VerificationOutcome.CORRECT,
            "thinklite_normalized_exact",
            "normalized answer matches a reference",
        )
    candidate_number = _parse_number(candidate_answer)
    reference_numbers = tuple(_parse_number(value) for value in references)
    if candidate_number is not None and all(
        value is not None for value in reference_numbers
    ):
        correct = candidate_number in reference_numbers
        return DeterministicVerification(
            VerificationOutcome.CORRECT if correct else VerificationOutcome.INCORRECT,
            "thinklite_numeric",
            f"numeric_equivalence={correct}",
        )
    return DeterministicVerification(
        VerificationOutcome.SEMANTIC_REQUIRED,
        "thinklite_semantic_required",
        "deterministic exact/numeric rules are inconclusive",
    )


def verify_vstar_answer(
    candidate_answer: str | None, expected_answer: Any
) -> DeterministicVerification:
    references = _reference_answers(expected_answer)
    if candidate_answer is None:
        return DeterministicVerification(
            VerificationOutcome.INCORRECT,
            "vstar_missing_final_answer",
            "no non-empty suffix follows the last </think>",
        )
    candidate = _normalize_answer(candidate_answer)
    if candidate in {_normalize_answer(value) for value in references}:
        return DeterministicVerification(
            VerificationOutcome.CORRECT,
            "vstar_normalized_exact",
            "normalized answer matches a reference",
        )
    return DeterministicVerification(
        VerificationOutcome.SEMANTIC_REQUIRED,
        "vstar_semantic_required",
        "deterministic exact rule is inconclusive",
    )


def verify_t1_answer(
    *,
    source: SelectionSource | str,
    candidate_answer: str | None,
    expected_answer: Any,
    option_count: int | None = None,
) -> DeterministicVerification:
    try:
        normalized_source = SelectionSource(source)
    except ValueError as exc:
        raise ValueError("source is unsupported") from exc
    if normalized_source in {SelectionSource.ARXIVQA, SelectionSource.TEACHER} and (
        option_count is not None
    ):
        verified = verify_arxivqa_answer(
            candidate_answer, expected_answer, option_count=option_count
        )
        if normalized_source is SelectionSource.TEACHER:
            return DeterministicVerification(
                verified.outcome,
                verified.route.replace("arxivqa", "teacher_mcq"),
                verified.evidence,
            )
        return verified
    if normalized_source is SelectionSource.ARXIVQA:
        raise ValueError("ArxivQA verification requires option_count")
    if option_count is not None:
        raise ValueError("option_count is valid only for ArxivQA or teacher MCQ")
    if normalized_source is SelectionSource.THINKLITE:
        return verify_thinklite_answer(candidate_answer, expected_answer)
    if normalized_source is SelectionSource.TEACHER:
        verified = verify_thinklite_answer(candidate_answer, expected_answer)
        return DeterministicVerification(
            verified.outcome,
            verified.route.replace("thinklite", "teacher_open"),
            verified.evidence,
        )
    return verify_vstar_answer(candidate_answer, expected_answer)


_PUBLIC_RUNTIME_MODULE = "tgvf_rl.data.policy_selection_runtime"
for _public_type in (VerificationOutcome, DeterministicVerification):
    rebind_public_class(
        _public_type,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNTIME_MODULE,
    )
for _public_function in (
    extract_final_answer,
    extract_direct_completion,
    parse_t1_answer,
    verify_arxivqa_answer,
    verify_thinklite_answer,
    verify_vstar_answer,
    verify_t1_answer,
):
    rebind_public_function(
        _public_function,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNTIME_MODULE,
    )
del _public_function, _public_type

__all__ = [
    "DeterministicVerification",
    "VerificationOutcome",
    "extract_direct_completion",
    "extract_final_answer",
    "parse_t1_answer",
    "verify_arxivqa_answer",
    "verify_t1_answer",
    "verify_thinklite_answer",
    "verify_vstar_answer",
]
