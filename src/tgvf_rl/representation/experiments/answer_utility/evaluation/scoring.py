"""Conservative deterministic scoring for Qwen3-VL-Instruct answer text.

Only complete candidate fields are decided locally.  Verbose prose, ambiguous
answers, and incomplete length-capped text remain unresolved for the separately
identified semantic-judge pass; the scorer never promotes a gold substring in
free-form prose to a correct answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Literal
import unicodedata

from tgvf_rl.representation.training.oracle_d_utility import (
    OracleAnswerScore,
    OracleDUtilityGroundTruth,
    _strip_terminal_markers,
)


INSTRUCT_READER_CONTRACT_VERSION = "answer-utility-instruct-short-reader-v2"
INSTRUCT_READER_INSTRUCTION = (
    "Answer the question directly with only a short phrase or number. Do not explain."
)
INSTRUCT_SCORING_CONTRACT_VERSION = "answer-utility-instruct-conservative-vqa-score-v3"

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_EXPLICIT_ANSWER = re.compile(
    r"(?im)^\s*(?:final\s+answer|answer)\s*(?:is\s*|[:=\-]\s*)(.+?)\s*$"
)
_BOXED_START = re.compile(r"\\boxed\s*\{")
_MARKDOWN = re.compile(r"[*_`]")
_SPACE = re.compile(r"\s+")
_PERIOD = re.compile(r"(?<!\d)\.(?!\d)")
_COMMA_BETWEEN_DIGITS = re.compile(r"(\d),(\d)")
_LATEX_FRACTION = re.compile(r"^\\frac\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}$")
_TOOL_CALL_ONLY = re.compile(
    r"^\s*<tool_call>.*?</tool_call>\s*$", re.DOTALL | re.IGNORECASE
)
_VQA_PUNCTUATION = (
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
)
_VQA_ARTICLES = frozenset({"a", "an", "the"})
_VQA_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
# The official VQA normalization also canonicalizes common un-apostrophized
# contractions.  Keeping the mapping local makes the scoring contract runnable
# without importing a mutable external VLMEvalKit checkout.
_VQA_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "itll": "it'll",
    "ive": "i've",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "shouldnt": "shouldn't",
    "shouldve": "should've",
    "thats": "that's",
    "theres": "there's",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wheres": "where's",
    "whos": "who's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "wouldve": "would've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}


@dataclass(frozen=True, slots=True)
class _CandidateWindow:
    kind: str
    value: str
    start: int
    closed: bool


@dataclass(frozen=True, slots=True)
class _WindowEvidence:
    route: str
    candidate: str
    normalized_candidate: str
    correct: bool


def reader_question(question: str) -> str:
    """Append the exact evaluation-only short-answer instruction."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("reader question must be non-empty text")
    return f"{question.rstrip()}\n\n{INSTRUCT_READER_INSTRUCTION}"


def score_instruct_generated_answer(
    generated_text: str,
    ground_truth: OracleDUtilityGroundTruth,
    *,
    generation_stop_reason: Literal["natural_stop", "length_cap"] = "natural_stop",
) -> OracleAnswerScore:
    """Apply exact rules to complete answer fields and defer everything else.

    Choice *labels* are never parsed because the reader does not see the
    choices.  Choice text may still act as a scoring-only reference, but only
    by whole-candidate VQA-normalized exact match.  ``candidate_choice_label``
    therefore remains ``None``: no model-emitted label was observed.
    """

    if not isinstance(generated_text, str):
        raise TypeError("generated_text must be text")
    if not isinstance(ground_truth, OracleDUtilityGroundTruth):
        raise TypeError("scoring requires separated ground truth")
    if generation_stop_reason not in {"natural_stop", "length_cap"}:
        raise ValueError("unknown generation_stop_reason")

    clean = _strip_terminal_markers(generated_text).strip()
    expected = _vqa_normalize(ground_truth.short_answer)
    expected_label = _expected_choice_label(ground_truth, expected=expected)
    if not clean:
        return OracleAnswerScore(
            correct=False,
            route="missing_answer",
            final_answer=None,
            normalized_candidate=None,
            normalized_expected=expected,
            expected_choice_label=expected_label,
            candidate_choice_label=None,
        )
    if _TOOL_CALL_ONLY.fullmatch(clean) is not None:
        return OracleAnswerScore(
            correct=False,
            route="non_answer_tool_call",
            final_answer=clean,
            normalized_candidate=_vqa_normalize(clean),
            normalized_expected=expected,
            expected_choice_label=expected_label,
            candidate_choice_label=None,
        )

    for window in _candidate_windows(
        clean,
        generation_stop_reason=generation_stop_reason,
    ):
        if not window.closed:
            continue
        evidence = _window_evidence(
            window.value,
            ground_truth=ground_truth,
            expected=expected,
            expected_label=expected_label,
        )
        if evidence is None:
            continue
        route = f"{window.kind}_{evidence.route}"
        if generation_stop_reason == "length_cap":
            route += "_before_length_cap"
        return OracleAnswerScore(
            correct=evidence.correct,
            route=route,
            final_answer=evidence.candidate,
            normalized_candidate=evidence.normalized_candidate,
            normalized_expected=expected,
            expected_choice_label=expected_label,
            candidate_choice_label=None,
        )

    return OracleAnswerScore(
        correct=None,
        route=(
            "length_cap_unresolved"
            if generation_stop_reason == "length_cap"
            else "semantic_unresolved"
        ),
        final_answer=clean,
        normalized_candidate=_vqa_normalize(clean),
        normalized_expected=expected,
        expected_choice_label=expected_label,
        candidate_choice_label=None,
    )


def _candidate_windows(
    clean: str,
    *,
    generation_stop_reason: Literal["natural_stop", "length_cap"],
) -> tuple[_CandidateWindow, ...]:
    structured: list[_CandidateWindow] = []
    for match in _ANSWER_TAG.finditer(clean):
        candidate = match.group(1).strip()
        if candidate:
            structured.append(
                _CandidateWindow("answer_tag", candidate, match.start(), True)
            )
    structured.extend(_boxed_windows(clean))
    for match in _EXPLICIT_ANSWER.finditer(clean):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        closed = generation_stop_reason == "natural_stop" or (
            match.end() < len(clean) or _has_terminal_punctuation(candidate)
        )
        structured.append(
            _CandidateWindow("explicit_answer", candidate, match.start(), closed)
        )

    # A later explicit/structured answer supersedes an earlier one.  The full
    # response is eligible only after a natural stop; a length cap does not
    # establish that its trailing text is a complete candidate.
    ordered = sorted(structured, key=lambda item: item.start, reverse=True)
    if generation_stop_reason == "natural_stop":
        ordered.append(_CandidateWindow("full_answer", clean, -1, True))

    seen: set[str] = set()
    result: list[_CandidateWindow] = []
    for window in ordered:
        identity = _vqa_normalize(window.value)
        if identity and identity not in seen:
            seen.add(identity)
            result.append(window)
    return tuple(result)


def _boxed_windows(clean: str) -> tuple[_CandidateWindow, ...]:
    windows: list[_CandidateWindow] = []
    for match in _BOXED_START.finditer(clean):
        depth = 1
        cursor = match.end()
        while cursor < len(clean) and depth:
            if clean[cursor] == "{":
                depth += 1
            elif clean[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            candidate = clean[match.end() : cursor - 1].strip()
            if candidate:
                windows.append(
                    _CandidateWindow("boxed_answer", candidate, match.start(), True)
                )
    return tuple(windows)


def _has_terminal_punctuation(candidate: str) -> bool:
    value = candidate.rstrip()
    return bool(value) and value[-1] in ".!?。！？"


def _window_evidence(
    candidate: str,
    *,
    ground_truth: OracleDUtilityGroundTruth,
    expected: str,
    expected_label: str | None,
) -> _WindowEvidence | None:
    normalized = _vqa_normalize(candidate)
    if normalized == expected:
        return _WindowEvidence(
            route="vqa_exact",
            candidate=candidate,
            normalized_candidate=normalized,
            correct=True,
        )

    candidate_number = _numeric_value(candidate)
    expected_number = _numeric_value(ground_truth.short_answer)
    if candidate_number is not None and expected_number is not None:
        return _WindowEvidence(
            route="numeric_equivalence",
            candidate=candidate,
            normalized_candidate=normalized,
            correct=candidate_number == expected_number,
        )

    labels = tuple(
        choice.label.upper()
        for choice in ground_truth.choices
        if _vqa_normalize(choice.text) == normalized
    )
    if len(labels) == 1 and expected_label is not None:
        return _WindowEvidence(
            route="choice_text_exact",
            candidate=candidate,
            normalized_candidate=normalized,
            correct=labels[0] == expected_label,
        )
    return None


def _expected_choice_label(
    ground_truth: OracleDUtilityGroundTruth,
    *,
    expected: str,
) -> str | None:
    labels = tuple(
        choice.label.upper()
        for choice in ground_truth.choices
        if _vqa_normalize(choice.text) == expected
    )
    return labels[0] if len(labels) == 1 else None


def _vqa_normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", _strip_terminal_markers(value))
    text = _MARKDOWN.sub("", text.replace("\n", " ").replace("\t", " "))
    text = _process_vqa_punctuation(text)
    words: list[str] = []
    for raw_word in text.casefold().split():
        word = _VQA_NUMBER_WORDS.get(raw_word, raw_word)
        if word in _VQA_ARTICLES:
            continue
        words.append(_VQA_CONTRACTIONS.get(word, word))
    return _SPACE.sub(" ", " ".join(words)).strip()


def _process_vqa_punctuation(value: str) -> str:
    text = value
    comma_between_digits = _COMMA_BETWEEN_DIGITS.search(value) is not None
    for punctuation in _VQA_PUNCTUATION:
        if (
            f"{punctuation} " in value
            or f" {punctuation}" in value
            or comma_between_digits
        ):
            text = text.replace(punctuation, "")
        else:
            text = text.replace(punctuation, " ")
    return _PERIOD.sub("", text)


def _numeric_value(value: str) -> Fraction | None:
    compact = unicodedata.normalize("NFKC", _strip_terminal_markers(value)).strip()
    compact = _MARKDOWN.sub("", compact)
    answer_tag = _ANSWER_TAG.fullmatch(compact)
    if answer_tag is not None:
        compact = answer_tag.group(1).strip()
    boxed = _boxed_windows(compact)
    if len(boxed) == 1 and boxed[0].start == 0:
        compact = boxed[0].value
    compact = compact.rstrip(".。").strip().replace(",", "")
    compact = _VQA_NUMBER_WORDS.get(compact.casefold(), compact)
    latex = _LATEX_FRACTION.fullmatch(compact)
    if latex is not None:
        denominator = int(latex.group(2))
        return None if denominator == 0 else Fraction(int(latex.group(1)), denominator)
    percent = compact.endswith("%")
    if percent:
        compact = compact[:-1].strip()
    try:
        result = Fraction(Decimal(compact))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        try:
            result = Fraction(compact)
        except (ValueError, ZeroDivisionError):
            return None
    return result / 100 if percent else result


__all__ = [
    "INSTRUCT_READER_CONTRACT_VERSION",
    "INSTRUCT_READER_INSTRUCTION",
    "INSTRUCT_SCORING_CONTRACT_VERSION",
    "reader_question",
    "score_instruct_generated_answer",
]
