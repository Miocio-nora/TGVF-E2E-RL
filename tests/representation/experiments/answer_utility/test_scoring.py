from __future__ import annotations

import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test lane
    import tomli as tomllib

    sys.modules.setdefault("tomllib", tomllib)

import pytest

from tgvf_rl.representation.experiments.answer_utility.evaluation.scoring import (
    score_instruct_generated_answer,
)
from tgvf_rl.representation.training.oracle_d_utility import (
    OracleDUtilityGroundTruth,
)
from tgvf_rl.representation.training.schema import RepresentationChoice


def _truth(expected: str, *choices: str) -> OracleDUtilityGroundTruth:
    return OracleDUtilityGroundTruth(
        sample_id="sample",
        short_answer=expected,
        choices=tuple(
            RepresentationChoice(label=chr(ord("A") + index), text=text)
            for index, text in enumerate(choices)
        ),
    )


@pytest.mark.parametrize(
    ("candidate", "expected", "route"),
    (
        ("white<|im_end|>", "white", "full_answer_vqa_exact"),
        ("A glove.<|im_end|>", "glove", "full_answer_vqa_exact"),
        ("one", "1", "full_answer_vqa_exact"),
        ("1.0", "1", "full_answer_numeric_equivalence"),
        ("0.5", "50%", "full_answer_numeric_equivalence"),
        (r"\frac{1}{2}", "0.5", "full_answer_numeric_equivalence"),
    ),
)
def test_complete_short_answers_use_vqa_and_numeric_exact_rules(
    candidate: str,
    expected: str,
    route: str,
) -> None:
    score = score_instruct_generated_answer(candidate, _truth(expected))

    assert score.correct is True
    assert score.route == route
    assert score.candidate_choice_label is None


def test_explicit_a_glove_is_text_not_hidden_choice_label() -> None:
    score = score_instruct_generated_answer(
        "The player holds a dark object.\n\nAnswer: A glove<|im_end|>",
        _truth("glove", "bat", "glove", "helmet", "ball"),
    )

    assert score.correct is True
    assert score.route == "explicit_answer_vqa_exact"
    assert score.expected_choice_label == "B"
    assert score.candidate_choice_label is None


def test_bare_hidden_choice_label_is_not_an_answer_route() -> None:
    score = score_instruct_generated_answer(
        "Answer: B<|im_end|>",
        _truth("glove", "bat", "glove", "helmet", "ball"),
    )

    assert score.correct is None
    assert score.route == "semantic_unresolved"
    assert score.candidate_choice_label is None


@pytest.mark.parametrize(
    ("candidate", "route"),
    (
        ("Reasoning. <answer>white</answer>", "answer_tag_vqa_exact"),
        (r"Reasoning. \boxed{white}", "boxed_answer_vqa_exact"),
    ),
)
def test_closed_structured_answer_fields_are_scoreable(
    candidate: str,
    route: str,
) -> None:
    score = score_instruct_generated_answer(
        candidate,
        _truth("white", "white", "gray", "blue", "black"),
    )

    assert score.correct is True
    assert score.route == route
    assert score.candidate_choice_label is None


@pytest.mark.parametrize(
    "candidate",
    (
        "The pants appear white across both legs.",
        "The player isn't holding a glove.",
        "A glove is not what he is holding.",
        "It is either **white** or **black**.",
        "The pants are black, not white.",
        (
            "Based on the image, the player is holding a baseball.\n\n"
            "He is also wearing a baseball glove."
        ),
    ),
)
def test_free_form_gold_mentions_never_become_deterministic_true(
    candidate: str,
) -> None:
    expected = "glove" if "glove" in candidate.casefold() else "white"
    choices = (
        ("bat", "glove", "helmet", "ball")
        if expected == "glove"
        else ("white", "gray", "blue", "black")
    )

    score = score_instruct_generated_answer(candidate, _truth(expected, *choices))

    assert score.correct is None
    assert score.route == "semantic_unresolved"
    assert score.candidate_choice_label is None


def test_complete_exact_distractor_can_be_rejected_without_parsing_a_label() -> None:
    score = score_instruct_generated_answer(
        "Answer: black<|im_end|>",
        _truth("white", "white", "gray", "blue", "black"),
    )

    assert score.correct is False
    assert score.route == "explicit_answer_choice_text_exact"
    assert score.candidate_choice_label is None


@pytest.mark.parametrize(
    "candidate",
    (
        "white",
        "The pants appear white across both legs, with",
        "The pants appear white, but",
        "Answer: white",
    ),
)
def test_unclosed_length_capped_text_stays_unresolved(candidate: str) -> None:
    score = score_instruct_generated_answer(
        candidate,
        _truth("white", "white", "gray", "blue", "black"),
        generation_stop_reason="length_cap",
    )

    assert score.correct is None
    assert score.route == "length_cap_unresolved"


@pytest.mark.parametrize(
    ("candidate", "route"),
    (
        ("Answer: white.", "explicit_answer_vqa_exact_before_length_cap"),
        ("<answer>white</answer>", "answer_tag_vqa_exact_before_length_cap"),
        (r"\boxed{white}", "boxed_answer_vqa_exact_before_length_cap"),
    ),
)
def test_closed_answer_field_before_length_cap_remains_scoreable(
    candidate: str,
    route: str,
) -> None:
    score = score_instruct_generated_answer(
        candidate,
        _truth("white", "white", "gray", "blue", "black"),
        generation_stop_reason="length_cap",
    )

    assert score.correct is True
    assert score.route == route


def test_tool_call_only_is_a_complete_non_answer() -> None:
    score = score_instruct_generated_answer(
        '<tool_call>{"name":"tgvf_focus_tool"}</tool_call><|im_end|>',
        _truth("standing on the mound"),
    )

    assert score.correct is False
    assert score.route == "non_answer_tool_call"


def test_empty_terminal_only_output_is_missing_answer() -> None:
    score = score_instruct_generated_answer(" <|im_end|> ", _truth("white"))

    assert score.correct is False
    assert score.route == "missing_answer"
    assert score.final_answer is None
