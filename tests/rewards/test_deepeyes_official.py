from __future__ import annotations

import pytest

from tgvf_rl.rewards.deepeyes_official import (
    DEEPEYES_MATH_VERIFY_PROMPT,
    DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
    DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
    DeepEyesBinaryJudgeRequest,
    build_visual_judge_user_prompt,
    extract_thinklite_answer,
    extract_visual_answer,
    parse_binary_judge_output,
    score_thinklite_trajectory,
    score_visual_trajectory,
)


class _Judge:
    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.requests: list[DeepEyesBinaryJudgeRequest] = []

    def judge(self, request: DeepEyesBinaryJudgeRequest) -> bool:
        self.requests.append(request)
        return self.verdict


def _score_visual(
    response: str,
    *,
    verdict: bool = True,
    successful_crop_count: int = 0,
) -> tuple[object, _Judge]:
    judge = _Judge(verdict)
    result = score_visual_trajectory(
        trajectory_id="sample/rollout-0",
        sample_id="sample",
        question="What color?",
        reference_answer="blue",
        response=response,
        task_kind="open",
        successful_crop_count=successful_crop_count,
        judge=judge,
    )
    return result, judge


def test_visual_extraction_uses_clean_direct_final_answer() -> None:
    plain = extract_visual_answer("blue")
    assert (plain.answer, plain.format_penalty) == ("blue", 0)

    malformed = extract_visual_answer(
        "<think>x<answer>red</answer><answer>blue</answer><|vision_start|><|image_pad|>"
    )
    assert malformed.answer == "blue"
    assert malformed.format_penalty == -1
    assert set(malformed.reason.split(",")) == {
        "think_count",
        "vision_count",
        "legacy_answer_wrapper",
    }

    multi_turn = extract_visual_answer(
        "<think>zoom</think><tool_call>crop</tool_call>\nuser\n"
        "<tool_response>crop</tool_response>\nassistant\n"
        "<think>done</think>blue"
    )
    assert multi_turn.answer == "blue"
    assert multi_turn.format_penalty == 0

    legacy = extract_visual_answer("<think>x</think><answer>blue</answer>")
    assert legacy.answer == "blue"
    assert legacy.format_penalty == -1
    assert legacy.reason == "legacy_answer_wrapper"


def test_visual_accuracy_and_format_are_independent_official_truth_table() -> None:
    result, judge = _score_visual("<think>unterminated blue", successful_crop_count=1)
    assert len(judge.requests) == 1
    assert result.accuracy == 1
    assert result.format_penalty == -1
    assert result.conditional_tool == 1
    assert result.total == pytest.approx(1.8)

    no_crop, _ = _score_visual(
        "<think>x</think>blue",
        successful_crop_count=0,
    )
    assert no_crop.accuracy == 1
    assert no_crop.conditional_tool == 0
    assert no_crop.total == pytest.approx(0.8)

    actual_crop, _ = _score_visual("<think>x</think>blue", successful_crop_count=1)
    assert actual_crop.total == pytest.approx(2.0)


def test_visual_overlong_answer_forces_wrong_and_format_penalty() -> None:
    response = "<think>x</think>" + "a" * 1000
    result, judge = _score_visual(response, successful_crop_count=1)
    assert len(judge.requests) == 1  # every visual trajectory is still judged
    assert result.accuracy == 0
    assert result.format_penalty == -1
    assert result.conditional_tool == 0
    assert result.total == pytest.approx(-0.2)


def test_thinklite_uses_last_boxed_and_does_not_gate_accuracy_on_format() -> None:
    extraction = extract_thinklite_answer(
        "<think>x</think> first \\boxed{3}, final \\boxed{4}"
    )
    assert extraction.answer == "4"
    assert extraction.format_penalty == -1

    judge = _Judge(False)
    result = score_thinklite_trajectory(
        trajectory_id="math/rollout-0",
        sample_id="math",
        question="2+2?",
        reference_answer="4",
        response="<think>x</think> first \\boxed{3}, final \\boxed{4}",
        task_kind="math",
        math_verify=lambda expected, candidate: expected == candidate,
        judge=judge,
    )
    assert judge.requests == []
    assert result.accuracy == 1
    assert result.format_penalty == -1
    assert result.total == pytest.approx(0.8)


def test_thinklite_missing_and_semantic_fallback_match_official_routes() -> None:
    missing = score_thinklite_trajectory(
        trajectory_id="math/rollout-0",
        sample_id="math",
        question="2+2?",
        reference_answer="4",
        response="<think>x</think> four",
        task_kind="math",
        math_verify=lambda _expected, _candidate: False,
        judge=_Judge(True),
    )
    assert missing.accuracy == 0
    assert missing.format_penalty == -1
    assert missing.total == pytest.approx(-0.4)

    judge = _Judge(True)
    fallback = score_thinklite_trajectory(
        trajectory_id="math/rollout-1",
        sample_id="math",
        question="2+2?",
        reference_answer="4",
        response="<think>unterminated \\boxed{four}",
        task_kind="math",
        math_verify=lambda _expected, _candidate: False,
        judge=judge,
    )
    assert len(judge.requests) == 1
    assert judge.requests[0].prompt_kind == DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND
    assert fallback.accuracy == 1
    assert fallback.format_penalty == -1
    assert fallback.total == pytest.approx(0.8)


def test_official_judge_prompt_profiles_and_parsers_are_explicit() -> None:
    visual = DeepEyesBinaryJudgeRequest.build(
        trajectory_id="v/0",
        sample_id="v",
        question="Where?",
        reference_answer="left",
        candidate_answer="on the left",
        task_kind="open",
        prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
    )
    assert visual.messages[0] == {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    assert visual.messages[1]["content"] == build_visual_judge_user_prompt(
        question="Where?",
        reference_answer="left",
        candidate_answer="on the left",
    )
    assert parse_binary_judge_output(
        "Judgement: 1", prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND
    )
    assert not parse_binary_judge_output(
        "0", prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND
    )

    math_request = DeepEyesBinaryJudgeRequest.build(
        trajectory_id="m/0",
        sample_id="m",
        question="2+2?",
        reference_answer="4",
        candidate_answer="four",
        task_kind="math",
        prompt_kind=DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
    )
    assert math_request.messages == (
        {
            "role": "user",
            "content": DEEPEYES_MATH_VERIFY_PROMPT.format(
                question="2+2?", reference_answer="4", candidate_answer="four"
            ),
        },
    )
    assert parse_binary_judge_output(
        "## Equivalence Judgement\nTRUE",
        prompt_kind=DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
    )
    with pytest.raises(ValueError, match="unambiguous"):
        parse_binary_judge_output(
            "TRUE or FALSE", prompt_kind=DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND
        )
