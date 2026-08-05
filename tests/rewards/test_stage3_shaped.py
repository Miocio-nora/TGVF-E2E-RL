from __future__ import annotations

from dataclasses import replace

import pytest

from tgvf_rl.rewards.stage3_shaped import (
    QualityJudgeScore,
    STAGE3_SHAPED_REWARD_VERSION,
    Stage3ShapedComponentName,
    Stage3ShapedRewardFacts,
    Stage3ShapedRewardKernel,
    ToolNecessityLabel,
)


def _component_scores(result) -> dict[Stage3ShapedComponentName, float]:
    return {component.name: component.score for component in result.components}


def test_stage3_shaped_exact_equation_and_audit_evidence() -> None:
    result = Stage3ShapedRewardKernel().score(
        Stage3ShapedRewardFacts(
            answer_correct=True,
            tool_label=ToolNecessityLabel.NEEDED,
            tool_call_count=1,
            successful_tgvf_observation_count=1,
            focus_score=QualityJudgeScore.PASS,
            grounding_score=QualityJudgeScore.PASS,
        )
    )

    assert result.version == STAGE3_SHAPED_REWARD_VERSION
    assert result.answer_gated is False
    assert _component_scores(result) == {
        Stage3ShapedComponentName.ANSWER: 2.0,
        Stage3ShapedComponentName.TOOL: 0.5,
        Stage3ShapedComponentName.FOCUS: 1.0,
        Stage3ShapedComponentName.GROUNDING: 1.0,
        Stage3ShapedComponentName.PROTOCOL: 0.0,
    }
    assert result.total == pytest.approx(4.5)
    assert "label=needed" in result.component(Stage3ShapedComponentName.TOOL).evidence
    assert (
        "judge_score=2"
        in result.component(Stage3ShapedComponentName.GROUNDING).evidence
    )


@pytest.mark.parametrize(
    ("label", "used", "expected"),
    (
        (ToolNecessityLabel.NEEDED, True, 0.5),
        (ToolNecessityLabel.NEEDED, False, -1.0),
        (ToolNecessityLabel.OPTIONAL, True, 0.25),
        (ToolNecessityLabel.OPTIONAL, False, 0.0),
        (ToolNecessityLabel.UNNECESSARY, True, -0.25),
        (ToolNecessityLabel.UNNECESSARY, False, 0.5),
    ),
)
def test_tool_decision_table_uses_default_half_confidence(
    label: ToolNecessityLabel,
    used: bool,
    expected: float,
) -> None:
    facts = Stage3ShapedRewardFacts(
        answer_correct=False,
        tool_label=label,
        tool_call_count=int(used),
        successful_tgvf_observation_count=int(used),
        focus_score=QualityJudgeScore.FAIL if used else None,
        grounding_score=QualityJudgeScore.FAIL if used else None,
    )

    result = Stage3ShapedRewardKernel().score(facts)

    assert result.component(Stage3ShapedComponentName.TOOL).score == pytest.approx(
        expected
    )


def test_needed_answer_gate_requires_a_successful_observation() -> None:
    failed_call = Stage3ShapedRewardFacts(
        answer_correct=True,
        tool_label=ToolNecessityLabel.NEEDED,
        tool_call_count=1,
        successful_tgvf_observation_count=0,
    )

    result = Stage3ShapedRewardKernel().score(failed_call)

    assert result.answer_gated is True
    assert result.component(Stage3ShapedComponentName.ANSWER).score == 0.0
    assert result.component(Stage3ShapedComponentName.TOOL).score == -1.0
    assert (
        "tool_succeeded=False"
        in result.component(Stage3ShapedComponentName.ANSWER).evidence
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    (
        (ToolNecessityLabel.NEEDED, -1.0),
        (ToolNecessityLabel.OPTIONAL, 0.0),
        (ToolNecessityLabel.UNNECESSARY, 0.5),
    ),
)
def test_failed_call_uses_no_use_branch_for_every_tool_label(
    label: ToolNecessityLabel,
    expected: float,
) -> None:
    result = Stage3ShapedRewardKernel().score(
        Stage3ShapedRewardFacts(
            answer_correct=False,
            tool_label=label,
            tool_call_count=1,
            successful_tgvf_observation_count=0,
        )
    )

    tool = result.component(Stage3ShapedComponentName.TOOL)
    assert tool.score == expected
    assert "tool_attempted=True" in tool.evidence
    assert "tool_used=False" in tool.evidence


def test_non_needed_answer_is_not_gated_when_no_tool_is_used() -> None:
    result = Stage3ShapedRewardKernel().score(
        Stage3ShapedRewardFacts(
            answer_correct=True,
            tool_label=ToolNecessityLabel.UNNECESSARY,
        )
    )

    assert result.answer_gated is False
    assert result.component(Stage3ShapedComponentName.ANSWER).score == 2.0
    assert result.component(Stage3ShapedComponentName.FOCUS).score == 0.0
    assert (
        "not_applicable" in result.component(Stage3ShapedComponentName.FOCUS).evidence
    )


def test_each_extra_call_costs_point_zero_five_without_confidence_scaling() -> None:
    result = Stage3ShapedRewardKernel().score(
        Stage3ShapedRewardFacts(
            answer_correct=False,
            tool_label=ToolNecessityLabel.OPTIONAL,
            tool_call_count=3,
            successful_tgvf_observation_count=1,
            focus_score=QualityJudgeScore.FAIL,
            grounding_score=QualityJudgeScore.FAIL,
        )
    )

    tool = result.component(Stage3ShapedComponentName.TOOL)
    assert tool.score == pytest.approx(0.5 * 0.5 - 2 * 0.05)
    assert "extra_call_count=2" in tool.evidence
    assert "extra_call_penalty=-0.1" in tool.evidence


@pytest.mark.parametrize(
    ("raw_score", "focus", "grounding"),
    (
        (QualityJudgeScore.PASS, 1.0, 1.0),
        (QualityJudgeScore.PARTIAL, 0.5, 0.5),
        (QualityJudgeScore.FAIL, 0.0, -1.0),
    ),
)
def test_focus_and_grounding_mappings_are_distinct(
    raw_score: QualityJudgeScore,
    focus: float,
    grounding: float,
) -> None:
    result = Stage3ShapedRewardKernel().score(
        Stage3ShapedRewardFacts(
            answer_correct=False,
            tool_label=ToolNecessityLabel.OPTIONAL,
            tool_call_count=1,
            successful_tgvf_observation_count=1,
            focus_score=raw_score,
            grounding_score=raw_score,
        )
    )

    assert result.component(Stage3ShapedComponentName.FOCUS).score == focus
    assert result.component(Stage3ShapedComponentName.GROUNDING).score == grounding


def test_sample_local_quality_judge_failure_is_zero_not_raw_fail() -> None:
    result = Stage3ShapedRewardKernel().score(
        Stage3ShapedRewardFacts(
            answer_correct=False,
            tool_label=ToolNecessityLabel.OPTIONAL,
            tool_call_count=1,
            successful_tgvf_observation_count=1,
            quality_judge_failure="transport",
        )
    )

    assert result.quality_judge_applicable is True
    assert result.quality_judge_covered is False
    assert result.quality_judge_failure == "transport"
    assert result.component(Stage3ShapedComponentName.FOCUS).score == 0.0
    assert result.component(Stage3ShapedComponentName.GROUNDING).score == 0.0
    assert (
        "coverage=0" in result.component(Stage3ShapedComponentName.GROUNDING).evidence
    )


def test_any_number_of_protocol_errors_produces_one_protocol_penalty() -> None:
    result = Stage3ShapedRewardKernel().score(
        Stage3ShapedRewardFacts(
            answer_correct=False,
            tool_label=ToolNecessityLabel.OPTIONAL,
            protocol_errors=("invalid_json", "missing_final_answer"),
        )
    )

    protocol = result.component(Stage3ShapedComponentName.PROTOCOL)
    assert protocol.score == -1.0
    assert "invalid_json,missing_final_answer" in protocol.evidence
    assert result.total == -1.0


@pytest.mark.parametrize(
    "changes",
    (
        {"answer_correct": 1},
        {"tool_label": "needed"},
        {"tool_call_count": True},
        {"successful_tgvf_observation_count": 0.0},
        {"label_confidence": 1},
        {"focus_score": 2},
        {"grounding_score": 2},
        {"protocol_errors": ["invalid_json"]},
        {"protocol_errors": (1,)},
    ),
)
def test_input_rejects_implicit_type_coercion(changes: dict[str, object]) -> None:
    base = Stage3ShapedRewardFacts(
        answer_correct=True,
        tool_label=ToolNecessityLabel.OPTIONAL,
    )

    with pytest.raises(TypeError):
        replace(base, **changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"tool_call_count": -1},
        {"successful_tgvf_observation_count": -1},
        {"tool_call_count": 0, "successful_tgvf_observation_count": 1},
        {"label_confidence": float("nan")},
        {"label_confidence": -0.1},
        {"label_confidence": 1.1},
        {"protocol_errors": ("",)},
        {"protocol_errors": (" invalid_json",)},
        {"protocol_errors": ("invalid_json", "invalid_json")},
    ),
)
def test_input_rejects_invalid_values(changes: dict[str, object]) -> None:
    base = Stage3ShapedRewardFacts(
        answer_correct=True,
        tool_label=ToolNecessityLabel.OPTIONAL,
    )

    with pytest.raises(ValueError):
        replace(base, **changes)


def test_successful_observation_requires_both_quality_scores() -> None:
    with pytest.raises(ValueError, match="require focus and grounding"):
        Stage3ShapedRewardFacts(
            answer_correct=True,
            tool_label=ToolNecessityLabel.NEEDED,
            tool_call_count=1,
            successful_tgvf_observation_count=1,
            focus_score=QualityJudgeScore.PASS,
        )


def test_quality_scores_are_rejected_without_an_observation() -> None:
    with pytest.raises(ValueError, match="require a successful"):
        Stage3ShapedRewardFacts(
            answer_correct=True,
            tool_label=ToolNecessityLabel.OPTIONAL,
            focus_score=QualityJudgeScore.PASS,
            grounding_score=QualityJudgeScore.PASS,
        )


def test_kernel_rejects_legacy_reward_context_instead_of_guessing() -> None:
    with pytest.raises(TypeError, match="Stage3ShapedRewardFacts"):
        Stage3ShapedRewardKernel().score(object())  # type: ignore[arg-type]
