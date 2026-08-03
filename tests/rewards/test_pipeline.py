from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.judges import JudgeResult, JudgeUsage
from tgvf_rl.rewards.pipeline import (
    ExactTextVerifier,
    PilotRewardPipeline,
    RewardPipeline,
)
from tgvf_rl.rewards.schema import (
    AnswerTaskKind,
    AnswerVerificationResult,
    NormalizationSpec,
    PilotRewardSpec,
    RewardComponentSpec,
    RewardContext,
    RewardPipelineSpec,
)
from tgvf_rl.rewards.verifiers import RuleFirstAnswerVerifier


def test_explicit_reward_pipeline_is_decomposed_and_deterministic() -> None:
    identity = ArtifactIdentity("smoke", "exact", "v1", "0" * 64)
    spec = RewardPipelineSpec(identity, (RewardComponentSpec("answer", 2.0, identity),))
    pipeline = RewardPipeline(
        spec,
        {"answer": ExactTextVerifier(NormalizationSpec(True, True, True))},
    )
    result = pipeline.score(RewardContext("s", "q", " Blue  label ", "blue label", 1))
    assert result.total == 2.0
    assert result.components[0].raw_score == 1.0


def _identity(name: str, digit: str) -> ArtifactIdentity:
    return ArtifactIdentity("policy-pilot-v1", name, "v1", digit * 64)


@dataclass(frozen=True)
class _FixedVerifier:
    correct: bool
    identity: ArtifactIdentity

    def verify(self, context: RewardContext) -> AnswerVerificationResult:
        return AnswerVerificationResult(
            self.correct, "fixture", "fixture verdict", self.identity
        )


def _pilot_spec() -> PilotRewardSpec:
    return PilotRewardSpec(
        pipeline_identity=_identity("reward", "1"),
        answer_verifier_identity=_identity("answer", "2"),
        format_verifier_identity=_identity("format", "3"),
        tool_verifier_identity=_identity("tool", "4"),
    )


def test_policy_pilot_reward_equation_and_one_time_tool_bonus() -> None:
    spec = _pilot_spec()
    pipeline = PilotRewardPipeline(
        spec, _FixedVerifier(True, spec.answer_verifier_identity)
    )
    context = RewardContext(
        "sample",
        "question",
        "answer",
        "answer",
        tool_call_count=4,
        successful_tgvf_observation_count=3,
        tool_error_codes=("tool_execution_failed",),
    )
    result = pipeline.score(context)

    assert result.total == pytest.approx(2.0)
    assert tuple(component.raw_score for component in result.components) == (
        1.0,
        0.0,
        1.0,
    )
    assert tuple(component.weighted_score for component in result.components) == (
        0.8,
        0.0,
        1.2,
    )


def test_answer_primary_reward_profile_lowers_only_the_tool_bonus() -> None:
    spec = replace(_pilot_spec(), conditional_tool_weight=0.2)
    pipeline = PilotRewardPipeline(
        spec, _FixedVerifier(True, spec.answer_verifier_identity)
    )

    result = pipeline.score(
        RewardContext(
            "sample",
            "question",
            "answer",
            "answer",
            tool_call_count=1,
            successful_tgvf_observation_count=1,
        )
    )

    assert spec.weight_profile_name == "answer-primary"
    assert result.total == pytest.approx(1.0)
    assert tuple(component.weighted_score for component in result.components) == (
        0.8,
        0.0,
        0.2,
    )


@pytest.mark.parametrize(
    ("answer_weight", "format_weight", "tool_weight"),
    (
        (0.8, 0.2, 0.4),
        (0.9, 0.2, 0.2),
        (0.8, 0.1, 0.2),
    ),
)
def test_policy_reward_spec_rejects_unnamed_weight_tuples(
    answer_weight: float,
    format_weight: float,
    tool_weight: float,
) -> None:
    with pytest.raises(ValueError, match="accepted profile"):
        replace(
            _pilot_spec(),
            answer_weight=answer_weight,
            format_weight=format_weight,
            conditional_tool_weight=tool_weight,
        )


def test_policy_pilot_format_penalty_and_no_invented_tool_error_penalty() -> None:
    spec = _pilot_spec()
    pipeline = PilotRewardPipeline(
        spec, _FixedVerifier(False, spec.answer_verifier_identity)
    )
    result = pipeline.score(
        RewardContext(
            "sample",
            "question",
            "bad",
            "good",
            tool_call_count=1,
            protocol_valid=False,
            successful_tgvf_observation_count=0,
            tool_error_codes=("invalid_json",),
        )
    )

    assert result.total == pytest.approx(-0.2)
    assert tuple(component.raw_score for component in result.components) == (
        0.0,
        -1.0,
        0.0,
    )


class _Judge:
    def __init__(self, result: JudgeResult) -> None:
        self.result = result
        self.calls = 0

    def judge(self, request):
        self.calls += 1
        return self.result


def _rule_first_verifier(
    *, judge_score: float = 1.0, judge_usage: JudgeUsage | None = None
):
    model = _identity("qwen2.5-72b", "5")
    service = _identity("judge-service", "6")
    sampling = _identity("judge-sampling", "7")
    calibration = _identity("judge-calibration", "8")
    prompt = _identity("judge-prompt", "9")
    judge = _Judge(
        JudgeResult(
            score=judge_score,
            rationale="binary semantic verdict",
            service_identity=service,
            model_identity=model,
            sampling_identity=sampling,
            calibration_identity=calibration,
            usage=judge_usage,
        )
    )
    verifier = RuleFirstAnswerVerifier(
        rule_identity=_identity("rules", "a"),
        normalization=NormalizationSpec(True, True, True),
        judge=judge,
        judge_prompt_identity=prompt,
        judge_model_identity=model,
        judge_service_identity=service,
        judge_sampling_identity=sampling,
        judge_calibration_identity=calibration,
    )
    return verifier, judge


def test_answer_router_uses_rules_for_mcq_and_numeric_math_before_judge() -> None:
    verifier, judge = _rule_first_verifier()
    mcq = verifier.verify(
        RewardContext(
            "mcq",
            "choose",
            "(B). explanation",
            "B",
            0,
            task_kind=AnswerTaskKind.MULTIPLE_CHOICE,
        )
    )
    math = verifier.verify(
        RewardContext(
            "math",
            "compute",
            "0.5",
            r"\frac{1}{2}",
            0,
            task_kind=AnswerTaskKind.MATH,
        )
    )

    assert mcq.correct and mcq.route == "multiple_choice_rule"
    assert math.correct and math.route == "math_numeric_rule"
    assert judge.calls == 0


@pytest.mark.parametrize(
    ("candidate", "expected", "correct"),
    (
        ("C", "C", True),
        ("(C). explanation", "C", True),
        (r"\boxed{C}", "C", True),
        ("<answer>C</answer>", "C", True),
        ("The final answer is **C**.<|im_end|>", "C", True),
        ("This corresponds to option **C**.<|im_end|>", "C", True),
        ("Option A is sparse. Therefore the majority lies in range C.", "C", True),
        ("The answer is option B.", "C", False),
        ("The majority lies in an interval without an option label.", "T", False),
        ("Clearly, option B is correct.", "C", False),
        ("Based on the image, the answer is A.", "B", False),
    ),
)
def test_mcq_parser_requires_a_canonical_or_explicit_decision(
    candidate: str,
    expected: str,
    correct: bool,
) -> None:
    verifier, judge = _rule_first_verifier()
    result = verifier.verify(
        RewardContext(
            "mcq",
            "choose",
            candidate,
            expected,
            0,
            task_kind=AnswerTaskKind.MULTIPLE_CHOICE,
        )
    )

    assert result.correct is correct
    assert result.route == "multiple_choice_rule"
    assert judge.calls == 0


@pytest.mark.parametrize(
    "candidate",
    (
        "A long visual analysis precedes the decision.\n\nB<|im_end|>",
        "A long visual analysis precedes the decision.\n\nB.\n<|im_end|>",
        "A long visual analysis precedes the decision.\n\nB. option text<|im_end|>",
        "A long visual analysis precedes the decision.\n\n**B**<|im_end|>",
        "A long visual analysis precedes the decision.\n\n`B.` option text<|endoftext|>",
        "The answer is A in the initial hypothesis.\nThat hypothesis is contradicted.\nB.",
    ),
)
def test_mcq_parser_accepts_decision_at_start_of_final_nonempty_line(
    candidate: str,
) -> None:
    verifier, judge = _rule_first_verifier()

    result = verifier.verify(
        RewardContext(
            "mcq",
            "choose",
            candidate,
            "B",
            0,
            task_kind=AnswerTaskKind.MULTIPLE_CHOICE,
        )
    )

    assert result.correct
    assert result.evidence == "candidate=B; expected=B"
    assert judge.calls == 0


@pytest.mark.parametrize(
    "candidate",
    (
        "B. is an intermediate branch.\nThe evidence remains inconclusive.",
        "The diagram labels one branch B.\nNo final choice was made.",
        "The reasoning compares A and B.\nBased on the evidence, it is inconclusive.",
        "The reasoning is incomplete.\nA possible interpretation remains.",
    ),
)
def test_mcq_parser_does_not_take_arbitrary_reasoning_letters_as_decisions(
    candidate: str,
) -> None:
    verifier, judge = _rule_first_verifier()

    result = verifier.verify(
        RewardContext(
            "mcq",
            "choose",
            candidate,
            "B",
            0,
            task_kind=AnswerTaskKind.MULTIPLE_CHOICE,
        )
    )

    assert not result.correct
    assert result.evidence == "letter_parse_failed; normalized_exact=False"
    assert judge.calls == 0


def test_open_vqa_and_undecidable_math_use_bound_72b_judge() -> None:
    usage = JudgeUsage(201, 17, 218, 0.00007916)
    verifier, judge = _rule_first_verifier(
        judge_score=1.0,
        judge_usage=usage,
    )
    result = verifier.verify(
        RewardContext(
            "open",
            "what is shown?",
            "a crimson automobile",
            "a red car",
            0,
            task_kind=AnswerTaskKind.OPEN_VQA,
        )
    )

    assert result.correct
    assert result.route == "qwen2.5_72b_semantic_fallback"
    assert result.judge_usage == usage
    assert judge.calls == 1


def test_formal_judge_must_return_binary_verdict() -> None:
    verifier, _ = _rule_first_verifier(judge_score=0.5)
    with pytest.raises(ValueError, match="binary"):
        verifier.verify(
            RewardContext(
                "open",
                "question",
                "candidate",
                "reference",
                0,
                task_kind=AnswerTaskKind.OPEN_VQA,
            )
        )
