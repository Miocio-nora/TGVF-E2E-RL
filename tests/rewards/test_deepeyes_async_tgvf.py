from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import OwnedTokenSequence, TokenOwnership, TokenSpan
from tgvf_rl.judges.base import JudgeUsage
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.rewards.deepeyes_async_tgvf import (
    DEEPEYES_ASYNC_PIPELINE_IDENTITY,
    AsyncDeepEyesTGVFTrajectoryRewardScorer,
)
from tgvf_rl.rewards.deepeyes_official import (
    DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
    DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
)
from tgvf_rl.rewards.deepeyes_verl_reward import AsyncJudgeOutcome
from tgvf_rl.rewards.schema import AnswerTaskKind
from tgvf_rl.rewards.verl_adapter import PILOT_VERL_JUDGE_USAGE_FIELD
from tgvf_rl.trajectories.behavior import BehaviorTraceHandle
from tgvf_rl.trajectories.schema import (
    AssistantTurnRecord,
    ToolCallRecord,
    ToolObservationRecord,
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)


SHA0 = "0" * 64
SHA1 = "1" * 64
JUDGE_USAGE = JudgeUsage(211, 7, 218, 0.00007916)


class FakeAsyncJudge:
    def __init__(self, *outcomes: AsyncJudgeOutcome) -> None:
        self.outcomes = list(outcomes)
        self.requests = []

    async def judge(self, request):
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("unexpected judge call")
        return self.outcomes.pop(0)


def _outcome(
    verdict: bool,
    *,
    failure_kind: str | None = None,
    usage: JudgeUsage | None = JUDGE_USAGE,
) -> AsyncJudgeOutcome:
    return AsyncJudgeOutcome(
        verdict=verdict,
        calls=1,
        retries=0,
        cache_hit=0,
        failure_kind=failure_kind,
        latency_seconds=0.25,
        usage=usage,
    )


def _turn(index: int, raw_text: str, *, tool: bool = False) -> AssistantTurnRecord:
    tokens = OwnedTokenSequence((index + 1,), (TokenOwnership.POLICY_SAMPLED,))
    digest = SHA0 if index == 0 else SHA1
    return AssistantTurnRecord(
        turn_index=index,
        raw_text=raw_text,
        tokens=tokens,
        behavior_trace=BehaviorTraceHandle(f"behavior-sha256:{digest}", digest),
        think_span=None,
        is_tool_call=tool,
    )


def _trajectory(
    response: str,
    *,
    successful_tgvf: bool = False,
    sample_id: str = "sample",
) -> TrajectoryRecord:
    identity = TrajectoryIdentity("async-reward-test", sample_id, 0, "group")
    turns = [_turn(0, response)]
    calls = ()
    observations = ()
    if successful_tgvf:
        turns = [
            _turn(0, "<think>inspect</think><tool_call>...</tool_call>", tool=True),
            _turn(1, response),
        ]
        calls = (
            ToolCallRecord(
                call_index=0,
                assistant_turn_index=0,
                function_name="tgvf_focus_tool",
                target="answer-bearing visual detail",
                target_token_span=TokenSpan(0, 1),
                target_char_span=(0, 28),
                raw_call_text="fixture",
            ),
        )
        observations = (
            ToolObservationRecord(
                call_index=0,
                handle=ObservationHandle("observation-0", SHA0),
                template_token_ids=(151665,),
            ),
        )
    return TrajectoryRecord(
        schema_version="trajectory-v1",
        identity=identity,
        model=ModelIdentity("qwen3_vl", "fixture", "/fixture", 1, SHA0),
        behavior_policy=PolicyVersion(identity.run_id, 0, SHA0),
        assistant_turns=tuple(turns),
        tool_calls=calls,
        observations=observations,
        final_answer=response,
        stop=TrajectoryStop.FINAL_ANSWER,
    )


def _score(
    scorer: AsyncDeepEyesTGVFTrajectoryRewardScorer,
    trajectory: TrajectoryRecord,
):
    return asyncio.run(
        scorer.score_async(
            request=SimpleNamespace(identity=trajectory.identity),
            trajectory=trajectory,
        )
    )


def test_visual_always_judges_and_applies_the_1000_character_guard() -> None:
    judge = FakeAsyncJudge(_outcome(False), _outcome(True))
    scorer = AsyncDeepEyesTGVFTrajectoryRewardScorer(
        question="What color?",
        reference_answer="blue",
        task_kind="open",
        data_source="vstar",
        judge_transport=judge,
    )

    missing = _score(scorer, _trajectory("<think>unsure</think>", sample_id="a"))
    overlong = _score(
        scorer,
        _trajectory(
            "<think>done</think>" + "x" * 1000,
            sample_id="b",
        ),
    )

    assert len(judge.requests) == 2
    assert all(
        request.prompt_kind == DEEPEYES_VISUAL_JUDGE_PROMPT_KIND
        for request in judge.requests
    )
    assert judge.requests[0].candidate_answer == "[NO VALID FINAL ANSWER]"
    assert missing.raw_components == (
        ("answer_reward", 0.0),
        ("format_reward", -1.0),
        ("conditional_tool_reward", 0.0),
    )
    assert overlong.total == -0.2
    assert overlong.raw_components == missing.raw_components


def test_visual_conditional_reward_requires_a_successful_tgvf_observation() -> None:
    judge = FakeAsyncJudge(_outcome(True), _outcome(True))
    scorer = AsyncDeepEyesTGVFTrajectoryRewardScorer(
        question="What color?",
        reference_answer="blue",
        task_kind=AnswerTaskKind.OPEN_VQA,
        data_source="arxivqa",
        judge_transport=judge,
    )
    direct = _score(scorer, _trajectory("<think>see</think>blue", sample_id="a"))
    focused = _score(
        scorer,
        _trajectory(
            "<think>see</think>blue",
            successful_tgvf=True,
            sample_id="b",
        ),
    )

    assert direct.total == 0.8
    assert focused.total == 2.0
    assert direct.raw_components[-1] == ("conditional_tool_reward", 0.0)
    assert focused.raw_components[-1] == ("conditional_tool_reward", 1.0)
    assert focused.result.answer_verification.judge_usage == JUDGE_USAGE
    assert focused.reward_sidecars()[PILOT_VERL_JUDGE_USAGE_FIELD] == (
        JUDGE_USAGE.prompt_tokens,
        JUDGE_USAGE.completion_tokens,
        JUDGE_USAGE.total_tokens,
        JUDGE_USAGE.cost_usd,
    )
    assert focused.reward_extra_info()["judge_prompt_tokens"] == 211
    assert focused.pipeline_sha256 == DEEPEYES_ASYNC_PIPELINE_IDENTITY.sha256


def test_thinklite_math_uses_math_verify_before_official_72b_fallback() -> None:
    rule_judge = FakeAsyncJudge()
    rule = AsyncDeepEyesTGVFTrajectoryRewardScorer(
        question="Compute it.",
        reference_answer="42",
        task_kind="math",
        data_source="thinklite",
        judge_transport=rule_judge,
        math_verify=lambda _reference, _candidate: True,
    )
    rule_result = _score(rule, _trajectory("<think>x</think>\\boxed{42}"))
    assert rule_result.total == 1.2
    assert rule_result.result.answer_verification.route == "math_verify"
    assert rule_result.result.answer_verification.judge_usage is None
    assert rule_judge.requests == []

    fallback_judge = FakeAsyncJudge(_outcome(True))

    def parsing_failure(_reference: str, _candidate: str) -> bool:
        raise ValueError("fixture parse failure")

    fallback = AsyncDeepEyesTGVFTrajectoryRewardScorer(
        question="Compute it.",
        reference_answer="42",
        task_kind=AnswerTaskKind.MATH,
        data_source="thinklite_eureka",
        judge_transport=fallback_judge,
        math_verify=parsing_failure,
    )
    fallback_result = _score(
        fallback,
        _trajectory("<think>x</think>\\boxed{42}", sample_id="fallback"),
    )
    assert fallback_result.total == 1.2
    assert len(fallback_judge.requests) == 1
    assert (
        fallback_judge.requests[0].prompt_kind == DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND
    )
    assert fallback_result.result.answer_verification.judge_usage == JUDGE_USAGE


def test_thinklite_open_and_mcq_use_the_official_visual_judge_prompt() -> None:
    for index, task_kind in enumerate(("open", "mcq")):
        judge = FakeAsyncJudge(_outcome(True))
        scorer = AsyncDeepEyesTGVFTrajectoryRewardScorer(
            question="Choose.",
            reference_answer="B",
            task_kind=task_kind,
            data_source="thinklite",
            judge_transport=judge,
        )
        result = _score(
            scorer,
            _trajectory("<think>x</think>\\boxed{B}", sample_id=str(index)),
        )
        assert result.total == 1.2
        assert judge.requests[0].prompt_kind == DEEPEYES_VISUAL_JUDGE_PROMPT_KIND
        assert judge.requests[0].candidate_answer == "B"


def test_judge_failure_zeros_components_but_retains_real_completed_usage() -> None:
    judge = FakeAsyncJudge(_outcome(False, failure_kind="completed_invalid_output"))
    scorer = AsyncDeepEyesTGVFTrajectoryRewardScorer(
        question="Compute it.",
        reference_answer="42",
        task_kind="math",
        data_source="xince",
        judge_transport=judge,
        math_verify=lambda _reference, _candidate: False,
    )
    result = _score(scorer, _trajectory("<think>x</think>\\boxed{41}"))

    assert result.total == 0.0
    assert result.raw_components == (
        ("answer_reward", 0.0),
        ("format_reward", 0.0),
        ("conditional_tool_reward", 0.0),
    )
    assert all(
        component.weighted_score == 0.0 for component in result.result.components
    )
    usage = result.result.answer_verification.judge_usage
    assert usage == JUDGE_USAGE


def test_cache_hit_does_not_fabricate_a_judge_call_or_usage() -> None:
    judge = FakeAsyncJudge(
        AsyncJudgeOutcome(
            verdict=True,
            calls=0,
            retries=0,
            cache_hit=1,
            failure_kind=None,
            latency_seconds=0.01,
            usage=None,
        )
    )
    scorer = AsyncDeepEyesTGVFTrajectoryRewardScorer(
        question="What color?",
        reference_answer="blue",
        task_kind="open",
        data_source="vstar",
        judge_transport=judge,
    )

    result = _score(scorer, _trajectory("<think>see</think>blue"))

    assert result.result.answer_verification.judge_usage is None
    assert result.reward_sidecars()[PILOT_VERL_JUDGE_USAGE_FIELD] is None
    assert result.reward_extra_info()["judge_calls"] == 0
