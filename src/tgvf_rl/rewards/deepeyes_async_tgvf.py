"""Official DeepEyes reward scoring for native asynchronous TGVF rollouts.

The veRL AgentLoop owns a complete :class:`TrajectoryRecord`; this module
turns that immutable record into the repo-owned, decomposed Pilot reward.  It
does not own request binding, rollout execution, or DataProto mutation.  The
only asynchronous boundary is the official DeepEyes binary judge transport.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import hashlib
import json
from typing import Protocol

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.trajectories.schema import TrajectoryRecord

from .context import reward_context_from_trajectory
from .deepeyes_official import (
    DEEPEYES_BINARY_JUDGE_MODEL,
    DEEPEYES_OFFICIAL_REWARD_SCHEMA,
    DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
    DEEPEYES_THINKLITE_JUDGE_PROMPT_SHA256,
    DEEPEYES_VISUAL_ANSWER_LIMIT,
    DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
    DEEPEYES_VISUAL_JUDGE_PROMPT_SHA256,
    DeepEyesAnswerExtraction,
    DeepEyesBinaryJudgeRequest,
    extract_thinklite_answer,
    extract_visual_answer,
)
from .deepeyes_verl_reward import AsyncJudgeOutcome, _official_math_verify
from .schema import (
    AnswerTaskKind,
    AnswerVerificationResult,
    PILOT_REWARD_EQUATION_DEEPEYES_VISUAL,
    RewardComponentResult,
    RewardResult,
    deepeyes_reward_equation_for_data_source,
)
from .verl_adapter import PilotVerlTrajectoryReward


DEEPEYES_ASYNC_TGVF_REWARD_SCHEMA = "tgvf.deepeyes-async-tgvf-reward.v1"
_MISSING_FINAL_ANSWER = "[NO VALID FINAL ANSWER]"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _identity(name: str, payload: object) -> ArtifactIdentity:
    return ArtifactIdentity(
        namespace="tgvf_rl.rewards",
        name=name,
        version="v1",
        sha256=_canonical_sha256(payload),
    )


DEEPEYES_ASYNC_ANSWER_VERIFIER_IDENTITY = _identity(
    "deepeyes-async-answer-verifier",
    {
        "schema": DEEPEYES_ASYNC_TGVF_REWARD_SCHEMA,
        "official_reward_schema": DEEPEYES_OFFICIAL_REWARD_SCHEMA,
        "judge_model": DEEPEYES_BINARY_JUDGE_MODEL,
        "visual_prompt_sha256": DEEPEYES_VISUAL_JUDGE_PROMPT_SHA256,
        "thinklite_prompt_sha256": DEEPEYES_THINKLITE_JUDGE_PROMPT_SHA256,
        "thinklite_math_route": "official_math_verify_then_72b_fallback",
    },
)
DEEPEYES_ASYNC_FORMAT_VERIFIER_IDENTITY = _identity(
    "deepeyes-async-format-verifier",
    {
        "schema": DEEPEYES_ASYNC_TGVF_REWARD_SCHEMA,
        "visual_extractor": "extract_visual_answer",
        "thinklite_extractor": "extract_thinklite_answer_last_post_think_boxed",
        "visual_answer_limit": DEEPEYES_VISUAL_ANSWER_LIMIT,
        "visual_answer_limit_comparison": ">=",
    },
)
DEEPEYES_ASYNC_TOOL_VERIFIER_IDENTITY = _identity(
    "deepeyes-async-conditional-tgvf-verifier",
    {
        "schema": DEEPEYES_ASYNC_TGVF_REWARD_SCHEMA,
        "rule": "answer_correct_and_successful_tgvf_observation_count_gte_1",
    },
)
DEEPEYES_ASYNC_PIPELINE_IDENTITY = _identity(
    "deepeyes-async-tgvf-pipeline",
    {
        "schema": DEEPEYES_ASYNC_TGVF_REWARD_SCHEMA,
        "official_reward_schema": DEEPEYES_OFFICIAL_REWARD_SCHEMA,
        "components": (
            "answer_reward",
            "format_reward",
            "conditional_tool_reward",
        ),
        "visual_equation": (0.8, 0.2, 1.2),
        "thinklite_equation": (1.2, 0.4, 0.0),
        "answer_verifier": DEEPEYES_ASYNC_ANSWER_VERIFIER_IDENTITY.sha256,
        "format_verifier": DEEPEYES_ASYNC_FORMAT_VERIFIER_IDENTITY.sha256,
        "tool_verifier": DEEPEYES_ASYNC_TOOL_VERIFIER_IDENTITY.sha256,
        "judge_failure": "scalar_and_all_components_zero",
    },
)


class AsyncDeepEyesJudgePort(Protocol):
    """Structural seam implemented by ``AsyncDeepEyesOpenRouterJudge``."""

    async def judge(self, request: DeepEyesBinaryJudgeRequest) -> AsyncJudgeOutcome: ...


def _normalize_task_kind(value: AnswerTaskKind | str) -> AnswerTaskKind:
    if isinstance(value, AnswerTaskKind):
        return value
    if not isinstance(value, str):
        raise TypeError("DeepEyes task_kind must be AnswerTaskKind or text")
    normalized = value.strip().lower()
    aliases = {
        "math": AnswerTaskKind.MATH,
        "open": AnswerTaskKind.OPEN_VQA,
        "open_vqa": AnswerTaskKind.OPEN_VQA,
        "mcq": AnswerTaskKind.MULTIPLE_CHOICE,
        "multiple_choice": AnswerTaskKind.MULTIPLE_CHOICE,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError("DeepEyes task_kind must be math, open, or mcq") from error


def _official_task_kind(value: AnswerTaskKind) -> str:
    return {
        AnswerTaskKind.MATH: "math",
        AnswerTaskKind.OPEN_VQA: "open",
        AnswerTaskKind.MULTIPLE_CHOICE: "mcq",
    }[value]


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"DeepEyes bound {name} must be non-empty text")
    return value


def _terminal_response(trajectory: TrajectoryRecord) -> str:
    if trajectory.assistant_turns:
        raw_text = trajectory.assistant_turns[-1].raw_text
        if not isinstance(raw_text, str):
            raise TypeError("terminal assistant raw_text must be text")
        return raw_text
    return trajectory.final_answer or ""


def _judge_evidence(
    request: DeepEyesBinaryJudgeRequest,
    outcome: AsyncJudgeOutcome,
) -> str:
    return json.dumps(
        {
            "request_id": request.request_id,
            "prompt_kind": request.prompt_kind,
            "verdict": outcome.verdict,
            "calls": outcome.calls,
            "retries": outcome.retries,
            "cache_hit": outcome.cache_hit,
            "failure_kind": outcome.failure_kind,
            "latency_seconds": outcome.latency_seconds,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class AsyncDeepEyesTGVFTrajectoryRewardScorer:
    """Score one bound native TGVF trajectory with official DeepEyes rules."""

    def __init__(
        self,
        *,
        question: str,
        reference_answer: str,
        task_kind: AnswerTaskKind | str,
        data_source: str,
        judge_transport: AsyncDeepEyesJudgePort,
        math_verify: Callable[[str, str], bool] = _official_math_verify,
    ) -> None:
        self.question = _required_text(question, "question")
        self.reference_answer = _required_text(reference_answer, "reference_answer")
        self.task_kind = _normalize_task_kind(task_kind)
        self.data_source = _required_text(data_source, "data_source")
        if not callable(getattr(judge_transport, "judge", None)):
            raise TypeError("judge_transport must implement async judge()")
        if not callable(math_verify):
            raise TypeError("math_verify must be callable")
        self.judge_transport = judge_transport
        self.math_verify = math_verify
        self.equation_route, self.applied_weights = (
            deepeyes_reward_equation_for_data_source(self.data_source)
        )
        self.component_weights = self.applied_weights

    async def _judge(
        self,
        *,
        trajectory: TrajectoryRecord,
        candidate_answer: str,
        prompt_kind: str,
    ) -> tuple[DeepEyesBinaryJudgeRequest, AsyncJudgeOutcome]:
        request = DeepEyesBinaryJudgeRequest.build(
            trajectory_id=trajectory.identity.canonical_id,
            sample_id=trajectory.identity.sample_id,
            question=self.question,
            reference_answer=self.reference_answer,
            candidate_answer=candidate_answer,
            task_kind=_official_task_kind(self.task_kind),
            prompt_kind=prompt_kind,
        )
        outcome = await self.judge_transport.judge(request)
        if not isinstance(outcome, AsyncJudgeOutcome):
            raise TypeError("DeepEyes async judge must return AsyncJudgeOutcome")
        if type(outcome.verdict) is not bool:
            raise TypeError("DeepEyes async judge verdict must be bool")
        return request, outcome

    async def score_async(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
    ) -> PilotVerlTrajectoryReward:
        if not isinstance(trajectory, TrajectoryRecord):
            raise TypeError("trajectory must be TrajectoryRecord")
        if getattr(request, "identity", None) != trajectory.identity:
            raise IdentityMismatchError(
                "reward request and trajectory identities differ"
            )

        # Resolve again at execution time so no caller can replace a cached
        # route/weight pair after construction.
        equation_route, applied_weights = deepeyes_reward_equation_for_data_source(
            self.data_source
        )
        response = _terminal_response(trajectory)
        base_context = reward_context_from_trajectory(
            trajectory,
            question=self.question,
            expected_answer=self.reference_answer,
            task_kind=self.task_kind,
            data_source=self.data_source,
        )

        if equation_route == PILOT_REWARD_EQUATION_DEEPEYES_VISUAL:
            scored = await self._score_visual(trajectory, response)
        else:
            scored = await self._score_thinklite(trajectory, response)

        extraction = scored.extraction
        context = replace(
            base_context,
            candidate_answer=extraction.answer,
            protocol_valid=(
                base_context.protocol_valid and extraction.valid and not scored.too_long
            ),
            has_valid_final_answer=bool(extraction.answer.strip()),
        )

        accuracy = scored.accuracy
        format_penalty = scored.format_penalty
        conditional_tool = scored.conditional_tool
        if scored.judge_outcome is not None and (
            scored.judge_outcome.failure_kind is not None
        ):
            # The exact bridge consumes components as well as the scalar.  A
            # transport/output failure therefore cannot retain a format term.
            accuracy = 0
            format_penalty = 0
            conditional_tool = 0

        raw_scores = (accuracy, format_penalty, conditional_tool)
        component_identities = (
            DEEPEYES_ASYNC_ANSWER_VERIFIER_IDENTITY,
            DEEPEYES_ASYNC_FORMAT_VERIFIER_IDENTITY,
            DEEPEYES_ASYNC_TOOL_VERIFIER_IDENTITY,
        )
        component_names = (
            "answer_reward",
            "format_reward",
            "conditional_tool_reward",
        )
        evidences = (
            scored.answer_evidence,
            (
                f"extractor={scored.extractor};reason={extraction.reason};"
                f"too_long={int(scored.too_long)}"
            ),
            (
                "rule=correct_and_successful_tgvf_observation;"
                f"successful={context.successful_tgvf_observation_count}"
            ),
        )
        components = tuple(
            RewardComponentResult(
                name=name,
                raw_score=float(raw),
                weighted_score=float(raw * weight),
                verifier_identity=identity,
                evidence=evidence,
            )
            for name, raw, weight, identity, evidence in zip(
                component_names,
                raw_scores,
                applied_weights,
                component_identities,
                evidences,
                strict=True,
            )
        )
        judge_usage = (
            None if scored.judge_outcome is None else scored.judge_outcome.usage
        )
        verification = AnswerVerificationResult(
            correct=bool(accuracy),
            route=scored.answer_route,
            evidence=scored.answer_evidence,
            verifier_identity=DEEPEYES_ASYNC_ANSWER_VERIFIER_IDENTITY,
            judge_usage=judge_usage,
        )
        result = RewardResult(
            total=sum(component.weighted_score for component in components),
            components=components,
            pipeline_identity=DEEPEYES_ASYNC_PIPELINE_IDENTITY,
            answer_verification=verification,
        )
        return PilotVerlTrajectoryReward(
            trajectory_id=trajectory.identity.canonical_id,
            group_uid=trajectory.identity.group_id,
            rollout_index=trajectory.identity.rollout_index,
            context=context,
            result=result,
            equation_route=equation_route,
            applied_weights=applied_weights,
        )

    async def _score_visual(
        self,
        trajectory: TrajectoryRecord,
        response: str,
    ) -> _AsyncScoreFacts:
        extraction = extract_visual_answer(response)
        judge_request, judge_outcome = await self._judge(
            trajectory=trajectory,
            candidate_answer=extraction.answer or _MISSING_FINAL_ANSWER,
            prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
        )
        too_long = len(extraction.answer) >= DEEPEYES_VISUAL_ANSWER_LIMIT
        accuracy = int(judge_outcome.verdict and not too_long)
        format_penalty = -1 if too_long else extraction.format_penalty
        successful = len(trajectory.observations)
        return _AsyncScoreFacts(
            extraction=extraction,
            accuracy=accuracy,
            format_penalty=format_penalty,
            conditional_tool=int(accuracy == 1 and successful >= 1),
            too_long=too_long,
            extractor="visual_direct_answer",
            answer_route="qwen2.5_72b_every_visual_trajectory",
            answer_evidence=_judge_evidence(judge_request, judge_outcome),
            judge_outcome=judge_outcome,
        )

    async def _score_thinklite(
        self,
        trajectory: TrajectoryRecord,
        response: str,
    ) -> _AsyncScoreFacts:
        extraction = extract_thinklite_answer(response)
        if self.task_kind in {
            AnswerTaskKind.OPEN_VQA,
            AnswerTaskKind.MULTIPLE_CHOICE,
        }:
            judge_request, judge_outcome = await self._judge(
                trajectory=trajectory,
                candidate_answer=extraction.answer or _MISSING_FINAL_ANSWER,
                prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
            )
            too_long = len(extraction.answer) >= DEEPEYES_VISUAL_ANSWER_LIMIT
            accuracy = int(judge_outcome.verdict and not too_long)
            format_penalty = -1 if too_long else extraction.format_penalty
            return _AsyncScoreFacts(
                extraction=extraction,
                accuracy=accuracy,
                format_penalty=format_penalty,
                conditional_tool=0,
                too_long=too_long,
                extractor="thinklite_last_boxed",
                answer_route="thinklite_boxed_qwen2.5_72b",
                answer_evidence=_judge_evidence(judge_request, judge_outcome),
                judge_outcome=judge_outcome,
            )
        if self.task_kind is not AnswerTaskKind.MATH:
            raise ValueError("ThinkLite task_kind must be math, open, or mcq")

        rule_correct = False
        if extraction.answer:
            try:
                rule_correct = self.math_verify(
                    self.reference_answer, extraction.answer
                )
            except Exception:
                # Official ThinkLite treats parser failures as judge fallback.
                rule_correct = False
            if type(rule_correct) is not bool:
                raise TypeError("math_verify must return bool")

        judge_outcome: AsyncJudgeOutcome | None = None
        if rule_correct:
            accuracy = 1
            answer_route = "math_verify"
            answer_evidence = "route=math_verify;verdict=true"
        elif extraction.answer:
            judge_request, judge_outcome = await self._judge(
                trajectory=trajectory,
                candidate_answer=extraction.answer,
                prompt_kind=DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
            )
            accuracy = int(judge_outcome.verdict)
            answer_route = "math_verify_then_qwen2.5_72b"
            answer_evidence = _judge_evidence(judge_request, judge_outcome)
        else:
            accuracy = 0
            answer_route = "missing_boxed_answer"
            answer_evidence = "route=missing_boxed_answer;verdict=false"
        return _AsyncScoreFacts(
            extraction=extraction,
            accuracy=accuracy,
            format_penalty=extraction.format_penalty,
            conditional_tool=0,
            too_long=False,
            extractor="thinklite_last_boxed",
            answer_route=answer_route,
            answer_evidence=answer_evidence,
            judge_outcome=judge_outcome,
        )


class _AsyncScoreFacts:
    __slots__ = (
        "accuracy",
        "answer_evidence",
        "answer_route",
        "conditional_tool",
        "extraction",
        "extractor",
        "format_penalty",
        "judge_outcome",
        "too_long",
    )

    def __init__(
        self,
        *,
        extraction: DeepEyesAnswerExtraction,
        accuracy: int,
        format_penalty: int,
        conditional_tool: int,
        too_long: bool,
        extractor: str,
        answer_route: str,
        answer_evidence: str,
        judge_outcome: AsyncJudgeOutcome | None,
    ) -> None:
        self.extraction = extraction
        self.accuracy = accuracy
        self.format_penalty = format_penalty
        self.conditional_tool = conditional_tool
        self.too_long = too_long
        self.extractor = extractor
        self.answer_route = answer_route
        self.answer_evidence = answer_evidence
        self.judge_outcome = judge_outcome


__all__ = [
    "DEEPEYES_ASYNC_ANSWER_VERIFIER_IDENTITY",
    "DEEPEYES_ASYNC_FORMAT_VERIFIER_IDENTITY",
    "DEEPEYES_ASYNC_PIPELINE_IDENTITY",
    "DEEPEYES_ASYNC_TGVF_REWARD_SCHEMA",
    "DEEPEYES_ASYNC_TOOL_VERIFIER_IDENTITY",
    "AsyncDeepEyesJudgePort",
    "AsyncDeepEyesTGVFTrajectoryRewardScorer",
]
