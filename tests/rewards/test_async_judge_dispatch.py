from __future__ import annotations

import asyncio
from dataclasses import replace
import threading
import time

import pytest

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.judges import (
    BoundedJudgeDispatcher,
    JudgeDispatchConfig,
    JudgeDispatchMode,
    JudgeModelRoute,
    JudgeRequest,
    JudgeResult,
)
from tgvf_rl.rewards.schema import AnswerTaskKind, NormalizationSpec, RewardContext
from tgvf_rl.rewards.verifiers import RuleFirstAnswerVerifier


def _identity(name: str, digit: str) -> ArtifactIdentity:
    return ArtifactIdentity("async-judge-test", name, "v1", digit * 64)


_PROMPT = _identity("prompt", "1")
_MODEL = _identity("qwen2.5-72b", "2")
_SERVICE = _identity("service", "3")
_SAMPLING = _identity("sampling", "4")
_CALIBRATION = _identity("calibration", "5")


def _request(index: int) -> JudgeRequest:
    return JudgeRequest(
        request_id=f"request-{index}",
        task_kind="open_vqa",
        question="What is shown?",
        candidate_answer=f"candidate-{index}",
        reference_answer="reference",
        prompt_identity=_PROMPT,
    )


def _result() -> JudgeResult:
    return JudgeResult(
        score=1.0,
        rationale="equivalent",
        service_identity=_SERVICE,
        model_identity=_MODEL,
        sampling_identity=_SAMPLING,
        calibration_identity=_CALIBRATION,
    )


class _ConcurrentProvider:
    def __init__(self, *, delay: float = 0.02) -> None:
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.maximum_active = 0
        self.thread_names: set[str] = set()
        self.lock = threading.Lock()

    def judge(self, request: JudgeRequest) -> JudgeResult:
        del request
        with self.lock:
            self.calls += 1
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.thread_names.add(threading.current_thread().name)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return _result()


def _dedicated_config(maximum_concurrency: int) -> JudgeDispatchConfig:
    return JudgeDispatchConfig(
        mode=JudgeDispatchMode.DEDICATED_THREAD_POOL,
        maximum_concurrency=maximum_concurrency,
    )


def test_dedicated_dispatcher_bounds_concurrency_and_closes_its_pool() -> None:
    provider = _ConcurrentProvider()
    dispatcher = BoundedJudgeDispatcher(
        provider,
        config=_dedicated_config(2),
        bound_model_name="Qwen2.5-72B-Instruct",
        bound_model_identity=_MODEL,
    )

    async def run() -> tuple[JudgeResult, ...]:
        async with dispatcher:
            return tuple(
                await asyncio.gather(
                    *(dispatcher.judge_async(_request(index)) for index in range(8))
                )
            )

    results = asyncio.run(run())

    assert len(results) == 8
    assert all(result.score == 1.0 for result in results)
    assert provider.calls == 8
    assert provider.maximum_active == 2
    assert provider.thread_names
    assert all(name.startswith("tgvf-reward-judge") for name in provider.thread_names)
    with pytest.raises(RuntimeError, match="closed"):
        dispatcher.judge(_request(9))


def test_dispatcher_preserves_provider_failure_without_fallback() -> None:
    class _PermanentFailureProvider:
        def judge(self, request: JudgeRequest) -> JudgeResult:
            del request
            raise RuntimeError("HTTP 401")

    dispatcher = BoundedJudgeDispatcher(
        _PermanentFailureProvider(),
        config=_dedicated_config(4),
        bound_model_name="qwen/qwen-2.5-72b-instruct",
        bound_model_identity=_MODEL,
    )

    async def run() -> None:
        async with dispatcher:
            await dispatcher.judge_async(_request(0))

    with pytest.raises(RuntimeError, match="HTTP 401"):
        asyncio.run(run())


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        (
            {
                "alternate_model_identity": _identity("small", "6"),
                "alternate_semantics_acknowledged": True,
            },
            "model name",
        ),
        (
            {
                "alternate_model_name": "small-judge",
                "alternate_semantics_acknowledged": True,
            },
            "model identity",
        ),
        (
            {
                "alternate_model_name": "small-judge",
                "alternate_model_identity": _identity("small", "6"),
            },
            "acknowledgement",
        ),
    ),
)
def test_alternate_judge_requires_name_identity_and_acknowledgement(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        JudgeDispatchConfig(
            model_route=JudgeModelRoute.EXPLICIT_ALTERNATE,
            **updates,
        )


def test_alternate_judge_binding_is_exact_and_never_automatic() -> None:
    alternate_identity = _identity("small", "6")
    alternate = JudgeDispatchConfig(
        mode=JudgeDispatchMode.DEDICATED_THREAD_POOL,
        maximum_concurrency=2,
        model_route=JudgeModelRoute.EXPLICIT_ALTERNATE,
        alternate_model_name="small-judge",
        alternate_model_identity=alternate_identity,
        alternate_semantics_acknowledged=True,
    )

    with pytest.raises(ValueError, match="model name differs"):
        BoundedJudgeDispatcher(
            _ConcurrentProvider(delay=0.0),
            config=alternate,
            bound_model_name="another-small-judge",
            bound_model_identity=alternate_identity,
        )
    with pytest.raises(ValueError, match="model identity differs"):
        BoundedJudgeDispatcher(
            _ConcurrentProvider(delay=0.0),
            config=alternate,
            bound_model_name="small-judge",
            bound_model_identity=_identity("wrong-small", "7"),
        )

    dispatcher = BoundedJudgeDispatcher(
        _ConcurrentProvider(delay=0.0),
        config=alternate,
        bound_model_name="small-judge",
        bound_model_identity=alternate_identity,
    )
    assert dispatcher.semantic_fallback_route.startswith(
        "explicit_alternate_semantic_fallback:async-judge-test/small@v1:"
    )
    assert "qwen2.5" not in dispatcher.semantic_fallback_route
    dispatcher.close()

    with pytest.raises(ValueError, match="must remain Qwen2.5-72B"):
        BoundedJudgeDispatcher(
            _ConcurrentProvider(delay=0.0),
            bound_model_name="small-judge",
            bound_model_identity=alternate_identity,
        )


def test_rule_first_async_path_never_dispatches_decidable_samples() -> None:
    provider = _ConcurrentProvider(delay=0.0)
    dispatcher = BoundedJudgeDispatcher(
        provider,
        config=_dedicated_config(2),
        bound_model_name="Qwen2.5-72B-Instruct",
        bound_model_identity=_MODEL,
    )
    verifier = RuleFirstAnswerVerifier(
        rule_identity=_identity("rules", "8"),
        normalization=NormalizationSpec(True, True, True),
        judge=dispatcher,
        judge_prompt_identity=_PROMPT,
        judge_model_identity=_MODEL,
        judge_service_identity=_SERVICE,
        judge_sampling_identity=_SAMPLING,
        judge_calibration_identity=_CALIBRATION,
    )

    async def run() -> None:
        async with dispatcher:
            exact = await verifier.verify_async(
                RewardContext(
                    "exact",
                    "What is shown?",
                    " blue ",
                    "blue",
                    0,
                    task_kind=AnswerTaskKind.OPEN_VQA,
                )
            )
            numeric = await verifier.verify_async(
                RewardContext(
                    "math",
                    "One half?",
                    "0.5",
                    "1/2",
                    0,
                    task_kind=AnswerTaskKind.MATH,
                )
            )
            multiple_choice = await verifier.verify_async(
                RewardContext(
                    "mcq",
                    "Choose one",
                    "B",
                    "B",
                    0,
                    task_kind=AnswerTaskKind.MULTIPLE_CHOICE,
                )
            )
            assert provider.calls == 0
            assert exact.route == "normalized_exact"
            assert numeric.route == "math_numeric_rule"
            assert multiple_choice.route == "multiple_choice_rule"

            fallback = await verifier.verify_async(
                RewardContext(
                    "open",
                    "What is shown?",
                    "a crimson automobile",
                    "a red car",
                    0,
                    task_kind=AnswerTaskKind.OPEN_VQA,
                )
            )
            assert fallback.route == "qwen2.5_72b_semantic_fallback"
            assert fallback.correct
            assert provider.calls == 1

    asyncio.run(run())


def test_default_route_rejects_all_alternate_fields() -> None:
    with pytest.raises(ValueError, match="cannot carry alternate"):
        replace(
            JudgeDispatchConfig(),
            alternate_model_name="small-judge",
        )
