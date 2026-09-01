from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.framework.verl import policy_live_runtime
from tgvf_rl.judges import (
    BoundedJudgeDispatcher,
    JUDGE_SAMPLE_FAILURE_ABORT,
    JudgeDispatchMode,
    JudgeResult,
)
from tgvf_rl.rewards.schema import AnswerTaskKind, RewardContext


def _identity(name: str, digit: str) -> ArtifactIdentity:
    return ArtifactIdentity("live-reward-dispatch-test", name, "v1", digit * 64)


def test_canonical_live_reward_pipeline_shares_configured_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _identity("prompt", "1")
    model = _identity("qwen2.5-72b", "2")
    service = _identity("service", "3")
    sampling = _identity("sampling", "4")
    calibration = _identity("calibration", "5")
    active = 0
    maximum_active = 0
    calls = 0
    lock = threading.Lock()

    class Provider:
        config = SimpleNamespace(model_name="Qwen2.5-72B-Instruct")

        def validate_credentials(self) -> None:
            return None

        def judge(self, request: object) -> JudgeResult:
            nonlocal active, maximum_active, calls
            del request
            with lock:
                calls += 1
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return JudgeResult(
                score=1.0,
                rationale="equivalent",
                service_identity=service,
                model_identity=model,
                sampling_identity=sampling,
                calibration_identity=calibration,
            )

    provider = Provider()
    bound = SimpleNamespace(
        provider=provider,
        prompt_identity=prompt,
        model_identity=model,
        service_identity=service,
        sampling_identity=sampling,
        calibration_identity=calibration,
        sample_failure_mode=JUDGE_SAMPLE_FAILURE_ABORT,
    )
    monkeypatch.setattr(
        policy_live_runtime,
        "load_openai_compatible_judge",
        lambda _path, *, expected_file_sha256: (
            bound if expected_file_sha256 == "a" * 64 else None
        ),
    )
    config = SimpleNamespace(
        schema_version="live-dispatch-test",
        identity_sha256="b" * 64,
        performance=SimpleNamespace(
            judge_dispatch_mode="dedicated_thread_pool",
            judge_max_concurrency_per_worker=2,
        ),
        reward=SimpleNamespace(
            profile="pilot-v1",
            answer_verifier="rule-first",
            answer_verifier_sha256="c" * 64,
            answer_weight=0.8,
            format_weight=0.2,
            conditional_tool_weight=1.2,
            judge_config_path=Path("/explicit/qwen72b-judge.json"),
            judge_config_sha256="a" * 64,
        ),
        protocol=SimpleNamespace(enabled_tool_names=("image_zoom_in",)),
    )

    pipeline = policy_live_runtime._build_reward_pipeline(config)
    dispatcher = pipeline.answer_verifier.judge
    assert isinstance(dispatcher, BoundedJudgeDispatcher)
    assert dispatcher.config.mode is JudgeDispatchMode.DEDICATED_THREAD_POOL
    assert dispatcher.config.maximum_concurrency == 2

    async def score_all() -> list[object]:
        return await asyncio.gather(
            *(
                pipeline.score_async(
                    RewardContext(
                        sample_id=f"sample-{index}",
                        question="What is shown?",
                        candidate_answer=f"candidate-{index}",
                        expected_answer="reference",
                        tool_call_count=0,
                        task_kind=AnswerTaskKind.OPEN_VQA,
                    )
                )
                for index in range(6)
            )
        )

    try:
        results = asyncio.run(score_all())
    finally:
        dispatcher.close()

    assert calls == 6
    assert maximum_active == 2
    assert all(result.total == pytest.approx(0.8) for result in results)
    assert all(
        result.answer_verification.route == "qwen2.5_72b_semantic_fallback"
        for result in results
    )


@pytest.mark.parametrize("mode", ("inherit", "inline"))
def test_legacy_or_inline_live_reward_keeps_original_provider(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    model = _identity("qwen2.5-72b", "2")

    class Provider:
        config = SimpleNamespace(model_name="Qwen2.5-72B-Instruct")

        def validate_credentials(self) -> None:
            return None

        def judge(self, _request: object) -> JudgeResult:
            raise AssertionError("binding test does not score")

    provider = Provider()
    bound = SimpleNamespace(
        provider=provider,
        prompt_identity=_identity("prompt", "1"),
        model_identity=model,
        service_identity=_identity("service", "3"),
        sampling_identity=_identity("sampling", "4"),
        calibration_identity=_identity("calibration", "5"),
        sample_failure_mode=JUDGE_SAMPLE_FAILURE_ABORT,
    )
    monkeypatch.setattr(
        policy_live_runtime,
        "load_openai_compatible_judge",
        lambda *_args, **_kwargs: bound,
    )
    config = SimpleNamespace(
        schema_version="legacy-or-inline-test",
        identity_sha256="b" * 64,
        performance=SimpleNamespace(
            judge_dispatch_mode=mode,
            judge_max_concurrency_per_worker=1,
        ),
        reward=SimpleNamespace(
            profile="pilot-v1",
            answer_verifier="rule-first",
            answer_verifier_sha256="c" * 64,
            answer_weight=0.8,
            format_weight=0.2,
            conditional_tool_weight=1.2,
            judge_config_path=Path("/explicit/qwen72b-judge.json"),
            judge_config_sha256="a" * 64,
        ),
        protocol=SimpleNamespace(enabled_tool_names=("image_zoom_in",)),
    )

    pipeline = policy_live_runtime._build_reward_pipeline(config)

    assert pipeline.answer_verifier.judge is provider
    assert not isinstance(pipeline.answer_verifier.judge, BoundedJudgeDispatcher)


def test_live_explicit_alternate_route_binds_loaded_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _identity("prompt", "1")
    alternate_model = _identity("small-judge", "6")
    service = _identity("service", "3")
    sampling = _identity("sampling", "4")
    calibration = _identity("calibration", "5")

    class Provider:
        config = SimpleNamespace(model_name="small-judge")

        def validate_credentials(self) -> None:
            return None

        def judge(self, _request: object) -> JudgeResult:
            return JudgeResult(
                score=1.0,
                rationale="explicit alternate verdict",
                service_identity=service,
                model_identity=alternate_model,
                sampling_identity=sampling,
                calibration_identity=calibration,
            )

    provider = Provider()
    monkeypatch.setattr(
        policy_live_runtime,
        "load_openai_compatible_judge",
        lambda *_args, **_kwargs: SimpleNamespace(
            provider=provider,
            prompt_identity=prompt,
            model_identity=alternate_model,
            service_identity=service,
            sampling_identity=sampling,
            calibration_identity=calibration,
            sample_failure_mode=JUDGE_SAMPLE_FAILURE_ABORT,
        ),
    )
    config = SimpleNamespace(
        schema_version="explicit-alternate-test",
        identity_sha256="b" * 64,
        performance=SimpleNamespace(
            judge_dispatch_mode="dedicated_thread_pool",
            judge_max_concurrency_per_worker=2,
        ),
        reward=SimpleNamespace(
            profile="pilot-v1",
            answer_verifier="rule-first",
            answer_verifier_sha256="c" * 64,
            answer_weight=0.8,
            format_weight=0.2,
            conditional_tool_weight=1.2,
            judge_config_path=Path("/explicit/small-judge.json"),
            judge_config_sha256="a" * 64,
            judge_model_route="explicit_alternate",
            alternate_judge_model_name="small-judge",
            alternate_judge_model_identity=alternate_model,
            alternate_semantics_acknowledged=True,
        ),
        protocol=SimpleNamespace(enabled_tool_names=("image_zoom_in",)),
    )

    pipeline = policy_live_runtime._build_reward_pipeline(config)
    dispatcher = pipeline.answer_verifier.judge
    assert isinstance(dispatcher, BoundedJudgeDispatcher)
    try:
        result = asyncio.run(
            pipeline.score_async(
                RewardContext(
                    sample_id="alternate",
                    question="What is shown?",
                    candidate_answer="candidate",
                    expected_answer="reference",
                    tool_call_count=0,
                    task_kind=AnswerTaskKind.OPEN_VQA,
                )
            )
        )
    finally:
        dispatcher.close()

    route = result.answer_verification.route
    assert route.startswith(
        "explicit_alternate_semantic_fallback:live-reward-dispatch-test/small-judge@v1:"
    )
    assert "qwen2.5" not in route
    assert result.answer_verification.verifier_identity == alternate_model
