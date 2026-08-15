from __future__ import annotations

from collections import Counter
import threading
import time

import pytest

from tgvf_rl.rewards.deepeyes_batch import (
    BatchJudgeConfig,
    BatchJudgeFailure,
    BoundedBatchSemanticJudge,
    DeepEyesTrajectoryRewardInput,
    JudgeGlobalFailure,
    JudgeSampleOutputError,
    score_deepeyes_trajectory_batch,
)


def _trajectory(
    trajectory_id: str,
    source: str,
    response: str,
    *,
    successful_crop_count: int = 0,
) -> DeepEyesTrajectoryRewardInput:
    return DeepEyesTrajectoryRewardInput(
        trajectory_id=trajectory_id,
        sample_id=trajectory_id.split("/")[0],
        data_source=source,
        task_kind="math" if source == "thinklite" else "open",
        question="question",
        reference_answer="blue" if source != "thinklite" else "4",
        response=response,
        successful_crop_count=successful_crop_count,
    )


def test_batch_routes_every_visual_and_only_thinklite_fallback() -> None:
    calls: list[str] = []

    def transport(request: object) -> bool:
        calls.append(request.trajectory_id)
        return True

    trajectories = (
        _trajectory("v/0", "vstar", "blue"),
        _trajectory("a/0", "arxivqa", "blue"),
        _trajectory("m/0", "thinklite", "<think>x</think>\\boxed{4}"),
        _trajectory("m/1", "thinklite", "<think>x</think>\\boxed{four}"),
    )
    result = score_deepeyes_trajectory_batch(
        trajectories,
        math_verify=lambda expected, candidate: expected == candidate,
        judge=BoundedBatchSemanticJudge(transport),
    )
    assert Counter(calls) == Counter({"v/0": 1, "a/0": 1, "m/1": 1})
    assert result.judge_metrics.requested == 3
    assert result.judge_metrics.visual_requested == 2
    assert result.judge_metrics.thinklite_fallback_requested == 1
    assert result.judge_metrics.judge_calls == 3
    assert result.judge_metrics.cache_hits == 0
    assert result.judge_metrics.failures == 0
    assert [reward.accuracy for reward in result.rewards] == [1, 1, 1, 1]


@pytest.mark.parametrize("task_kind", ("mcq", "open"))
def test_teacher_batch_route_is_visual_and_keeps_teacher_identity(
    task_kind: str,
) -> None:
    calls: list[object] = []
    teacher = DeepEyesTrajectoryRewardInput(
        trajectory_id=f"teacher-{task_kind}/0",
        sample_id=f"teacher-{task_kind}",
        data_source="teacher",
        task_kind=task_kind,
        question="What is visible?",
        reference_answer="blue",
        response="blue",
        successful_crop_count=0,
    )
    result = score_deepeyes_trajectory_batch(
        (teacher,),
        math_verify=lambda _expected, _candidate: False,
        judge=BoundedBatchSemanticJudge(
            lambda request: calls.append(request) is None or True
        ),
    )
    assert len(calls) == 1
    assert calls[0].task_kind == task_kind
    assert result.judge_metrics.visual_requested == 1
    assert result.judge_metrics.thinklite_fallback_requested == 0
    assert result.rewards[0].accuracy == 1


def test_teacher_batch_route_rejects_math() -> None:
    with pytest.raises(ValueError, match="mcq.*open"):
        DeepEyesTrajectoryRewardInput(
            trajectory_id="teacher/0",
            sample_id="teacher",
            data_source="teacher",
            task_kind="math",
            question="1+1?",
            reference_answer="2",
            response="2",
            successful_crop_count=0,
        )


def test_equal_answers_in_different_trajectories_are_never_deduplicated() -> None:
    calls: list[str] = []
    judge = BoundedBatchSemanticJudge(
        lambda request: calls.append(request.trajectory_id) is None or True
    )
    result = score_deepeyes_trajectory_batch(
        (
            _trajectory("same/0", "vstar", "blue"),
            _trajectory("same/1", "vstar", "blue"),
        ),
        math_verify=lambda _expected, _candidate: False,
        judge=judge,
    )
    assert sorted(calls) == ["same/0", "same/1"]
    assert result.judge_metrics.judge_calls == result.judge_metrics.requested == 2
    assert result.judge_metrics.cache_hits == 0


def test_cache_is_only_same_trajectory_resume_idempotence() -> None:
    call_count = 0

    def transport(_request: object) -> bool:
        nonlocal call_count
        call_count += 1
        return True

    judge = BoundedBatchSemanticJudge(transport)
    batch = (_trajectory("v/0", "vstar", "blue"),)
    first = score_deepeyes_trajectory_batch(
        batch, math_verify=lambda _a, _b: False, judge=judge
    )
    second = score_deepeyes_trajectory_batch(
        batch, math_verify=lambda _a, _b: False, judge=judge
    )
    assert first.judge_metrics.judge_calls == 1
    assert first.judge_metrics.cache_hits == 0
    assert second.judge_metrics.judge_calls == 0
    assert second.judge_metrics.cache_hits == 1
    assert call_count == 1


def test_retry_is_bounded_and_reported() -> None:
    attempts: Counter[str] = Counter()
    sleeps: list[float] = []

    def transport(request: object) -> bool:
        attempts[request.request_id] += 1
        if attempts[request.request_id] == 1:
            raise TimeoutError("transient")
        return True

    result = score_deepeyes_trajectory_batch(
        (_trajectory("v/0", "vstar", "blue"),),
        math_verify=lambda _a, _b: False,
        judge=BoundedBatchSemanticJudge(
            transport,
            config=BatchJudgeConfig(
                max_concurrency=1,
                max_attempts=2,
                retry_backoff_seconds=0.25,
                retry_maximum_seconds=0.25,
                cache_max_entries=0,
            ),
            sleeper=sleeps.append,
        ),
    )
    assert result.judge_metrics.judge_calls == 2
    assert result.judge_metrics.retries == 1
    assert result.judge_metrics.failures == 0
    assert sleeps == [0.25]


def test_completed_malformed_output_zeros_only_current_sample() -> None:
    def transport(request: object) -> bool:
        if request.trajectory_id == "bad/0":
            raise JudgeSampleOutputError("nonbinary")
        return True

    result = score_deepeyes_trajectory_batch(
        (
            _trajectory("bad/0", "vstar", "blue"),
            _trajectory("good/0", "vstar", "blue"),
        ),
        math_verify=lambda _a, _b: False,
        judge=BoundedBatchSemanticJudge(transport),
    )
    assert result.rewards[0].total == 0.0
    assert result.rewards[0].accuracy == 0
    assert result.rewards[1].accuracy == 1
    assert result.judge_metrics.failures == 1
    assert result.judge_metrics.judge_calls == 2


def test_transient_failure_threshold_and_global_failure_abort() -> None:
    tolerant = score_deepeyes_trajectory_batch(
        (_trajectory("v/0", "vstar", "blue"),),
        math_verify=lambda _a, _b: False,
        judge=BoundedBatchSemanticJudge(
            lambda _request: (_ for _ in ()).throw(TimeoutError("down")),
            config=BatchJudgeConfig(
                max_concurrency=1,
                max_attempts=1,
                retry_backoff_seconds=0.0,
                retry_maximum_seconds=0.0,
                cache_max_entries=0,
                maximum_failure_fraction=1.0,
            ),
        ),
    )
    assert tolerant.rewards[0].total == 0.0
    assert tolerant.judge_metrics.failures == 1

    with pytest.raises(BatchJudgeFailure, match="exhausted"):
        score_deepeyes_trajectory_batch(
            (_trajectory("v/1", "vstar", "blue"),),
            math_verify=lambda _a, _b: False,
            judge=BoundedBatchSemanticJudge(
                lambda _request: (_ for _ in ()).throw(TimeoutError("down")),
                config=BatchJudgeConfig(
                    max_concurrency=1,
                    max_attempts=1,
                    retry_backoff_seconds=0.0,
                    retry_maximum_seconds=0.0,
                    cache_max_entries=0,
                    maximum_failure_fraction=0.0,
                ),
            ),
        )

    with pytest.raises(JudgeGlobalFailure):
        score_deepeyes_trajectory_batch(
            (_trajectory("v/2", "vstar", "blue"),),
            math_verify=lambda _a, _b: False,
            judge=BoundedBatchSemanticJudge(
                lambda _request: (_ for _ in ()).throw(
                    JudgeGlobalFailure("model identity mismatch")
                )
            ),
        )


def test_worker_pool_never_exceeds_configured_concurrency() -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()

    def transport(_request: object) -> bool:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return True

    trajectories = tuple(
        _trajectory(f"v/{index}", "vstar", "blue") for index in range(8)
    )
    score_deepeyes_trajectory_batch(
        trajectories,
        math_verify=lambda _a, _b: False,
        judge=BoundedBatchSemanticJudge(
            transport, config=BatchJudgeConfig(max_concurrency=2)
        ),
    )
    assert 1 < maximum <= 2
