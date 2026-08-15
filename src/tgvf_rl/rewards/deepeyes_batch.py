"""Bounded concurrent semantic-judge batch manager for PRL13.

DeepEyes' visual arm produces 197 prompts x 16 trajectories = 3,152
mandatory semantic judgements per stratified optimizer step.  A veRL naive
row-at-a-time reward manager is therefore forbidden.  This implementation
collects the complete reward batch, resolves deterministic/cache hits first,
and dispatches unique Qwen2.5-72B requests through a bounded worker pool with
bounded retries.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
import threading
import time
from typing import Protocol

from tgvf_rl.policy.deepeyes_official_protocol import (
    THINKLITE_SOURCE,
    VISUAL_SOURCES,
    validate_source_task_kind,
)

from .deepeyes_official import (
    DEEPEYES_THINKLITE_ANSWER_WEIGHT,
    DEEPEYES_THINKLITE_FORMAT_WEIGHT,
    DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
    DEEPEYES_VISUAL_ANSWER_LIMIT,
    DEEPEYES_VISUAL_ANSWER_WEIGHT,
    DEEPEYES_VISUAL_CONDITIONAL_TOOL_WEIGHT,
    DEEPEYES_VISUAL_FORMAT_WEIGHT,
    DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
    DeepEyesBinaryJudgeRequest,
    DeepEyesRewardResult,
    MathVerifier,
    extract_thinklite_answer,
    extract_visual_answer,
)


DEEPEYES_BATCH_JUDGE_SCHEMA = "tgvf.deepeyes-batch-semantic-judge.v1"
DEEPEYES_BATCH_MAX_CONCURRENCY = 64
DEEPEYES_BATCH_MAX_ATTEMPTS = 4
DEEPEYES_BATCH_RETRY_BACKOFF_SECONDS = 0.25
DEEPEYES_BATCH_RETRY_MAXIMUM_SECONDS = 2.0
DEEPEYES_BATCH_CACHE_MAX_ENTRIES = 100_000


class JudgeTransport(Protocol):
    """One synchronous text-only Qwen2.5-72B request transport."""

    def __call__(self, request: DeepEyesBinaryJudgeRequest) -> bool: ...


@dataclass(frozen=True, slots=True)
class BatchJudgeConfig:
    max_concurrency: int = DEEPEYES_BATCH_MAX_CONCURRENCY
    max_attempts: int = DEEPEYES_BATCH_MAX_ATTEMPTS
    retry_backoff_seconds: float = DEEPEYES_BATCH_RETRY_BACKOFF_SECONDS
    retry_maximum_seconds: float = DEEPEYES_BATCH_RETRY_MAXIMUM_SECONDS
    cache_max_entries: int = DEEPEYES_BATCH_CACHE_MAX_ENTRIES
    maximum_failure_fraction: float = 0.01

    def __post_init__(self) -> None:
        if (
            type(self.max_concurrency) is not int
            or not 1 <= self.max_concurrency <= 256
        ):
            raise ValueError("judge max_concurrency must lie in [1,256]")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 16:
            raise ValueError("judge max_attempts must lie in [1,16]")
        for name in ("retry_backoff_seconds", "retry_maximum_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"judge {name} must be finite and non-negative")
        if self.retry_maximum_seconds < self.retry_backoff_seconds:
            raise ValueError("judge maximum retry delay is smaller than base delay")
        if type(self.cache_max_entries) is not int or self.cache_max_entries < 0:
            raise ValueError("judge cache_max_entries must be non-negative")
        if (
            not math.isfinite(self.maximum_failure_fraction)
            or not 0.0 <= self.maximum_failure_fraction <= 1.0
        ):
            raise ValueError("judge maximum_failure_fraction must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class BatchJudgeMetrics:
    requested: int
    visual_requested: int
    thinklite_fallback_requested: int
    unique_requests: int
    judge_calls: int
    cache_hits: int
    retries: int
    failures: int
    latency_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "requested",
            "visual_requested",
            "thinklite_fallback_requested",
            "unique_requests",
            "judge_calls",
            "cache_hits",
            "retries",
            "failures",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"batch judge {name} must be non-negative")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("batch judge latency must be finite and non-negative")
        if self.requested != (
            self.visual_requested + self.thinklite_fallback_requested
        ):
            raise ValueError("batch judge route request counts differ")

    def as_metrics(self) -> dict[str, int | float]:
        return {
            "judge_requested": self.requested,
            "visual_judge_requested": self.visual_requested,
            "thinklite_fallback_judge_requested": (self.thinklite_fallback_requested),
            "judge_unique_requests": self.unique_requests,
            "judge_calls": self.judge_calls,
            "judge_cache_hits": self.cache_hits,
            "judge_retries": self.retries,
            "judge_failures": self.failures,
            "judge_latency_seconds": self.latency_seconds,
        }


@dataclass(frozen=True, slots=True)
class BatchJudgeResult:
    verdicts: tuple[bool, ...]
    failure_kinds: tuple[str | None, ...]
    metrics: BatchJudgeMetrics


class BatchJudgeFailure(RuntimeError):
    def __init__(self, request_id: str, *, attempts: int, cause: BaseException):
        super().__init__(
            f"DeepEyes semantic judge exhausted {attempts} attempts for {request_id}"
        )
        self.request_id = request_id
        self.attempts = attempts
        self.__cause__ = cause


class JudgeSampleOutputError(ValueError):
    """A completed response is malformed/nonbinary; only that sample is zero."""


class JudgeGlobalFailure(RuntimeError):
    """Authentication/model-identity/global service errors abort the batch."""


class BoundedBatchSemanticJudge:
    """Thread-safe bounded batch dispatcher with LRU request-id cache."""

    def __init__(
        self,
        transport: JudgeTransport,
        *,
        config: BatchJudgeConfig = BatchJudgeConfig(),
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(transport):
            raise TypeError("judge transport must be callable")
        self.transport = transport
        self.config = config
        self.sleeper = sleeper
        self.clock = clock
        self._cache: OrderedDict[str, bool] = OrderedDict()
        self._lock = threading.Lock()

    def judge_many(
        self, requests: Sequence[DeepEyesBinaryJudgeRequest]
    ) -> BatchJudgeResult:
        started = self.clock()
        request_tuple = tuple(requests)
        if any(
            not isinstance(item, DeepEyesBinaryJudgeRequest) for item in request_tuple
        ):
            raise TypeError("judge_many requires DeepEyesBinaryJudgeRequest values")
        request_by_id: dict[str, DeepEyesBinaryJudgeRequest] = {}
        for request in request_tuple:
            existing = request_by_id.setdefault(request.request_id, request)
            if existing != request:
                raise ValueError("judge request_id collision has different payloads")

        verdict_by_id: dict[str, bool] = {}
        failure_by_id: dict[str, str | None] = {}
        with self._lock:
            for request_id in request_by_id:
                if request_id in self._cache:
                    verdict_by_id[request_id] = self._cache[request_id]
                    failure_by_id[request_id] = None
                    self._cache.move_to_end(request_id)

        missing = [
            request
            for request_id, request in request_by_id.items()
            if request_id not in verdict_by_id
        ]
        judge_calls = 0
        retries = 0
        failures = 0
        if missing:
            with ThreadPoolExecutor(
                max_workers=min(self.config.max_concurrency, len(missing)),
                thread_name_prefix="deepeyes-judge",
            ) as executor:
                futures: dict[
                    Future[tuple[bool, int, str | None]],
                    DeepEyesBinaryJudgeRequest,
                ] = {
                    executor.submit(self._judge_with_retry, request): request
                    for request in missing
                }
                for future in as_completed(futures):
                    request = futures[future]
                    verdict, attempts, failure_kind = future.result()
                    judge_calls += attempts
                    retries += attempts - 1
                    verdict_by_id[request.request_id] = verdict
                    failure_by_id[request.request_id] = failure_kind
                    if failure_kind is None:
                        self._cache_put(request.request_id, verdict)
                    else:
                        failures += 1

        transient_failures = sum(
            failure == "transient_exhausted" for failure in failure_by_id.values()
        )
        if missing and transient_failures / len(missing) > (
            self.config.maximum_failure_fraction
        ):
            raise BatchJudgeFailure(
                "batch_transient_failure_threshold",
                attempts=self.config.max_attempts,
                cause=RuntimeError(
                    "transient judge failures exceed configured batch fraction"
                ),
            )

        # A duplicate in the current batch receives a verdict but only one
        # physical request.  Count both persistent-cache and in-batch reuse.
        cache_hits = len(request_tuple) - len(missing)
        elapsed = max(0.0, self.clock() - started)
        return BatchJudgeResult(
            verdicts=tuple(
                verdict_by_id[request.request_id] for request in request_tuple
            ),
            failure_kinds=tuple(
                failure_by_id[request.request_id] for request in request_tuple
            ),
            metrics=BatchJudgeMetrics(
                requested=len(request_tuple),
                visual_requested=sum(
                    request.prompt_kind == DEEPEYES_VISUAL_JUDGE_PROMPT_KIND
                    for request in request_tuple
                ),
                thinklite_fallback_requested=sum(
                    request.prompt_kind == DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND
                    for request in request_tuple
                ),
                unique_requests=len(request_by_id),
                judge_calls=judge_calls,
                cache_hits=cache_hits,
                retries=retries,
                failures=failures,
                latency_seconds=elapsed,
            ),
        )

    def _judge_with_retry(
        self, request: DeepEyesBinaryJudgeRequest
    ) -> tuple[bool, int, str | None]:
        last_error: BaseException | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                verdict = self.transport(request)
                if type(verdict) is not bool:
                    raise JudgeSampleOutputError(
                        "completed judge output is malformed/nonbinary"
                    )
                return verdict, attempt, None
            except JudgeSampleOutputError:
                return False, attempt, "completed_invalid_output"
            except JudgeGlobalFailure:
                raise
            except (ConnectionError, TimeoutError, RuntimeError) as error:
                last_error = error
                if attempt == self.config.max_attempts:
                    break
                delay = min(
                    self.config.retry_maximum_seconds,
                    self.config.retry_backoff_seconds * (2 ** (attempt - 1)),
                )
                self.sleeper(delay)
        assert last_error is not None
        return False, self.config.max_attempts, "transient_exhausted"

    def _cache_put(self, request_id: str, verdict: bool) -> None:
        if self.config.cache_max_entries == 0:
            return
        with self._lock:
            self._cache[request_id] = verdict
            self._cache.move_to_end(request_id)
            while len(self._cache) > self.config.cache_max_entries:
                self._cache.popitem(last=False)


@dataclass(frozen=True, slots=True)
class DeepEyesTrajectoryRewardInput:
    trajectory_id: str
    sample_id: str
    data_source: str
    task_kind: str
    question: str
    reference_answer: str
    response: str
    successful_crop_count: int

    def __post_init__(self) -> None:
        for name in (
            "trajectory_id",
            "sample_id",
            "data_source",
            "task_kind",
            "question",
            "reference_answer",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"trajectory {name} must be non-empty text")
        validate_source_task_kind(self.data_source, self.task_kind)
        if (
            type(self.successful_crop_count) is not int
            or self.successful_crop_count < 0
        ):
            raise ValueError("successful_crop_count must be non-negative")
        if self.data_source == THINKLITE_SOURCE and self.successful_crop_count:
            raise ValueError("ThinkLite trajectory cannot have a successful crop")


@dataclass(frozen=True, slots=True)
class DeepEyesRewardBatchResult:
    rewards: tuple[DeepEyesRewardResult, ...]
    judge_metrics: BatchJudgeMetrics


@dataclass(slots=True)
class _PreparedTrajectory:
    input: DeepEyesTrajectoryRewardInput
    answer: str
    format_penalty: int
    rule_correct: bool = False
    judge_index: int | None = None


def score_deepeyes_trajectory_batch(
    trajectories: Sequence[DeepEyesTrajectoryRewardInput],
    *,
    math_verify: MathVerifier,
    judge: BoundedBatchSemanticJudge,
) -> DeepEyesRewardBatchResult:
    """Score one veRL reward batch with one bounded semantic-judge dispatch."""

    inputs = tuple(trajectories)
    if len({item.trajectory_id for item in inputs}) != len(inputs):
        raise ValueError("reward batch contains duplicate trajectory_id")
    prepared: list[_PreparedTrajectory] = []
    judge_requests: list[DeepEyesBinaryJudgeRequest] = []
    for item in inputs:
        if item.data_source in VISUAL_SOURCES:
            extraction = extract_visual_answer(item.response)
            candidate = extraction.answer or "[NO VALID FINAL ANSWER]"
            judge_index = len(judge_requests)
            judge_requests.append(
                DeepEyesBinaryJudgeRequest.build(
                    trajectory_id=item.trajectory_id,
                    sample_id=item.sample_id,
                    question=item.question,
                    reference_answer=item.reference_answer,
                    candidate_answer=candidate,
                    task_kind=item.task_kind,
                    prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
                )
            )
            prepared.append(
                _PreparedTrajectory(
                    input=item,
                    answer=extraction.answer,
                    format_penalty=extraction.format_penalty,
                    judge_index=judge_index,
                )
            )
            continue

        if item.task_kind in {"open", "mcq"}:
            extraction = extract_thinklite_answer(item.response)
            judge_index = len(judge_requests)
            judge_requests.append(
                DeepEyesBinaryJudgeRequest.build(
                    trajectory_id=item.trajectory_id,
                    sample_id=item.sample_id,
                    question=item.question,
                    reference_answer=item.reference_answer,
                    candidate_answer=extraction.answer or "[NO VALID FINAL ANSWER]",
                    task_kind=item.task_kind,
                    prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
                )
            )
            prepared.append(
                _PreparedTrajectory(
                    input=item,
                    answer=extraction.answer,
                    format_penalty=extraction.format_penalty,
                    judge_index=judge_index,
                )
            )
            continue
        if item.task_kind != "math":
            raise ValueError("ThinkLite task_kind must be math, open, or mcq")
        extraction = extract_thinklite_answer(item.response)
        rule_correct = False
        if extraction.answer:
            try:
                rule_correct = math_verify(item.reference_answer, extraction.answer)
            except Exception:
                rule_correct = False
            if type(rule_correct) is not bool:
                raise TypeError("math_verify must return bool")
        judge_index = None
        if extraction.answer and not rule_correct:
            judge_index = len(judge_requests)
            judge_requests.append(
                DeepEyesBinaryJudgeRequest.build(
                    trajectory_id=item.trajectory_id,
                    sample_id=item.sample_id,
                    question=item.question,
                    reference_answer=item.reference_answer,
                    candidate_answer=extraction.answer,
                    task_kind=item.task_kind,
                    prompt_kind=DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
                )
            )
        prepared.append(
            _PreparedTrajectory(
                input=item,
                answer=extraction.answer,
                format_penalty=extraction.format_penalty,
                rule_correct=rule_correct,
                judge_index=judge_index,
            )
        )

    judged = judge.judge_many(judge_requests)
    rewards: list[DeepEyesRewardResult] = []
    for item in prepared:
        source = item.input.data_source
        if source in VISUAL_SOURCES:
            assert item.judge_index is not None
            sample_failure = judged.failure_kinds[item.judge_index]
            too_long = len(item.answer) >= DEEPEYES_VISUAL_ANSWER_LIMIT
            accuracy = int(judged.verdicts[item.judge_index] and not too_long)
            format_penalty = -1 if too_long else item.format_penalty
            conditional_tool = int(
                accuracy == 1 and item.input.successful_crop_count > 0
            )
            total = (
                DEEPEYES_VISUAL_ANSWER_WEIGHT * accuracy
                + DEEPEYES_VISUAL_FORMAT_WEIGHT * format_penalty
                + DEEPEYES_VISUAL_CONDITIONAL_TOOL_WEIGHT * conditional_tool
            )
            if sample_failure is not None:
                total = 0.0
                accuracy = 0
                conditional_tool = 0
            rewards.append(
                DeepEyesRewardResult(
                    total=total,
                    accuracy=accuracy,
                    format_penalty=format_penalty,
                    conditional_tool=conditional_tool,
                    answer=item.answer,
                    verifier_route=(
                        "qwen2.5_72b_every_visual_trajectory_batch"
                        if sample_failure is None
                        else "qwen2.5_72b_visual_" + sample_failure + "_zero"
                    ),
                    judge_called=True,
                )
            )
        else:
            if item.input.task_kind in {"open", "mcq"}:
                assert item.judge_index is not None
                sample_failure = judged.failure_kinds[item.judge_index]
                too_long = len(item.answer) >= DEEPEYES_VISUAL_ANSWER_LIMIT
                accuracy = int(judged.verdicts[item.judge_index] and not too_long)
                format_penalty = -1 if too_long else item.format_penalty
                route = "thinklite_no_tool_qwen2.5_72b_batch"
                judge_called = True
            elif item.rule_correct:
                accuracy = 1
                route = "math_verify"
                judge_called = False
                sample_failure = None
            elif item.judge_index is not None:
                sample_failure = judged.failure_kinds[item.judge_index]
                accuracy = int(judged.verdicts[item.judge_index])
                route = "math_verify_then_qwen2.5_72b_batch"
                judge_called = True
            else:
                accuracy = 0
                route = "missing_boxed_answer"
                judge_called = False
                sample_failure = None
            total = (
                DEEPEYES_THINKLITE_ANSWER_WEIGHT * accuracy
                + DEEPEYES_THINKLITE_FORMAT_WEIGHT
                * (
                    format_penalty
                    if item.input.task_kind in {"open", "mcq"}
                    else item.format_penalty
                )
            )
            if sample_failure is not None:
                total = 0.0
                accuracy = 0
                route = "thinklite_" + sample_failure + "_zero"
            rewards.append(
                DeepEyesRewardResult(
                    total=total,
                    accuracy=accuracy,
                    format_penalty=(
                        format_penalty
                        if item.input.task_kind in {"open", "mcq"}
                        else item.format_penalty
                    ),
                    conditional_tool=0,
                    answer=item.answer,
                    verifier_route=route,
                    judge_called=judge_called,
                )
            )
    return DeepEyesRewardBatchResult(tuple(rewards), judged.metrics)


__all__ = [
    "DEEPEYES_BATCH_CACHE_MAX_ENTRIES",
    "DEEPEYES_BATCH_JUDGE_SCHEMA",
    "DEEPEYES_BATCH_MAX_ATTEMPTS",
    "DEEPEYES_BATCH_MAX_CONCURRENCY",
    "BatchJudgeConfig",
    "BatchJudgeFailure",
    "BatchJudgeMetrics",
    "BatchJudgeResult",
    "BoundedBatchSemanticJudge",
    "DeepEyesRewardBatchResult",
    "DeepEyesTrajectoryRewardInput",
    "JudgeTransport",
    "JudgeGlobalFailure",
    "JudgeSampleOutputError",
    "score_deepeyes_trajectory_batch",
]
