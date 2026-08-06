"""Deterministic Policy Pilot v1 metric accumulation and checkpoint state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from tgvf_rl.protocol.schema import ToolErrorCode


POLICY_PILOT_V1_METRICS_SCHEMA = "policy-pilot-v1-metrics-v1"
POLICY_PILOT_V1_TRAJECTORIES_PER_PROMPT = 8
POLICY_SUPPORTED_TRAJECTORIES_PER_PROMPT = (8, 16)
POLICY_PILOT_V1_ADMITTED_TOOL_ATTEMPTS = 4
POLICY_PILOT_V1_MAX_RECORDED_TOOL_ATTEMPTS = 5


@dataclass(frozen=True, slots=True)
class MetricReductionContract:
    """One public mean/rate denominator and its zero-denominator behavior."""

    metric: str
    numerator: str
    denominator: str
    zero_denominator_value: float = 0.0


PILOT_V1_METRIC_REDUCTIONS = (
    MetricReductionContract(
        "tool_call_attempt_rate",
        "trajectories_with_tool_call_attempts",
        "trajectories",
    ),
    MetricReductionContract(
        "mean_tool_call_attempts", "tool_call_attempts", "trajectories"
    ),
    MetricReductionContract(
        "mean_answer_reward", "answer_reward_total", "trajectories"
    ),
    MetricReductionContract("format_error_rate", "format_errors", "trajectories"),
    MetricReductionContract(
        "mean_conditional_tool_reward",
        "conditional_tool_reward_total",
        "trajectories",
    ),
    MetricReductionContract(
        "mean_reasoning_length", "reasoning_tokens", "trajectories"
    ),
    MetricReductionContract(
        "mean_original_visual_tokens", "original_visual_tokens", "trajectories"
    ),
    MetricReductionContract(
        "mean_total_visual_tokens", "total_visual_tokens", "trajectories"
    ),
    MetricReductionContract(
        "mean_step_time_seconds", "step_time_seconds_total", "optimizer_steps"
    ),
)


@dataclass(frozen=True, slots=True)
class ToolErrorCount:
    """Exact error-code count; distinct codes are never collapsed together."""

    code: str
    count: int

    def __post_init__(self) -> None:
        if type(self.code) is not str or not self.code.strip():
            raise ValueError("tool error code must be a non-empty string")
        _nonnegative_int(self.count, "tool error count")
        if self.count == 0:
            raise ValueError("stored tool error counts must be positive")


@dataclass(frozen=True, slots=True)
class PilotTrajectoryMetricsObservation:
    """Raw metric facts for one retained Pilot trajectory.

    ``tool_call_attempts`` includes every assistant tool-call attempt, including
    the fifth cap-error attempt.  Each attempt must yield either one successful
    TGVF observation or one typed error code.  Reward fields are the raw
    unweighted Pilot components, not their weighted scalar contributions.
    """

    prompt_id: str
    trajectory_id: str
    generated_policy_tokens: int
    successful_tgvf_observations: int
    tool_call_attempts: int
    answer_reward: float
    format_error: bool
    conditional_tool_reward: float
    reasoning_tokens: int
    original_visual_tokens: int
    total_visual_tokens: int
    tool_error_codes: tuple[str, ...] = ()
    judge_calls: int = 0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_cost_usd: float = 0.0
    reward_profile: str = "pilot-v1"
    stage3_reward_components: tuple[float, float, float, float, float] | None = None
    stage3_quality_judge_applicable: bool = False
    stage3_quality_judge_covered: bool = False
    stage3_quality_judge_failure: str | None = None
    stage3_visual_judge_calls: int = 0
    stage3_visual_judge_prompt_tokens: int = 0
    stage3_visual_judge_completion_tokens: int = 0
    stage3_visual_judge_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if type(self.prompt_id) is not str or not self.prompt_id.strip():
            raise ValueError("prompt_id must be a non-empty string")
        if type(self.trajectory_id) is not str or not self.trajectory_id.strip():
            raise ValueError("trajectory_id must be a non-empty string")
        for name in (
            "generated_policy_tokens",
            "successful_tgvf_observations",
            "tool_call_attempts",
            "reasoning_tokens",
            "original_visual_tokens",
            "total_visual_tokens",
            "judge_calls",
            "judge_prompt_tokens",
            "judge_completion_tokens",
            "stage3_visual_judge_calls",
            "stage3_visual_judge_prompt_tokens",
            "stage3_visual_judge_completion_tokens",
        ):
            _nonnegative_int(getattr(self, name), name)
        if self.judge_calls not in {0, 1}:
            raise ValueError("one trajectory can invoke the answer judge at most once")
        judge_cost = _finite_float(self.judge_cost_usd, "judge_cost_usd")
        if judge_cost < 0.0:
            raise ValueError("judge_cost_usd must be non-negative")
        object.__setattr__(self, "judge_cost_usd", judge_cost)
        if self.judge_calls == 0 and (
            self.judge_prompt_tokens
            or self.judge_completion_tokens
            or self.judge_cost_usd
        ):
            raise ValueError("judge usage requires a judge call")
        if self.reward_profile not in {"pilot-v1", "stage3-shaped-v1"}:
            raise ValueError("unsupported trajectory metrics reward profile")
        maximum_attempts = (
            POLICY_PILOT_V1_MAX_RECORDED_TOOL_ATTEMPTS
            if self.reward_profile == "pilot-v1"
            else 2
        )
        admitted_attempts = (
            POLICY_PILOT_V1_ADMITTED_TOOL_ATTEMPTS
            if self.reward_profile == "pilot-v1"
            else 1
        )
        if self.tool_call_attempts > maximum_attempts:
            raise ValueError("tool-call attempts exceed the reward profile bound")
        if self.successful_tgvf_observations > admitted_attempts:
            raise ValueError("successful observations exceed the reward profile bound")
        if self.successful_tgvf_observations > self.tool_call_attempts:
            raise ValueError("successful observations cannot exceed tool attempts")
        if self.reasoning_tokens > self.generated_policy_tokens:
            raise ValueError("reasoning tokens cannot exceed generated policy tokens")
        if self.original_visual_tokens > self.total_visual_tokens:
            raise ValueError("total visual tokens must include original visual tokens")
        if type(self.format_error) is not bool:
            raise TypeError("format_error must be bool")

        answer = _binary_reward(self.answer_reward, "answer_reward")
        conditional = _binary_reward(
            self.conditional_tool_reward, "conditional_tool_reward"
        )
        object.__setattr__(self, "answer_reward", answer)
        object.__setattr__(self, "conditional_tool_reward", conditional)
        if not isinstance(self.tool_error_codes, Sequence) or isinstance(
            self.tool_error_codes, (str, bytes)
        ):
            raise TypeError("tool_error_codes must be a sequence of exact codes")
        object.__setattr__(self, "tool_error_codes", tuple(self.tool_error_codes))
        if any(
            type(code) is not str or not code.strip() for code in self.tool_error_codes
        ):
            raise ValueError("tool_error_codes must contain non-empty strings")
        if (
            self.successful_tgvf_observations + len(self.tool_error_codes)
            != self.tool_call_attempts
        ):
            raise ValueError(
                "each tool-call attempt must yield one TGVF observation or one error"
            )
        cap_code = ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value
        if self.tool_call_attempts == maximum_attempts:
            if self.tool_error_codes.count(cap_code) != 1:
                if self.reward_profile == "pilot-v1":
                    raise ValueError(
                        "the fifth Pilot attempt must record one cap error"
                    )
                raise ValueError("the second Stage3 attempt must record one cap error")
        elif cap_code in self.tool_error_codes:
            if self.reward_profile == "pilot-v1":
                raise ValueError("a cap error is valid only on the fifth attempt")
            raise ValueError("a cap error is valid only on the second Stage3 attempt")
        if conditional == 1.0 and (
            answer != 1.0 or self.successful_tgvf_observations == 0
        ):
            raise ValueError(
                "conditional tool reward requires a correct answer and successful D"
            )
        self._validate_stage3_metrics()

    def _validate_stage3_metrics(self) -> None:
        stage3 = self.stage3_reward_components
        if self.reward_profile == "pilot-v1":
            if stage3 is not None or any(
                (
                    self.stage3_quality_judge_applicable,
                    self.stage3_quality_judge_covered,
                    self.stage3_quality_judge_failure is not None,
                    self.stage3_visual_judge_calls,
                    self.stage3_visual_judge_prompt_tokens,
                    self.stage3_visual_judge_completion_tokens,
                    self.stage3_visual_judge_cost_usd,
                )
            ):
                raise ValueError("Pilot-v1 metrics cannot carry Stage3 fields")
            return
        if (
            not isinstance(stage3, tuple)
            or len(stage3) != 5
            or any(not math.isfinite(float(value)) for value in stage3)
        ):
            raise ValueError("Stage3 metrics require five finite reward components")
        for name in (
            "stage3_quality_judge_applicable",
            "stage3_quality_judge_covered",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if not self.stage3_quality_judge_applicable and (
            self.stage3_quality_judge_covered
            or self.stage3_quality_judge_failure is not None
        ):
            raise ValueError(
                "non-applicable Stage3 quality judge cannot be covered/failed"
            )
        if self.stage3_quality_judge_applicable and (
            self.stage3_quality_judge_covered
            == (self.stage3_quality_judge_failure is not None)
        ):
            raise ValueError("Stage3 quality coverage/failure fields differ")
        if self.stage3_quality_judge_failure is not None and (
            not isinstance(self.stage3_quality_judge_failure, str)
            or not self.stage3_quality_judge_failure.strip()
        ):
            raise ValueError("Stage3 quality failure code is invalid")
        if self.stage3_visual_judge_calls != int(self.stage3_quality_judge_applicable):
            raise ValueError("Stage3 visual judge calls differ from applicability")
        visual_cost = _finite_float(
            self.stage3_visual_judge_cost_usd,
            "stage3_visual_judge_cost_usd",
        )
        if visual_cost < 0.0:
            raise ValueError("stage3_visual_judge_cost_usd must be non-negative")
        object.__setattr__(self, "stage3_visual_judge_cost_usd", visual_cost)
        if self.stage3_visual_judge_calls == 0 and (
            self.stage3_visual_judge_prompt_tokens
            or self.stage3_visual_judge_completion_tokens
            or visual_cost
        ):
            raise ValueError("Stage3 visual judge usage requires a call")


@dataclass(frozen=True, slots=True)
class PilotOptimizerStepMetricsObservation:
    """One complete global optimizer step and its non-duplicated wall time."""

    optimizer_step: int
    step_time_seconds: float
    trajectories: tuple[PilotTrajectoryMetricsObservation, ...]
    trajectories_per_prompt: int = POLICY_PILOT_V1_TRAJECTORIES_PER_PROMPT

    def __post_init__(self) -> None:
        _positive_int(self.optimizer_step, "optimizer_step")
        elapsed = _finite_float(self.step_time_seconds, "step_time_seconds")
        if elapsed <= 0.0:
            raise ValueError("step_time_seconds must be positive")
        object.__setattr__(self, "step_time_seconds", elapsed)
        object.__setattr__(self, "trajectories", tuple(self.trajectories))
        _positive_int(self.trajectories_per_prompt, "trajectories_per_prompt")
        if self.trajectories_per_prompt not in POLICY_SUPPORTED_TRAJECTORIES_PER_PROMPT:
            raise ValueError(
                "unsupported trajectories_per_prompt; accepted="
                f"{POLICY_SUPPORTED_TRAJECTORIES_PER_PROMPT!r}"
            )
        if not self.trajectories:
            raise ValueError("an optimizer-step metric observation cannot be empty")
        if any(
            not isinstance(row, PilotTrajectoryMetricsObservation)
            for row in self.trajectories
        ):
            raise TypeError("trajectories must contain typed Pilot observations")
        trajectory_ids = tuple(row.trajectory_id for row in self.trajectories)
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("trajectory IDs must be unique within an optimizer step")
        prompt_sizes = Counter(row.prompt_id for row in self.trajectories)
        incomplete = {
            prompt_id: count
            for prompt_id, count in prompt_sizes.items()
            if count != self.trajectories_per_prompt
        }
        if incomplete:
            group_label = (
                "eight"
                if self.trajectories_per_prompt
                == POLICY_PILOT_V1_TRAJECTORIES_PER_PROMPT
                else str(self.trajectories_per_prompt)
            )
            raise ValueError(
                "each Policy prompt must contribute exactly "
                f"{group_label} trajectories: "
                f"{dict(sorted(incomplete.items()))!r}"
            )

    @property
    def prompt_count(self) -> int:
        """Prompt presentations in this step; repeated IDs in later steps recount."""

        return len({row.prompt_id for row in self.trajectories})


@dataclass(frozen=True, slots=True)
class PilotMetricsCheckpointState:
    """Lossless additive numerators persisted at every checkpoint boundary."""

    schema_version: str = POLICY_PILOT_V1_METRICS_SCHEMA
    optimizer_steps: int = 0
    prompts: int = 0
    trajectories: int = 0
    generated_policy_tokens: int = 0
    successful_tgvf_observations: int = 0
    trajectories_with_tool_call_attempts: int = 0
    tool_call_attempts: int = 0
    answer_reward_total: int = 0
    format_errors: int = 0
    conditional_tool_reward_total: int = 0
    reasoning_tokens: int = 0
    original_visual_tokens: int = 0
    total_visual_tokens: int = 0
    judge_calls: int = 0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_cost_usd: float = 0.0
    step_time_seconds_total: float = 0.0
    tool_error_counts: tuple[ToolErrorCount, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_PILOT_V1_METRICS_SCHEMA:
            raise ValueError("unsupported Policy Pilot metrics checkpoint schema")
        integer_fields = (
            "optimizer_steps",
            "prompts",
            "trajectories",
            "generated_policy_tokens",
            "successful_tgvf_observations",
            "trajectories_with_tool_call_attempts",
            "tool_call_attempts",
            "answer_reward_total",
            "format_errors",
            "conditional_tool_reward_total",
            "reasoning_tokens",
            "original_visual_tokens",
            "total_visual_tokens",
            "judge_calls",
            "judge_prompt_tokens",
            "judge_completion_tokens",
        )
        for name in integer_fields:
            _nonnegative_int(getattr(self, name), name)
        elapsed = _finite_float(self.step_time_seconds_total, "step_time_seconds_total")
        if elapsed < 0.0:
            raise ValueError("step_time_seconds_total must be non-negative")
        object.__setattr__(self, "step_time_seconds_total", elapsed)
        judge_cost = _finite_float(self.judge_cost_usd, "judge_cost_usd")
        if judge_cost < 0.0:
            raise ValueError("judge_cost_usd must be non-negative")
        object.__setattr__(self, "judge_cost_usd", judge_cost)
        object.__setattr__(self, "tool_error_counts", tuple(self.tool_error_counts))
        if any(not isinstance(item, ToolErrorCount) for item in self.tool_error_counts):
            raise TypeError("tool_error_counts must contain ToolErrorCount values")
        codes = tuple(item.code for item in self.tool_error_counts)
        if codes != tuple(sorted(codes)) or len(codes) != len(set(codes)):
            raise ValueError("tool_error_counts must be unique and sorted by code")
        self._validate_reduction_invariants()

    def _validate_reduction_invariants(self) -> None:
        if self.prompts == 0:
            if self.trajectories != 0:
                raise ValueError(
                    "zero-prompt metrics state cannot contain trajectories"
                )
            inferred_group_size = POLICY_PILOT_V1_TRAJECTORIES_PER_PROMPT
        elif self.trajectories % self.prompts:
            raise ValueError(
                "Policy trajectory count must be divisible by prompt count"
            )
        else:
            inferred_group_size = self.trajectories // self.prompts
        if inferred_group_size not in POLICY_SUPPORTED_TRAJECTORIES_PER_PROMPT:
            raise ValueError(
                "Policy trajectory count must equal prompts multiplied by 8 "
                "or 16; inferred group size="
                f"{inferred_group_size}"
            )
        if self.optimizer_steps == 0:
            if self.prompts != 0 or self.step_time_seconds_total != 0.0:
                raise ValueError(
                    "empty metrics state must have no prompts or step time"
                )
        elif self.prompts < self.optimizer_steps or self.step_time_seconds_total <= 0.0:
            raise ValueError(
                "each optimizer step must contain prompts and positive time"
            )
        bounded_by_trajectories = {
            "trajectories_with_tool_call_attempts": (
                self.trajectories_with_tool_call_attempts
            ),
            "answer_reward_total": self.answer_reward_total,
            "format_errors": self.format_errors,
            "conditional_tool_reward_total": self.conditional_tool_reward_total,
            "judge_calls": self.judge_calls,
        }
        if any(value > self.trajectories for value in bounded_by_trajectories.values()):
            raise ValueError("trajectory metric numerators cannot exceed trajectories")
        if self.tool_call_attempts > (
            POLICY_PILOT_V1_MAX_RECORDED_TOOL_ATTEMPTS * self.trajectories
        ):
            raise ValueError("tool-call attempts exceed the Pilot trajectory bound")
        if self.successful_tgvf_observations > (
            POLICY_PILOT_V1_ADMITTED_TOOL_ATTEMPTS * self.trajectories
        ):
            raise ValueError("successful observations exceed the Pilot bound")
        if self.successful_tgvf_observations > self.tool_call_attempts:
            raise ValueError("successful observations cannot exceed tool attempts")
        if self.trajectories_with_tool_call_attempts > self.tool_call_attempts:
            raise ValueError("attempting trajectories cannot exceed total attempts")
        if self.conditional_tool_reward_total > self.answer_reward_total:
            raise ValueError("conditional tool rewards require answer rewards")
        if self.conditional_tool_reward_total > self.successful_tgvf_observations:
            raise ValueError("conditional tool rewards require successful observations")
        if self.reasoning_tokens > self.generated_policy_tokens:
            raise ValueError("reasoning tokens cannot exceed generated policy tokens")
        if self.original_visual_tokens > self.total_visual_tokens:
            raise ValueError("total visual tokens must include original visual tokens")
        if self.judge_calls == 0 and (
            self.judge_prompt_tokens
            or self.judge_completion_tokens
            or self.judge_cost_usd
        ):
            raise ValueError("judge usage totals require judge calls")
        if self.judge_calls and self.judge_prompt_tokens < self.judge_calls:
            raise ValueError("each judge call requires prompt tokens")
        if (
            sum(item.count for item in self.tool_error_counts)
            + (self.successful_tgvf_observations)
            != self.tool_call_attempts
        ):
            raise ValueError("tool observations plus typed errors must equal attempts")

    def to_checkpoint_mapping(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible checkpoint payload."""

        return {
            "schema_version": self.schema_version,
            "optimizer_steps": self.optimizer_steps,
            "prompts": self.prompts,
            "trajectories": self.trajectories,
            "generated_policy_tokens": self.generated_policy_tokens,
            "successful_tgvf_observations": self.successful_tgvf_observations,
            "trajectories_with_tool_call_attempts": (
                self.trajectories_with_tool_call_attempts
            ),
            "tool_call_attempts": self.tool_call_attempts,
            "answer_reward_total": self.answer_reward_total,
            "format_errors": self.format_errors,
            "conditional_tool_reward_total": self.conditional_tool_reward_total,
            "judge_calls": self.judge_calls,
            "judge_prompt_tokens": self.judge_prompt_tokens,
            "judge_completion_tokens": self.judge_completion_tokens,
            "judge_cost_usd": self.judge_cost_usd,
            "reasoning_tokens": self.reasoning_tokens,
            "original_visual_tokens": self.original_visual_tokens,
            "total_visual_tokens": self.total_visual_tokens,
            "step_time_seconds_total": self.step_time_seconds_total,
            "tool_error_counts": [
                [item.code, item.count] for item in self.tool_error_counts
            ],
        }

    @classmethod
    def from_checkpoint_mapping(
        cls, payload: Mapping[str, object]
    ) -> "PilotMetricsCheckpointState":
        if not isinstance(payload, Mapping):
            raise TypeError("metrics checkpoint state must be a mapping")
        expected = {
            "schema_version",
            "optimizer_steps",
            "prompts",
            "trajectories",
            "generated_policy_tokens",
            "successful_tgvf_observations",
            "trajectories_with_tool_call_attempts",
            "tool_call_attempts",
            "answer_reward_total",
            "format_errors",
            "conditional_tool_reward_total",
            "judge_calls",
            "judge_prompt_tokens",
            "judge_completion_tokens",
            "judge_cost_usd",
            "reasoning_tokens",
            "original_visual_tokens",
            "total_visual_tokens",
            "step_time_seconds_total",
            "tool_error_counts",
        }
        if set(payload) != expected:
            missing = tuple(sorted(expected - set(payload)))
            extra = tuple(sorted(set(payload) - expected))
            raise ValueError(
                f"metrics checkpoint fields differ: missing={missing!r} extra={extra!r}"
            )
        raw_errors = payload["tool_error_counts"]
        if not isinstance(raw_errors, Sequence) or isinstance(raw_errors, (str, bytes)):
            raise TypeError("tool_error_counts checkpoint field must be a sequence")
        errors: list[ToolErrorCount] = []
        for row in raw_errors:
            if (
                not isinstance(row, Sequence)
                or isinstance(row, (str, bytes))
                or len(row) != 2
            ):
                raise TypeError("each checkpoint tool error must be [code, count]")
            errors.append(ToolErrorCount(row[0], row[1]))
        return cls(
            schema_version=payload["schema_version"],
            optimizer_steps=payload["optimizer_steps"],
            prompts=payload["prompts"],
            trajectories=payload["trajectories"],
            generated_policy_tokens=payload["generated_policy_tokens"],
            successful_tgvf_observations=payload["successful_tgvf_observations"],
            trajectories_with_tool_call_attempts=payload[
                "trajectories_with_tool_call_attempts"
            ],
            tool_call_attempts=payload["tool_call_attempts"],
            answer_reward_total=payload["answer_reward_total"],
            format_errors=payload["format_errors"],
            conditional_tool_reward_total=payload["conditional_tool_reward_total"],
            judge_calls=payload["judge_calls"],
            judge_prompt_tokens=payload["judge_prompt_tokens"],
            judge_completion_tokens=payload["judge_completion_tokens"],
            judge_cost_usd=payload["judge_cost_usd"],
            reasoning_tokens=payload["reasoning_tokens"],
            original_visual_tokens=payload["original_visual_tokens"],
            total_visual_tokens=payload["total_visual_tokens"],
            step_time_seconds_total=payload["step_time_seconds_total"],
            tool_error_counts=tuple(errors),
        )


@dataclass(frozen=True, slots=True)
class PilotMetricsSummary:
    """Published counts and zero-safe means/rates for one run prefix."""

    optimizer_steps: int
    prompts: int
    trajectories: int
    generated_policy_tokens: int
    successful_tgvf_observations: int
    tool_call_attempt_rate: float
    mean_tool_call_attempts: float
    mean_answer_reward: float
    format_error_rate: float
    mean_conditional_tool_reward: float
    mean_reasoning_length: float
    mean_original_visual_tokens: float
    mean_total_visual_tokens: float
    mean_step_time_seconds: float
    judge_calls: int
    judge_prompt_tokens: int
    judge_completion_tokens: int
    judge_cost_usd: float
    tool_error_counts: tuple[ToolErrorCount, ...]


class PilotMetricsAccumulator:
    """Order-stable additive reducer with atomic checkpoint restoration."""

    def __init__(self) -> None:
        self._state = PilotMetricsCheckpointState()

    @property
    def state(self) -> PilotMetricsCheckpointState:
        return self._state

    def record_optimizer_step(
        self, observation: PilotOptimizerStepMetricsObservation
    ) -> PilotMetricsSummary:
        if not isinstance(observation, PilotOptimizerStepMetricsObservation):
            raise TypeError("observation must be PilotOptimizerStepMetricsObservation")
        expected_step = self._state.optimizer_steps + 1
        if observation.optimizer_step != expected_step:
            raise ValueError(
                "optimizer steps must be contiguous across metric restore: "
                f"expected {expected_step}, got {observation.optimizer_step}"
            )
        rows = observation.trajectories
        if self._state.prompts:
            existing_group_size = self._state.trajectories // self._state.prompts
            if observation.trajectories_per_prompt != existing_group_size:
                raise ValueError(
                    "metrics cannot mix trajectories-per-prompt group sizes: "
                    f"existing={existing_group_size} observed="
                    f"{observation.trajectories_per_prompt}"
                )
        errors = Counter(
            {item.code: item.count for item in self._state.tool_error_counts}
        )
        errors.update(code for row in rows for code in row.tool_error_codes)
        self._state = PilotMetricsCheckpointState(
            optimizer_steps=expected_step,
            prompts=self._state.prompts + observation.prompt_count,
            trajectories=self._state.trajectories + len(rows),
            generated_policy_tokens=(
                self._state.generated_policy_tokens
                + sum(row.generated_policy_tokens for row in rows)
            ),
            successful_tgvf_observations=(
                self._state.successful_tgvf_observations
                + sum(row.successful_tgvf_observations for row in rows)
            ),
            trajectories_with_tool_call_attempts=(
                self._state.trajectories_with_tool_call_attempts
                + sum(row.tool_call_attempts > 0 for row in rows)
            ),
            tool_call_attempts=(
                self._state.tool_call_attempts
                + sum(row.tool_call_attempts for row in rows)
            ),
            answer_reward_total=(
                self._state.answer_reward_total
                + sum(int(row.answer_reward) for row in rows)
            ),
            format_errors=(
                self._state.format_errors + sum(row.format_error for row in rows)
            ),
            conditional_tool_reward_total=(
                self._state.conditional_tool_reward_total
                + sum(int(row.conditional_tool_reward) for row in rows)
            ),
            judge_calls=(
                self._state.judge_calls + sum(row.judge_calls for row in rows)
            ),
            judge_prompt_tokens=(
                self._state.judge_prompt_tokens
                + sum(row.judge_prompt_tokens for row in rows)
            ),
            judge_completion_tokens=(
                self._state.judge_completion_tokens
                + sum(row.judge_completion_tokens for row in rows)
            ),
            judge_cost_usd=math.fsum(
                (self._state.judge_cost_usd, *(row.judge_cost_usd for row in rows))
            ),
            reasoning_tokens=(
                self._state.reasoning_tokens + sum(row.reasoning_tokens for row in rows)
            ),
            original_visual_tokens=(
                self._state.original_visual_tokens
                + sum(row.original_visual_tokens for row in rows)
            ),
            total_visual_tokens=(
                self._state.total_visual_tokens
                + sum(row.total_visual_tokens for row in rows)
            ),
            step_time_seconds_total=math.fsum(
                (self._state.step_time_seconds_total, observation.step_time_seconds)
            ),
            tool_error_counts=tuple(
                ToolErrorCount(code, errors[code]) for code in sorted(errors)
            ),
        )
        return self.summary()

    def summary(self) -> PilotMetricsSummary:
        """Return zero, never NaN, for every mean/rate on an empty state."""

        state = self._state
        return PilotMetricsSummary(
            optimizer_steps=state.optimizer_steps,
            prompts=state.prompts,
            trajectories=state.trajectories,
            generated_policy_tokens=state.generated_policy_tokens,
            successful_tgvf_observations=state.successful_tgvf_observations,
            tool_call_attempt_rate=_zero_safe_mean(
                state.trajectories_with_tool_call_attempts, state.trajectories
            ),
            mean_tool_call_attempts=_zero_safe_mean(
                state.tool_call_attempts, state.trajectories
            ),
            mean_answer_reward=_zero_safe_mean(
                state.answer_reward_total, state.trajectories
            ),
            format_error_rate=_zero_safe_mean(state.format_errors, state.trajectories),
            mean_conditional_tool_reward=_zero_safe_mean(
                state.conditional_tool_reward_total, state.trajectories
            ),
            mean_reasoning_length=_zero_safe_mean(
                state.reasoning_tokens, state.trajectories
            ),
            mean_original_visual_tokens=_zero_safe_mean(
                state.original_visual_tokens, state.trajectories
            ),
            mean_total_visual_tokens=_zero_safe_mean(
                state.total_visual_tokens, state.trajectories
            ),
            mean_step_time_seconds=_zero_safe_mean(
                state.step_time_seconds_total, state.optimizer_steps
            ),
            judge_calls=state.judge_calls,
            judge_prompt_tokens=state.judge_prompt_tokens,
            judge_completion_tokens=state.judge_completion_tokens,
            judge_cost_usd=state.judge_cost_usd,
            tool_error_counts=state.tool_error_counts,
        )

    def checkpoint_state(self) -> dict[str, object]:
        return self._state.to_checkpoint_mapping()

    def restore_checkpoint_state(self, payload: object) -> None:
        """Validate fully before replacing state, so a failed restore is atomic."""

        restored = _coerce_checkpoint_state(payload)
        self._state = restored

    @classmethod
    def from_checkpoint_state(cls, payload: object) -> "PilotMetricsAccumulator":
        accumulator = cls()
        accumulator.restore_checkpoint_state(payload)
        return accumulator


def _coerce_checkpoint_state(payload: object) -> PilotMetricsCheckpointState:
    if isinstance(payload, PilotMetricsCheckpointState):
        return payload
    if isinstance(payload, Mapping):
        return PilotMetricsCheckpointState.from_checkpoint_mapping(payload)
    raise TypeError("metrics checkpoint must be a state object or mapping")


def _zero_safe_mean(numerator: int | float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / denominator


def _nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _positive_int(value: object, name: str) -> None:
    _nonnegative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _binary_reward(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result not in {0.0, 1.0}:
        raise ValueError(f"{name} must be exactly 0 or 1")
    return result


__all__ = [
    "POLICY_PILOT_V1_ADMITTED_TOOL_ATTEMPTS",
    "POLICY_PILOT_V1_MAX_RECORDED_TOOL_ATTEMPTS",
    "POLICY_PILOT_V1_METRICS_SCHEMA",
    "POLICY_PILOT_V1_TRAJECTORIES_PER_PROMPT",
    "POLICY_SUPPORTED_TRAJECTORIES_PER_PROMPT",
    "PILOT_V1_METRIC_REDUCTIONS",
    "MetricReductionContract",
    "PilotMetricsAccumulator",
    "PilotMetricsCheckpointState",
    "PilotMetricsSummary",
    "PilotOptimizerStepMetricsObservation",
    "PilotTrajectoryMetricsObservation",
    "ToolErrorCount",
]
