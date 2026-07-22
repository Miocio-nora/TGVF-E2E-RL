from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from tgvf_rl.policy.metrics import (
    PILOT_V1_METRIC_REDUCTIONS,
    PilotMetricsAccumulator,
    PilotMetricsCheckpointState,
    PilotOptimizerStepMetricsObservation,
    PilotTrajectoryMetricsObservation,
    ToolErrorCount,
)
from tgvf_rl.protocol.schema import ToolErrorCode


def _rows(prompt_id: str) -> tuple[PilotTrajectoryMetricsObservation, ...]:
    attempts = (1, 2, 3, 4, 5, 0, 0, 0)
    successes = (1, 1, 2, 4, 4, 0, 0, 0)
    errors = (
        (),
        (ToolErrorCode.TOOL_EXECUTION_FAILED.value,),
        (ToolErrorCode.TOOL_RESPONSE_APPEND_FAILED.value,),
        (),
        (ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,),
        (),
        (),
        (),
    )
    answer_rewards = (1, 0, 1, 0, 1, 0, 1, 0)
    format_errors = (False, True, False, False, False, True, False, False)
    conditional_rewards = (1, 0, 1, 0, 1, 0, 0, 0)
    rows = []
    for index in range(8):
        reasoning = index + 2
        original_visual = index + 10
        rows.append(
            PilotTrajectoryMetricsObservation(
                prompt_id=prompt_id,
                trajectory_id=f"{prompt_id}/trajectory-{index}",
                generated_policy_tokens=reasoning + 3,
                successful_tgvf_observations=successes[index],
                tool_call_attempts=attempts[index],
                answer_reward=answer_rewards[index],
                format_error=format_errors[index],
                conditional_tool_reward=conditional_rewards[index],
                reasoning_tokens=reasoning,
                original_visual_tokens=original_visual,
                total_visual_tokens=original_visual + 2 * successes[index],
                tool_error_codes=errors[index],
                judge_calls=int(index == 0),
                judge_prompt_tokens=201 if index == 0 else 0,
                judge_completion_tokens=17 if index == 0 else 0,
                judge_cost_usd=0.00007916 if index == 0 else 0.0,
            )
        )
    return tuple(rows)


def _step(
    optimizer_step: int, prompt_id: str, elapsed: float
) -> PilotOptimizerStepMetricsObservation:
    return PilotOptimizerStepMetricsObservation(
        optimizer_step=optimizer_step,
        step_time_seconds=elapsed,
        trajectories=_rows(prompt_id),
    )


def test_empty_summary_has_explicit_zero_denominator_contract() -> None:
    contracts = {item.metric: item for item in PILOT_V1_METRIC_REDUCTIONS}
    assert contracts["tool_call_attempt_rate"].denominator == "trajectories"
    assert contracts["mean_tool_call_attempts"].denominator == "trajectories"
    assert contracts["mean_answer_reward"].denominator == "trajectories"
    assert contracts["format_error_rate"].denominator == "trajectories"
    assert contracts["mean_conditional_tool_reward"].denominator == "trajectories"
    assert contracts["mean_reasoning_length"].denominator == "trajectories"
    assert contracts["mean_original_visual_tokens"].denominator == "trajectories"
    assert contracts["mean_total_visual_tokens"].denominator == "trajectories"
    assert contracts["mean_step_time_seconds"].denominator == "optimizer_steps"
    assert {item.zero_denominator_value for item in contracts.values()} == {0.0}

    summary = PilotMetricsAccumulator().summary()
    assert summary.optimizer_steps == 0
    assert summary.prompts == 0
    assert summary.trajectories == 0
    for metric in contracts:
        assert getattr(summary, metric) == 0.0
    assert summary.tool_error_counts == ()


def test_pilot_metrics_use_declared_trajectory_and_step_denominators() -> None:
    accumulator = PilotMetricsAccumulator()
    step = _step(1, "prompt-a", 12.5)
    summary = accumulator.record_optimizer_step(step)

    assert summary.optimizer_steps == 1
    assert summary.prompts == 1
    assert summary.trajectories == 8
    assert summary.generated_policy_tokens == 68
    assert summary.successful_tgvf_observations == 12
    assert summary.tool_call_attempt_rate == pytest.approx(5 / 8)
    assert summary.mean_tool_call_attempts == pytest.approx(15 / 8)
    assert summary.mean_answer_reward == 0.5
    assert summary.format_error_rate == 0.25
    assert summary.mean_conditional_tool_reward == pytest.approx(3 / 8)
    assert summary.mean_reasoning_length == 5.5
    assert summary.mean_original_visual_tokens == 13.5
    assert summary.mean_total_visual_tokens == 16.5
    assert summary.judge_calls == 1
    assert summary.judge_prompt_tokens == 201
    assert summary.judge_completion_tokens == 17
    assert summary.judge_cost_usd == pytest.approx(0.00007916)
    assert summary.mean_step_time_seconds == 12.5
    assert summary.tool_error_counts == (
        ToolErrorCount(ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value, 1),
        ToolErrorCount(ToolErrorCode.TOOL_EXECUTION_FAILED.value, 1),
        ToolErrorCount(ToolErrorCode.TOOL_RESPONSE_APPEND_FAILED.value, 1),
    )

    reordered = PilotMetricsAccumulator()
    reordered.record_optimizer_step(replace(step, trajectories=step.trajectories[::-1]))
    assert reordered.state == accumulator.state
    assert reordered.summary() == summary


def test_checkpoint_restore_is_lossless_and_continues_at_exact_next_step() -> None:
    first = _step(1, "prompt-a", 12.5)
    second = _step(2, "prompt-b", 7.25)

    uninterrupted = PilotMetricsAccumulator()
    uninterrupted.record_optimizer_step(first)
    uninterrupted.record_optimizer_step(second)

    staged = PilotMetricsAccumulator()
    staged.record_optimizer_step(first)
    payload = json.loads(json.dumps(staged.checkpoint_state(), sort_keys=True))
    restored = PilotMetricsAccumulator.from_checkpoint_state(payload)
    assert restored.checkpoint_state() == payload
    restored.record_optimizer_step(second)

    assert restored.state == uninterrupted.state
    assert restored.summary() == uninterrupted.summary()
    assert restored.summary().optimizer_steps == 2
    assert restored.summary().prompts == 2
    assert restored.summary().trajectories == 16
    assert restored.summary().mean_step_time_seconds == pytest.approx(9.875)
    assert tuple(item.count for item in restored.summary().tool_error_counts) == (
        2,
        2,
        2,
    )

    payload["optimizer_steps"] = 999
    assert restored.state == uninterrupted.state


def test_observation_step_and_atomic_restore_validation_fail_closed() -> None:
    valid = _rows("prompt-a")[0]
    with pytest.raises(ValueError, match="conditional tool reward"):
        replace(
            valid,
            successful_tgvf_observations=0,
            tool_call_attempts=0,
            conditional_tool_reward=1,
        )
    with pytest.raises(ValueError, match="each tool-call attempt"):
        replace(valid, tool_call_attempts=2)
    with pytest.raises(ValueError, match="total visual tokens"):
        replace(valid, total_visual_tokens=valid.original_visual_tokens - 1)
    with pytest.raises(ValueError, match="fifth Pilot attempt"):
        replace(
            valid,
            successful_tgvf_observations=4,
            tool_call_attempts=5,
            tool_error_codes=(ToolErrorCode.TOOL_EXECUTION_FAILED.value,),
        )

    with pytest.raises(ValueError, match="exactly eight"):
        PilotOptimizerStepMetricsObservation(1, 1.0, _rows("prompt-a")[:-1])

    accumulator = PilotMetricsAccumulator()
    accumulator.record_optimizer_step(_step(1, "prompt-a", 1.0))
    before = accumulator.state
    with pytest.raises(ValueError, match="contiguous"):
        accumulator.record_optimizer_step(_step(3, "prompt-b", 1.0))
    assert accumulator.state == before

    corrupt = deepcopy(accumulator.checkpoint_state())
    corrupt["trajectories"] = 7
    with pytest.raises(ValueError, match="multiplied by 8"):
        accumulator.restore_checkpoint_state(corrupt)
    assert accumulator.state == before

    missing = deepcopy(accumulator.checkpoint_state())
    del missing["format_errors"]
    with pytest.raises(ValueError, match="fields differ"):
        PilotMetricsCheckpointState.from_checkpoint_mapping(missing)
    assert accumulator.state == before
