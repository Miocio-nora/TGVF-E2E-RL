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


def _step_n16(
    optimizer_step: int, prompt_id: str, elapsed: float
) -> PilotOptimizerStepMetricsObservation:
    rows = tuple(
        replace(
            row,
            trajectory_id=f"{prompt_id}/trajectory-{copy_index * 8 + row_index}",
        )
        for copy_index in range(2)
        for row_index, row in enumerate(_rows(prompt_id))
    )
    return PilotOptimizerStepMetricsObservation(
        optimizer_step=optimizer_step,
        step_time_seconds=elapsed,
        trajectories=rows,
        trajectories_per_prompt=16,
    )


def _step_n2(
    optimizer_step: int, prompt_id: str, elapsed: float
) -> PilotOptimizerStepMetricsObservation:
    rows = tuple(
        replace(
            row,
            trajectory_id=f"{prompt_id}/trajectory-{row_index}",
        )
        for row_index, row in enumerate(_rows(prompt_id)[:2])
    )
    return PilotOptimizerStepMetricsObservation(
        optimizer_step=optimizer_step,
        step_time_seconds=elapsed,
        trajectories=rows,
        trajectories_per_prompt=2,
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


def test_n16_metrics_round_trip_and_reject_mixed_group_sizes() -> None:
    first = _step_n16(1, "prompt-a", 20.0)
    second = _step_n16(2, "prompt-b", 22.0)
    accumulator = PilotMetricsAccumulator()
    summary = accumulator.record_optimizer_step(first)
    assert summary.prompts == 1
    assert summary.trajectories == 16

    payload = json.loads(json.dumps(accumulator.checkpoint_state()))
    restored = PilotMetricsAccumulator.from_checkpoint_state(payload)
    restored.record_optimizer_step(second)
    assert restored.summary().prompts == 2
    assert restored.summary().trajectories == 32

    with pytest.raises(ValueError, match="cannot mix"):
        restored.record_optimizer_step(_step(3, "prompt-c", 1.0))

    with pytest.raises(ValueError, match="zero-prompt"):
        PilotMetricsCheckpointState(prompts=0, trajectories=16)


def test_n2_functional_canary_metrics_round_trip() -> None:
    first = _step_n2(1, "prompt-a", 2.0)
    accumulator = PilotMetricsAccumulator()
    summary = accumulator.record_optimizer_step(first)

    assert summary.prompts == 1
    assert summary.trajectories == 2
    restored = PilotMetricsAccumulator.from_checkpoint_state(
        json.loads(json.dumps(accumulator.checkpoint_state()))
    )
    assert restored.summary() == summary


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
    with pytest.raises(ValueError, match="protocol-bound final attempt"):
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
    with pytest.raises(ValueError, match="supported group sizes"):
        accumulator.restore_checkpoint_state(corrupt)
    assert accumulator.state == before

    missing = deepcopy(accumulator.checkpoint_state())
    del missing["format_errors"]
    with pytest.raises(ValueError, match="fields differ"):
        PilotMetricsCheckpointState.from_checkpoint_mapping(missing)
    assert accumulator.state == before


def test_metrics_preserve_legal_six_call_protocol_counts_without_clamping() -> None:
    """PRL15 may legally execute six TGVF calls before its cap attempt."""

    base = _rows("prl15-step4-prompt")[0]
    six_call_rows = tuple(
        replace(
            base,
            trajectory_id=f"prl15-step4-trajectory-{index}",
            successful_tgvf_observations=6,
            tool_call_attempts=6,
            maximum_tool_calls=6,
        )
        for index in range(8)
    )
    observation = PilotOptimizerStepMetricsObservation(
        1,
        1.0,
        six_call_rows,
    )
    accumulator = PilotMetricsAccumulator(maximum_tool_calls=6)
    summary = accumulator.record_optimizer_step(observation)

    assert summary.successful_tgvf_observations == 48
    assert summary.mean_tool_call_attempts == 6.0

    capped = replace(
        six_call_rows[0],
        tool_call_attempts=7,
        tool_error_codes=(ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,),
    )
    assert capped.tool_call_attempts == 7
    assert capped.successful_tgvf_observations == 6
    assert capped.tool_error_codes == (
        ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,
    )

    with pytest.raises(ValueError, match="tool-call attempts exceed"):
        replace(
            capped,
            tool_call_attempts=8,
            tool_error_codes=(
                ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,
                ToolErrorCode.TOOL_EXECUTION_FAILED.value,
            ),
        )
    with pytest.raises(ValueError, match="successful observations exceed"):
        replace(
            capped,
            successful_tgvf_observations=7,
            tool_error_codes=(),
        )
    with pytest.raises(ValueError, match="protocol-bound final attempt"):
        replace(
            capped,
            tool_error_codes=(ToolErrorCode.TOOL_EXECUTION_FAILED.value,),
        )

    with pytest.raises(ValueError, match="differs from the accumulator"):
        PilotMetricsAccumulator().record_optimizer_step(observation)
    with pytest.raises(ValueError, match="configured tool-call bound"):
        PilotMetricsAccumulator.from_checkpoint_state(
            accumulator.checkpoint_state(),
            maximum_tool_calls=4,
        )


def test_prl15_step3_v1_metrics_payload_resumes_under_six_call_contract() -> None:
    """The real pre-failure step-3 payload stays byte-schema compatible."""

    payload = {
        "schema_version": "policy-pilot-v1-metrics-v1",
        "optimizer_steps": 3,
        "prompts": 48,
        "trajectories": 768,
        "generated_policy_tokens": 186707,
        "successful_tgvf_observations": 593,
        "trajectories_with_tool_call_attempts": 592,
        "tool_call_attempts": 593,
        "answer_reward_total": 516,
        "format_errors": 20,
        "conditional_tool_reward_total": 395,
        "judge_calls": 768,
        "judge_prompt_tokens": 390028,
        "judge_completion_tokens": 1344,
        "judge_cost_usd": 0.14094768,
        "reasoning_tokens": 0,
        "original_visual_tokens": 347904,
        "total_visual_tokens": 666021,
        "step_time_seconds_total": 2638.7188123851083,
        "tool_error_counts": [],
    }
    accumulator = PilotMetricsAccumulator.from_checkpoint_state(
        payload,
        maximum_tool_calls=6,
    )
    assert accumulator.checkpoint_state() == payload

    base = _rows("prl15-step4-resume-prompt")[0]
    rows = tuple(
        replace(
            base,
            trajectory_id=f"prl15-step4-resume-trajectory-{index}",
            successful_tgvf_observations=6,
            tool_call_attempts=6,
            maximum_tool_calls=6,
        )
        for index in range(16)
    )
    summary = accumulator.record_optimizer_step(
        PilotOptimizerStepMetricsObservation(
            4,
            1.0,
            rows,
            trajectories_per_prompt=16,
        )
    )
    assert summary.optimizer_steps == 4
    assert summary.successful_tgvf_observations == 689


def test_accumulator_rejects_impossible_aggregate_cap_position() -> None:
    state = PilotMetricsCheckpointState(
        optimizer_steps=1,
        prompts=1,
        trajectories=8,
        trajectories_with_tool_call_attempts=1,
        tool_call_attempts=1,
        step_time_seconds_total=1.0,
        tool_error_counts=(
            ToolErrorCount(ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value, 1),
        ),
    )

    with pytest.raises(ValueError, match="before all admitted attempts"):
        PilotMetricsAccumulator().restore_checkpoint_state(state)
