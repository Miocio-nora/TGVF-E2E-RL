"""Post-hoc GRPO group-signal diagnostics for PRL13 trajectories.

This module is deliberately outside the training path.  It distinguishes
answer-outcome contrasts from reward variation caused only by the conditional
Crop bonus, which is necessary before treating a non-zero GRPO advantage as
evidence that Crop improved an answer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class SignalRow:
    step: int
    group_id: str
    source: str
    score: float
    correct: bool
    tool_attempted: bool
    successful_crop: bool


def parse_signal_row(record: Mapping[str, Any]) -> SignalRow:
    """Extract the minimal, stable fields needed for group diagnostics."""

    step = record.get("step")
    group_id = record.get("group_id")
    source = record.get("source")
    score = record.get("score")
    acc = record.get("acc")
    crop_call_count = record.get("crop_call_count")
    successful_crop_count = record.get("successful_crop_count")

    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be a non-empty string")
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("score must be numeric")
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    if acc not in (0, 1, False, True):
        raise ValueError("acc must be binary")
    if type(crop_call_count) is not int or crop_call_count < 0:
        raise ValueError("crop_call_count must be a non-negative integer")
    if type(successful_crop_count) is not int or successful_crop_count < 0:
        raise ValueError("successful_crop_count must be a non-negative integer")
    if successful_crop_count > crop_call_count:
        raise ValueError("successful_crop_count exceeds crop_call_count")

    return SignalRow(
        step=step,
        group_id=group_id,
        source=source,
        score=score,
        correct=bool(acc),
        tool_attempted=crop_call_count > 0,
        successful_crop=successful_crop_count > 0,
    )


def _rate(count: int, denominator: int) -> float:
    return count / denominator if denominator else math.nan


def _route_summary(rows: Sequence[SignalRow]) -> dict[str, float | int]:
    correct = sum(row.correct for row in rows)
    return {
        "trajectories": len(rows),
        "correct": correct,
        "accuracy": _rate(correct, len(rows)),
    }


def summarize_group_signal(
    records: Iterable[Mapping[str, Any]],
    *,
    include_groups: bool = False,
) -> dict[str, Any]:
    """Summarize which GRPO groups carry answer-relevant learning signal.

    ``answer_benefit_candidate`` is observational rather than causal: it means
    the same prompt produced at least one correct successful-Crop rollout and
    at least one incorrect strict-direct rollout.  The sampled reasoning also
    differs, so only a controlled counterfactual evaluation can establish
    causality.
    """

    rows = [parse_signal_row(record) for record in records]
    grouped: dict[tuple[int, str], list[SignalRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.step, row.group_id)].append(row)

    counters: Counter[str] = Counter()
    source_counters: dict[str, Counter[str]] = defaultdict(Counter)
    details: list[dict[str, Any]] = []

    for (step, group_id), group in sorted(grouped.items()):
        sources = {row.source for row in group}
        if len(sources) != 1:
            raise ValueError(f"group {group_id!r} mixes sources: {sorted(sources)}")
        source = next(iter(sources))
        scores = {row.score for row in group}
        answers = {row.correct for row in group}
        successful_routes = {row.successful_crop for row in group}
        attempted_routes = {row.tool_attempted for row in group}

        reward_varied = len(scores) > 1
        answer_varied = len(answers) > 1
        successful_tool_varied = len(successful_routes) > 1
        attempted_tool_varied = len(attempted_routes) > 1

        correct_tool = any(row.correct and row.successful_crop for row in group)
        wrong_tool = any(not row.correct and row.successful_crop for row in group)
        correct_direct = any(row.correct and not row.tool_attempted for row in group)
        wrong_direct = any(not row.correct and not row.tool_attempted for row in group)

        flags = {
            "nonzero_reward": reward_varied,
            "zero_reward": not reward_varied,
            "answer_contrast": answer_varied,
            "successful_tool_contrast": successful_tool_varied,
            "attempted_tool_contrast": attempted_tool_varied,
            "bonus_only_contrast": (
                reward_varied and not answer_varied and successful_tool_varied
            ),
            "nonanswer_reward_contrast": reward_varied and not answer_varied,
            "answer_benefit_candidate": correct_tool and wrong_direct,
            "answer_harm_candidate": wrong_tool and correct_direct,
            "correct_tool_and_direct": correct_tool and correct_direct,
            "all_correct": all(row.correct for row in group),
            "all_wrong": all(not row.correct for row in group),
        }
        counters["groups"] += 1
        source_counters[source]["groups"] += 1
        for name, enabled in flags.items():
            if enabled:
                counters[name] += 1
                source_counters[source][name] += 1

        if include_groups:
            details.append(
                {
                    "step": step,
                    "group_id": group_id,
                    "source": source,
                    "size": len(group),
                    "scores": sorted(scores),
                    "correct_count": sum(row.correct for row in group),
                    "strict_direct_count": sum(
                        not row.tool_attempted for row in group
                    ),
                    "successful_crop_count": sum(
                        row.successful_crop for row in group
                    ),
                    **flags,
                }
            )

    direct = [row for row in rows if not row.tool_attempted]
    attempted = [row for row in rows if row.tool_attempted]
    successful = [row for row in rows if row.successful_crop]
    groups = counters["groups"]
    group_sizes = Counter(len(group) for group in grouped.values())

    def _counter_record(counter: Counter[str]) -> dict[str, Any]:
        result: dict[str, Any] = dict(sorted(counter.items()))
        denominator = counter["groups"]
        result["rates"] = {
            name: _rate(counter[name], denominator)
            for name in (
                "nonzero_reward",
                "answer_contrast",
                "successful_tool_contrast",
                "bonus_only_contrast",
                "answer_benefit_candidate",
                "answer_harm_candidate",
                "correct_tool_and_direct",
            )
        }
        return result

    result: dict[str, Any] = {
        "schema_version": "prl13-posthoc-group-signal-v1",
        "interpretation": (
            "answer_benefit/harm are sampled within-prompt associations, not "
            "controlled causal effects"
        ),
        "trajectories": len(rows),
        "groups": groups,
        "group_size_histogram": {
            str(size): count for size, count in sorted(group_sizes.items())
        },
        "group_signal": _counter_record(counters),
        "routes": {
            "strict_direct": _route_summary(direct),
            "tool_attempted": _route_summary(attempted),
            "successful_crop": _route_summary(successful),
        },
        "sources": {
            source: _counter_record(counter)
            for source, counter in sorted(source_counters.items())
        },
    }
    if include_groups:
        result["group_details"] = details
    return result


__all__ = ["SignalRow", "parse_signal_row", "summarize_group_signal"]
