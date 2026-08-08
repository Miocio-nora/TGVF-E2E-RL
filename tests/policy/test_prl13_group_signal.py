from __future__ import annotations

import math

import pytest

from tgvf_rl.policy.prl13_group_signal import (
    parse_signal_row,
    summarize_group_signal,
)


def _row(
    group: str,
    *,
    score: float,
    correct: bool,
    calls: int,
    successful: int,
    source: str = "vstar",
) -> dict[str, object]:
    return {
        "step": 1,
        "group_id": group,
        "source": source,
        "score": score,
        "acc": int(correct),
        "crop_call_count": calls,
        "successful_crop_count": successful,
    }


def test_distinguishes_answer_signal_from_conditional_bonus_signal() -> None:
    records = [
        # Both answers are correct; only the successful Crop bonus differs.
        _row("bonus", score=0.8, correct=True, calls=0, successful=0),
        _row("bonus", score=2.0, correct=True, calls=1, successful=1),
        # This group contains the desired sampled answer-outcome contrast.
        _row("benefit", score=0.0, correct=False, calls=0, successful=0),
        _row("benefit", score=2.0, correct=True, calls=1, successful=1),
        # Crop correlates with harm for this prompt.
        _row("harm", score=0.8, correct=True, calls=0, successful=0),
        _row("harm", score=0.0, correct=False, calls=1, successful=1),
        # No GRPO signal at all.
        _row("zero", score=0.0, correct=False, calls=0, successful=0),
        _row("zero", score=0.0, correct=False, calls=1, successful=1),
    ]

    report = summarize_group_signal(records, include_groups=True)
    signal = report["group_signal"]
    assert report["groups"] == 4
    assert signal["nonzero_reward"] == 3
    assert signal["answer_contrast"] == 2
    assert signal["bonus_only_contrast"] == 1
    assert signal["answer_benefit_candidate"] == 1
    assert signal["answer_harm_candidate"] == 1
    assert signal["correct_tool_and_direct"] == 1
    assert signal["rates"]["answer_contrast"] == 0.5
    assert report["routes"]["strict_direct"]["accuracy"] == 0.5
    assert report["routes"]["successful_crop"]["accuracy"] == 0.5


def test_sources_receive_independent_denominators() -> None:
    report = summarize_group_signal(
        [
            _row("visual", score=0.0, correct=False, calls=0, successful=0),
            _row("visual", score=2.0, correct=True, calls=1, successful=1),
            _row(
                "math",
                score=0.0,
                correct=False,
                calls=0,
                successful=0,
                source="thinklite",
            ),
            _row(
                "math",
                score=-0.4,
                correct=False,
                calls=0,
                successful=0,
                source="thinklite",
            ),
        ]
    )
    assert report["sources"]["vstar"]["rates"]["answer_contrast"] == 1.0
    assert report["sources"]["thinklite"]["rates"]["answer_contrast"] == 0.0
    assert math.isnan(
        report["routes"]["successful_crop"]["accuracy"]
    ) is False


def test_rejects_internally_impossible_crop_counts() -> None:
    record = _row("bad", score=0.0, correct=False, calls=0, successful=1)
    with pytest.raises(ValueError, match="exceeds"):
        parse_signal_row(record)


def test_group_cannot_mix_sources() -> None:
    rows = [
        _row("mixed", score=0.0, correct=False, calls=0, successful=0),
        _row(
            "mixed",
            score=0.0,
            correct=False,
            calls=0,
            successful=0,
            source="arxivqa",
        ),
    ]
    with pytest.raises(ValueError, match="mixes sources"):
        summarize_group_signal(rows)
