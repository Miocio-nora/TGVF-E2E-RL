"""Isolated, deterministic ``stage3-shaped-v1`` reward semantics.

This module deliberately does not depend on, or alter, the legacy Pilot reward
pipeline.  It turns already-verified answer/tool/judge facts into one scalar and
keeps an auditable decomposition of every contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import math


STAGE3_SHAPED_REWARD_VERSION = "stage3-shaped-v1"


class ToolNecessityLabel(str, Enum):
    """Counterfactual label controlling the tool-decision reward."""

    NEEDED = "needed"
    OPTIONAL = "optional"
    UNNECESSARY = "unnecessary"


class QualityJudgeScore(IntEnum):
    """Raw three-level score emitted by the focus/grounding judge."""

    FAIL = 0
    PARTIAL = 1
    PASS = 2


class Stage3ShapedComponentName(str, Enum):
    ANSWER = "answer"
    TOOL = "tool"
    FOCUS = "focus"
    GROUNDING = "grounding"
    PROTOCOL = "protocol"


@dataclass(frozen=True, slots=True)
class Stage3ShapedRewardFacts:
    """Strict, side-effect-free inputs to ``stage3-shaped-v1``.

    ``tool_call_count`` records attempts and controls only the repeated-call
    term. ``successful_tgvf_observation_count`` records calls that actually
    triggered the tool and produced a TGVF observation; matching old Stage3's
    ``native_result.triggered``, it controls both the tool-choice reward and the
    answer gate. A successful observation must have both quality-judge scores;
    without an observation, those scores must be absent because there is no D to
    judge.
    """

    answer_correct: bool
    tool_label: ToolNecessityLabel
    tool_call_count: int = 0
    successful_tgvf_observation_count: int = 0
    focus_score: QualityJudgeScore | None = None
    grounding_score: QualityJudgeScore | None = None
    quality_judge_failure: str | None = None
    label_confidence: float = 0.5
    protocol_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.answer_correct) is not bool:
            raise TypeError("answer_correct must be bool")
        if type(self.tool_label) is not ToolNecessityLabel:
            raise TypeError("tool_label must be ToolNecessityLabel")
        for field_name, value in (
            ("tool_call_count", self.tool_call_count),
            (
                "successful_tgvf_observation_count",
                self.successful_tgvf_observation_count,
            ),
        ):
            if type(value) is not int:
                raise TypeError(f"{field_name} must be int")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.successful_tgvf_observation_count > self.tool_call_count:
            raise ValueError(
                "successful TGVF observations cannot exceed attempted tool calls"
            )

        if type(self.label_confidence) is not float:
            raise TypeError("label_confidence must be float")
        if not math.isfinite(self.label_confidence):
            raise ValueError("label_confidence must be finite")
        if not 0.0 <= self.label_confidence <= 1.0:
            raise ValueError("label_confidence must be within [0, 1]")

        for field_name, score in (
            ("focus_score", self.focus_score),
            ("grounding_score", self.grounding_score),
        ):
            if score is not None and type(score) is not QualityJudgeScore:
                raise TypeError(f"{field_name} must be QualityJudgeScore or None")
        has_observation = self.successful_tgvf_observation_count >= 1
        has_both_scores = (
            self.focus_score is not None and self.grounding_score is not None
        )
        has_any_score = self.focus_score is not None or self.grounding_score is not None
        if self.quality_judge_failure is not None and (
            type(self.quality_judge_failure) is not str
            or not self.quality_judge_failure.strip()
            or self.quality_judge_failure != self.quality_judge_failure.strip()
        ):
            raise ValueError(
                "quality_judge_failure must be non-empty stripped text when present"
            )
        judge_failed = self.quality_judge_failure is not None
        if has_observation and has_both_scores == judge_failed:
            raise ValueError(
                "successful TGVF observations require focus and grounding scores "
                "or one explicit sample-local judge failure"
            )
        if not has_observation and (has_any_score or judge_failed):
            raise ValueError(
                "quality scores/failure require a successful TGVF observation"
            )

        if type(self.protocol_errors) is not tuple:
            raise TypeError("protocol_errors must be tuple[str, ...]")
        for error in self.protocol_errors:
            if type(error) is not str:
                raise TypeError("protocol_errors must contain only str values")
            if not error or error != error.strip():
                raise ValueError(
                    "protocol error codes must be non-empty stripped strings"
                )
        if len(self.protocol_errors) != len(set(self.protocol_errors)):
            raise ValueError("protocol error codes must be unique")


@dataclass(frozen=True, slots=True)
class Stage3ShapedRewardComponent:
    """One exact scalar contribution and its deterministic audit evidence."""

    name: Stage3ShapedComponentName
    score: float
    evidence: str

    def __post_init__(self) -> None:
        if type(self.name) is not Stage3ShapedComponentName:
            raise TypeError("component name must be Stage3ShapedComponentName")
        if type(self.score) is not float:
            raise TypeError("component score must be float")
        if not math.isfinite(self.score):
            raise ValueError("component score must be finite")
        if type(self.evidence) is not str:
            raise TypeError("component evidence must be str")
        if not self.evidence.strip():
            raise ValueError("component evidence must be non-empty")


@dataclass(frozen=True, slots=True)
class Stage3ShapedRewardResult:
    """Versioned scalar reward with a complete five-part decomposition."""

    total: float
    components: tuple[Stage3ShapedRewardComponent, ...]
    answer_gated: bool
    quality_judge_applicable: bool
    quality_judge_covered: bool
    quality_judge_failure: str | None = None
    version: str = STAGE3_SHAPED_REWARD_VERSION

    def __post_init__(self) -> None:
        if type(self.total) is not float:
            raise TypeError("reward total must be float")
        if not math.isfinite(self.total):
            raise ValueError("reward total must be finite")
        if type(self.components) is not tuple:
            raise TypeError("reward components must be tuple")
        if any(
            type(component) is not Stage3ShapedRewardComponent
            for component in self.components
        ):
            raise TypeError(
                "reward components must contain Stage3ShapedRewardComponent values"
            )
        expected_names = tuple(Stage3ShapedComponentName)
        actual_names = tuple(component.name for component in self.components)
        if actual_names != expected_names:
            raise ValueError(
                "stage3-shaped reward components must use the canonical order"
            )
        component_total = math.fsum(component.score for component in self.components)
        if not math.isclose(self.total, component_total, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("reward total differs from its component sum")
        if type(self.answer_gated) is not bool:
            raise TypeError("answer_gated must be bool")
        if type(self.quality_judge_covered) is not bool:
            raise TypeError("quality_judge_covered must be bool")
        if type(self.quality_judge_applicable) is not bool:
            raise TypeError("quality_judge_applicable must be bool")
        if self.quality_judge_failure is not None and (
            type(self.quality_judge_failure) is not str
            or not self.quality_judge_failure.strip()
        ):
            raise ValueError("quality_judge_failure must be non-empty when present")
        if not self.quality_judge_applicable and (
            self.quality_judge_covered or self.quality_judge_failure is not None
        ):
            raise ValueError("non-applicable quality judge cannot be covered or failed")
        if self.quality_judge_applicable and self.quality_judge_covered == (
            self.quality_judge_failure is not None
        ):
            raise ValueError(
                "quality judge coverage and sample-local failure are inconsistent"
            )
        if self.version != STAGE3_SHAPED_REWARD_VERSION:
            raise ValueError("unexpected stage3-shaped reward version")

    def component(self, name: Stage3ShapedComponentName) -> Stage3ShapedRewardComponent:
        """Return one named component without exposing a mutable mapping."""

        if type(name) is not Stage3ShapedComponentName:
            raise TypeError("component lookup name must be Stage3ShapedComponentName")
        return self.components[tuple(Stage3ShapedComponentName).index(name)]


class Stage3ShapedRewardKernel:
    """Implement ``2*A_gated + T + F + G + P`` exactly."""

    _TOOL_DECISION_BASE = {
        ToolNecessityLabel.NEEDED: (1.0, -2.0),
        ToolNecessityLabel.OPTIONAL: (0.5, 0.0),
        ToolNecessityLabel.UNNECESSARY: (-0.5, 1.0),
    }
    _FOCUS_SCORE = {
        QualityJudgeScore.PASS: 1.0,
        QualityJudgeScore.PARTIAL: 0.5,
        QualityJudgeScore.FAIL: 0.0,
    }
    _GROUNDING_SCORE = {
        QualityJudgeScore.PASS: 1.0,
        QualityJudgeScore.PARTIAL: 0.5,
        QualityJudgeScore.FAIL: -1.0,
    }

    def score(self, facts: Stage3ShapedRewardFacts) -> Stage3ShapedRewardResult:
        if type(facts) is not Stage3ShapedRewardFacts:
            raise TypeError("facts must be Stage3ShapedRewardFacts")

        tool_attempted = facts.tool_call_count >= 1
        tool_succeeded = facts.successful_tgvf_observation_count >= 1
        answer_gated = (
            facts.tool_label is ToolNecessityLabel.NEEDED and not tool_succeeded
        )
        answer_score = 2.0 * float(facts.answer_correct and not answer_gated)

        used_base, unused_base = self._TOOL_DECISION_BASE[facts.tool_label]
        tool_decision_base = used_base if tool_succeeded else unused_base
        decision_score = facts.label_confidence * tool_decision_base
        extra_call_count = max(0, facts.tool_call_count - 1)
        extra_call_penalty = -0.05 * extra_call_count
        tool_score = decision_score + extra_call_penalty

        if tool_succeeded and facts.quality_judge_failure is None:
            assert facts.focus_score is not None
            assert facts.grounding_score is not None
            focus_score = self._FOCUS_SCORE[facts.focus_score]
            grounding_score = self._GROUNDING_SCORE[facts.grounding_score]
            focus_evidence = (
                f"judge_score={int(facts.focus_score)}; mapped_score={focus_score}"
            )
            grounding_evidence = (
                "judge_score="
                f"{int(facts.grounding_score)}; mapped_score={grounding_score}"
            )
            quality_judge_covered = True
        elif tool_succeeded:
            focus_score = 0.0
            grounding_score = 0.0
            failure = facts.quality_judge_failure
            assert failure is not None
            focus_evidence = f"judge_failure={failure}; coverage=0; fallback_score=0.0"
            grounding_evidence = (
                f"judge_failure={failure}; coverage=0; fallback_score=0.0"
            )
            quality_judge_covered = False
        else:
            focus_score = 0.0
            grounding_score = 0.0
            focus_evidence = "not_applicable=no_successful_tgvf_observation"
            grounding_evidence = "not_applicable=no_successful_tgvf_observation"
            quality_judge_covered = False

        protocol_score = -1.0 if facts.protocol_errors else 0.0
        error_summary = ",".join(facts.protocol_errors) or "none"
        components = (
            Stage3ShapedRewardComponent(
                Stage3ShapedComponentName.ANSWER,
                answer_score,
                (
                    f"answer_correct={facts.answer_correct}; "
                    f"answer_gated={answer_gated}; tool_succeeded={tool_succeeded}; "
                    "multiplier=2.0"
                ),
            ),
            Stage3ShapedRewardComponent(
                Stage3ShapedComponentName.TOOL,
                tool_score,
                (
                    f"label={facts.tool_label.value}; tool_attempted={tool_attempted}; "
                    f"tool_used={tool_succeeded}; "
                    f"decision_base={tool_decision_base}; "
                    f"label_confidence={facts.label_confidence}; "
                    f"decision_score={decision_score}; "
                    f"extra_call_count={extra_call_count}; "
                    f"extra_call_penalty={extra_call_penalty}"
                ),
            ),
            Stage3ShapedRewardComponent(
                Stage3ShapedComponentName.FOCUS,
                focus_score,
                focus_evidence,
            ),
            Stage3ShapedRewardComponent(
                Stage3ShapedComponentName.GROUNDING,
                grounding_score,
                grounding_evidence,
            ),
            Stage3ShapedRewardComponent(
                Stage3ShapedComponentName.PROTOCOL,
                protocol_score,
                f"protocol_errors={error_summary}; any_error={bool(facts.protocol_errors)}",
            ),
        )
        return Stage3ShapedRewardResult(
            total=float(math.fsum(component.score for component in components)),
            components=components,
            answer_gated=answer_gated,
            quality_judge_applicable=tool_succeeded,
            quality_judge_covered=quality_judge_covered,
            quality_judge_failure=facts.quality_judge_failure,
        )


__all__ = [
    "QualityJudgeScore",
    "STAGE3_SHAPED_REWARD_VERSION",
    "Stage3ShapedComponentName",
    "Stage3ShapedRewardComponent",
    "Stage3ShapedRewardFacts",
    "Stage3ShapedRewardKernel",
    "Stage3ShapedRewardResult",
    "ToolNecessityLabel",
]
