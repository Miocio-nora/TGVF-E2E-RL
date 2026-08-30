"""Stable schemas, prompt construction, and answer scoring for oracle-D utility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import re
from typing import Any, Literal

import torch

from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.protocol.schema import TGVF_FOCUS_TOOL_NAME
from tgvf_rl.public_api_compat import (
    freeze_public_class_annotations,
    rebind_public_class,
    rebind_public_function,
)
from tgvf_rl.qwen.base import InjectedForwardRequest, InjectedVisualBlock

from .native_pipeline import _qwen3_position_ids
from .readout import RepresentationVisualTensorBundle
from .runtime import Qwen3RepresentationRuntime
from .schema import RepresentationChoice, RepresentationTrainingSample
from .transcript import NATIVE_REPRESENTATION_PRE_REASONING


ORACLE_D_UTILITY_SCHEMA_VERSION = "representation_oracle_d_utility_v1"
ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION = "representation_oracle_d_utility_record_v1"
ORACLE_D_UTILITY_SUMMARY_SCHEMA_VERSION = "representation_oracle_d_utility_summary_v1"
DEFAULT_THINKING_EOS_TOKEN_IDS = (151645, 151643)
_TERMINAL_MARKERS = ("<|im_end|>", "<|endoftext|>")
_MCQ_CANONICAL = re.compile(
    r"^\s*(?:[\(\[]\s*([A-Z])\s*[\)\]]|([A-Z])\s*[\.:]|([A-Z])\s*$)",
    re.IGNORECASE,
)
_MCQ_ANSWER_MARKER = re.compile(
    r"\b(?:final\s+answer|answer)\s*(?:is|:|=|-)\s*"
    r"(?:(?:option|choice)\s*)?[\(\[]?\s*([A-Z])\s*[\)\]]?"
    r"(?=\s|[.,:;!?]|$)",
    re.IGNORECASE,
)


class OracleDUtilityArm(str, Enum):
    """One declared intervention in the oracle-target utility experiment."""

    IMAGE_ONLY = "image_only"
    DIRECT_ZERO_D_REPLACEMENT = "direct_zero_D_replacement"
    DIRECT_CORRECT_D_REPLACEMENT = "direct_correct_D_replacement"
    DIRECT_MATCHED_WRONG_D_REPLACEMENT = "direct_matched_wrong_D_replacement"
    TARGET_ZERO_D_ONLY = "target_zero_D_only"
    CORRECT_D_ONLY = "correct_D_only"
    IMAGE_TARGET_ZERO_D = "image_target_zero_D"
    IMAGE_CORRECT_D = "image_correct_D"
    MATCHED_WRONG_D = "matched_wrong_D"


class OracleBatchCompatibilityError(ValueError):
    """Raised when oracle arms cannot share one exact cached decode batch."""


DEFAULT_ORACLE_D_UTILITY_ARMS = (
    OracleDUtilityArm.IMAGE_ONLY,
    OracleDUtilityArm.TARGET_ZERO_D_ONLY,
    OracleDUtilityArm.CORRECT_D_ONLY,
    OracleDUtilityArm.IMAGE_TARGET_ZERO_D,
    OracleDUtilityArm.IMAGE_CORRECT_D,
)


@dataclass(frozen=True, slots=True)
class OracleDUtilityModelInput:
    """The complete set of row values permitted to reach the model."""

    sample_id: str
    image_group_key: str
    image: str
    question: str
    target: str
    sample_content_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "image_group_key",
            "image",
            "question",
            "target",
            "sample_content_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")


@dataclass(frozen=True, slots=True)
class OracleDUtilityGroundTruth:
    """Scoring-only values, deliberately absent from every model API."""

    sample_id: str
    short_answer: str
    choices: tuple[RepresentationChoice, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("ground-truth sample_id must be non-empty")
        if not isinstance(self.short_answer, str) or not self.short_answer.strip():
            raise ValueError("ground-truth short_answer must be non-empty")
        if not isinstance(self.choices, tuple) or any(
            not isinstance(choice, RepresentationChoice) for choice in self.choices
        ):
            raise TypeError("ground-truth choices must be typed and immutable")


@dataclass(frozen=True, slots=True)
class OracleAnswerScore:
    correct: bool | None
    route: str
    final_answer: str | None
    normalized_candidate: str | None
    normalized_expected: str
    expected_choice_label: str | None
    candidate_choice_label: str | None


@dataclass(frozen=True, slots=True)
class OracleGeneratedAnswer:
    token_ids: tuple[int, ...]
    text: str
    stop_reason: Literal["natural_stop", "length_cap"]


@dataclass(frozen=True, slots=True)
class OracleGroupVisuals:
    source: RepresentationVisualTensorBundle
    correct_d_by_sample_id: Mapping[str, RepresentationVisualTensorBundle]
    image_grid_thw: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class OracleImageOnlyParity:
    sample_id: str
    native_top1_token_id: int
    injected_top1_token_id: int
    top1_match: bool
    max_abs_logit_difference: float
    mean_abs_logit_difference: float
    native_prefix_token_count: int
    image_grid_thw: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class OracleArmContext:
    """One rendered generation prefix plus its selected visual interventions."""

    arm: OracleDUtilityArm
    rendered_text_sha256: str
    canonical_token_ids_sha256: str
    prefix_input_ids: torch.Tensor
    prefix_position_ids: torch.Tensor
    image_grid_thw: torch.Tensor
    visual_blocks: tuple[InjectedVisualBlock, ...]
    source_positions: tuple[int, ...]
    d_positions: tuple[int, ...]
    forbidden_multimodal_token_ids: frozenset[int]

    def __post_init__(self) -> None:
        if self.prefix_input_ids.dtype != torch.long or (
            self.prefix_input_ids.ndim != 2
            or self.prefix_input_ids.shape[0] != 1
            or self.prefix_input_ids.shape[1] == 0
        ):
            raise ValueError("oracle arm prefix must be non-empty long [1,S]")
        if self.prefix_position_ids.shape[-2:] != self.prefix_input_ids.shape:
            raise ValueError("oracle arm M-RoPE positions must align with its prefix")
        expected_blocks = {
            OracleDUtilityArm.IMAGE_ONLY: (True, False),
            OracleDUtilityArm.DIRECT_ZERO_D_REPLACEMENT: (False, True),
            OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT: (False, True),
            OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT: (False, True),
            OracleDUtilityArm.TARGET_ZERO_D_ONLY: (False, True),
            OracleDUtilityArm.CORRECT_D_ONLY: (False, True),
            OracleDUtilityArm.IMAGE_TARGET_ZERO_D: (True, True),
            OracleDUtilityArm.IMAGE_CORRECT_D: (True, True),
            OracleDUtilityArm.MATCHED_WRONG_D: (False, True),
        }[self.arm]
        if (bool(self.source_positions), bool(self.d_positions)) != expected_blocks:
            raise ValueError("oracle arm visual layout differs from its declaration")

    def materialize(
        self, suffix: tuple[int, ...], runtime: Qwen3RepresentationRuntime
    ) -> InjectedForwardRequest:
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in suffix
        ):
            raise ValueError("generated suffix must contain non-negative token IDs")
        if any(token_id in self.forbidden_multimodal_token_ids for token_id in suffix):
            raise RuntimeError("generated continuation introduced a multimodal token")
        suffix_tensor = torch.tensor(
            (suffix,), dtype=torch.long, device=self.prefix_input_ids.device
        )
        input_ids = torch.cat((self.prefix_input_ids, suffix_tensor), dim=1)
        attention_mask = torch.ones_like(input_ids)
        position_ids = _qwen3_position_ids(
            runtime.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=self.image_grid_thw,
        )
        if not torch.equal(
            position_ids[..., : self.prefix_input_ids.shape[1]],
            self.prefix_position_ids,
        ):
            raise RuntimeError("Qwen3 M-RoPE recompute changed the oracle arm prefix")
        return InjectedForwardRequest(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_blocks=self.visual_blocks,
            use_cache=False,
        )


def split_oracle_d_utility_sample(
    sample: RepresentationTrainingSample,
) -> tuple[OracleDUtilityModelInput, OracleDUtilityGroundTruth]:
    """Split a teacher row before any prompt, processor, or model operation."""

    if not isinstance(sample, RepresentationTrainingSample):
        raise TypeError("oracle utility row must be a RepresentationTrainingSample")
    return (
        OracleDUtilityModelInput(
            sample_id=sample.sample_id,
            image_group_key=sample.image_group_key,
            image=sample.image,
            question=sample.question,
            target=sample.target,
            sample_content_sha256=sample.content_sha256,
        ),
        OracleDUtilityGroundTruth(
            sample_id=sample.sample_id,
            short_answer=sample.short_answer,
            choices=sample.choices,
        ),
    )


def build_image_only_messages(
    model_input: OracleDUtilityModelInput,
) -> tuple[dict[str, Any], ...]:
    """Build the ordinary image/question prompt with no target transcript."""

    _require_model_input(model_input)
    return (
        {
            "role": "user",
            "content": (
                {"type": "image"},
                {"type": "text", "text": model_input.question},
            ),
        },
    )


def build_oracle_target_messages(
    model_input: OracleDUtilityModelInput,
    *,
    include_source_image: bool,
    assistant_dialect: NativeAssistantDialect,
) -> tuple[dict[str, Any], ...]:
    """Build user, oracle tool call, and D-backed tool response only."""

    _require_model_input(model_input)
    if not isinstance(include_source_image, bool):
        raise TypeError("include_source_image must be boolean")
    if not isinstance(assistant_dialect, NativeAssistantDialect):
        raise TypeError("assistant_dialect must be a native Qwen dialect")
    user_content: tuple[dict[str, str], ...]
    if include_source_image:
        user_content = (
            {"type": "image"},
            {"type": "text", "text": model_input.question},
        )
    else:
        user_content = ({"type": "text", "text": model_input.question},)
    tool_turn: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": (
            {
                "type": "function",
                "function": {
                    "name": TGVF_FOCUS_TOOL_NAME,
                    "arguments": {"target": model_input.target},
                },
            },
        ),
    }
    if assistant_dialect is NativeAssistantDialect.QWEN3_VL_THINKING:
        tool_turn["reasoning_content"] = NATIVE_REPRESENTATION_PRE_REASONING
    return (
        {"role": "user", "content": user_content},
        tool_turn,
        {"role": "tool", "content": ({"type": "image"},)},
    )


def score_oracle_generated_answer(
    generated_text: str,
    ground_truth: OracleDUtilityGroundTruth,
    *,
    assistant_dialect: NativeAssistantDialect = NativeAssistantDialect.QWEN3_VL_THINKING,
    generation_stop_reason: Literal["natural_stop", "length_cap"] = "natural_stop",
) -> OracleAnswerScore:
    """Apply a deterministic, judge-free answer rule after generation."""

    if not isinstance(generated_text, str):
        raise TypeError("generated_text must be text")
    if not isinstance(ground_truth, OracleDUtilityGroundTruth):
        raise TypeError("scoring requires separated ground truth")
    if assistant_dialect is NativeAssistantDialect.QWEN3_VL_THINKING:
        final_answer = _thinking_final_answer(generated_text)
    elif assistant_dialect is NativeAssistantDialect.QWEN3_VL_INSTRUCT:
        final_answer = _strip_terminal_markers(generated_text).strip() or None
    else:  # pragma: no cover - enum is closed
        raise ValueError("unsupported assistant dialect")
    expected = _normalize_answer(ground_truth.short_answer)
    expected_label = _choice_label_for_expected(ground_truth)
    if generation_stop_reason not in {"natural_stop", "length_cap"}:
        raise ValueError("unknown generation_stop_reason")
    if generation_stop_reason == "length_cap":
        return OracleAnswerScore(
            correct=False,
            route="length_cap",
            final_answer=final_answer,
            normalized_candidate=(
                _normalize_answer(final_answer) if final_answer is not None else None
            ),
            normalized_expected=expected,
            expected_choice_label=expected_label,
            candidate_choice_label=(
                _multiple_choice_label(final_answer, ground_truth.choices)
                if final_answer is not None
                else None
            ),
        )
    if final_answer is None:
        return OracleAnswerScore(
            correct=False,
            route="missing_final_answer",
            final_answer=None,
            normalized_candidate=None,
            normalized_expected=expected,
            expected_choice_label=expected_label,
            candidate_choice_label=None,
        )
    normalized = _normalize_answer(final_answer)
    candidate_label = _multiple_choice_label(final_answer, ground_truth.choices)
    if normalized == expected:
        route = "normalized_exact"
        correct = True
    elif expected_label is not None and candidate_label is not None:
        correct = candidate_label == expected_label
        route = "multiple_choice_label"
    else:
        candidate_number = _parse_number(normalized)
        expected_number = _parse_number(expected)
        if candidate_number is not None and expected_number is not None:
            correct = candidate_number == expected_number
            route = "numeric_equivalence"
        else:
            correct = None
            route = "semantic_unresolved"
    return OracleAnswerScore(
        correct=correct,
        route=route,
        final_answer=final_answer,
        normalized_candidate=normalized,
        normalized_expected=expected,
        expected_choice_label=expected_label,
        candidate_choice_label=candidate_label,
    )


def _thinking_final_answer(text: str) -> str | None:
    marker = "</think>"
    index = text.rfind(marker)
    if index < 0:
        return None
    value = _strip_terminal_markers(text[index + len(marker) :]).strip()
    return value or None


def _strip_terminal_markers(text: str) -> str:
    value = text.rstrip()
    changed = True
    while changed:
        changed = False
        for marker in _TERMINAL_MARKERS:
            if value.endswith(marker):
                value = value[: -len(marker)].rstrip()
                changed = True
    return value


def _normalize_answer(text: str) -> str:
    value = _strip_terminal_markers(text).strip()
    answer_tag = re.fullmatch(
        r"<answer>\s*(.*?)\s*</answer>", value, re.DOTALL | re.IGNORECASE
    )
    if answer_tag is not None:
        value = answer_tag.group(1).strip()
    boxed = re.fullmatch(r"\\boxed\s*\{(.*)\}", value, re.DOTALL)
    if boxed is not None:
        value = boxed.group(1).strip()
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _choice_label_for_expected(truth: OracleDUtilityGroundTruth) -> str | None:
    expected = _normalize_answer(truth.short_answer)
    labels = tuple(
        choice.label.upper()
        for choice in truth.choices
        if _normalize_answer(choice.text) == expected
    )
    return labels[0] if len(labels) == 1 else None


def _multiple_choice_label(
    final_answer: str, choices: tuple[RepresentationChoice, ...]
) -> str | None:
    labels = {choice.label.upper() for choice in choices}
    stripped = re.sub(r"[*_`]", "", final_answer.strip())
    canonical = _MCQ_CANONICAL.match(stripped)
    if canonical is not None:
        label = next(group.upper() for group in canonical.groups() if group is not None)
        return label if label in labels else None
    markers = tuple(_MCQ_ANSWER_MARKER.finditer(stripped))
    if markers:
        label = markers[-1].group(1).upper()
        return label if label in labels else None
    normalized = _normalize_answer(stripped)
    text_matches = tuple(
        choice.label.upper()
        for choice in choices
        if normalized == _normalize_answer(choice.text)
    )
    return text_matches[0] if len(text_matches) == 1 else None


def _parse_number(value: str) -> Fraction | None:
    compact = value.strip().replace(",", "")
    percent = compact.endswith("%")
    if percent:
        compact = compact[:-1].strip()
    try:
        result = Fraction(compact)
    except (ValueError, ZeroDivisionError):
        return None
    return result / 100 if percent else result


def _require_model_input(value: OracleDUtilityModelInput) -> None:
    if not isinstance(value, OracleDUtilityModelInput):
        raise TypeError("prompt construction requires OracleDUtilityModelInput")


_IMPLEMENTATION_MODULE = __name__
_PUBLIC_MODULE = "tgvf_rl.representation.training.oracle_d_utility"
_PUBLIC_TYPES = (
    OracleDUtilityArm,
    OracleBatchCompatibilityError,
    OracleDUtilityModelInput,
    OracleDUtilityGroundTruth,
    OracleAnswerScore,
    OracleGeneratedAnswer,
    OracleGroupVisuals,
    OracleImageOnlyParity,
    OracleArmContext,
)
for _annotated_public_type in (
    OracleDUtilityGroundTruth,
    OracleGeneratedAnswer,
    OracleGroupVisuals,
    OracleArmContext,
):
    freeze_public_class_annotations(
        _annotated_public_type,
        implementation_globals=globals(),
    )
del _annotated_public_type
for _public_type in _PUBLIC_TYPES:
    rebind_public_class(
        _public_type,
        implementation_module=_IMPLEMENTATION_MODULE,
        public_module=_PUBLIC_MODULE,
    )
for _public_function in (
    split_oracle_d_utility_sample,
    build_image_only_messages,
    build_oracle_target_messages,
    score_oracle_generated_answer,
    _thinking_final_answer,
    _strip_terminal_markers,
    _normalize_answer,
    _choice_label_for_expected,
    _multiple_choice_label,
    _parse_number,
    _require_model_input,
):
    rebind_public_function(
        _public_function,
        implementation_module=_IMPLEMENTATION_MODULE,
        public_module=_PUBLIC_MODULE,
    )


__all__ = [
    "DEFAULT_ORACLE_D_UTILITY_ARMS",
    "DEFAULT_THINKING_EOS_TOKEN_IDS",
    "ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION",
    "ORACLE_D_UTILITY_SCHEMA_VERSION",
    "ORACLE_D_UTILITY_SUMMARY_SCHEMA_VERSION",
    "OracleAnswerScore",
    "OracleArmContext",
    "OracleBatchCompatibilityError",
    "OracleDUtilityArm",
    "OracleDUtilityGroundTruth",
    "OracleDUtilityModelInput",
    "OracleGeneratedAnswer",
    "OracleGroupVisuals",
    "OracleImageOnlyParity",
    "build_image_only_messages",
    "build_oracle_target_messages",
    "score_oracle_generated_answer",
    "split_oracle_d_utility_sample",
]
