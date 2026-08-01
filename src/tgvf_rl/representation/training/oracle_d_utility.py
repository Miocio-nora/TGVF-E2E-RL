"""Resumable oracle-target evaluation of Stage1 focused-D answer utility.

This module intentionally separates model-visible fields from scoring-only
ground truth.  Model prompts can contain only the source image, question,
oracle trajectory target, and a visual block selected by the evaluation arm.
In particular, the representation teacher's evidence description, short
answer, and post-focus assistant turn are never accepted by a prompt builder.

The principal comparisons are ``correct_D_only - target_zero_D_only`` and
``image_correct_D - image_target_zero_D``.  They isolate the content of D while
holding the oracle target/tool transcript fixed.  Comparisons against
``image_only`` additionally include the oracle-target transcript intervention.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Iterator, Literal

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.conditioning import (
    TargetConditioningProviderKind,
    TargetConditioningRequest,
)
from tgvf_rl.conditioning.base import _bind_canonical_input_ids
from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.protocol.schema import TGVF_FOCUS_TOOL_NAME
from tgvf_rl.qwen.base import (
    CachedTokenForwardRequest,
    InjectedForwardRequest,
    InjectedVisualBlock,
    QwenVLMFamilyAdapter,
    batch_identical_injected_requests,
)
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.adapter import TGVFAdapterOutput

from .config import RepresentationTrainingConfig, load_representation_training_config
from .data import load_retained_representation_jsonl
from .distributed_checkpoint import load_rank_zero_adapter_owned_state_export
from .evaluation_runner import (
    _enable_determinism,
    _load_qwen,
    _load_rgb_image,
    _require_file_sha256,
    _seed_current_process,
    _torch_dtype,
    _validate_training_artifact_binding,
    load_representation_internal_evaluation_run_config,
)
from .native_pipeline import (
    ModelActionTarget,
    NativeActionTarget,
    Qwen3NativeRepresentationGroupBuilder,
    _adapter_output_bundle,
    _expand_native_visual_placeholders,
    _native_action_target_from_rendered,
    _move_processor_batch,
    _processor_batch,
    _qwen3_position_ids,
    _single_visual_expansion_count,
    _source_bundle,
)
from .readout import RepresentationVisualTensorBundle
from .runtime import (
    Qwen3ContextualHiddenStateStack,
    Qwen3RepresentationRuntime,
    Qwen3VisionFeatures,
    Qwen3VisionPreMergeRequest,
    create_qwen3_representation_runtime,
)
from .schema import RepresentationChoice, RepresentationTrainingSample
from .transcript import NATIVE_REPRESENTATION_PRE_REASONING


ORACLE_D_UTILITY_SCHEMA_VERSION = "representation_oracle_d_utility_v1"
ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION = "representation_oracle_d_utility_record_v1"
ORACLE_D_UTILITY_SUMMARY_SCHEMA_VERSION = "representation_oracle_d_utility_summary_v1"
DEFAULT_THINKING_EOS_TOKEN_IDS = (151645, 151643)
_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
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


def materialize_oracle_group_visuals(
    *,
    model_inputs: Sequence[OracleDUtilityModelInput],
    runtime: Qwen3RepresentationRuntime,
    group_builder: Qwen3NativeRepresentationGroupBuilder,
) -> OracleGroupVisuals:
    """Generate every correct D for one same-image group using the train path."""

    rows = tuple(model_inputs)
    if not rows or any(not isinstance(row, OracleDUtilityModelInput) for row in rows):
        raise ValueError("oracle visual materialization requires typed model inputs")
    if (
        len({row.image_group_key for row in rows}) != 1
        or len({row.image for row in rows}) != 1
    ):
        raise ValueError("oracle visual materialization requires one exact image group")
    if group_builder.runtime is not runtime:
        raise ValueError("group builder and oracle runtime differ")
    messages = tuple(
        build_oracle_target_messages(
            row,
            include_source_image=True,
            assistant_dialect=runtime.renderer.assistant_dialect,
        )
        for row in rows
    )
    with runtime.validated_group_execution():
        prefills = runtime.renderer.render_many(
            tuple(turns[:1] for turns in messages), add_generation_prompt=True
        )
        transcripts = runtime.renderer.render_many(
            tuple(turns[:2] for turns in messages), add_generation_prompt=False
        )
        actions: tuple[NativeActionTarget, ...] = tuple(
            _native_action_target_from_rendered(
                runtime,
                messages=turns,
                prefill=prefill,
                transcript=transcript,
            )
            for turns, prefill, transcript in zip(
                messages, prefills, transcripts, strict=True
            )
        )
        image = group_builder.image_loader(rows[0].image)
        if image is None:
            raise ValueError("image_loader returned None")
        first_action, first_expansion = (
            group_builder._materialize_action_with_expansion(actions[0], image)
        )
        visual_token_count = _single_visual_expansion_count(first_expansion)
        model_actions: tuple[ModelActionTarget, ...] = (
            first_action,
            *tuple(
                group_builder._materialize_action_from_shared_visual(
                    action,
                    reference=first_action,
                    visual_token_count=visual_token_count,
                )
                for action in actions[1:]
            ),
        )
        vision = runtime.extract_vision_features(
            Qwen3VisionPreMergeRequest(
                pixel_values=first_action.pixel_values,
                image_grid_thw=first_action.image_grid_thw,
            )
        )
        if int(vision.merged_main.shape[-2]) != visual_token_count:
            raise ValueError("source vision tokens differ from action expansion")
        correct: dict[str, RepresentationVisualTensorBundle] = {}
        for row, action in zip(rows, model_actions, strict=True):
            condition = _oracle_target_condition(
                runtime=runtime,
                model_input=row,
                action=action,
                vision=vision,
            )
            with torch.no_grad():
                output = runtime.adapter(runtime.make_adapter_input(condition, vision))
            if not isinstance(output, TGVFAdapterOutput):
                raise TypeError("Stage1 Adapter returned an invalid output")
            correct[row.sample_id] = _detached_bundle(_adapter_output_bundle(output))
        return OracleGroupVisuals(
            source=_detached_bundle(_source_bundle(vision)),
            correct_d_by_sample_id=correct,
            image_grid_thw=vision.image_grid_thw,
        )


def prepare_oracle_arm_context(
    *,
    model_input: OracleDUtilityModelInput,
    arm: OracleDUtilityArm,
    runtime: Qwen3RepresentationRuntime,
    source: RepresentationVisualTensorBundle,
    correct_d: RepresentationVisualTensorBundle,
    image_grid_thw: tuple[int, int, int],
    matched_wrong_d: RepresentationVisualTensorBundle | None = None,
) -> OracleArmContext:
    """Render one arm and bind exactly its source/D injection blocks."""

    _require_model_input(model_input)
    if not isinstance(arm, OracleDUtilityArm):
        raise TypeError("arm must be OracleDUtilityArm")
    direct_replacement = arm in {
        OracleDUtilityArm.DIRECT_ZERO_D_REPLACEMENT,
        OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT,
        OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT,
    }
    if direct_replacement and (
        not correct_d.d_deepstack_active or len(correct_d.deepstack) != 3
    ):
        raise ValueError(
            "direct D replacement requires active main D plus all three DeepStack branches"
        )
    if arm is OracleDUtilityArm.IMAGE_ONLY or direct_replacement:
        messages = build_image_only_messages(model_input)
        rendered_text, canonical_ids = _render_direct_without_tools(runtime, messages)
        include_source = arm is OracleDUtilityArm.IMAGE_ONLY
        if arm is OracleDUtilityArm.IMAGE_ONLY:
            selected_d = None
        elif arm is OracleDUtilityArm.DIRECT_ZERO_D_REPLACEMENT:
            selected_d = _zero_bundle(correct_d)
        elif arm is OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT:
            selected_d = correct_d
        else:
            if matched_wrong_d is None:
                raise ValueError(
                    "direct_matched_wrong_D_replacement requires a same-image wrong D"
                )
            selected_d = matched_wrong_d
    else:
        include_source = arm in {
            OracleDUtilityArm.IMAGE_TARGET_ZERO_D,
            OracleDUtilityArm.IMAGE_CORRECT_D,
        }
        messages = build_oracle_target_messages(
            model_input,
            include_source_image=include_source,
            assistant_dialect=runtime.renderer.assistant_dialect,
        )
        rendered = runtime.renderer.render(messages, add_generation_prompt=True)
        runtime.renderer.assert_generation_prefill(rendered, runtime.tokenizer)
        rendered_text, canonical_ids = rendered.text, rendered.token_ids
        if arm in {
            OracleDUtilityArm.TARGET_ZERO_D_ONLY,
            OracleDUtilityArm.IMAGE_TARGET_ZERO_D,
        }:
            selected_d = _zero_bundle(correct_d)
        elif arm in {
            OracleDUtilityArm.CORRECT_D_ONLY,
            OracleDUtilityArm.IMAGE_CORRECT_D,
        }:
            selected_d = correct_d
        elif arm is OracleDUtilityArm.MATCHED_WRONG_D:
            if matched_wrong_d is None:
                raise ValueError("matched_wrong_D requires a same-image wrong D")
            selected_d = matched_wrong_d
        else:  # pragma: no cover - enum is closed
            raise AssertionError("unhandled oracle D utility arm")
    _assert_visual_bundle_match(source, correct_d)
    if selected_d is not None:
        _assert_visual_bundle_match(correct_d, selected_d)
    token_count = int(source.main.shape[1])
    placeholder_count = (1 if include_source else 0) + (
        1 if selected_d is not None else 0
    )
    model_ids, expansion = _expand_native_visual_placeholders(
        runtime,
        canonical_ids,
        visual_token_counts=tuple(token_count for _ in range(placeholder_count)),
    )
    visual_id = runtime.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    blocks = tuple(
        mapped
        for token_id, mapped in zip(
            expansion.canonical_token_ids,
            expansion.canonical_to_model_positions,
            strict=True,
        )
        if token_id == visual_id
    )
    if len(blocks) != placeholder_count:
        raise RuntimeError("oracle arm visual expansion changed block count")
    block_index = 0
    source_positions: tuple[int, ...] = ()
    d_positions: tuple[int, ...] = ()
    injected: list[InjectedVisualBlock] = []
    if include_source:
        source_positions = blocks[block_index]
        block_index += 1
        injected.append(_injected_block("source_image", source_positions, source))
    if selected_d is not None:
        d_positions = blocks[block_index]
        injected.append(_injected_block("focused_d", d_positions, selected_d))
    # Qwen's M-RoPE helper is driven only by discrete IDs/grid metadata and
    # performs Python scalar/list operations.  Keep these tensors on CPU and
    # let the family adapter move the finished request once; putting them on
    # CUDA here creates many tiny kernels and synchronizations per decode token.
    prefix_ids = torch.tensor((model_ids,), dtype=torch.long)
    attention_mask = torch.ones_like(prefix_ids)
    grid = torch.tensor(
        tuple(image_grid_thw for _ in range(placeholder_count)),
        dtype=torch.long,
    )
    positions = _qwen3_position_ids(
        runtime.model,
        input_ids=prefix_ids,
        attention_mask=attention_mask,
        image_grid_thw=grid,
    )
    return OracleArmContext(
        arm=arm,
        rendered_text_sha256=sha256(rendered_text.encode("utf-8")).hexdigest(),
        canonical_token_ids_sha256=_integer_sequence_sha256(canonical_ids),
        prefix_input_ids=prefix_ids,
        prefix_position_ids=positions,
        image_grid_thw=grid,
        visual_blocks=tuple(injected),
        source_positions=source_positions,
        d_positions=d_positions,
        forbidden_multimodal_token_ids=_qwen3_multimodal_token_ids(runtime),
    )


def greedy_oracle_answer(
    *,
    context: OracleArmContext,
    runtime: Qwen3RepresentationRuntime,
    family_adapter: QwenVLMFamilyAdapter,
    eos_token_ids: tuple[int, ...],
    max_new_tokens: int,
    decode_mode: Literal["cached", "no_cache"],
) -> OracleGeneratedAnswer:
    """Run deterministic native Thinking greedy generation for one arm."""

    if not eos_token_ids or len(set(eos_token_ids)) != len(eos_token_ids):
        raise ValueError("eos_token_ids must be non-empty and unique")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in eos_token_ids
    ):
        raise ValueError("eos_token_ids must be non-negative integers")
    if isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if decode_mode not in {"cached", "no_cache"}:
        raise ValueError("decode_mode must be cached or no_cache")
    generated: list[int] = []
    stop_reason: Literal["natural_stop", "length_cap"] = "length_cap"
    if decode_mode == "cached":
        if not family_adapter.capabilities.native_injected_kv_cache:
            raise ValueError("family adapter has no injected KV-cache path")
        materialized = context.materialize((), runtime)
        with torch.no_grad():
            result = family_adapter.prefill_injected_cache(runtime.model, materialized)
        past_key_values = result.past_key_values
        next_logits = result.logits[0, -1].float()
        for token_index in range(max_new_tokens):
            token_id = _greedy_token(next_logits)
            generated.append(token_id)
            if token_id in eos_token_ids:
                stop_reason = "natural_stop"
                break
            if token_index + 1 == max_new_tokens:
                break
            full_request = context.materialize(tuple(generated), runtime)
            cache_position = torch.tensor(
                (full_request.input_ids.shape[1] - 1,),
                dtype=torch.long,
                device=full_request.input_ids.device,
            )
            with torch.no_grad():
                result = family_adapter.forward_cached_token(
                    runtime.model,
                    CachedTokenForwardRequest(
                        input_ids=full_request.input_ids[:, -1:],
                        attention_mask=full_request.attention_mask,
                        position_ids=full_request.position_ids[..., -1:],
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                    ),
                )
            past_key_values = result.past_key_values
            next_logits = result.logits[0, -1].float()
    else:
        for _ in range(max_new_tokens):
            materialized = context.materialize(tuple(generated), runtime)
            with torch.no_grad():
                result = family_adapter.forward_injected(runtime.model, materialized)
            token_id = _greedy_token(result.logits[0, -1].float())
            generated.append(token_id)
            if token_id in eos_token_ids:
                stop_reason = "natural_stop"
                break
    token_ids = tuple(generated)
    if not token_ids:
        raise RuntimeError("oracle greedy generation produced no token")
    if any(
        token_id in context.forbidden_multimodal_token_ids for token_id in token_ids
    ):
        raise RuntimeError("oracle greedy generation emitted a multimodal token")
    text = runtime.tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    runtime.renderer.assert_tokenizer_length()
    if not isinstance(text, str) or not text:
        raise RuntimeError("oracle greedy token IDs decoded to empty text")
    return OracleGeneratedAnswer(
        token_ids=token_ids,
        text=text,
        stop_reason=stop_reason,
    )


def greedy_oracle_answers_batched(
    *,
    contexts: Sequence[OracleArmContext],
    runtime: Qwen3RepresentationRuntime,
    family_adapter: QwenVLMFamilyAdapter,
    eos_token_ids: tuple[int, ...],
    max_new_tokens: int,
) -> tuple[OracleGeneratedAnswer, ...]:
    """Greedily decode compatible oracle arms through one shared KV-cache batch."""

    lanes = tuple(contexts)
    if len(lanes) < 2:
        raise OracleBatchCompatibilityError(
            "batched oracle generation requires at least two arms"
        )
    if not eos_token_ids or len(set(eos_token_ids)) != len(eos_token_ids):
        raise ValueError("eos_token_ids must be non-empty and unique")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in eos_token_ids
    ):
        raise ValueError("eos_token_ids must be non-negative integers")
    if isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not family_adapter.capabilities.native_injected_kv_cache:
        raise ValueError("family adapter has no injected KV-cache path")

    prefixes = tuple(context.materialize((), runtime) for context in lanes)
    try:
        batched_prefix = batch_identical_injected_requests(prefixes)
    except ValueError as error:
        raise OracleBatchCompatibilityError(
            "oracle arms do not share one exact native generation prefix"
        ) from error
    with torch.no_grad():
        result = family_adapter.prefill_injected_cache(
            runtime.model,
            batched_prefix,
        )
    past_key_values = result.past_key_values
    next_logits = _batched_next_logits(result.logits, lane_count=len(lanes))
    generated: list[list[int]] = [[] for _context in lanes]
    cache_suffixes: list[list[int]] = [[] for _context in lanes]
    finished = [False for _context in lanes]
    stop_reasons: list[Literal["natural_stop", "length_cap"]] = [
        "length_cap" for _context in lanes
    ]
    finished_fill_token_id = eos_token_ids[0]

    for token_index in range(max_new_tokens):
        predicted = tuple(
            int(token_id)
            for token_id in torch.argmax(next_logits, dim=-1).detach().cpu().tolist()
        )
        if len(predicted) != len(lanes):
            raise RuntimeError("batched oracle logits lost a decode lane")
        for lane_index, token_id in enumerate(predicted):
            if finished[lane_index]:
                cache_suffixes[lane_index].append(finished_fill_token_id)
                continue
            generated[lane_index].append(token_id)
            cache_suffixes[lane_index].append(token_id)
            if token_id in eos_token_ids:
                finished[lane_index] = True
                stop_reasons[lane_index] = "natural_stop"
        if all(finished) or token_index + 1 == max_new_tokens:
            break

        full_requests = tuple(
            context.materialize(tuple(cache_suffixes[lane_index]), runtime)
            for lane_index, context in enumerate(lanes)
        )
        sequence_lengths = {
            int(request.input_ids.shape[1]) for request in full_requests
        }
        if len(sequence_lengths) != 1:
            raise OracleBatchCompatibilityError(
                "batched oracle arms produced different cached sequence lengths"
            )
        first_request = full_requests[0]
        position_batch_dimension = 0 if first_request.position_ids.ndim == 2 else 1
        cache_position = torch.tensor(
            (first_request.input_ids.shape[1] - 1,),
            dtype=torch.long,
            device=first_request.input_ids.device,
        )
        with torch.no_grad():
            result = family_adapter.forward_cached_token(
                runtime.model,
                CachedTokenForwardRequest(
                    input_ids=torch.cat(
                        tuple(request.input_ids[:, -1:] for request in full_requests),
                        dim=0,
                    ),
                    attention_mask=torch.cat(
                        tuple(request.attention_mask for request in full_requests),
                        dim=0,
                    ),
                    position_ids=torch.cat(
                        tuple(
                            request.position_ids[..., -1:] for request in full_requests
                        ),
                        dim=position_batch_dimension,
                    ),
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                ),
            )
        past_key_values = result.past_key_values
        next_logits = _batched_next_logits(result.logits, lane_count=len(lanes))

    answers: list[OracleGeneratedAnswer] = []
    for context, token_values, stop_reason in zip(
        lanes,
        generated,
        stop_reasons,
        strict=True,
    ):
        token_ids = tuple(token_values)
        if not token_ids:
            raise RuntimeError("batched oracle greedy generation produced no token")
        if any(
            token_id in context.forbidden_multimodal_token_ids for token_id in token_ids
        ):
            raise RuntimeError("oracle greedy generation emitted a multimodal token")
        text = runtime.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        runtime.renderer.assert_tokenizer_length()
        if not isinstance(text, str) or not text:
            raise RuntimeError("oracle greedy token IDs decoded to empty text")
        answers.append(
            OracleGeneratedAnswer(
                token_ids=token_ids,
                text=text,
                stop_reason=stop_reason,
            )
        )
    return tuple(answers)


def _batched_next_logits(logits: torch.Tensor, *, lane_count: int) -> torch.Tensor:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[0] != lane_count
        or logits.shape[1] == 0
        or logits.shape[2] == 0
    ):
        raise RuntimeError("batched oracle generation returned invalid logits")
    next_logits = logits[:, -1].float()
    if not bool(torch.isfinite(next_logits).all()):
        raise RuntimeError("batched oracle generation returned invalid logits")
    return next_logits


def verify_image_only_injected_native_parity(
    *,
    model_input: OracleDUtilityModelInput,
    context: OracleArmContext,
    runtime: Qwen3RepresentationRuntime,
    family_adapter: QwenVLMFamilyAdapter,
    image_loader: Any,
    image_max_pixels: int | None,
) -> OracleImageOnlyParity:
    """Compare the first image-only next token to Qwen's native top-level path."""

    if context.arm is not OracleDUtilityArm.IMAGE_ONLY:
        raise ValueError("native parity requires the image_only arm")
    if not callable(image_loader):
        raise TypeError("image_loader must be callable")
    messages = build_image_only_messages(model_input)
    rendered_text, canonical_ids = _render_direct_without_tools(runtime, messages)
    if _integer_sequence_sha256(canonical_ids) != context.canonical_token_ids_sha256:
        raise RuntimeError("image-only parity rerender changed canonical token IDs")
    image = image_loader(model_input.image)
    if image is None:
        raise ValueError("image_loader returned None during image-only parity")
    processor_batch = _processor_batch(
        runtime.processor,
        text=rendered_text,
        images=(image,),
        image_max_pixels=image_max_pixels,
    )
    input_ids, attention_mask, pixel_values, grid = _move_processor_batch(
        runtime, processor_batch
    )
    if tuple(int(value) for value in input_ids[0].detach().cpu().tolist()) != tuple(
        int(value) for value in context.prefix_input_ids[0].tolist()
    ):
        raise RuntimeError(
            "manual injected source prefix differs from Qwen processor IDs"
        )
    observed_grid = tuple(int(value) for value in grid[0].detach().cpu().tolist())
    expected_grid = tuple(int(value) for value in context.image_grid_thw[0].tolist())
    if observed_grid != expected_grid:
        raise RuntimeError(
            "manual injected source grid differs from Qwen processor grid"
        )
    with torch.no_grad():
        native = runtime.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            use_cache=False,
            return_dict=True,
        )
        injected = family_adapter.forward_injected(
            runtime.model, context.materialize((), runtime)
        )
    native_all_logits = getattr(native, "logits", None)
    if not isinstance(native_all_logits, torch.Tensor):
        raise RuntimeError("native Qwen image-only forward returned no logits")
    native_logits = native_all_logits[0, -1].float()
    injected_logits = injected.logits[0, -1].float()
    if native_logits.shape != injected_logits.shape or not bool(
        torch.isfinite(native_logits).all() and torch.isfinite(injected_logits).all()
    ):
        raise RuntimeError("native/injected image-only logits are invalid")
    absolute = (native_logits - injected_logits).abs()
    native_top1 = int(torch.argmax(native_logits).item())
    injected_top1 = int(torch.argmax(injected_logits).item())
    result = OracleImageOnlyParity(
        sample_id=model_input.sample_id,
        native_top1_token_id=native_top1,
        injected_top1_token_id=injected_top1,
        top1_match=native_top1 == injected_top1,
        max_abs_logit_difference=float(absolute.max().item()),
        mean_abs_logit_difference=float(absolute.mean().item()),
        native_prefix_token_count=int(input_ids.shape[1]),
        image_grid_thw=observed_grid,
    )
    if not result.top1_match:
        raise RuntimeError(
            "image-only injected/native first-token top-1 parity failed: "
            f"native={native_top1}, injected={injected_top1}"
        )
    return result


def run_oracle_d_utility_evaluation(
    source_config_path: str | Path,
    *,
    output_root: str | Path,
    arms: Sequence[OracleDUtilityArm | str] = DEFAULT_ORACLE_D_UTILITY_ARMS,
    max_new_tokens: int,
    eos_token_ids: Sequence[int] = DEFAULT_THINKING_EOS_TOKEN_IDS,
    decode_mode: Literal["cached", "no_cache"] = "cached",
    group_start: int = 0,
    group_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    """Run or exactly resume one single-GPU, image-group-sharded evaluation."""

    selected_arms = _normalize_arms(arms)
    selected_eos_token_ids = _normalize_eos_token_ids(eos_token_ids)
    _validate_selection(
        max_new_tokens=max_new_tokens,
        eos_token_ids=selected_eos_token_ids,
        decode_mode=decode_mode,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    source_config = load_representation_internal_evaluation_run_config(
        source_config_path
    )
    if source_config.evaluation_data_path is None or (
        source_config.evaluation_data_source_sha256 is None
    ):
        raise ValueError("oracle D utility requires an explicit evaluation_data split")
    training = load_representation_training_config(source_config.training_config_path)
    if Path(training.model.local_path).name != "Qwen3-VL-8B-Thinking":
        raise ValueError("oracle D utility entry is pinned to Qwen3-VL-8B-Thinking")
    _require_file_sha256(
        source_config.training_config_path,
        source_config.training_config_sha256,
        name="training config",
    )
    _require_file_sha256(
        source_config.artifact_path,
        source_config.artifact_file_sha256,
        name="Adapter artifact",
    )
    export = load_rank_zero_adapter_owned_state_export(source_config.artifact_path)
    manifest = export.manifest
    run_identity = manifest.run_identity
    if state_digest(manifest) != source_config.artifact_manifest_sha256:
        raise ValueError("Adapter artifact manifest SHA256 mismatch")
    if (
        manifest.run_identity_sha256 != source_config.expected_run_identity_sha256
        or run_identity.identity_sha256 != source_config.expected_run_identity_sha256
    ):
        raise ValueError("Adapter artifact run identity mismatch")
    if manifest.global_step != source_config.expected_global_step:
        raise ValueError("Adapter artifact global step mismatch")
    _validate_training_artifact_binding(training, run_identity)
    data = load_retained_representation_jsonl(
        source_config.evaluation_data_path,
        expected_source_sha256=source_config.evaluation_data_source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    all_groups = _ordered_sample_groups(data.samples)
    selected_groups = _select_groups(
        all_groups,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    wrong_d_arms = {
        OracleDUtilityArm.MATCHED_WRONG_D,
        OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT,
    }
    if wrong_d_arms.intersection(selected_arms) and any(
        len(group) < 2 for _, group in selected_groups
    ):
        raise ValueError(
            "matched-wrong D arms require every selected image group to have K>=2"
        )
    model_truth_rows = tuple(
        split_oracle_d_utility_sample(sample)
        for _group_ordinal, group in selected_groups
        for sample in group
    )
    model_inputs = tuple(row[0] for row in model_truth_rows)
    identity_payload = _run_identity_payload(
        source_config=source_config,
        training=training,
        data_manifest_sha256=data.manifest.manifest_sha256,
        model_inputs=model_inputs,
        arms=selected_arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=selected_eos_token_ids,
        decode_mode=decode_mode,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    output = Path(output_root).resolve()
    ledger = _OracleRunLedger(
        output,
        identity_payload=identity_payload,
        expected_keys=tuple(
            (row.sample_id, arm.value) for row in model_inputs for arm in selected_arms
        ),
    )
    with ledger.locked():
        ledger.prepare()
        if ledger.complete:
            return ledger.summary()
        _require_single_gpu_environment()
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        _enable_determinism()
        _seed_current_process(source_config.evaluation.random_seed)
        processor, model = _load_qwen(training, device=device)
        tokenizer_length_before = len(processor.tokenizer)
        runtime = create_qwen3_representation_runtime(
            model=model,
            processor=processor,
            model_identity=training.model_identity,
            conditioning_config=training.provider,
            adapter_dtype=_torch_dtype(training.model.dtype),
            adapter_variant=training.adapter_variant,
            fixture_mode=False,
        )
        if runtime.renderer.assistant_dialect is not (
            NativeAssistantDialect.QWEN3_VL_THINKING
        ):
            raise ValueError("oracle D utility requires the native Thinking dialect")
        model_eos_token_ids = _model_eos_token_ids(model)
        if set(model_eos_token_ids) != set(selected_eos_token_ids):
            raise ValueError(
                "configured EOS IDs differ from the local Thinking generation config: "
                f"configured={selected_eos_token_ids}, model={model_eos_token_ids}"
            )
        if len(processor.tokenizer) != tokenizer_length_before:
            raise RuntimeError("oracle D utility changed tokenizer length")
        run_identity.adapter_contract.assert_matches(runtime.adapter)
        if export.state is None:
            raise RuntimeError("Adapter export has no tensor state")
        runtime.adapter.load_artifact_state_dict(export.state)
        runtime.adapter.requires_grad_(False)
        runtime.adapter.eval()
        model.requires_grad_(False)
        model.eval()
        family_adapter = Qwen3VLAdapter()
        group_builder = Qwen3NativeRepresentationGroupBuilder(
            runtime=runtime,
            family_adapter=family_adapter,
            prompt=training.prompt,
            image_loader=_load_rgb_image,
            image_max_pixels=training.model.image_max_pixels,
        )
        truth_by_id = {truth.sample_id: truth for _model, truth in model_truth_rows}
        started = time.monotonic()
        parity_path = output / "image_only_native_parity.json"
        parity_checked = OracleDUtilityArm.IMAGE_ONLY not in selected_arms
        if parity_path.exists():
            parity_payload = json.loads(parity_path.read_text(encoding="utf-8"))
            if (
                parity_payload.get("run_identity_sha256") != ledger.identity_sha256
                or parity_payload.get("top1_match") is not True
            ):
                raise ValueError(
                    "existing image-only native parity artifact is invalid"
                )
            parity_checked = True
        for group_ordinal, samples in selected_groups:
            group_models = tuple(
                split_oracle_d_utility_sample(sample)[0] for sample in samples
            )
            pending = tuple(
                row
                for row in group_models
                if any(
                    not ledger.has(row.sample_id, arm.value) for arm in selected_arms
                )
            )
            if not pending:
                continue
            visuals = materialize_oracle_group_visuals(
                model_inputs=group_models,
                runtime=runtime,
                group_builder=group_builder,
            )
            if not parity_checked:
                parity_context = prepare_oracle_arm_context(
                    model_input=group_models[0],
                    arm=OracleDUtilityArm.IMAGE_ONLY,
                    runtime=runtime,
                    source=visuals.source,
                    correct_d=visuals.correct_d_by_sample_id[group_models[0].sample_id],
                    image_grid_thw=visuals.image_grid_thw,
                )
                parity = verify_image_only_injected_native_parity(
                    model_input=group_models[0],
                    context=parity_context,
                    runtime=runtime,
                    family_adapter=family_adapter,
                    image_loader=group_builder.image_loader,
                    image_max_pixels=group_builder.image_max_pixels,
                )
                _atomic_write_json(
                    parity_path,
                    {
                        "schema_version": "oracle_image_only_native_parity_v1",
                        "run_identity_sha256": ledger.identity_sha256,
                        **asdict(parity),
                    },
                )
                parity_checked = True
            for row_index, row in enumerate(group_models):
                correct_d = visuals.correct_d_by_sample_id[row.sample_id]
                wrong_d = (
                    visuals.correct_d_by_sample_id[
                        group_models[(row_index + 1) % len(group_models)].sample_id
                    ]
                    if len(group_models) > 1
                    else None
                )
                for arm in selected_arms:
                    if ledger.has(row.sample_id, arm.value):
                        continue
                    context = prepare_oracle_arm_context(
                        model_input=row,
                        arm=arm,
                        runtime=runtime,
                        source=visuals.source,
                        correct_d=correct_d,
                        image_grid_thw=visuals.image_grid_thw,
                        matched_wrong_d=wrong_d,
                    )
                    generated = greedy_oracle_answer(
                        context=context,
                        runtime=runtime,
                        family_adapter=family_adapter,
                        eos_token_ids=selected_eos_token_ids,
                        max_new_tokens=max_new_tokens,
                        decode_mode=decode_mode,
                    )
                    score = score_oracle_generated_answer(
                        generated.text,
                        truth_by_id[row.sample_id],
                        assistant_dialect=runtime.renderer.assistant_dialect,
                        generation_stop_reason=generated.stop_reason,
                    )
                    ledger.commit(
                        {
                            "schema_version": ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION,
                            "run_identity_sha256": ledger.identity_sha256,
                            "sample_id": row.sample_id,
                            "sample_content_sha256": row.sample_content_sha256,
                            "image_group_key": row.image_group_key,
                            "group_ordinal": group_ordinal,
                            "arm": arm.value,
                            "arm_contract": _arm_contract(arm),
                            "wrong_d_source_sample_id": (
                                group_models[
                                    (row_index + 1) % len(group_models)
                                ].sample_id
                                if arm
                                in {
                                    OracleDUtilityArm.MATCHED_WRONG_D,
                                    OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT,
                                }
                                else None
                            ),
                            "rendered_prefix_text_sha256": context.rendered_text_sha256,
                            "canonical_prefix_token_ids_sha256": context.canonical_token_ids_sha256,
                            "prefix_token_count": int(
                                context.prefix_input_ids.shape[1]
                            ),
                            "source_visual_token_count": len(context.source_positions),
                            "d_visual_token_count": len(context.d_positions),
                            "generated_token_ids": list(generated.token_ids),
                            "generated_text": generated.text,
                            "generation_stop_reason": generated.stop_reason,
                            "score": asdict(score),
                            "expected_short_answer": truth_by_id[
                                row.sample_id
                            ].short_answer,
                            "elapsed_seconds_since_process_start": time.monotonic()
                            - started,
                        }
                    )
            del visuals
        if len(processor.tokenizer) != tokenizer_length_before:
            raise RuntimeError("oracle D utility changed tokenizer length")
        return ledger.summary()


def _oracle_target_condition(
    *,
    runtime: Qwen3RepresentationRuntime,
    model_input: OracleDUtilityModelInput,
    action: ModelActionTarget,
    vision: Qwen3VisionFeatures,
) -> Any:
    action.assert_bound_invariants()
    conditioning_ids = action.input_ids[0]
    request = TargetConditioningRequest(
        input_ids=conditioning_ids,
        target_span=action.target_span,
        expected_target_token_ids=action.target_token_ids,
        trajectory_id=f"oracle-d-utility:{model_input.sample_id}",
        call_index=0,
        model_identity=runtime.model_identity,
        canonical_input_ids_proof=_bind_canonical_input_ids(
            conditioning_ids, action.model_token_ids
        ),
    )
    contextual = None
    if runtime.conditioning_config.provider is (
        TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
    ):
        with torch.no_grad():
            output = runtime.model(
                input_ids=action.input_ids,
                attention_mask=action.attention_mask,
                pixel_values=action.pixel_values,
                image_grid_thw=action.image_grid_thw,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        raw_layers = getattr(output, "hidden_states", None)
        if not isinstance(raw_layers, (tuple, list)) or not raw_layers:
            raise RuntimeError("frozen Qwen did not return contextual hidden states")
        layers = tuple(layer.detach().clone() for layer in raw_layers)
        if any(layer.shape[:2] != action.input_ids.shape for layer in layers):
            raise ValueError("Qwen contextual states do not align with action IDs")
        contextual = Qwen3ContextualHiddenStateStack(
            tuple(layer[0] for layer in layers)
        )
    return runtime.build_target_condition(request, contextual_hidden_states=contextual)


def _render_direct_without_tools(
    runtime: Qwen3RepresentationRuntime,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[int, ...]]:
    """Render the true image-only baseline without exposing a tool schema."""

    runtime.renderer.assert_tokenizer_length()
    runtime.renderer.assert_chat_template_identity()
    try:
        text = runtime.processor.apply_chat_template(
            list(messages),
            tools=None,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError as error:
        raise TypeError(
            "Qwen processor rejected the direct image-only prompt"
        ) from error
    if not isinstance(text, str) or not text.endswith(
        runtime.renderer.assistant_dialect.generation_prefill_text
    ):
        raise ValueError("direct image-only transcript has an invalid native prefill")
    token_ids = tuple(
        int(token_id)
        for token_id in runtime.tokenizer.encode(text, add_special_tokens=False)
    )
    runtime.renderer.assert_tokenizer_length()
    runtime.renderer.assert_chat_template_identity()
    return text, token_ids


def _injected_block(
    kind: str,
    positions: tuple[int, ...],
    visual: RepresentationVisualTensorBundle,
) -> InjectedVisualBlock:
    return InjectedVisualBlock(
        kind=kind,
        positions=positions,
        embeddings=visual.main,
        deepstack=visual.deepstack,
        deepstack_positions=tuple(positions for _ in visual.deepstack),
    )


def _zero_bundle(
    reference: RepresentationVisualTensorBundle,
) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.zeros_like(reference.main),
        deepstack=tuple(torch.zeros_like(branch) for branch in reference.deepstack),
        branch_layers=reference.branch_layers,
        d_deepstack_active=reference.d_deepstack_active,
    )


def _detached_bundle(
    value: RepresentationVisualTensorBundle,
) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=value.main.detach().clone(),
        deepstack=tuple(branch.detach().clone() for branch in value.deepstack),
        branch_layers=value.branch_layers,
        d_deepstack_active=value.d_deepstack_active,
    )


def _assert_visual_bundle_match(
    reference: RepresentationVisualTensorBundle,
    candidate: RepresentationVisualTensorBundle,
) -> None:
    if (
        candidate.main.shape != reference.main.shape
        or candidate.main.dtype != reference.main.dtype
        or candidate.main.device != reference.main.device
        or candidate.branch_layers != reference.branch_layers
        or len(candidate.deepstack) != len(reference.deepstack)
        or any(
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.device != right.device
            for left, right in zip(
                candidate.deepstack, reference.deepstack, strict=True
            )
        )
    ):
        raise ValueError("oracle source/D visual bundle contracts differ")


def _ordered_sample_groups(
    samples: Sequence[RepresentationTrainingSample],
) -> tuple[tuple[int, tuple[RepresentationTrainingSample, ...]], ...]:
    grouped: OrderedDict[str, list[RepresentationTrainingSample]] = OrderedDict()
    for sample in samples:
        grouped.setdefault(sample.image_group_key, []).append(sample)
    return tuple(
        (ordinal, tuple(group)) for ordinal, group in enumerate(grouped.values())
    )


def _select_groups(
    groups: Sequence[tuple[int, tuple[RepresentationTrainingSample, ...]]],
    *,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> tuple[tuple[int, tuple[RepresentationTrainingSample, ...]], ...]:
    after_start = tuple(groups[group_start:])
    sharded = tuple(
        group
        for index, group in enumerate(after_start)
        if index % shard_count == shard_index
    )
    selected = sharded if group_limit is None else sharded[:group_limit]
    if not selected:
        raise ValueError("oracle D utility selection contains no image group")
    return selected


def _normalize_arms(
    arms: Sequence[OracleDUtilityArm | str],
) -> tuple[OracleDUtilityArm, ...]:
    if isinstance(arms, (str, bytes)):
        raise TypeError("arms must be a sequence")
    try:
        result = tuple(
            arm if isinstance(arm, OracleDUtilityArm) else OracleDUtilityArm(arm)
            for arm in arms
        )
    except ValueError as error:
        raise ValueError("unknown oracle D utility arm") from error
    if not result or len(set(result)) != len(result):
        raise ValueError("oracle D utility arms must be non-empty and unique")
    return result


def _validate_selection(
    *,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    decode_mode: str,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> None:
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if not eos_token_ids:
        raise ValueError("eos_token_ids must be non-empty")
    if decode_mode not in {"cached", "no_cache"}:
        raise ValueError("decode_mode must be cached or no_cache")
    for name, value in (("group_start", group_start), ("shard_index", shard_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 0
    ):
        raise ValueError("shard_count must be positive")
    if shard_index >= shard_count:
        raise ValueError("shard_index must be smaller than shard_count")
    if group_limit is not None and (
        isinstance(group_limit, bool)
        or not isinstance(group_limit, int)
        or group_limit <= 0
    ):
        raise ValueError("group_limit must be positive when set")


def _run_identity_payload(
    *,
    source_config: Any,
    training: RepresentationTrainingConfig,
    data_manifest_sha256: str,
    model_inputs: Sequence[OracleDUtilityModelInput],
    arms: tuple[OracleDUtilityArm, ...],
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    decode_mode: str,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    module_path = Path(__file__).resolve()
    return {
        "schema_version": ORACLE_D_UTILITY_SCHEMA_VERSION,
        "claim_scope": "oracle_target_conditioned_D_utility_not_end_to_end_tool_selection",
        "source_config_path": str(source_config.source_path),
        "source_config_sha256": source_config.source_sha256,
        "training_config_sha256": source_config.training_config_sha256,
        "artifact_file_sha256": source_config.artifact_file_sha256,
        "artifact_manifest_sha256": source_config.artifact_manifest_sha256,
        "training_run_identity_sha256": source_config.expected_run_identity_sha256,
        "expected_global_step": source_config.expected_global_step,
        "model_name": training.model.model_name,
        "model_path": str(training.model.local_path),
        "data_manifest_sha256": data_manifest_sha256,
        "ordered_selected_samples": [
            {
                "sample_id": row.sample_id,
                "sample_content_sha256": row.sample_content_sha256,
                "image_group_key": row.image_group_key,
            }
            for row in model_inputs
        ],
        "arms": [arm.value for arm in arms],
        "arm_contracts": {arm.value: _arm_contract(arm) for arm in arms},
        "max_new_tokens": max_new_tokens,
        "eos_token_ids": list(eos_token_ids),
        "legacy_source_config_eos_token_ids": list(
            source_config.evaluation.eos_token_ids
        ),
        "decode_mode": decode_mode,
        "greedy": True,
        "random_seed": source_config.evaluation.random_seed,
        "group_start": group_start,
        "group_limit": group_limit,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "image_only_tool_schema_exposed": False,
        "target_arm_tool_schema_exposed": True,
        "ground_truth_model_input": False,
        "post_focus_transcript_model_input": False,
        "scoring": "thinking_suffix_deterministic_mcq_exact_numeric_v1",
        "cached_decode_note": (
            "cached/no-cache token parity passed the bounded RP62 lane; strict full-logit "
            "parity was not established by RP61"
            if decode_mode == "cached"
            else "full-prefix no-cache oracle"
        ),
        "live_git_head": _git_head(root),
        "live_module_sha256": sha256(module_path.read_bytes()).hexdigest(),
    }


def _arm_contract(arm: OracleDUtilityArm) -> dict[str, Any]:
    return {
        OracleDUtilityArm.IMAGE_ONLY: {
            "prompt": "question_only_no_tool_schema",
            "source_image": True,
            "oracle_target_transcript": False,
            "d": "absent",
        },
        OracleDUtilityArm.DIRECT_ZERO_D_REPLACEMENT: {
            "prompt": "image_question_no_tool_schema_original_image_slot",
            "source_image": False,
            "oracle_target_transcript": False,
            "d": "all_zero_main_and_all_deepstack_in_original_image_slot",
        },
        OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT: {
            "prompt": "image_question_no_tool_schema_original_image_slot",
            "source_image": False,
            "oracle_target_transcript": False,
            "d": "correct_target_stage1_main_and_all_deepstack_in_original_image_slot",
        },
        OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT: {
            "prompt": "image_question_no_tool_schema_original_image_slot",
            "source_image": False,
            "oracle_target_transcript": False,
            "d": "cyclic_next_target_same_image_stage1_main_and_all_deepstack_in_original_image_slot",
        },
        OracleDUtilityArm.TARGET_ZERO_D_ONLY: {
            "prompt": "question_plus_oracle_target_tool_transcript",
            "source_image": False,
            "oracle_target_transcript": True,
            "d": "all_zero_main_and_all_deepstack",
        },
        OracleDUtilityArm.CORRECT_D_ONLY: {
            "prompt": "question_plus_oracle_target_tool_transcript",
            "source_image": False,
            "oracle_target_transcript": True,
            "d": "correct_target_stage1_main_and_all_deepstack",
        },
        OracleDUtilityArm.IMAGE_TARGET_ZERO_D: {
            "prompt": "image_question_plus_oracle_target_tool_transcript",
            "source_image": True,
            "oracle_target_transcript": True,
            "d": "all_zero_main_and_all_deepstack",
        },
        OracleDUtilityArm.IMAGE_CORRECT_D: {
            "prompt": "image_question_plus_oracle_target_tool_transcript",
            "source_image": True,
            "oracle_target_transcript": True,
            "d": "correct_target_stage1_main_and_all_deepstack",
        },
        OracleDUtilityArm.MATCHED_WRONG_D: {
            "prompt": "question_plus_oracle_target_tool_transcript",
            "source_image": False,
            "oracle_target_transcript": True,
            "d": "cyclic_next_target_same_image_stage1_main_and_all_deepstack",
        },
    }[arm]


class _OracleRunLedger:
    """Atomic per-arm records with a reconstructable JSONL convenience view."""

    def __init__(
        self,
        root: Path,
        *,
        identity_payload: Mapping[str, Any],
        expected_keys: tuple[tuple[str, str], ...],
    ) -> None:
        self.root = root
        self.records_dir = root / "records"
        self.identity_path = root / "identity.json"
        self.jsonl_path = root / "records.jsonl"
        self.progress_path = root / "progress.json"
        self.summary_path = root / "summary.json"
        self.lock_path = root / "run.lock"
        self.identity_payload = dict(identity_payload)
        self.identity_sha256 = _canonical_sha256(self.identity_payload)
        self.expected_keys = expected_keys
        if len(set(expected_keys)) != len(expected_keys):
            raise ValueError("oracle ledger expected keys must be unique")
        self._completed: dict[tuple[str, str], dict[str, Any]] = {}

    @contextmanager
    def locked(self) -> Iterator[None]:
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"oracle output root is already active: {self.root}"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def prepare(self) -> None:
        self.records_dir.mkdir(parents=True, exist_ok=True)
        declared = {
            "schema_version": ORACLE_D_UTILITY_SCHEMA_VERSION,
            "identity_sha256": self.identity_sha256,
            "identity": self.identity_payload,
        }
        if self.identity_path.exists():
            observed = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if observed != declared:
                raise ValueError(
                    "oracle output identity differs; choose another output root"
                )
        else:
            _atomic_write_json(self.identity_path, declared)
        expected = set(self.expected_keys)
        completed: dict[tuple[str, str], dict[str, Any]] = {}
        for path in sorted(self.records_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            key = (record.get("sample_id"), record.get("arm"))
            if record.get("run_identity_sha256") != self.identity_sha256:
                raise ValueError(f"oracle record has another run identity: {path}")
            if key not in expected or key in completed:
                raise ValueError(
                    f"oracle record key is unexpected or duplicate: {path}"
                )
            completed[key] = record
        self._completed = completed
        self._rebuild_jsonl()
        self._write_progress()

    @property
    def complete(self) -> bool:
        return len(self._completed) == len(self.expected_keys)

    def has(self, sample_id: str, arm: str) -> bool:
        return (sample_id, arm) in self._completed

    def commit(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        key = (payload.get("sample_id"), payload.get("arm"))
        if key not in set(self.expected_keys):
            raise ValueError("oracle record key was not declared")
        if key in self._completed:
            raise FileExistsError("oracle record was already committed")
        filename = sha256(f"{key[0]}\0{key[1]}".encode("utf-8")).hexdigest() + ".json"
        path = self.records_dir / filename
        _atomic_write_json(path, payload)
        self._completed[key] = payload
        with self.jsonl_path.open("ab") as handle:
            handle.write(_canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._write_progress()

    def summary(self) -> dict[str, object]:
        if not self.complete:
            raise RuntimeError("cannot summarize an incomplete oracle run")
        records = tuple(self._completed[key] for key in self.expected_keys)
        by_arm: dict[str, dict[str, object]] = {}
        for arm in self.identity_payload["arms"]:
            arm_records = tuple(record for record in records if record["arm"] == arm)
            correct = sum(record["score"]["correct"] is True for record in arm_records)
            incorrect = sum(
                record["score"]["correct"] is False for record in arm_records
            )
            unresolved = sum(
                record["score"]["correct"] is None for record in arm_records
            )
            routes: dict[str, int] = {}
            for record in arm_records:
                route = record["score"]["route"]
                routes[route] = routes.get(route, 0) + 1
            by_arm[arm] = {
                "correct": correct,
                "incorrect": incorrect,
                "unresolved": unresolved,
                "total": len(arm_records),
                "strict_lower_bound_accuracy": correct / len(arm_records),
                "formal_accuracy": (
                    correct / len(arm_records) if unresolved == 0 else None
                ),
                "score_routes": dict(sorted(routes.items())),
            }
        paired = {}
        for name, treatment, control in (
            (
                "direct_D_replacement_content_effect",
                OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT.value,
                OracleDUtilityArm.DIRECT_ZERO_D_REPLACEMENT.value,
            ),
            (
                "direct_D_replacement_specificity",
                OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT.value,
                OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT.value,
            ),
            (
                "direct_D_replacement_vs_native_image",
                OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT.value,
                OracleDUtilityArm.IMAGE_ONLY.value,
            ),
            (
                "D_only_content_effect",
                OracleDUtilityArm.CORRECT_D_ONLY.value,
                OracleDUtilityArm.TARGET_ZERO_D_ONLY.value,
            ),
            (
                "image_plus_D_content_effect",
                OracleDUtilityArm.IMAGE_CORRECT_D.value,
                OracleDUtilityArm.IMAGE_TARGET_ZERO_D.value,
            ),
        ):
            if treatment in by_arm and control in by_arm:
                paired[name] = _paired_summary(records, treatment, control)
        payload: dict[str, object] = {
            "schema_version": ORACLE_D_UTILITY_SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "run_identity_sha256": self.identity_sha256,
            "sample_count": len({key[0] for key in self.expected_keys}),
            "record_count": len(records),
            "arms": by_arm,
            "paired_effects": paired,
            "records_jsonl": str(self.jsonl_path),
            "records_jsonl_sha256": sha256(self.jsonl_path.read_bytes()).hexdigest(),
            "interpretation": (
                "Correct-vs-zero paired effects isolate D content conditional on an "
                "oracle trajectory target. They do not measure autonomous target/tool selection."
            ),
        }
        parity_path = self.root / "image_only_native_parity.json"
        payload["image_only_native_parity"] = (
            json.loads(parity_path.read_text(encoding="utf-8"))
            if parity_path.exists()
            else None
        )
        _atomic_write_json(self.summary_path, payload)
        return payload

    def _rebuild_jsonl(self) -> None:
        payload = b"".join(
            _canonical_json_bytes(self._completed[key]) + b"\n"
            for key in self.expected_keys
            if key in self._completed
        )
        _atomic_write_bytes(self.jsonl_path, payload)

    def _write_progress(self) -> None:
        completed = len(self._completed)
        total = len(self.expected_keys)
        _atomic_write_json(
            self.progress_path,
            {
                "schema_version": "representation_oracle_d_utility_progress_v1",
                "run_identity_sha256": self.identity_sha256,
                "completed_records": completed,
                "total_records": total,
                "fraction": completed / total,
                "status": "complete" if completed == total else "running_or_resumable",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )


def _paired_summary(
    records: Sequence[Mapping[str, Any]], treatment: str, control: str
) -> dict[str, object]:
    truth = {
        (record["sample_id"], record["arm"]): record["score"]["correct"]
        for record in records
    }
    sample_ids = sorted(
        sample_id
        for sample_id, arm in truth
        if arm == treatment and (sample_id, control) in truth
    )
    resolved = tuple(
        sample_id
        for sample_id in sample_ids
        if truth[(sample_id, treatment)] is not None
        and truth[(sample_id, control)] is not None
    )
    wins = sum(
        truth[(sample_id, treatment)] is True and truth[(sample_id, control)] is False
        for sample_id in resolved
    )
    losses = sum(
        truth[(sample_id, treatment)] is False and truth[(sample_id, control)] is True
        for sample_id in resolved
    )
    treatment_lower_bound = sum(
        truth[(sample_id, treatment)] is True for sample_id in sample_ids
    ) / len(sample_ids)
    control_lower_bound = sum(
        truth[(sample_id, control)] is True for sample_id in sample_ids
    ) / len(sample_ids)
    return {
        "treatment": treatment,
        "control": control,
        "paired_samples": len(sample_ids),
        "resolved_pairs": len(resolved),
        "unresolved_pairs": len(sample_ids) - len(resolved),
        "treatment_strict_lower_bound_accuracy": treatment_lower_bound,
        "control_strict_lower_bound_accuracy": control_lower_bound,
        "strict_lower_bound_accuracy_delta": (
            treatment_lower_bound - control_lower_bound
        ),
        "formal_accuracy_delta": (
            treatment_lower_bound - control_lower_bound
            if len(resolved) == len(sample_ids)
            else None
        ),
        "wins": wins,
        "losses": losses,
        "ties_among_resolved": len(resolved) - wins - losses,
    }


def _require_single_gpu_environment() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    required = {
        "CUBLAS_WORKSPACE_CONFIG": _REQUIRED_CUBLAS_WORKSPACE,
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    mismatches = {
        name: (expected, os.environ.get(name))
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if not visible or "," in visible:
        mismatches["CUDA_VISIBLE_DEVICES"] = ("one physical GPU ID", visible)
    if mismatches:
        raise ValueError(f"oracle D utility launch environment mismatch: {mismatches}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("oracle D utility requires exactly one visible CUDA GPU")


def _normalize_eos_token_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("eos_token_ids must be a sequence of integers")
    result = tuple(values)
    if (
        not result
        or len(set(result)) != len(result)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in result
        )
    ):
        raise ValueError("eos_token_ids must be unique non-negative integers")
    return result


def _model_eos_token_ids(model: torch.nn.Module) -> tuple[int, ...]:
    generation_config = getattr(model, "generation_config", None)
    value = getattr(generation_config, "eos_token_id", None)
    if isinstance(value, int) and not isinstance(value, bool):
        result = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(value)
    else:
        raise ValueError("local Thinking model has no explicit generation EOS IDs")
    return _normalize_eos_token_ids(result)


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


def _choice_label_for_expected(
    truth: OracleDUtilityGroundTruth,
) -> str | None:
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


def _greedy_token(logits: torch.Tensor) -> int:
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("oracle generation produced invalid logits")
    return int(torch.argmax(logits).item())


def _qwen3_multimodal_token_ids(
    runtime: Qwen3RepresentationRuntime,
) -> frozenset[int]:
    ids: list[int] = []
    for token in (
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ):
        token_id = runtime.tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"Qwen3 control token {token!r} has no integer ID")
        if runtime.tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(f"Qwen3 control token {token!r} does not round trip")
        ids.append(token_id)
    return frozenset(ids)


def _require_model_input(value: OracleDUtilityModelInput) -> None:
    if not isinstance(value, OracleDUtilityModelInput):
        raise TypeError("prompt construction requires OracleDUtilityModelInput")


def _integer_sequence_sha256(values: Sequence[int]) -> str:
    return sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {result.stderr.strip()}")
    return result.stdout.strip()


__all__ = [
    "DEFAULT_ORACLE_D_UTILITY_ARMS",
    "DEFAULT_THINKING_EOS_TOKEN_IDS",
    "ORACLE_D_UTILITY_SCHEMA_VERSION",
    "OracleAnswerScore",
    "OracleBatchCompatibilityError",
    "OracleDUtilityArm",
    "OracleDUtilityGroundTruth",
    "OracleDUtilityModelInput",
    "OracleGeneratedAnswer",
    "OracleImageOnlyParity",
    "build_image_only_messages",
    "build_oracle_target_messages",
    "greedy_oracle_answer",
    "greedy_oracle_answers_batched",
    "materialize_oracle_group_visuals",
    "prepare_oracle_arm_context",
    "run_oracle_d_utility_evaluation",
    "score_oracle_generated_answer",
    "split_oracle_d_utility_sample",
    "verify_image_only_injected_native_parity",
]
