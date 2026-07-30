"""Native Qwen answer supervision for the isolated D-utility experiment.

The accepted representation transcript labels only teacher evidence.  This
module deliberately leaves that contract untouched and constructs a second,
typed supervision view whose labels own exactly the short answer and the final
native ``<|im_end|>`` token.  The clean view contains one focused-D block and
no source-image block; the gold-evidence view exists only as a leakage
diagnostic for E1/E3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from tgvf_rl.protocol.native import NativeAssistantDialect, RenderedTranscript
from tgvf_rl.qwen.base import InjectedForwardRequest, InjectedVisualBlock
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.native_pipeline import (
    RepresentationPromptConfig,
    _expand_native_visual_placeholders,
    _qwen3_position_ids,
    build_native_representation_messages,
)
from tgvf_rl.representation.training.qwen3_counterfactual import (
    build_qwen3_d_only_messages,
)
from tgvf_rl.representation.training.readout import (
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
)
from tgvf_rl.representation.training.runtime import Qwen3RepresentationRuntime
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.transcript import (
    _evidence_owned_token_positions,
    _tokenize_with_exact_offsets,
)


ANSWER_SUPERVISION_SCHEMA_VERSION = "answer_utility_native_supervision_v1"
AnswerContextKind = Literal["clean_d_only", "gold_evidence"]


@dataclass(frozen=True, slots=True)
class NativeAnswerSupervision:
    """One exact teacher-forced answer view with explicit visual provenance."""

    sample_id: str
    context_kind: AnswerContextKind
    transcript: RenderedTranscript
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: tuple[int, ...]
    answer_positions: tuple[int, ...]
    eos_positions: tuple[int, ...]
    source_positions: tuple[int, ...]
    d_positions: tuple[int, ...]
    answer_text: str
    evidence_field_injected: bool
    pre_d_text_kv_reused: bool = False
    cache_mode: str = "no_cache"
    schema_version: str = ANSWER_SUPERVISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("answer supervision sample_id must be non-empty")
        if self.context_kind not in {"clean_d_only", "gold_evidence"}:
            raise ValueError("unknown answer supervision context kind")
        if not isinstance(self.transcript, RenderedTranscript):
            raise TypeError("answer supervision requires a rendered transcript")
        if self.schema_version != ANSWER_SUPERVISION_SCHEMA_VERSION:
            raise ValueError("answer supervision schema mismatch")
        if self.input_ids.dtype != torch.long or self.input_ids.ndim != 2:
            raise ValueError("answer input_ids must be long [B,S]")
        if self.input_ids.shape[0] != 1 or self.input_ids.shape[1] < 2:
            raise ValueError("answer supervision requires one non-trivial row")
        if self.attention_mask.shape != self.input_ids.shape or not bool(
            self.attention_mask.bool().all().item()
        ):
            raise ValueError("answer attention mask must be aligned and all-one")
        if self.position_ids.shape != (3, 1, self.input_ids.shape[1]):
            raise ValueError("Qwen3 answer M-RoPE must have shape [3,1,S]")
        if len(self.labels) != self.input_ids.shape[1]:
            raise ValueError("answer labels must align with input_ids")
        if not self.answer_text.strip():
            raise ValueError("answer supervision answer_text must be non-empty")
        if not self.answer_positions or not self.eos_positions:
            raise ValueError("answer and native EOS must both receive supervision")
        owned = (*self.answer_positions, *self.eos_positions)
        if tuple(sorted(set(owned))) != tuple(sorted(owned)):
            raise ValueError("answer/EOS positions must be unique")
        if any(position <= 0 or position >= len(self.labels) for position in owned):
            raise ValueError("answer/EOS position is not causally scoreable")
        owned_set = set(owned)
        cpu_ids = tuple(int(token_id) for token_id in self.input_ids[0].tolist())
        for position, (label, token_id) in enumerate(
            zip(self.labels, cpu_ids, strict=True)
        ):
            expected = token_id if position in owned_set else EVIDENCE_IGNORE_INDEX
            if label != expected:
                raise ValueError("answer labels own tokens outside answer plus EOS")
        visual = (*self.source_positions, *self.d_positions)
        if not self.d_positions or len(set(visual)) != len(visual):
            raise ValueError("answer visual positions must be non-empty and disjoint")
        if any(position < 0 or position >= len(self.labels) for position in visual):
            raise ValueError("answer visual position lies outside the transcript")
        if owned_set.intersection(visual):
            raise ValueError("answer labels cannot overlap visual placeholders")
        if self.pre_d_text_kv_reused or self.cache_mode != "no_cache":
            raise ValueError("answer utility requires a fresh no-cache context")
        if self.context_kind == "clean_d_only":
            if self.source_positions or self.evidence_field_injected:
                raise ValueError("clean answer view cannot contain image/gold evidence")
        elif not self.source_positions or not self.evidence_field_injected:
            raise ValueError("gold-evidence diagnostic must declare its leakage")

    @property
    def supervised_positions(self) -> tuple[int, ...]:
        return tuple(sorted((*self.answer_positions, *self.eos_positions)))

    @property
    def supervised_token_count(self) -> int:
        return len(self.supervised_positions)

    def request(
        self,
        *,
        observation: RepresentationVisualTensorBundle,
        source: RepresentationVisualTensorBundle | None = None,
    ) -> InjectedForwardRequest:
        """Inject one complete main-D/DeepStack bundle into this fixed view."""

        _assert_bundle_positions(observation, self.d_positions, name="focused D")
        blocks: list[InjectedVisualBlock] = []
        if self.source_positions:
            if source is None:
                raise ValueError("gold-evidence answer view requires source visual")
            _assert_bundle_positions(source, self.source_positions, name="source image")
            blocks.append(_visual_block("source_image", self.source_positions, source))
        elif source is not None:
            raise ValueError("clean D-only answer view rejects a source image")
        blocks.append(_visual_block("focused_d", self.d_positions, observation))
        return InjectedForwardRequest(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            visual_blocks=tuple(blocks),
            use_cache=False,
        )


def build_qwen3_clean_answer_supervision(
    runtime: Qwen3RepresentationRuntime,
    sample: RepresentationTrainingSample,
    prompt: RepresentationPromptConfig,
    *,
    d_token_count: int,
    image_grid_thw: tuple[int, int, int],
    device: torch.device,
) -> NativeAnswerSupervision:
    """Render question + oracle target + D, with no source image or teacher evidence."""

    _validate_build_inputs(runtime, sample, prompt, d_token_count, image_grid_thw)
    history = build_qwen3_d_only_messages(
        sample,
        prompt,
        assistant_dialect=runtime.renderer.assistant_dialect,
    )
    final_turn = _clean_answer_turn(
        sample.short_answer,
        runtime.renderer.assistant_dialect,
    )
    prefix = runtime.renderer.render(history, add_generation_prompt=True)
    runtime.renderer.assert_generation_prefill(prefix, runtime.tokenizer)
    transcript = runtime.renderer.render(
        (*history, final_turn),
        add_generation_prompt=False,
    )
    return _materialize_qwen3_answer_supervision(
        runtime,
        sample=sample,
        transcript=transcript,
        canonical_prefix=prefix,
        visual_token_counts=(d_token_count,),
        image_grids=(image_grid_thw,),
        device=device,
        context_kind="clean_d_only",
        evidence_field_injected=False,
    )


def build_qwen3_gold_evidence_answer_supervision(
    runtime: Qwen3RepresentationRuntime,
    sample: RepresentationTrainingSample,
    prompt: RepresentationPromptConfig,
    *,
    row: RepresentationReadoutRow,
    image_grid_thw: tuple[int, int, int],
) -> NativeAnswerSupervision:
    """Label answer+EOS in the existing evidence transcript without mutating it."""

    if not isinstance(row, RepresentationReadoutRow):
        raise TypeError("gold-evidence answer view requires a representation row")
    d_token_count = len(row.d_positions)
    _validate_build_inputs(runtime, sample, prompt, d_token_count, image_grid_thw)
    messages = build_native_representation_messages(
        sample,
        prompt,
        assistant_dialect=runtime.renderer.assistant_dialect,
    )
    transcript = runtime.renderer.render(messages, add_generation_prompt=False)
    result = _materialize_qwen3_answer_supervision(
        runtime,
        sample=sample,
        transcript=transcript,
        canonical_prefix=None,
        visual_token_counts=(len(row.source_positions), d_token_count),
        image_grids=(image_grid_thw, image_grid_thw),
        device=row.input_ids.device,
        context_kind="gold_evidence",
        evidence_field_injected=True,
    )
    if tuple(int(value) for value in row.input_ids[0].tolist()) != tuple(
        int(value) for value in result.input_ids[0].tolist()
    ):
        raise ValueError("gold answer view differs from existing readout input IDs")
    if result.source_positions != row.source_positions or (
        result.d_positions != row.d_positions
    ):
        raise ValueError("gold answer view changed source/D placeholder order")
    if not torch.equal(result.position_ids, row.position_ids):
        raise ValueError("gold answer view changed existing Qwen M-RoPE positions")
    return result


def _materialize_qwen3_answer_supervision(
    runtime: Qwen3RepresentationRuntime,
    *,
    sample: RepresentationTrainingSample,
    transcript: RenderedTranscript,
    canonical_prefix: RenderedTranscript | None,
    visual_token_counts: tuple[int, ...],
    image_grids: tuple[tuple[int, int, int], ...],
    device: torch.device,
    context_kind: AnswerContextKind,
    evidence_field_injected: bool,
) -> NativeAnswerSupervision:
    model_token_ids, expansion = _expand_native_visual_placeholders(
        runtime,
        transcript.token_ids,
        visual_token_counts=visual_token_counts,
    )
    if canonical_prefix is not None:
        prefix_ids, _prefix_expansion = _expand_native_visual_placeholders(
            runtime,
            canonical_prefix.token_ids,
            visual_token_counts=visual_token_counts,
        )
        if model_token_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "clean answer transcript retokenized its generation prefix"
            )
    answer_canonical = _canonical_answer_positions(
        runtime,
        transcript,
        answer_text=sample.short_answer,
    )
    eos_canonical = _canonical_terminal_eos_positions(
        runtime,
        transcript,
        after_position=answer_canonical[-1],
    )
    answer_model = _map_single_token_positions(expansion, answer_canonical, "answer")
    eos_model = _map_single_token_positions(expansion, eos_canonical, "EOS")
    owned = set((*answer_model, *eos_model))
    labels = tuple(
        token_id if position in owned else EVIDENCE_IGNORE_INDEX
        for position, token_id in enumerate(model_token_ids)
    )
    cpu_input_ids = torch.tensor((model_token_ids,), dtype=torch.long)
    cpu_attention_mask = torch.ones_like(cpu_input_ids)
    grid = torch.tensor(image_grids, dtype=torch.long)
    cpu_position_ids = _qwen3_position_ids(
        runtime.model,
        input_ids=cpu_input_ids,
        attention_mask=cpu_attention_mask,
        image_grid_thw=grid,
    )
    blocks = _visual_expansion_blocks(expansion)
    if len(blocks) != len(visual_token_counts):
        raise ValueError("answer transcript has an unexpected visual block count")
    source_positions = () if context_kind == "clean_d_only" else blocks[0]
    d_positions = blocks[0] if context_kind == "clean_d_only" else blocks[1]
    runtime.renderer.assert_tokenizer_length()
    runtime.renderer.assert_chat_template_identity()
    return NativeAnswerSupervision(
        sample_id=sample.sample_id,
        context_kind=context_kind,
        transcript=transcript,
        input_ids=cpu_input_ids.to(device=device),
        attention_mask=cpu_attention_mask.to(device=device),
        position_ids=cpu_position_ids.to(device=device),
        labels=labels,
        answer_positions=answer_model,
        eos_positions=eos_model,
        source_positions=source_positions,
        d_positions=d_positions,
        answer_text=sample.short_answer,
        evidence_field_injected=evidence_field_injected,
    )


def _clean_answer_turn(
    answer: str,
    dialect: NativeAssistantDialect,
) -> dict[str, str]:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("clean answer must be non-empty")
    if dialect is NativeAssistantDialect.QWEN3_VL_THINKING:
        return {"role": "assistant", "reasoning_content": "", "content": answer}
    if dialect is NativeAssistantDialect.QWEN3_VL_INSTRUCT:
        return {"role": "assistant", "content": answer}
    raise ValueError("unsupported native assistant dialect")


def _canonical_answer_positions(
    runtime: Qwen3RepresentationRuntime,
    transcript: RenderedTranscript,
    *,
    answer_text: str,
) -> tuple[int, ...]:
    content = transcript.text.rstrip("\n")
    terminal = "<|im_end|>"
    if not content.endswith(terminal):
        raise ValueError("native answer transcript lacks its terminal control token")
    answer_end = len(content) - len(terminal)
    answer_start = answer_end - len(answer_text)
    if answer_start < 0 or content[answer_start:answer_end] != answer_text:
        raise ValueError("native transcript does not end in the exact short answer")
    token_ids, offsets = _tokenize_with_exact_offsets(runtime.renderer, transcript)
    if token_ids != transcript.token_ids:
        raise ValueError("answer offset tokenization differs from rendered IDs")
    return _evidence_owned_token_positions(
        transcript.text,
        offsets,
        evidence_start=answer_start,
        evidence_end=answer_end,
    )


def _canonical_terminal_eos_positions(
    runtime: Qwen3RepresentationRuntime,
    transcript: RenderedTranscript,
    *,
    after_position: int,
) -> tuple[int, ...]:
    eos_id = runtime.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(eos_id, bool) or not isinstance(eos_id, int):
        raise TypeError("Qwen native <|im_end|> must resolve to an integer ID")
    if runtime.tokenizer.convert_ids_to_tokens(eos_id) != "<|im_end|>":
        raise ValueError("Qwen native <|im_end|> token does not round trip")
    positions = tuple(
        index
        for index, token_id in enumerate(transcript.token_ids)
        if index > after_position and token_id == eos_id
    )
    if len(positions) != 1:
        raise ValueError("answer must be followed by exactly one terminal <|im_end|>")
    return positions


def _map_single_token_positions(expansion, positions: tuple[int, ...], name: str):
    mapped: list[int] = []
    for position in positions:
        values = expansion.canonical_to_model_positions[position]
        if len(values) != 1:
            raise ValueError(f"a supervised {name} token cannot visually expand")
        mapped.append(values[0])
    return tuple(mapped)


def _visual_expansion_blocks(expansion) -> tuple[tuple[int, ...], ...]:
    visual = set(expansion.visual_model_positions)
    return tuple(
        mapped
        for mapped in expansion.canonical_to_model_positions
        if mapped and set(mapped).issubset(visual)
    )


def _visual_block(
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


def _assert_bundle_positions(
    visual: RepresentationVisualTensorBundle,
    positions: tuple[int, ...],
    *,
    name: str,
) -> None:
    if not isinstance(visual, RepresentationVisualTensorBundle):
        raise TypeError(f"{name} must be a representation visual bundle")
    if len(positions) != visual.main.shape[1]:
        raise ValueError(f"{name} token count differs from transcript positions")
    if len(visual.deepstack) != 3 or visual.branch_layers != (8, 16, 24):
        raise ValueError(f"{name} must atomically contain DeepStack 8/16/24")


def _validate_build_inputs(
    runtime: Qwen3RepresentationRuntime,
    sample: RepresentationTrainingSample,
    prompt: RepresentationPromptConfig,
    d_token_count: int,
    image_grid_thw: tuple[int, int, int],
) -> None:
    if not isinstance(runtime, Qwen3RepresentationRuntime):
        raise TypeError("answer supervision requires Qwen3RepresentationRuntime")
    if not isinstance(sample, RepresentationTrainingSample):
        raise TypeError("answer supervision requires a typed sample")
    if not isinstance(prompt, RepresentationPromptConfig):
        raise TypeError("answer supervision requires an explicit prompt")
    if sample.short_answer != sample.short_answer.strip():
        raise ValueError("short answer cannot contain surrounding whitespace")
    forbidden_fragments = (
        "<|im_start|>",
        "<|im_end|>",
        "<|image_pad|>",
        "<tool_call>",
        "</tool_call>",
    )
    if any(fragment in sample.short_answer for fragment in forbidden_fragments):
        raise ValueError("short answer contains a native protocol control fragment")
    if isinstance(d_token_count, bool) or not isinstance(d_token_count, int):
        raise TypeError("D token count must be an integer")
    if d_token_count <= 0:
        raise ValueError("D token count must be positive")
    if (
        not isinstance(image_grid_thw, tuple)
        or len(image_grid_thw) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in image_grid_thw
        )
    ):
        raise ValueError("image_grid_thw must contain three positive integers")
    expected = (
        image_grid_thw[0]
        * image_grid_thw[1]
        * image_grid_thw[2]
        // (runtime.architecture.spatial_merge_size**2)
    )
    if expected != d_token_count:
        raise ValueError("D token count differs from exact Qwen image grid")


__all__ = [
    "ANSWER_SUPERVISION_SCHEMA_VERSION",
    "NativeAnswerSupervision",
    "build_qwen3_clean_answer_supervision",
    "build_qwen3_gold_evidence_answer_supervision",
]
