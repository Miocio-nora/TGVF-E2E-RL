"""Native-Qwen representation transcript, ``Hq``, and readout construction.

The representation trainer consumes protocol-neutral rows.  This module is
the single executable boundary that turns those rows into the exact native
Qwen action transcript, extracts the tool-call target span, builds either
accepted target-conditioning representation, runs the TGVF Adapter, and
constructs the two-visual-block causal evidence readout.

No prompt wording is defaulted here.  A run must supply an exact, hashed prompt
template.  The second image passed to the processor is only a geometry carrier:
the resulting visual positions are replaced by the live TGVF Adapter tensors
before the frozen language-model readout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from string import Formatter
from typing import Any

import torch
from torch import nn

from tgvf_rl.conditioning import (
    TargetConditioningProviderKind,
    TargetConditioningRequest,
)
from tgvf_rl.contracts.tokens import TokenSpan
from tgvf_rl.protocol.native import RenderedTranscript
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import (
    SampledAssistantTurn,
    TGVF_FOCUS_TOOL_NAME,
    TokenByteSpan,
)
from tgvf_rl.qwen.base import QwenVLMFamilyAdapter, resolve_language_model
from tgvf_rl.representation.adapter import (
    TGVFAdapter,
    TGVFAdapterInput,
    TGVFAdapterOutput,
)

from .readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
)
from .runtime import (
    Qwen3ContextualHiddenStateStack,
    Qwen3RepresentationRuntime,
    Qwen3VisionFeatures,
    Qwen3VisionPreMergeRequest,
)
from .schema import RepresentationTrainingSample
from .transcript import (
    CanonicalToModelTokenExpansion,
    _build_visual_token_expansion,
    render_native_evidence_labels,
)


REPRESENTATION_PROMPT_SCHEMA_VERSION = "native_representation_prompt_v1"
NATIVE_ACTION_TARGET_SCHEMA_VERSION = "native_action_target_v1"
_ACTION_TEMPLATE_SUFFIX = "<|im_end|>\n"
_ALLOWED_PROMPT_FIELDS = frozenset({"question", "target"})


@dataclass(frozen=True, slots=True)
class RepresentationPromptConfig:
    """Exact late-bound prompt wording used by one representation run."""

    identity: str
    template: str
    expected_sha256: str
    schema_version: str = REPRESENTATION_PROMPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty_text(self.identity, field_name="prompt identity")
        _require_non_empty_text(self.template, field_name="prompt template")
        _require_sha256(self.expected_sha256, field_name="prompt expected_sha256")
        if self.schema_version != REPRESENTATION_PROMPT_SCHEMA_VERSION:
            raise ValueError("representation prompt schema mismatch")
        if self.sha256 != self.expected_sha256:
            raise ValueError("representation prompt template SHA256 mismatch")

        parsed = tuple(Formatter().parse(self.template))
        fields = tuple(field for _, field, _, _ in parsed if field is not None)
        if not fields or "question" not in fields:
            raise ValueError("representation prompt must reference {question}")
        if any(field not in _ALLOWED_PROMPT_FIELDS for field in fields):
            raise ValueError(
                "representation prompt fields are limited to {question} and {target}"
            )
        if any(conversion or format_spec for _, _, format_spec, conversion in parsed):
            raise ValueError("representation prompt conversions/format specs are forbidden")

    @property
    def sha256(self) -> str:
        return sha256(self.template.encode("utf-8")).hexdigest()

    def render(self, sample: RepresentationTrainingSample) -> str:
        if not isinstance(sample, RepresentationTrainingSample):
            raise TypeError("prompt sample must be RepresentationTrainingSample")
        rendered = self.template.format(
            question=sample.question,
            target=sample.target,
        )
        _require_non_empty_text(rendered, field_name="rendered representation prompt")
        return rendered


@dataclass(frozen=True, slots=True)
class NativeActionTarget:
    """Exact native action transcript and target span before visual expansion."""

    transcript: RenderedTranscript
    generation_prefill: RenderedTranscript
    sampled_turn: SampledAssistantTurn
    canonical_target_span: TokenSpan
    canonical_target_token_ids: tuple[int, ...]
    target_text: str
    schema_version: str = NATIVE_ACTION_TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_ACTION_TARGET_SCHEMA_VERSION:
            raise ValueError("native action-target schema mismatch")
        if self.transcript.chat_template_sha256 != (
            self.generation_prefill.chat_template_sha256
        ):
            raise ValueError("action transcript and prefill template identities differ")
        if self.transcript.tool_schema_sha256 != (
            self.generation_prefill.tool_schema_sha256
        ):
            raise ValueError("action transcript and prefill tool identities differ")
        if self.canonical_target_span.end > len(self.transcript.token_ids):
            raise ValueError("canonical target span lies outside action transcript")
        realized = self.transcript.token_ids[
            self.canonical_target_span.start : self.canonical_target_span.end
        ]
        if realized != self.canonical_target_token_ids:
            raise ValueError("canonical target IDs differ from action transcript")
        _require_non_empty_text(self.target_text, field_name="native target text")


@dataclass(frozen=True, slots=True)
class ModelActionTarget:
    """Target span after Qwen processor visual-token expansion."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    target_span: TokenSpan
    target_token_ids: tuple[int, ...]
    canonical: NativeActionTarget

    def __post_init__(self) -> None:
        _validate_single_input(self.input_ids, self.attention_mask)
        if not isinstance(self.pixel_values, torch.Tensor) or (
            self.pixel_values.ndim != 2 or not self.pixel_values.is_floating_point()
        ):
            raise ValueError("action pixel_values must be floating [N,P]")
        if not isinstance(self.image_grid_thw, torch.Tensor) or (
            self.image_grid_thw.shape != (1, 3)
        ):
            raise ValueError("native action requires exactly one image grid")
        if self.target_span.end > self.input_ids.shape[1]:
            raise ValueError("model target span lies outside input_ids")
        realized = tuple(
            int(value)
            for value in self.input_ids[
                0, self.target_span.start : self.target_span.end
            ].tolist()
        )
        if realized != self.target_token_ids:
            raise ValueError("model target IDs differ from expanded input_ids")
        if realized != self.canonical.canonical_target_token_ids:
            raise ValueError("Qwen visual expansion changed target token IDs")


def build_native_representation_messages(
    sample: RepresentationTrainingSample,
    prompt: RepresentationPromptConfig,
) -> tuple[dict[str, Any], ...]:
    """Construct the fixed four-turn representation transcript structure."""

    if not isinstance(sample, RepresentationTrainingSample):
        raise TypeError("sample must be RepresentationTrainingSample")
    if not isinstance(prompt, RepresentationPromptConfig):
        raise TypeError("prompt must be RepresentationPromptConfig")
    return (
        {
            "role": "user",
            "content": (
                {"type": "image"},
                {"type": "text", "text": prompt.render(sample)},
            ),
        },
        {
            "role": "assistant",
            "reasoning_content": "",
            "content": "",
            "tool_calls": (
                {
                    "type": "function",
                    "function": {
                        "name": TGVF_FOCUS_TOOL_NAME,
                        "arguments": {"target": sample.target},
                    },
                },
            ),
        },
        {"role": "tool", "content": ({"type": "image"},)},
        {
            "role": "assistant",
            "reasoning_content": sample.evidence_description,
            "content": "",
        },
    )


def render_native_action_target(
    runtime: Qwen3RepresentationRuntime,
    messages: Sequence[Mapping[str, Any]],
) -> NativeActionTarget:
    """Render and parse the exact policy-owned native tool-call completion."""

    if not isinstance(runtime, Qwen3RepresentationRuntime):
        raise TypeError("runtime must be Qwen3RepresentationRuntime")
    if len(messages) != 4:
        raise ValueError("native representation action requires four-turn messages")
    prefill = runtime.renderer.render(messages[:1], add_generation_prompt=True)
    runtime.renderer.assert_generation_prefill(prefill, runtime.tokenizer)
    transcript = runtime.renderer.render(messages[:2], add_generation_prompt=False)
    if not transcript.text.startswith(prefill.text) or not transcript.text.endswith(
        _ACTION_TEMPLATE_SUFFIX
    ):
        raise ValueError("native action transcript differs from generation prefill")
    sampled_text = transcript.text[
        len(prefill.text) : -len(_ACTION_TEMPLATE_SUFFIX)
    ]
    if not sampled_text:
        raise ValueError("native action completion is empty")

    sampled_ids, sampled_offsets = _tokenize_with_offsets(
        runtime.tokenizer, sampled_text
    )
    sampled_turn = SampledAssistantTurn(
        sampled_text=sampled_text,
        token_ids=sampled_ids,
        token_byte_spans=_byte_spans(sampled_text, sampled_ids, sampled_offsets),
    )
    parsed = StrictToolCallParser().parse(sampled_turn)

    full_ids, full_offsets = _tokenize_with_offsets(runtime.tokenizer, transcript.text)
    if full_ids != transcript.token_ids:
        raise ValueError("action offset tokenization differs from rendered IDs")
    full_target_start = len(prefill.text) + parsed.target_span.offsets.char_start
    full_target_end = len(prefill.text) + parsed.target_span.offsets.char_end
    positions = _owned_token_positions(
        transcript.text,
        full_offsets,
        span_start=full_target_start,
        span_end=full_target_end,
        name="native action target",
    )
    canonical_span = TokenSpan(positions[0], positions[-1] + 1)
    canonical_ids = transcript.token_ids[canonical_span.start : canonical_span.end]
    if canonical_ids != parsed.target_span.token_ids:
        raise ValueError("full-transcript target IDs differ from strict tool parser")
    if parsed.target != messages[1]["tool_calls"][0]["function"]["arguments"]["target"]:
        raise ValueError("rendered native target differs from structured message")
    return NativeActionTarget(
        transcript=transcript,
        generation_prefill=prefill,
        sampled_turn=sampled_turn,
        canonical_target_span=canonical_span,
        canonical_target_token_ids=canonical_ids,
        target_text=parsed.target,
    )


class Qwen3NativeRepresentationGroupBuilder:
    """Build differentiable same-image groups from real Qwen3 native state."""

    def __init__(
        self,
        *,
        runtime: Qwen3RepresentationRuntime,
        family_adapter: QwenVLMFamilyAdapter,
        prompt: RepresentationPromptConfig,
        image_loader: Callable[[str], Any],
    ) -> None:
        if not isinstance(runtime, Qwen3RepresentationRuntime):
            raise TypeError("runtime must be Qwen3RepresentationRuntime")
        # Keep this concrete check lazy: Qwen3VLAdapter imports the public
        # representation-training package for transcript types, so importing it
        # at module scope would create a package-initialization cycle.
        from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter

        if not isinstance(family_adapter, Qwen3VLAdapter):
            raise TypeError("native Qwen3 representation requires Qwen3VLAdapter")
        if not isinstance(prompt, RepresentationPromptConfig):
            raise TypeError("prompt must be RepresentationPromptConfig")
        if not callable(image_loader):
            raise TypeError("image_loader must be callable")
        self.runtime = runtime
        self.family_adapter = family_adapter
        self.prompt = prompt
        self.image_loader = image_loader

    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        adapter: TGVFAdapter,
        *,
        collective_candidate_count: int,
    ) -> SameImageReadoutGroup:
        if len(samples) < 2 or not all(
            isinstance(sample, RepresentationTrainingSample) for sample in samples
        ):
            raise ValueError("native representation group requires K>=2 typed samples")
        if adapter is not self.runtime.adapter:
            raise ValueError("group builder must use the runtime-owned TGVF Adapter")
        collective_candidate_count = _plain_int(
            collective_candidate_count,
            name="collective_candidate_count",
        )
        if collective_candidate_count < len(samples):
            raise ValueError("collective candidate count cannot be smaller than real K")
        group_keys = {sample.image_group_key for sample in samples}
        image_paths = {sample.image for sample in samples}
        if len(group_keys) != 1 or len(image_paths) != 1:
            raise ValueError("one Matrix-CE group must share one exact source image")
        if len({sample.target for sample in samples}) != len(samples):
            raise ValueError("same-image Matrix CE requires distinct targets")

        self.runtime.assert_bound_invariants()
        image_path = samples[0].image
        image = self.image_loader(image_path)
        if image is None:
            raise ValueError("image_loader returned None")

        messages_by_sample = tuple(
            build_native_representation_messages(sample, self.prompt)
            for sample in samples
        )
        action_by_sample = tuple(
            render_native_action_target(self.runtime, messages)
            for messages in messages_by_sample
        )
        model_actions = tuple(
            self._materialize_action(action, image)
            for action in action_by_sample
        )

        first_action = model_actions[0]
        vision = self.runtime.extract_vision_features(
            Qwen3VisionPreMergeRequest(
                pixel_values=first_action.pixel_values,
                image_grid_thw=first_action.image_grid_thw,
            )
        )
        for action in model_actions[1:]:
            if not torch.equal(action.image_grid_thw, first_action.image_grid_thw):
                raise ValueError("same source image produced inconsistent Qwen grids")
            if action.pixel_values.shape != first_action.pixel_values.shape or not (
                torch.equal(action.pixel_values, first_action.pixel_values)
            ):
                raise ValueError("same source image produced inconsistent pixel tensors")

        source_identity = _source_visual_identity(
            image_path=image_path,
            vision=vision,
        )
        source_visual = _source_bundle(vision)
        rows: list[RepresentationReadoutRow] = []
        candidates: list[RepresentationCandidateObservation] = []
        padding_input: TGVFAdapterInput | None = None
        for sample, messages, model_action in zip(
            samples, messages_by_sample, model_actions, strict=True
        ):
            condition = self._condition(sample, model_action)
            adapter_input = self.runtime.make_adapter_input(condition, vision)
            if padding_input is None:
                padding_input = adapter_input
            output = adapter(adapter_input)
            candidate_visual = _adapter_output_bundle(output)
            candidates.append(
                RepresentationCandidateObservation(
                    sample_id=sample.sample_id,
                    image_group_key=sample.image_group_key,
                    source_visual_identity=source_identity,
                    target_conditioning_provider=self.runtime.conditioning_config.provider,
                    projection_identities=(
                        output.metadata.main_projection_identity,
                        *output.metadata.deepstack_projection_identities,
                    ),
                    visual=candidate_visual,
                )
            )
            rows.append(
                self._readout_row(
                    sample=sample,
                    messages=messages,
                    image=image,
                    vision=vision,
                    source_identity=source_identity,
                )
            )

        if padding_input is None:  # guarded by K>=2 above
            raise RuntimeError("native group did not retain a collective padding input")
        collective_padding = tuple(
            _adapter_output_bundle(adapter(padding_input))
            for _ in range(collective_candidate_count - len(samples))
        )
        group = SameImageReadoutGroup(
            image_group_key=samples[0].image_group_key,
            source_visual_identity=source_identity,
            source_visual=source_visual,
            rows=tuple(rows),
            candidates=tuple(candidates),
            collective_padding=collective_padding,
        )
        self.runtime.assert_bound_invariants()
        return group

    def _materialize_action(
        self, canonical: NativeActionTarget, image: Any
    ) -> ModelActionTarget:
        batch = _processor_batch(
            self.runtime.processor,
            text=canonical.transcript.text,
            images=(image,),
        )
        input_ids, attention_mask, pixel_values, grid = _move_processor_batch(
            self.runtime, batch
        )
        expansion = _qwen3_expansion(
            self.runtime,
            canonical.transcript.token_ids,
            input_ids,
        )
        target_positions: list[int] = []
        for canonical_position in range(
            canonical.canonical_target_span.start,
            canonical.canonical_target_span.end,
        ):
            mapped = expansion.canonical_to_model_positions[canonical_position]
            if len(mapped) != 1:
                raise ValueError("a native target token cannot be visually expanded")
            target_positions.append(mapped[0])
        if tuple(target_positions) != tuple(
            range(target_positions[0], target_positions[-1] + 1)
        ):
            raise ValueError("expanded target positions are not contiguous")
        return ModelActionTarget(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            target_span=TokenSpan(target_positions[0], target_positions[-1] + 1),
            target_token_ids=canonical.canonical_target_token_ids,
            canonical=canonical,
        )

    def _condition(
        self,
        sample: RepresentationTrainingSample,
        action: ModelActionTarget,
    ):
        request = TargetConditioningRequest(
            input_ids=action.input_ids[0],
            target_span=action.target_span,
            expected_target_token_ids=action.target_token_ids,
            trajectory_id=f"representation:{sample.sample_id}",
            call_index=0,
            model_identity=self.runtime.model_identity,
        )
        contextual = None
        if (
            self.runtime.conditioning_config.provider
            is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
        ):
            with torch.no_grad():
                output = self.runtime.model(
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
        return self.runtime.build_target_condition(
            request,
            contextual_hidden_states=contextual,
        )

    def _readout_row(
        self,
        *,
        sample: RepresentationTrainingSample,
        messages: Sequence[Mapping[str, Any]],
        image: Any,
        vision: Qwen3VisionFeatures,
        source_identity: str,
    ) -> RepresentationReadoutRow:
        canonical = render_native_evidence_labels(
            self.runtime.renderer,
            messages,
            evidence_description=sample.evidence_description,
        )
        batch = _processor_batch(
            self.runtime.processor,
            text=canonical.transcript.text,
            images=(image, image),
        )
        input_ids, attention_mask, _pixel_values, grid = _move_processor_batch(
            self.runtime, batch
        )
        if grid.shape != (2, 3) or not torch.equal(grid[0], grid[1]):
            raise ValueError("readout source and D placeholders must share one grid")
        expected_grid = torch.tensor(
            vision.image_grid_thw,
            dtype=grid.dtype,
            device=grid.device,
        )
        if not torch.equal(grid[0], expected_grid):
            raise ValueError("readout visual grid differs from source vision features")
        supervision = self.family_adapter.materialize_representation_supervision(
            self.runtime.model,
            self.runtime.tokenizer,
            canonical,
            input_ids,
        )
        blocks = supervision.visual_expansion_blocks
        expected_tokens = int(vision.merged_main.shape[-2])
        if tuple(map(len, blocks)) != (expected_tokens, expected_tokens):
            raise ValueError("readout visual blocks differ from source/D token counts")
        position_ids = _qwen3_position_ids(
            self.runtime.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=grid,
        )
        return RepresentationReadoutRow(
            sample_id=sample.sample_id,
            image_group_key=sample.image_group_key,
            source_visual_identity=source_identity,
            supervision=supervision,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            source_positions=blocks[0],
            d_positions=blocks[1],
        )


def _processor_batch(
    processor: Any,
    *,
    text: str,
    images: tuple[Any, ...],
) -> Mapping[str, torch.Tensor]:
    try:
        batch = processor(
            text=[text],
            images=list(images),
            padding=False,
            return_tensors="pt",
        )
    except TypeError as error:
        raise TypeError("Qwen processor rejected native text/image inputs") from error
    if not isinstance(batch, Mapping):
        raise TypeError("Qwen processor output must be a mapping")
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if not required.issubset(batch):
        raise ValueError(
            f"Qwen processor output is missing: {sorted(required - set(batch))}"
        )
    if any(not isinstance(batch[name], torch.Tensor) for name in required):
        raise TypeError("Qwen processor outputs must be torch tensors")
    return batch


def _move_processor_batch(
    runtime: Qwen3RepresentationRuntime,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding = resolve_language_model(runtime.model).get_input_embeddings()
    embedding_weight = getattr(embedding, "weight", None)
    if not isinstance(embedding_weight, torch.Tensor):
        raise TypeError("Qwen input embedding must expose a weight tensor")
    vision_owner = next(runtime.vision_tower.parameters(), None)
    if vision_owner is None:
        vision_owner = next(runtime.vision_tower.buffers(), None)
    if vision_owner is None:
        raise ValueError("Qwen vision tower has no device-owning tensor")
    if embedding_weight.device != vision_owner.device:
        raise ValueError(
            "representation runtime currently requires one model device per rank"
        )
    device = embedding_weight.device
    input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = batch["attention_mask"].to(device=device)
    pixel_values = batch["pixel_values"].to(
        device=device,
        dtype=vision_owner.dtype if vision_owner.dtype.is_floating_point else None,
    )
    grid = batch["image_grid_thw"].to(device=device, dtype=torch.long)
    _validate_single_input(input_ids, attention_mask)
    return input_ids, attention_mask, pixel_values, grid


def _qwen3_expansion(
    runtime: Qwen3RepresentationRuntime,
    canonical_ids: Sequence[int],
    model_input_ids: torch.Tensor,
) -> CanonicalToModelTokenExpansion:
    visual_id = runtime.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if isinstance(visual_id, bool) or not isinstance(visual_id, int):
        raise TypeError("Qwen image placeholder did not resolve to an integer ID")
    if runtime.tokenizer.convert_ids_to_tokens(visual_id) != "<|image_pad|>":
        raise ValueError("Qwen image placeholder token does not round trip")
    return _build_visual_token_expansion(
        family="qwen3_vl",
        canonical_token_ids=canonical_ids,
        model_token_ids=tuple(int(value) for value in model_input_ids[0].tolist()),
        visual_placeholder_token_id=visual_id,
    )


def _qwen3_position_ids(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    core = getattr(model, "model", None)
    get_rope_index = getattr(core, "get_rope_index", None)
    if not callable(get_rope_index):
        raise TypeError("Qwen3 model core must expose get_rope_index")
    with torch.no_grad():
        result = get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("Qwen3 get_rope_index must return positions and delta")
    position_ids, _rope_delta = result
    if not isinstance(position_ids, torch.Tensor) or (
        position_ids.ndim != 3 or position_ids.shape[-2:] != input_ids.shape
    ):
        raise ValueError("Qwen3 position IDs do not align with readout input")
    return position_ids.detach().clone()


def _source_bundle(vision: Qwen3VisionFeatures) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=_as_batched(vision.merged_main),
        deepstack=tuple(_as_batched(branch) for branch in vision.merged_deepstack),
        branch_layers=vision.branch_layers,
    )


def _as_batched(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor
    raise ValueError("representation visual tensor must be [N,H] or [1,N,H]")


def _adapter_output_bundle(
    output: TGVFAdapterOutput,
) -> RepresentationVisualTensorBundle:
    if not isinstance(output, TGVFAdapterOutput):
        raise TypeError("TGVF Adapter must return a TGVFAdapterOutput")
    return RepresentationVisualTensorBundle(
        main=_as_batched(output.main_d),
        deepstack=tuple(
            _as_batched(branch) for branch in output.deepstack_visual_embeds
        ),
        branch_layers=output.metadata.branch_layers,
    )


def _source_visual_identity(
    *,
    image_path: str,
    vision: Qwen3VisionFeatures,
) -> str:
    path = Path(image_path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("representation image path must resolve to a file")
    image_sha256 = sha256(path.read_bytes()).hexdigest()
    model = vision.model_identity
    payload = {
        "schema": "representation_source_visual_identity_v1",
        "image_sha256": image_sha256,
        "model": {
            "family": model.family,
            "name": model.model_name,
            "revision_or_path": model.revision_or_path,
            "tokenizer_length": model.tokenizer_length,
            "chat_template_sha256": model.chat_template_sha256,
        },
        "image_grid_thw": vision.image_grid_thw,
        "spatial_merge_size": vision.spatial_merge_size,
        "branch_layers": vision.branch_layers,
        "projection_identities": vision.projection_identities,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{sha256(raw).hexdigest()}"


def _tokenize_with_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    if getattr(tokenizer, "is_fast", None) is not True:
        raise TypeError("native target extraction requires a fast tokenizer")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer offset output must be a mapping")
    ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise TypeError("tokenizer input_ids offset output is invalid")
    if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)):
        raise TypeError("tokenizer offset_mapping output is invalid")
    token_ids = tuple(_plain_int(value, name="token ID") for value in ids)
    normalized_offsets = tuple(_offset(value) for value in offsets)
    if len(token_ids) != len(normalized_offsets):
        raise ValueError("token IDs and offsets do not align")
    return token_ids, normalized_offsets


def _byte_spans(
    text: str,
    token_ids: tuple[int, ...],
    offsets: tuple[tuple[int, int], ...],
) -> tuple[TokenByteSpan, ...]:
    spans: list[TokenByteSpan] = []
    byte_cursor = 0
    char_cursor = 0
    for index, (token_id, (start, end)) in enumerate(
        zip(token_ids, offsets, strict=True)
    ):
        if start != char_cursor or end <= start:
            raise ValueError("sampled action token offsets must exactly cover text")
        byte_start = len(text[:start].encode("utf-8"))
        byte_end = len(text[:end].encode("utf-8"))
        if byte_start != byte_cursor or byte_end <= byte_start:
            raise ValueError("sampled action byte offsets must be contiguous")
        spans.append(
            TokenByteSpan(
                token_index=index,
                token_id=token_id,
                byte_start=byte_start,
                byte_end=byte_end,
            )
        )
        char_cursor = end
        byte_cursor = byte_end
    if char_cursor != len(text) or byte_cursor != len(text.encode("utf-8")):
        raise ValueError("sampled action token offsets do not cover completion")
    return tuple(spans)


def _owned_token_positions(
    text: str,
    offsets: tuple[tuple[int, int], ...],
    *,
    span_start: int,
    span_end: int,
    name: str,
) -> tuple[int, ...]:
    if not (0 <= span_start < span_end <= len(text)):
        raise ValueError(f"{name} character span is invalid")
    positions: list[int] = []
    cursor = span_start
    for index, (start, end) in enumerate(offsets):
        overlaps = start < span_end and end > span_start
        if not overlaps:
            continue
        if start < span_start or end > span_end or end <= start:
            raise ValueError(f"a tokenizer token crosses the {name} boundary")
        if start != cursor:
            raise ValueError(f"{name} token offsets are not contiguous")
        positions.append(index)
        cursor = end
    if not positions or cursor != span_end:
        raise ValueError(f"{name} owns no exact contiguous token span")
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError(f"{name} token positions are not contiguous")
    return tuple(positions)


def _validate_single_input(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    if input_ids.dtype != torch.long or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("native representation input_ids must be long [1,S]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("native representation attention_mask must match input_ids")
    if bool((attention_mask == 0).any().item()):
        raise ValueError("unpadded native representation input cannot contain masked tokens")


def _offset(value: object) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("token offset must be a two-integer sequence")
    items = tuple(value)
    if len(items) != 2:
        raise ValueError("token offset must contain two integers")
    start = _plain_int(items[0], name="offset start")
    end = _plain_int(items[1], name="offset end")
    if start < 0 or end < start:
        raise ValueError("token offset is invalid")
    return start, end


def _plain_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _require_non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")


__all__ = [
    "NATIVE_ACTION_TARGET_SCHEMA_VERSION",
    "REPRESENTATION_PROMPT_SCHEMA_VERSION",
    "ModelActionTarget",
    "NativeActionTarget",
    "Qwen3NativeRepresentationGroupBuilder",
    "RepresentationPromptConfig",
    "build_native_representation_messages",
    "render_native_action_target",
]
