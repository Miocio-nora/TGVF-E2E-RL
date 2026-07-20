"""Exact native-Qwen evidence supervision for the representation phase."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tgvf_rl.protocol.native import NativeProtocolRenderer, RenderedTranscript
from tgvf_rl.protocol.schema import TGVF_FOCUS_TOOL_NAME

from .losses import EVIDENCE_IGNORE_INDEX


CANONICAL_EVIDENCE_SCHEMA_VERSION = "canonical_evidence_supervision_v1"
MODEL_EVIDENCE_SCHEMA_VERSION = "model_evidence_supervision_v1"
TOKEN_EXPANSION_SCHEMA_VERSION = "canonical_to_model_token_expansion_v1"
_EXECUTABLE_REPRESENTATION_FAMILY = "qwen3_vl"

NATIVE_REPRESENTATION_PRE_REASONING = "I need visual focus before answering."
_QWEN_THINKING_COMPLETION_MIDDLE = "\n</think>\n\n"
_QWEN_ASSISTANT_TURN_SUFFIX = "<|im_end|>\n"
_NATIVE_CONTROL_FRAGMENTS = (
    "<|im_",
    "<|vision_",
    "<|image_pad|>",
    "<|video_pad|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)


@dataclass(frozen=True, slots=True)
class CanonicalEvidenceSupervision:
    """Evidence labels over canonical chat-template token positions.

    This is not yet a model-input label tensor. Qwen processors expand each
    canonical image placeholder into a run of visual positions, so callers must
    materialize :class:`ModelEvidenceSupervision` through an explicit family
    token-expansion map before computing a loss.
    """

    transcript: RenderedTranscript
    generation_prefill: RenderedTranscript
    evidence_text: str
    answer_text: str
    canonical_labels: tuple[int, ...]
    evidence_char_start: int
    evidence_char_end: int
    evidence_byte_start: int
    evidence_byte_end: int
    evidence_token_positions: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]
    schema_version: str = CANONICAL_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CANONICAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("canonical evidence-supervision schema mismatch")
        _require_non_empty_text(self.answer_text, field_name="canonical answer_text")
        if len(self.canonical_labels) != len(self.transcript.token_ids):
            raise ValueError("canonical labels must align with transcript token ids")
        if len(self.token_offsets) != len(self.transcript.token_ids):
            raise ValueError(
                "canonical token offsets must align with transcript token ids"
            )
        if not (
            0
            <= self.evidence_char_start
            < self.evidence_char_end
            <= len(self.transcript.text)
        ):
            raise ValueError("canonical evidence character span is invalid")
        encoded = self.transcript.text.encode("utf-8")
        if not (0 <= self.evidence_byte_start < self.evidence_byte_end <= len(encoded)):
            raise ValueError("canonical evidence byte span is invalid")
        if (
            self.transcript.text[self.evidence_char_start : self.evidence_char_end]
            != self.evidence_text
        ):
            raise ValueError("canonical evidence character span differs from evidence")
        if (
            encoded[self.evidence_byte_start : self.evidence_byte_end].decode("utf-8")
            != self.evidence_text
        ):
            raise ValueError("canonical evidence byte span differs from evidence")
        if (
            self.transcript.chat_template_sha256
            != self.generation_prefill.chat_template_sha256
            or self.transcript.tool_schema_sha256
            != self.generation_prefill.tool_schema_sha256
            or self.transcript.tokenizer_length
            != self.generation_prefill.tokenizer_length
        ):
            raise ValueError(
                "canonical transcript and generation prefill identities differ"
            )
        if self.evidence_char_start != len(self.generation_prefill.text) or (
            self.transcript.text
            != self.generation_prefill.text
            + self.evidence_text
            + _QWEN_THINKING_COMPLETION_MIDDLE
            + self.answer_text
            + _QWEN_ASSISTANT_TURN_SUFFIX
        ):
            raise ValueError("canonical evidence placement differs from native prefill")
        derived_positions = _evidence_owned_token_positions(
            self.transcript.text,
            self.token_offsets,
            evidence_start=self.evidence_char_start,
            evidence_end=self.evidence_char_end,
        )
        if derived_positions != self.evidence_token_positions:
            raise ValueError(
                "canonical evidence positions differ from tokenizer offsets"
            )
        _validate_label_ownership(
            self.canonical_labels,
            self.transcript.token_ids,
            self.evidence_token_positions,
            name="canonical evidence",
        )

    @property
    def evidence_token_count(self) -> int:
        return len(self.evidence_token_positions)


@dataclass(frozen=True, slots=True)
class CanonicalToModelTokenExpansion:
    """One family-produced map from canonical to expanded model positions."""

    family: str
    canonical_token_ids: tuple[int, ...]
    model_token_ids: tuple[int, ...]
    canonical_to_model_positions: tuple[tuple[int, ...], ...]
    visual_model_positions: tuple[int, ...]
    schema_version: str = TOKEN_EXPANSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty_text(self.family, field_name="family")
        if self.family != _EXECUTABLE_REPRESENTATION_FAMILY:
            raise ValueError(
                f"{self.family} has no accepted representation token-expansion contract"
            )
        if self.schema_version != TOKEN_EXPANSION_SCHEMA_VERSION:
            raise ValueError("canonical-to-model token expansion schema mismatch")
        if not self.canonical_token_ids or not self.model_token_ids:
            raise ValueError("token expansion requires non-empty token sequences")
        if len(self.canonical_to_model_positions) != len(self.canonical_token_ids):
            raise ValueError("token expansion must map every canonical token")
        flattened: list[int] = []
        for canonical_id, model_positions in zip(
            self.canonical_token_ids,
            self.canonical_to_model_positions,
            strict=True,
        ):
            if not model_positions:
                raise ValueError("a canonical token cannot disappear during expansion")
            if tuple(range(model_positions[0], model_positions[-1] + 1)) != (
                model_positions
            ):
                raise ValueError("each canonical token must map to a contiguous block")
            for position in model_positions:
                _validate_position(position, len(self.model_token_ids), name="model")
                if self.model_token_ids[position] != canonical_id:
                    raise ValueError(
                        "expanded model token ids differ from canonical ids"
                    )
            flattened.extend(model_positions)
        if tuple(flattened) != tuple(range(len(self.model_token_ids))):
            raise ValueError(
                "token expansion must cover model positions once and in order"
            )

        if (
            tuple(sorted(set(self.visual_model_positions)))
            != self.visual_model_positions
        ):
            raise ValueError("visual model positions must be unique and ordered")
        for position in self.visual_model_positions:
            _validate_position(position, len(self.model_token_ids), name="visual model")
        visual = set(self.visual_model_positions)
        for model_positions in self.canonical_to_model_positions:
            if len(model_positions) > 1 and not set(model_positions).issubset(visual):
                raise ValueError(
                    "only visual placeholders may expand to multiple positions"
                )


@dataclass(frozen=True, slots=True)
class ModelEvidenceSupervision:
    """Final evidence labels aligned to one expanded model-input sequence."""

    family: str
    model_token_ids: tuple[int, ...]
    labels: tuple[int, ...]
    evidence_token_positions: tuple[int, ...]
    visual_model_positions: tuple[int, ...]
    canonical_to_model_positions: tuple[tuple[int, ...], ...]
    schema_version: str = MODEL_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty_text(self.family, field_name="family")
        if self.family != _EXECUTABLE_REPRESENTATION_FAMILY:
            raise ValueError(
                f"{self.family} has no accepted model evidence-supervision contract"
            )
        if self.schema_version != MODEL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("model evidence-supervision schema mismatch")
        _validate_label_ownership(
            self.labels,
            self.model_token_ids,
            self.evidence_token_positions,
            name="model evidence",
        )
        if not self.canonical_to_model_positions:
            raise ValueError("model supervision requires a canonical expansion map")
        flattened: list[int] = []
        for mapped in self.canonical_to_model_positions:
            if not mapped or tuple(range(mapped[0], mapped[-1] + 1)) != mapped:
                raise ValueError(
                    "model supervision expansion blocks must be non-empty/contiguous"
                )
            flattened.extend(mapped)
        if tuple(flattened) != tuple(range(len(self.model_token_ids))):
            raise ValueError("model supervision expansion must cover every model token")
        if tuple(sorted(set(self.visual_model_positions))) != (
            self.visual_model_positions
        ):
            raise ValueError(
                "model supervision visual positions must be ordered/unique"
            )
        for position in self.visual_model_positions:
            _validate_position(position, len(self.model_token_ids), name="visual model")
        visual = set(self.visual_model_positions)
        visual_blocks: list[tuple[int, ...]] = []
        for mapped in self.canonical_to_model_positions:
            overlap = visual.intersection(mapped)
            if overlap and overlap != set(mapped):
                raise ValueError("a canonical token expansion cannot be partly visual")
            if overlap:
                visual_blocks.append(mapped)
        if (
            tuple(position for block in visual_blocks for position in block)
            != self.visual_model_positions
        ):
            raise ValueError(
                "visual expansion blocks must exactly cover visual positions"
            )
        if self.evidence_token_positions != tuple(
            range(
                self.evidence_token_positions[0],
                self.evidence_token_positions[-1] + 1,
            )
        ):
            raise ValueError("model evidence token positions must be contiguous")
        evidence = set(self.evidence_token_positions)
        if evidence.intersection(self.visual_model_positions):
            raise ValueError("evidence labels cannot overlap visual model positions")
        if self.visual_model_positions and (
            self.evidence_token_positions[0] <= self.visual_model_positions[-1]
        ):
            raise ValueError(
                "post-tool evidence must follow every visual model position"
            )

    @property
    def evidence_token_count(self) -> int:
        return len(self.evidence_token_positions)

    @property
    def visual_expansion_blocks(self) -> tuple[tuple[int, ...], ...]:
        """Ordered expanded blocks for each canonical visual placeholder."""

        visual = set(self.visual_model_positions)
        return tuple(
            mapped
            for mapped in self.canonical_to_model_positions
            if set(mapped).issubset(visual) and mapped
        )


def render_native_evidence_labels(
    renderer: NativeProtocolRenderer,
    messages: Sequence[Mapping[str, Any]],
    *,
    evidence_description: str,
) -> CanonicalEvidenceSupervision:
    """Render the accepted Qwen-thinking evidence turn and own its tokens.

    Prompt wording and target construction remain separately versioned. This
    helper accepts already-constructed native messages, proves that the full
    transcript is exactly the post-tool generation prefill plus evidence and
    the template-owned closing suffix, then labels only offset-owned evidence.
    """

    if not isinstance(renderer, NativeProtocolRenderer):
        raise TypeError("renderer must be a NativeProtocolRenderer")
    answer_text = _validate_native_evidence_request(messages, evidence_description)

    history = messages[:-1]
    generation_prefill = renderer.render(history, add_generation_prompt=True)
    renderer.assert_generation_prefill(generation_prefill, renderer.tokenizer)
    transcript = renderer.render(messages, add_generation_prompt=False)
    return _native_evidence_supervision_from_rendered(
        renderer,
        evidence_description=evidence_description,
        answer_text=answer_text,
        generation_prefill=generation_prefill,
        transcript=transcript,
    )


def _render_native_evidence_labels_batch(
    renderer: NativeProtocolRenderer,
    messages_batch: Sequence[Sequence[Mapping[str, Any]]],
    *,
    evidence_descriptions: Sequence[str],
) -> tuple[CanonicalEvidenceSupervision, ...]:
    """Render one same-image group's evidence transcripts in two batch calls."""

    if not isinstance(renderer, NativeProtocolRenderer):
        raise TypeError("renderer must be a NativeProtocolRenderer")
    if not isinstance(messages_batch, Sequence) or isinstance(
        messages_batch, (str, bytes)
    ):
        raise TypeError("messages_batch must be a sequence of message sequences")
    if not isinstance(evidence_descriptions, Sequence) or isinstance(
        evidence_descriptions, (str, bytes)
    ):
        raise TypeError("evidence_descriptions must be a sequence of strings")
    if not messages_batch:
        raise ValueError("native evidence batch cannot be empty")
    if len(messages_batch) != len(evidence_descriptions):
        raise ValueError("native evidence messages and descriptions must align")
    answer_texts = tuple(
        _validate_native_evidence_request(messages, evidence_description)
        for messages, evidence_description in zip(
            messages_batch, evidence_descriptions, strict=True
        )
    )

    generation_prefills = renderer.render_many(
        tuple(messages[:-1] for messages in messages_batch),
        add_generation_prompt=True,
    )
    if len(generation_prefills) != len(messages_batch):
        raise RuntimeError("native evidence prefill batch changed cardinality")
    for generation_prefill in generation_prefills:
        renderer.assert_generation_prefill(generation_prefill, renderer.tokenizer)
    transcripts = renderer.render_many(
        messages_batch,
        add_generation_prompt=False,
    )
    if len(transcripts) != len(messages_batch):
        raise RuntimeError("native evidence transcript batch changed cardinality")
    return tuple(
        _native_evidence_supervision_from_rendered(
            renderer,
            evidence_description=evidence_description,
            answer_text=answer_text,
            generation_prefill=generation_prefill,
            transcript=transcript,
        )
        for evidence_description, answer_text, generation_prefill, transcript in zip(
            evidence_descriptions,
            answer_texts,
            generation_prefills,
            transcripts,
            strict=True,
        )
    )


def _validate_native_evidence_request(
    messages: Sequence[Mapping[str, Any]],
    evidence_description: str,
) -> str:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError("messages must be a sequence of mappings")
    if not isinstance(evidence_description, str) or not evidence_description.strip():
        raise ValueError("evidence_description must be a non-empty string")
    if evidence_description != evidence_description.strip("\n"):
        raise ValueError("evidence_description cannot have leading/trailing newlines")
    if any(fragment in evidence_description for fragment in _NATIVE_CONTROL_FRAGMENTS):
        raise ValueError("evidence_description cannot contain native control tags")
    return _validate_post_tool_evidence_messages(messages, evidence_description)


def _native_evidence_supervision_from_rendered(
    renderer: NativeProtocolRenderer,
    *,
    evidence_description: str,
    answer_text: str,
    generation_prefill: RenderedTranscript,
    transcript: RenderedTranscript,
) -> CanonicalEvidenceSupervision:
    expected_text = (
        generation_prefill.text
        + evidence_description
        + _QWEN_THINKING_COMPLETION_MIDDLE
        + answer_text
        + _QWEN_ASSISTANT_TURN_SUFFIX
    )
    if transcript.text != expected_text:
        raise ValueError(
            "completed evidence transcript differs from the accepted native "
            "generation-prefill contract"
        )

    evidence_start = len(generation_prefill.text)
    evidence_end = evidence_start + len(evidence_description)
    token_ids, offsets = _tokenize_with_exact_offsets(renderer, transcript)
    positions = _evidence_owned_token_positions(
        transcript.text,
        offsets,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )
    owned = set(positions)
    canonical_labels = tuple(
        token_id if position in owned else EVIDENCE_IGNORE_INDEX
        for position, token_id in enumerate(token_ids)
    )
    evidence_byte_start = len(transcript.text[:evidence_start].encode("utf-8"))
    evidence_byte_end = evidence_byte_start + len(evidence_description.encode("utf-8"))
    return CanonicalEvidenceSupervision(
        transcript=transcript,
        generation_prefill=generation_prefill,
        evidence_text=evidence_description,
        answer_text=answer_text,
        canonical_labels=canonical_labels,
        evidence_char_start=evidence_start,
        evidence_char_end=evidence_end,
        evidence_byte_start=evidence_byte_start,
        evidence_byte_end=evidence_byte_end,
        evidence_token_positions=positions,
        token_offsets=offsets,
    )


def _build_visual_token_expansion(
    *,
    family: str,
    canonical_token_ids: Sequence[int],
    model_token_ids: Sequence[int],
    visual_placeholder_token_id: int,
) -> CanonicalToModelTokenExpansion:
    """Build the exact expansion map for repeated Qwen visual placeholders.

    Every non-visual canonical token must remain one identical model token.
    Each visual placeholder consumes the complete matching run in model input.
    Consecutive canonical visual placeholders are rejected as ambiguous.
    """

    _require_non_empty_text(family, field_name="family")
    canonical = tuple(
        _require_plain_int(value, field_name="canonical_token_ids")
        for value in canonical_token_ids
    )
    model = tuple(
        _require_plain_int(value, field_name="model_token_ids")
        for value in model_token_ids
    )
    visual_id = _require_plain_int(
        visual_placeholder_token_id, field_name="visual_placeholder_token_id"
    )
    if not canonical or not model:
        raise ValueError("visual token expansion requires non-empty token sequences")

    mapping: list[tuple[int, ...]] = []
    visual_positions: list[int] = []
    model_cursor = 0
    for canonical_position, token_id in enumerate(canonical):
        if model_cursor >= len(model):
            raise ValueError("model token sequence ended before canonical expansion")
        if token_id != visual_id:
            if model[model_cursor] != token_id:
                raise ValueError(
                    "non-visual model token differs from canonical transcript"
                )
            mapping.append((model_cursor,))
            model_cursor += 1
            continue

        if (
            canonical_position + 1 < len(canonical)
            and canonical[canonical_position + 1] == visual_id
        ):
            raise ValueError("consecutive canonical visual placeholders are ambiguous")
        start = model_cursor
        while model_cursor < len(model) and model[model_cursor] == visual_id:
            model_cursor += 1
        if model_cursor == start:
            raise ValueError("canonical visual placeholder is absent from model input")
        positions = tuple(range(start, model_cursor))
        mapping.append(positions)
        visual_positions.extend(positions)

    if model_cursor != len(model):
        raise ValueError("model token sequence has an unmatched expansion suffix")
    return CanonicalToModelTokenExpansion(
        family=family,
        canonical_token_ids=canonical,
        model_token_ids=model,
        canonical_to_model_positions=tuple(mapping),
        visual_model_positions=tuple(visual_positions),
    )


def _materialize_model_evidence_supervision(
    canonical: CanonicalEvidenceSupervision,
    expansion: CanonicalToModelTokenExpansion,
) -> ModelEvidenceSupervision:
    """Map canonical evidence labels onto exact expanded model positions."""

    if not isinstance(canonical, CanonicalEvidenceSupervision):
        raise TypeError("canonical must be CanonicalEvidenceSupervision")
    if not isinstance(expansion, CanonicalToModelTokenExpansion):
        raise TypeError("expansion must be CanonicalToModelTokenExpansion")
    if expansion.canonical_token_ids != canonical.transcript.token_ids:
        raise ValueError("token expansion belongs to a different canonical transcript")

    model_positions: list[int] = []
    for canonical_position in canonical.evidence_token_positions:
        mapped = expansion.canonical_to_model_positions[canonical_position]
        if len(mapped) != 1:
            raise ValueError(
                "an evidence token cannot expand to multiple model positions"
            )
        model_position = mapped[0]
        if (
            expansion.model_token_ids[model_position]
            != canonical.transcript.token_ids[canonical_position]
        ):
            raise ValueError("evidence token id changed during model expansion")
        model_positions.append(model_position)
    if tuple(model_positions) != tuple(
        range(model_positions[0], model_positions[-1] + 1)
    ):
        raise ValueError("expanded evidence token positions must remain contiguous")

    owned = set(model_positions)
    labels = tuple(
        token_id if position in owned else EVIDENCE_IGNORE_INDEX
        for position, token_id in enumerate(expansion.model_token_ids)
    )
    return ModelEvidenceSupervision(
        family=expansion.family,
        model_token_ids=expansion.model_token_ids,
        labels=labels,
        evidence_token_positions=tuple(model_positions),
        visual_model_positions=expansion.visual_model_positions,
        canonical_to_model_positions=expansion.canonical_to_model_positions,
    )


def _validate_post_tool_evidence_messages(
    messages: Sequence[Mapping[str, Any]], evidence_description: str
) -> str:
    if len(messages) != 4 or not all(
        isinstance(message, Mapping) for message in messages
    ):
        raise ValueError(
            "native representation transcript requires exactly four mapped turns"
        )
    user_turn = messages[0]
    user_content = user_turn.get("content")
    if (
        user_turn.get("role") != "user"
        or not isinstance(user_content, Sequence)
        or isinstance(user_content, (str, bytes))
        or sum(
            isinstance(item, Mapping) and item.get("type") == "image"
            for item in user_content
        )
        != 1
    ):
        raise ValueError(
            "representation user turn must contain exactly one original image"
        )
    evidence_turn = messages[-1]
    if evidence_turn.get("role") != "assistant":
        raise ValueError("evidence_description must be in the final assistant turn")
    if evidence_turn.get("reasoning_content") != evidence_description:
        raise ValueError(
            "final assistant reasoning_content must equal evidence_description exactly"
        )
    answer_text = evidence_turn.get("content")
    if not isinstance(answer_text, str):
        raise ValueError("the representation answer content must be a string")
    if evidence_turn.get("tool_calls"):
        raise ValueError("the evidence readout turn cannot contain another tool call")
    if messages[-2].get("role") != "tool":
        raise ValueError(
            "the evidence readout assistant turn must immediately follow a tool result"
        )
    tool_content = messages[-2].get("content")
    if (
        not isinstance(tool_content, Sequence)
        or isinstance(tool_content, (str, bytes))
        or len(tool_content) != 1
        or not isinstance(tool_content[0], Mapping)
        or tool_content[0].get("type") != "image"
    ):
        raise ValueError(
            "the representation tool result must be one latent image block"
        )

    call_turn = messages[-3]
    if call_turn.get("role") != "assistant":
        raise ValueError("the tool result must follow an assistant tool-call turn")
    if call_turn.get("content") != "":
        raise ValueError("the representation tool-call turn cannot contain answer text")
    if call_turn.get("reasoning_content") != NATIVE_REPRESENTATION_PRE_REASONING:
        raise ValueError(
            "the representation tool-call reasoning must equal the fixed "
            "native pre-reasoning"
        )
    if not answer_text.strip():
        raise ValueError(
            "native representation requires non-empty short answer content"
        )
    if any(fragment in answer_text for fragment in _NATIVE_CONTROL_FRAGMENTS):
        raise ValueError("short answer content cannot contain native control tags")
    calls = call_turn.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise ValueError(
            "the preceding assistant turn must contain one native tool call"
        )
    if len(calls) != 1 or not isinstance(calls[0], Mapping):
        raise ValueError(
            "the preceding assistant turn must contain exactly one tool call"
        )
    function = calls[0].get("function")
    if (
        not isinstance(function, Mapping)
        or function.get("name") != TGVF_FOCUS_TOOL_NAME
    ):
        raise ValueError(f"the preceding call must invoke {TGVF_FOCUS_TOOL_NAME}")
    arguments = function.get("arguments")
    if (
        not isinstance(arguments, Mapping)
        or set(arguments) != {"target"}
        or not isinstance(arguments.get("target"), str)
        or not arguments["target"].strip()
    ):
        raise ValueError("the representation tool call requires one non-empty target")
    return answer_text


def _tokenize_with_exact_offsets(
    renderer: NativeProtocolRenderer, transcript: RenderedTranscript
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    if getattr(renderer.tokenizer, "is_fast", None) is not True:
        raise TypeError("native evidence labeling requires a fast tokenizer")
    try:
        encoded = renderer.tokenizer(
            transcript.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
    except (TypeError, NotImplementedError) as error:
        raise TypeError(
            "native evidence labeling requires tokenizer offset mapping"
        ) from error
    renderer.assert_tokenizer_length()
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer offset result must be a mapping")
    raw_ids = encoded.get("input_ids")
    raw_offsets = encoded.get("offset_mapping")
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise TypeError("tokenizer offset result has invalid input_ids")
    token_ids = tuple(
        _require_plain_int(value, field_name="input_ids") for value in raw_ids
    )
    if token_ids != transcript.token_ids:
        raise ValueError(
            "offset tokenization differs from rendered transcript token ids"
        )
    if not isinstance(raw_offsets, Sequence) or isinstance(raw_offsets, (str, bytes)):
        raise TypeError("tokenizer did not return offset_mapping")
    offsets = tuple(_coerce_offset(value) for value in raw_offsets)
    if len(offsets) != len(token_ids):
        raise ValueError("token offsets must align with rendered token ids")
    return token_ids, offsets


def _evidence_owned_token_positions(
    text: str,
    offsets: Sequence[tuple[int, int]],
    *,
    evidence_start: int,
    evidence_end: int,
) -> tuple[int, ...]:
    positions: list[int] = []
    owned_offsets: list[tuple[int, int]] = []
    for position, (start, end) in enumerate(offsets):
        if start < evidence_start < end:
            raise ValueError(
                "a tokenizer token begins before and crosses into evidence"
            )
        if evidence_start <= start < evidence_end:
            if end <= start:
                raise ValueError(
                    "an evidence-owned tokenizer token has an empty offset"
                )
            if end > evidence_end and not text[evidence_end:end].isspace():
                raise ValueError(
                    "an evidence-owned token crosses into non-whitespace template content"
                )
            positions.append(position)
            owned_offsets.append((start, end))

    if not positions:
        raise ValueError("evidence_description owns no tokenizer positions")
    cursor = evidence_start
    for start, end in owned_offsets:
        if start > cursor:
            raise ValueError("tokenizer offsets leave an uncovered evidence character")
        cursor = max(cursor, min(end, evidence_end))
    if cursor != evidence_end:
        raise ValueError("tokenizer offsets do not cover the complete evidence span")
    if tuple(positions) != tuple(range(positions[0], positions[-1] + 1)):
        raise ValueError("canonical evidence token positions must be contiguous")
    return tuple(positions)


def _validate_label_ownership(
    labels: Sequence[int],
    token_ids: Sequence[int],
    positions: tuple[int, ...],
    *,
    name: str,
) -> None:
    if len(labels) != len(token_ids):
        raise ValueError(f"{name} labels must align with token ids")
    if not positions:
        raise ValueError(f"{name} must own at least one token")
    if tuple(sorted(set(positions))) != positions:
        raise ValueError(f"{name} token positions must be unique and ordered")
    if positions[0] == 0:
        raise ValueError("the first sequence token cannot receive a causal label")
    owned = set(positions)
    for position, (label, token_id) in enumerate(zip(labels, token_ids, strict=True)):
        expected = token_id if position in owned else EVIDENCE_IGNORE_INDEX
        if label != expected:
            raise ValueError(f"{name} labels do not match their ownership mask")


def _coerce_offset(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError("every tokenizer offset must be an integer pair")
    start = _require_plain_int(value[0], field_name="offset start")
    end = _require_plain_int(value[1], field_name="offset end")
    if start < 0 or end < start:
        raise ValueError("tokenizer offsets must satisfy 0 <= start <= end")
    return start, end


def _validate_position(position: object, length: int, *, name: str) -> None:
    value = _require_plain_int(position, field_name=f"{name} position")
    if value < 0 or value >= length:
        raise ValueError(f"{name} position is outside the token sequence")


def _require_plain_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} values must be integers")
    return value


def _require_non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
