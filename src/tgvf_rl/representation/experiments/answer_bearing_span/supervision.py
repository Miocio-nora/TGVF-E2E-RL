"""Sparse sidecar-span-plus-answer labels for the isolated RP70 objective."""

from __future__ import annotations

from hashlib import sha256
import json

from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.readout import (
    RepresentationReadoutLossSupervision,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.transcript import (
    CanonicalEvidenceSupervision,
    ModelEvidenceSupervision,
    _QWEN_ASSISTANT_TURN_SUFFIX,
    _completion_middle,
)

from .data import (
    ANSWER_BEARING_SPAN_MATCH_POLICY,
    AnswerBearingSpanIndex,
    AnswerBearingSpanIndexSet,
    AnswerBearingSpanRecord,
    answer_bearing_span_semantic_sha256,
)


ANSWER_BEARING_SPAN_SUPERVISION_SCHEMA_VERSION = "answer_bearing_span_supervision_v2"
ANSWER_BEARING_SPAN_SUPERVISION_POLICY = (
    "explicit_sidecar_offsets_only_plus_final_answer_no_separator_or_eos_v1"
)


class AnswerBearingSpanSupervisionError(ValueError):
    """One sample/transcript cannot satisfy the fixed RP70 label contract."""


class AnswerBearingSpanSupervisionFactory:
    """Map only audited sidecar offsets onto the unchanged native transcript."""

    def __init__(
        self,
        index: AnswerBearingSpanIndex | AnswerBearingSpanIndexSet,
    ) -> None:
        if not isinstance(index, (AnswerBearingSpanIndex, AnswerBearingSpanIndexSet)):
            raise TypeError("answer-bearing supervision requires a bound span index")
        self.index = index
        self._records = index.by_uid
        self._identity_sha256 = _canonical_sha256(
            {
                "schema_version": ANSWER_BEARING_SPAN_SUPERVISION_SCHEMA_VERSION,
                "supervision_policy": ANSWER_BEARING_SPAN_SUPERVISION_POLICY,
                "match_policy": ANSWER_BEARING_SPAN_MATCH_POLICY,
                "span_index_identity_sha256": index.identity_sha256,
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    @property
    def statistics(self):
        """Expose the immutable matched/unmatched/multiple population audit."""

        return self.index.statistics

    def __call__(
        self,
        sample: RepresentationTrainingSample,
        canonical: CanonicalEvidenceSupervision,
        model: ModelEvidenceSupervision,
    ) -> RepresentationReadoutLossSupervision:
        if not isinstance(sample, RepresentationTrainingSample):
            raise TypeError("RP70 supervision sample has an invalid type")
        if not isinstance(canonical, CanonicalEvidenceSupervision):
            raise TypeError("RP70 canonical supervision has an invalid type")
        if not isinstance(model, ModelEvidenceSupervision):
            raise TypeError("RP70 model supervision has an invalid type")
        try:
            record = self._records[sample.sample_id]
        except KeyError as error:
            raise AnswerBearingSpanSupervisionError(
                f"span index has no record for sample {sample.sample_id!r}"
            ) from error
        _validate_bound_inputs(sample, record, canonical, model)

        value_canonical_positions = _value_token_positions(record, canonical)
        answer_canonical_positions = _answer_token_positions(canonical)
        value_model_positions = _map_canonical_positions(
            value_canonical_positions,
            canonical=canonical,
            model=model,
            name="evidence value",
        )
        answer_model_positions = _map_canonical_positions(
            answer_canonical_positions,
            canonical=canonical,
            model=model,
            name="final answer",
        )
        supervised_positions = tuple(
            sorted(set((*value_model_positions, *answer_model_positions)))
        )
        supervised = set(supervised_positions)
        labels = tuple(
            token_id if position in supervised else EVIDENCE_IGNORE_INDEX
            for position, token_id in enumerate(model.model_token_ids)
        )
        identity = _sample_supervision_identity(
            factory_identity_sha256=self.identity_sha256,
            sample=sample,
            canonical=canonical,
            model=model,
            record=record,
            value_model_positions=value_model_positions,
            answer_model_positions=answer_model_positions,
        )
        return RepresentationReadoutLossSupervision(
            identity=identity,
            labels=labels,
            supervised_token_positions=supervised_positions,
            evidence_value_token_positions=value_model_positions,
            answer_token_positions=answer_model_positions,
            source_image_block_query_start=model.evidence_token_positions[0] - 1,
            source_image_block_query_end=answer_model_positions[-1],
        )


def _validate_bound_inputs(
    sample: RepresentationTrainingSample,
    record: AnswerBearingSpanRecord,
    canonical: CanonicalEvidenceSupervision,
    model: ModelEvidenceSupervision,
) -> None:
    if record.uid != sample.sample_id:
        raise AnswerBearingSpanSupervisionError("span record and sample UID differ")
    if record.semantic_content_sha256 != answer_bearing_span_semantic_sha256(sample):
        raise AnswerBearingSpanSupervisionError(
            "span record semantic identity differs from the representation sample"
        )
    if record.evidence_description != sample.evidence_description:
        raise AnswerBearingSpanSupervisionError(
            "span record evidence differs from the representation sample"
        )
    if record.short_answer != sample.short_answer:
        raise AnswerBearingSpanSupervisionError(
            "span record short answer differs from the representation sample"
        )
    if canonical.evidence_text != sample.evidence_description:
        raise AnswerBearingSpanSupervisionError(
            "canonical evidence differs from the representation sample"
        )
    if canonical.answer_text != sample.short_answer:
        raise AnswerBearingSpanSupervisionError(
            "canonical final answer differs from the representation sample"
        )
    if len(model.canonical_to_model_positions) != len(canonical.transcript.token_ids):
        raise AnswerBearingSpanSupervisionError(
            "canonical/model token expansion lengths differ"
        )
    for canonical_position, mapped in enumerate(model.canonical_to_model_positions):
        canonical_id = canonical.transcript.token_ids[canonical_position]
        if any(model.model_token_ids[position] != canonical_id for position in mapped):
            raise AnswerBearingSpanSupervisionError(
                "canonical/model token expansion identities differ"
            )


def _value_token_positions(
    record: AnswerBearingSpanRecord,
    canonical: CanonicalEvidenceSupervision,
) -> tuple[int, ...]:
    positions: set[int] = set()
    for evidence_span in record.value_character_spans:
        span_start = canonical.evidence_char_start + evidence_span.start
        span_end = canonical.evidence_char_start + evidence_span.end
        if canonical.transcript.text[span_start:span_end] != evidence_span.exact_text:
            raise AnswerBearingSpanSupervisionError(
                "bound evidence value span differs from the canonical transcript"
            )
        positions.update(
            _minimal_overlapping_token_positions(
                canonical.transcript.text,
                canonical.token_offsets,
                span_start=span_start,
                span_end=span_end,
                name="evidence value",
            )
        )
    ordered = tuple(sorted(positions))
    if not set(ordered).issubset(canonical.evidence_token_positions):
        raise AnswerBearingSpanSupervisionError(
            "evidence value token cover escapes the complete evidence span"
        )
    return ordered


def _answer_token_positions(
    canonical: CanonicalEvidenceSupervision,
) -> tuple[int, ...]:
    answer_start = canonical.evidence_char_end + len(
        _completion_middle(canonical.assistant_dialect)
    )
    answer_end = answer_start + len(canonical.answer_text)
    text = canonical.transcript.text
    if text[answer_start:answer_end] != canonical.answer_text:
        raise AnswerBearingSpanSupervisionError(
            "final answer span differs from the canonical transcript"
        )
    if text[answer_end:] != _QWEN_ASSISTANT_TURN_SUFFIX:
        raise AnswerBearingSpanSupervisionError(
            "final answer is not immediately followed by the native assistant suffix"
        )
    positions = _minimal_overlapping_token_positions(
        text,
        canonical.token_offsets,
        span_start=answer_start,
        span_end=answer_end,
        name="final answer",
    )
    first_start = canonical.token_offsets[positions[0]][0]
    final_end = canonical.token_offsets[positions[-1]][1]
    if first_start < answer_start or final_end > answer_end:
        raise AnswerBearingSpanSupervisionError(
            "final answer token cover includes separator or assistant-suffix content"
        )
    return positions


def _minimal_overlapping_token_positions(
    text: str,
    offsets: tuple[tuple[int, int], ...],
    *,
    span_start: int,
    span_end: int,
    name: str,
) -> tuple[int, ...]:
    if not (0 <= span_start < span_end <= len(text)):
        raise AnswerBearingSpanSupervisionError(f"{name} character span is invalid")
    positions = tuple(
        position
        for position, (start, end) in enumerate(offsets)
        if start < span_end and end > span_start
    )
    if not positions:
        raise AnswerBearingSpanSupervisionError(f"no tokenizer token overlaps {name}")
    if positions != tuple(range(positions[0], positions[-1] + 1)):
        raise AnswerBearingSpanSupervisionError(
            f"{name} token positions are not contiguous"
        )
    selected_offsets = tuple(offsets[position] for position in positions)
    if any(end <= start for start, end in selected_offsets):
        raise AnswerBearingSpanSupervisionError(
            f"a zero-width tokenizer token overlaps {name}"
        )
    # Fast tokenizers may report overlapping offsets for adjacent tokens (for
    # example, byte-fallback pieces that share a Unicode character boundary).
    # Require an unbroken union over the requested character span, but do not
    # incorrectly reject such legitimate overlap.
    covered_until = span_start
    for start, end in selected_offsets:
        clipped_start = max(start, span_start)
        clipped_end = min(end, span_end)
        if clipped_start > covered_until:
            raise AnswerBearingSpanSupervisionError(
                f"{name} token offsets leave a gap in its character span"
            )
        covered_until = max(covered_until, clipped_end)
    if covered_until < span_end:
        raise AnswerBearingSpanSupervisionError(
            f"{name} token positions do not cover its character span"
        )
    return positions


def _map_canonical_positions(
    positions: tuple[int, ...],
    *,
    canonical: CanonicalEvidenceSupervision,
    model: ModelEvidenceSupervision,
    name: str,
) -> tuple[int, ...]:
    mapped_positions: set[int] = set()
    for canonical_position in positions:
        if canonical_position >= len(model.canonical_to_model_positions):
            raise AnswerBearingSpanSupervisionError(
                f"{name} canonical position lies outside the expansion map"
            )
        mapped = model.canonical_to_model_positions[canonical_position]
        if len(mapped) != 1:
            raise AnswerBearingSpanSupervisionError(
                f"{name} token cannot expand to multiple model positions"
            )
        model_position = mapped[0]
        if (
            model.model_token_ids[model_position]
            != canonical.transcript.token_ids[canonical_position]
        ):
            raise AnswerBearingSpanSupervisionError(
                f"{name} token ID changed during model expansion"
            )
        mapped_positions.add(model_position)
    return tuple(sorted(mapped_positions))


def _sample_supervision_identity(
    *,
    factory_identity_sha256: str,
    sample: RepresentationTrainingSample,
    canonical: CanonicalEvidenceSupervision,
    model: ModelEvidenceSupervision,
    record: AnswerBearingSpanRecord,
    value_model_positions: tuple[int, ...],
    answer_model_positions: tuple[int, ...],
) -> str:
    payload = {
        "schema_version": ANSWER_BEARING_SPAN_SUPERVISION_SCHEMA_VERSION,
        "factory_identity_sha256": factory_identity_sha256,
        "sample_id": sample.sample_id,
        "sample_semantic_sha256": record.semantic_content_sha256,
        "transcript_text_sha256": canonical.transcript.text_sha256,
        "transcript_token_ids_sha256": canonical.transcript.token_ids_sha256,
        "model_token_ids_sha256": _canonical_sha256(list(model.model_token_ids)),
        "record_status": record.status.value,
        "record_reason": record.reason,
        "value_character_spans": [
            {
                "start": span.start,
                "end": span.end,
                "exact_text": span.exact_text,
            }
            for span in record.value_character_spans
        ],
        "evidence_value_token_positions": list(value_model_positions),
        "answer_token_positions": list(answer_model_positions),
    }
    return (
        f"{ANSWER_BEARING_SPAN_SUPERVISION_SCHEMA_VERSION}:{_canonical_sha256(payload)}"
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "ANSWER_BEARING_SPAN_SUPERVISION_POLICY",
    "ANSWER_BEARING_SPAN_SUPERVISION_SCHEMA_VERSION",
    "AnswerBearingSpanSupervisionError",
    "AnswerBearingSpanSupervisionFactory",
]
