from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.protocol.native import NativeAssistantDialect, RenderedTranscript
from tgvf_rl.representation.experiments.image_axis_grounding.matching import (
    ImageAxisDonorAssignment,
    ImageAxisDonorManifest,
    ImageAxisDonorSourceBinding,
    QwenImageGridContract,
)
from tgvf_rl.representation.experiments.image_axis_grounding.native_pipeline import (
    ImageAxisGroundedNativeGroupBuilder,
)
from tgvf_rl.representation.experiments.answer_bearing_span.data import (
    VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
    AnswerBearingSpanStatus,
    EvidenceCharacterSpan,
    load_answer_bearing_span_index,
    render_answer_bearing_span_sidecar,
)
from tgvf_rl.representation.experiments.answer_bearing_span.supervision import (
    ANSWER_BEARING_SPAN_SUPERVISION_SCHEMA_VERSION,
    AnswerBearingSpanSupervisionError,
    AnswerBearingSpanSupervisionFactory,
    _minimal_overlapping_token_positions,
)
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
)
from tgvf_rl.representation.training.data import (
    REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION,
    REPRESENTATION_DATA_TRANSFORM_VERSION,
    AcceptedRowManifestEntry,
    RepresentationDataManifest,
    RepresentationDataset,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.transcript import (
    CanonicalEvidenceSupervision,
    ModelEvidenceSupervision,
)


_MIDDLE = "\n\n"
_SUFFIX = "<|im_end|>\n"


def test_factory_labels_only_explicit_sidecar_occurrence_and_final_answer(
    tmp_path: Path,
) -> None:
    sample = _sample(evidence="red then red.", answer="red")
    canonical, model, surfaces = _supervisions(
        evidence_pieces=("red", " then", " red", "."),
        answer_pieces=("red",),
    )
    factory = _factory(
        tmp_path,
        sample=sample,
        spans=(EvidenceCharacterSpan(start=9, end=12, exact_text="red"),),
    )

    result = factory(sample, canonical, model)

    assert result.identity.startswith(
        f"{ANSWER_BEARING_SPAN_SUPERVISION_SCHEMA_VERSION}:"
    )
    assert tuple(
        surfaces[position] for position in result.evidence_value_token_positions
    ) == (" red",)
    assert tuple(surfaces[position] for position in result.answer_token_positions) == (
        "red",
    )
    assert result.supervised_token_positions == tuple(
        sorted(
            (
                *result.evidence_value_token_positions,
                *result.answer_token_positions,
            )
        )
    )
    assert (
        tuple(
            position
            for position, label in enumerate(result.labels)
            if label != EVIDENCE_IGNORE_INDEX
        )
        == result.supervised_token_positions
    )
    assert result.source_image_block_query_start == (
        model.evidence_token_positions[0] - 1
    )
    assert result.source_image_block_query_end == result.answer_token_positions[-1]
    separator_positions = tuple(
        position for position, surface in enumerate(surfaces) if surface == _MIDDLE
    )
    suffix_positions = tuple(
        position for position, surface in enumerate(surfaces) if surface == _SUFFIX
    )
    assert not set(result.supervised_token_positions).intersection(
        (*separator_positions, *suffix_positions)
    )


def test_verified_no_answer_bearing_evidence_is_explicitly_answer_only(
    tmp_path: Path,
) -> None:
    sample = _sample(evidence="There are four RED objects.", answer="4")
    canonical, model, _ = _supervisions(
        evidence_pieces=("There", " are", " four", " RED", " objects", "."),
        answer_pieces=("4",),
        answer="4",
    )
    factory = _factory(
        tmp_path,
        sample=sample,
        status=AnswerBearingSpanStatus.VERIFIED_NO_ANSWER_BEARING_EVIDENCE,
        reason=VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
        spans=(),
    )

    result = factory(sample, canonical, model)

    assert result.evidence_value_token_positions == ()
    assert result.supervised_token_positions == result.answer_token_positions
    assert factory.statistics.matched_rows == 0
    assert factory.statistics.unmatched_rows == 1
    assert factory.statistics.coverage == 0.0


def test_factory_consumes_explicit_offsets_without_answer_literal_matching(
    tmp_path: Path,
) -> None:
    sample = _sample(evidence="There are four objects.", answer="4")
    canonical, model, surfaces = _supervisions(
        evidence_pieces=("There", " are", " four", " objects", "."),
        answer_pieces=("4",),
        answer="4",
    )
    factory = _factory(
        tmp_path,
        sample=sample,
        spans=(EvidenceCharacterSpan(start=10, end=14, exact_text="four"),),
    )

    result = factory(sample, canonical, model)

    assert tuple(
        surfaces[position] for position in result.evidence_value_token_positions
    ) == (" four",)
    assert tuple(surfaces[position] for position in result.answer_token_positions) == (
        "4",
    )


def test_value_uses_minimal_overlapping_token_cover_with_leading_space(
    tmp_path: Path,
) -> None:
    sample = _sample(evidence="The color is red.", answer="red")
    canonical, model, surfaces = _supervisions(
        evidence_pieces=("The", " color", " is", " red", "."),
        answer_pieces=("red",),
    )
    factory = _factory(
        tmp_path,
        sample=sample,
        spans=(EvidenceCharacterSpan(start=13, end=16, exact_text="red"),),
    )

    result = factory(sample, canonical, model)

    assert len(result.evidence_value_token_positions) == 1
    assert surfaces[result.evidence_value_token_positions[0]] == " red"


def test_value_minimal_cover_keeps_answer_bearing_token_with_trailing_punctuation(
    tmp_path: Path,
) -> None:
    sample = _sample(evidence="Rate is 20%, exactly.", answer="20%")
    canonical, model, surfaces = _supervisions(
        evidence_pieces=("Rate", " is", " 20", "%,", " exactly", "."),
        answer_pieces=("20%",),
        answer="20%",
    )
    factory = _factory(
        tmp_path,
        sample=sample,
        spans=(EvidenceCharacterSpan(start=8, end=11, exact_text="20%"),),
    )

    result = factory(sample, canonical, model)

    assert tuple(
        surfaces[position] for position in result.evidence_value_token_positions
    ) == (
        " 20",
        "%,",
    )


def test_factory_maps_sparse_positions_across_visual_expansion_and_rejects_drift(
    tmp_path: Path,
) -> None:
    sample = _sample(evidence="The sign reads OPEN.", answer="OPEN")
    canonical, model, model_surfaces = _supervisions(
        evidence_pieces=("The", " sign", " reads", " OPEN", "."),
        answer_pieces=("OPEN",),
        answer="OPEN",
        visual_expansion_counts=(2, 3),
    )
    factory = _factory(
        tmp_path,
        sample=sample,
        spans=(EvidenceCharacterSpan(start=15, end=19, exact_text="OPEN"),),
    )

    result = factory(sample, canonical, model)

    assert tuple(
        model_surfaces[position] for position in result.evidence_value_token_positions
    ) == (" OPEN",)
    assert tuple(
        model_surfaces[position] for position in result.answer_token_positions
    ) == ("OPEN",)
    assert result.source_image_block_query_start == (
        model.evidence_token_positions[0] - 1
    )

    drifted_canonical, _, _ = _supervisions(
        evidence_pieces=("The sign", " reads", " OPEN", "."),
        answer_pieces=("OPEN",),
        answer="OPEN",
    )
    with pytest.raises(
        AnswerBearingSpanSupervisionError,
        match="expansion lengths differ",
    ):
        factory(sample, drifted_canonical, model)


def test_factory_identity_and_output_are_deterministic(tmp_path: Path) -> None:
    sample = _sample(evidence="The sign reads OPEN.", answer="OPEN")
    canonical, model, _ = _supervisions(
        evidence_pieces=("The", " sign", " reads", " OPEN", "."),
        answer_pieces=("OPEN",),
        answer="OPEN",
    )
    factory = _factory(
        tmp_path,
        sample=sample,
        spans=(EvidenceCharacterSpan(start=15, end=19, exact_text="OPEN"),),
    )

    first = factory(sample, canonical, model)
    second = factory(sample, canonical, model)

    assert first == second
    assert first.identity == second.identity
    assert len(factory.identity_sha256) == 64


def test_strict_factory_accepts_rp67_matched_donor_and_reuses_sparse_labels(
    tmp_path: Path,
) -> None:
    anchor_path = "/fixture/anchor.png"
    samples = tuple(
        replace(
            _sample(evidence="The sign reads OPEN.", answer="OPEN"),
            sample_id=f"sample-{index}",
            image=anchor_path,
            image_id="anchor",
            target=f"target-{index}",
            stable_image_uid="anchor-stable",
            item_content_hash=f"anchor-content-{index}",
        )
        for index in range(2)
    )
    span = EvidenceCharacterSpan(start=15, end=19, exact_text="OPEN")
    factory = _factory_for_samples(
        tmp_path,
        samples=samples,
        annotations=tuple(
            (AnswerBearingSpanStatus.RESOLVED, None, (span,)) for _ in samples
        ),
    )
    canonical, model, _ = _supervisions(
        evidence_pieces=("The", " sign", " reads", " OPEN", "."),
        answer_pieces=("OPEN",),
        answer="OPEN",
    )
    base_builder = _SpanApplyingBaseBuilder(factory, canonical, model)
    assignment = ImageAxisDonorAssignment(
        anchor_image_group_key="anchor",
        anchor_image=anchor_path,
        anchor_image_sha256=sha256(b"anchor-image").hexdigest(),
        image_grid_thw=(1, 1, 1),
        donor_sample_id="donor-representative",
        donor_sample_content_sha256=sha256(b"donor-sample").hexdigest(),
        donor_image_group_key="donor",
        donor_image="/fixture/donor.png",
        donor_image_sha256=sha256(b"donor-image").hexdigest(),
        match_tier="exact_grid_same_source_dataset",
    )
    manifest = ImageAxisDonorManifest(
        random_seed=42,
        grid_contract=QwenImageGridContract(16, 2, 1024, 4096),
        source_binding=ImageAxisDonorSourceBinding(
            train_source_sha256="1" * 64,
            retained_manifest_sha256="2" * 64,
            raw_image_manifest_sha256="3" * 64,
            preprocessor_config_sha256="4" * 64,
        ),
        anchor_population_sha256="5" * 64,
        donor_population_sha256="6" * 64,
        assignments=(assignment,),
    )
    wrapper = ImageAxisGroundedNativeGroupBuilder(
        base_builder=base_builder,  # type: ignore[arg-type]
        donor_manifest=manifest,
    )

    group = wrapper(
        samples,
        object(),  # type: ignore[arg-type]
        collective_candidate_count=2,
    )

    assert group.eligible
    assert len(base_builder.calls) == 2
    assert {sample.image for sample in base_builder.calls[1]} == {"/fixture/donor.png"}
    assert all(sample.stable_image_uid is None for sample in base_builder.calls[1])
    assert all(sample.item_content_hash is None for sample in base_builder.calls[1])
    for base_row, donor_row in zip(group.base.rows, group.donor.rows, strict=True):
        assert base_row.loss_supervision == donor_row.loss_supervision
        assert base_row.loss_labels == donor_row.loss_labels
        assert base_row.loss_supervised_token_positions == (
            donor_row.loss_supervised_token_positions
        )
        assert (
            sum(label != EVIDENCE_IGNORE_INDEX for label in base_row.loss_labels) == 2
        )


def test_factory_fails_closed_on_bound_sample_or_transcript_drift(
    tmp_path: Path,
) -> None:
    sample = _sample(evidence="The sign reads OPEN.", answer="OPEN")
    canonical, model, _ = _supervisions(
        evidence_pieces=("The", " sign", " reads", " OPEN", "."),
        answer_pieces=("OPEN",),
        answer="OPEN",
    )
    factory = _factory(
        tmp_path,
        sample=sample,
        spans=(EvidenceCharacterSpan(start=15, end=19, exact_text="OPEN"),),
    )

    with pytest.raises(
        AnswerBearingSpanSupervisionError,
        match="semantic identity differs",
    ):
        factory(
            replace(sample, evidence_description="The sign reads CLOSED."),
            canonical,
            model,
        )
    with pytest.raises(AnswerBearingSpanSupervisionError, match="no record"):
        factory(replace(sample, sample_id="missing"), canonical, model)


def test_final_answer_token_cover_rejects_separator_or_eos_spill(
    tmp_path: Path,
) -> None:
    sample = _sample(evidence="The sign reads OPEN.", answer="OPEN")
    canonical, model, _ = _supervisions(
        evidence_pieces=("The", " sign", " reads", " OPEN", "."),
        answer_pieces=("\n\nOPEN",),
        answer="OPEN",
        answer_piece_includes_middle=True,
    )
    factory = _factory(
        tmp_path,
        sample=sample,
        spans=(EvidenceCharacterSpan(start=15, end=19, exact_text="OPEN"),),
    )

    with pytest.raises(AnswerBearingSpanSupervisionError, match="includes separator"):
        factory(sample, canonical, model)


def test_minimal_token_cover_accepts_overlap_but_rejects_character_gap() -> None:
    text = "abcdef"

    assert _minimal_overlapping_token_positions(
        text,
        ((0, 4), (3, 6)),
        span_start=1,
        span_end=5,
        name="fixture",
    ) == (0, 1)

    with pytest.raises(AnswerBearingSpanSupervisionError, match="leave a gap"):
        _minimal_overlapping_token_positions(
            text,
            ((0, 2), (3, 6)),
            span_start=1,
            span_end=5,
            name="fixture",
        )


class _SpanApplyingBaseBuilder:
    """Minimal native-builder seam that executes the real RP70 callback."""

    def __init__(
        self,
        factory: AnswerBearingSpanSupervisionFactory,
        canonical: CanonicalEvidenceSupervision,
        model: ModelEvidenceSupervision,
    ) -> None:
        self.factory = factory
        self.canonical = canonical
        self.model = model
        self.calls: list[tuple[RepresentationTrainingSample, ...]] = []

    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        _adapter: object,
        *,
        collective_candidate_count: int,
    ) -> SameImageReadoutGroup:
        self.calls.append(samples)
        source_identity = f"source::{samples[0].image_group_key}"
        source = _visual_bundle(0.1)
        rows: list[RepresentationReadoutRow] = []
        candidates: list[RepresentationCandidateObservation] = []
        blocks = self.model.visual_expansion_blocks
        assert len(blocks) == 2
        for index, sample in enumerate(samples):
            sparse = self.factory(sample, self.canonical, self.model)
            sequence = len(self.model.model_token_ids)
            rows.append(
                RepresentationReadoutRow(
                    sample_id=sample.sample_id,
                    image_group_key=sample.image_group_key,
                    source_visual_identity=source_identity,
                    supervision=self.model,
                    input_ids=torch.tensor(
                        (self.model.model_token_ids,), dtype=torch.long
                    ),
                    attention_mask=torch.ones((1, sequence), dtype=torch.long),
                    position_ids=torch.arange(sequence).view(1, sequence),
                    source_positions=blocks[0],
                    d_positions=blocks[1],
                    loss_supervision=sparse,
                )
            )
            candidates.append(
                RepresentationCandidateObservation(
                    sample_id=sample.sample_id,
                    image_group_key=sample.image_group_key,
                    source_visual_identity=source_identity,
                    target_conditioning_provider=(
                        TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
                    ),
                    projection_identities=("main", "branch"),
                    visual=_visual_bundle(0.2 + index * 0.1),
                    image_grid_thw=(1, 1, 1),
                )
            )
        padding = tuple(
            _visual_bundle(0.9)
            for _ in range(collective_candidate_count - len(samples))
        )
        return SameImageReadoutGroup(
            image_group_key=samples[0].image_group_key,
            source_visual_identity=source_identity,
            source_visual=source,
            rows=tuple(rows),
            candidates=tuple(candidates),
            collective_padding=padding,
        )


def _visual_bundle(value: float) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.full((1, 1, 4), value),
        deepstack=(torch.full((1, 1, 4), value + 0.01),),
        branch_layers=(8,),
    )


def _sample(*, evidence: str, answer: str) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id="sample",
        image="/fixture/image.png",
        question="What is shown?",
        target="the relevant region",
        evidence_description=evidence,
        short_answer=answer,
    )


def _factory(
    tmp_path: Path,
    *,
    sample: RepresentationTrainingSample,
    spans: tuple[EvidenceCharacterSpan, ...],
    status: AnswerBearingSpanStatus = AnswerBearingSpanStatus.RESOLVED,
    reason: str | None = None,
) -> AnswerBearingSpanSupervisionFactory:
    return _factory_for_samples(
        tmp_path,
        samples=(sample,),
        annotations=((status, reason, spans),),
    )


def _factory_for_samples(
    tmp_path: Path,
    *,
    samples: tuple[RepresentationTrainingSample, ...],
    annotations: tuple[
        tuple[
            AnswerBearingSpanStatus,
            str | None,
            tuple[EvidenceCharacterSpan, ...],
        ],
        ...,
    ],
) -> AnswerBearingSpanSupervisionFactory:
    manifest = RepresentationDataManifest(
        schema_version=REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION,
        transform_version=REPRESENTATION_DATA_TRANSFORM_VERSION,
        source_path="/fixture/source.jsonl",
        source_sha256="a" * 64,
        accepted_rows=tuple(
            AcceptedRowManifestEntry(
                source_line=ordinal + 1,
                source_row_sha256=sha256(
                    f"fixture-source-row-{ordinal}".encode()
                ).hexdigest(),
                source_image_reference=sample.image,
                resolved_image_path=sample.image,
                sample=sample.identity,
            )
            for ordinal, sample in enumerate(samples)
        ),
        excluded_rows=(),
        duplicate_records=(),
        leakage_records=(),
    )
    dataset = RepresentationDataset(samples=samples, manifest=manifest)
    payload = render_answer_bearing_span_sidecar(
        dataset,
        annotations,
        annotator_identity="test-auditor:v1",
    )
    path = tmp_path / f"span-{sha256(payload).hexdigest()}.jsonl"
    path.write_bytes(payload)
    index = load_answer_bearing_span_index(
        dataset,
        path,
        expected_sidecar_sha256=sha256(payload).hexdigest(),
    )
    return AnswerBearingSpanSupervisionFactory(index)


def _supervisions(
    *,
    evidence_pieces: tuple[str, ...],
    answer_pieces: tuple[str, ...],
    answer: str = "red",
    answer_piece_includes_middle: bool = False,
    visual_expansion_counts: tuple[int, int] = (1, 1),
) -> tuple[CanonicalEvidenceSupervision, ModelEvidenceSupervision, tuple[str, ...]]:
    evidence = "".join(evidence_pieces)
    prefix_pieces = ("<source>", "<focused-d>")
    if answer_piece_includes_middle:
        body_pieces = (*evidence_pieces, *answer_pieces, _SUFFIX)
    else:
        body_pieces = (*evidence_pieces, _MIDDLE, *answer_pieces, _SUFFIX)
    pieces = (*prefix_pieces, *body_pieces)
    text = "".join(pieces)
    expected = "".join(prefix_pieces) + evidence + _MIDDLE + answer + _SUFFIX
    assert text == expected
    token_ids = tuple(range(100, 100 + len(pieces)))
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    evidence_start = len("".join(prefix_pieces))
    evidence_end = evidence_start + len(evidence)
    evidence_positions = tuple(
        position
        for position, (start, _end) in enumerate(offsets)
        if evidence_start <= start < evidence_end
    )
    owned = set(evidence_positions)
    labels = tuple(
        token_id if position in owned else EVIDENCE_IGNORE_INDEX
        for position, token_id in enumerate(token_ids)
    )
    shared = {
        "chat_template_sha256": "1" * 64,
        "tool_schema_sha256": "2" * 64,
        "tokenizer_length": 4096,
    }
    transcript = RenderedTranscript(
        text=text,
        token_ids=token_ids,
        token_ids_sha256=_ids_sha256(token_ids),
        text_sha256=sha256(text.encode("utf-8")).hexdigest(),
        **shared,
    )
    prefix_text = "".join(prefix_pieces)
    generation_prefill = RenderedTranscript(
        text=prefix_text,
        token_ids=token_ids[: len(prefix_pieces)],
        token_ids_sha256=_ids_sha256(token_ids[: len(prefix_pieces)]),
        text_sha256=sha256(prefix_text.encode("utf-8")).hexdigest(),
        **shared,
    )
    canonical = CanonicalEvidenceSupervision(
        transcript=transcript,
        generation_prefill=generation_prefill,
        evidence_text=evidence,
        answer_text=answer,
        canonical_labels=labels,
        evidence_char_start=evidence_start,
        evidence_char_end=evidence_end,
        evidence_byte_start=len(prefix_text.encode("utf-8")),
        evidence_byte_end=len((prefix_text + evidence).encode("utf-8")),
        evidence_token_positions=evidence_positions,
        token_offsets=tuple(offsets),
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    if len(visual_expansion_counts) != 2 or any(
        count <= 0 for count in visual_expansion_counts
    ):
        raise ValueError("fixture visual expansion counts must be positive")
    expansion_values: list[tuple[int, ...]] = []
    model_ids: list[int] = []
    model_surfaces: list[str] = []
    for canonical_position, (token_id, surface) in enumerate(
        zip(token_ids, pieces, strict=True)
    ):
        count = (
            visual_expansion_counts[canonical_position] if canonical_position < 2 else 1
        )
        start = len(model_ids)
        model_ids.extend((token_id,) * count)
        model_surfaces.extend((surface,) * count)
        expansion_values.append(tuple(range(start, start + count)))
    expansion = tuple(expansion_values)
    model_evidence_positions = tuple(
        expansion[position][0] for position in evidence_positions
    )
    model_owned = set(model_evidence_positions)
    model_labels = tuple(
        token_id if position in model_owned else EVIDENCE_IGNORE_INDEX
        for position, token_id in enumerate(model_ids)
    )
    visual_positions = (*expansion[0], *expansion[1])
    model = ModelEvidenceSupervision(
        family="qwen3_vl",
        model_token_ids=tuple(model_ids),
        labels=model_labels,
        evidence_token_positions=model_evidence_positions,
        visual_model_positions=visual_positions,
        canonical_to_model_positions=expansion,
    )
    return canonical, model, tuple(model_surfaces)


def _ids_sha256(token_ids: tuple[int, ...]) -> str:
    return sha256(
        json.dumps(list(token_ids), separators=(",", ":")).encode()
    ).hexdigest()
