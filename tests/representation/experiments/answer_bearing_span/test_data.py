from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from tgvf_rl.representation.experiments.answer_bearing_span.data import (
    ANSWER_BEARING_SPAN_INDEX_SCHEMA_VERSION,
    ANSWER_BEARING_SPAN_MATCH_POLICY,
    ANSWER_BEARING_SPAN_SIDECAR_SCHEMA_VERSION,
    VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
    AnswerBearingSpanDataError,
    AnswerBearingSpanStatus,
    EvidenceCharacterSpan,
    answer_bearing_span_population_sha256,
    answer_bearing_span_semantic_sha256,
    load_answer_bearing_span_index,
    merge_answer_bearing_span_indices,
    render_answer_bearing_span_sidecar,
)
from tgvf_rl.representation.training.data import (
    REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION,
    REPRESENTATION_DATA_TRANSFORM_VERSION,
    AcceptedRowManifestEntry,
    RepresentationDataManifest,
    RepresentationDataset,
)
from tgvf_rl.representation.training.schema import (
    RepresentationChoice,
    RepresentationTrainingSample,
)


def test_loader_binds_complete_semantic_population_and_explicit_statuses(
    tmp_path: Path,
) -> None:
    samples = (
        _sample("one", evidence="red and red.", answer="red"),
        _sample("two", evidence="The inputs are 3 and 5.", answer="8"),
    )
    dataset = _dataset(samples, source_sha256="a" * 64)
    sidecar, digest = _write_sidecar(
        tmp_path / "spans.jsonl",
        dataset,
        annotations=(
            (
                "resolved",
                ((0, 3, "red"), (8, 11, "red")),
                None,
            ),
            (
                "verified_no_answer_bearing_evidence",
                (),
                VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
            ),
        ),
    )

    index = load_answer_bearing_span_index(
        dataset,
        sidecar,
        expected_sidecar_sha256=digest,
    )

    assert index.schema_version == ANSWER_BEARING_SPAN_INDEX_SCHEMA_VERSION
    assert index.match_policy == ANSWER_BEARING_SPAN_MATCH_POLICY
    assert index.sidecar_sha256 == digest
    assert index.retained_semantic_population_sha256 == (
        answer_bearing_span_population_sha256(dataset)
    )
    assert tuple(record.uid for record in index.records) == ("one", "two")
    assert index.record_for("one").status is AnswerBearingSpanStatus.RESOLVED
    assert tuple(
        span.exact_text for span in index.record_for("one").value_character_spans
    ) == ("red", "red")
    assert index.record_for("two").value_character_spans == ()
    assert index.statistics.total_rows == 2
    assert index.statistics.resolved_rows == 1
    assert index.statistics.verified_no_answer_bearing_evidence_rows == 1
    assert index.statistics.multiple_span_rows == 1
    assert index.statistics.total_spans == 2
    assert index.statistics.matched_rows == 1
    assert index.statistics.unmatched_rows == 1
    assert index.statistics.coverage == pytest.approx(0.5)
    assert index.by_uid is index.by_uid
    assert len(index.identity_sha256) == 64


def test_offsets_are_python_unicode_code_points_and_exact_text_is_mandatory(
    tmp_path: Path,
) -> None:
    sample = _sample("unicode", evidence="α😀蓝色β", answer="蓝色")
    dataset = _dataset((sample,), source_sha256="b" * 64)
    sidecar, digest = _write_sidecar(
        tmp_path / "unicode.jsonl",
        dataset,
        annotations=(("resolved", ((2, 4, "蓝色"),), None),),
    )

    index = load_answer_bearing_span_index(
        dataset,
        sidecar,
        expected_sidecar_sha256=digest,
    )

    span = index.records[0].value_character_spans[0]
    assert sample.evidence_description[span.start : span.end] == "蓝色"

    rows = _read_jsonl(sidecar)
    rows[1]["spans"][0] = {"start": 1, "end": 3, "exact_text": "蓝色"}
    drifted, drifted_sha = _write_rows(tmp_path / "unicode-drift.jsonl", rows)
    with pytest.raises(AnswerBearingSpanDataError, match="exact_text differs"):
        load_answer_bearing_span_index(
            dataset,
            drifted,
            expected_sidecar_sha256=drifted_sha,
        )


@pytest.mark.parametrize(
    "mutate, message",
    (
        (
            lambda rows: rows[0].__setitem__(
                "retained_semantic_population_sha256", "f" * 64
            ),
            "retained semantic population digest differs",
        ),
        (
            lambda rows: rows[0].__setitem__("retained_count", 2),
            "statistics disagree with retained_count",
        ),
        (
            lambda rows: rows[0].__setitem__("annotator_identity", ""),
            "annotator_identity",
        ),
        (
            lambda rows: rows[0]["status_statistics"].__setitem__("total_spans", 2),
            "statistics differ",
        ),
    ),
)
def test_header_population_count_statistics_and_annotator_fail_closed(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    dataset = _dataset(
        (_sample("one", evidence="The value is red.", answer="red"),),
        source_sha256="c" * 64,
    )
    sidecar, _ = _write_sidecar(
        tmp_path / "base.jsonl",
        dataset,
        annotations=(("resolved", ((13, 16, "red"),), None),),
    )
    rows = _read_jsonl(sidecar)
    mutate(rows)
    mutated, digest = _write_rows(tmp_path / f"mutated-{message[:4]}.jsonl", rows)

    with pytest.raises(AnswerBearingSpanDataError, match=message):
        load_answer_bearing_span_index(
            dataset,
            mutated,
            expected_sidecar_sha256=digest,
        )


def test_missing_extra_and_semantic_drift_fail_but_reordering_is_allowed(
    tmp_path: Path,
) -> None:
    samples = (
        _sample("one", evidence="red", answer="red"),
        _sample("two", evidence="blue", answer="blue"),
    )
    dataset = _dataset(samples, source_sha256="d" * 64)
    sidecar, _ = _write_sidecar(
        tmp_path / "base.jsonl",
        dataset,
        annotations=(
            ("resolved", ((0, 3, "red"),), None),
            ("resolved", ((0, 4, "blue"),), None),
        ),
    )
    rows = _read_jsonl(sidecar)

    missing, missing_sha = _write_rows(tmp_path / "missing.jsonl", rows[:-1])
    with pytest.raises(AnswerBearingSpanDataError, match="exactly one record"):
        load_answer_bearing_span_index(
            dataset,
            missing,
            expected_sidecar_sha256=missing_sha,
        )

    extra, extra_sha = _write_rows(tmp_path / "extra.jsonl", (*rows, rows[-1]))
    with pytest.raises(AnswerBearingSpanDataError, match="exactly one record"):
        load_answer_bearing_span_index(
            dataset,
            extra,
            expected_sidecar_sha256=extra_sha,
        )

    reordered, reordered_sha = _write_rows(
        tmp_path / "reordered.jsonl",
        (rows[0], rows[2], rows[1]),
    )
    reordered_index = load_answer_bearing_span_index(
        dataset,
        reordered,
        expected_sidecar_sha256=reordered_sha,
    )
    assert tuple(record.uid for record in reordered_index.records) == ("one", "two")

    drifted_dataset = _dataset(
        (
            _sample("one", evidence="RED", answer="red"),
            samples[1],
        ),
        source_sha256="d" * 64,
    )
    original_sha = sha256(sidecar.read_bytes()).hexdigest()
    with pytest.raises(
        AnswerBearingSpanDataError, match="semantic population digest differs"
    ):
        load_answer_bearing_span_index(
            drifted_dataset,
            sidecar,
            expected_sidecar_sha256=original_sha,
        )


def test_binding_tolerates_visual_provenance_source_line_and_dataset_reorder(
    tmp_path: Path,
) -> None:
    samples = (
        _sample("one", evidence="value red", answer="red"),
        _sample("two", evidence="value blue", answer="blue"),
    )
    original = _dataset(samples, source_sha256="7" * 64)
    sidecar, sidecar_sha = _write_sidecar(
        tmp_path / "semantic.jsonl",
        original,
        annotations=(
            ("resolved", ((6, 9, "red"),), None),
            ("resolved", ((6, 10, "blue"),), None),
        ),
    )
    donor_compatible = _dataset(
        tuple(
            replace(
                sample,
                image=f"/different/{sample.sample_id}.png",
                image_id=f"donor-{sample.sample_id}",
                stable_image_uid=None,
                item_content_hash=None,
                source_dataset="changed-provenance",
                source_profile="unrelated-metadata",
            )
            for sample in reversed(samples)
        ),
        source_sha256="8" * 64,
    )

    index = load_answer_bearing_span_index(
        donor_compatible,
        sidecar,
        expected_sidecar_sha256=sidecar_sha,
    )

    assert tuple(record.uid for record in index.records) == ("one", "two")
    assert index.retained_semantic_population_sha256 == (
        answer_bearing_span_population_sha256(original)
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("question", "A different question?"),
        ("target", "a different target"),
        ("evidence_description", "value green"),
        ("short_answer", "green"),
        ("choices", (RepresentationChoice(label="A", text="green"),)),
    ),
)
def test_binding_rejects_model_visible_semantic_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    sample = _sample("one", evidence="value red", answer="red")
    original = _dataset((sample,), source_sha256="9" * 64)
    sidecar, sidecar_sha = _write_sidecar(
        tmp_path / f"semantic-drift-{field}.jsonl",
        original,
        annotations=(("resolved", ((6, 9, "red"),), None),),
    )
    drifted = _dataset(
        (replace(sample, **{field: value}),),
        source_sha256="0" * 64,
    )

    with pytest.raises(
        AnswerBearingSpanDataError,
        match="semantic population digest differs",
    ):
        load_answer_bearing_span_index(
            drifted,
            sidecar,
            expected_sidecar_sha256=sidecar_sha,
        )


def test_loader_rejects_duplicate_and_unknown_sidecar_uids(tmp_path: Path) -> None:
    dataset = _dataset(
        (
            _sample("one", evidence="red", answer="red"),
            _sample("two", evidence="blue", answer="blue"),
        ),
        source_sha256="6" * 64,
    )
    sidecar, _ = _write_sidecar(
        tmp_path / "uid-base.jsonl",
        dataset,
        annotations=(
            ("resolved", ((0, 3, "red"),), None),
            ("resolved", ((0, 4, "blue"),), None),
        ),
    )
    rows = _read_jsonl(sidecar)
    rows[2]["uid"] = "one"
    duplicate, duplicate_sha = _write_rows(tmp_path / "duplicate.jsonl", rows)
    with pytest.raises(AnswerBearingSpanDataError, match="duplicate UID"):
        load_answer_bearing_span_index(
            dataset,
            duplicate,
            expected_sidecar_sha256=duplicate_sha,
        )

    rows = _read_jsonl(sidecar)
    rows[2]["uid"] = "unknown"
    unknown, unknown_sha = _write_rows(tmp_path / "unknown.jsonl", rows)
    with pytest.raises(AnswerBearingSpanDataError, match="unknown UID"):
        load_answer_bearing_span_index(
            dataset,
            unknown,
            expected_sidecar_sha256=unknown_sha,
        )


@pytest.mark.parametrize(
    "spans, message",
    (
        (((4, 7, "def"), (0, 3, "abc")), "sorted and unique"),
        (((0, 3, "abc"), (0, 3, "abc")), "sorted and unique"),
        (((0, 3, "abc"), (2, 5, "cde")), "must not overlap"),
        (((0, 7, "abcdefg"),), "outside evidence_description"),
        (((0, 3, "abd"),), "exact_text differs"),
    ),
)
def test_spans_must_be_sorted_unique_nonoverlapping_in_bounds_and_exact(
    tmp_path: Path,
    spans: tuple[tuple[int, int, str], ...],
    message: str,
) -> None:
    dataset = _dataset(
        (_sample("one", evidence="abcdef", answer="abc"),),
        source_sha256="e" * 64,
    )
    sidecar, digest = _write_sidecar(
        tmp_path / f"invalid-{message[:4]}.jsonl",
        dataset,
        annotations=(("resolved", spans, None),),
    )

    with pytest.raises(AnswerBearingSpanDataError, match=message):
        load_answer_bearing_span_index(
            dataset,
            sidecar,
            expected_sidecar_sha256=digest,
        )


@pytest.mark.parametrize(
    "status, spans, reason, message",
    (
        ("unresolved", (), None, "unsupported or unresolved status"),
        ("resolved", (), None, "inconsistent with resolved rows|require non-empty"),
        (
            "resolved",
            ((0, 3, "red"),),
            VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
            "reason=null",
        ),
        (
            "verified_no_answer_bearing_evidence",
            ((0, 3, "red"),),
            VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
            "require empty spans",
        ),
        (
            "verified_no_answer_bearing_evidence",
            (),
            "free-form reason",
            "invalid reason",
        ),
    ),
)
def test_only_fully_resolved_or_fixed_verified_empty_status_is_accepted(
    tmp_path: Path,
    status: str,
    spans: tuple[tuple[int, int, str], ...],
    reason: str | None,
    message: str,
) -> None:
    dataset = _dataset(
        (_sample("one", evidence="red", answer="red"),),
        source_sha256="f" * 64,
    )
    sidecar, digest = _write_sidecar(
        tmp_path / f"status-{status}.jsonl",
        dataset,
        annotations=((status, spans, reason),),
    )

    with pytest.raises(AnswerBearingSpanDataError, match=message):
        load_answer_bearing_span_index(
            dataset,
            sidecar,
            expected_sidecar_sha256=digest,
        )


def test_identity_binds_sidecar_bytes_canonical_content_and_index_order(
    tmp_path: Path,
) -> None:
    train_data = _dataset(
        (_sample("train", evidence="OPEN", answer="OPEN"),),
        source_sha256="1" * 64,
    )
    test_data = _dataset(
        (_sample("test", evidence="CLOSED", answer="CLOSED"),),
        source_sha256="2" * 64,
    )
    train_path, train_sha = _write_sidecar(
        tmp_path / "train.jsonl",
        train_data,
        annotations=(("resolved", ((0, 4, "OPEN"),), None),),
    )
    copy_path, copy_sha = _write_sidecar(
        tmp_path / "train-copy.jsonl",
        train_data,
        annotations=(("resolved", ((0, 4, "OPEN"),), None),),
    )
    test_path, test_sha = _write_sidecar(
        tmp_path / "test.jsonl",
        test_data,
        annotations=(("resolved", ((0, 6, "CLOSED"),), None),),
    )
    train = load_answer_bearing_span_index(
        train_data,
        train_path,
        expected_sidecar_sha256=train_sha,
    )
    copy = load_answer_bearing_span_index(
        train_data,
        copy_path,
        expected_sidecar_sha256=copy_sha,
    )
    test = load_answer_bearing_span_index(
        test_data,
        test_path,
        expected_sidecar_sha256=test_sha,
    )

    assert train.identity_sha256 == copy.identity_sha256
    merged = merge_answer_bearing_span_indices(train, test)
    assert tuple(record.uid for record in merged.records) == ("train", "test")
    assert merged.by_uid is merged.by_uid
    assert merge_answer_bearing_span_indices(test, train).identity_sha256 != (
        merged.identity_sha256
    )
    with pytest.raises(ValueError, match="repeats a source index"):
        merge_answer_bearing_span_indices(train, copy)

    rows = _read_jsonl(train_path)
    rows[0]["annotator_identity"] = "different-auditor:v1"
    changed_path, changed_sha = _write_rows(tmp_path / "changed.jsonl", rows)
    changed = load_answer_bearing_span_index(
        train_data,
        changed_path,
        expected_sidecar_sha256=changed_sha,
    )
    assert changed.identity_sha256 != train.identity_sha256


def test_loader_rejects_wrong_sidecar_sha_before_parsing(tmp_path: Path) -> None:
    dataset = _dataset(
        (_sample("one", evidence="red", answer="red"),),
        source_sha256="3" * 64,
    )
    sidecar, _ = _write_sidecar(
        tmp_path / "source.jsonl",
        dataset,
        annotations=(("resolved", ((0, 3, "red"),), None),),
    )

    with pytest.raises(AnswerBearingSpanDataError, match="sidecar SHA256 mismatch"):
        load_answer_bearing_span_index(
            dataset,
            sidecar,
            expected_sidecar_sha256="0" * 64,
        )


def test_canonical_writer_round_trips_complete_ordered_annotations(
    tmp_path: Path,
) -> None:
    dataset = _dataset(
        (
            _sample("one", evidence="red", answer="red"),
            _sample("two", evidence="inputs 3 and 5", answer="8"),
        ),
        source_sha256="4" * 64,
    )
    annotations = (
        (
            AnswerBearingSpanStatus.RESOLVED,
            None,
            (EvidenceCharacterSpan(start=0, end=3, exact_text="red"),),
        ),
        (
            AnswerBearingSpanStatus.VERIFIED_NO_ANSWER_BEARING_EVIDENCE,
            VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
            (),
        ),
    )

    first = render_answer_bearing_span_sidecar(
        dataset,
        annotations,
        annotator_identity="test-auditor:v1",
    )
    second = render_answer_bearing_span_sidecar(
        dataset,
        annotations,
        annotator_identity="test-auditor:v1",
    )

    assert first == second
    assert first.endswith(b"\n")
    path = tmp_path / "canonical.jsonl"
    path.write_bytes(first)
    index = load_answer_bearing_span_index(
        dataset,
        path,
        expected_sidecar_sha256=sha256(first).hexdigest(),
    )
    assert index.statistics.resolved_rows == 1
    assert index.statistics.verified_no_answer_bearing_evidence_rows == 1


def _sample(
    uid: str,
    *,
    evidence: str,
    answer: str,
) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=uid,
        image=f"/fixture/{uid}.png",
        image_id=f"image-{uid}",
        question="What is shown?",
        target=f"the relevant region for {uid}",
        evidence_description=evidence,
        short_answer=answer,
    )


def _dataset(
    samples: tuple[RepresentationTrainingSample, ...],
    *,
    source_sha256: str,
) -> RepresentationDataset:
    accepted = tuple(
        AcceptedRowManifestEntry(
            source_line=ordinal + 1,
            source_row_sha256=sha256(
                f"source-row-{ordinal}-{sample.sample_id}".encode()
            ).hexdigest(),
            source_image_reference=sample.image,
            resolved_image_path=sample.image,
            sample=sample.identity,
        )
        for ordinal, sample in enumerate(samples)
    )
    manifest = RepresentationDataManifest(
        schema_version=REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION,
        transform_version=REPRESENTATION_DATA_TRANSFORM_VERSION,
        source_path="/fixture/source.jsonl",
        source_sha256=source_sha256,
        accepted_rows=accepted,
        excluded_rows=(),
        duplicate_records=(),
        leakage_records=(),
    )
    return RepresentationDataset(samples=samples, manifest=manifest)


def _write_sidecar(
    path: Path,
    dataset: RepresentationDataset,
    *,
    annotations: tuple[tuple[str, tuple[tuple[int, int, str], ...], str | None], ...],
) -> tuple[Path, str]:
    assert len(annotations) == len(dataset.samples)
    resolved = sum(status == "resolved" for status, _spans, _reason in annotations)
    verified = len(annotations) - resolved
    multiple = sum(len(spans) > 1 for _status, spans, _reason in annotations)
    total_spans = sum(len(spans) for _status, spans, _reason in annotations)
    rows: list[dict[str, Any]] = [
        {
            "record_type": "header",
            "schema_version": ANSWER_BEARING_SPAN_SIDECAR_SCHEMA_VERSION,
            "policy": ANSWER_BEARING_SPAN_MATCH_POLICY,
            "retained_semantic_population_sha256": (
                answer_bearing_span_population_sha256(dataset)
            ),
            "retained_count": len(dataset.samples),
            "status_statistics": {
                "total_rows": len(dataset.samples),
                "resolved_rows": resolved,
                "verified_no_answer_bearing_evidence_rows": verified,
                "multiple_span_rows": multiple,
                "total_spans": total_spans,
            },
            "annotator_identity": "test-auditor:v1",
        }
    ]
    for sample, annotation in zip(
        dataset.samples,
        annotations,
        strict=True,
    ):
        status, spans, reason = annotation
        rows.append(
            {
                "record_type": "sample",
                "uid": sample.sample_id,
                "semantic_content_sha256": (
                    answer_bearing_span_semantic_sha256(sample)
                ),
                "question_sha256": _text_sha256(sample.question),
                "target_sha256": _text_sha256(sample.target),
                "evidence_description_sha256": _text_sha256(
                    sample.evidence_description
                ),
                "short_answer_sha256": _text_sha256(sample.short_answer),
                "choices_sha256": _canonical_sha256([]),
                "status": status,
                "reason": reason,
                "spans": [
                    {"start": start, "end": end, "exact_text": exact_text}
                    for start, end, exact_text in spans
                ],
            }
        )
    return _write_rows(path, rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_rows(
    path: Path,
    rows: Any,
) -> tuple[Path, str]:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return path, sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()
