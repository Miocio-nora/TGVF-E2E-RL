from __future__ import annotations

import json

import pytest

from tgvf_rl.representation.experiments.answer_bearing_span import (
    annotation_overrides,
)
from tgvf_rl.representation.experiments.answer_bearing_span.data import (
    AnswerBearingSpanStatus,
    EvidenceCharacterSpan,
    VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


def _sample(uid: str = "one") -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=uid,
        image="image.png",
        question="Which shape is red?",
        target="Find the red shape.",
        evidence_description="red square, blue circle",
        short_answer="square",
    )


def _line(
    uid: str,
    *,
    status: str = "resolved",
    token_indices: list[int] | None = None,
) -> str:
    return json.dumps(
        {
            "uid": uid,
            "status": status,
            "token_indices": [0] if token_indices is None else token_indices,
        }
    )


def test_partial_overrides_are_strict_and_semantically_validated(tmp_path) -> None:
    path = tmp_path / "overrides.jsonl"
    path.write_text(_line("one") + "\n", encoding="utf-8")
    overrides, source, digest = annotation_overrides._load_overrides(
        path,
        samples_by_uid={"one": _sample(), "two": _sample("two")},
    )
    assert source == path.resolve()
    assert len(digest) == 64
    assert overrides == {
        "one": (
            AnswerBearingSpanStatus.RESOLVED,
            None,
            (EvidenceCharacterSpan(start=0, end=3, exact_text="red"),),
        )
    }

    path.write_text(_line("one") + "\n" + _line("one") + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="duplicate UIDs"):
        annotation_overrides._load_overrides(
            path,
            samples_by_uid={"one": _sample()},
        )

    path.write_text(_line("unknown") + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="unknown UIDs"):
        annotation_overrides._load_overrides(
            path,
            samples_by_uid={"one": _sample()},
        )

    path.write_text(_line("one", token_indices=[99]) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="out of bounds"):
        annotation_overrides._load_overrides(
            path,
            samples_by_uid={"one": _sample()},
        )

    path.write_text(_line("one", token_indices=[-1]) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="out of bounds"):
        annotation_overrides._load_overrides(
            path,
            samples_by_uid={"one": _sample()},
        )

    path.write_text(_line("one", token_indices=[1, 0]) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="sorted and unique"):
        annotation_overrides._load_overrides(
            path,
            samples_by_uid={"one": _sample()},
        )


def test_token_indices_are_merged_into_exact_character_spans(tmp_path) -> None:
    path = tmp_path / "overrides.jsonl"
    path.write_text(_line("one", token_indices=[0, 1, 3]) + "\n", encoding="utf-8")
    overrides, _, _ = annotation_overrides._load_overrides(
        path,
        samples_by_uid={"one": _sample()},
    )
    assert overrides["one"][2] == (
        EvidenceCharacterSpan(start=0, end=10, exact_text="red square"),
        EvidenceCharacterSpan(start=12, end=16, exact_text="blue"),
    )


def test_reviewed_no_span_uses_canonical_reason(tmp_path) -> None:
    path = tmp_path / "overrides.jsonl"
    path.write_text(
        _line("one", status="no_span", token_indices=[]) + "\n",
        encoding="utf-8",
    )
    overrides, _, _ = annotation_overrides._load_overrides(
        path,
        samples_by_uid={"one": _sample()},
    )
    assert overrides["one"] == (
        AnswerBearingSpanStatus.VERIFIED_NO_ANSWER_BEARING_EVIDENCE,
        VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
        (),
    )


def test_render_annotations_is_canonical_and_supports_reviewed_no_span() -> None:
    payload = annotation_overrides._render_annotations(
        sample_uids=("resolved", "none"),
        annotations=(
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
        ),
    )
    rows = [json.loads(line) for line in payload.splitlines()]
    assert rows[0]["spans"] == [{"end": 3, "exact_text": "red", "start": 0}]
    assert rows[1] == {
        "reason": VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
        "spans": [],
        "status": "verified_no_answer_bearing_evidence",
        "uid": "none",
    }


def test_render_rejects_population_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        annotation_overrides._render_annotations(
            sample_uids=("one",),
            annotations=(),
        )
