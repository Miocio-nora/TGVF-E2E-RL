from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tgvf_rl.representation.experiments.answer_bearing_span import sidecar_tool
from tgvf_rl.representation.experiments.answer_bearing_span.data import (
    VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
    load_answer_bearing_span_index,
)
from tgvf_rl.representation.training.data import (
    REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION,
    REPRESENTATION_DATA_TRANSFORM_VERSION,
    AcceptedRowManifestEntry,
    RepresentationDataManifest,
    RepresentationDataset,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


def test_materializer_loads_bound_split_and_publishes_verified_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    training_path = tmp_path / "training.toml"
    training_path.write_text("strict fixture\n", encoding="utf-8")
    source_path = Path(dataset.manifest.source_path)
    training = _training(training_path, source_path, dataset.manifest.source_sha256)
    calls: list[tuple[object, ...]] = []

    def load_training(path: object) -> object:
        calls.append(("config", path))
        return training

    def load_dataset(
        path: object,
        *,
        expected_source_sha256: str,
        warn_on_leakage: bool,
    ) -> RepresentationDataset:
        calls.append(("dataset", path, expected_source_sha256, warn_on_leakage))
        return dataset

    monkeypatch.setattr(
        sidecar_tool, "load_representation_training_config", load_training
    )
    monkeypatch.setattr(
        sidecar_tool, "load_retained_representation_jsonl", load_dataset
    )

    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [
            {
                "uid": "two",
                "status": "verified_no_answer_bearing_evidence",
                "reason": VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
                "spans": [],
            },
            {
                "uid": "one",
                "status": "resolved",
                "reason": None,
                "spans": [{"start": 7, "end": 10, "exact_text": "red"}],
            },
        ],
    )
    output = tmp_path / "published" / "train-spans.jsonl"
    summary = sidecar_tool.materialize_answer_bearing_span_sidecar(
        training_config_path=training_path,
        split="train",
        annotations_path=annotations,
        annotator_identity="auditor:test-v1",
        output_path=output,
    )

    assert calls == [
        ("config", training_path),
        ("dataset", source_path, dataset.manifest.source_sha256, False),
    ]
    assert output.is_file()
    assert summary["source_sha256"] == dataset.manifest.source_sha256
    assert summary["sidecar_sha256"] == sha256(output.read_bytes()).hexdigest()
    assert summary["statistics"] == {
        "total_rows": 2,
        "resolved_rows": 1,
        "verified_no_answer_bearing_evidence_rows": 1,
        "multiple_span_rows": 0,
        "total_spans": 1,
    }
    index = load_answer_bearing_span_index(
        dataset,
        output,
        expected_sidecar_sha256=str(summary["sidecar_sha256"]),
    )
    assert summary["index_sha256"] == index.identity_sha256
    assert summary["population_sha256"] == index.retained_semantic_population_sha256

    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sidecar_tool.materialize_answer_bearing_span_sidecar(
            training_config_path=training_path,
            split="train",
            annotations_path=annotations,
            annotator_identity="auditor:test-v1",
            output_path=output,
        )
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "exactly one row"),
        ("extra", "exactly one row"),
        ("duplicate", "duplicate UIDs"),
        ("unknown", "unknown UIDs"),
        ("unresolved", "unsupported or unresolved status"),
        ("drift", "drifted"),
        ("row-extra-field", "fields differ"),
        ("span-extra-field", "fields differ"),
    ],
)
def test_materializer_rejects_incomplete_or_drifted_annotations_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    dataset = _dataset(tmp_path)
    training_path = tmp_path / "training.toml"
    training_path.write_text("strict fixture\n", encoding="utf-8")
    training = _training(
        training_path,
        Path(dataset.manifest.source_path),
        dataset.manifest.source_sha256,
    )
    monkeypatch.setattr(
        sidecar_tool,
        "load_representation_training_config",
        lambda _path: training,
    )
    monkeypatch.setattr(
        sidecar_tool,
        "load_retained_representation_jsonl",
        lambda *_args, **_kwargs: dataset,
    )
    rows = _valid_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(dict(rows[-1], uid="extra"))
    elif mutation == "duplicate":
        rows[1]["uid"] = rows[0]["uid"]
    elif mutation == "unknown":
        rows[1]["uid"] = "unknown"
    elif mutation == "unresolved":
        rows[0]["status"] = "unresolved"
    elif mutation == "drift":
        rows[0]["spans"][0]["exact_text"] = "RED"
    elif mutation == "row-extra-field":
        rows[0]["note"] = "not allowed"
    elif mutation == "span-extra-field":
        rows[0]["spans"][0]["confidence"] = 1.0
    else:  # pragma: no cover - parameter list is closed above
        raise AssertionError(mutation)
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(annotations, rows)
    output = tmp_path / "must-not-exist" / "sidecar.jsonl"

    with pytest.raises(sidecar_tool.AnswerBearingSpanAnnotationError, match=match):
        sidecar_tool.materialize_answer_bearing_span_sidecar(
            training_config_path=training_path,
            split="validation",
            annotations_path=annotations,
            annotator_identity="auditor:test-v1",
            output_path=output,
        )

    assert not output.exists()
    assert not output.parent.exists()


def test_main_prints_machine_readable_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "source_sha256": "a" * 64,
        "sidecar_sha256": "b" * 64,
        "index_sha256": "c" * 64,
        "population_sha256": "d" * 64,
        "statistics": {"total_rows": 2},
    }
    captured: dict[str, object] = {}

    def materialize(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        sidecar_tool,
        "materialize_answer_bearing_span_sidecar",
        materialize,
    )
    status = sidecar_tool.main(
        [
            "--training-config",
            str(tmp_path / "training.toml"),
            "--split",
            "train",
            "--annotations",
            str(tmp_path / "annotations.jsonl"),
            "--annotator-identity",
            "auditor:test-v1",
            "--output",
            str(tmp_path / "output.jsonl"),
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert captured["split"] == "train"


def _valid_rows() -> list[dict[str, Any]]:
    return [
        {
            "uid": "one",
            "status": "resolved",
            "reason": None,
            "spans": [{"start": 7, "end": 10, "exact_text": "red"}],
        },
        {
            "uid": "two",
            "status": "verified_no_answer_bearing_evidence",
            "reason": VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
            "spans": [],
        },
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _training(
    path: Path,
    source_path: Path,
    source_sha256: str,
) -> SimpleNamespace:
    split = SimpleNamespace(jsonl_path=source_path, source_sha256=source_sha256)
    return SimpleNamespace(
        source_path=path.resolve(),
        source_toml_sha256=sha256(path.read_bytes()).hexdigest(),
        canonical_config_sha256="f" * 64,
        data=SimpleNamespace(
            train=split,
            validation=split,
            warn_on_target_leakage=False,
        ),
    )


def _dataset(tmp_path: Path) -> RepresentationDataset:
    samples = (
        RepresentationTrainingSample(
            sample_id="one",
            image=str(tmp_path / "one.png"),
            image_id="image-one",
            question="What color?",
            target="the object color",
            evidence_description="color: red",
            short_answer="red",
        ),
        RepresentationTrainingSample(
            sample_id="two",
            image=str(tmp_path / "two.png"),
            image_id="image-two",
            question="What total?",
            target="the displayed total",
            evidence_description="inputs 3 and 5",
            short_answer="8",
        ),
    )
    source_path = tmp_path / "source.jsonl"
    source_path.write_text("fixture source\n", encoding="utf-8")
    source_sha256 = sha256(source_path.read_bytes()).hexdigest()
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
    return RepresentationDataset(
        samples=samples,
        manifest=RepresentationDataManifest(
            schema_version=REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION,
            transform_version=REPRESENTATION_DATA_TRANSFORM_VERSION,
            source_path=str(source_path.resolve()),
            source_sha256=source_sha256,
            accepted_rows=accepted,
            excluded_rows=(),
            duplicate_records=(),
            leakage_records=(),
        ),
    )
