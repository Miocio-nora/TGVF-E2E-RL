from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tgvf_rl.data import policy_selection_sources as implementation


PIL = pytest.importorskip("PIL.Image")


def _image_bytes() -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    PIL.new("RGB", (32, 24), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_image_bytes())


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_vstar_image_manifest(
    image_root: Path, annotation_hashes: dict[str, str]
) -> None:
    descriptor = {
        "schema_version": implementation.VSTAR_IMAGE_MANIFEST_SCHEMA,
        "annotation_files": annotation_hashes,
        "archives": {"fixture": {"sha256": "0" * 64}},
        "image_count": 4,
        "extraction": "fixture",
    }
    content_sha256 = hashlib.sha256(
        implementation._canonical_json_bytes(descriptor)
    ).hexdigest()
    (image_root / "manifest.json").write_bytes(
        implementation._canonical_json_bytes(
            {**descriptor, "content_sha256": content_sha256}
        )
        + b"\n"
    )


def test_vstar_image_materializer_extracts_only_required_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotations = tmp_path / "annotations"
    archives = tmp_path / "archives"
    annotations.mkdir()
    archives.mkdir()
    payload = _image_bytes()
    annotation_rows = {
        "GQA_data.json": [{"image": "gqa/images/gqa.jpg"}],
        "llava_focus_data.json": [{"image": "coco2017/train2017/coco2017.jpg"}],
        "spatial_relation_data.json": [{"image": "coco2014/train2014/coco2014.jpg"}],
        "vaw_attribute_data.json": [{"image": "gqa/images/gqa.jpg"}],
    }
    annotation_hashes = {}
    for filename, rows in annotation_rows.items():
        source = annotations / filename
        source.write_text(json.dumps(rows))
        annotation_hashes[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    archive_members = {
        "gqa": ("gqa-images.zip", "images/gqa.jpg"),
        "coco2014": ("train2014.zip", "train2014/coco2014.jpg"),
        "coco2017": ("train2017.zip", "train2017/coco2017.jpg"),
    }
    specs = {}
    for source, (filename, member) in archive_members.items():
        archive_path = archives / filename
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(member, payload)
            archive.writestr(f"unused/{source}.jpg", payload)
        specs[source] = {
            "filename": filename,
            "expected_bytes": archive_path.stat().st_size,
            "member_prefix": member.split("/", 1)[0],
            "source_url": f"fixture://{filename}",
        }
    monkeypatch.setattr(implementation, "VSTAR_ANNOTATION_FILES", annotation_hashes)
    monkeypatch.setattr(implementation, "VSTAR_IMAGE_ARCHIVE_SPECS", specs)

    result = implementation.materialize_vstar_images(
        annotations, archives, tmp_path / "images"
    )

    assert result.image_count == 3
    assert (result.output_root / "gqa/images/gqa.jpg").read_bytes() == payload
    assert not (result.output_root / "unused").exists()
    manifest = json.loads((result.output_root / "manifest.json").read_text())
    assert manifest["image_count"] == 3


def test_vstar_source_adapter_extracts_conversation_and_canonical_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    annotations.mkdir()
    source_files = (
        "GQA_data.json",
        "llava_focus_data.json",
        "spatial_relation_data.json",
        "vaw_attribute_data.json",
    )
    source_hashes: dict[str, str] = {}
    relative_images = (
        "gqa/images/gqa-a.jpg",
        "coco2017/train2017/coco-a.jpg",
        "coco2014/train2014/coco-b.jpg",
        "gqa/images/gqa-b.jpg",
    )
    for index, (filename, relative_image) in enumerate(
        zip(source_files, relative_images, strict=True)
    ):
        _write_image(images / relative_image)
        row = {
            "image": relative_image,
            "target_instances": [
                {
                    "name": "detail",
                    "instance_id": index,
                    "bbox": [-1.2, 2.1, 8.4, 9.2],
                }
            ],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nAdditional visual information to focus on: <object>.\nWhat is visible?",
                },
                {"from": "gpt", "value": "A detail."},
            ],
            "search": 1,
        }
        payload = json.dumps([row]).encode()
        (annotations / filename).write_bytes(payload)
        source_hashes[filename] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(implementation, "VSTAR_ANNOTATION_FILES", source_hashes)
    monkeypatch.setattr(implementation, "VSTAR_SOURCE_QUALITY_QUARANTINES", ())
    _write_vstar_image_manifest(images, source_hashes)

    result = implementation.materialize_vstar_candidates(
        annotations, images, tmp_path / "output"
    )
    records = _records(result.output_root / "candidates.jsonl")

    assert result.source_rows == result.candidate_rows == 4
    assert result.rejected_rows == 0
    assert records[0]["question"] == "What is visible?"
    assert records[0]["ground_truth"] == "A detail."
    assert records[0]["gt_regions"] == [[0, 2, 8, 12]]


def test_vstar_source_quality_quarantine_is_exact_audited_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    annotations.mkdir()
    source_files = (
        "GQA_data.json",
        "llava_focus_data.json",
        "spatial_relation_data.json",
        "vaw_attribute_data.json",
    )
    relative_images = (
        "gqa/images/gqa-a.jpg",
        "coco2017/train2017/coco-a.jpg",
        "coco2014/train2014/coco-b.jpg",
        "gqa/images/gqa-b.jpg",
    )
    rows_by_file: dict[str, list[dict[str, object]]] = {}
    source_hashes: dict[str, str] = {}
    for index, (filename, relative_image) in enumerate(
        zip(source_files, relative_images, strict=True)
    ):
        _write_image(images / relative_image)
        row: dict[str, object] = {
            "image": relative_image,
            "target_instances": [
                {
                    "name": "detail",
                    "instance_id": index,
                    "bbox": [1, 2, 8, 9],
                }
            ],
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is visible?"},
                {"from": "gpt", "value": "There are"},
            ],
            "search": 1,
        }
        rows_by_file[filename] = [row]
        payload = json.dumps([row]).encode()
        (annotations / filename).write_bytes(payload)
        source_hashes[filename] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(implementation, "VSTAR_ANNOTATION_FILES", source_hashes)
    _write_vstar_image_manifest(images, source_hashes)

    monkeypatch.setattr(implementation, "VSTAR_SOURCE_QUALITY_QUARANTINES", ())
    baseline = implementation.materialize_vstar_candidates(
        annotations, images, tmp_path / "baseline"
    )
    baseline_records = _records(baseline.output_root / "candidates.jsonl")
    quarantined_record = next(
        record
        for record in baseline_records
        if record["provenance"]["source_file"] == "llava_focus_data.json"
    )
    candidate = implementation.SelectionCandidate.from_record(quarantined_record)
    quarantined_row = rows_by_file["llava_focus_data.json"][0]
    quarantine = implementation.VstarSourceQualityQuarantine(
        quarantine_id="fixture-llava-row-0-v1",
        dataset_id=implementation.VSTAR_DATASET_ID,
        revision=implementation.VSTAR_REVISION,
        source_file="llava_focus_data.json",
        source_file_sha256=source_hashes["llava_focus_data.json"],
        source_row_index=0,
        source_row_sha256=hashlib.sha256(
            implementation._canonical_json_bytes(quarantined_row)
        ).hexdigest(),
        sample_id=candidate.sample_id,
        observed_candidate_sha256=candidate.identity_sha256,
        reason="source_ground_truth_truncated",
        question=candidate.question,
        ground_truth=str(candidate.ground_truth),
    )

    monkeypatch.setattr(
        implementation, "VSTAR_SOURCE_QUALITY_QUARANTINES", (quarantine,)
    )
    result = implementation.materialize_vstar_candidates(
        annotations, images, tmp_path / "quarantined"
    )
    retained = _records(result.output_root / "candidates.jsonl")
    rejected = _records(result.output_root / "rejected.jsonl")
    manifest = json.loads((result.output_root / "manifest.json").read_text())

    assert result.source_rows == 4
    assert result.candidate_rows == 3
    assert result.rejected_rows == 1
    assert all(record["ground_truth"] == "There are" for record in retained)
    assert candidate.sample_id not in {record["sample_id"] for record in retained}
    assert rejected == [
        {
            "schema_version": implementation.POLICY_SELECTION_REJECTION_SCHEMA,
            "source": "vstar",
            "source_file": "llava_focus_data.json",
            "source_row_index": 0,
            "reason": "SourceQualityQuarantine: source_ground_truth_truncated",
            "metadata": {
                "quarantine_id": "fixture-llava-row-0-v1",
                "sample_id": candidate.sample_id,
                "candidate_sha256": candidate.identity_sha256,
                "source_row_sha256": quarantine.source_row_sha256,
            },
        }
    ]
    assert manifest["source_identity"]["source_quality_quarantine"] == {
        "version": implementation.VSTAR_SOURCE_QUALITY_QUARANTINE_VERSION,
        "entries": [quarantine.as_record()],
    }
    assert manifest["statistics"]["source_quality_quarantines"] == 1

    monkeypatch.setattr(
        implementation,
        "VSTAR_SOURCE_QUALITY_QUARANTINES",
        (replace(quarantine, source_row_sha256="0" * 64),),
    )
    with pytest.raises(RuntimeError, match="row SHA-256 mismatch"):
        implementation.materialize_vstar_candidates(
            annotations, images, tmp_path / "mismatch"
        )
    assert not (tmp_path / "mismatch").exists()


def test_arxivqa_source_adapter_keeps_choices_and_verifier_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_root = tmp_path / "source"
    _write_image(image_root / "images" / "figure.jpg")
    annotations = tmp_path / "arxivqa.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "id": "paper-1",
                "image": "images/figure.jpg",
                "question": "Which line is highest?",
                "options": ["A. red", "B. blue"],
                "label": "B",
                "rationale": "The blue line is highest.",
            }
        )
        + "\n"
    )
    archive = tmp_path / "images.tgz"
    archive.write_bytes(b"fixture archive identity")
    monkeypatch.setattr(
        implementation,
        "ARXIVQA_ANNOTATIONS_SHA256",
        hashlib.sha256(annotations.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        implementation,
        "ARXIVQA_IMAGES_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )

    result = implementation.materialize_arxivqa_candidates(
        annotations, archive, image_root, tmp_path / "output"
    )
    record = _records(result.output_root / "candidates.jsonl")[0]

    assert result.candidate_rows == 1
    assert record["ground_truth"] == "B"
    assert record["selection_metadata"] == {
        "label_clean_index": 1,
        "label_source_index": 1,
        "option_count": 2,
        "option_transform_version": "arxivqa-canonical-options-v2",
        "options": ["A. red", "B. blue"],
        "rationale": "The blue line is highest.",
        "raw_label": "B",
        "raw_options": ["A. red", "B. blue"],
        "removed_options": [],
        "source_option_indices": [0, 1],
    }
    assert record["question"].endswith("Choices:\nA. red\nB. blue")
    manifest = json.loads((result.output_root / "manifest.json").read_text())
    assert manifest["source_identity"]["question_render"] == (
        "question-plus-canonical-choices-v2"
    )
    assert manifest["source_identity"]["option_transform_version"] == (
        "arxivqa-canonical-options-v2"
    )


def test_arxivqa_options_remove_explicit_garbage_and_remap_source_label() -> None:
    transformed = implementation._canonicalize_arxivqa_options(
        [
            "A. first",
            "-",
            "C) correct",
            "## Figure (d)",
            "E: final",
        ],
        "C) correct",
    )

    assert transformed == {
        "options": ["A. first", "B. correct", "C. final"],
        "raw_options": [
            "A. first",
            "-",
            "C) correct",
            "## Figure (d)",
            "E: final",
        ],
        "source_option_indices": [0, 2, 4],
        "removed_options": [
            {"source_index": 1, "raw_option": "-", "reason": "separator"},
            {
                "source_index": 3,
                "raw_option": "## Figure (d)",
                "reason": "markdown_figure_heading",
            },
        ],
        "option_count": 3,
        "option_transform_version": "arxivqa-canonical-options-v2",
        "label": "B",
        "label_source_index": 2,
        "label_clean_index": 1,
    }


@pytest.mark.parametrize(
    "heading",
    [
        "# Figure (b)",
        "## For both Figures (a) and (b):",
        "###### Next figure: right panel",
    ],
)
def test_arxivqa_options_remove_only_explicit_markdown_figure_headings(
    heading: str,
) -> None:
    transformed = implementation._canonicalize_arxivqa_options(
        ["A. first", "B. second", heading, "# Question 2", "plain Figure B"],
        "B",
    )

    assert transformed["options"] == [
        "A. first",
        "B. second",
        "C. # Question 2",
        "D. plain Figure B",
    ]
    assert transformed["source_option_indices"] == [0, 1, 3, 4]
    assert transformed["removed_options"] == [
        {
            "source_index": 2,
            "raw_option": heading,
            "reason": "markdown_figure_heading",
        }
    ]


def test_arxivqa_options_do_not_strip_scientific_name_prefixes() -> None:
    transformed = implementation._canonicalize_arxivqa_options(
        ["B. pumilus", "B. sphaericus", "B. cereus"],
        "B. B. sphaericus",
    )

    assert transformed["options"] == [
        "A. B. pumilus",
        "B. B. sphaericus",
        "C. B. cereus",
    ]
    assert transformed["label"] == "B"


def test_arxivqa_options_retain_raw_whitespace_in_provenance() -> None:
    transformed = implementation._canonicalize_arxivqa_options(
        ["  A. first  ", "\tB) second\t"], " B "
    )

    assert transformed["raw_options"] == ["  A. first  ", "\tB) second\t"]
    assert transformed["options"] == ["A. first", "B. second"]


def test_arxivqa_options_support_ten_positional_choices_and_j_label() -> None:
    raw_options = [f"{chr(ord('A') + index)}) choice {index}" for index in range(10)]

    transformed = implementation._canonicalize_arxivqa_options(
        raw_options, "J) choice 9"
    )

    assert transformed["option_count"] == 10
    assert transformed["options"][0] == "A. choice 0"
    assert transformed["options"][-1] == "J. choice 9"
    assert transformed["source_option_indices"] == list(range(10))
    assert transformed["label"] == "J"
    assert transformed["label_source_index"] == 9
    assert transformed["label_clean_index"] == 9


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [("B. blue", "B"), ("[C]", "C")],
)
def test_arxivqa_label_normalizes_unambiguous_source_variants(
    raw_label: str, expected: str
) -> None:
    assert implementation._arxivqa_label(raw_label, ["A", "B", "C"]) == expected


@pytest.mark.parametrize(
    "raw_label",
    ["[Correct answer choice based on the figure]", "G", ""],
)
def test_arxivqa_label_rejects_placeholder_or_out_of_range_values(
    raw_label: str,
) -> None:
    with pytest.raises(ValueError):
        implementation._arxivqa_label(raw_label, ["A", "B", "C", "D"])


def test_arxivqa_label_rejects_an_option_removed_by_cleanup() -> None:
    with pytest.raises(
        ValueError, match="label points to an option removed by canonical cleanup"
    ):
        implementation._canonicalize_arxivqa_options(["A. first", "-", "C. third"], "B")


def test_arxivqa_options_reject_more_than_twenty_six_retained_choices() -> None:
    with pytest.raises(ValueError, match="exceed canonical A-Z label capacity"):
        implementation._canonicalize_arxivqa_options(
            [f"choice {index}" for index in range(27)], "A"
        )


def test_thinklite_source_adapter_extracts_and_deduplicates_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    table_module = pytest.importorskip("pyarrow")
    source_path = tmp_path / "thinklite.parquet"
    payload = _image_bytes()
    table = table_module.Table.from_pylist(
        [
            {
                "image": payload,
                "problem": "<image>What is one plus one?",
                "answer": "2",
                "id": index,
                "choices": None,
                "ground_truth": "2",
            }
            for index in range(2)
        ]
    )
    parquet.write_table(table, source_path)
    monkeypatch.setattr(
        implementation,
        "THINKLITE_PARQUET_SHA256",
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )

    result = implementation.materialize_thinklite_candidates(
        source_path, tmp_path / "output"
    )
    records = _records(result.output_root / "candidates.jsonl")

    assert result.source_rows == result.candidate_rows == 2
    assert result.unique_images == 1
    assert records[0]["question"] == "What is one plus one?"
    assert records[0]["image"]["path"] == records[1]["image"]["path"]
    assert len(list((result.output_root / "images").iterdir())) == 1
