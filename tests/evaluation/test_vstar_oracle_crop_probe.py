from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tgvf_rl.evaluation.vstar_oracle_crop_probe import (
    MEDIUM_CONTROL_ORDINALS,
    PROBE_SAMPLE_ORDINALS,
    TINY_SAMPLE_ORDINALS,
    VStarOracleCropSample,
    build_vstar_oracle_probe_cases,
    expand_gt_xywh_to_square,
    extract_exact_option_label,
    load_vstar_oracle_crop_samples,
    make_gray_placebo,
    make_oracle_crop,
    make_oracle_crop_pair,
    summarize_exact_option_predictions,
)


def _write_pinned_population(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "vstar"
    questions_path = dataset_root / "test_questions.jsonl"
    rows: list[dict[str, object]] = []
    tiny = set(TINY_SAMPLE_ORDINALS)
    controls = set(MEDIUM_CONTROL_ORDINALS)
    for ordinal in range(191):
        category = "direct_attributes" if ordinal < 115 else "relative_position"
        directory = dataset_root / category
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / f"sample-{ordinal}.png"
        Image.new("RGB", (100, 100), (ordinal % 256, 20, 30)).save(image_path)
        question = f"What is the value for object {ordinal}?"
        rows.append(
            {
                "image": f"{category}/{image_path.name}",
                "text": (
                    f"{question}\n"
                    "(A) alpha\n"
                    "(B) beta\n"
                    "Answer with the option's letter from the given choices directly."
                ),
                "category": category,
                "question_id": str(ordinal),
                "label": "A" if ordinal % 2 == 0 else "B",
            }
        )
        if ordinal in tiny:
            bbox = [40, 40, 1, 1]
        elif ordinal in controls:
            bbox = [40, 40, 3, 3]
        elif category == "direct_attributes":
            bbox = [35, 35, 5, 5]
        else:
            bbox = [25, 25, 10, 10]
        image_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "target_object": [f"object {ordinal}"],
                    "bbox": [bbox],
                    "question": question,
                    "options": ["alpha", "beta"],
                }
            ),
            encoding="utf-8",
        )
    questions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return questions_path


def test_loads_exact_fixed_tiny_and_medium_probe(tmp_path: Path) -> None:
    questions_path = _write_pinned_population(tmp_path)

    samples = load_vstar_oracle_crop_samples(questions_path)

    assert len(samples) == 32
    assert tuple(sample.ordinal for sample in samples) == PROBE_SAMPLE_ORDINALS
    assert [sample.probe_index for sample in samples] == list(range(32))
    assert sum(sample.stratum == "tiny" for sample in samples) == 27
    assert sum(sample.stratum == "medium_control" for sample in samples) == 5
    assert all(
        sample.bbox_area_ratio == pytest.approx(0.0001) for sample in samples[:27]
    )
    assert all(
        sample.bbox_area_ratio == pytest.approx(0.0009) for sample in samples[27:]
    )
    assert samples[0].sample_id.endswith("/7_000007")
    assert samples[0].image_path.is_absolute()
    assert samples[0].option_map == {"A": "alpha", "B": "beta"}
    assert build_vstar_oracle_probe_cases(questions_path.parent) == samples
    assert samples[0].row_id == samples[0].ordinal
    assert samples[0].gt_xywh == samples[0].bbox_xywh
    assert samples[0].as_manifest_record()["row_id"] == samples[0].ordinal


def test_probe_builder_rejects_membership_override(tmp_path: Path) -> None:
    questions_path = _write_pinned_population(tmp_path)

    with pytest.raises(ValueError, match="tiny_ids must equal"):
        build_vstar_oracle_probe_cases(questions_path.parent, tiny_ids=(7,))


def test_population_drift_that_changes_tiny_membership_is_rejected(
    tmp_path: Path,
) -> None:
    questions_path = _write_pinned_population(tmp_path)
    extra_tiny = questions_path.parent / "direct_attributes" / "sample-0.json"
    payload = json.loads(extra_tiny.read_text(encoding="utf-8"))
    payload["bbox"] = [[40, 40, 1, 1]]
    extra_tiny.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="tiny-object membership differs"):
        load_vstar_oracle_crop_samples(questions_path)


def test_population_requires_sidecar_question_identity(tmp_path: Path) -> None:
    questions_path = _write_pinned_population(tmp_path)
    sidecar = questions_path.parent / "direct_attributes" / "sample-7.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["question"] = "A different question"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar question differs"):
        load_vstar_oracle_crop_samples(questions_path)


def test_expand_gt_box_preserves_requested_square_at_edges() -> None:
    assert expand_gt_xywh_to_square((40, 40, 10, 5), image_size=(100, 100)) == (
        25,
        22,
        65,
        62,
    )
    assert expand_gt_xywh_to_square((1, 1, 8, 8), image_size=(100, 100)) == (
        0,
        0,
        32,
        32,
    )
    assert expand_gt_xywh_to_square((91, 91, 8, 8), image_size=(100, 100)) == (
        68,
        68,
        100,
        100,
    )


def test_expand_gt_box_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        expand_gt_xywh_to_square((95, 95, 10, 10), image_size=(100, 100))
    with pytest.raises(ValueError, match="does not fit"):
        expand_gt_xywh_to_square((1, 1, 10, 10), image_size=(30, 100))
    with pytest.raises(ValueError, match="four integers"):
        expand_gt_xywh_to_square((1, 2, 3.0, 4), image_size=(100, 100))  # type: ignore[arg-type]


def test_oracle_and_placebo_have_identical_geometry_but_distinct_information() -> None:
    image = Image.new("RGB", (100, 100), (1, 2, 3))
    for x in range(40, 50):
        for y in range(40, 50):
            image.putpixel((x, y), (200, 10, 20))

    pair = make_oracle_crop_pair(image, (40, 40, 10, 10), placebo_gray=127)

    assert pair.source_bbox_xyxy == (25, 25, 65, 65)
    assert pair.oracle.mode == pair.placebo.mode == "RGB"
    assert pair.oracle.size == pair.placebo.size == (40, 40)
    assert pair.oracle.getpixel((20, 20)) == (200, 10, 20)
    assert pair.placebo.getextrema() == ((127, 127), (127, 127), (127, 127))
    assert image.size == (100, 100)

    oracle, source_box = make_oracle_crop(image, (40, 40, 10, 10))
    placebo = make_gray_placebo(oracle.size, gray=127)
    assert source_box == pair.source_bbox_xyxy
    assert oracle.tobytes() == pair.oracle.tobytes()
    assert placebo.tobytes() == pair.placebo.tobytes()


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("B", "B"),
        ("(B).<|im_end|>", "B"),
        ("Reasoning mentions A.\nThe correct option is B.", "B"),
        ("The object is blue, matching option B.", "B"),
        ("B. blue", "B"),
        ("blue", "B"),
        ("A or B", None),
        ("The answer is C.", None),
        ("Maybe red, maybe blue", None),
        ("", None),
    ],
)
def test_extract_exact_option_label(prediction: str, expected: str | None) -> None:
    assert extract_exact_option_label(prediction, {"A": "red", "B": "blue"}) == expected


def _sample(
    *, probe_index: int, ordinal: int, stratum: str, answer: str
) -> VStarOracleCropSample:
    return VStarOracleCropSample(
        probe_index=probe_index,
        ordinal=ordinal,
        sample_id=f"sample-{ordinal}",
        stratum=stratum,  # type: ignore[arg-type]
        question_id=str(ordinal),
        category="direct_attributes",
        question="Which color?",
        options=(("A", "red"), ("B", "blue")),
        answer=answer,
        image_path=Path(f"/{ordinal}.png"),
        image_size=(100, 100),
        target_object="object",
        bbox_xywh=(1, 1, 2, 2),
        bbox_area_ratio=0.0004,
    )


def test_summary_reports_strata_parse_and_paired_transitions() -> None:
    samples = (
        _sample(probe_index=0, ordinal=1, stratum="tiny", answer="A"),
        _sample(probe_index=1, ordinal=2, stratum="tiny", answer="B"),
        _sample(probe_index=2, ordinal=3, stratum="medium_control", answer="A"),
    )

    summary = summarize_exact_option_predictions(
        samples,
        {
            "current": {1: "B", 2: "B", 3: "unparsed"},
            "oracle": {1: "A", 2: "B", 3: "A"},
            "placebo": {1: "B", 2: "A", 3: "A"},
        },
    )

    assert summary["sample_count"] == 3
    assert summary["conditions"]["current"]["correct_count"] == 1  # type: ignore[index]
    assert summary["conditions"]["current"]["parsed_count"] == 2  # type: ignore[index]
    assert summary["conditions"]["oracle"]["accuracy"] == 1.0  # type: ignore[index]
    assert (
        summary["conditions"]["oracle"]["by_stratum"][  # type: ignore[index]
            "medium_control"
        ]["accuracy"]
        == 1.0
    )
    paired = summary["pairwise"]["current__to__oracle"]  # type: ignore[index]
    assert paired["right_only_correct"] == 2
    assert paired["left_only_correct"] == 0
    assert paired["right_minus_left_accuracy"] == pytest.approx(2 / 3)


def test_summary_rejects_partial_prediction_coverage() -> None:
    samples = (_sample(probe_index=0, ordinal=1, stratum="tiny", answer="A"),)

    with pytest.raises(ValueError, match="prediction coverage differs"):
        summarize_exact_option_predictions(samples, {"oracle": {}})
