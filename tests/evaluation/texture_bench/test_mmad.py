from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from tgvf_rl.evaluation.texture_bench.mmad import (
    MMAD_CANVAS_PADDING,
    MMAD_PANEL_GAP,
    MMAD_PANEL_LABEL_HEIGHT,
    MMAD_PANEL_SIZE,
    MMAD_QUERY_LABEL,
    MMAD_TEMPLATE_LABEL,
    build_mmad_task_rows,
    canonical_mmad_manifest_bytes,
    mmad_manifest_identity,
    normalize_mmad_question_type,
    normalize_mmad_source,
)
from tgvf_rl.evaluation.texture_bench.task import TextureTask


def _save_rgb(
    path: Path, *, size: tuple[int, int], color: tuple[int, int, int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="PNG")


def _snapshot_fixture(tmp_path: Path) -> tuple[Path, str]:
    snapshot = tmp_path / "snapshot"
    query = "DS-MVTec/widget/image/broken/000.png"
    random_template = "MVTec-AD/widget/train/good/000.png"
    similar_template = "MVTec-AD/widget/train/good/001.png"
    second_query = "GoodsAD/capsule/test/good/001.png"
    second_template = "GoodsAD/capsule/train/good/000.png"
    _save_rgb(snapshot / query, size=(80, 40), color=(220, 20, 20))
    _save_rgb(snapshot / random_template, size=(40, 80), color=(20, 210, 20))
    _save_rgb(snapshot / similar_template, size=(40, 80), color=(20, 20, 220))
    _save_rgb(snapshot / second_query, size=(32, 32), color=(180, 40, 40))
    _save_rgb(snapshot / second_template, size=(32, 32), color=(40, 180, 40))
    annotation = {
        query: {
            "image_path": "image/broken/000.png",
            "conversation": [
                {
                    "Question": "Which structure is visible?",
                    "Answer": "Y",
                    "Options": {"X": "first choice", "Y": "second choice"},
                    "type": "Object Structure",
                    "annotation": True,
                },
                {
                    "Question": "Which detail is visible?",
                    "Answer": "B",
                    "Options": {"A": "detail one", "B": "detail two"},
                    "type": "Object Details",
                    "annotation": False,
                },
            ],
            # This path intentionally does not exist.  The adapter must never
            # resolve or expose segmentation masks as model input.
            "mask_path": "SECRET-MASK-DO-NOT-READ.png",
            "random_templates": [random_template],
            "similar_templates": [similar_template],
        },
        second_query: {
            "conversation": [
                {
                    "Question": "Is there any defect?",
                    "Answer": "A",
                    "Options": {"A": "Yes.", "B": "No."},
                    "type": "Anomaly Detection",
                    "annotation": True,
                }
            ],
            "mask_path": "ANOTHER-SECRET-MASK.png",
            "random_templates": [second_template],
            "similar_templates": [second_template],
        },
    }
    payload = json.dumps(annotation, ensure_ascii=False).encode("utf-8")
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "mmad.json").write_bytes(payload)
    return snapshot, hashlib.sha256(payload).hexdigest()


def test_zero_shot_flattens_questions_and_preserves_official_option_order(
    tmp_path: Path,
) -> None:
    snapshot, source_sha256 = _snapshot_fixture(tmp_path)
    rows = build_mmad_task_rows(
        snapshot_root=snapshot,
        shot=0,
        stable_prefix=2,
        verify_official_source=False,
    )

    assert len(rows) == 2
    assert [row["ordinal"] for row in rows] == [0, 1]
    assert rows[0]["options"] == [
        ["A", "first choice"],
        ["B", "second choice"],
    ]
    assert rows[0]["answer"] == "B"
    prompt = str(rows[0]["question"])
    assert prompt.index("A. first choice") < prompt.index("B. second choice")
    assert "Question 1: Which structure is visible?" in prompt
    assert "Which detail is visible?" not in prompt
    assert MMAD_TEMPLATE_LABEL not in prompt
    assert len(rows[0]["image_paths"]) == 1
    assert rows[0]["image_paths"] == [
        str((snapshot / "DS-MVTec/widget/image/broken/000.png").resolve())
    ]
    assert "SECRET-MASK-DO-NOT-READ" not in json.dumps(rows)

    metadata = rows[0]["metadata"]
    assert metadata["source_dataset_raw"] == "DS-MVTec"
    assert metadata["source_dataset"] == "MVTec-AD"
    assert metadata["score_dataset"] == "MVTec-AD"
    assert metadata["question_type_raw"] == "Object Structure"
    assert metadata["question_type"] == "Object Analysis"
    assert metadata["question_type_score"] == "Object Analysis"
    assert metadata["is_normal"] == "false"
    assert metadata["category"] == "Object Analysis"
    assert metadata["cycle_category"] == "MVTec-AD"
    assert metadata["effective_image_layout"] == "query_only"
    assert rows[0]["sample_id"] == rows[0]["index"]
    assert rows[0]["image_sha256s"]
    assert rows[0]["image_dimensions"]
    assert TextureTask(**rows[0]).single_image is True

    identity = mmad_manifest_identity(
        rows,
        source_json_sha256=source_sha256,
        stable_prefix=2,
    )
    assert identity["task_count"] == 2
    assert identity["single_image_count"] == 2
    assert (
        identity["manifest_sha256"]
        == hashlib.sha256(canonical_mmad_manifest_bytes(rows)).hexdigest()
    )

    all_rows = build_mmad_task_rows(
        snapshot_root=snapshot,
        shot=0,
        verify_official_source=False,
    )
    assert all_rows[-1]["metadata"]["is_normal"] == "true"


def test_one_shot_canvas_is_labelled_ordered_letterboxed_and_deterministic(
    tmp_path: Path,
) -> None:
    snapshot, _ = _snapshot_fixture(tmp_path)
    canvas_root = tmp_path / "canvases"
    build = dict(
        snapshot_root=snapshot,
        shot=1,
        canvas_root=canvas_root,
        stable_prefix=1,
        verify_official_source=False,
    )
    first = build_mmad_task_rows(**build)
    second = build_mmad_task_rows(**build)

    assert first == second
    row = first[0]
    assert row["metadata"]["template_kind"] == "random"
    assert row["metadata"]["effective_image_layout"] == (
        "normal_template_left__query_right"
    )
    assert MMAD_TEMPLATE_LABEL in row["question"]
    assert MMAD_QUERY_LABEL in row["question"]
    assert row["question"].index(MMAD_TEMPLATE_LABEL) < row["question"].index(
        MMAD_QUERY_LABEL
    )
    assert "SECRET-MASK-DO-NOT-READ" not in json.dumps(row)

    canvas_path = Path(row["image_paths"][0])
    assert canvas_path.is_absolute()
    assert canvas_path.suffix == ".png"
    assert row["image_sha256s"] == [
        hashlib.sha256(canvas_path.read_bytes()).hexdigest()
    ]
    panel_width, panel_height = MMAD_PANEL_SIZE
    expected_size = (
        2 * MMAD_CANVAS_PADDING + 2 * panel_width + MMAD_PANEL_GAP,
        2 * MMAD_CANVAS_PADDING + MMAD_PANEL_LABEL_HEIGHT + panel_height,
    )
    assert row["image_dimensions"] == [list(expected_size)]

    with Image.open(canvas_path) as canvas:
        canvas = canvas.convert("RGB")
        image_top = MMAD_CANVAS_PADDING + MMAD_PANEL_LABEL_HEIGHT
        left_center = (
            MMAD_CANVAS_PADDING + panel_width // 2,
            image_top + panel_height // 2,
        )
        right_left = MMAD_CANVAS_PADDING + panel_width + MMAD_PANEL_GAP
        right_center = (
            right_left + panel_width // 2,
            image_top + panel_height // 2,
        )
        assert canvas.getpixel(left_center) == (20, 210, 20)
        assert canvas.getpixel(right_center) == (220, 20, 20)
        # The portrait template and landscape query are fitted, not stretched.
        assert canvas.getpixel((MMAD_CANVAS_PADDING, left_center[1])) == (18, 18, 18)
        assert canvas.getpixel((right_center[0], image_top)) == (18, 18, 18)
        label_band = canvas.crop(
            (
                0,
                MMAD_CANVAS_PADDING,
                canvas.width,
                image_top,
            )
        )
        assert any(sum(pixel) < 100 for pixel in label_band.getdata())


def test_one_shot_similar_template_and_official_report_merges(tmp_path: Path) -> None:
    snapshot, _ = _snapshot_fixture(tmp_path)
    rows = build_mmad_task_rows(
        snapshot_root=snapshot,
        shot=1,
        canvas_root=tmp_path / "similar-canvases",
        template_kind="similar",
        stable_prefix=1,
        verify_official_source=False,
    )

    row = rows[0]
    assert row["metadata"]["template_kind"] == "similar"
    assert row["metadata"]["template_image"].endswith("good/001.png")
    with Image.open(row["image_paths"][0]) as canvas:
        center = (
            MMAD_CANVAS_PADDING + MMAD_PANEL_SIZE[0] // 2,
            MMAD_CANVAS_PADDING + MMAD_PANEL_LABEL_HEIGHT + MMAD_PANEL_SIZE[1] // 2,
        )
        assert canvas.convert("RGB").getpixel(center) == (20, 20, 220)

    assert normalize_mmad_source("DS-MVTec") == "MVTec-AD"
    assert normalize_mmad_source("MVTec-AD") == "MVTec-AD"
    assert normalize_mmad_source("VisA") == "VisA"
    assert normalize_mmad_question_type("Object Structure") == "Object Analysis"
    assert normalize_mmad_question_type("Object Details") == "Object Analysis"
    assert normalize_mmad_question_type("Defect Analysis") == "Defect Analysis"
