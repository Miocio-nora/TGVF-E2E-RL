"""Pure utilities for the bounded VStar oracle-crop diagnostic.

This module is deliberately separate from the production policy evaluator.  It
loads a fixed, small VStarBench slice whose source annotations contain object
boxes, constructs a positive-control crop and a same-sized no-information
placebo, and performs deterministic multiple-choice scoring.  It does not
launch a model, alter the Crop/TGVF protocols, or write experiment artifacts.

VStar sidecar boxes use source-image pixel ``[x, y, width, height]`` coordinates.
The probe converts them to an expanded square ``[left, top, right, bottom]``
region while retaining the exact requested side length at image boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

from PIL import Image


VSTAR_ORACLE_CROP_PROBE_SCHEMA = "tgvf.vstar-oracle-crop-probe.v1"
VSTAR_EXPECTED_ROW_COUNT = 191
VSTAR_EXPECTED_CATEGORY_COUNTS = {
    "direct_attributes": 115,
    "relative_position": 76,
}
TINY_MAX_AREA_RATIO = 0.0002

# These are exactly the single-object direct-attribute rows at or below
# TINY_MAX_AREA_RATIO in the pinned 191-row VStarBench snapshot.
TINY_SAMPLE_ORDINALS = (
    7,
    10,
    20,
    29,
    34,
    35,
    41,
    44,
    46,
    49,
    51,
    70,
    71,
    76,
    78,
    82,
    83,
    86,
    87,
    93,
    95,
    96,
    99,
    101,
    105,
    106,
    110,
)

# Five progressively larger single-object controls.  Their source-object area
# ratios are approximately 0.055%, 0.091%, 0.155%, 0.270%, and 0.437%.
MEDIUM_CONTROL_ORDINALS = (81, 59, 91, 40, 73)
PROBE_SAMPLE_ORDINALS = TINY_SAMPLE_ORDINALS + MEDIUM_CONTROL_ORDINALS

# Compatibility names consumed by the isolated runner.  They are aliases of
# the pinned membership, not separately configurable sample lists.
DEFAULT_TINY_PRIMARY_IDS = TINY_SAMPLE_ORDINALS
DEFAULT_MEDIUM_CONTROL_IDS = MEDIUM_CONTROL_ORDINALS

_OPTION_LINE = re.compile(r"^\(([A-Z])\)\s*(.*?)\s*$")
_PLAIN_LABEL = re.compile(r"^\(?\s*([A-Z])\s*\)?[.:]?$")
_LABEL_WITH_TEXT = re.compile(r"^\(?\s*([A-Z])\s*\)?\s*[.:-]\s*(.+?)\s*$")
_ANSWER_LABEL = re.compile(
    r"^(?:(?:thus|therefore)[,:]?\s+)?(?:the\s+)?(?:correct\s+)?"
    r"(?:answer|option|choice)(?:\s+is|\s*:)?\s*\(?([A-Z])\)?[.!]?$",
    re.IGNORECASE,
)
_MATCHING_LABEL = re.compile(
    r"^.*\b(?:matching|corresponding\s+to)\s+(?:option|choice)\s+"
    r"\(?([A-Z])\)?[.!]?$",
    re.IGNORECASE,
)
_TERMINAL_MODEL_TOKEN = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$")


ProbeStratum = Literal["tiny", "medium_control"]
XYWH = tuple[int, int, int, int]
XYXY = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class VStarOracleCropSample:
    """One fully validated member of the fixed internal probe."""

    probe_index: int
    ordinal: int
    sample_id: str
    stratum: ProbeStratum
    question_id: str
    category: str
    question: str
    options: tuple[tuple[str, str], ...]
    answer: str
    image_path: Path
    image_size: tuple[int, int]
    target_object: str
    bbox_xywh: XYWH
    bbox_area_ratio: float

    @property
    def option_map(self) -> dict[str, str]:
        return dict(self.options)

    @property
    def row_id(self) -> int:
        return self.ordinal

    @property
    def gt_xywh(self) -> XYWH:
        return self.bbox_xywh

    @property
    def bbox_area_fraction(self) -> float:
        return self.bbox_area_ratio

    def as_manifest_record(self) -> dict[str, object]:
        """Return the JSON-safe case identity consumed by the probe runner."""

        return {
            "probe_index": self.probe_index,
            "row_id": self.row_id,
            "sample_id": self.sample_id,
            "stratum": self.stratum,
            "question_id": self.question_id,
            "category": self.category,
            "question": self.question,
            "options": [list(option) for option in self.options],
            "answer": self.answer,
            "image_path": str(self.image_path),
            "image_size": list(self.image_size),
            "target_object": self.target_object,
            "gt_xywh": list(self.gt_xywh),
            "bbox_area_fraction": self.bbox_area_fraction,
        }


@dataclass(frozen=True, slots=True)
class OracleCropPair:
    """A source-derived positive-control crop and its visual placebo."""

    source_bbox_xyxy: XYXY
    oracle: Image.Image
    placebo: Image.Image


@dataclass(frozen=True, slots=True)
class _ValidatedRow:
    ordinal: int
    question_id: str
    category: str
    question: str
    options: tuple[tuple[str, str], ...]
    answer: str
    image_path: Path
    image_size: tuple[int, int]
    target_objects: tuple[str, ...]
    boxes_xywh: tuple[XYWH, ...]
    bbox_area_ratio: float


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _parse_question_and_options(
    text: object, *, ordinal: int
) -> tuple[str, tuple[tuple[str, str], ...]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"VStar row {ordinal} has invalid text")
    question_lines: list[str] = []
    options: list[tuple[str, str]] = []
    saw_options = False
    for line in text.splitlines():
        match = _OPTION_LINE.fullmatch(line.strip())
        if match is not None:
            saw_options = True
            label, value = match.groups()
            if not value:
                raise ValueError(f"VStar row {ordinal} has an empty option {label}")
            options.append((label, value))
        elif not saw_options:
            if line.strip():
                question_lines.append(line.strip())
    question = "\n".join(question_lines)
    if not question:
        raise ValueError(f"VStar row {ordinal} has no question")
    expected_labels = [chr(ord("A") + index) for index in range(len(options))]
    actual_labels = [label for label, _ in options]
    if len(options) < 2 or actual_labels != expected_labels:
        raise ValueError(
            f"VStar row {ordinal} options are not contiguous from A: {actual_labels}"
        )
    return question, tuple(options)


def _strict_xywh(value: object, *, ordinal: int, box_index: int) -> XYWH:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 4
        or any(type(coordinate) is not int for coordinate in value)
    ):
        raise ValueError(
            f"VStar row {ordinal} bbox {box_index} must be four integer xywh values"
        )
    x, y, width, height = value
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"VStar row {ordinal} bbox {box_index} is not positive xywh")
    return x, y, width, height


def _safe_image_path(dataset_root: Path, value: object, *, ordinal: int) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"VStar row {ordinal} has no image path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"VStar row {ordinal} image path must be relative")
    root = dataset_root.resolve()
    image_path = (root / relative).resolve()
    if not image_path.is_relative_to(root):
        raise ValueError(f"VStar row {ordinal} image escapes the dataset root")
    if not image_path.is_file():
        raise FileNotFoundError(
            f"VStar row {ordinal} image does not exist: {image_path}"
        )
    return image_path


def _validate_row(
    raw: Mapping[str, object], *, ordinal: int, dataset_root: Path
) -> _ValidatedRow:
    question_id = raw.get("question_id")
    if not isinstance(question_id, str) or question_id != str(ordinal):
        raise ValueError(
            f"VStar row {ordinal} question_id must equal its zero-based ordinal"
        )
    category = raw.get("category")
    if category not in VSTAR_EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"VStar row {ordinal} has unknown category {category!r}")
    question, options = _parse_question_and_options(raw.get("text"), ordinal=ordinal)
    answer = raw.get("label")
    if not isinstance(answer, str) or answer not in dict(options):
        raise ValueError(f"VStar row {ordinal} answer is not a valid option label")
    image_path = _safe_image_path(dataset_root, raw.get("image"), ordinal=ordinal)
    sidecar_path = image_path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"VStar row {ordinal} sidecar does not exist: {sidecar_path}"
        )
    sidecar = _json_object(sidecar_path)
    if sidecar.get("question") != question:
        raise ValueError(f"VStar row {ordinal} sidecar question differs from test row")
    target_value = sidecar.get("target_object")
    if (
        not isinstance(target_value, list)
        or not target_value
        or any(not isinstance(target, str) or not target for target in target_value)
    ):
        raise ValueError(f"VStar row {ordinal} has invalid target_object annotations")
    bbox_value = sidecar.get("bbox")
    if not isinstance(bbox_value, list) or not bbox_value:
        raise ValueError(f"VStar row {ordinal} has no bbox annotations")
    boxes = tuple(
        _strict_xywh(box, ordinal=ordinal, box_index=box_index)
        for box_index, box in enumerate(bbox_value)
    )
    if len(target_value) != len(boxes):
        raise ValueError(
            f"VStar row {ordinal} target_object and bbox counts do not match"
        )
    try:
        with Image.open(image_path) as image:
            image_size = image.size
    except OSError as error:
        raise ValueError(
            f"VStar row {ordinal} image is unreadable: {image_path}"
        ) from error
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"VStar row {ordinal} image has invalid dimensions")
    for box_index, (x, y, width, height) in enumerate(boxes):
        if x + width > image_width or y + height > image_height:
            raise ValueError(
                f"VStar row {ordinal} bbox {box_index} exceeds the source image"
            )
    area_ratio = sum(width * height for _, _, width, height in boxes) / (
        image_width * image_height
    )
    return _ValidatedRow(
        ordinal=ordinal,
        question_id=question_id,
        category=category,
        question=question,
        options=options,
        answer=answer,
        image_path=image_path,
        image_size=image_size,
        target_objects=tuple(target_value),
        boxes_xywh=boxes,
        bbox_area_ratio=area_ratio,
    )


def load_vstar_oracle_crop_samples(
    test_questions_path: str | Path,
) -> tuple[VStarOracleCropSample, ...]:
    """Load and identity-check the fixed 27-tiny plus 5-control probe.

    The loader scans the complete pinned 191-row population.  This makes a
    silent dataset revision fail explicitly instead of changing the meaning of
    the tiny-object threshold or the fixed probe membership.
    """

    questions_path = Path(test_questions_path).resolve()
    if not questions_path.is_file():
        raise FileNotFoundError(f"VStar test questions do not exist: {questions_path}")
    raw_rows: list[dict[str, Any]] = []
    with questions_path.open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if not line.strip():
                raise ValueError(f"VStar test questions contain blank row {ordinal}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid VStar JSONL row {ordinal}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"VStar row {ordinal} is not a JSON object")
            raw_rows.append(raw)
    if len(raw_rows) != VSTAR_EXPECTED_ROW_COUNT:
        raise ValueError(
            "VStar oracle-crop probe requires exactly "
            f"{VSTAR_EXPECTED_ROW_COUNT} rows, found {len(raw_rows)}"
        )
    rows = tuple(
        _validate_row(raw, ordinal=ordinal, dataset_root=questions_path.parent)
        for ordinal, raw in enumerate(raw_rows)
    )
    category_counts = {
        category: sum(row.category == category for row in rows)
        for category in VSTAR_EXPECTED_CATEGORY_COUNTS
    }
    if category_counts != VSTAR_EXPECTED_CATEGORY_COUNTS:
        raise ValueError(
            "VStar category counts differ from the pinned population: "
            f"{category_counts}"
        )
    discovered_tiny = tuple(
        row.ordinal
        for row in rows
        if row.category == "direct_attributes"
        and len(row.target_objects) == 1
        and len(row.boxes_xywh) == 1
        and row.bbox_area_ratio <= TINY_MAX_AREA_RATIO
    )
    if discovered_tiny != TINY_SAMPLE_ORDINALS:
        raise ValueError(
            "VStar tiny-object membership differs from the fixed probe: "
            f"expected {TINY_SAMPLE_ORDINALS}, found {discovered_tiny}"
        )
    selected: list[VStarOracleCropSample] = []
    for probe_index, ordinal in enumerate(PROBE_SAMPLE_ORDINALS):
        row = rows[ordinal]
        if (
            row.category != "direct_attributes"
            or len(row.target_objects) != 1
            or len(row.boxes_xywh) != 1
        ):
            raise ValueError(
                f"fixed VStar probe row {ordinal} is not single-object direct_attributes"
            )
        stratum: ProbeStratum = (
            "tiny" if ordinal in TINY_SAMPLE_ORDINALS else "medium_control"
        )
        if stratum == "medium_control" and not (
            row.bbox_area_ratio > TINY_MAX_AREA_RATIO
        ):
            raise ValueError(f"fixed medium-control row {ordinal} is no longer medium")
        selected.append(
            VStarOracleCropSample(
                probe_index=probe_index,
                ordinal=row.ordinal,
                sample_id=(
                    "vstar_test_questions_191/"
                    "vstar_bench_snapshot_test_questions_jsonl/"
                    f"{row.ordinal}_{row.ordinal:06d}"
                ),
                stratum=stratum,
                question_id=row.question_id,
                category=row.category,
                question=row.question,
                options=row.options,
                answer=row.answer,
                image_path=row.image_path,
                image_size=row.image_size,
                target_object=row.target_objects[0],
                bbox_xywh=row.boxes_xywh[0],
                bbox_area_ratio=row.bbox_area_ratio,
            )
        )
    return tuple(selected)


def build_vstar_oracle_probe_cases(
    vstar_root: str | Path,
    *,
    tiny_ids: Sequence[int] = DEFAULT_TINY_PRIMARY_IDS,
    medium_ids: Sequence[int] = DEFAULT_MEDIUM_CONTROL_IDS,
) -> tuple[VStarOracleCropSample, ...]:
    """Build cases from a dataset root while refusing membership drift."""

    if tuple(tiny_ids) != DEFAULT_TINY_PRIMARY_IDS:
        raise ValueError("tiny_ids must equal the fixed oracle-probe membership")
    if tuple(medium_ids) != DEFAULT_MEDIUM_CONTROL_IDS:
        raise ValueError("medium_ids must equal the fixed oracle-probe membership")
    root = Path(vstar_root)
    questions_path = (
        root if root.name == "test_questions.jsonl" else root / "test_questions.jsonl"
    )
    return load_vstar_oracle_crop_samples(questions_path)


def expand_gt_xywh_to_square(
    bbox_xywh: Sequence[int],
    *,
    image_size: tuple[int, int],
    expansion: int = 4,
    minimum_side: int = 32,
) -> XYXY:
    """Expand source-pixel xywh to an exact square, shifting at boundaries."""

    if (
        not isinstance(bbox_xywh, Sequence)
        or isinstance(bbox_xywh, (str, bytes, bytearray))
        or len(bbox_xywh) != 4
        or any(type(coordinate) is not int for coordinate in bbox_xywh)
    ):
        raise ValueError("bbox_xywh must contain four integers")
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(type(dimension) is not int for dimension in image_size)
    ):
        raise ValueError("image_size must be an integer (width, height) tuple")
    if type(expansion) is not int or expansion <= 0:
        raise ValueError("expansion must be a positive integer")
    if type(minimum_side) is not int or minimum_side <= 0:
        raise ValueError("minimum_side must be a positive integer")
    x, y, width, height = bbox_xywh
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("bbox_xywh must be positive and non-empty")
    if x + width > image_width or y + height > image_height:
        raise ValueError("bbox_xywh exceeds the source image")
    side = max(minimum_side, expansion * max(width, height))
    if side > image_width or side > image_height:
        raise ValueError(
            f"requested square side {side} does not fit image {image_size}"
        )
    left = math.floor(x + width / 2 - side / 2)
    top = math.floor(y + height / 2 - side / 2)
    left = min(max(left, 0), image_width - side)
    top = min(max(top, 0), image_height - side)
    return left, top, left + side, top + side


def make_oracle_crop_pair(
    image: Image.Image,
    bbox_xywh: Sequence[int],
    *,
    expansion: int = 4,
    minimum_side: int = 32,
    placebo_gray: int = 128,
) -> OracleCropPair:
    """Create an RGB oracle crop and an exactly same-sized gray placebo."""

    if type(placebo_gray) is not int or not 0 <= placebo_gray <= 255:
        raise ValueError("placebo_gray must be an integer in [0, 255]")
    oracle, source_bbox = make_oracle_crop(
        image,
        bbox_xywh,
        expansion=expansion,
        minimum_side=minimum_side,
    )
    placebo = make_gray_placebo(oracle.size, gray=placebo_gray)
    return OracleCropPair(
        source_bbox_xyxy=source_bbox,
        oracle=oracle,
        placebo=placebo,
    )


def make_oracle_crop(
    image: Image.Image,
    bbox_xywh: Sequence[int],
    *,
    expansion: int = 4,
    minimum_side: int = 32,
) -> tuple[Image.Image, XYXY]:
    """Create the RGB positive-control crop and return its source-pixel box."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    source_bbox = expand_gt_xywh_to_square(
        bbox_xywh,
        image_size=image.size,
        expansion=expansion,
        minimum_side=minimum_side,
    )
    return image.convert("RGB").crop(source_bbox), source_bbox


def make_gray_placebo(size: tuple[int, int], *, gray: int = 128) -> Image.Image:
    """Create an RGB, spatially uniform no-information observation."""

    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in size)
    ):
        raise ValueError("size must be a positive integer (width, height) tuple")
    if type(gray) is not int or not 0 <= gray <= 255:
        raise ValueError("gray must be an integer in [0, 255]")
    return Image.new("RGB", size, (gray, gray, gray))


def _normalized_option_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def extract_exact_option_label(
    prediction: object,
    options: Mapping[str, str] | Sequence[tuple[str, str]],
) -> str | None:
    """Extract an unambiguous final option without semantic judging.

    Accepted forms are a bare final label, a conventional exact answer phrase,
    a label followed by its exact option text, or the exact option text alone.
    Only the final non-empty line is inspected, preventing reasoning mentions
    of distractors from becoming votes.
    """

    if not isinstance(prediction, str) or not prediction.strip():
        return None
    option_map = dict(options)
    if len(option_map) < 2 or any(
        not isinstance(label, str)
        or len(label) != 1
        or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        or not isinstance(value, str)
        or not value.strip()
        for label, value in option_map.items()
    ):
        raise ValueError("options must contain at least two uppercase labels and text")
    cleaned = prediction.strip()
    while True:
        without_token = _TERMINAL_MODEL_TOKEN.sub("", cleaned).strip()
        if without_token == cleaned:
            break
        cleaned = without_token
    if not cleaned:
        return None
    final_line = next(
        (line.strip() for line in reversed(cleaned.splitlines()) if line.strip()), ""
    )
    for pattern in (_PLAIN_LABEL, _ANSWER_LABEL, _MATCHING_LABEL):
        match = pattern.fullmatch(final_line)
        if match is not None:
            label = match.group(1).upper()
            return label if label in option_map else None
    labeled = _LABEL_WITH_TEXT.fullmatch(final_line)
    if labeled is not None:
        label, value = labeled.groups()
        label = label.upper()
        if label in option_map and _normalized_option_text(
            value.rstrip(".! ")
        ) == _normalized_option_text(option_map[label].rstrip(".! ")):
            return label
        return None
    normalized_final = _normalized_option_text(final_line.rstrip(".! "))
    text_matches = [
        label
        for label, value in option_map.items()
        if normalized_final == _normalized_option_text(value.rstrip(".! "))
    ]
    return text_matches[0] if len(text_matches) == 1 else None


def summarize_exact_option_predictions(
    samples: Sequence[VStarOracleCropSample],
    predictions_by_condition: Mapping[str, Mapping[int, str]],
) -> dict[str, object]:
    """Return deterministic arm, stratum, row, and paired-transition metrics."""

    if not samples:
        raise ValueError("samples must not be empty")
    ordinals = [sample.ordinal for sample in samples]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("samples contain duplicate ordinals")
    if not predictions_by_condition:
        raise ValueError("predictions_by_condition must not be empty")
    expected = set(ordinals)
    condition_names = list(predictions_by_condition)
    if any(not isinstance(name, str) or not name for name in condition_names):
        raise ValueError("condition names must be non-empty strings")
    extracted: dict[str, dict[int, str | None]] = {}
    arm_summaries: dict[str, object] = {}
    for condition, predictions in predictions_by_condition.items():
        actual = set(predictions)
        if actual != expected:
            raise ValueError(
                f"condition {condition!r} prediction coverage differs: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        labels = {
            sample.ordinal: extract_exact_option_label(
                predictions[sample.ordinal], sample.options
            )
            for sample in samples
        }
        extracted[condition] = labels
        correct = {
            sample.ordinal: labels[sample.ordinal] == sample.answer
            for sample in samples
        }
        by_stratum: dict[str, object] = {}
        for stratum in ("tiny", "medium_control"):
            members = [sample for sample in samples if sample.stratum == stratum]
            parsed_count = sum(labels[sample.ordinal] is not None for sample in members)
            correct_count = sum(correct[sample.ordinal] for sample in members)
            by_stratum[stratum] = {
                "sample_count": len(members),
                "parsed_count": parsed_count,
                "parse_rate": parsed_count / len(members) if members else 0.0,
                "correct_count": correct_count,
                "accuracy": correct_count / len(members) if members else 0.0,
            }
        parsed_count = sum(label is not None for label in labels.values())
        correct_count = sum(correct.values())
        arm_summaries[condition] = {
            "sample_count": len(samples),
            "parsed_count": parsed_count,
            "parse_rate": parsed_count / len(samples),
            "correct_count": correct_count,
            "accuracy": correct_count / len(samples),
            "by_stratum": by_stratum,
        }
    pairwise: dict[str, object] = {}
    for left_index, left in enumerate(condition_names):
        for right in condition_names[left_index + 1 :]:
            left_correct = {
                sample.ordinal: extracted[left][sample.ordinal] == sample.answer
                for sample in samples
            }
            right_correct = {
                sample.ordinal: extracted[right][sample.ordinal] == sample.answer
                for sample in samples
            }
            left_only = sum(
                left_correct[ordinal] and not right_correct[ordinal]
                for ordinal in ordinals
            )
            right_only = sum(
                right_correct[ordinal] and not left_correct[ordinal]
                for ordinal in ordinals
            )
            both_correct = sum(
                left_correct[ordinal] and right_correct[ordinal] for ordinal in ordinals
            )
            both_wrong = len(samples) - left_only - right_only - both_correct
            pairwise[f"{left}__to__{right}"] = {
                "left": left,
                "right": right,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "left_only_correct": left_only,
                "right_only_correct": right_only,
                "right_minus_left_correct": right_only - left_only,
                "right_minus_left_accuracy": (right_only - left_only) / len(samples),
            }
    rows = [
        {
            "probe_index": sample.probe_index,
            "ordinal": sample.ordinal,
            "sample_id": sample.sample_id,
            "stratum": sample.stratum,
            "answer": sample.answer,
            "extracted": {
                condition: extracted[condition][sample.ordinal]
                for condition in condition_names
            },
            "correct": {
                condition: extracted[condition][sample.ordinal] == sample.answer
                for condition in condition_names
            },
        }
        for sample in samples
    ]
    return {
        "schema_version": VSTAR_ORACLE_CROP_PROBE_SCHEMA,
        "sample_count": len(samples),
        "condition_order": condition_names,
        "conditions": arm_summaries,
        "pairwise": pairwise,
        "rows": rows,
    }


__all__ = [
    "DEFAULT_MEDIUM_CONTROL_IDS",
    "DEFAULT_TINY_PRIMARY_IDS",
    "MEDIUM_CONTROL_ORDINALS",
    "PROBE_SAMPLE_ORDINALS",
    "TINY_MAX_AREA_RATIO",
    "TINY_SAMPLE_ORDINALS",
    "VSTAR_ORACLE_CROP_PROBE_SCHEMA",
    "OracleCropPair",
    "VStarOracleCropSample",
    "build_vstar_oracle_probe_cases",
    "expand_gt_xywh_to_square",
    "extract_exact_option_label",
    "load_vstar_oracle_crop_samples",
    "make_gray_placebo",
    "make_oracle_crop",
    "make_oracle_crop_pair",
    "summarize_exact_option_predictions",
]
