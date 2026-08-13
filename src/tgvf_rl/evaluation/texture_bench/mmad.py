"""Deterministic MMAD task adaptation for the single-image tool protocol.

MMAD's official Qwen evaluator asks one multiple-choice question per request.
In zero-shot mode the request contains only the query image.  The official
one-shot protocol supplies a normal reference followed by the query image;
our policy evaluator accepts one source image, so this module renders those
roles into one immutable, labelled PNG without changing their aspect ratios.

Segmentation masks belong to MMAD's annotation surface, not its model input.
This adapter deliberately never resolves, opens, copies, or records a mask
path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont


MMAD_ADAPTER_SCHEMA = "tgvf-mmad-single-image-adapter-v1"
MMAD_MANIFEST_SCHEMA = "tgvf-mmad-coredev-task-manifest-v1"
MMAD_PINNED_JSON_SHA256 = (
    "639343b491bc67b2abb3c5d719f221ce27f83b2ed97948f4e88055aaa31f1c1e"
)
MMAD_OFFICIAL_QUERY_COUNT = 8_366
MMAD_OFFICIAL_QUESTION_COUNT = 39_670

MMAD_OFFICIAL_INSTRUCTION = (
    "You are an industrial inspector who checks products by images. You "
    "should judge whether there is a defect in the query image and answer "
    "the questions about it.\n"
    "Answer with the option's letter from the given choices directly."
)
MMAD_DIRECT_ANSWER_INSTRUCTION = (
    "Answer with the option's letter from the given choices directly!"
)

MMAD_PANEL_SIZE = (512, 512)
MMAD_PANEL_LABEL_HEIGHT = 32
MMAD_PANEL_GAP = 8
MMAD_CANVAS_PADDING = 8
MMAD_PNG_COMPRESSION_LEVEL = 1
MMAD_CANVAS_WORKERS = min(16, max(1, os.cpu_count() or 1))
MMAD_TEMPLATE_LABEL = "NORMAL TEMPLATE"
MMAD_QUERY_LABEL = "QUERY"

MmadShot = Literal[0, 1]
MmadTemplateKind = Literal["random", "similar"]


def normalize_mmad_source(source: str) -> str:
    """Apply the official MMAD DS-MVTec/MVTec-AD reporting merge."""

    if not isinstance(source, str) or not source:
        raise ValueError("MMAD source name must be non-empty")
    return "MVTec-AD" if source in {"DS-MVTec", "MVTec-AD"} else source


def normalize_mmad_question_type(question_type: str) -> str:
    """Apply the official Object Structure/Details reporting merge."""

    if not isinstance(question_type, str) or not question_type:
        raise ValueError("MMAD question type must be non-empty")
    if question_type in {"Object Structure", "Object Details"}:
        return "Object Analysis"
    return question_type


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_safe_relative_path(value: object, *, owner: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"{owner} must stay below the MMAD snapshot root")
    return relative


def _resolve_snapshot_image(
    snapshot_root: Path, value: object, *, owner: str
) -> tuple[Path, str]:
    relative = _require_safe_relative_path(value, owner=owner)
    candidate = snapshot_root.joinpath(*relative.parts).resolve(strict=True)
    if not candidate.is_relative_to(snapshot_root) or not candidate.is_file():
        raise ValueError(f"{owner} is missing or escapes the MMAD snapshot root")
    return candidate, relative.as_posix()


def _read_decodable_image(path: Path) -> tuple[bytes, tuple[int, int]]:
    payload = path.read_bytes()
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            dimensions = (int(opened.width), int(opened.height))
    except (OSError, ValueError) as error:
        raise ValueError(f"MMAD image cannot be decoded: {path}") from error
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"MMAD image has invalid dimensions: {path}")
    return payload, dimensions


def _open_rgb(payload: bytes, *, path: Path) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            return opened.convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"MMAD image cannot be decoded: {path}") from error


def _letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = image.size
    target_width, target_height = size
    scale = min(target_width / width, target_height / height)
    resized_size = (
        max(1, min(target_width, round(width * scale))),
        max(1, min(target_height, round(height * scale))),
    )
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (18, 18, 18))
    offset = (
        (target_width - resized.width) // 2,
        (target_height - resized.height) // 2,
    )
    panel.paste(resized, offset)
    return panel


def _draw_centered_label(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    panel_left: int,
    panel_width: int,
    label_top: int,
    label_height: int,
) -> None:
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), label, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = panel_left + (panel_width - text_width) // 2 - bounds[0]
    y = label_top + (label_height - text_height) // 2 - bounds[1]
    draw.text((x, y), label, fill=(0, 0, 0), font=font)


def _render_one_shot_canvas(
    *,
    template_payload: bytes,
    template_path: Path,
    query_payload: bytes,
    query_path: Path,
) -> tuple[bytes, tuple[int, int]]:
    template = _letterbox(
        _open_rgb(template_payload, path=template_path), MMAD_PANEL_SIZE
    )
    query = _letterbox(_open_rgb(query_payload, path=query_path), MMAD_PANEL_SIZE)
    panel_width, panel_height = MMAD_PANEL_SIZE
    canvas_size = (
        2 * MMAD_CANVAS_PADDING + 2 * panel_width + MMAD_PANEL_GAP,
        2 * MMAD_CANVAS_PADDING + MMAD_PANEL_LABEL_HEIGHT + panel_height,
    )
    canvas = Image.new("RGB", canvas_size, (245, 245, 245))
    left_x = MMAD_CANVAS_PADDING
    right_x = left_x + panel_width + MMAD_PANEL_GAP
    image_y = MMAD_CANVAS_PADDING + MMAD_PANEL_LABEL_HEIGHT
    canvas.paste(template, (left_x, image_y))
    canvas.paste(query, (right_x, image_y))

    draw = ImageDraw.Draw(canvas)
    _draw_centered_label(
        draw,
        label=MMAD_TEMPLATE_LABEL,
        panel_left=left_x,
        panel_width=panel_width,
        label_top=MMAD_CANVAS_PADDING,
        label_height=MMAD_PANEL_LABEL_HEIGHT,
    )
    _draw_centered_label(
        draw,
        label=MMAD_QUERY_LABEL,
        panel_left=right_x,
        panel_width=panel_width,
        label_top=MMAD_CANVAS_PADDING,
        label_height=MMAD_PANEL_LABEL_HEIGHT,
    )
    divider_x = left_x + panel_width + MMAD_PANEL_GAP // 2
    draw.line(
        (
            divider_x,
            MMAD_CANVAS_PADDING,
            divider_x,
            canvas.height - MMAD_CANVAS_PADDING - 1,
        ),
        fill=(128, 128, 128),
        width=1,
    )

    encoded = io.BytesIO()
    canvas.save(
        encoded,
        format="PNG",
        optimize=False,
        compress_level=MMAD_PNG_COMPRESSION_LEVEL,
    )
    return encoded.getvalue(), canvas_size


def materialize_mmad_one_shot_image(
    *,
    snapshot_root: str | Path,
    query_image: str,
    template_image: str,
    canvas_root: str | Path,
) -> tuple[Path, str, tuple[int, int]]:
    """Render and content-address one labelled normal/query comparison PNG."""

    root = Path(snapshot_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("MMAD snapshot root must be a directory")
    query_path, _ = _resolve_snapshot_image(root, query_image, owner="query image")
    template_path, _ = _resolve_snapshot_image(
        root, template_image, owner="normal template image"
    )
    query_payload, _ = _read_decodable_image(query_path)
    template_payload, _ = _read_decodable_image(template_path)
    canvas_payload, dimensions = _render_one_shot_canvas(
        template_payload=template_payload,
        template_path=template_path,
        query_payload=query_payload,
        query_path=query_path,
    )
    digest = _sha256_bytes(canvas_payload)
    output_root = Path(canvas_root).resolve()
    target = output_root / digest[:2] / f"{digest}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise RuntimeError("MMAD one-shot canvas target is not a regular file")
        if target.read_bytes() != canvas_payload:
            raise RuntimeError("MMAD one-shot canvas content-address collision")
    else:
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(canvas_payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target.resolve(strict=True), digest, dimensions


def _official_prompt(
    *,
    question_number: int,
    question: str,
    options: Sequence[tuple[str, str]],
    shot: MmadShot,
) -> str:
    lines = [MMAD_OFFICIAL_INSTRUCTION, MMAD_DIRECT_ANSWER_INSTRUCTION]
    if shot == 1:
        lines.extend(
            (
                "Following is 1 image of normal sample, which can be used as "
                "a template to compare the image being queried.",
                "The single input is a two-panel image: the NORMAL TEMPLATE "
                "is on the left and the QUERY is on the right.",
            )
        )
    lines.extend(
        (
            "Following is the query image:",
            "Following is the question list:",
            f"Question {question_number}: {question}",
            *(f"{letter}. {value}" for letter, value in options),
        )
    )
    return "\n".join(lines) + "\n"


def _normalize_question(
    raw: object, *, question_number: int, shot: MmadShot
) -> tuple[str, str, list[list[str]], str, str]:
    if not isinstance(raw, Mapping):
        raise ValueError("MMAD conversation item must be an object")
    question = raw.get("Question")
    question_type = raw.get("type")
    raw_answer = raw.get("Answer")
    raw_options = raw.get("Options")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("MMAD question text must be non-empty")
    if not isinstance(question_type, str) or not question_type:
        raise ValueError("MMAD question type must be non-empty")
    if not isinstance(raw_answer, str) or not raw_answer:
        raise ValueError("MMAD answer must be non-empty")
    if not isinstance(raw_options, Mapping) or not 2 <= len(raw_options) <= 5:
        raise ValueError("MMAD question must have two to five ordered options")

    options: list[tuple[str, str]] = []
    answer: str | None = None
    for option_index, (source_key, value) in enumerate(raw_options.items()):
        if not isinstance(source_key, str) or not source_key:
            raise ValueError("MMAD option key must be non-empty")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("MMAD option text must be non-empty")
        canonical_key = chr(ord("A") + option_index)
        options.append((canonical_key, value))
        if source_key == raw_answer:
            answer = canonical_key
    if answer is None:
        raise ValueError("MMAD answer is absent from its ordered options")
    normalized_type = normalize_mmad_question_type(question_type)
    prompt = _official_prompt(
        question_number=question_number,
        question=question,
        options=options,
        shot=shot,
    )
    return (
        prompt,
        answer,
        [[letter, value] for letter, value in options],
        question_type,
        normalized_type,
    )


def _load_annotation(
    snapshot_root: Path, *, verify_official_source: bool
) -> tuple[Mapping[str, object], str, int]:
    annotation_path = snapshot_root / "mmad.json"
    if not annotation_path.is_file() or annotation_path.is_symlink():
        raise ValueError("MMAD snapshot must contain a regular-file annotation JSON")
    payload = annotation_path.read_bytes()
    digest = _sha256_bytes(payload)
    if verify_official_source and digest != MMAD_PINNED_JSON_SHA256:
        raise ValueError("MMAD annotation SHA256 differs from the pinned snapshot")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("MMAD annotation JSON is unreadable") from error
    if not isinstance(decoded, Mapping):
        raise ValueError("MMAD annotation root must be an object")
    question_count = 0
    for query_image, raw_entry in decoded.items():
        _require_safe_relative_path(query_image, owner="query image")
        if not isinstance(raw_entry, Mapping):
            raise ValueError("MMAD image annotation must be an object")
        conversation = raw_entry.get("conversation")
        if not isinstance(conversation, list) or not conversation:
            raise ValueError("MMAD image annotation must carry a conversation")
        question_count += len(conversation)
    if verify_official_source and (
        len(decoded) != MMAD_OFFICIAL_QUERY_COUNT
        or question_count != MMAD_OFFICIAL_QUESTION_COUNT
    ):
        raise ValueError("MMAD annotation cardinality differs from the official suite")
    return decoded, digest, question_count


def _sample_id(
    *, shot: MmadShot, template_kind: str, query_image: str, question_index: int
) -> str:
    canonical = "\0".join(
        (
            MMAD_ADAPTER_SCHEMA,
            str(shot),
            template_kind,
            query_image,
            str(question_index),
        )
    ).encode("utf-8")
    return f"mmad-{shot}shot-{hashlib.sha256(canonical).hexdigest()[:32]}"


def build_mmad_task_rows(
    *,
    snapshot_root: str | Path,
    shot: MmadShot,
    canvas_root: str | Path | None = None,
    template_kind: MmadTemplateKind = "random",
    stable_prefix: int | None = None,
    verify_official_source: bool = True,
) -> tuple[dict[str, object], ...]:
    """Flatten MMAD into one CoreDevTask-compatible row per question.

    ``stable_prefix`` takes the first N questions in the annotation's pinned
    insertion order.  It is intended for deterministic smoke tests; task IDs
    are identical to the corresponding prefix of a full build.
    """

    if shot not in (0, 1) or isinstance(shot, bool):
        raise ValueError("MMAD shot must be integer 0 or 1")
    if template_kind not in {"random", "similar"}:
        raise ValueError("MMAD template_kind must be random or similar")
    if stable_prefix is not None and (
        type(stable_prefix) is not int or stable_prefix <= 0
    ):
        raise ValueError("MMAD stable_prefix must be a positive integer")
    if shot == 1 and canvas_root is None:
        raise ValueError("MMAD one-shot adaptation requires canvas_root")

    root = Path(snapshot_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("MMAD snapshot root must be a directory")
    annotation, annotation_sha256, full_question_count = _load_annotation(
        root, verify_official_source=verify_official_source
    )
    target_count = full_question_count if stable_prefix is None else stable_prefix
    if target_count > full_question_count:
        raise ValueError("MMAD stable_prefix exceeds the annotation question count")

    selected_entries: list[tuple[int, str, Mapping[str, object]]] = []
    selected_question_count = 0
    for query_index, (raw_query_image, raw_entry) in enumerate(annotation.items()):
        assert isinstance(raw_query_image, str)
        assert isinstance(raw_entry, Mapping)
        selected_entries.append((query_index, raw_query_image, raw_entry))
        conversation = raw_entry["conversation"]
        assert isinstance(conversation, list)
        selected_question_count += len(conversation)
        if selected_question_count >= target_count:
            break

    one_shot_images: dict[int, tuple[Path, str, tuple[int, int], str]] = {}
    if shot == 1:
        assert canvas_root is not None

        def prepare_one_shot(
            item: tuple[int, str, Mapping[str, object]],
        ) -> tuple[int, Path, str, tuple[int, int], str]:
            query_index, raw_query_image, raw_entry = item
            _, query_image = _resolve_snapshot_image(
                root, raw_query_image, owner="query image"
            )
            template_field = f"{template_kind}_templates"
            raw_templates = raw_entry.get(template_field)
            if not isinstance(raw_templates, list) or not raw_templates:
                raise ValueError(f"MMAD entry has no {template_field}")
            _, template_image = _resolve_snapshot_image(
                root,
                raw_templates[0],
                owner=f"{template_kind} normal template",
            )
            path, digest, dimensions = materialize_mmad_one_shot_image(
                snapshot_root=root,
                query_image=query_image,
                template_image=template_image,
                canvas_root=canvas_root,
            )
            return query_index, path, digest, dimensions, template_image

        with ThreadPoolExecutor(
            max_workers=MMAD_CANVAS_WORKERS,
            thread_name_prefix="mmad-canvas",
        ) as executor:
            for query_index, path, digest, dimensions, template_image in executor.map(
                prepare_one_shot, selected_entries
            ):
                one_shot_images[query_index] = (
                    path,
                    digest,
                    dimensions,
                    template_image,
                )

    rows: list[dict[str, object]] = []
    selected_template_kind = template_kind if shot == 1 else "none"
    for query_index, raw_query_image, raw_entry in selected_entries:
        if len(rows) == target_count:
            break
        assert isinstance(raw_query_image, str)
        assert isinstance(raw_entry, Mapping)
        query_path, query_image = _resolve_snapshot_image(
            root, raw_query_image, owner="query image"
        )
        raw_source = PurePosixPath(query_image).parts[0]
        source = normalize_mmad_source(raw_source)
        is_normal = any(
            component.casefold() in {"good", "normal"}
            for component in PurePosixPath(query_image).parts
        )
        object_category = (
            PurePosixPath(query_image).parts[1]
            if len(PurePosixPath(query_image).parts) > 1
            else "unknown"
        )

        template_image: str | None = None
        if shot == 0:
            query_payload, dimensions = _read_decodable_image(query_path)
            effective_path = query_path
            effective_sha256 = _sha256_bytes(query_payload)
            effective_dimensions = dimensions
        else:
            (
                effective_path,
                effective_sha256,
                effective_dimensions,
                template_image,
            ) = one_shot_images[query_index]

        conversation = raw_entry["conversation"]
        assert isinstance(conversation, list)
        for question_index, raw_question in enumerate(conversation):
            if len(rows) == target_count:
                break
            prompt, answer, options, raw_type, normalized_type = _normalize_question(
                raw_question,
                question_number=question_index + 1,
                shot=shot,
            )
            assert isinstance(raw_question, Mapping)
            annotation_flag = raw_question.get("annotation")
            metadata = {
                "benchmark": "MMAD",
                "adapter_schema": MMAD_ADAPTER_SCHEMA,
                "annotation_sha256": annotation_sha256,
                "shot": str(shot),
                "template_kind": selected_template_kind,
                "effective_image_layout": (
                    "query_only" if shot == 0 else "normal_template_left__query_right"
                ),
                "query_image": query_image,
                "query_index": str(query_index),
                "question_index": str(question_index),
                "question_number": str(question_index + 1),
                "source_dataset_raw": raw_source,
                "source_dataset": source,
                "score_dataset": source,
                "object_category": object_category,
                "question_type_raw": raw_type,
                "question_type": normalized_type,
                "question_type_score": normalized_type,
                "is_normal": str(is_normal).lower(),
                # Generic policy scoring groups these two established keys.
                "category": normalized_type,
                "cycle_category": source,
                "annotation": (
                    str(annotation_flag).lower()
                    if isinstance(annotation_flag, bool)
                    else "unspecified"
                ),
            }
            if template_image is not None:
                metadata["template_image"] = template_image
            sample_id = _sample_id(
                shot=shot,
                template_kind=selected_template_kind,
                query_image=query_image,
                question_index=question_index,
            )
            rows.append(
                {
                    "ordinal": len(rows),
                    "dataset": "MMAD",
                    "row_number": len(rows),
                    "index": sample_id,
                    "sample_id": sample_id,
                    "question": prompt,
                    "image_paths": [str(effective_path)],
                    "answer": answer,
                    "options": options,
                    "metadata": metadata,
                    "image_sha256s": [effective_sha256],
                    "image_dimensions": [list(effective_dimensions)],
                }
            )

    validate_mmad_task_rows(rows, expected_count=target_count, verify_images=True)
    return tuple(rows)


def canonical_mmad_manifest_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    """Encode rows exactly as the generic policy benchmark JSONL boundary."""

    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def validate_mmad_task_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_count: int | None = None,
    verify_images: bool = True,
) -> None:
    """Enforce the single-image and immutable-identity MMAD invariants."""

    if expected_count is not None and len(rows) != expected_count:
        raise ValueError("MMAD task row count differs")
    expected_ordinals = tuple(range(len(rows)))
    observed_ordinals = tuple(row.get("ordinal") for row in rows)
    if observed_ordinals != expected_ordinals:
        raise ValueError("MMAD task row order differs")
    sample_ids: set[str] = set()
    image_cache: dict[Path, tuple[str, tuple[int, int]]] = {}
    for row in rows:
        if row.get("dataset") != "MMAD":
            raise ValueError("MMAD task dataset differs")
        sample_id = row.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or row.get("index") != sample_id
            or sample_id in sample_ids
        ):
            raise ValueError("MMAD task sample identity is malformed or duplicated")
        sample_ids.add(sample_id)
        image_paths = row.get("image_paths")
        image_sha256s = row.get("image_sha256s")
        image_dimensions = row.get("image_dimensions")
        if (
            not isinstance(image_paths, list)
            or len(image_paths) != 1
            or not isinstance(image_paths[0], str)
            or not Path(image_paths[0]).is_absolute()
            or not isinstance(image_sha256s, list)
            or len(image_sha256s) != 1
            or not isinstance(image_dimensions, list)
            or len(image_dimensions) != 1
        ):
            raise ValueError("MMAD task must bind exactly one absolute image")
        digest = image_sha256s[0]
        dimensions = image_dimensions[0]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(dimensions, list)
            or len(dimensions) != 2
            or any(type(value) is not int or value <= 0 for value in dimensions)
        ):
            raise ValueError("MMAD task image identity is malformed")
        options = row.get("options")
        answer = row.get("answer")
        if not isinstance(options, list) or answer not in {
            item[0] for item in options if isinstance(item, list) and len(item) == 2
        }:
            raise ValueError("MMAD task answer/options are malformed")
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("MMAD task metadata is missing")
        if any("mask" in str(key).lower() for key in metadata):
            raise ValueError("MMAD mask annotations must not enter task rows")
        normalized_source = normalize_mmad_source(
            str(metadata.get("source_dataset_raw", ""))
        )
        if (
            metadata.get("source_dataset") != normalized_source
            or metadata.get("score_dataset") != normalized_source
        ):
            raise ValueError("MMAD source normalization differs")
        normalized_type = normalize_mmad_question_type(
            str(metadata.get("question_type_raw", ""))
        )
        if (
            metadata.get("question_type") != normalized_type
            or metadata.get("question_type_score") != normalized_type
        ):
            raise ValueError("MMAD question-type normalization differs")
        query_image = metadata.get("query_image")
        if not isinstance(query_image, str) or not query_image:
            raise ValueError("MMAD query-image metadata is missing")
        expected_normal = any(
            component.casefold() in {"good", "normal"}
            for component in PurePosixPath(query_image).parts
        )
        if metadata.get("is_normal") != str(expected_normal).lower():
            raise ValueError("MMAD normal/anomalous metadata differs")
        if verify_images:
            image_path = Path(image_paths[0])
            if image_path not in image_cache:
                payload, actual_dimensions = _read_decodable_image(image_path)
                image_cache[image_path] = (
                    _sha256_bytes(payload),
                    actual_dimensions,
                )
            actual_digest, actual_dimensions = image_cache[image_path]
            if actual_digest != digest or list(actual_dimensions) != dimensions:
                raise ValueError("MMAD task image identity changed")


def mmad_manifest_identity(
    rows: Sequence[Mapping[str, object]],
    *,
    source_json_sha256: str,
    stable_prefix: int | None,
) -> dict[str, object]:
    """Return the portable identity fields for a validated MMAD manifest."""

    validate_mmad_task_rows(rows, verify_images=False)
    if (
        not isinstance(source_json_sha256, str)
        or len(source_json_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_json_sha256)
    ):
        raise ValueError("MMAD source JSON SHA256 is malformed")
    if stable_prefix is not None and stable_prefix != len(rows):
        raise ValueError("MMAD manifest stable_prefix differs from its task count")
    payload = canonical_mmad_manifest_bytes(rows)
    return {
        "schema_version": MMAD_MANIFEST_SCHEMA,
        "adapter_schema": MMAD_ADAPTER_SCHEMA,
        "source_json_sha256": source_json_sha256,
        "stable_prefix": stable_prefix,
        "task_count": len(rows),
        "single_image_count": len(rows),
        "manifest_sha256": _sha256_bytes(payload),
    }


__all__ = [
    "MMAD_ADAPTER_SCHEMA",
    "MMAD_CANVAS_WORKERS",
    "MMAD_MANIFEST_SCHEMA",
    "MMAD_OFFICIAL_QUERY_COUNT",
    "MMAD_OFFICIAL_QUESTION_COUNT",
    "MMAD_PANEL_SIZE",
    "MMAD_PINNED_JSON_SHA256",
    "MMAD_PNG_COMPRESSION_LEVEL",
    "MMAD_QUERY_LABEL",
    "MMAD_TEMPLATE_LABEL",
    "build_mmad_task_rows",
    "canonical_mmad_manifest_bytes",
    "materialize_mmad_one_shot_image",
    "mmad_manifest_identity",
    "normalize_mmad_question_type",
    "normalize_mmad_source",
    "validate_mmad_task_rows",
]
