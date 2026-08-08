"""Fail-closed Crop grounding diagnostics over frozen rollout-zero probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tgvf_rl.qwen.crop_coordinates import (
    QWEN3_CROP_CONVERSION_VERSION,
    QWEN3_CROP_COORDINATE_SPACE,
    map_qwen3_crop_bbox_to_source,
)

from .policy_benchmark_scoring import load_policy_evaluation_identity
from .policy_coredev import (
    CoreDevTask,
    load_benchmark_tasks,
    load_policy_benchmark_results,
)


CROP_GROUNDING_PROBE_SCHEMA = "crop-grounding-probe-sample-id-manifest-v1"
CROP_GROUNDING_REPORT_SCHEMA = "crop-grounding-report-v2"
POLICY_TRAJECTORY_AUDIT_SCHEMA = "policy-trajectory-audit-v1"
_CANDIDATE_SCHEMA = "tgvf.policy-selection.candidate.v1"
_CANDIDATE_MANIFEST_SCHEMA = "tgvf.policy-selection.source-manifest.v1"
_CROP_TOOL_NAME = "image_zoom_in_tool"
_AUDIT_MODES = frozenset({"training", "benchmark"})
_ORIGINAL_IMAGE_PIXEL_COORDINATE_SPACE = "original_image_pixels_xyxy_v1"
_ORIGINAL_IMAGE_PIXEL_CONVERSION_VERSION = "deepeyes-source-pixel-clipping-v1"


@dataclass(frozen=True, slots=True)
class GroundingCandidate:
    """Immutable source-image geometry for one frozen probe sample."""

    sample_id: str
    question: str
    image_path: str
    image_sha256: str
    source_width: int
    source_height: int
    gt_regions: tuple[tuple[int, int, int, int], ...]


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 of the exact file bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _require_exact_file_sha256(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> str:
    expected = _require_sha256(expected_sha256, name=f"expected {name} SHA256")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{name} SHA256 mismatch: expected {expected}, observed {actual}"
        )
    return actual


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _bbox(
    value: object,
    *,
    name: str,
    width: int | None = None,
    height: int | None = None,
) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must contain exactly four integers")
    left, top, right, bottom = value
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError(f"{name} must be a non-empty non-negative xyxy box")
    if width is not None and height is not None and (right > width or bottom > height):
        raise ValueError(f"{name} lies outside the source image")
    return left, top, right, bottom


def _numeric_bbox(value: object, *, name: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain exactly four finite numbers")
    left, top, right, bottom = (float(item) for item in value)
    if not left < right or not top < bottom:
        raise ValueError(f"{name} must be a non-empty xyxy box")
    return left, top, right, bottom


def _official_source_bbox(
    requested: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = requested
    effective = (
        math.floor(max(0.0, left)),
        math.floor(max(0.0, top)),
        math.ceil(min(float(width), right)),
        math.ceil(min(float(height), bottom)),
    )
    crop_width = effective[2] - effective[0]
    crop_height = effective[3] - effective[1]
    if crop_width <= 30 or crop_height <= 30:
        raise ValueError("official source-pixel crop dimensions must exceed 30 pixels")
    if max(crop_width, crop_height) / min(crop_width, crop_height) > 100:
        raise ValueError("official source-pixel crop aspect ratio exceeds 100")
    return effective


def _load_probe_manifest(
    path: Path,
    *,
    expected_sha256: str,
    candidate_manifest_file_sha256: str,
    candidates_jsonl_sha256: str,
) -> tuple[tuple[str, ...], str, Path, str]:
    probe_file_sha256 = _require_exact_file_sha256(
        path, expected_sha256, name="probe manifest file"
    )
    payload = _load_json_object(path, name="probe manifest")
    if payload.get("schema_version") != CROP_GROUNDING_PROBE_SCHEMA:
        raise ValueError("unsupported crop grounding probe manifest schema")
    if payload.get("candidate_manifest_file_sha256") != candidate_manifest_file_sha256:
        raise ValueError("probe manifest is bound to a different candidate manifest")
    if payload.get("candidates_jsonl_sha256") != candidates_jsonl_sha256:
        raise ValueError("probe manifest is bound to a different candidates JSONL")
    raw_ids = payload.get("ordered_sample_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("probe ordered_sample_ids must be a non-empty list")
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in raw_ids):
        raise ValueError("probe sample IDs must be non-empty strings")
    sample_ids = tuple(raw_ids)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("probe ordered_sample_ids contains duplicates")
    declared_count = payload.get("sample_count")
    if declared_count != len(sample_ids):
        raise ValueError("probe sample_count differs from ordered_sample_ids")
    task_binding = payload.get("task_manifest")
    if not isinstance(task_binding, Mapping):
        raise ValueError("probe task_manifest binding is missing")
    relative = task_binding.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("probe task manifest path must be safe and relative")
    task_path = path.parent / relative
    task_sha256 = _require_sha256(
        task_binding.get("sha256"), name="probe task manifest SHA256"
    )
    _require_exact_file_sha256(task_path, task_sha256, name="probe task manifest")
    if task_binding.get("row_count") != len(sample_ids):
        raise ValueError("probe task manifest row count differs")
    return sample_ids, probe_file_sha256, task_path, task_sha256


def _candidate_jsonl_from_manifest(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[Path, str, int, str]:
    manifest_file_sha256 = _require_exact_file_sha256(
        path, expected_sha256, name="candidate manifest file"
    )
    payload = _load_json_object(path, name="candidate manifest")
    if payload.get("schema_version") != _CANDIDATE_MANIFEST_SCHEMA:
        raise ValueError("unsupported candidate manifest schema")
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("candidate manifest candidates must be an object")
    relative = candidates.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("candidate JSONL path must be a safe relative path")
    candidates_path = (path.parent / relative).resolve()
    try:
        candidates_path.relative_to(path.parent.resolve())
    except ValueError as error:
        raise ValueError("candidate JSONL escapes the manifest directory") from error
    candidates_sha256 = _require_sha256(
        candidates.get("sha256"), name="candidate JSONL manifest SHA256"
    )
    _require_exact_file_sha256(
        candidates_path,
        candidates_sha256,
        name="candidate JSONL",
    )
    candidate_rows = _nonnegative_int(
        payload.get("candidate_rows"), name="candidate_rows"
    )
    return (
        candidates_path,
        candidates_sha256,
        candidate_rows,
        manifest_file_sha256,
    )


def _load_candidates(
    path: Path,
    *,
    expected_rows: int,
    probe_sample_ids: Sequence[str],
) -> dict[str, GroundingCandidate]:
    wanted = set(probe_sample_ids)
    found: dict[str, GroundingCandidate] = {}
    seen: set[str] = set()
    row_count = 0
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read candidate JSONL: {path}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            row_count += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"candidate JSONL contains invalid JSON at line {line_number}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(f"candidate line {line_number} must be an object")
            if payload.get("schema_version") != _CANDIDATE_SCHEMA:
                raise ValueError(
                    f"candidate line {line_number} has an unsupported schema"
                )
            sample_id = payload.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"candidate line {line_number} has no valid sample_id")
            if sample_id in seen:
                raise ValueError(
                    f"candidate JSONL contains duplicate sample_id {sample_id}"
                )
            seen.add(sample_id)
            if sample_id not in wanted:
                continue
            image = payload.get("image")
            if not isinstance(image, dict):
                raise ValueError(f"probe candidate {sample_id} has no image geometry")
            width = _positive_int(image.get("width"), name=f"{sample_id} image width")
            height = _positive_int(
                image.get("height"), name=f"{sample_id} image height"
            )
            question = payload.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"probe candidate {sample_id} has no question")
            image_path = image.get("path")
            if not isinstance(image_path, str) or not Path(image_path).is_absolute():
                raise ValueError(
                    f"probe candidate {sample_id} image path must be absolute"
                )
            image_sha256 = _require_sha256(
                image.get("sha256"), name=f"{sample_id} image SHA256"
            )
            raw_regions = payload.get("gt_regions")
            if not isinstance(raw_regions, list) or not raw_regions:
                raise ValueError(f"probe candidate {sample_id} has no GT regions")
            regions = tuple(
                _bbox(
                    region,
                    name=f"{sample_id} GT region {region_index}",
                    width=width,
                    height=height,
                )
                for region_index, region in enumerate(raw_regions)
            )
            found[sample_id] = GroundingCandidate(
                sample_id=sample_id,
                question=question,
                image_path=image_path,
                image_sha256=image_sha256,
                source_width=width,
                source_height=height,
                gt_regions=regions,
            )
    if row_count != expected_rows:
        raise ValueError(
            f"candidate JSONL row count mismatch: expected {expected_rows}, observed {row_count}"
        )
    missing = wanted.difference(found)
    if missing:
        raise ValueError(
            f"probe sample IDs are absent from candidates: {sorted(missing)[:5]}"
        )
    return found


def _audit_tree_sha256(step_dir: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(step_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _load_rollout_zero_audits(
    audit_root: Path,
    *,
    behavior_step: int,
    probe_sample_ids: Sequence[str],
) -> tuple[dict[str, tuple[Path, dict[str, Any]]], Path, str, int]:
    step = _nonnegative_int(behavior_step, name="behavior_step")
    step_dir = audit_root / f"step-{step:08d}"
    if not step_dir.is_dir():
        raise ValueError(f"trajectory audit step directory does not exist: {step_dir}")
    paths = sorted(step_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"trajectory audit step directory is empty: {step_dir}")
    expected = set(probe_sample_ids)
    accepted: dict[str, tuple[Path, dict[str, Any]]] = {}
    ignored_nonzero = 0
    for path in paths:
        payload = _load_json_object(path, name="trajectory audit")
        if payload.get("schema_version") != POLICY_TRAJECTORY_AUDIT_SCHEMA:
            raise ValueError(
                f"{path}: unsupported trajectory audit schema; "
                "a training-writer-compatible audit with rollout_index is required"
            )
        if payload.get("optimizer_step") != step:
            raise ValueError(
                f"{path}: optimizer_step differs from behavior step {step}"
            )
        sample_id = payload.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}: sample_id must be a non-empty string")
        if sample_id not in expected:
            raise ValueError(f"{path}: unknown sample_id {sample_id}")
        rollout_index = _nonnegative_int(
            payload.get("rollout_index"), name=f"{path} rollout_index"
        )
        if rollout_index != 0:
            ignored_nonzero += 1
            continue
        if sample_id in accepted:
            raise ValueError(
                f"duplicate rollout_index=0 audit for sample_id {sample_id}"
            )
        reasons = payload.get("selection_reasons")
        if (
            not isinstance(reasons, list)
            or "representative_rollout_zero" not in reasons
        ):
            raise ValueError(
                f"{path}: rollout zero lacks its representative selection reason"
            )
        accepted[sample_id] = (path, payload)
    missing = expected.difference(accepted)
    if missing:
        raise ValueError(f"missing rollout_index=0 audits: {sorted(missing)[:5]}")
    return accepted, step_dir, _audit_tree_sha256(step_dir, paths), ignored_nonzero


def _rank_jsonl_paths(
    inference_dir: Path, *, expected_world_size: int | None = None
) -> tuple[Path, ...]:
    if not inference_dir.is_dir():
        raise ValueError(
            f"benchmark inference directory does not exist: {inference_dir}"
        )
    indexed: list[tuple[int, Path]] = []
    for path in inference_dir.glob("rank-*.jsonl"):
        suffix = path.stem.removeprefix("rank-")
        if not suffix.isdigit() or str(int(suffix)) != suffix:
            raise ValueError(f"invalid benchmark rank JSONL name: {path.name}")
        indexed.append((int(suffix), path))
    indexed.sort()
    if not indexed:
        raise ValueError(f"benchmark inference has no rank JSONLs: {inference_dir}")
    observed_ranks = tuple(rank for rank, _path in indexed)
    expected_ranks = tuple(
        range(len(indexed) if expected_world_size is None else expected_world_size)
    )
    if observed_ranks != expected_ranks:
        raise ValueError(
            "benchmark inference rank JSONLs differ from the bound world size"
        )
    return tuple(path for _rank, path in indexed)


def _validate_probe_tasks(
    *,
    probe_sample_ids: Sequence[str],
    candidates: Mapping[str, GroundingCandidate],
    tasks: Sequence[CoreDevTask],
) -> None:
    if len(tasks) != len(probe_sample_ids):
        raise ValueError("probe task count differs from frozen probe")
    for ordinal, (sample_id, task) in enumerate(
        zip(probe_sample_ids, tasks, strict=True)
    ):
        candidate = candidates[sample_id]
        expected_task = {
            "ordinal": ordinal,
            "sample_id": sample_id,
            "index": sample_id,
            "question": candidate.question,
            "image_paths": (candidate.image_path,),
            "image_sha256s": (candidate.image_sha256,),
            "image_dimensions": ((candidate.source_width, candidate.source_height),),
        }
        for field, expected_value in expected_task.items():
            observed = (
                task.bound_sample_id if field == "sample_id" else getattr(task, field)
            )
            if observed != expected_value:
                raise ValueError(f"probe task {field} differs from frozen candidate")


def _load_benchmark_audits(
    audit_root: Path,
    *,
    behavior_step: int,
    probe_sample_ids: Sequence[str],
    tasks: Sequence[CoreDevTask],
    evaluation_identity: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[Path, dict[str, Any]]],
    Path,
    str,
    int,
    dict[str, Any],
]:
    step = _nonnegative_int(behavior_step, name="behavior_step")
    policy_snapshot = evaluation_identity.get("policy_snapshot")
    execution = evaluation_identity.get("execution")
    if not isinstance(policy_snapshot, Mapping) or not isinstance(execution, Mapping):
        raise ValueError("benchmark evaluation identity is malformed")
    if policy_snapshot.get("optimizer_step") != step:
        raise ValueError("benchmark optimizer_step differs from evaluation identity")
    world_size = _positive_int(
        execution.get("world_size"), name="benchmark evaluation world_size"
    )
    inference_dir = audit_root / "inference"
    rank_paths = _rank_jsonl_paths(inference_dir, expected_world_size=world_size)
    try:
        records = load_policy_benchmark_results(
            inference_dir,
            tasks=tasks,
            evaluation_identity=evaluation_identity,
            require_complete=True,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"benchmark audit identity validation failed: {error}"
        ) from error
    accepted = {
        sample_id: (
            inference_dir / f"rank-{ordinal % world_size}.jsonl",
            records[ordinal],
        )
        for ordinal, sample_id in enumerate(probe_sample_ids)
    }
    checkpoint_identity = {
        "evaluation_id": evaluation_identity["evaluation_id"],
        "evaluation_identity_sha256": evaluation_identity["identity_sha256"],
        "policy_run_id": policy_snapshot["run_id"],
        "policy_run_identity_sha256": policy_snapshot["run_identity_sha256"],
        "optimizer_step": policy_snapshot["optimizer_step"],
        "policy_weights_sha256": policy_snapshot["weights_sha256"],
    }
    snapshot_backend = policy_snapshot.get("snapshot_backend", "lora")
    if snapshot_backend == "full_model":
        checkpoint_identity.update(
            {
                "policy_snapshot_backend": "full_model",
                "policy_full_snapshot_identity_sha256": policy_snapshot[
                    "snapshot_identity_sha256"
                ],
                "policy_checkpoint_sha256": policy_snapshot["checkpoint_sha256"],
                "policy_source_tree_sha256": policy_snapshot["source_tree_sha256"],
                "policy_materialization_identity_sha256": policy_snapshot[
                    "materialization_identity_sha256"
                ],
                "policy_materialized_model_tree_sha256": policy_snapshot[
                    "materialized_model_tree_sha256"
                ],
            }
        )
    elif snapshot_backend in {"lora", "lora_adapter"}:
        checkpoint_identity.update(
            {
                "policy_pointer_file_sha256": policy_snapshot["pointer_file_sha256"],
                "policy_manifest_file_sha256": policy_snapshot["manifest_file_sha256"],
                "policy_tensor_file_sha256": policy_snapshot["tensor_file_sha256"],
            }
        )
    else:
        raise ValueError("benchmark policy snapshot backend differs")
    return (
        accepted,
        inference_dir,
        _audit_tree_sha256(inference_dir, rank_paths),
        0,
        checkpoint_identity,
    )


def _iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    if union <= 0:  # pragma: no cover - boxes are validated before this boundary
        raise ValueError("IoU requires non-empty boxes")
    return intersection / union


def _contains_gt_center(
    crop: tuple[int, int, int, int],
    gt: tuple[int, int, int, int],
) -> bool:
    center_x = (gt[0] + gt[2]) / 2.0
    center_y = (gt[1] + gt[3]) / 2.0
    return crop[0] <= center_x < crop[2] and crop[1] <= center_y < crop[3]


def _intersection_box(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    box = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


def _union_area(boxes: Sequence[tuple[int, int, int, int]]) -> int:
    """Return exact area of the union of integer xyxy rectangles."""

    if not boxes:
        return 0
    xs = sorted({coordinate for box in boxes for coordinate in (box[0], box[2])})
    area = 0
    for left, right in zip(xs, xs[1:], strict=False):
        if right <= left:
            continue
        intervals = sorted(
            (box[1], box[3]) for box in boxes if box[0] < right and box[2] > left
        )
        covered_height = 0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered_height += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered_height += end - start
        area += (right - left) * covered_height
    return area


def _intersection_union_area(
    first: Sequence[tuple[int, int, int, int]],
    second: Sequence[tuple[int, int, int, int]],
) -> int:
    intersections = [
        intersection
        for left in first
        for right in second
        if (intersection := _intersection_box(left, right)) is not None
    ]
    return _union_area(intersections)


def _geometry_metrics(
    crops: Sequence[tuple[int, int, int, int]],
    gt_regions: Sequence[tuple[int, int, int, int]],
) -> dict[str, float]:
    gt_area = _union_area(gt_regions)
    if gt_area <= 0:  # pragma: no cover - candidate GT boxes are validated
        raise ValueError("grounding GT union must have positive area")
    region_best_ious = [
        max((_iou(crop, gt) for crop in crops), default=0.0) for gt in gt_regions
    ]
    center_hits = [
        any(_contains_gt_center(crop, gt) for crop in crops) for gt in gt_regions
    ]
    return {
        "max_pair_iou": max(region_best_ious, default=0.0),
        "gt_area_recall": _intersection_union_area(crops, gt_regions) / gt_area,
        "gt_region_iou_recall_at_0_1": sum(value >= 0.1 for value in region_best_ious)
        / len(gt_regions),
        "gt_region_iou_recall_at_0_3": sum(value >= 0.3 for value in region_best_ious)
        / len(gt_regions),
        "gt_center_recall": sum(center_hits) / len(gt_regions),
    }


def _tool_attempted(payload: Mapping[str, Any], *, path: Path) -> bool:
    errors = payload.get("tool_errors")
    if not isinstance(errors, list) or any(
        not isinstance(error, dict) for error in errors
    ):
        raise ValueError(f"{path}: tool_errors must be a list of objects")
    if any(error.get("function_name") != _CROP_TOOL_NAME for error in errors):
        raise ValueError(f"{path}: tool error is not for {_CROP_TOOL_NAME}")
    return bool(payload.get("tool_calls")) or bool(errors)


def _score_sample(
    candidate: GroundingCandidate,
    *,
    audit_path: Path,
    payload: Mapping[str, Any],
    expected_coordinate_space: str,
) -> dict[str, Any]:
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list) or any(
        not isinstance(call, dict) for call in raw_calls
    ):
        raise ValueError(f"{audit_path}: tool_calls must be a list of objects")
    observation_count = _nonnegative_int(
        payload.get("successful_observation_count"),
        name=f"{audit_path} successful_observation_count",
    )
    if observation_count != len(raw_calls):
        raise ValueError(
            f"{audit_path}: successful observations must equal successful tool calls"
        )
    call_indices: set[int] = set()
    requested_boxes: list[
        tuple[int, int, int, int] | tuple[float, float, float, float]
    ] = []
    model_boxes: list[tuple[int, int, int, int] | None] = []
    source_boxes: list[tuple[int, int, int, int]] = []
    for call_number, call in enumerate(raw_calls):
        if call.get("function_name") != _CROP_TOOL_NAME:
            raise ValueError(f"{audit_path}: successful call is not {_CROP_TOOL_NAME}")
        call_index = _nonnegative_int(
            call.get("call_index"), name=f"{audit_path} call_index"
        )
        if call_index in call_indices:
            raise ValueError(f"{audit_path}: duplicate successful call_index")
        if call_index != call_number:
            raise ValueError(f"{audit_path}: successful call_index order differs")
        call_indices.add(call_index)
        coordinate_space = call.get("coordinate_space", QWEN3_CROP_COORDINATE_SPACE)
        if coordinate_space != expected_coordinate_space:
            raise ValueError(
                f"{audit_path}: crop coordinate space differs from evaluation identity"
            )
        if coordinate_space == QWEN3_CROP_COORDINATE_SPACE:
            model_bbox = _bbox(
                call.get("bbox_2d"), name=f"{audit_path} crop {call_number}"
            )
            try:
                mapping = map_qwen3_crop_bbox_to_source(
                    model_bbox,
                    source_width=candidate.source_width,
                    source_height=candidate.source_height,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{audit_path}: invalid Qwen3 crop coordinates for "
                    f"{candidate.sample_id}"
                ) from error
            if "source_bbox_2d" in call:
                audited_source = _bbox(
                    call.get("source_bbox_2d"),
                    name=f"{audit_path} crop {call_number} source bbox",
                    width=candidate.source_width,
                    height=candidate.source_height,
                )
                if audited_source != mapping.source_bbox_2d:
                    raise ValueError(
                        f"{audit_path}: Qwen3 source bbox differs from family mapping"
                    )
                for field in ("effective_bbox_2d",):
                    if field in call and _bbox(
                        call.get(field),
                        name=f"{audit_path} crop {call_number} {field}",
                        width=candidate.source_width,
                        height=candidate.source_height,
                    ) != mapping.source_bbox_2d:
                        raise ValueError(
                            f"{audit_path}: Qwen3 effective bbox differs from mapping"
                        )
                if call.get("conversion_version") != QWEN3_CROP_CONVERSION_VERSION:
                    raise ValueError(
                        f"{audit_path}: Qwen3 conversion version differs"
                    )
                if call.get("coordinate_reference_size") != [1000, 1000]:
                    raise ValueError(
                        f"{audit_path}: Qwen3 coordinate reference differs"
                    )
                if call.get("source_size") != [
                    candidate.source_width,
                    candidate.source_height,
                ]:
                    raise ValueError(f"{audit_path}: Qwen3 source size differs")
                if (
                    call.get("crop_width")
                    != mapping.source_bbox_2d[2] - mapping.source_bbox_2d[0]
                    or call.get("crop_height")
                    != mapping.source_bbox_2d[3] - mapping.source_bbox_2d[1]
                ):
                    raise ValueError(f"{audit_path}: Qwen3 crop dimensions differ")
                if call.get("crop_source") != "immutable_original_image":
                    raise ValueError(f"{audit_path}: Qwen3 crop source differs")
                _require_sha256(
                    call.get("crop_rgb_sha256"),
                    name=f"{audit_path} crop {call_number} RGB SHA256",
                )
            requested_boxes.append(model_bbox)
            model_boxes.append(mapping.model_bbox_2d)
            source_boxes.append(mapping.source_bbox_2d)
            continue
        if coordinate_space != _ORIGINAL_IMAGE_PIXEL_COORDINATE_SPACE:
            raise ValueError(f"{audit_path}: unsupported crop coordinate space")
        requested = _numeric_bbox(
            call.get("bbox_2d"), name=f"{audit_path} crop {call_number} request"
        )
        try:
            expected_source = _official_source_bbox(
                requested,
                width=candidate.source_width,
                height=candidate.source_height,
            )
            source_bbox = _bbox(
                call.get("source_bbox_2d"),
                name=f"{audit_path} crop {call_number} source bbox",
                width=candidate.source_width,
                height=candidate.source_height,
            )
        except ValueError as error:
            raise ValueError(
                f"{audit_path}: invalid official source-pixel crop for "
                f"{candidate.sample_id}"
            ) from error
        if source_bbox != expected_source:
            raise ValueError(
                f"{audit_path}: official source bbox differs from clipped request"
            )
        if call.get("crop_source") != "immutable_original_image":
            raise ValueError(f"{audit_path}: official crop source differs")
        if (
            call.get("crop_width") != source_bbox[2] - source_bbox[0]
            or call.get("crop_height") != source_bbox[3] - source_bbox[1]
        ):
            raise ValueError(f"{audit_path}: official crop dimensions differ")
        _require_sha256(
            call.get("crop_rgb_sha256"),
            name=f"{audit_path} crop {call_number} RGB SHA256",
        )
        requested_boxes.append(requested)
        model_boxes.append(None)
        source_boxes.append(source_bbox)

    first_call = _geometry_metrics(source_boxes[:1], candidate.gt_regions)
    all_calls = _geometry_metrics(source_boxes, candidate.gt_regions)
    source_area = candidate.source_width * candidate.source_height
    gt_area = _union_area(candidate.gt_regions)
    call_metrics: list[dict[str, Any]] = []
    prior: list[tuple[int, int, int, int]] = []
    prior_gt_coverage_area = 0
    prior_center_hits: set[int] = set()
    for call_index, (model_box, requested_box, crop) in enumerate(
        zip(model_boxes, requested_boxes, source_boxes, strict=True)
    ):
        crop_area = _union_area((crop,))
        crop_gt_intersection = _intersection_union_area((crop,), candidate.gt_regions)
        current_gt_coverage_area = _intersection_union_area(
            (*prior, crop), candidate.gt_regions
        )
        center_hits = {
            index
            for index, gt in enumerate(candidate.gt_regions)
            if _contains_gt_center(crop, gt)
        }
        incremental_area = current_gt_coverage_area - prior_gt_coverage_area
        call_metrics.append(
            {
                "call_index": call_index,
                "requested_bbox_2d": list(requested_box),
                "model_bbox_0_1000": (
                    list(model_box) if model_box is not None else None
                ),
                "source_bbox_xyxy": list(crop),
                "crop_area_ratio": crop_area / source_area,
                "spatial_precision": crop_gt_intersection / crop_area,
                "gt_area_recall": crop_gt_intersection / gt_area,
                "incremental_gt_area_recall": incremental_area / gt_area,
                "redundancy_ratio_with_prior_crops": (
                    _intersection_union_area((crop,), prior) / crop_area
                    if prior
                    else 0.0
                ),
                "nonincremental": incremental_area == 0,
                "max_pair_iou": max(
                    (_iou(crop, gt) for gt in candidate.gt_regions), default=0.0
                ),
                "gt_center_hit_count": len(center_hits),
                "new_gt_center_hit_count": len(center_hits - prior_center_hits),
            }
        )
        prior.append(crop)
        prior_gt_coverage_area = current_gt_coverage_area
        prior_center_hits.update(center_hits)
    attempted = _tool_attempted(payload, path=audit_path)
    return {
        "sample_id": candidate.sample_id,
        "trajectory_id": payload.get("trajectory_id"),
        "trajectory_sha256": payload.get("trajectory_sha256"),
        "tool_attempted": attempted,
        "successful_crop": observation_count > 0,
        "successful_crop_count": observation_count,
        "crop_coordinate_space": expected_coordinate_space,
        "requested_bboxes_2d": [list(box) for box in requested_boxes],
        "model_bboxes_0_1000": (
            [list(box) for box in model_boxes if box is not None]
            if expected_coordinate_space == QWEN3_CROP_COORDINATE_SPACE
            else None
        ),
        "source_bboxes_xyxy": [list(box) for box in source_boxes],
        "source_width": candidate.source_width,
        "source_height": candidate.source_height,
        "gt_regions_xyxy": [list(box) for box in candidate.gt_regions],
        "gt_union_area_pixels": gt_area,
        "first_call": first_call,
        "all_calls": all_calls,
        "calls": call_metrics,
        "mean_crop_area_ratio": (
            sum(float(call["crop_area_ratio"]) for call in call_metrics)
            / len(call_metrics)
            if call_metrics
            else 0.0
        ),
        "mean_spatial_precision": (
            sum(float(call["spatial_precision"]) for call in call_metrics)
            / len(call_metrics)
            if call_metrics
            else 0.0
        ),
        "redundant_call_count": sum(
            float(call["redundancy_ratio_with_prior_crops"]) > 0.0
            for call in call_metrics
        ),
        "nonincremental_call_count": sum(
            bool(call["nonincremental"]) for call in call_metrics
        ),
        # Backward-readable best-pair diagnostics. These are explicitly not
        # the primary grounding metrics because they can reward crop shotgun.
        "max_iou": all_calls["max_pair_iou"],
        "iou_at_least_0_1": all_calls["max_pair_iou"] >= 0.1,
        "iou_at_least_0_3": all_calls["max_pair_iou"] >= 0.3,
        "gt_center_hit": all_calls["gt_center_recall"] > 0.0,
    }


def _code_identity() -> dict[str, Any]:
    scorer_path = Path(__file__).resolve()
    coordinate_path = Path(map_qwen3_crop_bbox_to_source.__code__.co_filename).resolve()
    files = {
        "crop_grounding.py": file_sha256(scorer_path),
        "crop_coordinates.py": file_sha256(coordinate_path),
    }
    identity = {
        "files": files,
        "coordinate_space": QWEN3_CROP_COORDINATE_SPACE,
        "coordinate_conversion_version": QWEN3_CROP_CONVERSION_VERSION,
    }
    return {**identity, "sha256": _canonical_json_sha256(identity)}


_GROUNDING_METRICS = (
    "max_pair_iou",
    "gt_area_recall",
    "gt_region_iou_recall_at_0_1",
    "gt_region_iou_recall_at_0_3",
    "gt_center_recall",
)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sample_metric_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "mean_successful_crop_count": _mean(
            [float(sample["successful_crop_count"]) for sample in samples]
        ),
        "first_call": {
            f"mean_{metric}": _mean(
                [float(sample["first_call"][metric]) for sample in samples]
            )
            for metric in _GROUNDING_METRICS
        },
        "all_calls": {
            f"mean_{metric}": _mean(
                [float(sample["all_calls"][metric]) for sample in samples]
            )
            for metric in _GROUNDING_METRICS
        },
        "mean_crop_area_ratio": _mean(
            [float(sample["mean_crop_area_ratio"]) for sample in samples]
        ),
        "mean_spatial_precision": _mean(
            [float(sample["mean_spatial_precision"]) for sample in samples]
        ),
    }


def _benchmark_coordinate_space(
    evaluation_identity: Mapping[str, Any],
) -> str:
    protocol = evaluation_identity.get("protocol")
    if protocol is None:
        # Evaluation identities written before protocol binding all used the
        # project's historical normalized Qwen3 coordinate contract.
        return QWEN3_CROP_COORDINATE_SPACE
    if not isinstance(protocol, Mapping):
        raise ValueError("benchmark evaluation protocol identity is malformed")
    coordinate_space = protocol.get(
        "crop_coordinate_space", QWEN3_CROP_COORDINATE_SPACE
    )
    if coordinate_space not in {
        QWEN3_CROP_COORDINATE_SPACE,
        _ORIGINAL_IMAGE_PIXEL_COORDINATE_SPACE,
    }:
        raise ValueError("benchmark evaluation crop coordinate space is unsupported")
    return str(coordinate_space)


def _grounding_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(samples)
    if denominator <= 0:  # pragma: no cover - probe manifest is non-empty
        raise ValueError("grounding summary requires at least one sample")
    successful = tuple(sample for sample in samples if sample["successful_crop"])
    calls = tuple(call for sample in samples for call in sample["calls"])
    attempted_count = sum(bool(sample["tool_attempted"]) for sample in samples)
    successful_count = len(successful)
    nonincremental_count = sum(bool(call["nonincremental"]) for call in calls)
    redundant_count = sum(
        float(call["redundancy_ratio_with_prior_crops"]) > 0.0 for call in calls
    )
    return {
        "sample_count": denominator,
        "attempted_sample_count": attempted_count,
        "attempted_sample_rate": attempted_count / denominator,
        "successful_crop_sample_count": successful_count,
        "successful_crop_sample_rate": successful_count / denominator,
        "unconditional": _sample_metric_summary(samples),
        "conditional_on_successful_crop": _sample_metric_summary(successful),
        "call_level": {
            "successful_call_count": len(calls),
            "mean_crop_area_ratio": _mean(
                [float(call["crop_area_ratio"]) for call in calls]
            ),
            "mean_spatial_precision": _mean(
                [float(call["spatial_precision"]) for call in calls]
            ),
            "mean_incremental_gt_area_recall": _mean(
                [float(call["incremental_gt_area_recall"]) for call in calls]
            ),
            "mean_redundancy_ratio_with_prior_crops": _mean(
                [float(call["redundancy_ratio_with_prior_crops"]) for call in calls]
            ),
            "redundant_call_count": redundant_count,
            "redundant_call_rate": (redundant_count / len(calls) if calls else None),
            "nonincremental_call_count": nonincremental_count,
            "nonincremental_call_rate": (
                nonincremental_count / len(calls) if calls else None
            ),
        },
    }


def score_crop_grounding(
    *,
    candidate_manifest_path: str | Path,
    candidate_manifest_sha256: str,
    probe_manifest_path: str | Path,
    probe_manifest_sha256: str,
    trajectory_audit_root: str | Path,
    behavior_step: int,
    audit_mode: str = "training",
    evaluation_identity_path: str | Path | None = None,
    evaluation_identity_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Score a complete frozen probe at one exact behavior-policy step.

    Only rollout index zero contributes.  Missing, duplicate, unknown, malformed,
    or geometrically invalid records abort the entire report rather than changing
    the denominator.
    """

    candidate_manifest = Path(candidate_manifest_path).resolve()
    probe_manifest = Path(probe_manifest_path).resolve()
    audit_root = Path(trajectory_audit_root).resolve()
    if audit_mode not in _AUDIT_MODES:
        raise ValueError(f"audit_mode must be one of {sorted(_AUDIT_MODES)}")
    (
        candidates_path,
        candidates_jsonl_sha256,
        candidate_rows,
        candidate_manifest_file_sha256,
    ) = _candidate_jsonl_from_manifest(
        candidate_manifest,
        expected_sha256=candidate_manifest_sha256,
    )
    (
        ordered_ids,
        probe_manifest_file_sha256,
        task_manifest_path,
        task_manifest_sha256,
    ) = _load_probe_manifest(
        probe_manifest,
        expected_sha256=probe_manifest_sha256,
        candidate_manifest_file_sha256=candidate_manifest_file_sha256,
        candidates_jsonl_sha256=candidates_jsonl_sha256,
    )
    candidates = _load_candidates(
        candidates_path,
        expected_rows=candidate_rows,
        probe_sample_ids=ordered_ids,
    )
    tasks = load_benchmark_tasks(
        task_manifest_path,
        expected_task_count=len(ordered_ids),
        expected_single_image_count=len(ordered_ids),
        expected_sha256=task_manifest_sha256,
        verify_image_paths=False,
        verify_image_contents=False,
        require_explicit_sample_ids=True,
        require_image_identities=True,
    )
    if tuple(task.bound_sample_id for task in tasks) != ordered_ids:
        raise ValueError("probe task manifest order differs from ordered_sample_ids")
    _validate_probe_tasks(
        probe_sample_ids=ordered_ids,
        candidates=candidates,
        tasks=tasks,
    )
    if audit_mode == "training":
        if (
            evaluation_identity_path is not None
            or evaluation_identity_file_sha256 is not None
        ):
            raise ValueError(
                "evaluation identity arguments are benchmark-only; "
                "training audit identity comes from its step directory"
            )
        (
            audits,
            audit_input_dir,
            audit_tree_sha256,
            ignored_nonzero,
        ) = _load_rollout_zero_audits(
            audit_root,
            behavior_step=behavior_step,
            probe_sample_ids=ordered_ids,
        )
        checkpoint_identity: dict[str, Any] | None = None
        evaluation_identity_file_digest: str | None = None
        expected_coordinate_space = QWEN3_CROP_COORDINATE_SPACE
    else:
        if evaluation_identity_path is None or evaluation_identity_file_sha256 is None:
            raise ValueError(
                "benchmark audit requires an exact evaluation identity file"
            )
        evaluation_identity, evaluation_identity_file_digest = (
            load_policy_evaluation_identity(
                evaluation_identity_path,
                expected_file_sha256=evaluation_identity_file_sha256,
            )
        )
        task_binding = evaluation_identity.get("task_manifest")
        if not isinstance(task_binding, Mapping):
            raise ValueError("evaluation identity task manifest binding is malformed")
        if (
            task_binding.get("sha256") != task_manifest_sha256
            or task_binding.get("task_count") != len(ordered_ids)
            or task_binding.get("single_image_count") != len(ordered_ids)
        ):
            raise ValueError(
                "evaluation identity differs from the grounding probe tasks"
            )
        expected_coordinate_space = _benchmark_coordinate_space(evaluation_identity)
        (
            audits,
            audit_input_dir,
            audit_tree_sha256,
            ignored_nonzero,
            checkpoint_identity,
        ) = _load_benchmark_audits(
            audit_root,
            behavior_step=behavior_step,
            probe_sample_ids=ordered_ids,
            tasks=tasks,
            evaluation_identity=evaluation_identity,
        )
    samples = [
        _score_sample(
            candidates[sample_id],
            audit_path=audits[sample_id][0],
            payload=audits[sample_id][1],
            expected_coordinate_space=expected_coordinate_space,
        )
        for sample_id in ordered_ids
    ]
    summary = _grounding_summary(samples)
    code_identity = _code_identity()
    input_identity = {
        "candidate_manifest_file_sha256": candidate_manifest_file_sha256,
        "candidates_jsonl_sha256": candidates_jsonl_sha256,
        "probe_manifest_file_sha256": probe_manifest_file_sha256,
        "task_manifest_sha256": task_manifest_sha256,
        "trajectory_audit_tree_sha256": audit_tree_sha256,
        "behavior_step": behavior_step,
        "audit_mode": audit_mode,
        "crop_coordinate_space": expected_coordinate_space,
        "checkpoint_identity": checkpoint_identity,
        "evaluation_identity_file_sha256": evaluation_identity_file_digest,
    }
    report: dict[str, Any] = {
        "schema_version": CROP_GROUNDING_REPORT_SCHEMA,
        "behavior_step": behavior_step,
        "audit_mode": audit_mode,
        "coordinate_contract": {
            "space": expected_coordinate_space,
            "conversion_version": (
                QWEN3_CROP_CONVERSION_VERSION
                if expected_coordinate_space == QWEN3_CROP_COORDINATE_SPACE
                else _ORIGINAL_IMAGE_PIXEL_CONVERSION_VERSION
            ),
        },
        "metric_contract": {
            "primary_scopes": {
                "first_call": "first successful crop only; zero for no crop",
                "all_calls": "union/all successful crops; zero for no crop",
            },
            "gt_area_recall": (
                "area(intersection(union(crops), union(gt_regions))) / "
                "area(union(gt_regions))"
            ),
            "gt_region_iou_recall": (
                "fraction of every GT region matched by a crop at the threshold"
            ),
            "gt_center_recall": (
                "fraction of every GT-region center contained by a crop"
            ),
            "spatial_precision": (
                "area(intersection(crop, union(gt_regions))) / area(crop)"
            ),
            "incremental_gt_area_recall": (
                "new GT-union area covered by this call / GT-union area"
            ),
            "redundancy_ratio_with_prior_crops": (
                "crop area already covered by prior crops / crop area"
            ),
            "best_pair_iou_status": "secondary_diagnostic_not_primary",
        },
        "inputs": {
            "candidate_manifest_path": str(candidate_manifest),
            "candidate_manifest_file_sha256": candidate_manifest_file_sha256,
            "candidates_jsonl_path": str(candidates_path),
            "candidates_jsonl_sha256": candidates_jsonl_sha256,
            "probe_manifest_path": str(probe_manifest),
            "probe_manifest_file_sha256": probe_manifest_file_sha256,
            "task_manifest_path": str(task_manifest_path),
            "task_manifest_sha256": task_manifest_sha256,
            "trajectory_audit_input_dir": str(audit_input_dir),
            "trajectory_audit_tree_sha256": audit_tree_sha256,
            "ignored_nonzero_rollout_records": ignored_nonzero,
            "evaluation_identity_path": (
                str(Path(evaluation_identity_path).resolve())
                if evaluation_identity_path is not None
                else None
            ),
            "evaluation_identity_file_sha256": evaluation_identity_file_digest,
            "identity_sha256": _canonical_json_sha256(input_identity),
        },
        "checkpoint_identity": checkpoint_identity,
        "code_identity": code_identity,
        "summary": summary,
        "counterfactual": {
            "observation_masked_replay": {
                "supported": False,
                "status": "unsupported",
                "reason_code": "observation_masked_replay_artifact_not_provided",
                "metric_values": None,
                "required_artifact_schema": (
                    "crop-grounding-observation-masked-replay-v1"
                ),
            }
        },
        "samples": samples,
    }
    report["report_identity_sha256"] = _canonical_json_sha256(report)
    return report


__all__ = [
    "CROP_GROUNDING_PROBE_SCHEMA",
    "CROP_GROUNDING_REPORT_SCHEMA",
    "POLICY_TRAJECTORY_AUDIT_SCHEMA",
    "GroundingCandidate",
    "file_sha256",
    "score_crop_grounding",
]
