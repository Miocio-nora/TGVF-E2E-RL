from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.evaluation.policy_coredev import (
    POLICY_BENCHMARK_SCHEMA,
    POLICY_EVALUATION_IDENTITY_SCHEMA,
)
from tgvf_rl.trajectories.schema import TrajectoryIdentity

from tgvf_rl.evaluation.crop_grounding import (
    CROP_GROUNDING_PROBE_SCHEMA,
    CROP_GROUNDING_REPORT_SCHEMA,
    file_sha256,
    score_crop_grounding,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _candidate(sample_id: str) -> dict[str, object]:
    return {
        "schema_version": "tgvf.policy-selection.candidate.v1",
        "sample_id": sample_id,
        "source": "vstar",
        "question": f"question-{sample_id}",
        "ground_truth": "answer",
        "image": {
            "path": f"/immutable/{sample_id}.png",
            "sha256": "1" * 64,
            "width": 200,
            "height": 100,
        },
        "gt_regions": (
            [[20, 10, 40, 30], [160, 70, 180, 90]]
            if sample_id == "sample-multi"
            else [[30, 20, 50, 40]]
        ),
    }


def _audit(
    sample_id: str,
    *,
    step: int = 8,
    rollout_index: int = 0,
    calls: list[dict[str, object]] | None = None,
    errors: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    calls = [] if calls is None else calls
    errors = [] if errors is None else errors
    return {
        "schema_version": "policy-trajectory-audit-v1",
        "selection_reasons": (
            ["representative_rollout_zero"]
            if rollout_index == 0
            else ["correct_answer"]
        ),
        "optimizer_step": step,
        "trajectory_id": f"run/{sample_id}/{rollout_index}/group",
        "trajectory_sha256": hashlib.sha256(
            f"{sample_id}:{rollout_index}".encode()
        ).hexdigest(),
        "sample_id": sample_id,
        "group_uid": "group",
        "rollout_index": rollout_index,
        "tool_calls": calls,
        "tool_errors": errors,
        "successful_observation_count": len(calls),
    }


def _seal_result(payload: dict[str, object]) -> dict[str, object]:
    payload.pop("result_identity_sha256", None)
    payload["result_identity_sha256"] = _canonical_sha256(payload)
    return payload


def _benchmark_audit(
    sample_id: str,
    *,
    evaluation_identity: dict[str, object],
    ordinal: int = 0,
    step: int = 8,
    calls: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    policy = evaluation_identity["policy_snapshot"]
    task = evaluation_identity["task_manifest"]
    execution = evaluation_identity["execution"]
    assert isinstance(policy, dict)
    assert isinstance(task, dict)
    assert isinstance(execution, dict)
    world_size = execution["world_size"]
    assert isinstance(world_size, int)
    rank = ordinal % world_size
    payload = _audit(sample_id, step=step, calls=calls)
    snapshot_backend = policy.get("snapshot_backend", "lora")
    if snapshot_backend == "full_model":
        snapshot_audit = {
            "policy_snapshot_backend": "full_model",
            "policy_full_snapshot_identity_sha256": policy["snapshot_identity_sha256"],
            "policy_checkpoint_sha256": policy["checkpoint_sha256"],
            "policy_source_tree_sha256": policy["source_tree_sha256"],
            "policy_materialization_identity_sha256": policy[
                "materialization_identity_sha256"
            ],
            "policy_materialized_model_tree_sha256": policy[
                "materialized_model_tree_sha256"
            ],
        }
    else:
        snapshot_audit = {
            "policy_snapshot_backend": "lora",
            "policy_pointer_file_sha256": policy["pointer_file_sha256"],
            "policy_manifest_file_sha256": policy["manifest_file_sha256"],
            "policy_tensor_file_sha256": policy["tensor_file_sha256"],
        }
    payload.update(
        {
            "schema_version": "tgvf-policy-coredev-trajectory-audit-v1",
            "evaluation_identity_sha256": evaluation_identity["identity_sha256"],
            "policy_run_identity_sha256": policy["run_identity_sha256"],
            **snapshot_audit,
            "policy_config_identity_sha256": evaluation_identity[
                "policy_run_config_identity_sha256"
            ],
            "task_manifest_sha256": task["sha256"],
            "model_identity": evaluation_identity["model_identity"],
            "rank": rank,
            "world_size": world_size,
            "evaluation_id": evaluation_identity["evaluation_id"],
            "policy_run_id": policy["run_id"],
            "policy_weights_sha256": policy["weights_sha256"],
            "group_uid": f"benchmark:{ordinal}",
            "ordinal": ordinal,
            "dataset": "VStarGroundingProbe",
            "row_number": ordinal,
            "index": sample_id,
            "question": f"question-{sample_id}",
            "image_paths": [f"/immutable/{sample_id}.png"],
            "image_sha256s": ["1" * 64],
            "image_dimensions": [[200, 100]],
            "trajectory_id": TrajectoryIdentity(
                str(evaluation_identity["evaluation_id"]),
                sample_id,
                0,
                f"benchmark:{ordinal}",
            ).canonical_id,
        }
    )
    return _seal_result(payload)


def _fixture(
    tmp_path: Path,
    *,
    sample_ids: tuple[str, ...] = ("sample-a",),
) -> dict[str, object]:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    candidates_path = candidate_dir / "candidates.jsonl"
    candidates_path.write_text(
        "".join(
            json.dumps(_candidate(sample_id), sort_keys=True) + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    candidates_sha256 = file_sha256(candidates_path)
    candidate_manifest = candidate_dir / "manifest.json"
    _write_json(
        candidate_manifest,
        {
            "schema_version": "tgvf.policy-selection.source-manifest.v1",
            "source": "vstar",
            "candidate_rows": len(sample_ids),
            "candidates": {
                "path": "candidates.jsonl",
                "sha256": candidates_sha256,
            },
        },
    )
    candidate_manifest_sha256 = file_sha256(candidate_manifest)
    probe_manifest = tmp_path / "probe.json"
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(
            json.dumps(
                {
                    "ordinal": ordinal,
                    "dataset": "VStarGroundingProbe",
                    "row_number": ordinal,
                    "index": sample_id,
                    "sample_id": sample_id,
                    "question": f"question-{sample_id}",
                    "image_paths": [f"/immutable/{sample_id}.png"],
                    "image_sha256s": ["1" * 64],
                    "image_dimensions": [[200, 100]],
                },
                sort_keys=True,
            )
            + "\n"
            for ordinal, sample_id in enumerate(sample_ids)
        ),
        encoding="utf-8",
    )
    tasks_sha256 = file_sha256(tasks_path)
    _write_json(
        probe_manifest,
        {
            "schema_version": CROP_GROUNDING_PROBE_SCHEMA,
            "sample_count": len(sample_ids),
            "ordered_sample_ids": list(sample_ids),
            "candidate_manifest_file_sha256": candidate_manifest_sha256,
            "candidates_jsonl_sha256": candidates_sha256,
            "task_manifest": {
                "path": tasks_path.name,
                "sha256": tasks_sha256,
                "row_count": len(sample_ids),
            },
        },
    )
    return {
        "candidate_manifest_path": candidate_manifest,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "probe_manifest_path": probe_manifest,
        "probe_manifest_sha256": file_sha256(probe_manifest),
        "trajectory_audit_root": tmp_path / "trajectory_audit",
        "behavior_step": 8,
    }


def _write_audit(root: Path, name: str, payload: object, *, step: int = 8) -> None:
    _write_json(root / f"step-{step:08d}" / f"{name}.json", payload)


def _write_rank(root: Path, rank: int, payloads: list[object]) -> None:
    path = root / "inference" / f"rank-{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def _benchmark_context(
    inputs: dict[str, object],
    *,
    snapshot_backend: str = "lora",
) -> tuple[dict[str, object], dict[str, object]]:
    probe_path = inputs["probe_manifest_path"]
    audit_root = inputs["trajectory_audit_root"]
    assert isinstance(probe_path, Path)
    assert isinstance(audit_root, Path)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    task_binding = probe["task_manifest"]
    sample_count = probe["sample_count"]
    world_size = min(2, sample_count)
    if snapshot_backend == "full_model":
        policy_snapshot = {
            "snapshot_backend": "full_model",
            "run_id": "PRL-13",
            "run_identity_sha256": "5" * 64,
            "optimizer_step": 8,
            "weights_sha256": "a" * 64,
            "snapshot_identity_sha256": "6" * 64,
            "checkpoint_sha256": "7" * 64,
            "source_tree_sha256": "8" * 64,
            "materialization_identity_sha256": "9" * 64,
            "materialized_model_tree_sha256": "b" * 64,
            "lora_request": None,
        }
    elif snapshot_backend == "lora":
        policy_snapshot = {
            "snapshot_backend": "lora_adapter",
            "run_id": "PRL-11",
            "run_identity_sha256": "5" * 64,
            "optimizer_step": 8,
            "weights_sha256": "a" * 64,
            "pointer_file_sha256": "6" * 64,
            "manifest_file_sha256": "7" * 64,
            "tensor_file_sha256": "8" * 64,
            "request_sha256": "9" * 64,
        }
    else:
        raise ValueError("unsupported fixture snapshot backend")
    content: dict[str, object] = {
        "schema_version": POLICY_EVALUATION_IDENTITY_SCHEMA,
        "evaluation_id": "GROUNDING-EVAL-STEP8",
        "evaluation_schema_version": POLICY_BENCHMARK_SCHEMA,
        "policy_config_path": "/immutable/policy.toml",
        "policy_config_file_sha256": "2" * 64,
        "policy_run_config_identity_sha256": "3" * 64,
        "model_identity": {
            "family": "qwen3_vl",
            "model_name": "fixture",
            "revision_or_path": "fixture",
            "tokenizer_length": 1,
            "chat_template_sha256": "4" * 64,
        },
        "policy_snapshot": policy_snapshot,
        "task_manifest": {
            "path": str(probe_path.parent / task_binding["path"]),
            "sha256": task_binding["sha256"],
            "task_count": sample_count,
            "single_image_count": sample_count,
        },
        "execution": {
            "world_size": world_size,
            "gpu_ids": list(range(world_size)),
            "max_model_len": 32768,
            "max_num_batched_tokens": 32768,
            "enable_chunked_prefill": False,
            "inference_concurrency_per_gpu": 8,
        },
    }
    identity = {**content, "identity_sha256": _canonical_sha256(content)}
    identity_path = audit_root / "runtime/evaluation-identity.json"
    _write_json(identity_path, identity)
    return (
        {
            **inputs,
            "audit_mode": "benchmark",
            "evaluation_identity_path": identity_path,
            "evaluation_identity_file_sha256": file_sha256(identity_path),
        },
        identity,
    )


def test_no_call_is_zero_on_the_full_frozen_denominator(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, sample_ids=("sample-a", "sample-b"))
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    _write_audit(root, "a", _audit("sample-a"))
    _write_audit(root, "b", _audit("sample-b"))

    report = score_crop_grounding(**inputs)

    assert report["schema_version"] == CROP_GROUNDING_REPORT_SCHEMA
    summary = report["summary"]
    assert summary["sample_count"] == 2
    assert summary["attempted_sample_rate"] == 0.0
    assert summary["successful_crop_sample_rate"] == 0.0
    assert summary["unconditional"]["first_call"]["mean_max_pair_iou"] == 0.0
    assert summary["unconditional"]["all_calls"]["mean_gt_area_recall"] == 0.0
    assert summary["conditional_on_successful_crop"]["sample_count"] == 0
    assert (
        summary["conditional_on_successful_crop"]["all_calls"]["mean_gt_area_recall"]
        is None
    )
    assert summary["call_level"]["successful_call_count"] == 0
    assert summary["call_level"]["mean_spatial_precision"] is None
    assert [sample["max_iou"] for sample in report["samples"]] == [0.0, 0.0]
    assert report["counterfactual"]["observation_masked_replay"] == {
        "supported": False,
        "status": "unsupported",
        "reason_code": "observation_masked_replay_artifact_not_provided",
        "metric_values": None,
        "required_artifact_schema": "crop-grounding-observation-masked-replay-v1",
    }
    assert len(report["inputs"]["identity_sha256"]) == 64
    assert len(report["code_identity"]["sha256"]) == 64


def test_qwen3_coordinates_use_runtime_floor_mapping_and_score_iou(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    call = {
        "call_index": 0,
        "assistant_turn_index": 0,
        "function_name": "image_zoom_in_tool",
        "raw_call_text": "fixture",
        "bbox_2d": [100, 100, 500, 500],
        "source_bbox_2d": [20, 10, 100, 50],
        "effective_bbox_2d": [20, 10, 100, 50],
        "coordinate_space": "qwen3_relative_0_1000",
        "conversion_version": "qwen3-relative-1000-floor-v1",
        "coordinate_reference_size": [1000, 1000],
        "source_size": [200, 100],
        "crop_width": 80,
        "crop_height": 40,
        "crop_rgb_sha256": "a" * 64,
        "crop_source": "immutable_original_image",
        "label": None,
    }
    _write_audit(root, "rollout-zero", _audit("sample-a", calls=[call]))
    _write_audit(
        root,
        "rollout-one",
        _audit("sample-a", rollout_index=1, calls=[call]),
    )

    report = score_crop_grounding(**inputs)

    sample = report["samples"][0]
    assert sample["model_bboxes_0_1000"] == [[100, 100, 500, 500]]
    assert sample["source_bboxes_xyxy"] == [[20, 10, 100, 50]]
    assert sample["max_iou"] == pytest.approx(0.125)
    assert sample["iou_at_least_0_1"] is True
    assert sample["iou_at_least_0_3"] is False
    assert sample["gt_center_hit"] is True
    assert sample["first_call"]["gt_area_recall"] == 1.0
    assert sample["all_calls"]["gt_region_iou_recall_at_0_1"] == 1.0
    assert sample["calls"][0]["crop_area_ratio"] == pytest.approx(0.16)
    assert sample["calls"][0]["spatial_precision"] == pytest.approx(0.125)
    assert report["summary"]["attempted_sample_rate"] == 1.0
    assert report["summary"]["successful_crop_sample_rate"] == 1.0
    assert report["inputs"]["ignored_nonzero_rollout_records"] == 1


def test_qwen3_audit_rejects_double_mapped_source_box(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    call = {
        "call_index": 0,
        "assistant_turn_index": 0,
        "function_name": "image_zoom_in_tool",
        "raw_call_text": "fixture",
        "bbox_2d": [100, 100, 500, 500],
        # Correct source mapping is [20,10,100,50]. This is what incorrectly
        # mapping that source box a second time would produce.
        "source_bbox_2d": [4, 1, 20, 5],
        "effective_bbox_2d": [4, 1, 20, 5],
        "coordinate_space": "qwen3_relative_0_1000",
        "conversion_version": "qwen3-relative-1000-floor-v1",
        "coordinate_reference_size": [1000, 1000],
        "source_size": [200, 100],
        "crop_width": 16,
        "crop_height": 4,
        "crop_rgb_sha256": "a" * 64,
        "crop_source": "immutable_original_image",
        "label": None,
    }
    _write_audit(root, "double-map", _audit("sample-a", calls=[call]))

    with pytest.raises(ValueError, match="differs from family mapping"):
        score_crop_grounding(**inputs)


def test_first_and_all_call_coverage_expose_redundant_crop_shotgun(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path, sample_ids=("sample-multi",))
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    boxes = (
        [50, 50, 250, 350],
        [50, 50, 250, 350],
        [750, 650, 950, 950],
    )
    calls = [
        {
            "call_index": index,
            "assistant_turn_index": index,
            "function_name": "image_zoom_in_tool",
            "raw_call_text": "fixture",
            "bbox_2d": box,
            "label": None,
        }
        for index, box in enumerate(boxes)
    ]
    _write_audit(root, "multi", _audit("sample-multi", calls=calls))

    report = score_crop_grounding(**inputs)

    sample = report["samples"][0]
    assert sample["first_call"]["gt_area_recall"] == pytest.approx(0.5)
    assert sample["first_call"]["gt_center_recall"] == pytest.approx(0.5)
    assert sample["all_calls"]["gt_area_recall"] == pytest.approx(1.0)
    assert sample["all_calls"]["gt_center_recall"] == pytest.approx(1.0)
    assert sample["calls"][1]["redundancy_ratio_with_prior_crops"] == 1.0
    assert sample["calls"][1]["incremental_gt_area_recall"] == 0.0
    assert sample["calls"][1]["nonincremental"] is True
    assert sample["redundant_call_count"] == 1
    assert sample["nonincremental_call_count"] == 1
    assert report["summary"]["call_level"]["nonincremental_call_rate"] == (
        pytest.approx(1 / 3)
    )


def test_benchmark_rank_jsonl_scores_one_bound_record_per_probe_sample(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path, sample_ids=("sample-a", "sample-b"))
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    score_inputs, evaluation_identity = _benchmark_context(inputs)
    call = {
        "call_index": 0,
        "assistant_turn_index": 0,
        "function_name": "image_zoom_in_tool",
        "raw_call_text": "fixture",
        "bbox_2d": [100, 100, 500, 500],
        "label": None,
    }
    _write_rank(
        root,
        0,
        [
            _benchmark_audit(
                "sample-a",
                evaluation_identity=evaluation_identity,
                ordinal=0,
                calls=[call],
            )
        ],
    )
    _write_rank(
        root,
        1,
        [
            _benchmark_audit(
                "sample-b", evaluation_identity=evaluation_identity, ordinal=1
            )
        ],
    )

    report = score_crop_grounding(**score_inputs)

    assert report["audit_mode"] == "benchmark"
    assert report["checkpoint_identity"]["evaluation_id"] == ("GROUNDING-EVAL-STEP8")
    assert report["checkpoint_identity"]["policy_run_identity_sha256"] == "5" * 64
    assert report["checkpoint_identity"]["policy_pointer_file_sha256"] == "6" * 64
    assert report["checkpoint_identity"]["policy_tensor_file_sha256"] == "8" * 64
    assert report["summary"]["sample_count"] == 2
    assert report["summary"]["attempted_sample_rate"] == 0.5
    assert report["summary"]["successful_crop_sample_rate"] == 0.5
    assert report["summary"]["unconditional"]["all_calls"][
        "mean_max_pair_iou"
    ] == pytest.approx(0.0625)
    assert report["inputs"]["ignored_nonzero_rollout_records"] == 0


def test_benchmark_full_model_snapshot_scores_and_reports_checkpoint_identity(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    score_inputs, evaluation_identity = _benchmark_context(
        inputs, snapshot_backend="full_model"
    )
    _write_rank(
        root,
        0,
        [_benchmark_audit("sample-a", evaluation_identity=evaluation_identity)],
    )

    report = score_crop_grounding(**score_inputs)

    checkpoint = report["checkpoint_identity"]
    assert checkpoint["policy_snapshot_backend"] == "full_model"
    assert checkpoint["policy_run_id"] == "PRL-13"
    assert checkpoint["policy_full_snapshot_identity_sha256"] == "6" * 64
    assert checkpoint["policy_checkpoint_sha256"] == "7" * 64
    assert checkpoint["policy_source_tree_sha256"] == "8" * 64
    assert checkpoint["policy_materialization_identity_sha256"] == "9" * 64
    assert checkpoint["policy_materialized_model_tree_sha256"] == "b" * 64
    assert "policy_pointer_file_sha256" not in checkpoint
    assert report["summary"]["sample_count"] == 1


def test_benchmark_scores_identity_bound_official_source_pixel_crops(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    score_inputs, evaluation_identity = _benchmark_context(inputs)
    evaluation_identity.pop("identity_sha256")
    evaluation_identity["protocol"] = {
        "profile": "deepeyes_official_visible_native_crop_v1",
        "crop_coordinate_space": "original_image_pixels_xyxy_v1",
    }
    evaluation_identity["identity_sha256"] = _canonical_sha256(evaluation_identity)
    identity_path = score_inputs["evaluation_identity_path"]
    assert isinstance(identity_path, Path)
    _write_json(identity_path, evaluation_identity)
    score_inputs["evaluation_identity_file_sha256"] = file_sha256(identity_path)
    call = {
        "call_index": 0,
        "assistant_turn_index": 0,
        "function_name": "image_zoom_in_tool",
        "raw_call_text": "fixture",
        "bbox_2d": [-1.2, 5.1, 50.2, 42.1],
        "source_bbox_2d": [0, 5, 51, 43],
        "coordinate_space": "original_image_pixels_xyxy_v1",
        "label": None,
        "crop_width": 51,
        "crop_height": 38,
        "crop_rgb_sha256": "a" * 64,
        "crop_source": "immutable_original_image",
    }
    _write_rank(
        root,
        0,
        [
            _benchmark_audit(
                "sample-a", evaluation_identity=evaluation_identity, calls=[call]
            )
        ],
    )

    report = score_crop_grounding(**score_inputs)

    assert report["coordinate_contract"] == {
        "space": "original_image_pixels_xyxy_v1",
        "conversion_version": "deepeyes-source-pixel-clipping-v1",
    }
    sample = report["samples"][0]
    assert sample["requested_bboxes_2d"] == [[-1.2, 5.1, 50.2, 42.1]]
    assert sample["model_bboxes_0_1000"] is None
    assert sample["source_bboxes_xyxy"] == [[0, 5, 51, 43]]
    assert sample["calls"][0]["model_bbox_0_1000"] is None


def test_official_source_pixel_crop_must_match_its_clipped_request(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    score_inputs, evaluation_identity = _benchmark_context(inputs)
    evaluation_identity.pop("identity_sha256")
    evaluation_identity["protocol"] = {
        "crop_coordinate_space": "original_image_pixels_xyxy_v1"
    }
    evaluation_identity["identity_sha256"] = _canonical_sha256(evaluation_identity)
    identity_path = score_inputs["evaluation_identity_path"]
    assert isinstance(identity_path, Path)
    _write_json(identity_path, evaluation_identity)
    score_inputs["evaluation_identity_file_sha256"] = file_sha256(identity_path)
    call = {
        "call_index": 0,
        "assistant_turn_index": 0,
        "function_name": "image_zoom_in_tool",
        "raw_call_text": "fixture",
        "bbox_2d": [10, 10, 80, 80],
        "source_bbox_2d": [11, 10, 80, 80],
        "coordinate_space": "original_image_pixels_xyxy_v1",
        "crop_width": 69,
        "crop_height": 70,
        "crop_rgb_sha256": "a" * 64,
        "crop_source": "immutable_original_image",
    }
    _write_rank(
        root,
        0,
        [
            _benchmark_audit(
                "sample-a", evaluation_identity=evaluation_identity, calls=[call]
            )
        ],
    )

    with pytest.raises(ValueError, match="differs from clipped request"):
        score_crop_grounding(**score_inputs)


def test_benchmark_evaluation_identity_file_hash_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    score_inputs, evaluation_identity = _benchmark_context(inputs)
    _write_rank(
        root,
        0,
        [_benchmark_audit("sample-a", evaluation_identity=evaluation_identity)],
    )
    score_inputs["evaluation_identity_file_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="evaluation identity file SHA256 differs"):
        score_crop_grounding(**score_inputs)


@pytest.mark.parametrize(
    ("field", "wrong", "message"),
    [
        ("evaluation_id", "OTHER-EVAL", "evaluation_id differs"),
        ("optimizer_step", 9, "optimizer_step differs"),
        ("policy_run_id", "OTHER-RUN", "policy_run_id differs"),
        ("policy_weights_sha256", "b" * 64, "policy_weights_sha256 differs"),
        (
            "policy_run_identity_sha256",
            "b" * 64,
            "policy_run_identity_sha256 differs",
        ),
        (
            "policy_pointer_file_sha256",
            "b" * 64,
            "policy_pointer_file_sha256 differs",
        ),
        (
            "policy_tensor_file_sha256",
            "b" * 64,
            "policy_tensor_file_sha256 differs",
        ),
        (
            "policy_manifest_file_sha256",
            "b" * 64,
            "policy_manifest_file_sha256 differs",
        ),
        (
            "evaluation_identity_sha256",
            "b" * 64,
            "evaluation_identity_sha256 differs",
        ),
    ],
)
def test_benchmark_checkpoint_identity_mismatch_fails_closed(
    tmp_path: Path,
    field: str,
    wrong: object,
    message: str,
) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    score_inputs, evaluation_identity = _benchmark_context(inputs)
    payload = _benchmark_audit("sample-a", evaluation_identity=evaluation_identity)
    payload[field] = wrong
    _seal_result(payload)
    _write_rank(root, 0, [payload])

    with pytest.raises(ValueError, match=message):
        score_crop_grounding(**score_inputs)


@pytest.mark.parametrize(
    ("field", "wrong", "message"),
    [
        ("question", "different question", "result question differs"),
        ("image_paths", ["/immutable/other.png"], "result image_paths differs"),
        ("image_sha256s", ["b" * 64], "result image_sha256s differs"),
        ("image_dimensions", [[201, 100]], "result image_dimensions differs"),
        ("rollout_index", 1, "rollout_index differs"),
        ("group_uid", "benchmark:9", "group_uid differs"),
        ("sample_id", "unknown", "sample_id differs"),
    ],
)
def test_benchmark_sample_binding_mismatch_fails_closed(
    tmp_path: Path,
    field: str,
    wrong: object,
    message: str,
) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    score_inputs, evaluation_identity = _benchmark_context(inputs)
    payload = _benchmark_audit("sample-a", evaluation_identity=evaluation_identity)
    payload[field] = wrong
    _seal_result(payload)
    _write_rank(root, 0, [payload])

    with pytest.raises(ValueError, match=message):
        score_crop_grounding(**score_inputs)


def test_benchmark_duplicate_and_missing_records_fail_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, sample_ids=("sample-a", "sample-b"))
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    score_inputs, evaluation_identity = _benchmark_context(inputs)
    duplicate = _benchmark_audit(
        "sample-a", evaluation_identity=evaluation_identity, ordinal=0
    )
    _write_rank(root, 0, [duplicate, duplicate])
    _write_rank(root, 1, [])

    with pytest.raises(ValueError, match="duplicate/invalid policy benchmark ordinal"):
        score_crop_grounding(**score_inputs)

    inference = root / "inference"
    for path in inference.glob("rank-*.jsonl"):
        path.unlink()
    _write_rank(
        root,
        0,
        [
            _benchmark_audit(
                "sample-a", evaluation_identity=evaluation_identity, ordinal=0
            )
        ],
    )
    _write_rank(root, 1, [])
    with pytest.raises(ValueError, match="policy benchmark results are incomplete"):
        score_crop_grounding(**score_inputs)


def test_tool_error_counts_as_attempt_but_not_success(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    _write_audit(
        root,
        "error",
        _audit(
            "sample-a",
            errors=[
                {
                    "attempt_index": 0,
                    "assistant_turn_index": 0,
                    "function_name": "image_zoom_in_tool",
                    "code": "tool_execution_failed",
                    "payload_json": "{}",
                    "recoverable": True,
                }
            ],
        ),
    )

    report = score_crop_grounding(**inputs)

    assert report["summary"]["attempted_sample_rate"] == 1.0
    assert report["summary"]["successful_crop_sample_rate"] == 0.0
    assert report["summary"]["unconditional"]["all_calls"]["mean_max_pair_iou"] == 0.0


def test_duplicate_rollout_zero_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    _write_audit(root, "first", _audit("sample-a"))
    _write_audit(root, "second", _audit("sample-a"))

    with pytest.raises(ValueError, match="duplicate rollout_index=0"):
        score_crop_grounding(**inputs)


def test_missing_rollout_zero_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, sample_ids=("sample-a", "sample-b"))
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    _write_audit(root, "only-a", _audit("sample-a"))

    with pytest.raises(ValueError, match="missing rollout_index=0"):
        score_crop_grounding(**inputs)


def test_unknown_sample_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    _write_audit(root, "unknown", _audit("unknown"))

    with pytest.raises(ValueError, match="unknown sample_id unknown"):
        score_crop_grounding(**inputs)


@pytest.mark.parametrize("which", ["candidate", "probe"])
def test_input_file_hash_mismatch_fails_closed(tmp_path: Path, which: str) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    _write_audit(root, "a", _audit("sample-a"))
    inputs[f"{which}_manifest_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        score_crop_grounding(**inputs)


def test_probe_candidate_binding_mismatch_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    probe = inputs["probe_manifest_path"]
    assert isinstance(probe, Path)
    payload = json.loads(probe.read_text(encoding="utf-8"))
    payload["candidates_jsonl_sha256"] = "2" * 64
    _write_json(probe, payload)
    inputs["probe_manifest_sha256"] = file_sha256(probe)

    with pytest.raises(ValueError, match="different candidates JSONL"):
        score_crop_grounding(**inputs)


def test_out_of_range_crop_coordinates_fail_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    call = {
        "call_index": 0,
        "assistant_turn_index": 0,
        "function_name": "image_zoom_in_tool",
        "raw_call_text": "fixture",
        "bbox_2d": [900, 10, 1001, 100],
        "label": None,
    }
    _write_audit(root, "a", _audit("sample-a", calls=[call]))

    with pytest.raises(ValueError, match="invalid Qwen3 crop coordinates"):
        score_crop_grounding(**inputs)


def test_coredev_audit_without_rollout_index_is_explicitly_unsupported(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    root = inputs["trajectory_audit_root"]
    assert isinstance(root, Path)
    payload = _audit("sample-a")
    payload["schema_version"] = "tgvf-policy-coredev-trajectory-audit-v1"
    payload.pop("rollout_index")
    _write_audit(root, "coredev", payload)

    with pytest.raises(ValueError, match="training-writer-compatible audit"):
        score_crop_grounding(**inputs)
