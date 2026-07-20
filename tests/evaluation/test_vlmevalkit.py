from pathlib import Path

import pytest

from tgvf_rl.evaluation.vlmevalkit import (
    COREDEV_2511,
    COREDEV_2511_MANIFEST_SHA256,
    SHARED_BENCHMARK_ROOT,
    VLMEVALKIT_REVIEW_COMMIT,
    CoreDev2511Spec,
    CoreDevSliceSpec,
    TGVFPolicyEvaluationResult,
    VLMEvalKitLaunchPlan,
)


def test_coredev_2511_identity_and_official_scorer_slices_are_fixed() -> None:
    assert COREDEV_2511.manifest_sha256 == COREDEV_2511_MANIFEST_SHA256
    assert COREDEV_2511.seed == 20260625
    assert COREDEV_2511.sample_count == 2511
    assert tuple(item.vlmeval_dataset for item in COREDEV_2511.slices) == (
        "VStarBench",
        "HRBench4K",
        "BLINK",
        "OCRBench_v2",
        "MMMU_Pro_10c",
        "MathVista_MINI",
        "MathVerse_MINI",
    )


def test_coredev_2511_rejects_a_mixed_or_incomplete_suite() -> None:
    with pytest.raises(ValueError, match="exactly seven"):
        CoreDev2511Spec(
            manifest_sha256=COREDEV_2511_MANIFEST_SHA256,
            seed=20260625,
            slices=COREDEV_2511.slices[:1],
        )
    changed = list(COREDEV_2511.slices)
    changed[-1] = CoreDevSliceSpec(
        "mathverse", "mathverse_testmini_3940", "MathVerse_MINI", 499
    )
    with pytest.raises(ValueError, match="sample count drifted"):
        CoreDev2511Spec(
            manifest_sha256=COREDEV_2511_MANIFEST_SHA256,
            seed=20260625,
            slices=tuple(changed),
        )


def test_launch_plan_uses_pinned_external_checkout_and_shared_data() -> None:
    plan = VLMEvalKitLaunchPlan(
        checkout=Path("/opt/VLMEvalKit"),
        config_path=Path("/workspace/configs/coredev2511.json"),
        work_dir=Path("/workspace/results/run-1"),
    )
    assert plan.expected_commit == VLMEVALKIT_REVIEW_COMMIT
    assert plan.argv == (
        "python",
        "/opt/VLMEvalKit/run.py",
        "--config",
        "/workspace/configs/coredev2511.json",
        "--work-dir",
        "/workspace/results/run-1",
        "--mode",
        "all",
    )
    assert plan.environment == {
        "LMUData": str(SHARED_BENCHMARK_ROOT),
        "PRED_FORMAT": "tsv",
        "EVAL_FORMAT": "json",
    }


def test_policy_result_separates_prediction_from_extra_records() -> None:
    result = TGVFPolicyEvaluationResult(
        final_answer="42",
        extra_records={
            "trajectory_id": "trajectory-1",
            "ordered_tool_names": ["image_zoom_in_tool", "tgvf_focus_tool"],
        },
    )
    assert result.as_generate_inner_result() == (
        0,
        "42",
        {
            "trajectory_id": "trajectory-1",
            "ordered_tool_names": ["image_zoom_in_tool", "tgvf_focus_tool"],
        },
    )


def test_policy_result_rejects_non_json_extra_records() -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        TGVFPolicyEvaluationResult(
            final_answer="answer",
            extra_records={"bad": float("nan")},
        )
