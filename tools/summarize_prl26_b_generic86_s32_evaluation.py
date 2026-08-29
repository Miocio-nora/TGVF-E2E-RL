#!/usr/bin/env python3
"""Publish PRL-26-B's owner-generic86 seven-subset and tool-use result."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.coredev_results import (  # noqa: E402
    extract_coredev_macro_star,
    write_json_atomic,
)


EVALUATION_ID = (
    "PRL26-B-S32-OWNER-GENERIC86-TRAINING-RUN-COREDEV2511-PIXEL512-V1"
)
TRAINING_RUN_ID = (
    "PRL-26-B-TRAIN512-S32-PARITY-CROP-QWEN3-INSTRUCT-"
    "BS16-N16-TEACHER25-WS8"
)
PAIRED_SEED_NAMESPACE = (
    "coredev2511/prl26-b/owner-generic86/training-run/"
    "train512-eval512/s32/temp1/seed42/v1"
)
VARIANT = "prl26-b-e756546b-generic-crop-continuation-v1"
ENVIRONMENT_SHA256 = (
    "72a2caecb47a2b775a4497e5846c244061d9455fbb4b9690d3501cbc2521e187"
)
TRAINING_LAUNCH_COMMIT = "40f1728a69e0a3f868117776c80c45ad6de70b8c"
PIXEL512 = 262_144
STEP = 32
OFFICIAL_ROWS = {
    "VStarBench": 191,
    "HRBench4K": 200,
    "BLINK": 420,
    "OCRBench_v2": 600,
    "MMMU_Pro_10c": 300,
    "MathVista_MINI": 300,
    "MathVerse_MINI": 500,
}
EXPECTED_COVERAGE = {
    "official_manifest_rows": 2511,
    "evaluated_single_image_rows": 2240,
    "held_multi_image_rows": 271,
    "multi_image_policy": "unsupported_explicit_hold",
}


def _load_shared() -> ModuleType:
    path = REPOSITORY_ROOT / "tools/summarize_prl27_b_corrected_crop_s32_evaluation.py"
    spec = importlib.util.spec_from_file_location("_prl26_generic86_summary_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared Crop summarizer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EVALUATION_ID = EVALUATION_ID
    module.TRAINING_RUN_ID = TRAINING_RUN_ID
    module.PAIRED_SEED_NAMESPACE = PAIRED_SEED_NAMESPACE
    return module


_SHARED = _load_shared()
_sha256 = _SHARED._sha256
_canonical_sha256 = _SHARED._canonical_sha256
_read_json = _SHARED._read_json
_require_self_identity = _SHARED._require_self_identity
_read_rows = _SHARED._read_rows
_usage = _SHARED._usage
INFERENCE_ROWS = _SHARED.INFERENCE_ROWS


def summarize(evaluation_root: Path, output: Path) -> dict[str, object]:
    plan_path = evaluation_root / "runtime/bound-crop-plan.json"
    handoff_path = evaluation_root / "runtime/bound-handoff.json"
    reuse_path = evaluation_root / "runtime/full-model-materialization-reuse.json"
    proof_path = evaluation_root / "step32/runtime/pixel512-processor-proof.json"
    paired_path = evaluation_root / "paired-summary.json"
    summary_path = evaluation_root / (
        "step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json"
    )
    inference_root = evaluation_root / "step32/inference"

    plan = _read_json(plan_path)
    handoff = _read_json(handoff_path)
    reuse = _read_json(reuse_path)
    proof = _read_json(proof_path)
    paired = _read_json(paired_path)
    summary = _read_json(summary_path)
    expected_arm = {
        "name": "step32",
        "optimizer_step": STEP,
        "evaluation_id": EVALUATION_ID,
        "source": {
            "kind": "owner_checkpoint",
            "relative_path": "permanent-checkpoints/global_step_32",
        },
    }
    owner = plan.get("checkpoint_owner")
    protocol = plan.get("protocol")
    identity = (
        protocol.get("training_run_identity") if isinstance(protocol, dict) else None
    )
    if (
        plan.get("schema_version") != "tgvf.paired-policy-benchmark-plan.v3"
        or plan.get("evaluation_id") != EVALUATION_ID
        or plan.get("evaluation_image_max_pixels") != PIXEL512
        or plan.get("expected_task_count") != 2511
        or plan.get("expected_single_image_count") != 2240
        or plan.get("unsupported_multi_image_count") != 271
        or plan.get("arms") != [expected_arm]
        or not isinstance(owner, dict)
        or owner.get("run_id") != TRAINING_RUN_ID
        or not isinstance(protocol, dict)
        or protocol.get("evaluation_protocol") != "training_run"
        or protocol.get("training_run_variant") != VARIANT
        or protocol.get("action_boundary")
        != {
            "stop_strings": ["</tool_call>"],
            "stop_token_ids": [151645],
            "include_stop_str_in_output": True,
            "ignore_eos": False,
        }
        or not isinstance(identity, dict)
        or identity.get("tool_profile") != "crop_only"
        or identity.get("success_environment_renderer")
        != "render_qwen_native_success_environment_text"
        or identity.get("success_environment_text_sha256") != ENVIRONMENT_SHA256
        or identity.get("success_environment_token_count") != 86
        or identity.get("training_launch_project_commit") != TRAINING_LAUNCH_COMMIT
        or identity.get("response_budget_scope") != "total_response_tokens"
        or identity.get("single_response_max_tokens") != 10_240
        or plan.get("paired_rng", {}).get("seed_namespace")
        != PAIRED_SEED_NAMESPACE
        or plan.get("paired_rng", {}).get("protocol_sha256")
        != _canonical_sha256(identity)
    ):
        raise RuntimeError("PRL-26-B generic86 training-run plan identity differs")

    handoff_identity = _require_self_identity(handoff, name="handoff")
    crop = handoff.get("crop")
    if (
        handoff.get("schema_version")
        != "tgvf.prl26-b-generic86-training-run-evaluation-handoff.v1"
        or handoff.get("status") != "ready"
        or handoff.get("evaluation_id") != EVALUATION_ID
        or handoff.get("optimizer_step") != STEP
        or not isinstance(crop, dict)
        or crop.get("run_id") != TRAINING_RUN_ID
        or crop.get("training_run_variant") != VARIANT
        or crop.get("environment_text_sha256") != ENVIRONMENT_SHA256
        or crop.get("environment_token_count") != 86
        or crop.get("training_launch_project_commit") != TRAINING_LAUNCH_COMMIT
        or crop.get("bound_plan_file_sha256") != _sha256(plan_path)
    ):
        raise RuntimeError("PRL-26-B generic86 handoff identity differs")

    reuse_identity = _require_self_identity(reuse, name="materialization reuse")
    if (
        reuse.get("status") != "pass"
        or reuse.get("evaluation_id") != EVALUATION_ID
        or reuse.get("reuse_mode") != "read_only_existing_hf_tree_no_merge"
        or reuse.get("model_bytes_copied") is not False
        or reuse.get("merge_command_executed") is not False
    ):
        raise RuntimeError("PRL-26-B materialization reuse proof differs")

    proof_identity = _require_self_identity(proof, name="processor proof")
    proof_protocol = proof.get("protocol")
    dynamic_proof = proof.get("proof")
    if (
        proof.get("schema_version") != "tgvf.prl26-train512-processor-proof.v1"
        or proof.get("arm") != "crop"
        or proof.get("evaluation_id") != EVALUATION_ID
        or proof.get("optimizer_step") != STEP
        or proof.get("train_image_max_pixels") != PIXEL512
        or proof.get("evaluation_image_max_pixels") != PIXEL512
        or not isinstance(proof_protocol, dict)
        or proof_protocol.get("continuation_parity") is not True
        or proof_protocol.get("training_run_variant") != VARIANT
        or proof_protocol.get("success_environment_text_sha256")
        != ENVIRONMENT_SHA256
        or proof_protocol.get("continuation_environment_token_count") != 86
        or not isinstance(dynamic_proof, dict)
        or dynamic_proof.get("continuation_environment_token_count") != 86
        or dynamic_proof.get("success_environment_renderer")
        != "render_qwen_native_success_environment_text"
    ):
        raise RuntimeError("PRL-26-B generic86 processor proof differs")

    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "pass"
        or summary.get("phase") != "eval"
        or summary.get("sample_count") != 2511
        or summary.get("slice_count") != 7
        or {
            item.get("dataset"): item.get("sample_count")
            for item in summary.get("slices", [])
            if isinstance(item, dict)
        }
        != OFFICIAL_ROWS
    ):
        raise RuntimeError("PRL-26-B generic86 seven-subset summary differs")
    arms = paired.get("arms")
    contracts = paired.get("identity_contracts")
    if (
        paired.get("schema_version") != "tgvf.paired-coredev-summary.v2"
        or paired.get("evaluation_id") != EVALUATION_ID
        or paired.get("coverage") != EXPECTED_COVERAGE
        or not isinstance(arms, dict)
        or set(arms) != {"step32"}
        or paired.get("step32") != summary
        or not isinstance(contracts, dict)
        or contracts.get("training_run_protocol") != protocol
        or arms["step32"].get("evaluation_identity_sha256")
        != proof.get("evaluation_identity_sha256")
    ):
        raise RuntimeError("PRL-26-B generic86 paired summary identity differs")

    rows = _read_rows(inference_root, str(proof["evaluation_identity_sha256"]))
    headline = extract_coredev_macro_star(summary)
    payload: dict[str, object] = {
        "schema_version": "tgvf.prl26-b-generic86-s32-results.v1",
        "status": "pass",
        "evaluation_id": EVALUATION_ID,
        "contract": (
            "historical PRL-26-B Train@512 S32; exact owner-native generic86 "
            "training_run Eval@512"
        ),
        "coverage": {
            "official_manifest_rows": 2511,
            "evaluated_single_image_rows": 2240,
            "held_multi_image_rows": 271,
            "subset_count": 7,
        },
        "handoff_identity_sha256": handoff_identity,
        "materialization_reuse_identity_sha256": reuse_identity,
        "processor_proof_identity_sha256": proof_identity,
        "paired_rng_protocol_sha256": plan["paired_rng"]["protocol_sha256"],
        "length_unit": "sampled model tokens across all assistant turns",
        "arm": {
            "method": "Crop S32 owner-generic86 matched",
            "optimizer_step": STEP,
            "train_image_max_pixels": PIXEL512,
            "evaluation_image_max_pixels": PIXEL512,
            "macro_star_percent": headline["macro_star_percent"],
            "headline": headline,
            "seven_subset_statistics": summary["slices"],
            "tool_usage_overall": _usage(rows),
            "tool_usage_by_subset": {
                dataset: _usage([row for row in rows if row["dataset"] == dataset])
                for dataset in INFERENCE_ROWS
            },
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
        },
    }
    write_json_atomic(output.resolve(), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.evaluation_root.resolve(), args.output.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
