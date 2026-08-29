from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SUPERVISOR = _ROOT / "tools/supervise_prl26_e_atomic_train_and_eval.sh"
_EVALUATOR = _ROOT / "tools/supervise_prl26_e_atomic_s32_evaluation.sh"
_LAUNCHER = _ROOT / "tools/launch_prl26_e_atomic_train_and_eval_tmux.sh"
_VALIDATOR = _ROOT / "tools/validate_prl26_e_atomic_handoff.py"
_PLAN = _ROOT / (
    "configs/evaluation/"
    "prl26_e_atomic_crop_tgvf_train512_s32_pixel512_coredev2511_plan.json"
)


def _validator_module():
    spec = importlib.util.spec_from_file_location(
        "prl26_e_atomic_validator", _VALIDATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atomic_scripts_are_shell_valid_and_launcher_does_not_inline_work() -> None:
    for script in (_SUPERVISOR, _EVALUATOR, _LAUNCHER):
        subprocess.run(["bash", "-n", str(script)], check=True)
    launcher = _LAUNCHER.read_text(encoding="utf-8")
    assert "supervise_prl26_e_atomic_train_and_eval.sh" in launcher
    assert "tmux new-session -d" in launcher
    assert "trainable_tgvf_launcher" not in launcher


def test_atomic_relay_requires_completed_tgvf_eval_then_clean_resources() -> None:
    source = _SUPERVISOR.read_text(encoding="utf-8")

    prerequisite = "waiting_for_tgvf_paired_evaluation"
    audit = '"$validator" prerequisite'
    resources = "wait_for_resources prerequisite"
    canary = "running_atomic_c0"
    formal = "training_atomic_to_s32"
    evaluation = "starting_atomic_evaluation"
    for fragment in (prerequisite, audit, resources, canary, formal, evaluation):
        assert fragment in source
    assert 'while [[ ! -f "$prerequisite_complete" ]]' in source
    assert 'while [[ ! -s "$prerequisite_complete" ]]' not in source
    assert '[[ ! -f "$state_root/canary-accepted" ]]' in source
    assert '[[ ! -f "$state_root/atomic-s32-accepted" ]]' in source
    assert '[[ ! -s "$state_root/canary-accepted" ]]' not in source
    assert '[[ ! -s "$state_root/atomic-s32-accepted" ]]' not in source
    assert "--require-output-roots-absent" in source
    assert "release_stable_polls < 2" in source
    assert "control state belongs to another admitted HEAD" in source
    assert source.index(prerequisite) < source.index(audit)
    assert source.index(audit) < source.index(resources)
    assert source.index(resources) < source.index(canary)
    assert source.index(canary) < source.index(formal)
    assert source.index(formal) < source.index(evaluation)
    assert "--target-step 32" in source
    assert "WANDB_RUN_ID=prl26e_train512_atomic_s32" in source


def test_atomic_evaluator_proves_boundary_before_inference() -> None:
    source = _EVALUATOR.read_text(encoding="utf-8")

    proof = "validating_atomic_pixel512_processor_and_boundary"
    infer = "running_atomic_four_gpu_inference"
    score = "scoring_atomic_seven_subsets"
    assert "--arm atomic" in source
    assert source.index(proof) < source.index(infer) < source.index(score)
    assert "summarize_prl26_e_atomic_s32_evaluation.py" in source
    assert 'admitted_head_file="$control_root/admitted-head.txt"' in source
    assert "evaluation admitted HEAD differs from training control" in source
    assert source.count("validate_worktree") >= 10
    assert "release_stable_polls < 2" in source


def test_atomic_plan_and_development_contract_audit_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _validator_module()
    plan = json.loads(_PLAN.read_text(encoding="utf-8"))
    assert plan["evaluation_image_max_pixels"] == 262_144
    assert plan["arms"][0]["optimizer_step"] == 32
    assert plan["paired_rng"]["master_seed"] == 42
    assert plan["paired_rng"]["temperature"] == 1.0
    result = validator.validate_contracts(
        repository=_ROOT,
        canary_path=_ROOT
        / (
            "configs/policy/runs/prl_26_e_c0_qwen3_instruct_full_atomic_"
            "crop_tgvf_train512_parity_bs4_n2_teacher25_1step_ws4.toml"
        ),
        formal_path=_ROOT
        / (
            "configs/policy/runs/prl_26_e_qwen3_instruct_full_atomic_crop_"
            "tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
        ),
        reference_canary_path=_ROOT
        / (
            "configs/policy/runs/prl_26_c_c0_qwen3_instruct_short_tgvf_"
            "train512_parity_bs4_n2_teacher25_1step_ws4.toml"
        ),
        reference_formal_path=_ROOT
        / (
            "configs/policy/runs/prl_26_c_qwen3_instruct_short_tgvf_"
            "train512_parity_s32_bs16_n16_teacher25_ws8.toml"
        ),
        plan_path=_PLAN,
        require_clean=False,
    )
    assert result["status"] == "accepted"
    assert result["all_output_roots_absent"] is True
    assert result["runtime_baseline_commit"] == (
        "8e6b3d647d3a94c7768e3d8718b69d544010841e"
    )
    assert result["reference_configs"] == {
        "canary": {
            "path": str(
                _ROOT
                / (
                    "configs/policy/runs/prl_26_c_c0_qwen3_instruct_short_"
                    "tgvf_train512_parity_bs4_n2_teacher25_1step_ws4.toml"
                )
            ),
            "file_sha256": (
                "cfcb374b91261174c14bfcd23a843a23f17631efb38f0c44e7ed92e7d35bd724"
            ),
        },
        "formal": {
            "path": str(
                _ROOT
                / (
                    "configs/policy/runs/prl_26_c_qwen3_instruct_short_"
                    "tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
                )
            ),
            "file_sha256": (
                "17efa65a8622f36c337a0691bed2dd9acf799562a2109595d8a43acf3b7ebe17"
            ),
        },
    }

    monkeypatch.setattr(validator.os.path, "lexists", lambda _path: True)
    with pytest.raises(RuntimeError, match="requires fresh output roots"):
        validator.validate_contracts(
            repository=_ROOT,
            canary_path=_ROOT
            / (
                "configs/policy/runs/prl_26_e_c0_qwen3_instruct_full_atomic_"
                "crop_tgvf_train512_parity_bs4_n2_teacher25_1step_ws4.toml"
            ),
            formal_path=_ROOT
            / (
                "configs/policy/runs/prl_26_e_qwen3_instruct_full_atomic_crop_"
                "tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml"
            ),
            reference_canary_path=_ROOT
            / (
                "configs/policy/runs/prl_26_c_c0_qwen3_instruct_short_tgvf_"
                "train512_parity_bs4_n2_teacher25_1step_ws4.toml"
            ),
            reference_formal_path=_ROOT
            / (
                "configs/policy/runs/prl_26_c_qwen3_instruct_short_tgvf_"
                "train512_parity_s32_bs16_n16_teacher25_ws8.toml"
            ),
            plan_path=_PLAN,
            require_clean=False,
            require_output_roots_absent=True,
        )


def test_atomic_prerequisite_is_closed_over_exact_cd_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _validator_module()
    root = tmp_path / "PRL26-CD-TGVF-PROMPT-PAIR-S32-PIXEL512-COREDEV2511-V1"
    result_path = root / "tgvf-target-prompt-s32-pixel512-results.json"
    complete_marker = root / "runtime/evaluation-complete"
    failed_marker = root / "runtime/failed"
    paired_summary_path = root / "paired-summary.json"
    runner_complete_path = root / "evaluation-complete"
    for name, value in {
        "_PREREQUISITE_ROOT": root,
        "_PREREQUISITE_RESULT": result_path,
        "_PREREQUISITE_COMPLETE": complete_marker,
        "_PREREQUISITE_FAILED": failed_marker,
        "_PREREQUISITE_PAIRED_SUMMARY": paired_summary_path,
        "_PREREQUISITE_RUNNER_COMPLETE": runner_complete_path,
    }.items():
        monkeypatch.setattr(validator, name, value)
    headline = {"macro_star_percent": 61.25, "component_count": 7}
    monkeypatch.setattr(
        validator, "extract_coredev_macro_star", lambda _summary: headline
    )

    summaries: dict[str, dict[str, object]] = {}
    paired_arms: dict[str, dict[str, object]] = {}
    result_arms: dict[str, dict[str, object]] = {}
    for index, name in enumerate(("short", "full")):
        summary = {
            "schema_version": 1,
            "status": "pass",
            "phase": "eval",
            "sample_count": 2511,
            "slice_count": 7,
            "slices": [{"dataset": f"dataset-{item}"} for item in range(7)],
        }
        summary_path = (
            root
            / name
            / "scoring/coredev-official-v1/coredev-2511-eval-summary.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        summaries[name] = summary
        expected = validator._PREREQUISITE_ARM_CONTRACTS[name]
        paired_arms[name] = {
            "optimizer_step": 32,
            "evaluation_id": expected["evaluation_id"],
            "evaluation_image_max_pixels": 262_144,
            "arm_protocol_sha256": expected["arm_protocol_sha256"],
            "seed_protocol_sha256": validator._PREREQUISITE_SEED_PROTOCOL_SHA256,
            "evaluation_identity_sha256": ("a" if index == 0 else "b") * 64,
            "official_summary": summary,
        }
        result_arms[name] = {
            "method": expected["method"],
            "optimizer_step": 32,
            "train_image_max_pixels": 262_144,
            "evaluation_image_max_pixels": 262_144,
            "macro_star_percent": headline["macro_star_percent"],
            "headline": headline,
            "seven_subset_statistics": summary["slices"],
            "summary_path": str(summary_path),
            "summary_sha256": validator._sha256(summary_path),
        }

    paired_rng = {
        "mode": "common_random_numbers_per_task_turn",
        "master_seed": 42,
        "task_manifest_sha256": (
            "3f69119d24867c3f3210c8b01eb71304247725ddaf9ca983d2b41c2885403cbc"
        ),
        "seed_protocol_sha256": validator._PREREQUISITE_SEED_PROTOCOL_SHA256,
        "arm_protocol_sha256": {
            name: contract["arm_protocol_sha256"]
            for name, contract in validator._PREREQUISITE_ARM_CONTRACTS.items()
        },
        "protocol_projection": validator._PREREQUISITE_TARGET_PROMPT_PAIR,
        "temperature": 1.0,
        "do_sample": True,
    }
    paired = {
        "schema_version": "tgvf.paired-coredev-summary.v2",
        "evaluation_id": validator._PREREQUISITE_EVALUATION_ID,
        "coverage": validator._PREREQUISITE_PAIRED_COVERAGE,
        "target_prompt_pair": validator._PREREQUISITE_TARGET_PROMPT_PAIR,
        "sampling": {
            "source": "bound_policy_run_config",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "do_sample": True,
            "paired_rng": paired_rng,
        },
        "arms": paired_arms,
        "short": summaries["short"],
        "full": summaries["full"],
    }
    root.mkdir(parents=True, exist_ok=True)
    paired_summary_path.write_text(json.dumps(paired), encoding="utf-8")
    paired_sha256 = validator._sha256(paired_summary_path)
    result = {
        "schema_version": validator._RESULT_SCHEMA,
        "status": "pass",
        "evaluation_id": validator._PREREQUISITE_EVALUATION_ID,
        "contract": validator._PREREQUISITE_CONTRACT,
        "coverage": validator._PREREQUISITE_COVERAGE,
        "paired_rng_streams_equal": True,
        "paired_rng_stream_count": 2240,
        "paired_summary_path": str(paired_summary_path),
        "paired_summary_sha256": paired_sha256,
        "arms": result_arms,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    complete_marker.parent.mkdir(parents=True, exist_ok=True)
    complete_marker.touch()
    runner_complete_path.write_text(
        json.dumps(
            {
                "schema_version": "tgvf.paired-coredev-evaluation-complete.v1",
                "status": "complete",
                "evaluation_id": validator._PREREQUISITE_EVALUATION_ID,
                "paired_summary_path": str(paired_summary_path.resolve()),
                "paired_summary_sha256": paired_sha256,
            }
        ),
        encoding="utf-8",
    )

    accepted = validator.validate_prerequisite(
        result_path=result_path,
        complete_marker=complete_marker,
        failed_marker=failed_marker,
    )
    assert accepted["status"] == "accepted"
    assert accepted["evaluation_id"] == validator._PREREQUISITE_EVALUATION_ID

    stale_path = root / "stale-result.json"
    stale_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="path or file boundary differs"):
        validator.validate_prerequisite(
            result_path=stale_path,
            complete_marker=complete_marker,
            failed_marker=failed_marker,
        )

    result["evaluation_id"] = "stale-evaluation"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="prerequisite result differs"):
        validator.validate_prerequisite(
            result_path=result_path,
            complete_marker=complete_marker,
            failed_marker=failed_marker,
        )
