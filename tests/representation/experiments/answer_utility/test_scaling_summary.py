from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest


_TOOL_PATH = (
    Path(__file__).resolve().parents[4]
    / "tools"
    / "summarize_representation_answer_utility_scaling.py"
)
_SPEC = importlib.util.spec_from_file_location("answer_utility_scaling_tool", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
scaling = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scaling
_SPEC.loader.exec_module(scaling)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()  # synthetic fixtures are ASCII-only


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _metric(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(row["final_score"]["correct"] is True for row in rows)
    return {
        "total": len(rows),
        "diagnostic_final_correct": correct,
        "diagnostic_final_incorrect": len(rows) - correct,
        "diagnostic_semantic_accuracy": correct / len(rows),
    }


def _make_overlay(
    root: Path,
    *,
    step: int,
    sample_outcomes: Mapping[str, tuple[bool, bool]],
    training_run_sha: str,
    data_sha: str,
    ordered_sha: str,
) -> Path:
    """Create one fully hash-bound miniature semantic overlay.

    Outcome tuples are ``(image_only, image_correct_D)``.  Non-2000 steps
    intentionally omit image-only, matching the production curve protocol.
    """

    root.mkdir(parents=True)
    generation_root = root.parent / f"generation-step{step}"
    generation_root.mkdir()
    source_config = root.parent / f"source-step{step}.toml"
    source_config.write_text(
        "[artifact]\n"
        f'expected_run_identity_sha256 = "{training_run_sha}"\n'
        f"expected_global_step = {step}\n"
        "\n[evaluation]\n"
        f'ordered_group_manifest_sha256 = "{ordered_sha}"\n',
        encoding="utf-8",
    )
    source_config_sha = sha256(source_config.read_bytes()).hexdigest()
    arms = (
        [scaling.BASELINE_ARM, scaling.CORRECT_D_ARM]
        if step == 2000
        else [scaling.CORRECT_D_ARM]
    )
    selected = [
        {
            "sample_id": sample_id,
            "sample_content_sha256": sha256(sample_id.encode()).hexdigest(),
        }
        for sample_id in sample_outcomes
    ]
    generation_identity = {
        "schema_version": scaling.GENERATION_IDENTITY_SCHEMA,
        "candidate_id": f"rp66-step{step}",
        "candidate_kind": "production_source",
        "candidate_global_step": step,
        "production_source_global_step": step,
        "candidate_training_run_identity_sha256": training_run_sha,
        "production_source_run_identity_sha256": training_run_sha,
        "source_evaluation_config_sha256": source_config_sha,
        "data_manifest_sha256": data_sha,
        "ordered_group_manifest_sha256": ordered_sha,
        "arms": arms,
        "arm_contracts": {
            arm: scaling.EXPECTED_ARM_CONTRACTS[arm] for arm in arms
        },
        "ordered_selected_samples": selected,
    }
    generation_identity_sha = _sha(generation_identity)
    generation_outer = {
        "schema_version": "representation_oracle_d_utility_v1",
        "identity_sha256": generation_identity_sha,
        "identity": generation_identity,
    }
    _write_json(generation_root / "identity.json", generation_outer)

    generation_rows: list[dict[str, Any]] = []
    generation_sha_by_key: dict[tuple[str, str], str] = {}
    for sample_id in sample_outcomes:
        for arm in arms:
            row = {
                "schema_version": "answer-utility-instruct-evaluation-record-v2",
                "run_identity_sha256": generation_identity_sha,
                "candidate_id": f"rp66-step{step}",
                "sample_id": sample_id,
                "arm": arm,
            }
            generation_rows.append(row)
            generation_sha_by_key[(sample_id, arm)] = _sha(row)
    generation_records_payload = b"".join(
        _canonical(row) + b"\n" for row in generation_rows
    )
    (generation_root / "records.jsonl").write_bytes(generation_records_payload)
    generation_summary = {
        "status": "complete",
        "run_identity_sha256": generation_identity_sha,
        "record_count": len(generation_rows),
        "records_jsonl_sha256": sha256(generation_records_payload).hexdigest(),
    }
    _write_json(generation_root / "summary.json", generation_summary)
    label = f"rp66-step{step}@{generation_identity_sha[:12]}"
    source_binding = {
        "root": str(generation_root.resolve()),
        "candidate_id": f"rp66-step{step}",
        "label": label,
        "generation_identity_sha256": generation_identity_sha,
        "identity_file_sha256": sha256(
            (generation_root / "identity.json").read_bytes()
        ).hexdigest(),
        "records_file_sha256": sha256(generation_records_payload).hexdigest(),
        "summary_file_sha256": sha256(
            (generation_root / "summary.json").read_bytes()
        ).hexdigest(),
        "record_count": len(generation_rows),
    }
    semantic_identity = {
        "schema_version": scaling.SEMANTIC_SCHEMA_VERSION,
        "evaluation_data_manifest_sha256": data_sha,
        "source_evaluation_config_path": str(source_config.resolve()),
        "source_evaluation_config_sha256": source_config_sha,
        "deterministic_scoring_contract_version": "synthetic-score-v1",
        "generation_sources": [source_binding],
        "judge": {"identity": "synthetic-judge-v1"},
    }
    semantic_run_sha = _sha(semantic_identity)
    semantic_rows: list[dict[str, Any]] = []
    for sample_id, outcomes in sample_outcomes.items():
        for arm in arms:
            correct = outcomes[0] if arm == scaling.BASELINE_ARM else outcomes[1]
            record_without_sha = {
                "schema_version": scaling.SEMANTIC_RECORD_SCHEMA_VERSION,
                "run_identity_sha256": semantic_run_sha,
                "candidate_id": f"rp66-step{step}",
                "source_generation_identity_sha256": generation_identity_sha,
                "source_label": label,
                "source_root": str(generation_root.resolve()),
                "source_record_sha256": generation_sha_by_key[(sample_id, arm)],
                "sample_id": sample_id,
                "arm": arm,
                "final_score": {
                    "correct": correct,
                    "route": "synthetic",
                    "scope": scaling.EXPECTED_CLAIM_SCOPE,
                },
            }
            semantic_rows.append(
                {
                    **record_without_sha,
                    "overlay_record_sha256": _sha(record_without_sha),
                }
            )
    semantic_records_payload = b"".join(
        _canonical(row) + b"\n" for row in semantic_rows
    )
    (root / "records.jsonl").write_bytes(semantic_records_payload)
    by_arm_rows = {
        arm: [row for row in semantic_rows if row["arm"] == arm] for arm in arms
    }
    semantic_summary = {
        "schema_version": scaling.SEMANTIC_SCHEMA_VERSION,
        "status": "complete",
        "run_identity_sha256": semantic_run_sha,
        "claim_scope": scaling.EXPECTED_CLAIM_SCOPE,
        "overall": _metric(semantic_rows),
        "by_arm": {arm: _metric(rows) for arm, rows in by_arm_rows.items()},
        "by_candidate": {
            f"rp66-step{step}": {
                "overall": _metric(semantic_rows),
                "by_arm": {
                    arm: _metric(rows) for arm, rows in by_arm_rows.items()
                },
            }
        },
    }
    _write_json(root / "summary.json", semantic_summary)
    (root / "blind_requests.jsonl").write_bytes(b"")
    (root / "judge_evidence.jsonl").write_bytes(b"")
    files = {
        "blind_requests": {
            "path": "blind_requests.jsonl",
            "sha256": sha256(b"").hexdigest(),
            "rows": 0,
        },
        "judge_evidence": {
            "path": "judge_evidence.jsonl",
            "sha256": sha256(b"").hexdigest(),
            "rows": 0,
        },
        "overlay_records": {
            "path": "records.jsonl",
            "sha256": sha256(semantic_records_payload).hexdigest(),
            "rows": len(semantic_rows),
        },
        "summary": {
            "path": "summary.json",
            "sha256": sha256((root / "summary.json").read_bytes()).hexdigest(),
        },
    }
    manifest_without_sha = {
        "schema_version": scaling.SEMANTIC_SCHEMA_VERSION,
        "status": "complete",
        "run_identity_sha256": semantic_run_sha,
        "identity": semantic_identity,
        "files": files,
        "unique_judge_requests": 0,
        "judge_consumer_count": 0,
    }
    _write_json(
        root / "manifest.json",
        {**manifest_without_sha, "manifest_sha256": _sha(manifest_without_sha)},
    )
    return root


def _curve(tmp_path: Path) -> tuple[dict[int, Path], dict[str, str]]:
    identities = {
        "training": "1" * 64,
        "data": "2" * 64,
        "ordered": "3" * 64,
    }
    outcomes = {
        500: {
            "a": (True, True),
            "b": (True, False),
            "c": (False, True),
            "d": (False, False),
        },
        1000: {
            "a": (True, True),
            "b": (True, True),
            "c": (False, True),
            "d": (False, False),
        },
        1500: {
            "a": (True, False),
            "b": (True, True),
            "c": (False, True),
            "d": (False, True),
        },
        2000: {
            "a": (True, True),
            "b": (True, True),
            "c": (False, True),
            "d": (False, True),
        },
    }
    roots = {
        step: _make_overlay(
            tmp_path / f"semantic-step{step}",
            step=step,
            sample_outcomes=outcomes[step],
            training_run_sha=identities["training"],
            data_sha=identities["data"],
            ordered_sha=identities["ordered"],
        )
        for step in scaling.EXPECTED_STEPS
    }
    return roots, identities


def test_summarize_scaling_strictly_binds_steps_and_computes_paired_curve(
    tmp_path: Path,
) -> None:
    roots, identities = _curve(tmp_path)

    result = scaling.summarize_scaling(
        roots,
        expected_samples=4,
        expected_training_run_identity_sha256=identities["training"],
        expected_data_manifest_sha256=identities["data"],
        expected_ordered_group_manifest_sha256=identities["ordered"],
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )

    assert result["status"] == "complete"
    assert result["image_only_baseline"]["correct"] == 2
    curve = {point["global_step"]: point for point in result["checkpoint_curve"]}
    assert curve[500]["image_correct_D"]["correct"] == 2
    assert curve[1000]["image_correct_D"]["correct"] == 3
    assert curve[2000]["image_correct_D"]["correct"] == 4
    paired = curve[500]["paired_vs_shared_image_only"]
    assert paired["treatment_only_correct_wins"] == 1
    assert paired["baseline_only_correct_losses"] == 1
    assert paired["accuracy_delta"] == 0.0
    assert paired["baseline_prediction_identity_relabelled"] is False
    comparisons = {
        item["comparison_id"]: item
        for item in result["checkpoint_to_checkpoint_paired_comparisons"]
    }
    assert set(comparisons) == {
        "step1000_vs_step0500",
        "step1500_vs_step0500",
        "step2000_vs_step0500",
        "step1500_vs_step1000",
        "step2000_vs_step1500",
    }
    step1000_vs_500 = comparisons["step1000_vs_step0500"]
    assert step1000_vs_500["comparison_roles"] == ["vs_step0500", "adjacent"]
    assert step1000_vs_500["treatment_only_correct_wins"] == 1
    assert step1000_vs_500["control_only_correct_losses"] == 0
    assert step1000_vs_500["ties"] == 3
    assert step1000_vs_500["accuracy_delta"] == 0.25
    step1500_vs_500 = comparisons["step1500_vs_step0500"]
    assert step1500_vs_500["treatment_only_correct_wins"] == 2
    assert step1500_vs_500["control_only_correct_losses"] == 1
    assert step1500_vs_500["ties"] == 1
    step2000_vs_1500 = comparisons["step2000_vs_step1500"]
    assert step2000_vs_1500["treatment_only_correct_wins"] == 1
    assert step2000_vs_1500["control_only_correct_losses"] == 0
    assert step2000_vs_1500["paired_bootstrap_95"]["seed"] == (
        7 + 100_000_000 + 2000 * 10_000 + 1500
    )
    markdown = scaling.render_markdown(result)
    assert "Checkpoint-to-checkpoint paired comparisons" in markdown
    assert "| 2000 vs 1500 | adjacent |" in markdown
    assert result["summary_sha256"] == _sha(
        {key: value for key, value in result.items() if key != "summary_sha256"}
    )


def test_summarize_scaling_rejects_tampered_actual_checkpoint_step(
    tmp_path: Path,
) -> None:
    roots, identities = _curve(tmp_path)
    source = tmp_path / "source-step1000.toml"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "expected_global_step = 1000", "expected_global_step = 999"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source evaluation config SHA256 differs"):
        scaling.summarize_scaling(
            roots,
            expected_samples=4,
            expected_training_run_identity_sha256=identities["training"],
            expected_data_manifest_sha256=identities["data"],
            expected_ordered_group_manifest_sha256=identities["ordered"],
            bootstrap_replicates=10,
        )


def test_summarize_scaling_rejects_wrong_sample_count_before_statistics(
    tmp_path: Path,
) -> None:
    roots, identities = _curve(tmp_path)

    with pytest.raises(ValueError, match="semantic record count differs|row binding differs"):
        scaling.summarize_scaling(
            roots,
            expected_samples=5,
            expected_training_run_identity_sha256=identities["training"],
            expected_data_manifest_sha256=identities["data"],
            expected_ordered_group_manifest_sha256=identities["ordered"],
            bootstrap_replicates=10,
        )


def test_checkpoint_paired_bootstrap_is_seed_deterministic() -> None:
    arguments = {
        "comparison_id": "step1000_vs_step0500",
        "comparison_roles": ("vs_step0500", "adjacent"),
        "treatment_step": 1000,
        "control_step": 500,
        "treatment_candidate_id": "step1000",
        "control_candidate_id": "step0500",
        "bootstrap_replicates": 250,
        "bootstrap_seed": 123,
    }
    treatment = {"a": True, "b": True, "c": False, "d": False}
    control = {"a": True, "b": False, "c": True, "d": False}

    first = scaling._checkpoint_paired_effect(treatment, control, **arguments)
    second = scaling._checkpoint_paired_effect(treatment, control, **arguments)

    assert first == second
    assert first["paired_bootstrap_95"]["seed"] == (
        123 + 100_000_000 + 1000 * 10_000 + 500
    )
