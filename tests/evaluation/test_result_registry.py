from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tgvf_rl.evaluation.result_registry import (
    IncomparableResultsError,
    RegistryValidationError,
    ResultRegistry,
    ResultStatus,
    load_result_registry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPOSITORY_ROOT / "evidence/policy/result_registry_v2.json"
MATERIALIZER = REPOSITORY_ROOT / "tools/materialize_policy_result_table.py"


def _payload() -> dict[str, object]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _minimal_payload() -> dict[str, object]:
    payload = _payload()
    payload["comparison_definitions"] = []
    payload["results"] = [deepcopy(payload["results"][0])]
    payload["tables"] = [
        {
            "table_id": "one",
            "title": "One",
            "result_ids": [payload["results"][0]["result_id"]],
            "baseline_result_id": payload["results"][0]["result_id"],
        }
    ]
    return payload


def _add_comparison_definition(
    payload: dict[str, object],
    *,
    comparison_group: str,
    members: list[str],
    intervention_axes: list[str],
) -> None:
    payload["comparison_definitions"].append(
        {
            "comparison_group": comparison_group,
            "member_result_ids": sorted(members),
            "intervention_axes": sorted(intervention_axes),
            "preregistration_evidence": {
                "path": f"evidence/policy/comparisons/{comparison_group.replace('/', '-')}.json",
                "sha256": "0" * 64,
                "kind": "preregistration_json_v1",
            },
        }
    )


def _promote_pair(
    payload: dict[str, object],
    baseline: dict[str, object],
    treatment: dict[str, object],
    *,
    comparison_group: str,
    intervention_axes: list[str],
) -> None:
    for result in (baseline, treatment):
        result["status"] = "golden"
        result["comparison_group"] = comparison_group
        result["declared_intervention_axes"] = sorted(intervention_axes)
    _add_comparison_definition(
        payload,
        comparison_group=comparison_group,
        members=[baseline["result_id"], treatment["result_id"]],
        intervention_axes=intervention_axes,
    )


def _score_summary_payload(score: dict[str, object]) -> dict[str, object]:
    return {
        "headline": {
            "schema_version": "tgvf.coredev-2511-macro-star.v1",
            "macro_star_percent": score["macro_star_percent"],
            "components_percent": score["components_percent"],
        }
    }


def test_checked_in_registry_classifies_audited_results_fail_closed() -> None:
    registry = load_result_registry(REGISTRY_PATH)

    assert registry.comparison_definitions == ()
    assert not [
        record for record in registry.results if record.status is ResultStatus.GOLDEN
    ]
    assert registry.result("crop-prl27a-hybrid-s0-s4").status is ResultStatus.INVALID
    assert (
        registry.result("crop-prl27b-canonical-s32-eval").status is ResultStatus.PENDING
    )
    old_crop = registry.result("crop-prl25b-s80-boundaryfix-processor-default")
    assert old_crop.contract.declared_evaluation_image_max_pixels == 262_144
    assert old_crop.contract.effective_evaluation_image_max_pixels == 16_777_216
    no_tool = registry.result("notool-prl26a-s32-512")
    crop = registry.result("crop-prl26b-s32-512-official60")
    assert no_tool.contract.rng_identity != crop.contract.rng_identity
    assert "no-tool-overlap=0-of-2240" in crop.contract.rng_identity
    assert (
        crop.contract.action_boundary_identity
        == "historical/action-boundary-not-cryptographically-bound"
    )


def test_registry_rejects_unknown_and_missing_contract_fields() -> None:
    payload = _minimal_payload()
    payload["unexpected"] = True
    with pytest.raises(RegistryValidationError, match="unknown keys"):
        ResultRegistry.from_json(payload)

    payload = _minimal_payload()
    del payload["results"][0]["contract"]["parser_identity"]
    with pytest.raises(RegistryValidationError, match="parser_identity"):
        ResultRegistry.from_json(payload)

    payload = _minimal_payload()
    del payload["results"][0]["contract"]["action_boundary_identity"]
    with pytest.raises(RegistryValidationError, match="action_boundary_identity"):
        ResultRegistry.from_json(payload)


def test_measured_status_requires_score_artifact_and_present_weight_hash() -> None:
    payload = _minimal_payload()
    payload["results"][0]["score_artifact"] = None
    with pytest.raises(RegistryValidationError, match="bind score and score_artifact"):
        ResultRegistry.from_json(payload)

    payload = _minimal_payload()
    payload["results"][0]["weights"]["sha256"] = None
    with pytest.raises(RegistryValidationError, match="required for present weights"):
        ResultRegistry.from_json(payload)


def test_pending_result_cannot_smuggle_a_score() -> None:
    payload = _minimal_payload()
    result = payload["results"][0]
    result["status"] = "pending"
    result["weights"] = {
        "state": "expected",
        "identity": "future snapshot",
        "sha256": None,
    }
    with pytest.raises(RegistryValidationError, match="pending results cannot"):
        ResultRegistry.from_json(payload)


def test_declared_weights_intervention_permits_delta() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "same-contract-treatment"
    treatment["label"] = "same contract treatment"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    treatment["score"]["macro_star_percent"] += 1.0
    for name in treatment["score"]["components_percent"]:
        treatment["score"]["components_percent"][name] += 1.0
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/weights-ablation-v1",
        intervention_axes=["weights"],
    )
    payload["results"].append(treatment)
    payload["tables"][0]["result_ids"].append("same-contract-treatment")
    registry = ResultRegistry.from_json(payload)

    assert registry.delta(
        result_id="same-contract-treatment",
        baseline_result_id=baseline["result_id"],
    ) == pytest.approx(1.0)


def test_standalone_rows_never_emit_a_delta() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    baseline["comparison_group"] = "test/standalone-not-comparison-v1"
    baseline["declared_intervention_axes"] = ["weights"]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "standalone-treatment"
    treatment["label"] = "standalone treatment"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    payload["results"].append(treatment)
    payload["tables"][0]["result_ids"].append(treatment["result_id"])
    registry = ResultRegistry.from_json(payload)

    with pytest.raises(IncomparableResultsError, match="golden status"):
        registry.delta(
            result_id=treatment["result_id"],
            baseline_result_id=baseline["result_id"],
        )


def test_undeclared_treatment_difference_blocks_delta() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "undeclared-prompt-treatment"
    treatment["label"] = "undeclared prompt treatment"
    treatment["contract"]["prompt_identity"] = "different-prompt"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/weights-only-ablation-v1",
        intervention_axes=["weights"],
    )
    payload["results"].append(treatment)
    payload["tables"][0]["result_ids"].append(treatment["result_id"])
    registry = ResultRegistry.from_json(payload)

    with pytest.raises(
        IncomparableResultsError,
        match="undeclared intervention differences: prompt_identity",
    ):
        registry.delta(
            result_id=treatment["result_id"],
            baseline_result_id=baseline["result_id"],
        )


def test_declared_single_intervention_and_only_that_difference_permits_delta() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "declared-prompt-treatment"
    treatment["label"] = "declared prompt treatment"
    treatment["contract"]["prompt_identity"] = "different-prompt"
    treatment["score"]["macro_star_percent"] += 1.0
    for name in treatment["score"]["components_percent"]:
        treatment["score"]["components_percent"][name] += 1.0
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/prompt-ablation-v1",
        intervention_axes=["prompt_identity"],
    )
    payload["results"].append(treatment)
    payload["tables"][0]["result_ids"].append(treatment["result_id"])
    registry = ResultRegistry.from_json(payload)

    assert registry.delta(
        result_id=treatment["result_id"],
        baseline_result_id=baseline["result_id"],
    ) == pytest.approx(1.0)


def test_extra_difference_beyond_declared_intervention_blocks_delta() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "prompt-plus-runtime-treatment"
    treatment["label"] = "prompt plus runtime treatment"
    treatment["contract"]["prompt_identity"] = "different-prompt"
    treatment["contract"]["runtime_identity"] = "different-runtime"
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/prompt-ablation-v1",
        intervention_axes=["prompt_identity"],
    )
    payload["results"].append(treatment)
    payload["tables"][0]["result_ids"].append(treatment["result_id"])
    registry = ResultRegistry.from_json(payload)

    with pytest.raises(
        IncomparableResultsError,
        match="undeclared intervention differences: runtime_identity",
    ):
        registry.delta(
            result_id=treatment["result_id"],
            baseline_result_id=baseline["result_id"],
        )


def test_declared_intervention_must_actually_differ() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "prompt-only-treatment"
    treatment["label"] = "prompt only treatment"
    treatment["contract"]["prompt_identity"] = "different-prompt"
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/prompt-runtime-ablation-v1",
        intervention_axes=["prompt_identity", "runtime_identity"],
    )
    payload["results"].append(treatment)
    payload["tables"][0]["result_ids"].append(treatment["result_id"])
    registry = ResultRegistry.from_json(payload)

    with pytest.raises(
        IncomparableResultsError,
        match="declared intervention axes did not differ: runtime_identity",
    ):
        registry.delta(
            result_id=treatment["result_id"],
            baseline_result_id=baseline["result_id"],
        )


def test_different_comparison_groups_block_delta() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "different-group-treatment"
    treatment["label"] = "different group treatment"
    baseline_companion = deepcopy(baseline)
    baseline_companion["result_id"] = "baseline-companion"
    treatment_companion = deepcopy(baseline)
    treatment_companion["result_id"] = "treatment-companion"
    for result, group in (
        (baseline, "test/baseline-group-v1"),
        (treatment, "test/treatment-group-v1"),
    ):
        result["status"] = "golden"
        result["comparison_group"] = group
        result["declared_intervention_axes"] = ["weights"]
    _add_comparison_definition(
        payload,
        comparison_group="test/baseline-group-v1",
        members=[baseline["result_id"], baseline_companion["result_id"]],
        intervention_axes=["weights"],
    )
    _add_comparison_definition(
        payload,
        comparison_group="test/treatment-group-v1",
        members=[treatment["result_id"], treatment_companion["result_id"]],
        intervention_axes=["weights"],
    )
    payload["results"].append(treatment)
    payload["results"].extend((baseline_companion, treatment_companion))
    payload["tables"][0]["result_ids"].append(treatment["result_id"])
    registry = ResultRegistry.from_json(payload)

    with pytest.raises(IncomparableResultsError, match="comparison_group differs"):
        registry.delta(
            result_id=treatment["result_id"],
            baseline_result_id=baseline["result_id"],
        )


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("task_manifest_sha256", "2" * 64),
        ("training_contract_identity", "different-training-contract"),
        ("training_image_max_pixels", 262_144),
        ("declared_evaluation_image_max_pixels", 262_144),
        ("effective_evaluation_image_max_pixels", 262_144),
        ("runtime_identity", "different-runtime"),
        ("parser_identity", "different-parser"),
        ("action_boundary_identity", "different-action-boundary"),
        ("observation_identity", "different-observation"),
        ("prompt_identity", "different-prompt"),
        ("rng_identity", "different-rng"),
        ("generation_identity", "different-generation"),
        ("scorer_identity", "different-scorer"),
        ("inference_sample_count", 2_240),
        ("scored_sample_count", 2_240),
        ("slice_count", 8),
    ),
)
def test_every_comparison_field_blocks_delta(field: str, replacement: object) -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "mismatched-treatment"
    treatment["label"] = "mismatched treatment"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    treatment["contract"][field] = replacement
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/field-audit-v1",
        intervention_axes=["weights"],
    )
    payload["results"].append(treatment)
    payload["tables"][0]["result_ids"].append("mismatched-treatment")
    registry = ResultRegistry.from_json(payload)

    with pytest.raises(IncomparableResultsError, match=field):
        registry.delta(
            result_id="mismatched-treatment",
            baseline_result_id=baseline["result_id"],
        )


def test_confounded_and_invalid_statuses_block_delta_even_if_contract_matches() -> None:
    for status in ("confounded", "invalid"):
        payload = _minimal_payload()
        baseline = payload["results"][0]
        treatment = deepcopy(baseline)
        treatment["result_id"] = f"{status}-treatment"
        treatment["label"] = status
        treatment["status"] = status
        if status == "invalid":
            treatment["weights"]["state"] = "invalid"
        baseline["status"] = "golden"
        baseline["comparison_group"] = "test/status-audit-v1"
        baseline["declared_intervention_axes"] = ["weights"]
        treatment["comparison_group"] = "test/status-audit-v1"
        treatment["declared_intervention_axes"] = ["weights"]
        _add_comparison_definition(
            payload,
            comparison_group="test/status-audit-v1",
            members=[baseline["result_id"], treatment["result_id"]],
            intervention_axes=["weights"],
        )
        payload["results"].append(treatment)
        payload["tables"][0]["result_ids"].append(treatment["result_id"])
        registry = ResultRegistry.from_json(payload)
        with pytest.raises(IncomparableResultsError, match="status"):
            registry.delta(
                result_id=treatment["result_id"],
                baseline_result_id=baseline["result_id"],
            )


def test_two_rows_cannot_self_promote_without_top_level_preregistration() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "forged-treatment"
    treatment["weights"]["identity"] = "forged checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    for result in (baseline, treatment):
        result["status"] = "golden"
        result["comparison_group"] = "forged/by-result-rows"
        result["declared_intervention_axes"] = ["weights"]
    payload["results"].append(treatment)

    with pytest.raises(RegistryValidationError, match="no preregistered"):
        ResultRegistry.from_json(payload)


def test_golden_result_must_be_named_member_with_group_level_axes() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "registered-treatment"
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/membership-v1",
        intervention_axes=["weights"],
    )
    payload["results"].append(treatment)

    payload["comparison_definitions"][0]["member_result_ids"] = [
        baseline["result_id"],
        "unregistered-result",
    ]
    with pytest.raises(RegistryValidationError, match="unknown results"):
        ResultRegistry.from_json(payload)

    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "registered-treatment"
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/group-axes-v1",
        intervention_axes=["weights"],
    )
    payload["results"].append(treatment)
    treatment["declared_intervention_axes"] = ["prompt_identity"]
    with pytest.raises(RegistryValidationError, match="axes differ"):
        ResultRegistry.from_json(payload)


def test_action_boundary_is_a_preregisterable_treatment_axis() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "strict-action-treatment"
    treatment["contract"]["action_boundary_identity"] = (
        "qwen-native-action-boundary-single-terminal-tool-call-v2"
    )
    treatment["score"]["macro_star_percent"] += 1.0
    for name in treatment["score"]["components_percent"]:
        treatment["score"]["components_percent"][name] += 1.0
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/action-boundary-ablation-v1",
        intervention_axes=["action_boundary_identity"],
    )
    payload["results"].append(treatment)
    registry = ResultRegistry.from_json(payload)

    assert registry.delta(
        result_id=treatment["result_id"],
        baseline_result_id=baseline["result_id"],
    ) == pytest.approx(1.0)


def test_golden_result_requires_content_verifiable_score_summary() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "digest-only-treatment"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/digest-only-v1",
        intervention_axes=["weights"],
    )
    treatment["score_artifact"]["kind"] = "evaluation_run_summary_digest_only"
    payload["results"].append(treatment)

    with pytest.raises(RegistryValidationError, match="content-verifiable"):
        ResultRegistry.from_json(payload)


def test_materialized_mixed_tables_show_no_cross_contract_numeric_delta() -> None:
    registry = load_result_registry(REGISTRY_PATH)
    rendered = registry.render_markdown(
        ("true1m-evidence-inventory", "pixel512-evidence-inventory")
    )

    assert "— (contract differs)" in rendered
    assert "+2.4373 pp" not in rendered
    assert "not A/B" in rendered
    assert "edit the registry, not this table" in rendered
    assert "content-checked" in rendered
    assert "digest-only" in rendered
    assert "does not validate the registered score content" in rendered


def test_canonical_docs_embed_materialized_registry_tables_without_drift() -> None:
    registry = load_result_registry(REGISTRY_PATH)
    report = (
        REPOSITORY_ROOT
        / "docs/NEURIPS_WORKSHOP_TGVF_EXPERIMENT_PLAN_PROGRESS_REPORT_20260826.md"
    ).read_text(encoding="utf-8")
    ledger = (REPOSITORY_ROOT / "docs/EXPERIMENT_LEDGER.md").read_text(encoding="utf-8")

    assert registry.render_table("true1m-evidence-inventory") in report
    pixel512 = registry.render_table("pixel512-evidence-inventory")
    assert pixel512 in report
    assert pixel512 in ledger


def test_artifact_verification_checks_regular_file_and_sha(tmp_path: Path) -> None:
    payload = _minimal_payload()
    artifact_path = tmp_path / "evidence/score.json"
    artifact_path.parent.mkdir()
    result = payload["results"][0]
    artifact_path.write_text(
        json.dumps(_score_summary_payload(result["score"])), encoding="utf-8"
    )
    result["score_artifact"] = {
        "path": "evidence/score.json",
        "sha256": sha256(artifact_path.read_bytes()).hexdigest(),
        "kind": "score_summary",
    }
    registry = ResultRegistry.from_json(payload)
    registry.verify_artifacts(tmp_path)

    artifact_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="SHA-256 differs"):
        registry.verify_artifacts(tmp_path)


def test_artifact_verification_rejects_parent_symlink_escape(tmp_path: Path) -> None:
    payload = _minimal_payload()
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact_path = outside / "score.json"
    result = payload["results"][0]
    artifact_path.write_text(
        json.dumps(_score_summary_payload(result["score"])), encoding="utf-8"
    )
    (repository_root / "evidence").symlink_to(outside, target_is_directory=True)
    result["score_artifact"] = {
        "path": "evidence/score.json",
        "sha256": sha256(artifact_path.read_bytes()).hexdigest(),
        "kind": "score_summary",
    }
    registry = ResultRegistry.from_json(payload)

    with pytest.raises(RegistryValidationError, match="escapes"):
        registry.verify_artifacts(repository_root)


def test_score_summary_hash_match_does_not_hide_score_content_mismatch(
    tmp_path: Path,
) -> None:
    payload = _minimal_payload()
    result = payload["results"][0]
    artifact_score = deepcopy(result["score"])
    artifact_score["macro_star_percent"] += 1.0
    for name in artifact_score["components_percent"]:
        artifact_score["components_percent"][name] += 1.0
    artifact_path = tmp_path / "evidence/score.json"
    artifact_path.parent.mkdir()
    artifact_path.write_text(
        json.dumps(_score_summary_payload(artifact_score)), encoding="utf-8"
    )
    result["score_artifact"] = {
        "path": "evidence/score.json",
        "sha256": sha256(artifact_path.read_bytes()).hexdigest(),
        "kind": "score_summary",
    }
    registry = ResultRegistry.from_json(payload)

    with pytest.raises(RegistryValidationError, match="registered score differs"):
        registry.verify_artifacts(tmp_path)


def test_score_summary_requires_parseable_headline_schema(tmp_path: Path) -> None:
    payload = _minimal_payload()
    artifact_path = tmp_path / "evidence/score.json"
    artifact_path.parent.mkdir()
    artifact_path.write_text('{"status": "complete"}', encoding="utf-8")
    payload["results"][0]["score_artifact"] = {
        "path": "evidence/score.json",
        "sha256": sha256(artifact_path.read_bytes()).hexdigest(),
        "kind": "score_summary",
    }
    registry = ResultRegistry.from_json(payload)

    with pytest.raises(RegistryValidationError, match="headline must be a JSON object"):
        registry.verify_artifacts(tmp_path)


@pytest.mark.parametrize(
    "kind",
    ("evaluation_run_summary_digest_only", "curated_report_digest_only"),
)
def test_explicit_digest_only_artifacts_verify_provenance_not_score_content(
    tmp_path: Path, kind: str
) -> None:
    payload = _minimal_payload()
    artifact_path = tmp_path / "evidence/provenance.txt"
    artifact_path.parent.mkdir()
    artifact_path.write_text("does not contain a registered score\n", encoding="utf-8")
    payload["results"][0]["score_artifact"] = {
        "path": "evidence/provenance.txt",
        "sha256": sha256(artifact_path.read_bytes()).hexdigest(),
        "kind": kind,
    }
    registry = ResultRegistry.from_json(payload)

    registry.verify_artifacts(tmp_path)


def test_preregistration_evidence_hash_and_content_bind_group_definition(
    tmp_path: Path,
) -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "preregistered-treatment"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/preregistered-v1",
        intervention_axes=["weights"],
    )
    payload["results"].append(treatment)
    definition = payload["comparison_definitions"][0]
    preregistration = {
        "schema_version": "tgvf.comparison-preregistration.v1",
        "comparison_group": definition["comparison_group"],
        "member_result_ids": definition["member_result_ids"],
        "intervention_axes": definition["intervention_axes"],
    }
    preregistration_path = tmp_path / definition["preregistration_evidence"]["path"]
    preregistration_path.parent.mkdir(parents=True)
    preregistration_path.write_text(json.dumps(preregistration), encoding="utf-8")
    definition["preregistration_evidence"]["sha256"] = sha256(
        preregistration_path.read_bytes()
    ).hexdigest()
    score_path = tmp_path / baseline["score_artifact"]["path"]
    score_path.parent.mkdir(parents=True)
    score_path.write_text(
        json.dumps(_score_summary_payload(baseline["score"])), encoding="utf-8"
    )
    score_digest = sha256(score_path.read_bytes()).hexdigest()
    baseline["score_artifact"]["sha256"] = score_digest
    treatment["score_artifact"]["sha256"] = score_digest
    treatment["score_artifact"]["path"] = baseline["score_artifact"]["path"]
    registry = ResultRegistry.from_json(payload)
    registry.verify_artifacts(tmp_path)

    preregistration["intervention_axes"] = ["prompt_identity"]
    preregistration_path.write_text(json.dumps(preregistration), encoding="utf-8")
    definition["preregistration_evidence"]["sha256"] = sha256(
        preregistration_path.read_bytes()
    ).hexdigest()
    registry = ResultRegistry.from_json(payload)
    with pytest.raises(RegistryValidationError, match="content differs"):
        registry.verify_artifacts(tmp_path)


def test_materializer_verifies_by_default_and_unsafe_mode_cannot_write(
    tmp_path: Path,
) -> None:
    payload = _minimal_payload()
    result = payload["results"][0]
    artifact_path = tmp_path / "evidence/score.json"
    artifact_path.parent.mkdir()
    artifact_path.write_text(
        json.dumps(_score_summary_payload(result["score"])), encoding="utf-8"
    )
    result["score_artifact"] = {
        "path": "evidence/score.json",
        "sha256": sha256(artifact_path.read_bytes()).hexdigest(),
        "kind": "score_summary",
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    common = [
        sys.executable,
        str(MATERIALIZER),
        "--registry",
        str(registry_path),
        "--artifact-root",
        str(tmp_path),
    ]
    verified = subprocess.run(common, text=True, capture_output=True, check=False)
    assert verified.returncode == 0

    artifact_path.write_text("tampered", encoding="utf-8")
    rejected = subprocess.run(common, text=True, capture_output=True, check=False)
    assert rejected.returncode == 2
    assert "SHA-256 differs" in rejected.stderr

    unsafe = subprocess.run(
        [*common, "--unsafe-skip-artifact-verification"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert unsafe.returncode == 0
    output_path = tmp_path / "must-not-exist.md"
    unsafe_write = subprocess.run(
        [
            *common,
            "--unsafe-skip-artifact-verification",
            "--output",
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert unsafe_write.returncode == 2
    assert not output_path.exists()
