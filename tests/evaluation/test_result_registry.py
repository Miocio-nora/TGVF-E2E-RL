from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

import tgvf_rl.secure_file_read as secure_file_read
from tgvf_rl.evaluation.result_registry import (
    ComparisonDefinition,
    IncomparableResultsError,
    RegistryValidationError,
    ResultRecord,
    ResultRegistry,
    ResultStatus,
    _validate_comparison_contract,
    load_result_registry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPOSITORY_ROOT / "evidence/policy/result_registry_v2.json"


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


def _comparison_parts(
    payload: dict[str, object],
    *,
    baseline: dict[str, object],
    treatment: dict[str, object],
    comparison_group: str,
    intervention_axes: list[str],
) -> tuple[ResultRecord, ResultRecord, ComparisonDefinition]:
    for result in (baseline, treatment):
        result["comparison_group"] = comparison_group
        result["declared_intervention_axes"] = sorted(intervention_axes)
    _add_comparison_definition(
        payload,
        comparison_group=comparison_group,
        members=[baseline["result_id"], treatment["result_id"]],
        intervention_axes=intervention_axes,
    )
    return (
        ResultRecord.from_json(treatment, index=1),
        ResultRecord.from_json(baseline, index=0),
        ComparisonDefinition.from_json(payload["comparison_definitions"][0], index=0),
    )


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


def test_result_registry_declares_comparison_definitions_once() -> None:
    source = REPOSITORY_ROOT / "src/tgvf_rl/evaluation/result_registry.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    registry_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResultRegistry"
    )
    declarations = [
        node
        for node in registry_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "comparison_definitions"
    ]
    assert len(declarations) == 1


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


def test_v2_rejects_golden_promotion_even_with_preregistration() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "candidate-treatment"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/candidate-v1",
        intervention_axes=["weights"],
    )
    payload["results"].append(treatment)
    payload["tables"][0]["result_ids"].append(treatment["result_id"])

    with pytest.raises(
        RegistryValidationError,
        match="v2 cannot promote golden results.*trajectory-set identity.*weights",
    ):
        ResultRecord.from_json(baseline, index=0)
    with pytest.raises(
        RegistryValidationError,
        match="v2 cannot promote golden results.*trajectory-set identity.*weights",
    ):
        ResultRegistry.from_json(payload)


def test_v2_delta_api_is_closed_for_standalone_rows() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "standalone-treatment"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    payload["results"].append(treatment)
    registry = ResultRegistry.from_json(payload)

    with pytest.raises(
        IncomparableResultsError,
        match="v2 cannot promote golden results.*provenance-receipt schema",
    ):
        registry.delta(
            result_id=treatment["result_id"],
            baseline_result_id=baseline["result_id"],
        )


def test_shared_score_artifact_cannot_promote_different_weights(
    tmp_path: Path,
) -> None:
    """A score-content hash is not an evaluation-provenance receipt."""

    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "different-weights-same-score-file"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    payload["results"].append(treatment)

    artifact_path = tmp_path / "evidence/score.json"
    artifact_path.parent.mkdir()
    artifact_path.write_text(
        json.dumps(_score_summary_payload(baseline["score"])), encoding="utf-8"
    )
    artifact = {
        "path": "evidence/score.json",
        "sha256": sha256(artifact_path.read_bytes()).hexdigest(),
        "kind": "score_summary",
    }
    baseline["score_artifact"] = deepcopy(artifact)
    treatment["score_artifact"] = deepcopy(artifact)

    # V2 can verify the shared file's bytes and score content for inventory
    # rows.  That deliberately does not establish which weights produced it.
    registry = ResultRegistry.from_json(payload)
    registry.verify_artifacts(tmp_path)
    with pytest.raises(IncomparableResultsError, match="trajectory-set identity"):
        registry.delta(
            result_id=treatment["result_id"],
            baseline_result_id=baseline["result_id"],
        )

    _promote_pair(
        payload,
        baseline,
        treatment,
        comparison_group="test/shared-score-v1",
        intervention_axes=["weights"],
    )
    with pytest.raises(RegistryValidationError, match="v2 cannot promote golden"):
        ResultRegistry.from_json(payload)


def test_comparison_definition_still_validates_members_and_axes() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "standalone-treatment"
    payload["results"].append(treatment)
    _add_comparison_definition(
        payload,
        comparison_group="test/standalone-diagnostic-v1",
        members=[baseline["result_id"], treatment["result_id"]],
        intervention_axes=["action_boundary_identity"],
    )
    registry = ResultRegistry.from_json(payload)
    definition = registry.comparison_definition("test/standalone-diagnostic-v1")
    assert definition.intervention_axes == ("action_boundary_identity",)

    payload["comparison_definitions"][0]["member_result_ids"][1] = "unknown"
    with pytest.raises(RegistryValidationError, match="unknown results"):
        ResultRegistry.from_json(payload)


def test_private_comparison_validator_accepts_exact_declared_axis() -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "different-weights"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    result_record, baseline_record, definition = _comparison_parts(
        payload,
        baseline=baseline,
        treatment=treatment,
        comparison_group="test/weights-contract-diagnostic-v1",
        intervention_axes=["weights"],
    )

    assert (
        _validate_comparison_contract(
            result=result_record,
            baseline=baseline_record,
            definition=definition,
        )
        is None
    )


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("task_manifest_sha256", "2" * 64),
        ("rng_identity", "different-rng"),
        ("scorer_identity", "different-scorer"),
        ("inference_sample_count", 2_240),
        ("scored_sample_count", 2_240),
        ("slice_count", 8),
    ),
)
def test_private_comparison_validator_rejects_every_invariant_drift(
    field: str,
    replacement: object,
) -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "invariant-drift"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    treatment["contract"][field] = replacement
    result_record, baseline_record, definition = _comparison_parts(
        payload,
        baseline=baseline,
        treatment=treatment,
        comparison_group="test/invariant-diagnostic-v1",
        intervention_axes=["weights"],
    )

    with pytest.raises(IncomparableResultsError, match=field):
        _validate_comparison_contract(
            result=result_record,
            baseline=baseline_record,
            definition=definition,
        )


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("training_contract_identity", "different-training-contract"),
        ("training_image_max_pixels", 262_144),
        ("declared_evaluation_image_max_pixels", 262_144),
        ("effective_evaluation_image_max_pixels", 262_144),
        ("runtime_identity", "different-runtime"),
        ("parser_identity", "different-parser"),
        ("action_boundary_identity", "different-action-boundary"),
        ("observation_identity", "different-observation"),
        ("prompt_identity", "different-prompt"),
        ("generation_identity", "different-generation"),
    ),
)
def test_private_comparison_validator_rejects_undeclared_treatment_axis(
    field: str,
    replacement: object,
) -> None:
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "undeclared-treatment"
    treatment["weights"]["identity"] = "different checkpoint"
    treatment["weights"]["sha256"] = "1" * 64
    treatment["contract"][field] = replacement
    result_record, baseline_record, definition = _comparison_parts(
        payload,
        baseline=baseline,
        treatment=treatment,
        comparison_group="test/undeclared-axis-diagnostic-v1",
        intervention_axes=["weights"],
    )
    with pytest.raises(IncomparableResultsError, match=field):
        _validate_comparison_contract(
            result=result_record,
            baseline=baseline_record,
            definition=definition,
        )


def test_private_comparison_validator_rejects_declared_axis_that_did_not_change() -> (
    None
):
    payload = _minimal_payload()
    baseline = payload["results"][0]
    treatment = deepcopy(baseline)
    treatment["result_id"] = "missing-runtime-axis"
    treatment["contract"]["prompt_identity"] = "different-prompt"
    result_record, baseline_record, definition = _comparison_parts(
        payload,
        baseline=baseline,
        treatment=treatment,
        comparison_group="test/missing-axis-diagnostic-v1",
        intervention_axes=["prompt_identity", "runtime_identity"],
    )
    with pytest.raises(IncomparableResultsError, match="runtime_identity"):
        _validate_comparison_contract(
            result=result_record,
            baseline=baseline_record,
            definition=definition,
        )


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


def test_artifact_verification_rejects_repository_root_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _minimal_payload()
    repository_root = tmp_path / "repository"
    legitimate_evidence = repository_root / "evidence"
    legitimate_evidence.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside_evidence = outside / "evidence"
    outside_evidence.mkdir(parents=True)
    result = payload["results"][0]
    artifact_payload = json.dumps(_score_summary_payload(result["score"]))
    for directory in (legitimate_evidence, outside_evidence):
        (directory / "score.json").write_text(artifact_payload, encoding="utf-8")
    result["score_artifact"] = {
        "path": "evidence/score.json",
        "sha256": sha256(artifact_payload.encode()).hexdigest(),
        "kind": "score_summary",
    }
    registry = ResultRegistry.from_json(payload)
    archived_root = tmp_path / "repository-before-race"
    original_open = secure_file_read._open_path
    swapped = False

    def _swap_root_before_open(
        path: str,
        flags: int,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == repository_root.name and dir_fd is not None and not swapped:
            swapped = True
            repository_root.rename(archived_root)
            repository_root.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, dir_fd=dir_fd)

    monkeypatch.setattr(secure_file_read, "_open_path", _swap_root_before_open)

    with pytest.raises(RegistryValidationError, match="symlinks"):
        registry.verify_artifacts(repository_root)
    assert swapped


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
    for result in (baseline, treatment):
        result["comparison_group"] = "test/preregistered-v1"
        result["declared_intervention_axes"] = ["weights"]
    _add_comparison_definition(
        payload,
        comparison_group="test/preregistered-v1",
        members=[baseline["result_id"], treatment["result_id"]],
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


def test_verified_registry_api_renders_only_after_artifact_check(
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
    registry = ResultRegistry.from_json(payload)
    registry.verify_artifacts(tmp_path)
    assert "Original raw-direct true-1M" in registry.render_markdown()

    artifact_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="SHA-256 differs"):
        registry.verify_artifacts(tmp_path)
