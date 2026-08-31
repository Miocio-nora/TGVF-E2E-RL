from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from tgvf_rl.data import policy_selection_t1_scoring as scoring
from tgvf_rl.data.policy_selection import SelectionSource


def test_scoring_module_and_cli_help_are_cpu_only() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    repository_root = Path(__file__).resolve().parents[2]
    script = f"""
import sys
sys.path.insert(0, {str(source_root)!r})
import tgvf_rl.data.policy_selection_t1_scoring
for name in ('torch', 'vllm', 'transformers', 'PIL'):
    assert name not in sys.modules, (name, sorted(sys.modules))
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools" / "score_policy_data_selection_t1.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--judge-config" in completed.stdout
    assert "--quality-exclusions" in completed.stdout


def test_thinklite_semantic_queue_uses_sample_level_task_kind() -> None:
    run = SimpleNamespace(
        run_id="fixture-run",
        manifest_sha256="1" * 64,
        verifier={
            "semantic_judge": {
                "prompt_sha256": "2" * 64,
                "repository": "judge/repository",
                "served_name": "judge-model",
            }
        },
    )

    def queued(question: str, ground_truth: str) -> str:
        candidate = SimpleNamespace(
            identity_sha256="3" * 64,
            sample_id="thinklite-sample",
            source=SelectionSource.THINKLITE,
            question=question,
            ground_truth=ground_truth,
        )
        evidence = SimpleNamespace(
            request_id="request-1",
            attempt_index=0,
            evidence_sha256="4" * 64,
        )
        records = scoring._judge_queue(
            (
                {
                    "request_id": evidence.request_id,
                    "status": "verifier_error",
                    "semantic_required": True,
                    "answer": "candidate answer",
                },
            ),
            ((candidate, evidence, False),),
            run=run,
            judge_config_sha256="5" * 64,
        )
        return records[0]["task_kind"]

    assert queued("What establishment is serving this food?", "food truck") == (
        "open_vqa"
    )
    assert queued(r"Find \\angle ABC.", "45 degrees") == "math"


def test_latest_length_is_terminal_truncated_without_requiring_retry(
    monkeypatch,
) -> None:
    candidate = SimpleNamespace(
        identity_sha256="a" * 64,
        sample_id="sample-1",
        source=SelectionSource.ARXIVQA,
    )
    evidence = SimpleNamespace(
        request_id="request-1",
        candidate_sha256=candidate.identity_sha256,
        sample_id=candidate.sample_id,
        attempt_index=0,
        source=candidate.source,
        budget_revision=0,
        finish_reason="length",
        evidence_sha256="b" * 64,
        sampled_token_count=40_960,
    )
    manifest = SimpleNamespace(manifest_sha256="c" * 64)
    run = SimpleNamespace(
        run_id="T1-04-QWEN3-INSTRUCT-512-FULLIMAGE-271842-GPU0123",
        response_budgets=tuple(SimpleNamespace(revision=index) for index in range(3)),
    )
    monkeypatch.setattr(
        scoring,
        "_expected_requests",
        lambda _candidates: {evidence.request_id: (candidate, 0)},
    )
    monkeypatch.setattr(
        scoring,
        "_load_validated_evidence",
        lambda _run: ((manifest, evidence),),
    )

    selected, pointers = scoring._effective_generations(run, (candidate,))

    assert selected == ((candidate, evidence, True),)
    assert pointers[0]["selected_budget_revision"] == 0
    assert pointers[0]["finish_reason"] == "length"
    assert pointers[0]["budget_exhausted"] is True


def test_unretried_length_waiver_is_scoped_to_t1_04(monkeypatch) -> None:
    candidate = SimpleNamespace(
        identity_sha256="a" * 64,
        sample_id="sample-1",
        source=SelectionSource.ARXIVQA,
    )
    evidence = SimpleNamespace(
        request_id="request-1",
        candidate_sha256=candidate.identity_sha256,
        sample_id=candidate.sample_id,
        attempt_index=0,
        source=candidate.source,
        budget_revision=0,
        finish_reason="length",
        evidence_sha256="b" * 64,
        sampled_token_count=40_960,
    )
    manifest = SimpleNamespace(manifest_sha256="c" * 64)
    run = SimpleNamespace(
        run_id="ANOTHER-T1-RUN",
        response_budgets=tuple(SimpleNamespace(revision=index) for index in range(3)),
    )
    monkeypatch.setattr(
        scoring,
        "_expected_requests",
        lambda _candidates: {evidence.request_id: (candidate, 0)},
    )
    monkeypatch.setattr(
        scoring,
        "_load_validated_evidence",
        lambda _run: ((manifest, evidence),),
    )

    try:
        scoring._effective_generations(run, (candidate,))
    except ValueError as error:
        assert str(error) == "a length finish still requires a response-budget retry"
    else:
        raise AssertionError("unrelated T1 run unexpectedly inherited the waiver")


def test_completed_retry_still_selects_latest_stop_generation(monkeypatch) -> None:
    candidate = SimpleNamespace(
        identity_sha256="a" * 64,
        sample_id="sample-1",
        source=SelectionSource.ARXIVQA,
    )
    common = {
        "request_id": "request-1",
        "candidate_sha256": candidate.identity_sha256,
        "sample_id": candidate.sample_id,
        "attempt_index": 0,
        "source": candidate.source,
    }
    revision_0 = SimpleNamespace(
        **common,
        budget_revision=0,
        finish_reason="length",
        evidence_sha256="b" * 64,
        sampled_token_count=40_960,
    )
    revision_1 = SimpleNamespace(
        **common,
        budget_revision=1,
        finish_reason="stop",
        evidence_sha256="d" * 64,
        sampled_token_count=41_200,
    )
    manifest_0 = SimpleNamespace(manifest_sha256="c" * 64)
    manifest_1 = SimpleNamespace(manifest_sha256="e" * 64)
    run = SimpleNamespace(
        run_id="T1-04-QWEN3-INSTRUCT-512-FULLIMAGE-271842-GPU0123",
        response_budgets=tuple(SimpleNamespace(revision=index) for index in range(3)),
    )
    monkeypatch.setattr(
        scoring,
        "_expected_requests",
        lambda _candidates: {revision_0.request_id: (candidate, 0)},
    )
    monkeypatch.setattr(
        scoring,
        "_load_validated_evidence",
        lambda _run: ((manifest_0, revision_0), (manifest_1, revision_1)),
    )
    validated: list[tuple[object, object]] = []
    monkeypatch.setattr(
        scoring,
        "validate_length_retry_identity",
        lambda previous, current: validated.append((previous, current)),
    )

    selected, pointers = scoring._effective_generations(run, (candidate,))

    assert validated == [(revision_0, revision_1)]
    assert selected == ((candidate, revision_1, False),)
    assert pointers[0]["selected_budget_revision"] == 1
    assert pointers[0]["finish_reason"] == "stop"
    assert pointers[0]["budget_exhausted"] is False


def _pathological_teacher_fixture() -> tuple[
    object,
    object,
    object,
    object,
    dict[str, object],
    dict[str, object],
]:
    candidate_record = {
        "schema_version": "tgvf.policy-selection.candidate.v1",
        "sample_id": "teacher-pathological-sample",
        "source": "teacher",
        "question": "What is shown?",
        "ground_truth": "answer",
        "image": {"sha256": "9" * 64, "width": 16, "height": 16},
        "gt_regions": [],
        "provenance": {"source_uid": "teacher-pathological-source"},
        "selection_metadata": {"task_kind": "open", "answer_format": "open"},
    }
    candidate = scoring.SelectionCandidate.from_record(candidate_record)
    run = SimpleNamespace(
        run_id="T1-TEACHER-PATHOLOGICAL-FIXTURE",
        manifest_sha256="1" * 64,
        selection={"kind": "teacher_full"},
        response_budgets=tuple(SimpleNamespace(revision=index) for index in range(3)),
    )
    request_id = scoring.stable_selection_request_id(
        candidate_sha256=candidate.identity_sha256,
        branch=scoring.SelectionBranch.FULL_IMAGE,
        attempt_index=0,
    )
    common = {
        "request_id": request_id,
        "candidate_sha256": candidate.identity_sha256,
        "sample_id": candidate.sample_id,
        "attempt_index": 0,
        "source": candidate.source,
    }
    revision_0 = SimpleNamespace(
        **common,
        budget_revision=0,
        finish_reason="length",
        evidence_sha256="2" * 64,
        sampled_token_count=40_960,
    )
    revision_1 = SimpleNamespace(
        **common,
        budget_revision=1,
        finish_reason="length",
        evidence_sha256="3" * 64,
        sampled_token_count=98_304,
    )
    waiver = {
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "request_id": request_id,
        "sample_id": candidate.sample_id,
        "candidate_sha256": candidate.identity_sha256,
        "source": "teacher",
        "attempt_index": 0,
        "terminal_budget_revision": 1,
        "terminal_finish_reason": "length",
        "terminal_evidence_sha256": revision_1.evidence_sha256,
        "terminal_sampled_token_count": revision_1.sampled_token_count,
    }
    exclusion = {
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "candidate_sha256": candidate.identity_sha256,
        "sample_id": candidate.sample_id,
        "source": "teacher",
        "reason": "pathological_generation_length",
        "question": candidate.question,
        "ground_truth": candidate.ground_truth,
        "generation_length_waivers": [waiver],
    }
    return run, candidate, revision_0, revision_1, waiver, exclusion


def test_pathological_rev1_length_waiver_is_exact_and_consumed(monkeypatch) -> None:
    run, candidate, revision_0, revision_1, waiver, exclusion = (
        _pathological_teacher_fixture()
    )
    quality, waivers, records = scoring._parse_quality_exclusions(
        {
            "schema_version": "tgvf.policy-selection.t1-quality-exclusions.v1",
            "exclusions": [exclusion],
        },
        run=run,
        candidates_by_sha={candidate.identity_sha256: candidate},
    )
    assert quality[candidate.identity_sha256] == exclusion
    assert waivers == {revision_1.request_id: waiver}
    assert records == (waiver,)
    binding = scoring._pathological_waiver_binding(records)
    assert binding["pathological_generation_length_waiver_count"] == 1
    assert len(binding["pathological_generation_length_waivers_sha256"]) == 64

    monkeypatch.setattr(
        scoring,
        "_expected_requests",
        lambda _candidates: {revision_1.request_id: (candidate, 0)},
    )
    manifests = (
        SimpleNamespace(manifest_sha256="4" * 64),
        SimpleNamespace(manifest_sha256="5" * 64),
    )
    monkeypatch.setattr(
        scoring,
        "_load_validated_evidence",
        lambda _run: tuple(zip(manifests, (revision_0, revision_1), strict=True)),
    )
    monkeypatch.setattr(
        scoring, "validate_length_retry_identity", lambda _previous, _current: None
    )

    selected, pointers = scoring._effective_generations(
        run,
        (candidate,),
        pathological_length_waivers=waivers,
    )

    assert selected == ((candidate, revision_1, True),)
    assert pointers[0]["selected_budget_revision"] == 1
    assert pointers[0]["raw_generation_sha256"] == revision_1.evidence_sha256
    assert pointers[0]["budget_exhausted"] is True


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("request_id", "wrong-request"),
        ("candidate_sha256", "a" * 64),
        ("attempt_index", 1),
        ("terminal_budget_revision", 0),
        ("terminal_evidence_sha256", "b" * 64),
        ("terminal_sampled_token_count", 98_303),
    ),
)
def test_pathological_length_waiver_rejects_identity_drift(
    field: str, wrong_value: object
) -> None:
    run, candidate, _revision_0, revision_1, waiver, _exclusion = (
        _pathological_teacher_fixture()
    )
    changed = {**waiver, field: wrong_value}

    with pytest.raises(
        ValueError, match="pathological-generation waiver evidence identity differs"
    ):
        scoring._validate_pathological_length_waiver(
            changed,
            run=run,
            candidate=candidate,
            evidence=revision_1,
            terminal_revision=1,
        )


def test_pathological_length_waiver_must_be_consumed(monkeypatch) -> None:
    run, candidate, revision_0, revision_1, waiver, _exclusion = (
        _pathological_teacher_fixture()
    )
    completed = SimpleNamespace(
        **{
            **vars(revision_1),
            "finish_reason": "stop",
            "sampled_token_count": 107,
            "evidence_sha256": "6" * 64,
        }
    )
    monkeypatch.setattr(
        scoring,
        "_expected_requests",
        lambda _candidates: {revision_1.request_id: (candidate, 0)},
    )
    manifests = (
        SimpleNamespace(manifest_sha256="4" * 64),
        SimpleNamespace(manifest_sha256="5" * 64),
    )
    monkeypatch.setattr(
        scoring,
        "_load_validated_evidence",
        lambda _run: tuple(zip(manifests, (revision_0, completed), strict=True)),
    )
    monkeypatch.setattr(
        scoring, "validate_length_retry_identity", lambda _previous, _current: None
    )

    with pytest.raises(ValueError, match="request waiver was not consumed"):
        scoring._effective_generations(
            run,
            (candidate,),
            pathological_length_waivers={revision_1.request_id: waiver},
        )


def test_pathological_candidate_without_pending_request_has_no_waiver() -> None:
    run, candidate, _revision_0, _revision_1, _waiver, exclusion = (
        _pathological_teacher_fixture()
    )
    exclusion = {**exclusion, "generation_length_waivers": []}

    quality, waivers, records = scoring._parse_quality_exclusions(
        {
            "schema_version": "tgvf.policy-selection.t1-quality-exclusions.v1",
            "exclusions": [exclusion],
        },
        run=run,
        candidates_by_sha={candidate.identity_sha256: candidate},
    )

    assert quality[candidate.identity_sha256] == exclusion
    assert waivers == {}
    assert records == ()


def test_pathological_candidate_marks_all_attempts_unresolved() -> None:
    _run, candidate, _revision_0, _revision_1, _waiver, exclusion = (
        _pathological_teacher_fixture()
    )
    requests = scoring._expected_requests((candidate,))
    attempts = []
    for request_id, (_candidate, attempt_index) in requests.items():
        attempts.append(
            scoring._apply_quality_exclusion(
                {
                    "schema_version": "tgvf.policy-selection.attempt.v1",
                    "request_id": request_id,
                    "sample_id": candidate.sample_id,
                    "candidate_sha256": candidate.identity_sha256,
                    "source": "teacher",
                    "branch": "full_image",
                    "attempt_index": attempt_index,
                    "status": "scored",
                    "correct": True,
                    "verification_route": "teacher_open_normalized_exact",
                },
                exclusion,
            )
        )

    assert {item["verification_route"] for item in attempts} == {
        "source_generation_anomaly_invalid"
    }
    assert all(item["correct"] is None for item in attempts)
    decision = scoring.reduce_selection_attempts(
        (candidate.canonical_record,), attempts
    )[0]
    assert decision["t1"]["decision"] == "unresolved"
    assert decision["t1"]["full_image"]["status_counts"] == {
        "verifier_error": 8
    }


def test_materialized_report_and_manifest_bind_pathological_waivers(
    tmp_path: Path, monkeypatch
) -> None:
    run, candidate, _revision_0, _revision_1, waiver, exclusion = (
        _pathological_teacher_fixture()
    )
    prompt_sha256 = "7" * 64
    judge_payload = scoring._canonical_json_bytes(
        {"prompt": {"sha256": prompt_sha256}}
    ) + b"\n"
    judge_path = tmp_path / "judge.json"
    judge_path.write_bytes(judge_payload)
    quality_payload = scoring._canonical_json_bytes(
        {
            "schema_version": "tgvf.policy-selection.t1-quality-exclusions.v1",
            "exclusions": [exclusion],
        }
    ) + b"\n"
    quality_path = tmp_path / "quality.json"
    quality_path.write_bytes(quality_payload)
    run.output_root = tmp_path / "output"
    run.selection["candidates_sha256"] = "8" * 64
    run.verifier = {
        "answer_parser": "direct-completion-v1",
        "semantic_judge": {
            "config_sha256": scoring._sha256_bytes(judge_payload),
            "prompt_sha256": prompt_sha256,
            "repository": "judge/repository",
            "served_name": "judge-model",
        },
    }
    requests = scoring._expected_requests((candidate,))
    evidences = tuple(
        SimpleNamespace(
            request_id=request_id,
            attempt_index=attempt_index,
            evidence_sha256=f"{attempt_index + 1:064x}",
        )
        for request_id, (_candidate, attempt_index) in requests.items()
    )
    observed_waivers: list[dict[str, object]] = []

    def effective(_run, _candidates, *, pathological_length_waivers):
        observed_waivers.extend(pathological_length_waivers.values())
        return (
            tuple((candidate, evidence, False) for evidence in evidences),
            tuple(
                {"request_id": evidence.request_id, "attempt_index": evidence.attempt_index}
                for evidence in evidences
            ),
        )

    def attempt_record(evidence, **_kwargs):
        return {
            "schema_version": "tgvf.policy-selection.attempt.v1",
            "request_id": evidence.request_id,
            "sample_id": candidate.sample_id,
            "candidate_sha256": candidate.identity_sha256,
            "source": "teacher",
            "branch": "full_image",
            "attempt_index": evidence.attempt_index,
            "raw_generation_sha256": evidence.evidence_sha256,
            "status": "scored",
            "correct": True,
            "answer": "answer",
            "verification_route": "teacher_open_normalized_exact",
        }

    captured_attempts: list[dict[str, object]] = []

    def reduce(_candidates, attempts):
        captured_attempts.extend(attempts)
        return (
            {
                "schema_version": "tgvf.policy-selection.decision.v1",
                "sample_id": candidate.sample_id,
                "candidate_sha256": candidate.identity_sha256,
                "source": "teacher",
                "t1": {"decision": "unresolved"},
                "t2": {},
            },
        )

    monkeypatch.setattr(scoring, "load_t1_run_config", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(scoring, "_validate_prepared_output_root", lambda *_: None)
    monkeypatch.setattr(scoring, "load_t1_candidates", lambda _run: (candidate,))
    monkeypatch.setattr(scoring, "_effective_generations", effective)
    monkeypatch.setattr(scoring, "evidence_to_attempt_record", attempt_record)
    monkeypatch.setattr(scoring, "reduce_selection_attempts", reduce)
    monkeypatch.setattr(
        scoring,
        "summarize_selection_decisions",
        lambda _decisions: {"t1_retained_total": 0},
    )

    result = scoring.materialize_t1_deterministic_scoring(
        tmp_path / "run.json",
        judge_config_path=judge_path,
        quality_exclusions_path=quality_path,
    )

    expected_binding = scoring._pathological_waiver_binding((waiver,))
    assert observed_waivers == [waiver]
    assert result["pathological_generation_exclusion_candidate_count"] == 1
    assert result["pathological_generation_exclusion_attempt_count"] == 8
    assert result["quality_exclusion_attempt_count"] == 8
    assert all(
        attempt["verification_route"] == "source_generation_anomaly_invalid"
        for attempt in captured_attempts
    )
    for field, value in expected_binding.items():
        assert result[field] == value
    manifest = json.loads(
        (run.output_root / scoring.T1_SCORING_DIRECTORY / "manifest.json").read_bytes()
    )
    for field, value in expected_binding.items():
        assert manifest[field] == value
