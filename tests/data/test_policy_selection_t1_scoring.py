from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

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
