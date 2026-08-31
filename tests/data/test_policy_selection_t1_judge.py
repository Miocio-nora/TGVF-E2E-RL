from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tgvf_rl.data import policy_selection_t1_judge as judge
from tgvf_rl.data.policy_selection_t1_judge import (
    T1_JUDGE_MANIFEST_SCHEMA,
    _merge_semantic_verdict,
    _strict_response,
)


def test_strict_local_response_requires_model_stop_binary_and_usage() -> None:
    response = {
        "model": "Qwen2.5-72B-Instruct",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": '{"verdict":1,"rationale":"same"}'},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    verdict, rationale, usage = _strict_response(
        response, expected_model="Qwen2.5-72B-Instruct"
    )
    assert verdict == 1
    assert rationale == "same"
    assert usage["cost_usd"] == 0.0

    for changed, message in (
        ({**response, "model": "wrong"}, "model differs"),
        (
            {
                **response,
                "choices": [{**response["choices"][0], "finish_reason": "length"}],
            },
            "finish with stop",
        ),
    ):
        with pytest.raises(RuntimeError, match=message):
            _strict_response(changed, expected_model="Qwen2.5-72B-Instruct")


def test_judge_cli_help_does_not_import_vllm_or_transformers() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools" / "judge_policy_data_selection_t1.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "finalize" in completed.stdout
    assert "publish" in completed.stdout
    assert "run" in completed.stdout


@pytest.mark.parametrize(
    ("source", "verification_route"),
    (
        ("thinklite", "thinklite_semantic_required"),
        ("teacher", "teacher_open_semantic_required"),
    ),
)
def test_merge_semantic_verdict_changes_only_resolution_fields(
    source: str, verification_route: str
) -> None:
    deterministic = {
        "schema_version": "tgvf.policy-selection.attempt.v1",
        "request_id": "request-1",
        "sample_id": "sample-1",
        "candidate_sha256": "a" * 64,
        "source": source,
        "branch": "full_image",
        "attempt_index": 0,
        "run_id": "run-1",
        "run_manifest_sha256": "b" * 64,
        "raw_generation_sha256": "c" * 64,
        "budget_revision": 0,
        "status": "verifier_error",
        "correct": None,
        "answer": "two blue objects",
        "verification_route": verification_route,
        "verification_evidence": "deterministic rules are inconclusive",
        "semantic_required": True,
    }

    resolved = _merge_semantic_verdict(
        deterministic,
        verdict=True,
        evidence_sha256="d" * 64,
    )

    assert "semantic_required" not in resolved
    assert resolved["status"] == "scored"
    assert resolved["correct"] is True
    assert resolved["verification_route"] == "local_qwen25_72b_semantic_judge"
    assert resolved["semantic_judge_evidence_sha256"] == "d" * 64
    for field in (
        "request_id",
        "sample_id",
        "candidate_sha256",
        "raw_generation_sha256",
        "answer",
        "verification_evidence",
    ):
        assert resolved[field] == deterministic[field]


def test_publish_validates_each_completed_pair_once_without_source_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge_config = tmp_path / "judge.json"
    judge_config.write_bytes(b"{}\n")
    judge_config_sha256 = hashlib.sha256(judge_config.read_bytes()).hexdigest()
    run = SimpleNamespace(
        run_id="run-1",
        manifest_sha256="a" * 64,
        output_root=tmp_path / "output",
        selection={"candidates_sha256": "b" * 64},
        verifier={"semantic_judge": {"config_sha256": judge_config_sha256}},
    )
    requests = tuple(
        {
            "judge_request_id": f"judge-{index}",
            "payload_sha256": f"{index:064x}",
            "consumer_count": index + 1,
        }
        for index in range(2)
    )
    scoring_manifest_sha256 = "c" * 64
    observed_verify_flags: list[bool] = []
    validated: list[str] = []

    def load_run(_path: Path, *, verify_data_files: bool) -> SimpleNamespace:
        observed_verify_flags.append(verify_data_files)
        return run

    def load_index(
        _root: Path,
        request: dict[str, object],
        *,
        run_manifest_sha256: str,
        scoring_manifest_sha256: str,
    ) -> dict[str, object]:
        assert run_manifest_sha256 == run.manifest_sha256
        assert scoring_manifest_sha256 == "c" * 64
        request_id = str(request["judge_request_id"])
        validated.append(request_id)
        return {
            "judge_request_id": request_id,
            "evidence_sha256": "d" * 64,
            "verdict": int(request_id.endswith("1")),
        }

    monkeypatch.setattr(judge, "load_t1_run_config", load_run)
    monkeypatch.setattr(judge, "_validate_prepared_output_root", lambda *_: None)
    monkeypatch.setattr(
        judge,
        "_load_scoring_queue",
        lambda _run: ({}, requests, scoring_manifest_sha256),
    )
    monkeypatch.setattr(judge, "_load_completed_index_only", load_index)

    result = judge.publish_t1_semantic_judge_manifest(
        tmp_path / "run.json",
        judge_config_path=judge_config,
    )

    assert observed_verify_flags == [False]
    assert validated == ["judge-0", "judge-1"]
    assert result["request_count"] == 2
    assert result["validation_mode"] == judge.T1_JUDGE_INDEX_ONLY_VALIDATION
    manifest = json.loads(
        (run.output_root / judge.T1_JUDGE_DIRECTORY / "manifest.json").read_bytes()
    )
    assert manifest["schema_version"] == T1_JUDGE_MANIFEST_SCHEMA
    assert manifest["consumer_count"] == 3


def test_index_only_closure_does_not_read_evidence_payload(tmp_path: Path) -> None:
    payload_sha256 = "1" * 64
    evidence_sha256 = "2" * 64
    request = {
        "judge_request_id": f"t1-semantic-judge:{payload_sha256}",
        "payload_sha256": payload_sha256,
    }
    evidence_relative = (
        Path("evidence") / evidence_sha256[:2] / f"{evidence_sha256}.json"
    )
    evidence_path = tmp_path / evidence_relative
    evidence_path.parent.mkdir(parents=True)
    evidence_payload = b"deliberately not JSON"
    evidence_path.write_bytes(evidence_payload)
    index_identity = {
        "schema_version": judge.T1_JUDGE_INDEX_SCHEMA,
        "run_manifest_sha256": "3" * 64,
        "scoring_manifest_sha256": "4" * 64,
        "judge_request_id": request["judge_request_id"],
        "payload_sha256": payload_sha256,
        "evidence_sha256": evidence_sha256,
        "evidence_file": evidence_relative.as_posix(),
        "evidence_file_sha256": hashlib.sha256(evidence_payload).hexdigest(),
        "verdict": 1,
    }
    index = {
        **index_identity,
        "index_sha256": hashlib.sha256(
            judge._canonical_json_bytes(index_identity)
        ).hexdigest(),
    }
    index_path = tmp_path / "requests" / payload_sha256[:2] / f"{payload_sha256}.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_bytes(judge._canonical_json_bytes(index) + b"\n")

    loaded = judge._load_completed_index_only(
        tmp_path,
        request,
        run_manifest_sha256="3" * 64,
        scoring_manifest_sha256="4" * 64,
    )

    assert loaded == index
    with pytest.raises((ValueError, json.JSONDecodeError)):
        judge._load_completed_index(
            tmp_path,
            request,
            run_manifest_sha256="3" * 64,
            scoring_manifest_sha256="4" * 64,
        )


@pytest.mark.parametrize(
    ("source", "verification_route"),
    (
        ("vstar", "vstar_semantic_required"),
        ("teacher", "teacher_open_semantic_required"),
    ),
)
def test_finalize_merges_materialized_attempt_without_generation_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    verification_route: str,
) -> None:
    judge_config = tmp_path / "judge.json"
    judge_config.write_bytes(b"{}\n")
    judge_config_sha256 = hashlib.sha256(judge_config.read_bytes()).hexdigest()
    run = SimpleNamespace(
        run_id="run-1",
        manifest_sha256="a" * 64,
        output_root=tmp_path / "output",
        selection={"candidates_sha256": "b" * 64},
        verifier={"semantic_judge": {"config_sha256": judge_config_sha256}},
    )
    deterministic = {
        "schema_version": "tgvf.policy-selection.attempt.v1",
        "request_id": "request-1",
        "sample_id": "sample-1",
        "candidate_sha256": "c" * 64,
        "source": source,
        "branch": "full_image",
        "attempt_index": 0,
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "raw_generation_sha256": "d" * 64,
        "budget_revision": 0,
        "status": "verifier_error",
        "correct": None,
        "answer": "on the left",
        "verification_route": verification_route,
        "verification_evidence": "deterministic exact rule is inconclusive",
        "semantic_required": True,
    }
    attempts_payload = judge.canonical_json_line(deterministic)
    attempts_path = run.output_root / judge.T1_SCORING_DIRECTORY / "attempts.jsonl"
    attempts_path.parent.mkdir(parents=True)
    attempts_path.write_bytes(attempts_payload)
    scoring_manifest_sha256 = "e" * 64
    scoring_manifest = {
        "files": {
            "attempts": {
                "path": "attempts.jsonl",
                "rows": 1,
                "bytes": len(attempts_payload),
                "sha256": hashlib.sha256(attempts_payload).hexdigest(),
            }
        }
    }
    consumer = {
        field: deterministic[field]
        for field in (
            "request_id",
            "sample_id",
            "candidate_sha256",
            "source",
            "attempt_index",
            "raw_generation_sha256",
        )
    }
    request = {
        "judge_request_id": "judge-1",
        "consumer_count": 1,
        "consumers": [consumer],
    }
    judge._publish_judge_manifest(
        run=run,
        requests=(request,),
        scoring_manifest_sha256=scoring_manifest_sha256,
        judge_config_sha256=judge_config_sha256,
        indices=(
            {
                "judge_request_id": "judge-1",
                "evidence_sha256": "f" * 64,
                "verdict": 1,
            },
        ),
    )

    verify_flags: list[bool] = []
    captured_attempts: list[dict[str, object]] = []

    def load_run(_path: Path, *, verify_data_files: bool) -> SimpleNamespace:
        verify_flags.append(verify_data_files)
        return run

    def reduce(_candidates: object, attempts: object) -> tuple[dict[str, object], ...]:
        captured_attempts.extend(attempts)
        return (
            {
                "schema_version": "tgvf.policy-selection.decision.v1",
                "sample_id": "sample-1",
                "candidate_sha256": "c" * 64,
                "source": source,
                "t1": {},
                "t2": {},
            },
        )

    monkeypatch.setattr(judge, "load_t1_run_config", load_run)
    monkeypatch.setattr(judge, "_validate_prepared_output_root", lambda *_: None)
    monkeypatch.setattr(
        judge,
        "_load_scoring_queue",
        lambda _run: (scoring_manifest, (request,), scoring_manifest_sha256),
    )
    monkeypatch.setattr(
        judge,
        "load_t1_candidates",
        lambda _run: (SimpleNamespace(canonical_record={"candidate": 1}),),
    )
    monkeypatch.setattr(judge, "reduce_selection_attempts", reduce)
    monkeypatch.setattr(
        judge,
        "summarize_selection_decisions",
        lambda _decisions: {"t1_retained_total": 1},
    )

    result = judge.finalize_t1_scoring(
        tmp_path / "run.json",
        judge_config_path=judge_config,
    )

    assert verify_flags == [False]
    assert len(captured_attempts) == 1
    assert captured_attempts[0]["status"] == "scored"
    assert captured_attempts[0]["correct"] is True
    assert captured_attempts[0]["raw_generation_sha256"] == "d" * 64
    assert result["attempt_count"] == 1
