from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from tgvf_rl.data.policy_selection_t1_judge import (
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


def test_judge_cli_exposes_production_commands() -> None:
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


def test_merge_teacher_semantic_verdict_preserves_bound_identity() -> None:
    deterministic = {
        "schema_version": "tgvf.policy-selection.attempt.v1",
        "request_id": "request-1",
        "sample_id": "teacher-1",
        "candidate_sha256": "a" * 64,
        "source": "teacher",
        "branch": "full_image",
        "attempt_index": 0,
        "run_id": "run-1",
        "run_manifest_sha256": "b" * 64,
        "raw_generation_sha256": "c" * 64,
        "budget_revision": 0,
        "status": "verifier_error",
        "correct": None,
        "answer": "two blue objects",
        "verification_route": "teacher_open_semantic_required",
        "verification_evidence": "deterministic rules are inconclusive",
        "semantic_required": True,
        "semantic_judge_evidence_sha256": None,
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
    assert resolved["raw_generation_sha256"] == deterministic["raw_generation_sha256"]
