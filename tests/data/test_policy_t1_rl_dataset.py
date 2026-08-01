from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.data import (
    PolicyT1DecisionStage,
    PolicyT1RLDatasetValidationError,
    PolicyT1RLRuntimeBinding,
    SelectionCandidate,
    canonical_json_line,
    load_policy_t1_rl_runtime,
    materialize_policy_t1_arxivqa_rl_dataset,
)


def _candidate(tmp_path: Path, index: int) -> dict[str, object]:
    image_path = (tmp_path / f"image-{index}.bin").resolve()
    image_path.write_bytes(f"image-{index}".encode())
    return {
        "schema_version": "tgvf.policy-selection.candidate.v1",
        "sample_id": f"candidate:arxivqa:{index}",
        "source": "arxivqa",
        "question": f"Question {index}?\nA. no\nB. yes",
        "ground_truth": "B",
        "image": {
            "path": str(image_path),
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "width": 8,
            "height": 8,
        },
        "gt_regions": [],
        "provenance": {"fixture": True, "index": index},
        "selection_metadata": {"option_count": 2},
    }


def _decision(candidate: dict[str, object], decision: str) -> dict[str, object]:
    parsed = SelectionCandidate.from_record(candidate)
    correct = 4 if decision == "retain" else 8
    return {
        "schema_version": "tgvf.policy-selection.decision.v1",
        "sample_id": parsed.sample_id,
        "candidate_sha256": parsed.identity_sha256,
        "source": "arxivqa",
        "t1": {
            "decision": decision,
            "full_image": {
                "accuracy": correct / 8,
                "complete": True,
                "correct_count": correct,
                "expected_attempts": 8,
                "missing_indices": [],
                "observed_attempts": 8,
                "scoreable_attempts": 8,
                "status_counts": {"scored": 8},
            },
            "reason": "fixture",
        },
        "t2": {
            "decision": "not_applicable_preserve_t1",
            "gt_region": None,
            "reason": "fixture",
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json_line(record) for record in records))


def test_materializes_and_loads_provisional_retained_arxivqa(tmp_path: Path) -> None:
    candidates = [_candidate(tmp_path, 0), _candidate(tmp_path, 1)]
    decisions = [
        _decision(candidates[0], "retain"),
        _decision(candidates[1], "exclude_too_easy"),
    ]
    candidate_path = tmp_path / "candidates.jsonl"
    decision_path = tmp_path / "provisional-decisions.jsonl"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(decision_path, decisions)

    result = materialize_policy_t1_arxivqa_rl_dataset(
        candidate_path,
        decision_path,
        tmp_path / "artifact",
        decision_stage=PolicyT1DecisionStage.PROVISIONAL,
        shuffle_seed=42,
    )
    assert result.sample_count == 1
    row = json.loads((result.output_root / "samples.jsonl").read_text())
    assert row["sample_id"] == candidates[0]["sample_id"]
    assert row["data_source"] == "arxivqa"
    assert row["task_kind"] == "mcq"
    assert row["selection"]["decision_stage"] == "provisional"
    assert row["selection"]["t1"]["decision"] == "retain"

    binding = PolicyT1RLRuntimeBinding(
        manifest_file_sha256=result.manifest_file_sha256,
        content_sha256=result.content_sha256,
        shuffle_seed=42,
        decision_stage=PolicyT1DecisionStage.PROVISIONAL,
        expected_sample_count=1,
    )
    runtime = load_policy_t1_rl_runtime(result.output_root, binding=binding)
    assert runtime.iteration_identity_sha256 == result.iteration_identity_sha256
    assert runtime[0].question == candidates[0]["question"]
    assert runtime[0].ground_truth == "B"


def test_materializer_rejects_decision_candidate_identity_drift(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, 0)
    decision = _decision(candidate, "retain")
    decision["candidate_sha256"] = "0" * 64
    candidate_path = tmp_path / "candidates.jsonl"
    decision_path = tmp_path / "decisions.jsonl"
    _write_jsonl(candidate_path, [candidate])
    _write_jsonl(decision_path, [decision])

    with pytest.raises(
        PolicyT1RLDatasetValidationError, match="candidate identity differs"
    ):
        materialize_policy_t1_arxivqa_rl_dataset(
            candidate_path,
            decision_path,
            tmp_path / "artifact",
            decision_stage=PolicyT1DecisionStage.FINAL,
        )
