from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from tgvf_rl.representation.experiments.answer_utility.evaluation.semantic_rescore import (
    SEMANTIC_REQUEST_SCHEMA_VERSION,
    _EvaluatedRecord,
    _GenerationSource,
    _blind_queue,
    _canonical_json_bytes,
    _load_generation_source,
    _publish_complete_directory,
    _strict_response,
)
from tgvf_rl.representation.training.oracle_d_utility import (
    OracleDUtilityGroundTruth,
)
from tgvf_rl.representation.training.schema import RepresentationChoice


def _sha(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _source(tmp_path: Path, *, identity_sha: str = "a" * 64) -> _GenerationSource:
    return _GenerationSource(
        root=tmp_path / identity_sha[:4],
        candidate_id="candidate",
        label=f"candidate@{identity_sha[:12]}",
        identity_sha256=identity_sha,
        identity={},
        identity_file_sha256="b" * 64,
        records_file_sha256="c" * 64,
        summary_file_sha256="d" * 64,
        records=(),
    )


def _evaluated(
    tmp_path: Path,
    *,
    identity_sha: str,
    consumer_id: str,
    arm: str,
) -> _EvaluatedRecord:
    payload = {
        "task_kind": "open_vqa",
        "question": "What color is the shirt?",
        "candidate_answer": "It is blue.",
        "reference_answer": "blue",
    }
    record = {
        "sample_id": "sample-1",
        "arm": arm,
        "generated_text": "It is blue.",
        "generation_stop_reason": "natural_stop",
        "score": {"correct": None, "route": "old"},
    }
    return _EvaluatedRecord(
        source=_source(tmp_path, identity_sha=identity_sha),
        source_record=record,
        source_record_sha256=_sha(record),
        truth=OracleDUtilityGroundTruth(
            sample_id="sample-1",
            short_answer="blue",
            choices=(RepresentationChoice("A", "blue"),),
        ),
        original_question=payload["question"],
        deterministic_score={"correct": None, "route": "semantic_unresolved"},
        consumer_id=consumer_id,
        request_payload=payload,
        request_payload_sha256=_sha(payload),
    )


def test_blind_queue_hides_source_arm_and_d_and_deduplicates(tmp_path: Path) -> None:
    records = (
        _evaluated(
            tmp_path,
            identity_sha="1" * 64,
            consumer_id="2" * 64,
            arm="image_only",
        ),
        _evaluated(
            tmp_path,
            identity_sha="3" * 64,
            consumer_id="4" * 64,
            arm="image_correct_D",
        ),
    )

    queue = _blind_queue(records)

    assert len(queue) == 1
    assert queue[0]["schema_version"] == SEMANTIC_REQUEST_SCHEMA_VERSION
    assert queue[0]["consumer_count"] == 2
    assert set(queue[0]["payload"]) == {
        "task_kind",
        "question",
        "candidate_answer",
        "reference_answer",
    }
    assert queue[0]["payload"]["task_kind"] == "open_vqa"
    serialized_payload = json.dumps(queue[0]["payload"])
    assert "image_only" not in serialized_payload
    assert "image_correct_D" not in serialized_payload
    assert "candidate@" not in serialized_payload


def test_strict_response_requires_exact_model_stop_binary_json_and_usage() -> None:
    response = {
        "id": "completion-1",
        "model": "Qwen2.5-72B-Instruct",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {"verdict": 1, "rationale": "Equivalent answer."}
                    )
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    verdict, rationale, usage = _strict_response(
        response, expected_model="Qwen2.5-72B-Instruct"
    )

    assert verdict == 1
    assert rationale == "Equivalent answer."
    assert usage["total_tokens"] == 15
    response["model"] = "another-model"
    with pytest.raises(RuntimeError, match="model differs"):
        _strict_response(response, expected_model="Qwen2.5-72B-Instruct")
    response["model"] = "Qwen2.5-72B-Instruct"
    del response["choices"][0]["index"]
    with pytest.raises(RuntimeError, match="finish with stop"):
        _strict_response(response, expected_model="Qwen2.5-72B-Instruct")


def test_completed_source_loader_locks_identity_records_and_summary(
    tmp_path: Path,
) -> None:
    source_sha = "1" * 64
    data_sha = "2" * 64
    identity = {
        "schema_version": "answer-utility-instruct-evaluation-v2",
        "assistant_dialect": "qwen3-vl-instruct-v1",
        "source_evaluation_config_sha256": source_sha,
        "data_manifest_sha256": data_sha,
        "candidate_id": "E2",
        "arms": ["image_only"],
        "ordered_selected_samples": [
            {"sample_id": "sample-1", "sample_content_sha256": "3" * 64}
        ],
    }
    identity_sha = _sha(identity)
    (tmp_path / "identity.json").write_text(
        json.dumps(
            {
                "schema_version": "representation_oracle_d_utility_v1",
                "identity_sha256": identity_sha,
                "identity": identity,
            }
        ),
        encoding="utf-8",
    )
    record = {
        "schema_version": "answer-utility-instruct-evaluation-record-v2",
        "run_identity_sha256": identity_sha,
        "sample_id": "sample-1",
        "sample_content_sha256": "3" * 64,
        "arm": "image_only",
    }
    records_payload = _canonical_json_bytes(record) + b"\n"
    (tmp_path / "records.jsonl").write_bytes(records_payload)
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "run_identity_sha256": identity_sha,
                "records_jsonl_sha256": sha256(records_payload).hexdigest(),
                "record_count": 1,
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_generation_source(
        tmp_path,
        source_config_sha256=source_sha,
        data_manifest_sha256=data_sha,
    )

    assert loaded.identity_sha256 == identity_sha
    assert loaded.records_file_sha256 == sha256(records_payload).hexdigest()
    assert loaded.label.startswith("E2@")
    (tmp_path / "records.jsonl").write_bytes(records_payload + records_payload)
    with pytest.raises(ValueError, match="duplicate"):
        _load_generation_source(
            tmp_path,
            source_config_sha256=source_sha,
            data_manifest_sha256=data_sha,
        )


def test_complete_directory_publish_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "semantic"

    _publish_complete_directory(output, {"records.jsonl": b"{}\n"})

    assert (output / "records.jsonl").read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError, match="already exists"):
        _publish_complete_directory(output, {"records.jsonl": b"different\n"})
