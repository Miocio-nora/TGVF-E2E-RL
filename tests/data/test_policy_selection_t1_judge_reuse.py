from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

import pytest

from tgvf_rl.data.policy_selection import POLICY_SELECTION_TASK_KIND_POLICY
from tgvf_rl.data.policy_selection_t1_judge import (
    T1_JUDGE_EVIDENCE_SCHEMA,
    T1_JUDGE_INDEX_SCHEMA,
    _canonical_json_bytes,
    _identity_record,
    _load_completed_index,
    _request_payload,
    _sha256_bytes,
    _strict_response,
)
from tgvf_rl.data.policy_selection_t1_judge_reuse import (
    T1_JUDGE_REUSE_PROVENANCE_SCHEMA,
    T1_LEGACY_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
    T1_LEGACY_JUDGE_EVIDENCE_SCHEMA,
    T1_LEGACY_JUDGE_INDEX_SCHEMA,
    _RunIdentity,
    _reuse_legacy_judge_results,
)
import tgvf_rl.data.policy_selection_t1_judge_reuse as judge_reuse
from tgvf_rl.data.policy_selection_t1_scoring import (
    T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
    T1_SEMANTIC_JUDGE_REQUEST_SCHEMA,
)
from tgvf_rl.judges.openai_compatible import load_openai_compatible_judge


@dataclass(frozen=True)
class _Fixture:
    identity: _RunIdentity
    bound: Any
    source_scoring_root: Path
    source_judge_root: Path
    target_scoring_root: Path
    target_judge_root: Path
    source_request: dict[str, Any]
    target_request: dict[str, Any]
    source_scoring_manifest_sha256: str
    target_scoring_manifest_sha256: str


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _request(
    identity: _RunIdentity,
    *,
    task_kind: str,
    consumers: int,
    question: str = "What color is the car?",
    consumer_namespace: str = "default",
) -> dict[str, Any]:
    payload = {
        "task_kind": task_kind,
        "question": question,
        "candidate_answer": "The car is red.",
        "reference_answer": "red",
    }
    payload_sha256 = _sha256_bytes(_canonical_json_bytes(payload))
    consumer_records = [
        {
            "request_id": f"rollout-{consumer_namespace}-{index}",
            "sample_id": f"sample-{consumer_namespace}-{index}",
            "candidate_sha256": f"candidate-{index}",
            "source": "thinklite",
            "attempt_index": index,
            "raw_generation_sha256": f"generation-{index}",
        }
        for index in range(consumers)
    ]
    return {
        "schema_version": T1_SEMANTIC_JUDGE_REQUEST_SCHEMA,
        "judge_request_id": f"t1-semantic-judge:{payload_sha256}",
        "run_id": identity.run_id,
        "run_manifest_sha256": identity.run_manifest_sha256,
        "prompt_sha256": identity.prompt_sha256,
        "judge_config_sha256": identity.judge_config_sha256,
        "model_repository": identity.model_repository,
        "model_served_name": identity.model_served_name,
        "payload_sha256": payload_sha256,
        **payload,
        "consumers": consumer_records,
        "consumer_count": len(consumer_records),
    }


def _write_scoring(
    root: Path,
    *,
    schema: str,
    identity: _RunIdentity,
    requests: list[dict[str, Any]],
) -> str:
    requests = sorted(requests, key=lambda row: row["payload_sha256"])
    queue = b"".join(_canonical_json_bytes(row) + b"\n" for row in requests)
    (root / "semantic-judge-requests.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (root / "semantic-judge-requests.jsonl").write_bytes(queue)
    manifest_identity = {
        "schema_version": schema,
        "run_id": identity.run_id,
        "run_manifest_sha256": identity.run_manifest_sha256,
        "selection_candidates_sha256": identity.selection_candidates_sha256,
        "judge_config_sha256": identity.judge_config_sha256,
        "quality_exclusions_sha256": "a" * 64,
        "files": {
            "semantic_judge_requests": {
                "path": "semantic-judge-requests.jsonl",
                "rows": len(requests),
                "bytes": len(queue),
                "sha256": _sha256_bytes(queue),
            }
        },
    }
    if schema == T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA:
        manifest_identity["task_kind_policy"] = POLICY_SELECTION_TASK_KIND_POLICY
    manifest_sha256 = _sha256_bytes(_canonical_json_bytes(manifest_identity))
    _write_canonical(
        root / "manifest.json",
        {**manifest_identity, "manifest_sha256": manifest_sha256},
    )
    return manifest_sha256


def _write_legacy_result(
    fixture: _Fixture,
    *,
    request_payload_model: str | None = None,
    compacted: bool = False,
) -> None:
    request = fixture.source_request
    response = {
        "id": "chatcmpl-test",
        "model": fixture.identity.model_served_name,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": '{"verdict":1,"rationale":"equivalent"}'},
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    verdict, rationale, usage = _strict_response(
        response, expected_model=fixture.identity.model_served_name
    )
    request_payload = _request_payload(request, fixture.bound)
    if request_payload_model is not None:
        request_payload = {**request_payload, "model": request_payload_model}
    response_bytes = _canonical_json_bytes(response)
    evidence_identity = {
        "schema_version": T1_LEGACY_JUDGE_EVIDENCE_SCHEMA,
        "run_id": fixture.identity.run_id,
        "run_manifest_sha256": fixture.identity.run_manifest_sha256,
        "scoring_manifest_sha256": fixture.source_scoring_manifest_sha256,
        "judge_request_id": request["judge_request_id"],
        "payload_sha256": request["payload_sha256"],
        "consumer_count": request["consumer_count"],
        "judge_config_sha256": fixture.identity.judge_config_sha256,
        "prompt_identity": _identity_record(fixture.bound.prompt_identity),
        "service_identity": _identity_record(fixture.bound.service_identity),
        "model_identity": _identity_record(fixture.bound.model_identity),
        "sampling_identity": _identity_record(fixture.bound.sampling_identity),
        "calibration_identity": _identity_record(fixture.bound.calibration_identity),
        "failure_policy_identity": _identity_record(
            fixture.bound.failure_policy_identity
        ),
        "judge_input_compaction": (
            [{"trigger": "input_context_overflow"}] if compacted else []
        ),
        "runtime_replica_base_url": "http://127.0.0.1:8013/v1",
        "runtime_replica_index": 0,
        "request_payload": request_payload,
        "request_payload_sha256": _sha256_bytes(_canonical_json_bytes(request_payload)),
        "raw_response_bytes_sha256": _sha256_bytes(response_bytes),
        "response_json_sha256": _sha256_bytes(response_bytes),
        "response": response,
        "response_id": response["id"],
        "response_model": response["model"],
        "finish_reason": "stop",
        "usage": usage,
        "verdict": verdict,
        "rationale": rationale,
    }
    evidence_sha256 = _sha256_bytes(_canonical_json_bytes(evidence_identity))
    evidence = {**evidence_identity, "evidence_sha256": evidence_sha256}
    evidence_payload = _canonical_json_bytes(evidence) + b"\n"
    evidence_relative = (
        Path("evidence") / evidence_sha256[:2] / f"{evidence_sha256}.json"
    )
    path = fixture.source_judge_root / evidence_relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(evidence_payload)
    index_identity = {
        "schema_version": T1_LEGACY_JUDGE_INDEX_SCHEMA,
        "run_manifest_sha256": fixture.identity.run_manifest_sha256,
        "scoring_manifest_sha256": fixture.source_scoring_manifest_sha256,
        "judge_request_id": request["judge_request_id"],
        "payload_sha256": request["payload_sha256"],
        "evidence_sha256": evidence_sha256,
        "evidence_file": evidence_relative.as_posix(),
        "evidence_file_sha256": _sha256_bytes(evidence_payload),
        "verdict": verdict,
    }
    index_sha256 = _sha256_bytes(_canonical_json_bytes(index_identity))
    _write_canonical(
        fixture.source_judge_root
        / "requests"
        / str(request["payload_sha256"])[:2]
        / f"{request['payload_sha256']}.json",
        {**index_identity, "index_sha256": index_sha256},
    )


def _fixture(
    tmp_path: Path,
    *,
    source_task_kind: str = "open_vqa",
    target_task_kind: str = "open_vqa",
) -> _Fixture:
    repository_root = Path(__file__).resolve().parents[2]
    judge_path = (
        repository_root
        / "configs"
        / "policy"
        / "judges"
        / "qwen25_72b_rl_answer_judge_v1.json"
    )
    judge_bytes = judge_path.read_bytes()
    judge_sha256 = hashlib.sha256(judge_bytes).hexdigest()
    judge_config = json.loads(judge_bytes)
    bound = load_openai_compatible_judge(judge_path, expected_file_sha256=judge_sha256)
    identity = _RunIdentity(
        run_id="T1-test",
        run_manifest_sha256="b" * 64,
        selection_candidates_sha256="c" * 64,
        judge_config_sha256=judge_sha256,
        prompt_sha256=bound.prompt_identity.sha256,
        model_repository=judge_config["model"]["repository"],
        model_served_name=judge_config["model"]["served_name"],
    )
    source_request = _request(identity, task_kind=source_task_kind, consumers=1)
    target_request = _request(identity, task_kind=target_task_kind, consumers=2)
    source_scoring_root = tmp_path / "deterministic-v2"
    target_scoring_root = tmp_path / "deterministic-v3"
    source_scoring_sha256 = _write_scoring(
        source_scoring_root,
        schema=T1_LEGACY_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
        identity=identity,
        requests=[source_request],
    )
    target_scoring_sha256 = _write_scoring(
        target_scoring_root,
        schema=T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
        identity=identity,
        requests=[target_request],
    )
    return _Fixture(
        identity=identity,
        bound=bound,
        source_scoring_root=source_scoring_root,
        source_judge_root=tmp_path / "judge-v1",
        target_scoring_root=target_scoring_root,
        target_judge_root=tmp_path / "judge-v2",
        source_request=source_request,
        target_request=target_request,
        source_scoring_manifest_sha256=source_scoring_sha256,
        target_scoring_manifest_sha256=target_scoring_sha256,
    )


def _run(
    fixture: _Fixture, *, dry_run: bool = False, workers: int = 32
) -> dict[str, Any]:
    return _reuse_legacy_judge_results(
        identity=fixture.identity,
        bound=fixture.bound,
        source_scoring_root=fixture.source_scoring_root,
        source_judge_root=fixture.source_judge_root,
        target_scoring_root=fixture.target_scoring_root,
        target_judge_root=fixture.target_judge_root,
        dry_run=dry_run,
        workers=workers,
    )


def test_reuse_rebinds_only_manifest_schema_and_target_consumer_count(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _write_legacy_result(fixture)

    report = _run(fixture)

    assert report["unchanged_payload_count"] == 1
    assert report["workers"] == 32
    assert report["maximum_in_flight"] == 64
    assert report["reusable_count"] == 1
    assert report["records_written"] == 1
    index = _load_completed_index(
        fixture.target_judge_root,
        fixture.target_request,
        run_manifest_sha256=fixture.identity.run_manifest_sha256,
        scoring_manifest_sha256=fixture.target_scoring_manifest_sha256,
    )
    assert index is not None
    assert index["schema_version"] == T1_JUDGE_INDEX_SCHEMA
    evidence = json.loads(
        (fixture.target_judge_root / index["evidence_file"]).read_bytes()
    )
    assert evidence["schema_version"] == T1_JUDGE_EVIDENCE_SCHEMA
    assert evidence["consumer_count"] == 2
    assert evidence["scoring_manifest_sha256"] == fixture.target_scoring_manifest_sha256
    assert (
        evidence["reuse_provenance"]["schema_version"]
        == T1_JUDGE_REUSE_PROVENANCE_SCHEMA
    )
    assert evidence["request_payload"]["messages"][1]["content"].endswith(
        '"task_kind":"open_vqa"}'
    )
    resumed = _run(fixture)
    assert resumed["target_already_complete_count"] == 1
    assert resumed["records_written"] == 0


def test_changed_task_kind_has_no_reusable_payload(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, source_task_kind="math", target_task_kind="open_vqa")
    _write_legacy_result(fixture)

    report = _run(fixture)

    assert report["unchanged_payload_count"] == 0
    assert report["source_only_count"] == 1
    assert report["target_only_count"] == 1
    assert report["records_written"] == 0
    assert not fixture.target_judge_root.exists()


def test_self_consistent_but_wrong_legacy_http_payload_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _write_legacy_result(fixture, request_payload_model="wrong-model")

    with pytest.raises(ValueError, match="HTTP request payload differs"):
        _run(fixture)
    assert not fixture.target_judge_root.exists()


def test_compacted_legacy_request_is_deliberately_rejudged(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_legacy_result(fixture, compacted=True)

    report = _run(fixture)

    assert report["legacy_input_compaction_skipped_count"] == 1
    assert report["reusable_count"] == 0
    assert report["records_written"] == 0


def test_dry_run_validates_but_does_not_publish(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_legacy_result(fixture)

    report = _run(fixture, dry_run=True)

    assert report["reusable_count"] == 1
    assert report["records_written"] == 0
    assert not fixture.target_judge_root.exists()


@pytest.mark.parametrize("workers", [0, 65, True])
def test_worker_count_is_bounded(tmp_path: Path, workers: int) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match=r"workers must be in \[1, 64\]"):
        _run(fixture, workers=workers)


def test_target_v3_manifest_requires_exact_task_kind_policy(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture.target_scoring_root / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["task_kind_policy"] = "wrong-policy"
    identity = dict(manifest)
    identity.pop("manifest_sha256")
    manifest["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(identity))
    _write_canonical(path, manifest)

    with pytest.raises(ValueError, match="task_kind_policy differs"):
        _run(fixture, workers=1)


def test_exact_intersection_work_is_bounded_and_concurrent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    pairs = [
        (
            _request(
                fixture.identity,
                task_kind="open_vqa",
                consumers=1,
                question=f"Question {index}?",
                consumer_namespace=str(index),
            ),
            _request(
                fixture.identity,
                task_kind="open_vqa",
                consumers=2,
                question=f"Question {index}?",
                consumer_namespace=str(index),
            ),
        )
        for index in range(8)
    ]
    source_manifest_sha256 = _write_scoring(
        fixture.source_scoring_root,
        schema=T1_LEGACY_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
        identity=fixture.identity,
        requests=[pair[0] for pair in pairs],
    )
    target_manifest_sha256 = _write_scoring(
        fixture.target_scoring_root,
        schema=T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
        identity=fixture.identity,
        requests=[pair[1] for pair in pairs],
    )
    fixture = replace(
        fixture,
        source_scoring_manifest_sha256=source_manifest_sha256,
        target_scoring_manifest_sha256=target_manifest_sha256,
    )
    for source_request, target_request in pairs:
        _write_legacy_result(
            replace(
                fixture,
                source_request=source_request,
                target_request=target_request,
            )
        )

    original = judge_reuse._validated_legacy_evidence
    lock = threading.Lock()
    active = 0
    peak = 0

    def observed(*args: Any, **kwargs: Any) -> Any:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.02)
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(judge_reuse, "_validated_legacy_evidence", observed)
    progress: list[Mapping[str, Any]] = []
    report = _reuse_legacy_judge_results(
        identity=fixture.identity,
        bound=fixture.bound,
        source_scoring_root=fixture.source_scoring_root,
        source_judge_root=fixture.source_judge_root,
        target_scoring_root=fixture.target_scoring_root,
        target_judge_root=fixture.target_judge_root,
        workers=3,
        progress_every=2,
        progress_callback=progress.append,
    )

    assert peak == 3
    assert report["records_written"] == 8
    assert report["maximum_in_flight"] == 6
    assert progress
    assert all(record["in_flight"] <= 6 for record in progress)
    assert progress[-1]["unchanged_payload_completed"] == 8
