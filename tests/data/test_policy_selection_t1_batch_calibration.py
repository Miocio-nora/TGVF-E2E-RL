from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.data.policy_selection_t1_batch_calibration import (
    T1_STRICT_EVIDENCE_SCHEMA,
    T1_STRICT_INDEX_SCHEMA,
    build_batch_request_payload,
    select_completed_candidate_groups,
    strict_batch_response,
    strict_compact_batch_response,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _queue_request(
    *, candidate_sha256: str, attempt_index: int, answer: str
) -> dict[str, object]:
    payload = {
        "task_kind": "open_vqa",
        "question": "What color is the car?",
        "candidate_answer": answer,
        "reference_answer": "red",
    }
    payload_sha256 = _sha(_canonical(payload))
    return {
        "judge_request_id": f"t1-semantic-judge:{payload_sha256}",
        "payload_sha256": payload_sha256,
        **payload,
        "consumer_count": 1,
        "consumers": [
            {
                "candidate_sha256": candidate_sha256,
                "sample_id": "sample-1",
                "source": "vstar",
                "attempt_index": attempt_index,
            }
        ],
    }


def _write_strict_artifact(
    root: Path, request: dict[str, object], *, verdict: int
) -> None:
    evidence_identity = {
        "schema_version": T1_STRICT_EVIDENCE_SCHEMA,
        "judge_request_id": request["judge_request_id"],
        "payload_sha256": request["payload_sha256"],
        "verdict": verdict,
    }
    evidence_sha256 = _sha(_canonical(evidence_identity))
    evidence = {**evidence_identity, "evidence_sha256": evidence_sha256}
    evidence_payload = _canonical(evidence) + b"\n"
    evidence_relative = (
        Path("evidence") / evidence_sha256[:2] / (f"{evidence_sha256}.json")
    )
    evidence_path = root / evidence_relative
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(evidence_payload)
    index_identity = {
        "schema_version": T1_STRICT_INDEX_SCHEMA,
        "judge_request_id": request["judge_request_id"],
        "payload_sha256": request["payload_sha256"],
        "verdict": verdict,
        "evidence_sha256": evidence_sha256,
        "evidence_file": evidence_relative.as_posix(),
        "evidence_file_sha256": _sha(evidence_payload),
    }
    index = {
        **index_identity,
        "index_sha256": _sha(_canonical(index_identity)),
    }
    payload_sha256 = str(request["payload_sha256"])
    index_path = root / "requests" / payload_sha256[:2] / f"{payload_sha256}.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(_canonical(index) + b"\n")


def test_selects_same_candidate_requests_with_strict_identity(tmp_path: Path) -> None:
    candidate_sha256 = "a" * 64
    requests = [
        _queue_request(
            candidate_sha256=candidate_sha256,
            attempt_index=0,
            answer="red",
        ),
        _queue_request(
            candidate_sha256=candidate_sha256,
            attempt_index=1,
            answer="It is red.",
        ),
    ]
    queue = tmp_path / "queue.jsonl"
    queue.write_bytes(b"".join(_canonical(item) + b"\n" for item in requests))
    strict_root = tmp_path / "strict"
    _write_strict_artifact(strict_root, requests[0], verdict=1)
    _write_strict_artifact(strict_root, requests[1], verdict=1)

    groups, audit = select_completed_candidate_groups(
        queue,
        strict_root=strict_root,
        candidate_count=1,
        minimum_items=2,
    )

    assert groups[0]["candidate_sha256"] == candidate_sha256
    assert [item["item_index"] for item in groups[0]["items"]] == [0, 1]
    assert {item["judge_request_id"] for item in groups[0]["items"]} == {
        request["judge_request_id"] for request in requests
    }
    assert audit["selected_original_requests"] == 2


def test_batch_payload_and_response_preserve_item_mapping() -> None:
    group = {
        "task_kind": "open_vqa",
        "question": "What color is the car?",
        "reference_answer": "red",
        "items": [
            {"candidate_answer": "red"},
            {"candidate_answer": "blue"},
        ],
    }
    payload = build_batch_request_payload(
        group,
        model_name="Qwen2.5-72B-Instruct",
        max_tokens=512,
        seed=42,
    )
    user = json.loads(payload["messages"][1]["content"])
    assert user["candidate_answers"] == [
        {"item_index": 0, "candidate_answer": "red"},
        {"item_index": 1, "candidate_answer": "blue"},
    ]
    response = {
        "model": "Qwen2.5-72B-Instruct",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "verdicts": [
                                {
                                    "item_index": 0,
                                    "verdict": 1,
                                    "rationale": "same",
                                },
                                {
                                    "item_index": 1,
                                    "verdict": 0,
                                    "rationale": "different",
                                },
                            ]
                        }
                    )
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }
    verdicts, usage = strict_batch_response(
        response,
        expected_model="Qwen2.5-72B-Instruct",
        expected_count=2,
    )
    assert [item["verdict"] for item in verdicts] == [1, 0]
    assert usage["total_tokens"] == 30


def test_batch_response_rejects_reordered_or_partial_items() -> None:
    response = {
        "model": "Qwen2.5-72B-Instruct",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "verdicts": [
                                {
                                    "item_index": 1,
                                    "verdict": 1,
                                    "rationale": "same",
                                }
                            ]
                        }
                    )
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }
    with pytest.raises(RuntimeError, match="item order differs"):
        strict_batch_response(
            response,
            expected_model="Qwen2.5-72B-Instruct",
            expected_count=1,
        )
    with pytest.raises(RuntimeError, match="verdict count differs"):
        strict_batch_response(
            response,
            expected_model="Qwen2.5-72B-Instruct",
            expected_count=2,
        )


def test_compact_batch_response_is_ordered_binary_list_only() -> None:
    response = {
        "model": "Qwen2.5-72B-Instruct",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": '{"verdicts":[1,0,1]}'},
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }
    verdicts, _ = strict_compact_batch_response(
        response,
        expected_model="Qwen2.5-72B-Instruct",
        expected_count=3,
    )
    assert [item["verdict"] for item in verdicts] == [1, 0, 1]

    changed = json.loads(json.dumps(response))
    changed["choices"][0]["message"]["content"] = (
        '{"verdicts":[1,0,1],"rationales":["a","b","c"]}'
    )
    with pytest.raises(RuntimeError, match="JSON schema differs"):
        strict_compact_batch_response(
            changed,
            expected_model="Qwen2.5-72B-Instruct",
            expected_count=3,
        )


def test_compact_v3_prompt_explicitly_maps_correctness_not_yes_no() -> None:
    payload = build_batch_request_payload(
        {
            "task_kind": "open_vqa",
            "question": "Is it red?",
            "reference_answer": "No",
            "items": [{"candidate_answer": "No"}],
        },
        model_name="Qwen2.5-72B-Instruct",
        max_tokens=128,
        seed=42,
        protocol="compact-v3",
    )
    prompt = payload["messages"][0]["content"]
    assert "1 means the candidate is correct" in prompt
    assert "never encodes the yes/no polarity" in prompt


def test_compact_v4_has_dynamic_count_and_no_copyable_fixed_list() -> None:
    payload = build_batch_request_payload(
        {
            "task_kind": "open_vqa",
            "question": "Is it red?",
            "reference_answer": "No",
            "items": [
                {"candidate_answer": "No"},
                {"candidate_answer": "Yes"},
                {"candidate_answer": "No, it is blue"},
            ],
        },
        model_name="Qwen2.5-72B-Instruct",
        max_tokens=128,
        seed=42,
        protocol="compact-v4",
    )
    prompt = payload["messages"][0]["content"]
    user = json.loads(payload["messages"][1]["content"])
    assert '{"verdicts":[0,1]}' not in prompt
    assert user["expected_verdict_count"] == 3
