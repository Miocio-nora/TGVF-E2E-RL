"""Resumable local semantic judging and final reduction for T1."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from tgvf_rl.judges.openai_compatible import (
    QWEN25_72B_RL_JUDGE_SYSTEM_PROMPT,
    _binary_verdict,
    load_openai_compatible_judge,
)

from .policy_selection import (
    canonical_json_line,
    reduce_selection_attempts,
    summarize_selection_decisions,
)
from .policy_selection_runtime import (
    _atomic_write_immutable,
    evidence_to_attempt_record,
    load_t1_run_config,
)
from .policy_selection_t1_scoring import (
    T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
    T1_SCORING_DIRECTORY,
    _effective_generations,
    _option_count,
)
from .policy_selection_vllm import (
    _validate_prepared_output_root,
    load_t1_candidates,
)


T1_JUDGE_EVIDENCE_SCHEMA = "tgvf.policy-selection.t1-semantic-judge-evidence.v1"
T1_JUDGE_INDEX_SCHEMA = "tgvf.policy-selection.t1-semantic-judge-index.v1"
T1_JUDGE_MANIFEST_SCHEMA = "tgvf.policy-selection.t1-semantic-judge-manifest.v1"
T1_FINAL_SCORING_MANIFEST_SCHEMA = "tgvf.policy-selection.t1-final-scoring-manifest.v1"
T1_JUDGE_DIRECTORY = Path("scoring") / "judge-v1"
T1_FINAL_DIRECTORY = Path("scoring") / "final-v1"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _records_payload(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_line(record) for record in records)


def _identity_record(identity: Any) -> dict[str, str]:
    record = asdict(identity)
    if set(record) != {"namespace", "name", "version", "sha256"}:
        raise ValueError("judge ArtifactIdentity schema differs")
    return record


def _load_scoring_queue(
    run: Any,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], str]:
    root = run.output_root / T1_SCORING_DIRECTORY
    manifest_path = root / "manifest.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version")
        != T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA
        or manifest.get("run_manifest_sha256") != run.manifest_sha256
    ):
        raise ValueError("deterministic scoring manifest identity differs")
    identity = dict(manifest)
    manifest_sha256 = identity.pop("manifest_sha256", None)
    if manifest_sha256 != _sha256_bytes(_canonical_json_bytes(identity)):
        raise ValueError("deterministic scoring manifest SHA-256 differs")
    file_record = manifest.get("files", {}).get("semantic_judge_requests")
    if not isinstance(file_record, Mapping):
        raise ValueError("deterministic scoring judge-queue file is missing")
    queue_path = root / str(file_record["path"])
    queue_payload = queue_path.read_bytes()
    if _sha256_bytes(queue_payload) != file_record.get("sha256"):
        raise ValueError("semantic judge queue SHA-256 differs")
    records = tuple(
        json.loads(line) for line in queue_payload.decode("utf-8").splitlines()
    )
    if len(records) != file_record.get("rows"):
        raise ValueError("semantic judge queue row count differs")
    request_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("semantic judge queue row must be an object")
        request_id = record.get("judge_request_id")
        payload_sha256 = record.get("payload_sha256")
        payload = {
            "task_kind": record.get("task_kind"),
            "question": record.get("question"),
            "candidate_answer": record.get("candidate_answer"),
            "reference_answer": record.get("reference_answer"),
        }
        if request_id != f"t1-semantic-judge:{payload_sha256}":
            raise ValueError("semantic judge request identity differs")
        if payload_sha256 != _sha256_bytes(_canonical_json_bytes(payload)):
            raise ValueError("semantic judge payload SHA-256 differs")
        if request_id in request_ids:
            raise ValueError("duplicate semantic judge request ID")
        request_ids.add(str(request_id))
    return dict(manifest), records, str(manifest_sha256)


def _request_payload(request: Mapping[str, Any], bound: Any) -> dict[str, Any]:
    sampling = bound.provider.config
    user_payload = {
        "task_kind": request["task_kind"],
        "question": request["question"],
        "candidate_answer": request["candidate_answer"],
        "reference_answer": request["reference_answer"],
    }
    return {
        "model": sampling.model_name,
        "messages": [
            {"role": "system", "content": QWEN25_72B_RL_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _canonical_json_bytes(user_payload).decode("utf-8"),
            },
        ],
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_tokens,
        "seed": sampling.seed,
        "response_format": {"type": "json_object"},
    }


def _strict_response(
    response: object, *, expected_model: str
) -> tuple[int, str, dict[str, Any]]:
    if not isinstance(response, Mapping):
        raise RuntimeError("semantic judge returned a non-object response")
    if response.get("model") != expected_model:
        raise RuntimeError("semantic judge response model differs")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("semantic judge must return exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping) or choice.get("finish_reason") != "stop":
        raise RuntimeError("semantic judge did not finish with stop")
    if choice.get("index", 0) != 0:
        raise RuntimeError("semantic judge response choice index differs")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("semantic judge returned empty content")
    verdict, rationale = _binary_verdict(content)
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise RuntimeError("semantic judge response has no usage")
    try:
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        total_tokens = usage["total_tokens"]
    except KeyError as exc:
        raise RuntimeError("semantic judge usage fields are missing") from exc
    if (
        type(prompt_tokens) is not int
        or type(completion_tokens) is not int
        or type(total_tokens) is not int
        or min(prompt_tokens, completion_tokens, total_tokens) < 0
        or total_tokens != prompt_tokens + completion_tokens
    ):
        raise RuntimeError("semantic judge usage fields are invalid")
    normalized_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": 0.0,
    }
    return verdict, rationale, normalized_usage


def _index_path(root: Path, request: Mapping[str, Any]) -> Path:
    return root / "requests" / f"{request['payload_sha256']}.json"


def _load_completed_index(
    root: Path,
    request: Mapping[str, Any],
    *,
    run_manifest_sha256: str,
    scoring_manifest_sha256: str,
) -> dict[str, Any] | None:
    index_path = _index_path(root, request)
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index_identity = dict(index)
    index_sha256 = index_identity.pop("index_sha256", None)
    if index_sha256 != _sha256_bytes(_canonical_json_bytes(index_identity)):
        raise ValueError("semantic judge resume index SHA-256 differs")
    if (
        index.get("schema_version") != T1_JUDGE_INDEX_SCHEMA
        or index.get("judge_request_id") != request["judge_request_id"]
        or index.get("payload_sha256") != request["payload_sha256"]
        or index.get("run_manifest_sha256") != run_manifest_sha256
        or index.get("scoring_manifest_sha256") != scoring_manifest_sha256
    ):
        raise ValueError("semantic judge resume index identity differs")
    evidence_path = root / str(index.get("evidence_file"))
    payload = evidence_path.read_bytes()
    if _sha256_bytes(payload) != index.get("evidence_file_sha256"):
        raise ValueError("semantic judge evidence file SHA-256 differs")
    evidence = json.loads(payload)
    evidence_identity = dict(evidence)
    evidence_sha256 = evidence_identity.pop("evidence_sha256", None)
    if evidence_sha256 != _sha256_bytes(_canonical_json_bytes(evidence_identity)):
        raise ValueError("semantic judge evidence identity SHA-256 differs")
    if evidence_sha256 != index.get("evidence_sha256"):
        raise ValueError("semantic judge index/evidence SHA-256 differs")
    return index


async def run_t1_semantic_judge(
    config_path: str | Path,
    *,
    judge_config_path: str | Path,
    concurrency: int = 32,
) -> dict[str, Any]:
    """Execute and durably publish every unique local semantic-judge request."""

    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("judge concurrency must be in [1, 64]")
    path = Path(config_path).resolve()
    run = load_t1_run_config(path, verify_data_files=True)
    _validate_prepared_output_root(run, path)
    _, requests, scoring_manifest_sha256 = _load_scoring_queue(run)
    judge_path = Path(judge_config_path).resolve()
    expected_config_sha256 = str(run.verifier["semantic_judge"]["config_sha256"])
    bound = load_openai_compatible_judge(
        judge_path, expected_file_sha256=expected_config_sha256
    )
    expected_model = str(run.verifier["semantic_judge"]["served_name"])
    if bound.provider.config.model_name != expected_model:
        raise ValueError("semantic judge served-model identity differs")
    output_root = run.output_root / T1_JUDGE_DIRECTORY
    output_root.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for request in requests:
        existing = _load_completed_index(
            output_root,
            request,
            run_manifest_sha256=run.manifest_sha256,
            scoring_manifest_sha256=scoring_manifest_sha256,
        )
        if existing is None:
            pending.append(request)
        else:
            completed[str(request["judge_request_id"])] = existing

    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=bound.provider.config.timeout_seconds)
    endpoint = bound.provider.config.base_url.rstrip("/") + "/chat/completions"

    async def execute(session: Any, request: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            payload = _request_payload(request, bound)
            request_bytes = _canonical_json_bytes(payload)
            last_error: Exception | None = None
            for retry_index in range(5):
                try:
                    async with session.post(
                        endpoint,
                        data=request_bytes,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        raw_response = await response.read()
                        if response.status != 200:
                            raise RuntimeError(
                                f"semantic judge HTTP status {response.status}"
                            )
                    decoded = json.loads(raw_response)
                    verdict, rationale, usage = _strict_response(
                        decoded, expected_model=expected_model
                    )
                    response_canonical = _canonical_json_bytes(decoded)
                    evidence_identity = {
                        "schema_version": T1_JUDGE_EVIDENCE_SCHEMA,
                        "run_id": run.run_id,
                        "run_manifest_sha256": run.manifest_sha256,
                        "scoring_manifest_sha256": scoring_manifest_sha256,
                        "judge_request_id": request["judge_request_id"],
                        "payload_sha256": request["payload_sha256"],
                        "consumer_count": request["consumer_count"],
                        "judge_config_sha256": expected_config_sha256,
                        "prompt_identity": _identity_record(bound.prompt_identity),
                        "service_identity": _identity_record(bound.service_identity),
                        "model_identity": _identity_record(bound.model_identity),
                        "sampling_identity": _identity_record(bound.sampling_identity),
                        "calibration_identity": _identity_record(
                            bound.calibration_identity
                        ),
                        "failure_policy_identity": _identity_record(
                            bound.failure_policy_identity
                        ),
                        "request_payload": payload,
                        "request_payload_sha256": _sha256_bytes(request_bytes),
                        "raw_response_bytes_sha256": _sha256_bytes(raw_response),
                        "response_json_sha256": _sha256_bytes(response_canonical),
                        "response": decoded,
                        "response_id": decoded.get("id"),
                        "response_model": decoded.get("model"),
                        "finish_reason": decoded["choices"][0]["finish_reason"],
                        "usage": usage,
                        "verdict": verdict,
                        "rationale": rationale,
                    }
                    evidence_sha256 = _sha256_bytes(
                        _canonical_json_bytes(evidence_identity)
                    )
                    evidence = {
                        **evidence_identity,
                        "evidence_sha256": evidence_sha256,
                    }
                    evidence_payload = _canonical_json_bytes(evidence) + b"\n"
                    evidence_relative = Path("evidence") / f"{evidence_sha256}.json"
                    _atomic_write_immutable(
                        output_root / evidence_relative, evidence_payload
                    )
                    index_identity = {
                        "schema_version": T1_JUDGE_INDEX_SCHEMA,
                        "run_manifest_sha256": run.manifest_sha256,
                        "scoring_manifest_sha256": scoring_manifest_sha256,
                        "judge_request_id": request["judge_request_id"],
                        "payload_sha256": request["payload_sha256"],
                        "evidence_sha256": evidence_sha256,
                        "evidence_file": evidence_relative.as_posix(),
                        "evidence_file_sha256": _sha256_bytes(evidence_payload),
                        "verdict": verdict,
                    }
                    index_sha256 = _sha256_bytes(_canonical_json_bytes(index_identity))
                    index = {**index_identity, "index_sha256": index_sha256}
                    _atomic_write_immutable(
                        _index_path(output_root, request),
                        _canonical_json_bytes(index) + b"\n",
                    )
                    print(
                        json.dumps(
                            {
                                "event": "semantic_judge_committed",
                                "judge_request_id": request["judge_request_id"],
                                "verdict": verdict,
                                "evidence_sha256": evidence_sha256,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    return index
                except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
                    last_error = exc
                    if retry_index == 4:
                        break
                    await asyncio.sleep(min(2**retry_index, 8))
            assert last_error is not None
            raise RuntimeError(
                f"semantic judge request failed: {request['judge_request_id']}"
            ) from last_error

    if pending:
        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            results = await asyncio.gather(
                *(execute(session, request) for request in pending)
            )
        for result in results:
            completed[str(result["judge_request_id"])] = result

    if len(completed) != len(requests):
        raise ValueError("semantic judge completion count differs from queue")
    indices = [completed[key] for key in sorted(completed)]
    verdict_counts = Counter(str(index["verdict"]) for index in indices)
    manifest_identity = {
        "schema_version": T1_JUDGE_MANIFEST_SCHEMA,
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "scoring_manifest_sha256": scoring_manifest_sha256,
        "judge_config_sha256": expected_config_sha256,
        "request_count": len(requests),
        "consumer_count": sum(int(item["consumer_count"]) for item in requests),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "indices_sha256": _sha256_bytes(_records_payload(indices)),
        "indices": indices,
    }
    manifest_sha256 = _sha256_bytes(_canonical_json_bytes(manifest_identity))
    manifest = {**manifest_identity, "manifest_sha256": manifest_sha256}
    _atomic_write_immutable(
        output_root / "manifest.json", _canonical_json_bytes(manifest) + b"\n"
    )
    return {
        "run_id": run.run_id,
        "request_count": len(requests),
        "records_resumed": len(requests) - len(pending),
        "records_written": len(pending),
        "consumer_count": manifest_identity["consumer_count"],
        "verdict_counts": manifest_identity["verdict_counts"],
        "manifest_sha256": manifest_sha256,
    }


def finalize_t1_scoring(
    config_path: str | Path, *, judge_config_path: str | Path
) -> dict[str, Any]:
    """Merge semantic evidence, reduce all candidate decisions, and publish final T1."""

    path = Path(config_path).resolve()
    run = load_t1_run_config(path, verify_data_files=True)
    _validate_prepared_output_root(run, path)
    scoring_manifest, queue, scoring_manifest_sha256 = _load_scoring_queue(run)
    judge_manifest_path = run.output_root / T1_JUDGE_DIRECTORY / "manifest.json"
    judge_manifest_payload = judge_manifest_path.read_bytes()
    judge_manifest = json.loads(judge_manifest_payload)
    judge_identity = dict(judge_manifest)
    judge_manifest_sha256 = judge_identity.pop("manifest_sha256", None)
    if (
        judge_manifest.get("schema_version") != T1_JUDGE_MANIFEST_SCHEMA
        or judge_manifest.get("run_manifest_sha256") != run.manifest_sha256
        or judge_manifest.get("scoring_manifest_sha256")
        != scoring_manifest_sha256
        or judge_manifest_sha256
        != _sha256_bytes(_canonical_json_bytes(judge_identity))
    ):
        raise ValueError("semantic judge manifest identity differs")
    index_by_judge_id = {
        index["judge_request_id"]: index for index in judge_manifest["indices"]
    }
    verdict_by_request: dict[str, tuple[bool, str]] = {}
    for request in queue:
        index = index_by_judge_id.get(request["judge_request_id"])
        if index is None:
            raise ValueError("semantic judge result is missing")
        verdict = index["verdict"]
        if verdict not in {0, 1}:
            raise ValueError("semantic judge verdict is nonbinary")
        for consumer in request["consumers"]:
            request_id = consumer["request_id"]
            if request_id in verdict_by_request:
                raise ValueError("duplicate semantic verdict consumer")
            verdict_by_request[request_id] = (
                bool(verdict),
                str(index["evidence_sha256"]),
            )

    judge_config_sha256 = _sha256_bytes(Path(judge_config_path).resolve().read_bytes())
    if judge_config_sha256 != run.verifier["semantic_judge"]["config_sha256"]:
        raise ValueError("semantic judge config SHA-256 differs during finalize")
    candidates = load_t1_candidates(run)
    selected, _ = _effective_generations(run, candidates)
    deterministic_attempts_path = (
        run.output_root
        / T1_SCORING_DIRECTORY
        / str(scoring_manifest["files"]["attempts"]["path"])
    )
    deterministic_attempts = {
        record["request_id"]: record
        for record in (
            json.loads(line)
            for line in deterministic_attempts_path.read_text(
                encoding="utf-8"
            ).splitlines()
        )
    }
    attempts: list[dict[str, Any]] = []
    for candidate, evidence, budget_exhausted in selected:
        deterministic = deterministic_attempts[evidence.request_id]
        if deterministic.get("semantic_required") is True:
            verdict = verdict_by_request.get(evidence.request_id)
            if verdict is None:
                raise ValueError("semantic-required attempt has no judge verdict")
            attempt = evidence_to_attempt_record(
                evidence,
                expected_answer=candidate.ground_truth,
                option_count=_option_count(candidate),
                budget_exhausted=budget_exhausted,
                semantic_verdict=verdict[0],
                semantic_judge_evidence_sha256=verdict[1],
                answer_parser=str(run.verifier["answer_parser"]),
            )
            assert attempt is not None
        else:
            attempt = deterministic
        attempts.append(attempt)
    attempts.sort(key=lambda item: (item["sample_id"], item["attempt_index"]))
    decisions = reduce_selection_attempts(
        (candidate.canonical_record for candidate in candidates), attempts
    )
    summary = summarize_selection_decisions(decisions)
    report = {
        "schema_version": "tgvf.policy-selection.t1-final-report.v1",
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "scoring_manifest_sha256": scoring_manifest_sha256,
        "judge_manifest_sha256": judge_manifest_sha256,
        "candidate_count": len(candidates),
        "attempt_count": len(attempts),
        "status_counts": dict(sorted(Counter(item["status"] for item in attempts).items())),
        "correct_count": sum(item.get("correct") is True for item in attempts),
        "incorrect_count": sum(item.get("correct") is False for item in attempts),
        "unscored_count": sum(item.get("correct") is None for item in attempts),
        "selection_summary": summary,
    }
    attempts_payload = _records_payload(attempts)
    decisions_payload = _records_payload(decisions)
    report_payload = _canonical_json_bytes(report) + b"\n"
    output_root = run.output_root / T1_FINAL_DIRECTORY
    files = {
        "attempts": {
            "path": "attempts.jsonl",
            "rows": len(attempts),
            "sha256": _sha256_bytes(attempts_payload),
        },
        "decisions": {
            "path": "decisions.jsonl",
            "rows": len(decisions),
            "sha256": _sha256_bytes(decisions_payload),
        },
        "report": {"path": "report.json", "sha256": _sha256_bytes(report_payload)},
    }
    _atomic_write_immutable(output_root / "attempts.jsonl", attempts_payload)
    _atomic_write_immutable(output_root / "decisions.jsonl", decisions_payload)
    _atomic_write_immutable(output_root / "report.json", report_payload)
    manifest_identity = {
        "schema_version": T1_FINAL_SCORING_MANIFEST_SCHEMA,
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "scoring_manifest_sha256": scoring_manifest_sha256,
        "judge_manifest_sha256": judge_manifest_sha256,
        "files": files,
    }
    final_manifest_sha256 = _sha256_bytes(_canonical_json_bytes(manifest_identity))
    final_manifest = {
        **manifest_identity,
        "manifest_sha256": final_manifest_sha256,
    }
    _atomic_write_immutable(
        output_root / "manifest.json",
        _canonical_json_bytes(final_manifest) + b"\n",
    )
    return {**report, "manifest_sha256": final_manifest_sha256, "files": files}


__all__ = [
    "T1_FINAL_DIRECTORY",
    "T1_FINAL_SCORING_MANIFEST_SCHEMA",
    "T1_JUDGE_DIRECTORY",
    "T1_JUDGE_EVIDENCE_SCHEMA",
    "T1_JUDGE_INDEX_SCHEMA",
    "T1_JUDGE_MANIFEST_SCHEMA",
    "finalize_t1_scoring",
    "run_t1_semantic_judge",
]
