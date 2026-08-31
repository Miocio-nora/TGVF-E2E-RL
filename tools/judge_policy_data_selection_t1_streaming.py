#!/usr/bin/env python3
"""Bounded-memory, multi-replica executor for the accepted T1 judge queue.

This only publishes the same immutable per-request evidence/index records as
``judge_policy_data_selection_t1.py run``.  After every request is present, the
original command is rerun to validate all records and publish the canonical
judge manifest before ``finalize`` is allowed.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Any

from tgvf_rl.data.policy_selection_runtime import _atomic_write_immutable
from tgvf_rl.data.policy_selection import POLICY_SELECTION_TASK_KIND_POLICY
from tgvf_rl.data.policy_selection_t1_judge import (
    T1_JUDGE_DIRECTORY,
    T1_JUDGE_EVIDENCE_SCHEMA,
    T1_JUDGE_INDEX_SCHEMA,
    _canonical_json_bytes,
    _index_path,
    _load_completed_index,
    _request_payload,
    _sha256_bytes,
    _strict_response,
)
from tgvf_rl.data.policy_selection_t1_judge_reuse import (
    T1_LEGACY_JUDGE_DIRECTORY,
)
from tgvf_rl.data.policy_selection_t1_scoring import (
    T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
    T1_SCORING_DIRECTORY,
)
from tgvf_rl.data.policy_selection_vllm import _validate_prepared_output_root
from tgvf_rl.data.policy_selection_runtime import load_t1_run_config
from tgvf_rl.judges.openai_compatible import load_openai_compatible_judge


_CONTEXT_LIMIT_ERROR = re.compile(
    r"maximum context length is (?P<maximum>\d+) tokens.*?"
    r"request has (?P<input>\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)
_COMPACTION_MARKER = "\n\n[... middle omitted by T1 judge input compaction v1 ...]\n\n"
_SHA256_FILE = re.compile(r"(?P<sha256>[0-9a-f]{64})\.json\Z")


def _identity_record(identity: Any) -> dict[str, str]:
    record = asdict(identity)
    if set(record) != {"namespace", "name", "version", "sha256"}:
        raise ValueError("judge ArtifactIdentity schema differs")
    return record


def _scoring_queue_identity(run: Any) -> tuple[Path, int, str, str]:
    root = run.output_root / T1_SCORING_DIRECTORY
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (
        manifest.get("schema_version") != T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA
        or manifest.get("run_manifest_sha256") != run.manifest_sha256
        or manifest.get("task_kind_policy") != POLICY_SELECTION_TASK_KIND_POLICY
    ):
        raise ValueError("deterministic scoring manifest identity differs")
    identity = dict(manifest)
    manifest_sha256 = identity.pop("manifest_sha256", None)
    if manifest_sha256 != _sha256_bytes(_canonical_json_bytes(identity)):
        raise ValueError("deterministic scoring manifest SHA-256 differs")
    record = manifest.get("files", {}).get("semantic_judge_requests")
    if not isinstance(record, dict):
        raise ValueError("deterministic scoring judge queue is missing")
    return (
        root / str(record["path"]),
        int(record["rows"]),
        str(record["sha256"]),
        str(manifest_sha256),
    )


def _verify_queue_file(path: Path, *, expected_rows: int, expected_sha256: str) -> None:
    """Verify the complete immutable queue before publishing any judge result."""

    hasher = hashlib.sha256()
    observed = 0
    with path.open("rb") as handle:
        for line in handle:
            hasher.update(line)
            observed += 1
    if observed != expected_rows:
        raise ValueError("semantic judge queue row count differs")
    if hasher.hexdigest() != expected_sha256:
        raise ValueError("semantic judge queue SHA-256 differs")


def _has_legacy_judge_index(root: Path, payload_sha256: str) -> bool:
    """Return whether either immutable legacy index layout claims a payload.

    Deliberately do not parse or compare legacy contents here.  When concurrent
    reuse is enabled, the legacy-index presence alone partitions ownership:
    the reuse process owns every payload with an old index, and this executor
    owns only payloads without one.  Reuse and the canonical publisher remain
    responsible for validating the old index and its evidence fail-closed.
    """

    sharded = root / "requests" / payload_sha256[:2] / f"{payload_sha256}.json"
    flat = root / "requests" / f"{payload_sha256}.json"
    found = False
    for path in (sharded, flat):
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("legacy semantic judge index is not a regular file")
        found = True
    return found


def _completed_index_inventory(root: Path) -> frozenset[str]:
    """Inventory canonical request-index names with sequential directory I/O.

    This is intentionally only a resume-presence optimization.  The canonical
    publisher remains responsible for parsing every index and its evidence.
    Walking the 256 hash shards once avoids millions of random NFS ``stat``
    calls when a completed 1.69M-request queue needs a tiny missing-tail repair.
    """

    requests_root = root / "requests"
    if not requests_root.is_dir():
        return frozenset()
    payloads: set[str] = set()

    def add_file(entry: os.DirEntry[str], *, expected_prefix: str | None) -> None:
        if not entry.is_file(follow_symlinks=False):
            raise ValueError("semantic judge index inventory contains a non-file")
        match = _SHA256_FILE.fullmatch(entry.name)
        if match is None:
            raise ValueError("semantic judge index inventory filename is invalid")
        payload_sha256 = match.group("sha256")
        if expected_prefix is not None and payload_sha256[:2] != expected_prefix:
            raise ValueError("semantic judge index inventory shard differs")
        if payload_sha256 in payloads:
            raise ValueError("semantic judge index inventory has a duplicate payload")
        payloads.add(payload_sha256)

    with os.scandir(requests_root) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                prefix = entry.name
                if len(prefix) != 2 or any(c not in "0123456789abcdef" for c in prefix):
                    raise ValueError("semantic judge index shard name is invalid")
                with os.scandir(entry.path) as shard_entries:
                    for shard_entry in shard_entries:
                        add_file(shard_entry, expected_prefix=prefix)
            else:
                # Preserve the flat layout used by the earliest canary runs.
                add_file(entry, expected_prefix=None)
    return frozenset(payloads)


def _context_limit_counts(raw_response: bytes) -> tuple[int, int] | None:
    match = _CONTEXT_LIMIT_ERROR.search(raw_response.decode("utf-8", errors="replace"))
    if match is None:
        return None
    maximum = int(match.group("maximum"))
    input_tokens = int(match.group("input"))
    if maximum <= 0 or input_tokens <= maximum:
        return None
    return maximum, input_tokens


def _compact_answer(original: str, *, character_budget: int) -> str:
    if character_budget >= len(original):
        raise ValueError("judge input compaction did not reduce the answer")
    content_budget = character_budget - len(_COMPACTION_MARKER)
    if content_budget < 2:
        raise ValueError("judge input compaction budget is too small")
    head = content_budget // 2
    tail = content_budget - head
    return original[:head] + _COMPACTION_MARKER + original[-tail:]


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.replica_base_url:
        raise ValueError("at least one --replica-base-url is required")
    if not 1 <= args.concurrency_per_replica <= 64:
        raise ValueError("concurrency per replica must be in [1, 64]")
    if args.queue_capacity < len(args.replica_base_url):
        raise ValueError("queue capacity is smaller than replica count")
    if args.maximum_attempts < 1:
        raise ValueError("maximum attempts must be positive")
    if args.progress_every < 1:
        raise ValueError("progress interval must be positive")

    config_path = args.config.resolve()
    # The prepared run identity and deterministic scoring manifest already pin
    # the source/candidate artifacts.  Avoid reparsing the multi-GB candidate
    # corpus on every resumable executor restart; the canonical finalizer will
    # perform the full source verification once before publishing its manifest.
    run = load_t1_run_config(config_path, verify_data_files=False)
    _validate_prepared_output_root(run, config_path)
    (
        requests_path,
        request_count,
        requests_sha256,
        scoring_manifest_sha256,
    ) = _scoring_queue_identity(run)
    if args.startup_queue_validation == "full":
        _verify_queue_file(
            requests_path,
            expected_rows=request_count,
            expected_sha256=requests_sha256,
        )

    judge_path = args.judge_config.resolve()
    expected_config_sha256 = str(run.verifier["semantic_judge"]["config_sha256"])
    bound = load_openai_compatible_judge(
        judge_path, expected_file_sha256=expected_config_sha256
    )
    expected_model = str(run.verifier["semantic_judge"]["served_name"])
    if bound.provider.config.model_name != expected_model:
        raise ValueError("semantic judge served-model identity differs")

    output_root = run.output_root / T1_JUDGE_DIRECTORY
    output_root.mkdir(parents=True, exist_ok=True)
    completed_index_inventory = (
        _completed_index_inventory(output_root)
        if args.resume_validation == "inventory-index-only"
        else None
    )
    legacy_judge_root: Path | None = None
    if args.defer_existing_legacy_indices:
        legacy_judge_root = run.output_root / T1_LEGACY_JUDGE_DIRECTORY
        if not legacy_judge_root.is_dir():
            raise ValueError("legacy semantic judge root is not a directory")
        if legacy_judge_root.resolve() == output_root.resolve():
            raise ValueError("legacy and target semantic judge roots must differ")
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
        maxsize=args.queue_capacity
    )
    counters: Counter[str] = Counter()
    started = time.monotonic()
    lock = asyncio.Lock()

    import aiohttp

    timeout = aiohttp.ClientTimeout(total=bound.provider.config.timeout_seconds)
    sessions = [
        aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=args.concurrency_per_replica),
        )
        for _ in args.replica_base_url
    ]

    async def publish(
        session: aiohttp.ClientSession,
        endpoint: str,
        replica_index: int,
        request: dict[str, Any],
    ) -> None:
        original_answer = str(request["candidate_answer"])
        active_request = request
        payload = _request_payload(active_request, bound)
        request_bytes = _canonical_json_bytes(payload)
        compaction_history: list[dict[str, Any]] = []
        active_character_budget = len(original_answer)
        last_error: Exception | None = None
        for retry_index in range(args.maximum_attempts):
            try:
                async with session.post(
                    endpoint.rstrip("/") + "/chat/completions",
                    data=request_bytes,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    raw_response = await response.read()
                    if response.status != 200:
                        context_counts = (
                            _context_limit_counts(raw_response)
                            if response.status == 400
                            else None
                        )
                        if context_counts is not None:
                            maximum_tokens, input_tokens = context_counts
                            target_input_tokens = max(
                                1,
                                maximum_tokens
                                - int(bound.provider.config.max_tokens)
                                - 512,
                            )
                            next_budget = int(
                                active_character_budget
                                * target_input_tokens
                                / input_tokens
                                * 0.9
                            )
                            next_budget = min(
                                next_budget,
                                active_character_budget - 1,
                            )
                            compacted_answer = _compact_answer(
                                original_answer,
                                character_budget=next_budget,
                            )
                            compaction_history.append(
                                {
                                    "schema_version": (
                                        "tgvf.policy-selection."
                                        "t1-judge-input-compaction.v1"
                                    ),
                                    "trigger": "input_context_overflow",
                                    "retry_index": retry_index,
                                    "original_character_count": len(original_answer),
                                    "previous_character_budget": (
                                        active_character_budget
                                    ),
                                    "compacted_character_count": len(compacted_answer),
                                    "reported_input_tokens": input_tokens,
                                    "maximum_context_tokens": maximum_tokens,
                                    "target_input_tokens": target_input_tokens,
                                    "error_response_sha256": _sha256_bytes(
                                        raw_response
                                    ),
                                    "preserved_segments": "equal_head_and_tail",
                                }
                            )
                            active_character_budget = len(compacted_answer)
                            active_request = {
                                **request,
                                "candidate_answer": compacted_answer,
                            }
                            payload = _request_payload(active_request, bound)
                            request_bytes = _canonical_json_bytes(payload)
                            last_error = RuntimeError(
                                "semantic judge input exceeded context; compacted"
                            )
                            continue
                        raise RuntimeError(
                            f"semantic judge HTTP status {response.status}"
                        )
                decoded = json.loads(raw_response)
                choices = decoded.get("choices") if isinstance(decoded, dict) else None
                choice = (
                    choices[0]
                    if isinstance(choices, list)
                    and len(choices) == 1
                    and isinstance(choices[0], dict)
                    else None
                )
                if choice is not None and choice.get("finish_reason") == "length":
                    # A small number of long, repetitive candidate answers make
                    # the judge copy a pattern indefinitely into its rationale.
                    # Raising max_tokens only lengthens that loop.  Preserve the
                    # semantically important answer head/final-answer tail and
                    # deterministically halve the middle-bearing input instead.
                    if active_character_budget <= 2_048:
                        raise RuntimeError(
                            "semantic judge output remained length-truncated "
                            "after input compaction"
                        )
                    next_budget = max(2_048, active_character_budget // 2)
                    compacted_answer = _compact_answer(
                        original_answer,
                        character_budget=next_budget,
                    )
                    compaction_history.append(
                        {
                            "schema_version": (
                                "tgvf.policy-selection.t1-judge-input-compaction.v1"
                            ),
                            "trigger": "judge_output_length",
                            "retry_index": retry_index,
                            "original_character_count": len(original_answer),
                            "previous_character_budget": active_character_budget,
                            "compacted_character_count": len(compacted_answer),
                            "finish_reason": "length",
                            "truncated_response_sha256": _sha256_bytes(raw_response),
                            "preserved_segments": "equal_head_and_tail",
                        }
                    )
                    active_character_budget = len(compacted_answer)
                    active_request = {
                        **request,
                        "candidate_answer": compacted_answer,
                    }
                    payload = _request_payload(active_request, bound)
                    request_bytes = _canonical_json_bytes(payload)
                    last_error = RuntimeError(
                        "semantic judge output reached its token budget; "
                        "compacted input"
                    )
                    continue
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
                    "judge_input_compaction": compaction_history,
                    # Both replicas have the exact bound model, TP2 topology,
                    # prompt and sampling identity.  Record routing explicitly
                    # instead of pretending every request hit port 8013.
                    "runtime_replica_base_url": endpoint,
                    "runtime_replica_index": replica_index,
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
                evidence = {**evidence_identity, "evidence_sha256": evidence_sha256}
                evidence_payload = _canonical_json_bytes(evidence) + b"\n"
                evidence_relative = (
                    Path("evidence") / evidence_sha256[:2] / f"{evidence_sha256}.json"
                )
                # File flushes are deliberately durable, but doing them on the
                # asyncio thread serializes every in-flight judge request.  Run
                # each immutable commit in the bounded default I/O pool while
                # preserving evidence-before-index crash consistency.
                await asyncio.to_thread(
                    _atomic_write_immutable,
                    output_root / evidence_relative,
                    evidence_payload,
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
                await asyncio.to_thread(
                    _atomic_write_immutable,
                    _index_path(output_root, request),
                    _canonical_json_bytes(index) + b"\n",
                )
                async with lock:
                    counters["written"] += 1
                    counters[f"verdict_{verdict}"] += 1
                    done = (
                        counters["deferred_legacy"]
                        + counters["resumed"]
                        + counters["written"]
                    )
                    if done % args.progress_every == 0:
                        elapsed = max(time.monotonic() - started, 1.0e-9)
                        print(
                            json.dumps(
                                {
                                    "event": "semantic_judge_progress",
                                    "completed": done,
                                    "total": request_count,
                                    "written": counters["written"],
                                    "resumed": counters["resumed"],
                                    "deferred_legacy": counters["deferred_legacy"],
                                    "requests_per_second": counters["written"]
                                    / elapsed,
                                    "elapsed_seconds": elapsed,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                return
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as error:
                last_error = error
                if retry_index + 1 == args.maximum_attempts:
                    break
                await asyncio.sleep(min(2**retry_index, 8))
        assert last_error is not None
        raise RuntimeError(
            f"semantic judge request failed: {request['judge_request_id']}"
        ) from last_error

    async def worker(replica_index: int) -> None:
        session = sessions[replica_index]
        endpoint = args.replica_base_url[replica_index]
        while True:
            request = await queue.get()
            try:
                if request is None:
                    return
                await publish(session, endpoint, replica_index, request)
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker(replica_index))
        for replica_index in range(len(sessions))
        for _ in range(args.concurrency_per_replica)
    ]

    async def produce() -> None:
        hasher = hashlib.sha256()
        observed = 0
        with requests_path.open("rb") as handle:
            for line in handle:
                hasher.update(line)
                request = json.loads(line)
                payload = {
                    "task_kind": request.get("task_kind"),
                    "question": request.get("question"),
                    "candidate_answer": request.get("candidate_answer"),
                    "reference_answer": request.get("reference_answer"),
                }
                payload_sha256 = _sha256_bytes(_canonical_json_bytes(payload))
                if (
                    request.get("payload_sha256") != payload_sha256
                    or request.get("judge_request_id")
                    != f"t1-semantic-judge:{payload_sha256}"
                ):
                    raise ValueError("semantic judge request identity differs")
                observed += 1
                if legacy_judge_root is not None and _has_legacy_judge_index(
                    legacy_judge_root, payload_sha256
                ):
                    # Presence defines exclusive ownership by the concurrent
                    # reuse job.  Do not inspect target judge-v2 first: even if
                    # reuse has already published it, this executor must never
                    # enter the legacy-owned write set.
                    counters["deferred_legacy"] += 1
                    continue
                if completed_index_inventory is not None:
                    is_complete = payload_sha256 in completed_index_inventory
                elif args.resume_validation == "index-only":
                    index_path = _index_path(output_root, request)
                    if index_path.exists() and not index_path.is_file():
                        raise ValueError(
                            "semantic judge resume index is not a regular file"
                        )
                    is_complete = index_path.is_file()
                else:
                    existing = _load_completed_index(
                        output_root,
                        request,
                        run_manifest_sha256=run.manifest_sha256,
                        scoring_manifest_sha256=scoring_manifest_sha256,
                    )
                    is_complete = existing is not None
                if is_complete:
                    counters["resumed"] += 1
                    continue
                await queue.put(request)
        if observed != request_count:
            raise ValueError("semantic judge queue row count differs")
        if hasher.hexdigest() != requests_sha256:
            raise ValueError("semantic judge queue SHA-256 differs")
        for _ in workers:
            await queue.put(None)

    producer = asyncio.create_task(produce())
    try:
        # Await producer and consumers together so any failed request aborts
        # immediately instead of leaving a full queue with no live workers.
        await asyncio.gather(producer, *workers)
    except BaseException:
        producer.cancel()
        for task in workers:
            task.cancel()
        await asyncio.gather(producer, *workers, return_exceptions=True)
        raise
    finally:
        await asyncio.gather(*(session.close() for session in sessions))

    elapsed = time.monotonic() - started
    partitioned = (
        counters["deferred_legacy"] + counters["resumed"] + counters["written"]
    )
    if partitioned != request_count:
        raise RuntimeError("semantic judge ownership partition count differs")
    return {
        "run_id": run.run_id,
        "request_count": request_count,
        "records_resumed": counters["resumed"],
        "records_deferred_legacy": counters["deferred_legacy"],
        "records_written": counters["written"],
        "verdict_counts": {
            "0": counters["verdict_0"],
            "1": counters["verdict_1"],
        },
        "elapsed_seconds": elapsed,
        "requests_per_second": counters["written"] / max(elapsed, 1.0e-9),
        "resume_validation": args.resume_validation,
        "startup_queue_validation": args.startup_queue_validation,
        "next_step": (
            "run the canonical judge_policy_data_selection_t1.py publish command "
            "to validate each index/evidence pair once and publish "
            "judge-v2/manifest.json"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--replica-base-url", action="append", required=True)
    parser.add_argument("--concurrency-per-replica", type=int, default=32)
    parser.add_argument("--queue-capacity", type=int, default=4096)
    parser.add_argument("--maximum-attempts", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--defer-existing-legacy-indices",
        action="store_true",
        help=(
            "while legacy judge-v1 reuse runs concurrently, reserve every "
            "payload with a flat or sharded legacy index for that reuse job "
            "and judge only payloads with no legacy index"
        ),
    )
    parser.add_argument(
        "--resume-validation",
        choices=("full", "index-only", "inventory-index-only"),
        default="full",
        help=(
            "use index-only modes only after a prior full validation; "
            "inventory-index-only avoids per-request NFS stats by walking the "
            "request shards once; the canonical publisher still revalidates "
            "every index and evidence file"
        ),
    )
    parser.add_argument(
        "--startup-queue-validation",
        choices=("full", "deferred"),
        default="full",
        help=(
            "deferred starts immediately and validates row count/SHA at EOF; "
            "use only after this immutable queue has passed a full preflight"
        ),
    )
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
