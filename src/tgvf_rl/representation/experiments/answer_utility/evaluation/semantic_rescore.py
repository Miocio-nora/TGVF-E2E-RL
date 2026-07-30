"""Blind semantic rescoring of immutable answer-utility generation ledgers.

This is an experiment-private, inference-only overlay.  Generation artifacts
are read and hash-bound, never modified.  The deterministic Instruct scorer is
reapplied first; only unresolved answers are sent to the pinned local semantic
judge.  Judge requests deliberately omit checkpoint, arm, and intervention
metadata.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import tgvf_rl.judges.openai_compatible as openai_compatible_judge_module
from tgvf_rl.judges.openai_compatible import (
    QWEN25_72B_RL_JUDGE_SYSTEM_PROMPT,
    _binary_verdict,
    load_openai_compatible_judge,
)
from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.representation.training.data import (
    load_retained_representation_jsonl,
)
from tgvf_rl.representation.training.evaluation_runner import (
    load_representation_internal_evaluation_run_config,
)
from tgvf_rl.representation.training.oracle_d_utility import (
    OracleDUtilityGroundTruth,
    split_oracle_d_utility_sample,
)

from .scoring import (
    INSTRUCT_SCORING_CONTRACT_VERSION,
    score_instruct_generated_answer,
)


SEMANTIC_RESCORE_SCHEMA_VERSION = "answer-utility-semantic-rescore-v1"
SEMANTIC_RESCORE_RECORD_SCHEMA_VERSION = "answer-utility-semantic-rescore-record-v1"
SEMANTIC_REQUEST_SCHEMA_VERSION = "answer-utility-blind-semantic-request-v1"
SEMANTIC_EVIDENCE_SCHEMA_VERSION = "answer-utility-semantic-evidence-v1"
_REQUEST_PREFIX = "answer-utility-semantic:"
_RECORD_SCHEMAS = {
    "answer-utility-instruct-evaluation-record-v1",
    "answer-utility-instruct-evaluation-record-v2",
}


@dataclass(frozen=True, slots=True)
class _GenerationSource:
    root: Path
    candidate_id: str
    label: str
    identity_sha256: str
    identity: Mapping[str, Any]
    identity_file_sha256: str
    records_file_sha256: str
    summary_file_sha256: str
    records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _EvaluatedRecord:
    source: _GenerationSource
    source_record: Mapping[str, Any]
    source_record_sha256: str
    truth: OracleDUtilityGroundTruth
    original_question: str
    deterministic_score: Mapping[str, Any]
    consumer_id: str
    request_payload: Mapping[str, str] | None
    request_payload_sha256: str | None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_line(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _identity_record(identity: Any) -> dict[str, str]:
    result = asdict(identity)
    if set(result) != {"namespace", "name", "version", "sha256"}:
        raise ValueError("judge ArtifactIdentity schema differs")
    return result


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _load_truth(
    source_evaluation_config_path: str | Path,
) -> tuple[Any, dict[str, tuple[str, OracleDUtilityGroundTruth]], str]:
    config_path = Path(source_evaluation_config_path).expanduser().resolve()
    source_evaluation = load_representation_internal_evaluation_run_config(config_path)
    data = load_retained_representation_jsonl(
        source_evaluation.evaluation_data_path,
        expected_source_sha256=source_evaluation.evaluation_data_source_sha256,
        warn_on_leakage=False,
    )
    truth: dict[str, tuple[str, OracleDUtilityGroundTruth]] = {}
    for sample in data.samples:
        model_input, ground_truth = split_oracle_d_utility_sample(sample)
        truth[sample.sample_id] = (model_input.question, ground_truth)
    return source_evaluation, truth, data.manifest.manifest_sha256


def _source_candidate_id(identity: Mapping[str, Any], root: Path) -> str:
    candidate = identity.get("candidate_id")
    if not isinstance(candidate, str) or not candidate.strip():
        run_config = identity.get("answer_utility_run_config_path")
        candidate = (
            Path(run_config).stem
            if isinstance(run_config, str) and run_config.strip()
            else root.name
        )
    return candidate


def _load_generation_source(
    root: Path,
    *,
    source_config_sha256: str,
    data_manifest_sha256: str,
) -> _GenerationSource:
    identity_path = root / "identity.json"
    records_path = root / "records.jsonl"
    summary_path = root / "summary.json"
    for path in (identity_path, records_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"completed generation artifact is missing: {path}")

    identity_outer = _require_mapping(
        json.loads(identity_path.read_text(encoding="utf-8")), name="identity"
    )
    if set(identity_outer) != {"schema_version", "identity_sha256", "identity"}:
        raise ValueError(f"generation identity fields differ: {root}")
    identity = _require_mapping(identity_outer["identity"], name="generation identity")
    identity_sha = _sha256_bytes(_canonical_json_bytes(identity))
    if identity_outer["identity_sha256"] != identity_sha:
        raise ValueError(f"generation identity SHA256 differs: {root}")
    if (
        identity.get("assistant_dialect")
        != NativeAssistantDialect.QWEN3_VL_INSTRUCT.value
    ):
        raise ValueError(
            f"semantic rescore accepts only Qwen3-VL Instruct runs: {root}"
        )
    if identity.get("source_evaluation_config_sha256") != source_config_sha256:
        raise ValueError(f"generation/source evaluation config differs: {root}")
    if identity.get("data_manifest_sha256") != data_manifest_sha256:
        raise ValueError(f"generation/source data manifest differs: {root}")

    raw_records = records_path.read_bytes()
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(raw_records.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank generation record at {root}:{line_number}")
        record = _require_mapping(json.loads(line), name="generation record")
        if record.get("schema_version") not in _RECORD_SCHEMAS:
            raise ValueError(
                f"unsupported generation record schema at {root}:{line_number}"
            )
        if record.get("run_identity_sha256") != identity_sha:
            raise ValueError(
                f"generation record identity differs at {root}:{line_number}"
            )
        records.append(record)
    if not records:
        raise ValueError(f"generation records are empty: {root}")
    keys = tuple((row.get("sample_id"), row.get("arm")) for row in records)
    if len(set(keys)) != len(keys):
        raise ValueError(
            f"generation records contain duplicate sample/arm keys: {root}"
        )

    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")), name="generation summary"
    )
    records_sha = _sha256_bytes(raw_records)
    if (
        summary.get("status") != "complete"
        or summary.get("run_identity_sha256") != identity_sha
        or summary.get("records_jsonl_sha256") != records_sha
        or summary.get("record_count") != len(records)
    ):
        raise ValueError(f"generation summary does not bind completed records: {root}")
    declared_arms = identity.get("arms")
    selected = identity.get("ordered_selected_samples")
    if not isinstance(declared_arms, list) or not isinstance(selected, list):
        raise ValueError(f"generation identity selection is missing: {root}")
    expected_keys = tuple(
        (item.get("sample_id"), arm)
        for item in selected
        if isinstance(item, Mapping)
        for arm in declared_arms
    )
    if keys != expected_keys:
        raise ValueError(
            f"generation records differ from declared ordered selection: {root}"
        )
    candidate_id = _source_candidate_id(identity, root)
    return _GenerationSource(
        root=root,
        candidate_id=candidate_id,
        label=f"{candidate_id}@{identity_sha[:12]}",
        identity_sha256=identity_sha,
        identity=identity,
        identity_file_sha256=_file_sha256(identity_path),
        records_file_sha256=records_sha,
        summary_file_sha256=_file_sha256(summary_path),
        records=tuple(records),
    )


def _load_sources(
    roots: Sequence[str | Path],
    *,
    source_config_sha256: str,
    data_manifest_sha256: str,
) -> tuple[_GenerationSource, ...]:
    if isinstance(roots, (str, bytes)) or not roots:
        raise ValueError("at least one generation output root is required")
    resolved = tuple(sorted({Path(root).expanduser().resolve() for root in roots}))
    if len(resolved) != len(roots):
        raise ValueError("generation output roots must be unique")
    sources = tuple(
        _load_generation_source(
            root,
            source_config_sha256=source_config_sha256,
            data_manifest_sha256=data_manifest_sha256,
        )
        for root in resolved
    )
    identities = tuple(source.identity_sha256 for source in sources)
    if len(set(identities)) != len(identities):
        raise ValueError("the same generation identity was supplied more than once")
    return sources


def _evaluate_records(
    sources: Sequence[_GenerationSource],
    truth_by_id: Mapping[str, tuple[str, OracleDUtilityGroundTruth]],
) -> tuple[_EvaluatedRecord, ...]:
    evaluated: list[_EvaluatedRecord] = []
    consumer_ids: set[str] = set()
    for source in sources:
        for record in source.records:
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in truth_by_id:
                raise ValueError(
                    f"generation sample is absent from source data: {sample_id!r}"
                )
            question, truth = truth_by_id[sample_id]
            if record.get("sample_content_sha256") != next(
                item.get("sample_content_sha256")
                for item in source.identity["ordered_selected_samples"]
                if item.get("sample_id") == sample_id
            ):
                raise ValueError(
                    f"generation sample content identity differs: {sample_id}"
                )
            if record.get("expected_short_answer") != truth.short_answer:
                raise ValueError(
                    f"generation expected answer differs from source data: {sample_id}"
                )
            generated = record.get("generated_text")
            stop_reason = record.get("generation_stop_reason")
            if not isinstance(generated, str) or stop_reason not in {
                "natural_stop",
                "length_cap",
            }:
                raise ValueError(
                    f"generation answer/stop reason is invalid: {sample_id}"
                )
            score = asdict(
                score_instruct_generated_answer(
                    generated,
                    truth,
                    generation_stop_reason=stop_reason,
                )
            )
            source_record_sha = _sha256_bytes(_canonical_json_bytes(record))
            consumer_identity = {
                "source_identity_sha256": source.identity_sha256,
                "source_record_sha256": source_record_sha,
                "sample_id": sample_id,
                "arm": record["arm"],
            }
            consumer_id = _sha256_bytes(_canonical_json_bytes(consumer_identity))
            if consumer_id in consumer_ids:
                raise ValueError("duplicate semantic-rescore consumer identity")
            consumer_ids.add(consumer_id)
            request_payload: Mapping[str, str] | None = None
            payload_sha: str | None = None
            if score["correct"] is None:
                # This exact payload is the complete information shown to the judge.
                # No source, checkpoint, arm, target, D, or choices are included.
                request_payload = {
                    "task_kind": "open_vqa",
                    "question": question,
                    "candidate_answer": generated,
                    "reference_answer": truth.short_answer,
                }
                payload_sha = _sha256_bytes(_canonical_json_bytes(request_payload))
            evaluated.append(
                _EvaluatedRecord(
                    source=source,
                    source_record=record,
                    source_record_sha256=source_record_sha,
                    truth=truth,
                    original_question=question,
                    deterministic_score=score,
                    consumer_id=consumer_id,
                    request_payload=request_payload,
                    request_payload_sha256=payload_sha,
                )
            )
    return tuple(evaluated)


def _blind_queue(evaluated: Sequence[_EvaluatedRecord]) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in evaluated:
        if item.request_payload is None or item.request_payload_sha256 is None:
            continue
        payload_sha = item.request_payload_sha256
        consumer = {
            "consumer_id": item.consumer_id,
            "source_identity_sha256": item.source.identity_sha256,
            "source_record_sha256": item.source_record_sha256,
            "sample_id": item.source_record["sample_id"],
            "arm": item.source_record["arm"],
        }
        existing = grouped.get(payload_sha)
        if existing is None:
            grouped[payload_sha] = {
                "schema_version": SEMANTIC_REQUEST_SCHEMA_VERSION,
                "judge_request_id": _REQUEST_PREFIX + payload_sha,
                "payload_sha256": payload_sha,
                "payload": dict(item.request_payload),
                "consumers": [consumer],
            }
        else:
            if existing["payload"] != item.request_payload:
                raise RuntimeError("semantic request SHA256 collision")
            if not any(
                value["consumer_id"] == consumer["consumer_id"]
                for value in existing["consumers"]
            ):
                existing["consumers"].append(consumer)
    result: list[dict[str, Any]] = []
    for payload_sha in sorted(grouped):
        request = grouped[payload_sha]
        request["consumers"] = sorted(
            request["consumers"], key=lambda value: value["consumer_id"]
        )
        request["consumer_count"] = len(request["consumers"])
        result.append(request)
    return tuple(result)


def _judge_request_body(request: Mapping[str, Any], bound: Any) -> dict[str, Any]:
    config = bound.provider.config
    payload: dict[str, Any] = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": QWEN25_72B_RL_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _canonical_json_bytes(request["payload"]).decode("utf-8"),
            },
        ],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
    }
    if config.send_json_response_format:
        payload["response_format"] = {"type": "json_object"}
    if config.provider_routing is not None:
        payload["provider"] = dict(config.provider_routing)
    return payload


def _strict_response(
    response: object, *, expected_model: str
) -> tuple[int, str, dict[str, int | float]]:
    value = _require_mapping(response, name="semantic judge response")
    if value.get("model") != expected_model:
        raise RuntimeError("semantic judge response model differs")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("semantic judge must return exactly one choice")
    choice = _require_mapping(choices[0], name="semantic judge choice")
    if choice.get("index") != 0 or choice.get("finish_reason") != "stop":
        raise RuntimeError("semantic judge choice did not finish with stop")
    message = _require_mapping(choice.get("message"), name="semantic judge message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("semantic judge returned empty content")
    verdict, rationale = _binary_verdict(content)
    usage = _require_mapping(value.get("usage"), name="semantic judge usage")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if (
        type(prompt) is not int
        or type(completion) is not int
        or type(total) is not int
        or min(prompt, completion, total) < 0
        or total != prompt + completion
    ):
        raise RuntimeError("semantic judge usage fields are invalid")
    return (
        verdict,
        rationale,
        {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "cost_usd": 0.0,
        },
    )


async def _judge_all(
    requests: Sequence[Mapping[str, Any]],
    *,
    bound: Any,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("judge concurrency must be in [1, 64]")
    if not requests:
        return {}
    import aiohttp

    config = bound.provider.config
    bound.provider.validate_credentials()
    headers = {"Content-Type": "application/json"}
    if config.api_key_env is not None:
        headers["Authorization"] = "Bearer " + os.environ[config.api_key_env].strip()
    if config.http_referer is not None:
        headers["HTTP-Referer"] = config.http_referer
    if config.application_title is not None:
        headers["X-OpenRouter-Title"] = config.application_title
    endpoint = config.base_url.rstrip("/") + "/chat/completions"
    expected_model = config.expected_response_model or config.model_name
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
    semaphore = asyncio.Semaphore(concurrency)

    async def execute(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
        async with semaphore:
            body = _judge_request_body(request, bound)
            body_bytes = _canonical_json_bytes(body)
            try:
                async with session.post(
                    endpoint, data=body_bytes, headers=headers
                ) as response:
                    raw = await response.read()
                    if response.status != 200:
                        raise RuntimeError(
                            f"semantic judge HTTP status {response.status}"
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                raise RuntimeError(
                    f"semantic judge request failed: {request['judge_request_id']}"
                ) from error
            try:
                decoded = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    "semantic judge returned invalid response JSON"
                ) from error
            verdict, rationale, usage = _strict_response(
                decoded, expected_model=expected_model
            )
            evidence_identity = {
                "schema_version": SEMANTIC_EVIDENCE_SCHEMA_VERSION,
                "judge_request_id": request["judge_request_id"],
                "payload_sha256": request["payload_sha256"],
                "request_body_sha256": _sha256_bytes(body_bytes),
                "raw_response_sha256": _sha256_bytes(raw),
                "response_json_sha256": _sha256_bytes(_canonical_json_bytes(decoded)),
                "response_id": decoded.get("id"),
                "response_model": decoded.get("model"),
                "finish_reason": decoded["choices"][0]["finish_reason"],
                "usage": usage,
                "verdict": verdict,
                "rationale": rationale,
            }
            return {
                **evidence_identity,
                "evidence_sha256": _sha256_bytes(
                    _canonical_json_bytes(evidence_identity)
                ),
            }

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        results = await asyncio.gather(
            *(execute(session, request) for request in requests)
        )
    return {
        str(request["payload_sha256"]): evidence
        for request, evidence in zip(requests, results, strict=True)
    }


def _source_binding(source: _GenerationSource) -> dict[str, Any]:
    return {
        "root": str(source.root),
        "candidate_id": source.candidate_id,
        "label": source.label,
        "generation_identity_sha256": source.identity_sha256,
        "identity_file_sha256": source.identity_file_sha256,
        "records_file_sha256": source.records_file_sha256,
        "summary_file_sha256": source.summary_file_sha256,
        "record_count": len(source.records),
    }


def _judge_binding(bound: Any) -> dict[str, Any]:
    return {
        "config_file_sha256": bound.config_file_sha256,
        "prompt_identity": _identity_record(bound.prompt_identity),
        "service_identity": _identity_record(bound.service_identity),
        "model_identity": _identity_record(bound.model_identity),
        "sampling_identity": _identity_record(bound.sampling_identity),
        "calibration_identity": _identity_record(bound.calibration_identity),
        "failure_policy_identity": _identity_record(bound.failure_policy_identity),
        "formal_pilot_accepted": bound.formal_pilot_accepted,
    }


def _overlay_records(
    evaluated: Sequence[_EvaluatedRecord],
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    run_identity_sha256: str,
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for item in evaluated:
        deterministic = item.deterministic_score
        semantic: dict[str, Any] | None = None
        if deterministic["correct"] is None:
            assert item.request_payload_sha256 is not None
            judged = evidence.get(item.request_payload_sha256)
            if judged is None:
                raise RuntimeError("semantic judge evidence is missing")
            semantic = {
                "judge_request_id": _REQUEST_PREFIX + item.request_payload_sha256,
                "payload_sha256": item.request_payload_sha256,
                "evidence_sha256": judged["evidence_sha256"],
                "verdict": judged["verdict"],
                "rationale": judged["rationale"],
                "usage": judged["usage"],
            }
            correct = bool(judged["verdict"])
            route = "qwen2.5_72b_semantic_fallback"
        else:
            correct = bool(deterministic["correct"])
            route = "deterministic:" + str(deterministic["route"])
        record = item.source_record
        choices = [asdict(choice) for choice in item.truth.choices]
        overlay_identity = {
            "schema_version": SEMANTIC_RESCORE_RECORD_SCHEMA_VERSION,
            "run_identity_sha256": run_identity_sha256,
            "source_label": item.source.label,
            "candidate_id": item.source.candidate_id,
            "source_root": str(item.source.root),
            "source_generation_identity_sha256": item.source.identity_sha256,
            "source_record_sha256": item.source_record_sha256,
            "sample_id": record["sample_id"],
            "arm": record["arm"],
            "original_question": item.original_question,
            "expected_short_answer": item.truth.short_answer,
            "choices": choices,
            "generated_text": record["generated_text"],
            "generation_stop_reason": record["generation_stop_reason"],
            "original_score": record.get("score"),
            "deterministic_score": dict(deterministic),
            "semantic_judge": semantic,
            "final_score": {
                "correct": correct,
                "route": route,
                "scope": "diagnostic_semantic_overlay_not_formal_pilot",
            },
        }
        result.append(
            {
                **overlay_identity,
                "overlay_record_sha256": _sha256_bytes(
                    _canonical_json_bytes(overlay_identity)
                ),
            }
        )
    return tuple(result)


def _metric(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        raise ValueError("cannot summarize an empty semantic-rescore slice")
    deterministic_correct = sum(
        row["deterministic_score"]["correct"] is True for row in records
    )
    deterministic_incorrect = sum(
        row["deterministic_score"]["correct"] is False for row in records
    )
    unresolved = total - deterministic_correct - deterministic_incorrect
    final_correct = sum(row["final_score"]["correct"] is True for row in records)
    return {
        "total": total,
        "deterministic_correct": deterministic_correct,
        "deterministic_incorrect": deterministic_incorrect,
        "semantic_judge_count": unresolved,
        "deterministic_strict_lower_bound_accuracy": deterministic_correct / total,
        "diagnostic_final_correct": final_correct,
        "diagnostic_final_incorrect": total - final_correct,
        "diagnostic_semantic_accuracy": final_correct / total,
    }


_PAIRED_COMPARISONS = (
    ("D_only_content_effect", "correct_D_only", "target_zero_D_only"),
    ("D_only_specificity", "correct_D_only", "matched_wrong_D"),
    ("image_plus_D_content_effect", "image_correct_D", "image_target_zero_D"),
    ("image_plus_D_specificity", "image_correct_D", "image_matched_wrong_D"),
    (
        "image_plus_D_image_grounding",
        "image_correct_D",
        "image_same_target_wrong_image_D",
    ),
    (
        "direct_D_replacement_content_effect",
        "direct_correct_D_replacement",
        "direct_zero_D_replacement",
    ),
    (
        "direct_D_replacement_specificity",
        "direct_correct_D_replacement",
        "direct_matched_wrong_D_replacement",
    ),
    (
        "direct_D_replacement_vs_native_image",
        "direct_correct_D_replacement",
        "image_only",
    ),
)


def _paired_effects(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in records:
        key = (
            str(record["candidate_id"]),
            str(record["sample_id"]),
            str(record["arm"]),
        )
        if key in truth:
            raise ValueError("candidate summary repeats one sample/arm")
        truth[key] = record
    result: dict[str, Any] = {}
    for name, treatment, control in _PAIRED_COMPARISONS:
        pair_keys = sorted(
            (candidate, sample_id)
            for candidate, sample_id, arm in truth
            if arm == treatment and (candidate, sample_id, control) in truth
        )
        if not pair_keys:
            continue
        treatment_final = tuple(
            bool(truth[(*key, treatment)]["final_score"]["correct"])
            for key in pair_keys
        )
        control_final = tuple(
            bool(truth[(*key, control)]["final_score"]["correct"]) for key in pair_keys
        )
        treatment_deterministic = tuple(
            truth[(*key, treatment)]["deterministic_score"]["correct"]
            for key in pair_keys
        )
        control_deterministic = tuple(
            truth[(*key, control)]["deterministic_score"]["correct"]
            for key in pair_keys
        )
        pair_count = len(pair_keys)
        treatment_accuracy = sum(treatment_final) / pair_count
        control_accuracy = sum(control_final) / pair_count
        wins = sum(
            left and not right
            for left, right in zip(treatment_final, control_final, strict=True)
        )
        losses = sum(
            not left and right
            for left, right in zip(treatment_final, control_final, strict=True)
        )
        result[name] = {
            "treatment": treatment,
            "control": control,
            "paired_samples": pair_count,
            "treatment_diagnostic_semantic_accuracy": treatment_accuracy,
            "control_diagnostic_semantic_accuracy": control_accuracy,
            "diagnostic_semantic_accuracy_delta": (
                treatment_accuracy - control_accuracy
            ),
            "wins": wins,
            "losses": losses,
            "ties": pair_count - wins - losses,
            "treatment_deterministic_lower_bound_accuracy": (
                sum(value is True for value in treatment_deterministic) / pair_count
            ),
            "control_deterministic_lower_bound_accuracy": (
                sum(value is True for value in control_deterministic) / pair_count
            ),
        }
    return result


def _summary(
    records: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    run_identity_sha256: str,
) -> dict[str, Any]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record["source_label"])].append(record)
        by_candidate[str(record["candidate_id"])].append(record)
        by_arm[str(record["arm"])].append(record)
    verdicts = Counter(str(item["verdict"]) for item in evidence.values())
    usage = {
        key: sum(int(item["usage"][key]) for item in evidence.values())
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "schema_version": SEMANTIC_RESCORE_SCHEMA_VERSION,
        "status": "complete",
        "run_identity_sha256": run_identity_sha256,
        "claim_scope": "diagnostic_semantic_overlay_not_formal_pilot",
        "overall": _metric(records),
        "by_source": {key: _metric(value) for key, value in sorted(by_source.items())},
        "by_candidate": {
            key: {
                "overall": _metric(value),
                "by_arm": {
                    arm: _metric(arm_records)
                    for arm, arm_records in sorted(
                        (
                            arm,
                            [row for row in value if row["arm"] == arm],
                        )
                        for arm in {str(row["arm"]) for row in value}
                    )
                },
                "paired_effects": _paired_effects(value),
            }
            for key, value in sorted(by_candidate.items())
        },
        "by_arm": {key: _metric(value) for key, value in sorted(by_arm.items())},
        "paired_effects": _paired_effects(records),
        "unique_judge_requests": len(evidence),
        "judge_verdict_counts": dict(sorted(verdicts.items())),
        "judge_usage": usage,
    }


def _publish_complete_directory(output_root: Path, files: Mapping[str, bytes]) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"semantic-rescore output already exists; choose a new root: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent)
    )
    try:
        for relative, payload in files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.rename(staging, output_root)
        directory_fd = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


async def run_semantic_rescore(
    generation_output_roots: Sequence[str | Path],
    source_evaluation_config_path: str | Path,
    judge_config_path: str | Path,
    *,
    expected_judge_config_sha256: str,
    output_root: str | Path,
    concurrency: int = 32,
) -> dict[str, Any]:
    """Re-score immutable generation roots and atomically publish one overlay."""

    output = Path(output_root).expanduser().resolve()
    config_path = Path(source_evaluation_config_path).expanduser().resolve()
    judge_path = Path(judge_config_path).expanduser().resolve()
    if _file_sha256(judge_path) != expected_judge_config_sha256:
        raise ValueError("semantic judge config SHA256 differs from the pinned value")
    source_evaluation, truth, data_manifest_sha = _load_truth(config_path)
    source_config_sha = _file_sha256(config_path)
    if source_evaluation.source_sha256 != source_config_sha:
        raise ValueError("source evaluation config loader/file SHA256 differs")
    sources = _load_sources(
        generation_output_roots,
        source_config_sha256=source_config_sha,
        data_manifest_sha256=data_manifest_sha,
    )
    for source in sources:
        if (
            output == source.root
            or source.root in output.parents
            or output in source.root.parents
        ):
            raise ValueError(
                "semantic-rescore output must not overlap generation roots"
            )
    evaluated = _evaluate_records(sources, truth)
    requests = _blind_queue(evaluated)
    bound = load_openai_compatible_judge(
        judge_path, expected_file_sha256=expected_judge_config_sha256
    )
    identity = {
        "schema_version": SEMANTIC_RESCORE_SCHEMA_VERSION,
        "source_evaluation_config_path": str(config_path),
        "source_evaluation_config_sha256": source_config_sha,
        "evaluation_data_manifest_sha256": data_manifest_sha,
        "generation_sources": [_source_binding(source) for source in sources],
        "deterministic_scoring_contract_version": INSTRUCT_SCORING_CONTRACT_VERSION,
        "deterministic_scorer_file_sha256": _file_sha256(
            Path(__file__).with_name("scoring.py")
        ),
        "semantic_rescorer_file_sha256": _file_sha256(Path(__file__)),
        "openai_compatible_judge_client_file_sha256": _file_sha256(
            Path(openai_compatible_judge_module.__file__).resolve()
        ),
        "judge": _judge_binding(bound),
        "blind_request_fields": [
            "task_kind",
            "question",
            "candidate_answer",
            "reference_answer",
        ],
        "blind_task_kind": "open_vqa",
        "selection_rule": "latest_deterministic_score_correct_is_none",
        "choices_model_visible": False,
        "checkpoint_arm_and_D_hidden_from_judge": True,
    }
    run_identity_sha = _sha256_bytes(_canonical_json_bytes(identity))
    evidence = await _judge_all(requests, bound=bound, concurrency=concurrency)
    overlay = _overlay_records(
        evaluated, evidence, run_identity_sha256=run_identity_sha
    )
    summary = _summary(overlay, evidence, run_identity_sha256=run_identity_sha)
    queue_payload = b"".join(_canonical_json_line(item) for item in requests)
    evidence_rows = tuple(evidence[key] for key in sorted(evidence))
    evidence_payload = b"".join(_canonical_json_line(item) for item in evidence_rows)
    records_payload = b"".join(_canonical_json_line(item) for item in overlay)
    summary_payload = _canonical_json_line(summary)
    manifest_identity = {
        "schema_version": SEMANTIC_RESCORE_SCHEMA_VERSION,
        "status": "complete",
        "run_identity_sha256": run_identity_sha,
        "identity": identity,
        "files": {
            "blind_requests": {
                "path": "blind_requests.jsonl",
                "sha256": _sha256_bytes(queue_payload),
                "rows": len(requests),
            },
            "judge_evidence": {
                "path": "judge_evidence.jsonl",
                "sha256": _sha256_bytes(evidence_payload),
                "rows": len(evidence_rows),
            },
            "overlay_records": {
                "path": "records.jsonl",
                "sha256": _sha256_bytes(records_payload),
                "rows": len(overlay),
            },
            "summary": {
                "path": "summary.json",
                "sha256": _sha256_bytes(summary_payload),
            },
        },
        "unique_judge_requests": len(requests),
        "judge_consumer_count": sum(item["consumer_count"] for item in requests),
    }
    manifest = {
        **manifest_identity,
        "manifest_sha256": _sha256_bytes(_canonical_json_bytes(manifest_identity)),
    }
    _publish_complete_directory(
        output,
        {
            "blind_requests.jsonl": queue_payload,
            "judge_evidence.jsonl": evidence_payload,
            "records.jsonl": records_payload,
            "summary.json": summary_payload,
            "manifest.json": _canonical_json_line(manifest),
        },
    )
    return {
        "status": "complete",
        "output_root": str(output),
        "run_identity_sha256": run_identity_sha,
        "source_count": len(sources),
        "record_count": len(overlay),
        "deterministic_resolved_count": len(overlay)
        - sum(item["consumer_count"] for item in requests),
        "judge_consumer_count": sum(item["consumer_count"] for item in requests),
        "unique_judge_requests": len(requests),
        "summary": summary,
        "manifest_sha256": manifest["manifest_sha256"],
    }


__all__ = [
    "SEMANTIC_RESCORE_SCHEMA_VERSION",
    "run_semantic_rescore",
]
