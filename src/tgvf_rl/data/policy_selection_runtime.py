"""CPU-only runtime contracts for Qwen3 Policy-RL T1 scoring.

The module owns identities, validation, deterministic routing, evidence, and
resume metadata.  It deliberately imports neither Torch nor a model runtime.
GPU orchestration is expected to consume these contracts rather than recreate
their semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal as Decimal, InvalidOperation as InvalidOperation
from enum import Enum as Enum
from fractions import Fraction as Fraction
import hashlib as hashlib
import json
import math as math
import os
from pathlib import Path
import re as re
import tempfile
from types import MappingProxyType as MappingProxyType
from typing import Any

from .policy_selection import (
    POLICY_SELECTION_ATTEMPT_SCHEMA,
    AttemptStatus,
    SelectionBranch,
    SelectionCandidate as SelectionCandidate,
    SelectionSource as SelectionSource,
    stable_selection_request_id as stable_selection_request_id,
)
from .policy_selection_config import (
    T1_ATTEMPTS as T1_ATTEMPTS,
    T1_ATTEMPT_SEED_SCHEMA as T1_ATTEMPT_SEED_SCHEMA,
    T1_INSTRUCT_ANSWER_PARSER as T1_INSTRUCT_ANSWER_PARSER,
    T1_MAX_PIXELS as T1_MAX_PIXELS,
    T1_MODEL_IDENTITY_SCHEMA as T1_MODEL_IDENTITY_SCHEMA,
    T1_MODEL_PATH_BY_REPOSITORY as T1_MODEL_PATH_BY_REPOSITORY,
    T1_PROCESSOR_IDENTITY_SCHEMA as T1_PROCESSOR_IDENTITY_SCHEMA,
    T1_PROMPT_IDENTITY_SCHEMA as T1_PROMPT_IDENTITY_SCHEMA,
    T1_PROMPT_SCHEMA as T1_PROMPT_SCHEMA,
    T1_RUN_CONFIG_SCHEMA as T1_RUN_CONFIG_SCHEMA,
    T1_RUNTIME_IDENTITY_SCHEMA as T1_RUNTIME_IDENTITY_SCHEMA,
    T1_SEED_MODULUS as T1_SEED_MODULUS,
    T1_SHARD_COUNT as T1_SHARD_COUNT,
    T1_SOURCE_RGB_SCHEMA as T1_SOURCE_RGB_SCHEMA,
    T1_THINKING_ANSWER_PARSER as T1_THINKING_ANSWER_PARSER,
    T1DataSource as T1DataSource,
    T1ResponseBudget as T1ResponseBudget,
    T1RunConfig as T1RunConfig,
    candidate_rank as candidate_rank,
    derive_t1_attempt_seed as derive_t1_attempt_seed,
    load_t1_run_config as load_t1_run_config,
)
from .policy_selection_config_values import (
    _canonical_json_bytes,
    _exact_fields,
    _json_clone,
    _load_json_object,
    _mapping,
    _reject_duplicate_keys,
    _required_int,
    _required_sha256,
    _required_string,
    _safe_relative_path,
    _sha256_bytes,
    _sha256_json,
)
from .policy_selection_evidence import (
    GenerationDisposition as GenerationDisposition,
    T1_RAW_GENERATION_SCHEMA as T1_RAW_GENERATION_SCHEMA,
    T1_TOKEN_IDS_SCHEMA as T1_TOKEN_IDS_SCHEMA,
    T1RawGenerationEvidence as T1RawGenerationEvidence,
    classify_generation_finish as classify_generation_finish,
    sampled_token_ids_sha256 as sampled_token_ids_sha256,
)
from .policy_selection_recommended import (
    T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION as T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION,
    T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA as T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA,
    T1_RECOMMENDED_SELECTION_NAMESPACE as T1_RECOMMENDED_SELECTION_NAMESPACE,
    T1_RECOMMENDED_SELECTION_ROWS as T1_RECOMMENDED_SELECTION_ROWS,
    T1_RECOMMENDED_SOURCE_QUOTAS as T1_RECOMMENDED_SOURCE_QUOTAS,
)
from .policy_selection_verification import (
    DeterministicVerification as DeterministicVerification,
    VerificationOutcome as VerificationOutcome,
    extract_direct_completion as extract_direct_completion,
    extract_final_answer as extract_final_answer,
    parse_t1_answer as parse_t1_answer,
    verify_arxivqa_answer as verify_arxivqa_answer,
    verify_t1_answer as verify_t1_answer,
    verify_thinklite_answer as verify_thinklite_answer,
    verify_vstar_answer as verify_vstar_answer,
)

T1_CHUNK_MANIFEST_SCHEMA = "tgvf.policy-selection.t1-chunk-manifest.v1"
T1_RENDERED_PROMPT_TOKEN_IDS_SCHEMA = (
    "tgvf.policy-selection.rendered-prompt-token-ids.v1"
)

_CHUNK_FIELDS = {
    "schema_version",
    "run_manifest_sha256",
    "shard_rank",
    "world_size",
    "chunk_index",
    "record_count",
    "evidence_file",
    "evidence_sha256",
    "logical_keys_sha256",
    "manifest_sha256",
}


def native_user_message_descriptor(
    *, image: str, question: str
) -> list[dict[str, Any]]:
    """Return the sole user message accepted by the tool-free T1 prompt."""

    image_value = _required_string(image, field_name="image")
    question_value = _required_string(question, field_name="question")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_value},
                {"type": "text", "text": question_value},
            ],
        }
    ]


def native_prompt_identity_sha256(
    *, question: str, image_sha256: str, chat_template_sha256: str
) -> str:
    """Bind prompt semantics without making a relocatable image path an identity."""

    question_value = _required_string(question, field_name="question")
    image_identity = _required_sha256(image_sha256, field_name="image_sha256")
    template_identity = _required_sha256(
        chat_template_sha256, field_name="chat_template_sha256"
    )
    return _sha256_json(
        {
            "schema": T1_PROMPT_IDENTITY_SCHEMA,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_sha256": image_identity},
                        {"type": "text", "text": question_value},
                    ],
                }
            ],
            "system": None,
            "tools": None,
            "add_generation_prompt": True,
            "chat_template_sha256": template_identity,
        }
    )


def rendered_prompt_token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash the exact processor/vLLM-expanded prompt token sequence."""

    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ValueError("rendered prompt token IDs must be a sequence")
    normalized = tuple(token_ids)
    if not normalized or any(
        type(token_id) is not int or token_id < 0 for token_id in normalized
    ):
        raise ValueError(
            "rendered prompt token IDs must be a non-empty non-negative sequence"
        )
    return _sha256_json(
        {"schema": T1_RENDERED_PROMPT_TOKEN_IDS_SCHEMA, "token_ids": normalized}
    )


def source_rgb_sha256(*, width: int, height: int, pixel_bytes: bytes) -> str:
    """Hash exact row-major 8-bit RGB pixels given to the native processor."""

    parsed_width = _required_int(width, field_name="width", minimum=1)
    parsed_height = _required_int(height, field_name="height", minimum=1)
    if not isinstance(pixel_bytes, bytes):
        raise TypeError("pixel_bytes must be bytes")
    expected_bytes = parsed_width * parsed_height * 3
    if len(pixel_bytes) != expected_bytes:
        raise ValueError(
            f"RGB pixel byte length {len(pixel_bytes)} != {expected_bytes}"
        )
    metadata = _canonical_json_bytes(
        {
            "schema": T1_SOURCE_RGB_SCHEMA,
            "mode": "RGB",
            "width": parsed_width,
            "height": parsed_height,
            "byte_length": expected_bytes,
        }
    )
    return _sha256_bytes(b"tgvf-source-rgb-v1\0" + metadata + b"\0" + pixel_bytes)


def evidence_to_attempt_record(
    evidence: T1RawGenerationEvidence | Mapping[str, Any],
    *,
    expected_answer: Any,
    option_count: int | None = None,
    budget_exhausted: bool = False,
    semantic_verdict: bool | None = None,
    semantic_judge_evidence_sha256: str | None = None,
    answer_parser: str = T1_THINKING_ANSWER_PARSER,
) -> dict[str, Any] | None:
    """Convert raw evidence to the reducer schema without losing uncertainty.

    A non-final length finish returns ``None`` so the caller schedules the next
    budget revision.  A semantic-required rule result becomes verifier_error
    until identified judge evidence is supplied.
    """

    raw = (
        evidence
        if isinstance(evidence, T1RawGenerationEvidence)
        else T1RawGenerationEvidence.from_record(evidence)
    )
    if type(budget_exhausted) is not bool:
        raise TypeError("budget_exhausted must be boolean")
    if semantic_verdict is not None and type(semantic_verdict) is not bool:
        raise TypeError("semantic_verdict must be boolean or None")
    if semantic_judge_evidence_sha256 is not None:
        _required_sha256(
            semantic_judge_evidence_sha256,
            field_name="semantic_judge_evidence_sha256",
        )

    base: dict[str, Any] = {
        "schema_version": POLICY_SELECTION_ATTEMPT_SCHEMA,
        "request_id": raw.request_id,
        "sample_id": raw.sample_id,
        "candidate_sha256": raw.candidate_sha256,
        "source": raw.source.value,
        "branch": SelectionBranch.FULL_IMAGE.value,
        "attempt_index": raw.attempt_index,
        "run_id": raw.run_id,
        "run_manifest_sha256": raw.run_manifest_sha256,
        "raw_generation_sha256": raw.evidence_sha256,
        "budget_revision": raw.budget_revision,
    }
    if raw.disposition is GenerationDisposition.TRUNCATED:
        if not budget_exhausted:
            return None
        return {
            **base,
            "status": AttemptStatus.TRUNCATED.value,
            "correct": None,
            "answer": None,
            "verification_route": "response_budgets_exhausted",
        }
    if raw.disposition is GenerationDisposition.GENERATION_ERROR:
        return {
            **base,
            "status": AttemptStatus.GENERATION_ERROR.value,
            "correct": None,
            "answer": None,
            "verification_route": "generation_error",
            "generation_error": raw.generation_error,
        }

    answer = parse_t1_answer(raw.raw_text, answer_parser=answer_parser)
    verification = verify_t1_answer(
        source=raw.source,
        candidate_answer=answer,
        expected_answer=expected_answer,
        option_count=option_count,
    )
    if verification.outcome is VerificationOutcome.SEMANTIC_REQUIRED:
        if semantic_verdict is None:
            return {
                **base,
                "status": AttemptStatus.VERIFIER_ERROR.value,
                "correct": None,
                "answer": answer,
                "verification_route": verification.route,
                "verification_evidence": verification.evidence,
                "semantic_required": True,
            }
        if semantic_judge_evidence_sha256 is None:
            raise ValueError(
                "a semantic verdict requires semantic_judge_evidence_sha256"
            )
        correct = semantic_verdict
        route = "local_qwen25_72b_semantic_judge"
    else:
        if semantic_verdict is not None or semantic_judge_evidence_sha256 is not None:
            raise ValueError("semantic judge evidence is forbidden for a rule decision")
        assert verification.correct is not None
        correct = verification.correct
        route = verification.route
    return {
        **base,
        "status": AttemptStatus.SCORED.value,
        "correct": correct,
        "answer": answer,
        "verification_route": route,
        "verification_evidence": verification.evidence,
        "semantic_judge_evidence_sha256": semantic_judge_evidence_sha256,
    }


@dataclass(frozen=True, slots=True)
class T1ChunkManifest:
    run_manifest_sha256: str
    shard_rank: int
    world_size: int
    chunk_index: int
    record_count: int
    evidence_file: Path
    evidence_sha256: str
    logical_keys_sha256: str
    manifest_sha256: str
    _record_bytes: bytes

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "T1ChunkManifest":
        value = _mapping(record, field_name="chunk manifest")
        _exact_fields(value, _CHUNK_FIELDS, field_name="chunk manifest")
        if value["schema_version"] != T1_CHUNK_MANIFEST_SCHEMA:
            raise ValueError(
                f"chunk manifest schema_version must be {T1_CHUNK_MANIFEST_SCHEMA!r}"
            )
        run_sha = _required_sha256(
            value["run_manifest_sha256"], field_name="run_manifest_sha256"
        )
        world_size = _required_int(
            value["world_size"], field_name="world_size", minimum=1
        )
        shard_rank = _required_int(
            value["shard_rank"],
            field_name="shard_rank",
            maximum=world_size - 1,
        )
        chunk_index = _required_int(value["chunk_index"], field_name="chunk_index")
        record_count = _required_int(
            value["record_count"], field_name="record_count", minimum=1
        )
        evidence_file = _safe_relative_path(
            value["evidence_file"], field_name="evidence_file"
        )
        evidence_sha = _required_sha256(
            value["evidence_sha256"], field_name="evidence_sha256"
        )
        expected_file = Path("chunks") / f"{evidence_sha}.jsonl"
        if evidence_file != expected_file:
            raise ValueError("evidence_file is not named by evidence_sha256")
        logical_sha = _required_sha256(
            value["logical_keys_sha256"], field_name="logical_keys_sha256"
        )
        manifest_sha = _required_sha256(
            value["manifest_sha256"], field_name="manifest_sha256"
        )
        identity = dict(_json_clone(value))
        del identity["manifest_sha256"]
        if _sha256_json(identity) != manifest_sha:
            raise ValueError("chunk manifest SHA-256 mismatch")
        normalized = _json_clone(value)
        return cls(
            run_manifest_sha256=run_sha,
            shard_rank=shard_rank,
            world_size=world_size,
            chunk_index=chunk_index,
            record_count=record_count,
            evidence_file=evidence_file,
            evidence_sha256=evidence_sha,
            logical_keys_sha256=logical_sha,
            manifest_sha256=manifest_sha,
            _record_bytes=_canonical_json_bytes(normalized),
        )

    def as_record(self) -> dict[str, Any]:
        return json.loads(self._record_bytes)


def _logical_evidence_key(evidence: T1RawGenerationEvidence) -> tuple[Any, ...]:
    return (
        evidence.candidate_sha256,
        evidence.attempt_index,
        evidence.budget_revision,
    )


def _logical_keys_sha256(evidences: Sequence[T1RawGenerationEvidence]) -> str:
    return _sha256_json(
        {
            "schema": "tgvf.policy-selection.t1-chunk-logical-keys.v1",
            "keys": [
                {
                    "candidate_sha256": evidence.candidate_sha256,
                    "attempt_index": evidence.attempt_index,
                    "budget_revision": evidence.budget_revision,
                }
                for evidence in evidences
            ],
        }
    )


def _atomic_write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing immutable artifact differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"existing immutable artifact differs: {path}")
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_chunk_manifest(
    path: str | Path, manifest: T1ChunkManifest | Mapping[str, Any]
) -> T1ChunkManifest:
    parsed = (
        manifest
        if isinstance(manifest, T1ChunkManifest)
        else T1ChunkManifest.from_record(manifest)
    )
    _atomic_write_immutable(Path(path), parsed._record_bytes + b"\n")
    return parsed


def write_content_addressed_chunk(
    output_root: str | Path,
    records: Sequence[T1RawGenerationEvidence | Mapping[str, Any]],
    *,
    run: T1RunConfig,
    shard_rank: int,
    chunk_index: int,
) -> T1ChunkManifest:
    """Atomically publish one immutable evidence chunk and logical manifest."""

    root = Path(output_root)
    if root.absolute() != run.output_root:
        raise ValueError("chunk output_root differs from the run configuration")
    rank = _required_int(
        shard_rank,
        field_name="shard_rank",
        maximum=int(run.runtime["world_size"]) - 1,
    )
    index = _required_int(chunk_index, field_name="chunk_index")
    evidences = [
        item
        if isinstance(item, T1RawGenerationEvidence)
        else T1RawGenerationEvidence.from_record(item)
        for item in records
    ]
    if not evidences:
        raise ValueError("a chunk must contain at least one evidence record")
    for evidence in evidences:
        evidence.validate_against_run(run)
        if (
            candidate_rank(
                evidence.candidate_sha256, world_size=int(run.runtime["world_size"])
            )
            != rank
        ):
            raise ValueError("evidence candidate belongs to a different shard")
    evidences.sort(key=_logical_evidence_key)
    logical_keys = [_logical_evidence_key(evidence) for evidence in evidences]
    if len(logical_keys) != len(set(logical_keys)):
        raise ValueError("chunk contains duplicate logical evidence keys")
    evidence_bytes = b"".join(
        _canonical_json_bytes(evidence.as_record()) + b"\n" for evidence in evidences
    )
    evidence_sha = _sha256_bytes(evidence_bytes)
    evidence_relative = Path("chunks") / f"{evidence_sha}.jsonl"
    _atomic_write_immutable(root / evidence_relative, evidence_bytes)
    identity = {
        "schema_version": T1_CHUNK_MANIFEST_SCHEMA,
        "run_manifest_sha256": run.manifest_sha256,
        "shard_rank": rank,
        "world_size": int(run.runtime["world_size"]),
        "chunk_index": index,
        "record_count": len(evidences),
        "evidence_file": evidence_relative.as_posix(),
        "evidence_sha256": evidence_sha,
        "logical_keys_sha256": _logical_keys_sha256(evidences),
    }
    manifest_record = {**identity, "manifest_sha256": _sha256_json(identity)}
    manifest = T1ChunkManifest.from_record(manifest_record)
    manifest_path = root / "manifests" / f"rank-{rank:02d}-chunk-{index:06d}.json"
    atomic_write_chunk_manifest(manifest_path, manifest)
    return manifest


def validate_chunk_manifest(
    manifest: T1ChunkManifest | Mapping[str, Any],
    *,
    output_root: str | Path,
    run: T1RunConfig,
    expected_rank: int | None = None,
    expected_chunk_index: int | None = None,
) -> T1ChunkManifest:
    """Validate a manifest, its canonical JSONL, and every raw record."""

    parsed = (
        manifest
        if isinstance(manifest, T1ChunkManifest)
        else T1ChunkManifest.from_record(manifest)
    )
    root = Path(output_root)
    if root.absolute() != run.output_root:
        raise ValueError("chunk output_root differs from the run configuration")
    if parsed.run_manifest_sha256 != run.manifest_sha256:
        raise ValueError("chunk run identity mismatch")
    if parsed.world_size != run.runtime["world_size"]:
        raise ValueError("chunk world_size mismatch")
    if expected_rank is not None and parsed.shard_rank != expected_rank:
        raise ValueError("chunk shard rank mismatch")
    if expected_chunk_index is not None and parsed.chunk_index != expected_chunk_index:
        raise ValueError("chunk index mismatch")
    evidence_path = root / parsed.evidence_file
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    payload = evidence_path.read_bytes()
    if _sha256_bytes(payload) != parsed.evidence_sha256:
        raise ValueError("chunk evidence SHA-256 mismatch")
    lines = payload.splitlines()
    if len(lines) != parsed.record_count or any(not line for line in lines):
        raise ValueError("chunk evidence record count mismatch")
    evidences: list[T1RawGenerationEvidence] = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {constant}")
                ),
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid evidence JSON at line {index + 1}") from exc
        evidence = T1RawGenerationEvidence.from_record(record)
        evidence.validate_against_run(run)
        if (
            candidate_rank(evidence.candidate_sha256, world_size=parsed.world_size)
            != parsed.shard_rank
        ):
            raise ValueError("chunk contains an evidence record for another shard")
        if _canonical_json_bytes(record) != line:
            raise ValueError("chunk evidence JSONL is not canonical")
        evidences.append(evidence)
    if evidences != sorted(evidences, key=_logical_evidence_key):
        raise ValueError("chunk evidence records are not in canonical logical order")
    logical_keys = [_logical_evidence_key(evidence) for evidence in evidences]
    if len(logical_keys) != len(set(logical_keys)):
        raise ValueError("chunk contains duplicate logical evidence keys")
    if _logical_keys_sha256(evidences) != parsed.logical_keys_sha256:
        raise ValueError("chunk logical-key SHA-256 mismatch")
    return parsed


def load_resumable_chunk(
    manifest_path: str | Path,
    *,
    output_root: str | Path,
    run: T1RunConfig,
    expected_rank: int,
    expected_chunk_index: int,
) -> T1ChunkManifest | None:
    """Return an already complete chunk, or ``None`` if it was never committed."""

    path = Path(manifest_path)
    if not path.exists():
        return None
    record = _load_json_object(path)
    return validate_chunk_manifest(
        record,
        output_root=output_root,
        run=run,
        expected_rank=expected_rank,
        expected_chunk_index=expected_chunk_index,
    )


__all__ = [
    "DeterministicVerification",
    "GenerationDisposition",
    "T1_ATTEMPTS",
    "T1_ATTEMPT_SEED_SCHEMA",
    "T1_CHUNK_MANIFEST_SCHEMA",
    "T1_MAX_PIXELS",
    "T1_INSTRUCT_ANSWER_PARSER",
    "T1_MODEL_PATH_BY_REPOSITORY",
    "T1_PROMPT_SCHEMA",
    "T1_RENDERED_PROMPT_TOKEN_IDS_SCHEMA",
    "T1_RAW_GENERATION_SCHEMA",
    "T1_RUN_CONFIG_SCHEMA",
    "T1_SHARD_COUNT",
    "T1_SOURCE_RGB_SCHEMA",
    "T1_THINKING_ANSWER_PARSER",
    "T1ChunkManifest",
    "T1DataSource",
    "T1RawGenerationEvidence",
    "T1ResponseBudget",
    "T1RunConfig",
    "VerificationOutcome",
    "atomic_write_chunk_manifest",
    "candidate_rank",
    "classify_generation_finish",
    "derive_t1_attempt_seed",
    "evidence_to_attempt_record",
    "extract_final_answer",
    "extract_direct_completion",
    "parse_t1_answer",
    "load_resumable_chunk",
    "load_t1_run_config",
    "native_prompt_identity_sha256",
    "native_user_message_descriptor",
    "rendered_prompt_token_ids_sha256",
    "sampled_token_ids_sha256",
    "source_rgb_sha256",
    "validate_chunk_manifest",
    "verify_arxivqa_answer",
    "verify_t1_answer",
    "verify_thinklite_answer",
    "verify_vstar_answer",
    "write_content_addressed_chunk",
]
