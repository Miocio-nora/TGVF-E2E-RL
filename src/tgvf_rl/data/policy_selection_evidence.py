"""Immutable raw-generation evidence contracts for T1 policy selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from typing import Any

from tgvf_rl.public_api_compat import rebind_public_class, rebind_public_function

from .policy_selection import (
    SelectionBranch,
    SelectionSource,
    stable_selection_request_id,
)
from .policy_selection_config_schema import T1RunConfig
from .policy_selection_config_values import (
    T1_ATTEMPTS,
    T1_SEED_MODULUS,
    _canonical_json_bytes,
    _exact_fields,
    _freeze_json,
    _json_clone,
    _mapping,
    _required_int,
    _required_sha256,
    _required_string,
    _sha256_bytes,
    _sha256_json,
)

T1_RAW_GENERATION_SCHEMA = "tgvf.policy-selection.t1-raw-generation.v1"
T1_TOKEN_IDS_SCHEMA = "tgvf.policy-selection.sampled-token-ids.v1"

_RAW_REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "run_manifest_sha256",
    "request_id",
    "sample_id",
    "candidate_sha256",
    "source",
    "branch",
    "attempt_index",
    "attempt_seed",
    "budget_revision",
    "max_model_len",
    "max_new_tokens",
    "prompt_sha256",
    "rendered_prompt_token_ids_sha256",
    "prompt_token_count",
    "image_sha256",
    "image_evidence",
    "sampled_token_ids_sha256",
    "sampled_token_count",
    "raw_text",
    "finish_reason",
    "stop_reason",
    "backend",
}
_RAW_OPTIONAL_FIELDS = {"sampled_token_ids", "generation_error"}
_IMAGE_EVIDENCE_FIELDS = {
    "source_width",
    "source_height",
    "source_mode",
    "source_rgb_sha256",
    "processed_width",
    "processed_height",
}
_BACKEND_EVIDENCE_FIELDS = {
    "name",
    "version",
    "runtime_sha256",
    "model_sha256",
    "processor_sha256",
}


def sampled_token_ids_sha256(token_ids: Sequence[int]) -> str:
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ValueError("sampled token IDs must be a sequence")
    normalized = tuple(token_ids)
    if any(type(token_id) is not int or token_id < 0 for token_id in normalized):
        raise ValueError("sampled token IDs must be non-negative integers")
    return _sha256_json({"schema": T1_TOKEN_IDS_SCHEMA, "token_ids": normalized})


class GenerationDisposition(str, Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    GENERATION_ERROR = "generation_error"


def classify_generation_finish(finish_reason: str) -> GenerationDisposition:
    try:
        reason = _required_string(finish_reason, field_name="finish_reason")
    except ValueError as exc:
        raise ValueError("finish_reason must be stop, length, or error") from exc
    if reason == "length":
        return GenerationDisposition.TRUNCATED
    if reason == "stop":
        return GenerationDisposition.COMPLETED
    if reason == "error":
        return GenerationDisposition.GENERATION_ERROR
    raise ValueError("finish_reason must be stop, length, or error")


@dataclass(frozen=True, slots=True)
class T1RawGenerationEvidence:
    run_id: str
    run_manifest_sha256: str
    request_id: str
    sample_id: str
    candidate_sha256: str
    source: SelectionSource
    attempt_index: int
    attempt_seed: int
    budget_revision: int
    max_model_len: int
    max_new_tokens: int
    prompt_sha256: str
    rendered_prompt_token_ids_sha256: str
    prompt_token_count: int
    image_sha256: str
    source_width: int
    source_height: int
    source_mode: str
    source_rgb_sha256: str
    processed_width: int
    processed_height: int
    sampled_token_ids_sha256: str
    sampled_token_count: int
    sampled_token_ids: tuple[int, ...] | None
    raw_text: str
    finish_reason: str
    stop_reason: str | int | None
    backend: Mapping[str, Any]
    generation_error: str | None
    _record_bytes: bytes

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "T1RawGenerationEvidence":
        value = _mapping(record, field_name="raw generation")
        actual_fields = set(value)
        if not _RAW_REQUIRED_FIELDS <= actual_fields or actual_fields - (
            _RAW_REQUIRED_FIELDS | _RAW_OPTIONAL_FIELDS
        ):
            missing = sorted(_RAW_REQUIRED_FIELDS - actual_fields)
            unknown = sorted(
                actual_fields - (_RAW_REQUIRED_FIELDS | _RAW_OPTIONAL_FIELDS)
            )
            raise ValueError(
                f"raw generation fields differ; missing={missing}, unknown={unknown}"
            )
        if value["schema_version"] != T1_RAW_GENERATION_SCHEMA:
            raise ValueError(
                f"raw generation schema_version must be {T1_RAW_GENERATION_SCHEMA!r}"
            )
        run_id = _required_string(value["run_id"], field_name="run_id")
        run_sha = _required_sha256(
            value["run_manifest_sha256"], field_name="run_manifest_sha256"
        )
        request_id = _required_string(value["request_id"], field_name="request_id")
        sample_id = _required_string(value["sample_id"], field_name="sample_id")
        candidate_sha = _required_sha256(
            value["candidate_sha256"], field_name="candidate_sha256"
        )
        try:
            source = SelectionSource(value["source"])
        except ValueError as exc:
            raise ValueError("source is unsupported") from exc
        if value["branch"] != SelectionBranch.FULL_IMAGE.value:
            raise ValueError("T1 raw generation branch must be full_image")
        attempt_index = _required_int(
            value["attempt_index"],
            field_name="attempt_index",
            maximum=T1_ATTEMPTS - 1,
        )
        expected_request_id = stable_selection_request_id(
            candidate_sha256=candidate_sha,
            branch=SelectionBranch.FULL_IMAGE,
            attempt_index=attempt_index,
        )
        if request_id != expected_request_id:
            raise ValueError("raw generation request identity mismatch")
        attempt_seed = _required_int(
            value["attempt_seed"],
            field_name="attempt_seed",
            maximum=T1_SEED_MODULUS - 1,
        )
        budget_revision = _required_int(
            value["budget_revision"], field_name="budget_revision"
        )
        max_model_len = _required_int(
            value["max_model_len"], field_name="max_model_len", minimum=1
        )
        max_new_tokens = _required_int(
            value["max_new_tokens"], field_name="max_new_tokens", minimum=1
        )
        prompt_sha = _required_sha256(
            value["prompt_sha256"], field_name="prompt_sha256"
        )
        rendered_prompt_sha = _required_sha256(
            value["rendered_prompt_token_ids_sha256"],
            field_name="rendered_prompt_token_ids_sha256",
        )
        prompt_token_count = _required_int(
            value["prompt_token_count"], field_name="prompt_token_count", minimum=1
        )
        image_sha = _required_sha256(value["image_sha256"], field_name="image_sha256")

        image_evidence = _mapping(value["image_evidence"], field_name="image_evidence")
        _exact_fields(
            image_evidence, _IMAGE_EVIDENCE_FIELDS, field_name="image_evidence"
        )
        source_width = _required_int(
            image_evidence["source_width"],
            field_name="image_evidence.source_width",
            minimum=1,
        )
        source_height = _required_int(
            image_evidence["source_height"],
            field_name="image_evidence.source_height",
            minimum=1,
        )
        source_mode = _required_string(
            image_evidence["source_mode"], field_name="image_evidence.source_mode"
        )
        source_rgb_sha = _required_sha256(
            image_evidence["source_rgb_sha256"],
            field_name="image_evidence.source_rgb_sha256",
        )
        processed_width = _required_int(
            image_evidence["processed_width"],
            field_name="image_evidence.processed_width",
            minimum=1,
        )
        processed_height = _required_int(
            image_evidence["processed_height"],
            field_name="image_evidence.processed_height",
            minimum=1,
        )

        token_sha = _required_sha256(
            value["sampled_token_ids_sha256"], field_name="sampled_token_ids_sha256"
        )
        token_count = _required_int(
            value["sampled_token_count"], field_name="sampled_token_count"
        )
        token_ids_value = value.get("sampled_token_ids")
        token_ids: tuple[int, ...] | None
        if token_ids_value is None:
            token_ids = None
        else:
            if not isinstance(token_ids_value, Sequence) or isinstance(
                token_ids_value, (str, bytes)
            ):
                raise ValueError("sampled_token_ids must be a list when present")
            token_ids = tuple(token_ids_value)
            actual_token_sha = sampled_token_ids_sha256(token_ids)
            if actual_token_sha != token_sha:
                raise ValueError("sampled token IDs SHA-256 mismatch")
            if len(token_ids) != token_count:
                raise ValueError("sampled_token_count differs from sampled_token_ids")

        raw_text = value["raw_text"]
        if not isinstance(raw_text, str):
            raise ValueError("raw_text must be a string")
        finish_reason = _required_string(
            value["finish_reason"], field_name="finish_reason"
        )
        disposition = classify_generation_finish(finish_reason)
        stop_reason = value["stop_reason"]
        if stop_reason is not None and type(stop_reason) not in {str, int}:
            raise ValueError("stop_reason must be a string, integer, or null")
        if disposition is GenerationDisposition.COMPLETED:
            if stop_reason is not None and stop_reason not in {151_645, 151_643}:
                raise ValueError("normal stop_reason is outside the effective EOS set")
        elif stop_reason is not None:
            raise ValueError("length/error evidence must have a null stop_reason")
        backend = _mapping(value["backend"], field_name="backend")
        _exact_fields(backend, _BACKEND_EVIDENCE_FIELDS, field_name="backend")
        _required_string(backend["name"], field_name="backend.name")
        _required_string(backend["version"], field_name="backend.version")
        for field in ("runtime_sha256", "model_sha256", "processor_sha256"):
            _required_sha256(backend[field], field_name=f"backend.{field}")
        generation_error = value.get("generation_error")
        if disposition is GenerationDisposition.GENERATION_ERROR:
            generation_error = _required_string(
                generation_error, field_name="generation_error"
            )
            if token_count != 0 or raw_text:
                raise ValueError(
                    "generation error evidence must have no sampled output"
                )
        elif generation_error is not None:
            raise ValueError("generation_error is only valid for finish_reason=error")

        normalized = _json_clone(value)
        record_bytes = _canonical_json_bytes(normalized)
        return cls(
            run_id=run_id,
            run_manifest_sha256=run_sha,
            request_id=request_id,
            sample_id=sample_id,
            candidate_sha256=candidate_sha,
            source=source,
            attempt_index=attempt_index,
            attempt_seed=attempt_seed,
            budget_revision=budget_revision,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            prompt_sha256=prompt_sha,
            rendered_prompt_token_ids_sha256=rendered_prompt_sha,
            prompt_token_count=prompt_token_count,
            image_sha256=image_sha,
            source_width=source_width,
            source_height=source_height,
            source_mode=source_mode,
            source_rgb_sha256=source_rgb_sha,
            processed_width=processed_width,
            processed_height=processed_height,
            sampled_token_ids_sha256=token_sha,
            sampled_token_count=token_count,
            sampled_token_ids=token_ids,
            raw_text=raw_text,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            backend=_freeze_json(dict(normalized["backend"])),
            generation_error=generation_error,
            _record_bytes=record_bytes,
        )

    @property
    def evidence_sha256(self) -> str:
        return _sha256_bytes(self._record_bytes)

    @property
    def disposition(self) -> GenerationDisposition:
        return classify_generation_finish(self.finish_reason)

    def as_record(self) -> dict[str, Any]:
        return json.loads(self._record_bytes)

    def validate_against_run(self, run: T1RunConfig) -> None:
        if not isinstance(run, T1RunConfig):
            raise TypeError("run must be T1RunConfig")
        if self.run_id != run.run_id or self.run_manifest_sha256 != run.manifest_sha256:
            raise ValueError("raw generation run identity mismatch")
        if self.attempt_seed != run.attempt_seed(
            candidate_sha256=self.candidate_sha256,
            attempt_index=self.attempt_index,
        ):
            raise ValueError("raw generation attempt seed mismatch")
        budget = run.budget(self.budget_revision)
        if (self.max_model_len, self.max_new_tokens) != (
            budget.max_model_len,
            budget.max_new_tokens,
        ):
            raise ValueError("raw generation response budget mismatch")
        expected_backend = {
            "name": run.runtime["backend"],
            "version": run.runtime["version"],
            "runtime_sha256": run.runtime_identity_sha256,
            "model_sha256": run.model_identity_sha256,
            "processor_sha256": run.processor_identity_sha256,
        }
        if dict(self.backend) != expected_backend:
            raise ValueError("raw generation backend identity mismatch")
        if self.processed_width * self.processed_height > run.image["max_pixels"]:
            raise ValueError("processed image exceeds the run max_pixels")


_PUBLIC_RUNTIME_MODULE = "tgvf_rl.data.policy_selection_runtime"
for _public_type in (GenerationDisposition, T1RawGenerationEvidence):
    rebind_public_class(
        _public_type,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNTIME_MODULE,
    )
for _public_function in (sampled_token_ids_sha256, classify_generation_finish):
    rebind_public_function(
        _public_function,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNTIME_MODULE,
    )
del _public_function, _public_type

__all__ = [
    "GenerationDisposition",
    "T1_RAW_GENERATION_SCHEMA",
    "T1_TOKEN_IDS_SCHEMA",
    "T1RawGenerationEvidence",
    "classify_generation_finish",
    "sampled_token_ids_sha256",
]
