"""Shared contracts for exact target-conditioned TGVF inputs."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.contracts.tokens import TokenSpan


TARGET_CONDITIONING_SCHEMA_VERSION = "tgvf-target-conditioning-v1"
CONTEXTUAL_HIDDEN_STATE = "contextual_hidden_state"
TARGET_TOKEN_EMBEDDING = "target_token_embedding"
_PROVIDER_NAMES = {CONTEXTUAL_HIDDEN_STATE, TARGET_TOKEN_EMBEDDING}
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_CANONICAL_INPUT_IDS_PROOF_SEAL = object()


class TargetConditioningProviderKind(str, Enum):
    """Closed set of provider choices recorded in every run identity."""

    CONTEXTUAL_HIDDEN_STATE = "contextual_hidden_state"
    TARGET_TOKEN_EMBEDDING = "target_token_embedding"


@dataclass(frozen=True, slots=True)
class TargetConditioningConfig:
    """Fail-closed configuration for one explicitly selected provider."""

    provider: TargetConditioningProviderKind
    hidden_layer: int | None = None
    embedding_identity: str | None = None
    schema_version: str = TARGET_CONDITIONING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_CONDITIONING_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported target-conditioning schema {self.schema_version!r}"
            )
        if not isinstance(self.provider, TargetConditioningProviderKind):
            raise TypeError("provider must be a TargetConditioningProviderKind")
        if self.provider is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE:
            if not isinstance(self.hidden_layer, int) or isinstance(
                self.hidden_layer, bool
            ):
                raise ValueError(
                    "contextual_hidden_state requires an explicit integer hidden_layer"
                )
            if self.embedding_identity is not None:
                raise ValueError(
                    "contextual_hidden_state cannot configure embedding_identity"
                )
            return
        if self.hidden_layer is not None:
            raise ValueError("target_token_embedding cannot configure hidden_layer")
        if not self.embedding_identity or not self.embedding_identity.strip():
            raise ValueError(
                "target_token_embedding requires an explicit embedding_identity"
            )


@dataclass(frozen=True, slots=True)
class _CanonicalInputIdsProof:
    """CPU token authority bound to one exact, still-unmodified tensor."""

    rows: tuple[tuple[int, ...], ...]
    digest: str
    tensor_identity: int
    tensor_data_ptr: int
    tensor_version: int
    tensor_shape: tuple[int, ...]
    tensor_stride: tuple[int, ...]
    tensor_storage_offset: int
    tensor_device: torch.device
    tensor_dtype: torch.dtype
    seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.seal is not _CANONICAL_INPUT_IDS_PROOF_SEAL:
            raise ValueError("canonical input-ID proofs require the trusted binder")


@dataclass(frozen=True, slots=True)
class TargetConditioningRequest:
    """One common request consumed by either provider implementation."""

    input_ids: torch.Tensor
    target_span: TokenSpan
    expected_target_token_ids: Sequence[int] | Sequence[Sequence[int]]
    trajectory_id: str | Sequence[str]
    call_index: int | Sequence[int]
    model_identity: ModelIdentity
    contextual_hidden_states: torch.Tensor | None = None
    canonical_input_ids_proof: _CanonicalInputIdsProof | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if not isinstance(self.target_span, TokenSpan):
            raise TypeError("target_span must be a TokenSpan")
        if not isinstance(self.model_identity, ModelIdentity):
            raise TypeError("model_identity must be a ModelIdentity")
        if self.contextual_hidden_states is not None and not isinstance(
            self.contextual_hidden_states, torch.Tensor
        ):
            raise TypeError("contextual_hidden_states must be a torch.Tensor")
        if self.canonical_input_ids_proof is not None and not isinstance(
            self.canonical_input_ids_proof,
            _CanonicalInputIdsProof,
        ):
            raise TypeError(
                "canonical_input_ids_proof must be a bound canonical proof"
            )


@dataclass(frozen=True, slots=True)
class TargetConditioningProvenance:
    """Identity of the source tokens and trajectory used to construct ``Hq``."""

    provider: str
    model: ModelIdentity
    target_span: TokenSpan
    target_token_ids: tuple[tuple[int, ...], ...]
    trajectory_ids: tuple[str, ...]
    call_indices: tuple[int, ...]
    source_sequence_length: int
    source_batch_size: int
    source_input_ids_sha256: str
    batched: bool
    hidden_layer: int | None = None
    embedding_identity: str | None = None
    schema_version: str = TARGET_CONDITIONING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_CONDITIONING_SCHEMA_VERSION:
            raise ValueError(f"unsupported conditioning schema {self.schema_version!r}")
        if self.provider not in _PROVIDER_NAMES:
            raise ValueError(f"unknown target-conditioning provider {self.provider!r}")
        if self.source_sequence_length <= 0 or self.source_batch_size <= 0:
            raise ValueError("conditioning source dimensions must be positive")
        if self.target_span.end > self.source_sequence_length:
            raise ValueError("target span lies outside the source token sequence")
        if not (
            len(self.target_token_ids)
            == len(self.trajectory_ids)
            == len(self.call_indices)
            == self.source_batch_size
        ):
            raise ValueError(
                "conditioning provenance rows must match source batch size"
            )
        span_length = self.target_span.end - self.target_span.start
        if any(len(row) != span_length for row in self.target_token_ids):
            raise ValueError(
                "target-token provenance must exactly cover the target span"
            )
        if any(not trajectory_id for trajectory_id in self.trajectory_ids):
            raise ValueError("trajectory IDs must be non-empty")
        if any(call_index < 0 for call_index in self.call_indices):
            raise ValueError("tool call indices must be non-negative")
        if len(self.source_input_ids_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.source_input_ids_sha256
        ):
            raise ValueError("source_input_ids_sha256 must be a lowercase SHA256")
        if not self.batched and self.source_batch_size != 1:
            raise ValueError("multi-row conditioning provenance must be batched")
        if self.provider == CONTEXTUAL_HIDDEN_STATE:
            if self.hidden_layer is None:
                raise ValueError("contextual hidden-state provenance needs a layer")
            if self.embedding_identity is not None:
                raise ValueError(
                    "contextual hidden-state provenance cannot name an embedding"
                )
        else:
            if self.hidden_layer is not None:
                raise ValueError(
                    "token-embedding provenance cannot name a hidden layer"
                )
            if not self.embedding_identity:
                raise ValueError(
                    "token-embedding provenance needs an embedding identity"
                )


@dataclass(frozen=True, slots=True)
class TargetConditioningOutput:
    """Target token features plus enough identity to reject trajectory drift."""

    values: torch.Tensor
    provenance: TargetConditioningProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("conditioning values must be a torch.Tensor")
        expected_rank = 3 if self.provenance.batched else 2
        if self.values.ndim != expected_rank:
            raise ValueError(
                f"conditioning values must have rank {expected_rank} for this provenance"
            )
        if self.values.shape[-2] != (
            self.provenance.target_span.end - self.provenance.target_span.start
        ):
            raise ValueError("conditioning token count differs from the target span")
        if self.values.shape[-1] <= 0:
            raise ValueError("conditioning feature dimension must be positive")
        if (
            self.provenance.batched
            and self.values.shape[0] != self.provenance.source_batch_size
        ):
            raise ValueError("conditioning batch size differs from provenance")
        if not self.values.is_floating_point():
            raise TypeError("conditioning values must use a floating-point dtype")

    @property
    def target_hidden_states(self) -> torch.Tensor:
        return self.values

    @property
    def hq(self) -> torch.Tensor:
        return self.values


@runtime_checkable
class TargetConditionProvider(Protocol):
    """Strong common boundary for a configuration-selected provider."""

    provider_name: str
    model_identity: ModelIdentity

    def build(
        self, request: TargetConditioningRequest, /
    ) -> TargetConditioningOutput: ...


class BoundTargetConditionProvider(nn.Module):
    """Base module bound to one tokenizer/model identity."""

    provider_name: str

    def __init__(self, *, model_identity: ModelIdentity) -> None:
        super().__init__()
        if not isinstance(model_identity, ModelIdentity):
            raise TypeError("model_identity must be a ModelIdentity")
        self.model_identity = model_identity

    def _check_runtime_model(self, runtime_model: ModelIdentity | None) -> None:
        if runtime_model is not None and runtime_model != self.model_identity:
            raise ValueError("runtime model identity differs from the provider binding")


@dataclass(frozen=True, slots=True)
class _ValidatedTargetSelection:
    input_ids: torch.Tensor
    rows: tuple[tuple[int, ...], ...]
    trajectories: tuple[str, ...]
    call_indices: tuple[int, ...]
    batched: bool
    digest: str

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])


def _validate_target_selection(
    *,
    input_ids: torch.Tensor,
    target_span: TokenSpan,
    expected_target_token_ids: Sequence[int] | Sequence[Sequence[int]],
    trajectory_id: str | Sequence[str],
    call_index: int | Sequence[int],
    tokenizer_length: int,
    canonical_input_ids_proof: _CanonicalInputIdsProof | None = None,
) -> _ValidatedTargetSelection:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim not in (1, 2):
        raise ValueError("input_ids must have shape [S] or [B, S]")
    if input_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("input_ids must use an integer dtype")
    if not isinstance(target_span, TokenSpan):
        raise TypeError("target_span must be a TokenSpan")

    batched = input_ids.ndim == 2
    normalized_ids = input_ids if batched else input_ids.unsqueeze(0)
    batch_size, sequence_length = normalized_ids.shape
    if batch_size == 0 or sequence_length == 0:
        raise ValueError("input_ids batch and sequence dimensions must be non-empty")
    if target_span.end > sequence_length:
        raise ValueError("target span lies outside input_ids")
    rows = _normalize_expected_rows(
        expected_target_token_ids,
        batch_size=int(batch_size),
        span_length=target_span.end - target_span.start,
    )
    if canonical_input_ids_proof is None:
        if (
            int(normalized_ids.min()) < 0
            or int(normalized_ids.max()) >= tokenizer_length
        ):
            raise ValueError(
                "input_ids contain an ID outside the bound tokenizer vocabulary"
            )
        expected = torch.tensor(
            rows, device=normalized_ids.device, dtype=normalized_ids.dtype
        )
        selected = normalized_ids[:, target_span.start : target_span.end]
        if not torch.equal(selected, expected):
            raise ValueError(
                "target token IDs do not exactly match input_ids at target_span"
            )
        digest = _hash_input_ids(normalized_ids)
    else:
        canonical_rows, digest = _validate_canonical_input_ids_proof(
            canonical_input_ids_proof,
            input_ids=input_ids,
            batched=batched,
            batch_size=int(batch_size),
            sequence_length=int(sequence_length),
        )
        if any(
            token_id < 0 or token_id >= tokenizer_length
            for row in canonical_rows
            for token_id in row
        ):
            raise ValueError(
                "input_ids contain an ID outside the bound tokenizer vocabulary"
            )
        realized_rows = tuple(
            row[target_span.start : target_span.end] for row in canonical_rows
        )
        if realized_rows != rows:
            raise ValueError(
                "target token IDs do not exactly match input_ids at target_span"
            )

    trajectories = _normalize_string_rows(
        trajectory_id, batch_size=int(batch_size), name="trajectory_id"
    )
    call_indices = _normalize_integer_rows(
        call_index, batch_size=int(batch_size), name="call_index"
    )
    if any(index < 0 for index in call_indices):
        raise ValueError("call_index must be non-negative")
    return _ValidatedTargetSelection(
        input_ids=normalized_ids,
        rows=rows,
        trajectories=trajectories,
        call_indices=call_indices,
        batched=batched,
        digest=digest,
    )


def _bind_canonical_input_ids(
    input_ids: torch.Tensor,
    canonical_input_ids: Sequence[int] | Sequence[Sequence[int]],
) -> _CanonicalInputIdsProof:
    """Bind trusted CPU IDs to one tensor without reading a CUDA tensor.

    Native materialization calls this only after the processor-owned expansion
    has produced the CPU token tuple (or after that tuple created the tensor).
    CPU callers are checked directly.  The version/identity contract then
    rejects replacement or in-place mutation before provider consumption.
    """

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim not in (1, 2):
        raise ValueError("input_ids must have shape [S] or [B, S]")
    if input_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("input_ids must use an integer dtype")
    batched = input_ids.ndim == 2
    normalized = input_ids if batched else input_ids.unsqueeze(0)
    batch_size, sequence_length = normalized.shape
    if batch_size == 0 or sequence_length == 0:
        raise ValueError("input_ids batch and sequence dimensions must be non-empty")
    rows = _normalize_canonical_input_rows(
        canonical_input_ids,
        batched=batched,
        batch_size=int(batch_size),
        sequence_length=int(sequence_length),
    )
    if input_ids.device.type == "cpu":
        actual_rows = tuple(
            tuple(int(token_id) for token_id in row)
            for row in normalized.detach().to(dtype=torch.int64).tolist()
        )
        if actual_rows != rows:
            raise ValueError("canonical input IDs differ from the bound CPU tensor")
    return _CanonicalInputIdsProof(
        rows=rows,
        digest=_hash_input_id_rows(rows),
        tensor_identity=id(input_ids),
        tensor_data_ptr=input_ids.data_ptr(),
        tensor_version=input_ids._version,
        tensor_shape=tuple(input_ids.shape),
        tensor_stride=tuple(input_ids.stride()),
        tensor_storage_offset=input_ids.storage_offset(),
        tensor_device=input_ids.device,
        tensor_dtype=input_ids.dtype,
        seal=_CANONICAL_INPUT_IDS_PROOF_SEAL,
    )


def _validate_canonical_input_ids_proof(
    proof: _CanonicalInputIdsProof,
    *,
    input_ids: torch.Tensor,
    batched: bool,
    batch_size: int,
    sequence_length: int,
) -> tuple[tuple[tuple[int, ...], ...], str]:
    if not isinstance(proof, _CanonicalInputIdsProof):
        raise TypeError("canonical_input_ids_proof must be a bound canonical proof")
    if proof.seal is not _CANONICAL_INPUT_IDS_PROOF_SEAL:
        raise ValueError("canonical input-ID proof seal changed")
    tensor_contract = (
        id(input_ids),
        input_ids.data_ptr(),
        input_ids._version,
        tuple(input_ids.shape),
        tuple(input_ids.stride()),
        input_ids.storage_offset(),
        input_ids.device,
        input_ids.dtype,
    )
    expected_contract = (
        proof.tensor_identity,
        proof.tensor_data_ptr,
        proof.tensor_version,
        proof.tensor_shape,
        proof.tensor_stride,
        proof.tensor_storage_offset,
        proof.tensor_device,
        proof.tensor_dtype,
    )
    if tensor_contract != expected_contract:
        raise ValueError("canonical input-ID proof does not bind this tensor state")
    rows = proof.rows
    if len(rows) != batch_size or any(len(row) != sequence_length for row in rows):
        raise ValueError("canonical input-ID proof shape differs from input_ids")
    if (not batched) != (len(proof.tensor_shape) == 1):
        raise ValueError("canonical input-ID proof batching differs from input_ids")
    if proof.digest != _hash_input_id_rows(rows):
        raise ValueError("canonical input-ID proof digest changed")
    return rows, proof.digest


def _normalize_canonical_input_rows(
    values: Sequence[int] | Sequence[Sequence[int]],
    *,
    batched: bool,
    batch_size: int,
    sequence_length: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(values, torch.Tensor) or isinstance(values, (str, bytes)):
        raise TypeError("canonical_input_ids must be CPU integer sequences")
    items = tuple(values)
    if not items:
        raise ValueError("canonical_input_ids must be non-empty")
    if not batched:
        if any(not isinstance(item, int) or isinstance(item, bool) for item in items):
            raise TypeError("canonical_input_ids must contain integers")
        rows = (tuple(int(item) for item in items),)
    else:
        materialized: list[tuple[int, ...]] = []
        for row in items:
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise TypeError("batched canonical_input_ids require integer rows")
            normalized_row = tuple(row)
            if any(
                not isinstance(item, int) or isinstance(item, bool)
                for item in normalized_row
            ):
                raise TypeError("canonical_input_ids must contain integers")
            materialized.append(tuple(int(item) for item in normalized_row))
        rows = tuple(materialized)
    if len(rows) != batch_size or any(len(row) != sequence_length for row in rows):
        raise ValueError("canonical_input_ids shape must match input_ids")
    return rows


def _normalize_expected_rows(
    values: Sequence[int] | Sequence[Sequence[int]],
    *,
    batch_size: int,
    span_length: int,
) -> tuple[tuple[int, ...], ...]:
    items = tuple(values)
    if not items:
        raise ValueError("expected_target_token_ids must be non-empty")
    if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        if batch_size != 1:
            raise ValueError(
                "batched input requires one expected target-token row per item"
            )
        rows = (tuple(int(item) for item in items),)
    else:
        rows_list: list[tuple[int, ...]] = []
        for row in items:
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise TypeError(
                    "expected target-token rows must contain integer sequences"
                )
            normalized_row = tuple(row)
            if any(
                not isinstance(item, int) or isinstance(item, bool)
                for item in normalized_row
            ):
                raise TypeError("expected target token IDs must be integers")
            rows_list.append(tuple(int(item) for item in normalized_row))
        rows = tuple(rows_list)
    if len(rows) != batch_size:
        raise ValueError("expected target-token row count must match batch size")
    if any(len(row) != span_length for row in rows):
        raise ValueError("expected target-token rows must exactly cover target_span")
    if any(token_id < 0 for row in rows for token_id in row):
        raise ValueError("expected target token IDs must be non-negative")
    return rows


def _normalize_string_rows(
    value: str | Sequence[str], *, batch_size: int, name: str
) -> tuple[str, ...]:
    if isinstance(value, str):
        if batch_size != 1:
            raise ValueError(f"batched input requires one {name} per item")
        rows = (value,)
    else:
        rows = tuple(value)
    if len(rows) != batch_size or any(
        not isinstance(item, str) or not item for item in rows
    ):
        raise ValueError(f"{name} values must be non-empty strings matching batch size")
    return rows


def _normalize_integer_rows(
    value: int | Sequence[int], *, batch_size: int, name: str
) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        if batch_size != 1:
            raise ValueError(f"batched input requires one {name} per item")
        rows = (int(value),)
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{name} must contain integers")
        rows = tuple(value)
    if len(rows) != batch_size or any(
        not isinstance(item, int) or isinstance(item, bool) for item in rows
    ):
        raise ValueError(f"{name} values must be integers matching batch size")
    return tuple(int(item) for item in rows)


def _hash_input_ids(input_ids: torch.Tensor) -> str:
    canonical = input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    rows = tuple(tuple(int(token_id) for token_id in row) for row in canonical.tolist())
    return _hash_input_id_rows(rows)


def _hash_input_id_rows(rows: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<II", len(rows), len(rows[0])))
    for row in rows:
        for token_id in row:
            digest.update(struct.pack("<q", token_id))
    return digest.hexdigest()


__all__ = [
    "CONTEXTUAL_HIDDEN_STATE",
    "TARGET_CONDITIONING_SCHEMA_VERSION",
    "TARGET_TOKEN_EMBEDDING",
    "BoundTargetConditionProvider",
    "TargetConditioningConfig",
    "TargetConditionProvider",
    "TargetConditioningOutput",
    "TargetConditioningProviderKind",
    "TargetConditioningProvenance",
    "TargetConditioningRequest",
]
