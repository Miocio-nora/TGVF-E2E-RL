"""Schemas and identity validation for distributed representation checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.public_api_compat import (
    freeze_public_class_annotations,
    rebind_public_class,
    rebind_public_function,
)

from .checkpoint import (
    REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION,
    REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3,
    RepresentationRunIdentity,
    RepresentationRunIdentityV3,
)
from .history import RepresentationMetricsHistoryIdentity


DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION = (
    "distributed-representation-checkpoint-v1"
)
DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2 = (
    "distributed-representation-checkpoint-v2"
)
DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION = (
    "distributed-representation-rank-state-v1"
)
_BORROWED_QWEN_PREFIXES = (
    "main_projection.",
    "d_deepstack_projections.",
)
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class DistributedRepresentationRankState:
    rank: int
    sampler_identity_sha256: str
    sampler_state: dict[str, object]
    sampler_state_sha256: str
    rng_state: dict[str, object]
    rng_state_sha256: str
    scheduler_type: str | None
    scheduler_state: dict[str, object] | None
    scheduler_state_sha256: str | None
    schema_version: str = DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_negative_int(self.rank, field_name="rank")
        _sha256(self.sampler_identity_sha256, field_name="sampler_identity_sha256")
        if not isinstance(self.sampler_state, dict):
            raise TypeError("rank sampler state must be a dict")
        _sha256(self.sampler_state_sha256, field_name="sampler_state_sha256")
        if state_digest(self.sampler_state) != self.sampler_state_sha256:
            raise ValueError("rank sampler state digest mismatch")
        if self.sampler_state.get("identity_sha256") != self.sampler_identity_sha256:
            raise ValueError("rank sampler state carries a different identity")
        if not isinstance(self.rng_state, dict):
            raise TypeError("rank RNG state must be a dict")
        _sha256(self.rng_state_sha256, field_name="rng_state_sha256")
        if state_digest(self.rng_state) != self.rng_state_sha256:
            raise ValueError("rank RNG state digest mismatch")
        if (self.scheduler_type is None) != (self.scheduler_state is None):
            raise ValueError("rank scheduler type/state presence must align")
        if (self.scheduler_type is None) != (self.scheduler_state_sha256 is None):
            raise ValueError("rank scheduler type/digest presence must align")
        if self.scheduler_type is not None:
            _non_empty_text(self.scheduler_type, field_name="scheduler_type")
            if not isinstance(self.scheduler_state, dict):
                raise TypeError("rank scheduler state must be a dict")
            _sha256(
                self.scheduler_state_sha256,
                field_name="scheduler_state_sha256",
            )
            if state_digest(self.scheduler_state) != self.scheduler_state_sha256:
                raise ValueError("rank scheduler state digest mismatch")
        if self.schema_version != DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION:
            raise ValueError("distributed rank-state schema mismatch")


@dataclass(frozen=True, slots=True)
class DistributedRepresentationCheckpointManifest:
    run_identity: RepresentationRunIdentity
    run_identity_sha256: str
    global_step: int
    world_size: int
    fsdp_reshard_after_forward: bool
    owned_state_names: tuple[str, ...]
    optimizer_type: str
    optimizer_identity_sha256: str
    accumulation_identity_sha256: str
    trainer_execution_identity_sha256: str
    sampler_contract_identity_sha256: str
    scheduler_identity_sha256: str | None
    rank_state_sha256: tuple[str, ...]
    model_local_shard_sha256: tuple[str, ...]
    optimizer_local_shard_sha256: tuple[str, ...]
    torch_version: str
    schema_version: str = DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION
    metrics_history: RepresentationMetricsHistoryIdentity | None = None
    metrics_history_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_run_identity(self.run_identity)
        _sha256(self.run_identity_sha256, field_name="run_identity_sha256")
        if self.run_identity_sha256 != self.run_identity.identity_sha256:
            raise ValueError("distributed checkpoint run identity digest mismatch")
        _non_negative_int(self.global_step, field_name="global_step")
        _positive_int(self.world_size, field_name="world_size")
        if not isinstance(self.fsdp_reshard_after_forward, bool):
            raise TypeError("fsdp_reshard_after_forward must be bool")
        _sorted_unique_names(self.owned_state_names, field_name="owned_state_names")
        if any(
            name.startswith(_BORROWED_QWEN_PREFIXES) for name in self.owned_state_names
        ):
            raise ValueError("borrowed Qwen state is forbidden in DCP manifest")
        _non_empty_text(self.optimizer_type, field_name="optimizer_type")
        _sha256(self.optimizer_identity_sha256, field_name="optimizer_identity_sha256")
        if (
            self.optimizer_identity_sha256
            != self.run_identity.optimizer.identity_sha256
        ):
            raise ValueError("distributed optimizer identity digest mismatch")
        _sha256(
            self.accumulation_identity_sha256,
            field_name="accumulation_identity_sha256",
        )
        if (
            self.accumulation_identity_sha256
            != self.run_identity.accumulation.identity_sha256
        ):
            raise ValueError("distributed accumulation identity digest mismatch")
        _sha256(
            self.trainer_execution_identity_sha256,
            field_name="trainer_execution_identity_sha256",
        )
        if (
            self.trainer_execution_identity_sha256
            != self.run_identity.trainer_execution.identity_sha256
        ):
            raise ValueError("distributed trainer execution identity digest mismatch")
        _sha256(
            self.sampler_contract_identity_sha256,
            field_name="sampler_contract_identity_sha256",
        )
        if (
            self.sampler_contract_identity_sha256
            != self.run_identity.sampler_contract.identity_sha256
        ):
            raise ValueError("distributed sampler contract identity digest mismatch")
        if (
            self.scheduler_identity_sha256
            != self.run_identity.scheduler_identity_sha256
        ):
            raise ValueError("distributed scheduler identity digest mismatch")
        if len(self.rank_state_sha256) != self.world_size:
            raise ValueError("one rank-state digest is required per FSDP rank")
        for digest in self.rank_state_sha256:
            _sha256(digest, field_name="rank_state_sha256")
        if len(set(self.rank_state_sha256)) != len(self.rank_state_sha256):
            raise ValueError("rank-state digests must be rank-specific")
        for field_name, digests in (
            ("model_local_shard_sha256", self.model_local_shard_sha256),
            ("optimizer_local_shard_sha256", self.optimizer_local_shard_sha256),
        ):
            if not isinstance(digests, tuple) or len(digests) != self.world_size:
                raise ValueError(f"{field_name} must contain one digest per FSDP rank")
            for digest in digests:
                _sha256(digest, field_name=field_name)
        _non_empty_text(self.torch_version, field_name="torch_version")
        metrics_history = getattr(self, "metrics_history", None)
        metrics_history_identity_sha256 = getattr(
            self,
            "metrics_history_identity_sha256",
            None,
        )
        if self.schema_version == DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION:
            if (
                metrics_history is not None
                or metrics_history_identity_sha256 is not None
            ):
                raise ValueError(
                    "distributed checkpoint v1 cannot bind metrics history"
                )
        elif (
            self.schema_version
            == DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2
        ):
            _validate_metrics_history_binding(
                metrics_history,
                expected_run_identity=self.run_identity,
                expected_global_step=self.global_step,
            )
            assert isinstance(metrics_history, RepresentationMetricsHistoryIdentity)
            _sha256(
                metrics_history_identity_sha256,
                field_name="metrics_history_identity_sha256",
            )
            if metrics_history_identity_sha256 != metrics_history.identity_sha256:
                raise ValueError("distributed metrics-history digest mismatch")
        else:
            raise ValueError("distributed representation checkpoint schema mismatch")


@dataclass(frozen=True, slots=True)
class DistributedRepresentationMetadata:
    manifest: DistributedRepresentationCheckpointManifest
    rank_states: tuple[DistributedRepresentationRankState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, DistributedRepresentationCheckpointManifest):
            raise TypeError("distributed metadata manifest has the wrong type")
        self.manifest.__post_init__()
        for record in self.rank_states:
            if not isinstance(record, DistributedRepresentationRankState):
                raise TypeError("distributed metadata rank state has the wrong type")
            record.__post_init__()
        if len(self.rank_states) != self.manifest.world_size:
            raise ValueError("distributed metadata rank-state count mismatch")
        ranks = tuple(record.rank for record in self.rank_states)
        if ranks != tuple(range(self.manifest.world_size)):
            raise ValueError("distributed rank states must be sorted and complete")
        digests = tuple(_rank_state_digest(record) for record in self.rank_states)
        if digests != self.manifest.rank_state_sha256:
            raise ValueError("distributed rank-state manifest digest mismatch")
        sampler_identities = tuple(
            record.sampler_identity_sha256 for record in self.rank_states
        )
        if len(set(sampler_identities)) != len(sampler_identities):
            raise ValueError("rank-local sampler identities must be unique")
        scheduler_digests = {
            record.scheduler_state_sha256 for record in self.rank_states
        }
        if len(scheduler_digests) != 1:
            raise ValueError("scheduler state must agree across all ranks")


@dataclass(frozen=True, slots=True)
class DistributedRepresentationResumeResult:
    global_step: int
    next_global_step: int
    run_identity_sha256: str
    exact: bool = True
    next_validation_event_index: int | None = None


def _rank_state_digest(record: DistributedRepresentationRankState) -> str:
    payload = {
        "schema_version": record.schema_version,
        "rank": record.rank,
        "sampler_identity_sha256": record.sampler_identity_sha256,
        "sampler_state_sha256": record.sampler_state_sha256,
        "rng_state_sha256": record.rng_state_sha256,
        "scheduler_type": record.scheduler_type,
        "scheduler_state_sha256": record.scheduler_state_sha256,
    }
    return state_digest(payload)


def _validate_run_identity(identity: object) -> None:
    if not isinstance(identity, RepresentationRunIdentity):
        raise TypeError("run identity must be a RepresentationRunIdentity")
    if type(identity) is RepresentationRunIdentity:
        expected_schema_version = REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION
    elif type(identity) is RepresentationRunIdentityV3:
        expected_schema_version = REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3
    else:
        raise TypeError("unsupported representation run identity type")
    if identity.schema_version != expected_schema_version:
        raise ValueError("representation run identity schema mismatch")
    identity.__post_init__()
    # Re-hashing traverses every nested identity and fails on unsupported drift.
    _sha256(identity.identity_sha256, field_name="run identity digest")


def _validate_metrics_history_binding(
    metrics_history: object,
    *,
    expected_run_identity: RepresentationRunIdentity,
    expected_global_step: int,
) -> None:
    if not isinstance(metrics_history, RepresentationMetricsHistoryIdentity):
        raise TypeError(
            "metrics_history must be a RepresentationMetricsHistoryIdentity"
        )
    metrics_history.__post_init__()
    if metrics_history.run_id != expected_run_identity.run_id:
        raise IdentityMismatchError("metrics history run_id mismatch")
    if metrics_history.run_identity_sha256 != expected_run_identity.identity_sha256:
        raise IdentityMismatchError("metrics history run identity mismatch")
    if metrics_history.checkpoint_global_step != expected_global_step:
        raise IdentityMismatchError("metrics history checkpoint step mismatch")


def _validate_expected_metrics_history(
    manifest: DistributedRepresentationCheckpointManifest,
    *,
    expected_metrics_history: RepresentationMetricsHistoryIdentity | None,
    expected_run_identity: RepresentationRunIdentity,
) -> None:
    recorded = getattr(manifest, "metrics_history", None)
    recorded_digest = getattr(manifest, "metrics_history_identity_sha256", None)
    if manifest.schema_version == DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION:
        if expected_metrics_history is not None:
            raise IdentityMismatchError(
                "distributed checkpoint v1 has no metrics-history binding"
            )
        return
    if (
        manifest.schema_version
        != DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2
    ):
        raise ValueError("distributed representation checkpoint schema mismatch")
    if expected_metrics_history is None:
        raise ReplayMismatchError(
            "distributed checkpoint v2 requires expected metrics history"
        )
    _validate_metrics_history_binding(
        expected_metrics_history,
        expected_run_identity=expected_run_identity,
        expected_global_step=manifest.global_step,
    )
    if (
        recorded != expected_metrics_history
        or recorded_digest != expected_metrics_history.identity_sha256
    ):
        raise ReplayMismatchError("distributed checkpoint metrics history mismatch")


def _non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _sha256(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
        raise ValueError(f"{field_name} must be a lowercase SHA256")


def _positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _non_negative_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _sorted_unique_names(values: object, *, field_name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be sorted and unique")


_DISTRIBUTED_CHECKPOINT_SCHEMA_TYPES = (
    DistributedRepresentationRankState,
    DistributedRepresentationCheckpointManifest,
    DistributedRepresentationMetadata,
    DistributedRepresentationResumeResult,
)
_DISTRIBUTED_CHECKPOINT_SCHEMA_FUNCTIONS = (
    _rank_state_digest,
    _validate_run_identity,
    _validate_metrics_history_binding,
    _validate_expected_metrics_history,
    _non_empty_text,
    _sha256,
    _positive_int,
    _non_negative_int,
    _sorted_unique_names,
)
_LEGACY_PUBLIC_MODULE = "tgvf_rl.representation.training.distributed_checkpoint"

for _schema_type in (
    DistributedRepresentationCheckpointManifest,
    DistributedRepresentationMetadata,
):
    freeze_public_class_annotations(
        _schema_type,
        implementation_globals=globals(),
    )
for _schema_type in _DISTRIBUTED_CHECKPOINT_SCHEMA_TYPES:
    rebind_public_class(
        _schema_type,
        implementation_module=__name__,
        public_module=_LEGACY_PUBLIC_MODULE,
    )
del _schema_type

for _schema_function in _DISTRIBUTED_CHECKPOINT_SCHEMA_FUNCTIONS:
    rebind_public_function(
        _schema_function,
        implementation_module=__name__,
        public_module=_LEGACY_PUBLIC_MODULE,
    )
del _schema_function

__all__ = [
    "DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION",
    "DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2",
    "DISTRIBUTED_REPRESENTATION_RANK_STATE_SCHEMA_VERSION",
    "DistributedRepresentationCheckpointManifest",
    "DistributedRepresentationMetadata",
    "DistributedRepresentationRankState",
    "DistributedRepresentationResumeResult",
]
