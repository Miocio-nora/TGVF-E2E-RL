"""Content-bound metrics history used by representation checkpoints.

The JSONL file is part of resumable state rather than an informal log.  A
checkpoint binds the exact durable byte prefix that preceded its save, along
with the next validation cursor.  Restore can therefore reject truncated,
advanced, forked, or semantically reordered history before applying model or
optimizer state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile


REPRESENTATION_METRICS_HISTORY_SCHEMA_VERSION = "representation-metrics-history-v1"

_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class RepresentationMetricsHistoryIdentity:
    run_id: str
    run_identity_sha256: str
    checkpoint_global_step: int
    raw_bytes_sha256: str
    byte_count: int
    line_count: int
    next_validation_event_index: int
    schema_version: str = REPRESENTATION_METRICS_HISTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty(self.run_id, field_name="run_id")
        _digest(self.run_identity_sha256, field_name="run_identity_sha256")
        _non_negative_int(
            self.checkpoint_global_step,
            field_name="checkpoint_global_step",
        )
        _digest(self.raw_bytes_sha256, field_name="raw_bytes_sha256")
        _positive_int(self.byte_count, field_name="byte_count")
        _positive_int(self.line_count, field_name="line_count")
        _non_negative_int(
            self.next_validation_event_index,
            field_name="next_validation_event_index",
        )
        if self.schema_version != REPRESENTATION_METRICS_HISTORY_SCHEMA_VERSION:
            raise ValueError("representation metrics-history schema mismatch")

    @property
    def identity_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_identity_sha256": self.run_identity_sha256,
            "checkpoint_global_step": self.checkpoint_global_step,
            "raw_bytes_sha256": self.raw_bytes_sha256,
            "byte_count": self.byte_count,
            "line_count": self.line_count,
            "next_validation_event_index": self.next_validation_event_index,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ParsedRepresentationMetricsHistory:
    identity: RepresentationMetricsHistoryIdentity
    records: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RepresentationMetricsHistoryIdentity):
            raise TypeError("history identity must be typed")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("history records must be a non-empty tuple")


@dataclass(frozen=True, slots=True)
class RepresentationMetricsHistoryRecoveryResult:
    """Auditable outcome of reconciling a live ledger with a checkpoint.

    ``committed_history`` is the exact byte prefix bound into the checkpoint.
    When ``archive_path`` is present, that path contains the complete live
    ledger observed before rollback, including its uncommitted suffix.
    """

    active_path: Path
    committed_history: ParsedRepresentationMetricsHistory
    observed_raw_bytes_sha256: str
    observed_byte_count: int
    suffix_byte_count: int
    suffix_raw_bytes_sha256: str | None
    archive_path: Path | None

    def __post_init__(self) -> None:
        if not isinstance(self.active_path, Path):
            raise TypeError("metrics-history recovery active_path must be a Path")
        if not isinstance(
            self.committed_history, ParsedRepresentationMetricsHistory
        ):
            raise TypeError("metrics-history recovery must carry parsed history")
        _digest(
            self.observed_raw_bytes_sha256,
            field_name="observed_raw_bytes_sha256",
        )
        _positive_int(self.observed_byte_count, field_name="observed_byte_count")
        _non_negative_int(self.suffix_byte_count, field_name="suffix_byte_count")
        committed_bytes = self.committed_history.identity.byte_count
        if self.observed_byte_count != committed_bytes + self.suffix_byte_count:
            raise ValueError("metrics-history recovery byte counts do not align")
        if (self.suffix_raw_bytes_sha256 is None) != (self.suffix_byte_count == 0):
            raise ValueError("metrics-history recovery suffix digest does not align")
        if self.suffix_raw_bytes_sha256 is not None:
            _digest(
                self.suffix_raw_bytes_sha256,
                field_name="suffix_raw_bytes_sha256",
            )
        if (self.archive_path is None) != (self.suffix_byte_count == 0):
            raise ValueError("metrics-history recovery archive does not align")
        if self.archive_path is not None and not isinstance(self.archive_path, Path):
            raise TypeError("metrics-history recovery archive_path must be a Path")

    @property
    def rolled_back(self) -> bool:
        return self.archive_path is not None


def load_representation_metrics_history(
    path: str | Path,
    *,
    run_id: str,
    run_identity_sha256: str,
    checkpoint_global_step: int,
    runner_schema_version: str,
) -> ParsedRepresentationMetricsHistory:
    """Read and validate the exact unfinished JSONL prefix for a checkpoint."""

    _non_empty(run_id, field_name="run_id")
    _digest(run_identity_sha256, field_name="run_identity_sha256")
    _non_negative_int(
        checkpoint_global_step,
        field_name="checkpoint_global_step",
    )
    _non_empty(runner_schema_version, field_name="runner_schema_version")
    raw = Path(path).read_bytes()
    return _parse_representation_metrics_history(
        raw,
        run_id=run_id,
        run_identity_sha256=run_identity_sha256,
        checkpoint_global_step=checkpoint_global_step,
        runner_schema_version=runner_schema_version,
    )


def recover_representation_metrics_history_prefix(
    path: str | Path,
    *,
    checkpoint_identity: RepresentationMetricsHistoryIdentity,
    runner_schema_version: str,
) -> RepresentationMetricsHistoryRecoveryResult:
    """Restore a live JSONL ledger to its checkpoint-bound WAL prefix.

    A checkpoint commits the first ``byte_count`` bytes and their SHA256.  Any
    later bytes are an uncommitted write-ahead-log suffix.  The committed
    prefix is verified semantically and byte-for-byte before any mutation.  A
    suffix is preserved through a deterministic, no-overwrite hard-link in the
    same directory before the active path is atomically replaced with the
    committed prefix.

    The caller must have exclusive ownership of the metrics path while this
    recovery operation runs.
    """

    if not isinstance(checkpoint_identity, RepresentationMetricsHistoryIdentity):
        raise TypeError("checkpoint_identity must be a metrics-history identity")
    checkpoint_identity.__post_init__()
    _non_empty(runner_schema_version, field_name="runner_schema_version")

    active_path = Path(path)
    if active_path.is_symlink():
        raise ValueError("metrics-history recovery refuses a symlink active path")
    if not active_path.is_file():
        raise FileNotFoundError(f"metrics-history file does not exist: {active_path}")
    observed = active_path.read_bytes()
    committed_byte_count = checkpoint_identity.byte_count
    if len(observed) < committed_byte_count:
        raise ValueError(
            "live metrics JSONL is shorter than the checkpoint-bound prefix"
        )
    committed = observed[:committed_byte_count]
    if sha256(committed).hexdigest() != checkpoint_identity.raw_bytes_sha256:
        raise ValueError("checkpoint-bound metrics prefix SHA256 mismatch")

    parsed = _parse_representation_metrics_history(
        committed,
        run_id=checkpoint_identity.run_id,
        run_identity_sha256=checkpoint_identity.run_identity_sha256,
        checkpoint_global_step=checkpoint_identity.checkpoint_global_step,
        runner_schema_version=runner_schema_version,
    )
    if parsed.identity != checkpoint_identity:
        raise ValueError("checkpoint-bound metrics prefix identity mismatch")

    suffix = observed[committed_byte_count:]
    observed_digest = sha256(observed).hexdigest()
    if not suffix:
        return RepresentationMetricsHistoryRecoveryResult(
            active_path=active_path,
            committed_history=parsed,
            observed_raw_bytes_sha256=observed_digest,
            observed_byte_count=len(observed),
            suffix_byte_count=0,
            suffix_raw_bytes_sha256=None,
            archive_path=None,
        )

    archive_path = _metrics_history_archive_path(active_path, checkpoint_identity)
    mode = active_path.stat(follow_symlinks=False).st_mode & 0o777
    temporary_path = _write_temporary_file(
        active_path.parent,
        name_prefix=f".{active_path.name}.rollback-",
        raw=committed,
        mode=mode,
    )
    archive_published = False
    try:
        # A same-directory hard link is atomic and refuses to overwrite an
        # existing audit archive.  Once the active name is replaced below, the
        # hard link remains the complete pre-rollback ledger.
        os.link(active_path, archive_path, follow_symlinks=False)
        archive_published = True
        _fsync_directory(active_path.parent)
        os.replace(temporary_path, active_path)
        _fsync_directory(active_path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    if not archive_published:  # pragma: no cover - os.link either returns or raises
        raise RuntimeError("metrics-history archive was not published")
    archived = archive_path.read_bytes()
    if archived != observed:
        raise RuntimeError("metrics-history archive changed during exclusive recovery")
    if active_path.read_bytes() != committed:
        raise RuntimeError("metrics-history active prefix rollback verification failed")
    return RepresentationMetricsHistoryRecoveryResult(
        active_path=active_path,
        committed_history=parsed,
        observed_raw_bytes_sha256=observed_digest,
        observed_byte_count=len(observed),
        suffix_byte_count=len(suffix),
        suffix_raw_bytes_sha256=sha256(suffix).hexdigest(),
        archive_path=archive_path,
    )


def _parse_representation_metrics_history(
    raw: bytes,
    *,
    run_id: str,
    run_identity_sha256: str,
    checkpoint_global_step: int,
    runner_schema_version: str,
) -> ParsedRepresentationMetricsHistory:
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("metrics JSONL must be non-empty and end with a newline")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("metrics JSONL must be valid UTF-8") from error

    records: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"metrics JSONL line {line_number} is empty")
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"metrics JSONL line {line_number} is not strict JSON"
            ) from error
        if not isinstance(value, dict):
            raise TypeError(f"metrics JSONL line {line_number} is not an object")
        records.append(value)

    first = records[0]
    if first.get("event") != "start":
        raise ValueError("metrics history must begin with a start event")
    if first.get("schema_version") != runner_schema_version:
        raise ValueError("metrics start event has a different runner schema")
    if first.get("run_id") != run_id:
        raise ValueError("metrics start event has a different run_id")
    if first.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError("metrics start event has a different run identity")

    previous_step = -1
    checkpoint_train_events = 0
    validation_indices: list[int] = []
    for line_number, record in enumerate(records, start=1):
        _validate_record_identity(
            record,
            line_number=line_number,
            run_id=run_id,
            run_identity_sha256=run_identity_sha256,
        )
        event = record.get("event")
        if not isinstance(event, str) or not event:
            raise ValueError(f"metrics JSONL line {line_number} has no event")
        if event in {"complete", "paused"}:
            raise ValueError(
                "checkpoint-bound metrics history cannot contain a terminal or "
                "operational pause event"
            )
        step = record.get("global_step")
        if step is not None:
            _non_negative_int(step, field_name=f"line {line_number} global_step")
            if step < previous_step:
                raise ValueError("metrics global_step values must be non-decreasing")
            if step > checkpoint_global_step:
                raise ValueError("metrics history is advanced beyond the checkpoint")
            previous_step = step
        initial_step = record.get("initial_global_step")
        if initial_step is not None:
            _non_negative_int(
                initial_step,
                field_name=f"line {line_number} initial_global_step",
            )
            if initial_step > checkpoint_global_step:
                raise ValueError("metrics initial step is beyond the checkpoint")
        if event == "train":
            if step is None:
                raise ValueError("every train event must carry global_step")
            if step == checkpoint_global_step:
                checkpoint_train_events += 1
        if event == "validation":
            index = record.get("validation_event_index")
            _non_negative_int(
                index,
                field_name=f"line {line_number} validation_event_index",
            )
            validation_indices.append(index)

    if checkpoint_train_events != 1:
        raise ValueError(
            "metrics history must contain exactly one train event at the checkpoint"
        )
    expected_indices = list(range(len(validation_indices)))
    if validation_indices != expected_indices:
        raise ValueError(
            "validation event indices must be complete, unique, and contiguous"
        )
    identity = RepresentationMetricsHistoryIdentity(
        run_id=run_id,
        run_identity_sha256=run_identity_sha256,
        checkpoint_global_step=checkpoint_global_step,
        raw_bytes_sha256=sha256(raw).hexdigest(),
        byte_count=len(raw),
        line_count=len(records),
        next_validation_event_index=len(validation_indices),
    )
    return ParsedRepresentationMetricsHistory(identity, tuple(records))


def _metrics_history_archive_path(
    active_path: Path,
    checkpoint_identity: RepresentationMetricsHistoryIdentity,
) -> Path:
    return active_path.with_name(
        f"{active_path.name}.uncommitted-after-step-"
        f"{checkpoint_identity.checkpoint_global_step:08d}."
        f"{checkpoint_identity.identity_sha256}.jsonl"
    )


def _write_temporary_file(
    directory: Path,
    *,
    name_prefix: str,
    raw: bytes,
    mode: int,
) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=name_prefix, dir=directory)
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return path


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_record_identity(
    record: Mapping[str, object],
    *,
    line_number: int,
    run_id: str,
    run_identity_sha256: str,
) -> None:
    if "run_id" in record and record["run_id"] != run_id:
        raise ValueError(f"metrics JSONL line {line_number} changes run_id")
    if (
        "run_identity_sha256" in record
        and record["run_identity_sha256"] != run_identity_sha256
    ):
        raise ValueError(f"metrics JSONL line {line_number} changes run identity")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _non_empty(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _digest(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")


def _non_negative_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


__all__ = [
    "ParsedRepresentationMetricsHistory",
    "REPRESENTATION_METRICS_HISTORY_SCHEMA_VERSION",
    "RepresentationMetricsHistoryIdentity",
    "RepresentationMetricsHistoryRecoveryResult",
    "load_representation_metrics_history",
    "recover_representation_metrics_history_prefix",
]
