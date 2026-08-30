"""Fail-closed launch authorization and bounded prerequisite waiting.

The gate deliberately separates four operations:

1. a controller proves its prerequisite and publishes ``ready.json``;
2. an operator explicitly issues a short-lived authorization token;
3. the controller atomically consumes that token immediately before launch;
4. a durable consumption receipt prevents replay, even from a copied token.

Ready receipts and tokens are bound to a canonical run identity.  Artifact
evidence is hashed when readiness is published and revalidated at consume
time.  All mutable coordination happens under one filesystem lock.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import getpass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import secrets
import socket
import time
from typing import Any, Callable, Iterator, Mapping


READY_SCHEMA = "tgvf-launch-ready-v1"
AUTHORIZATION_SCHEMA = "tgvf-launch-authorization-v1"
CONSUMPTION_SCHEMA = "tgvf-launch-authorization-consumption-v1"
LIVENESS_SCHEMA = "tgvf-process-liveness-v1"
EXECUTION_POLICY_SCHEMA = "tgvf-experiment-execution-policy-v2"
FREEZE_OVERRIDE_SCHEMA = "tgvf-experiment-freeze-override-v1"
FREEZE_OVERRIDE_CONSUMPTION_SCHEMA = "tgvf-freeze-override-consumption-v1"
MAX_AUTHORIZATION_TTL_SECONDS = 24 * 60 * 60
MAX_FREEZE_OVERRIDE_TTL_SECONDS = 60 * 60
MAX_WAIT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
_TOKEN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LaunchGateError(RuntimeError):
    """Base class for a launch gate refusal."""


class LaunchAuthorizationError(LaunchGateError):
    """Raised when an authorization is absent, invalid, or already consumed."""


class LaunchLivenessError(LaunchGateError):
    """Raised when the process expected to produce an artifact is no longer live."""


class LaunchTimeoutError(LaunchGateError):
    """Raised when a bounded prerequisite wait expires."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise LaunchGateError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise LaunchGateError(f"{field} is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LaunchGateError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LaunchGateError("launch identity must be finite JSON data") from error


def _sha256_value(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    if path.is_symlink():
        raise LaunchGateError(f"{label} must not be a symlink: {path}")
    try:
        stat = path.stat()
    except FileNotFoundError as error:
        raise LaunchGateError(f"{label} is missing: {path}") from error
    if not path.is_file():
        raise LaunchGateError(f"{label} must be a regular file: {path}")
    return stat


def _sha256_file(path: Path) -> str:
    _regular_file(path, label="artifact")
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_gate_directory(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.exists() and path.is_symlink():
        raise LaunchGateError(f"gate directory must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise LaunchGateError(f"gate path is not a directory: {path}")
    return path


def _write_json_exclusive(path: Path, value: object, *, mode: int) -> None:
    if path.is_symlink():
        raise LaunchGateError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError:
        raise
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _write_json_atomic(path: Path, value: object, *, mode: int) -> None:
    if path.is_symlink():
        raise LaunchGateError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    try:
        _write_json_exclusive(temporary, value, mode=mode)
        os.replace(temporary, path)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchGateError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise LaunchGateError(f"{label} must contain one JSON object: {path}")
    return value


@contextmanager
def _gate_lock(gate_directory: Path) -> Iterator[None]:
    lock_path = gate_directory / ".launch-gate.lock"
    if lock_path.is_symlink():
        raise LaunchGateError(f"launch gate lock must not be a symlink: {lock_path}")
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def make_run_identity(
    *,
    run_id: str,
    phase: str,
    command_id: str,
    parameters: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build a canonical identity for exactly one launchable phase."""

    for field, value in (
        ("run_id", run_id),
        ("phase", phase),
        ("command_id", command_id),
    ):
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise LaunchGateError(f"{field} must be a non-empty string without NUL")
    body: dict[str, Any] = {
        "run_id": run_id,
        "phase": phase,
        "command_id": command_id,
        "parameters": dict(parameters or {}),
    }
    identity_sha256 = _sha256_value(body)
    return {**body, "identity_sha256": identity_sha256}


def _validate_run_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LaunchGateError("run_identity must be one JSON object")
    if set(value) != {
        "run_id",
        "phase",
        "command_id",
        "parameters",
        "identity_sha256",
    }:
        raise LaunchGateError("run_identity has an unexpected field set")
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise LaunchGateError("run_identity.parameters must be one JSON object")
    expected = make_run_identity(
        run_id=value.get("run_id"),
        phase=value.get("phase"),
        command_id=value.get("command_id"),
        parameters=parameters,
    )
    if value != expected:
        raise LaunchGateError("run_identity identity_sha256 does not match its content")
    return expected


def _artifact_evidence(name: str, path: Path) -> dict[str, Any]:
    if not name or "\x00" in name:
        raise LaunchGateError("evidence name must be non-empty and contain no NUL")
    resolved = path.expanduser().resolve(strict=True)
    stat = _regular_file(resolved, label=f"evidence {name}")
    if stat.st_size <= 0:
        raise LaunchGateError(f"evidence {name} is empty: {resolved}")
    return {
        "name": name,
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _validate_evidence(records: object, *, rehash: bool) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise LaunchGateError("ready receipt must contain at least one evidence record")
    observed: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "name",
            "path",
            "size_bytes",
            "sha256",
        }:
            raise LaunchGateError(f"evidence[{index}] has an unexpected field set")
        name = record.get("name")
        path_text = record.get("path")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(name, str) or not name or name in names:
            raise LaunchGateError("evidence names must be unique non-empty strings")
        if not isinstance(path_text, str) or not Path(path_text).is_absolute():
            raise LaunchGateError(f"evidence {name} path must be absolute")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise LaunchGateError(f"evidence {name} size_bytes must be positive")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise LaunchGateError(f"evidence {name} sha256 is malformed")
        names.add(name)
        normalized = dict(record)
        if rehash:
            current = _artifact_evidence(name, Path(path_text))
            if current != normalized:
                raise LaunchAuthorizationError(
                    f"ready evidence changed after authorization: {name}"
                )
        observed.append(normalized)
    return sorted(observed, key=lambda item: item["name"])


def _validate_ready(value: object, *, rehash: bool) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "status",
        "created_at",
        "run_identity",
        "run_identity_sha256",
        "evidence",
    }:
        raise LaunchGateError("ready receipt has an unexpected field set")
    if value.get("schema_version") != READY_SCHEMA or value.get("status") != "ready":
        raise LaunchGateError("ready receipt schema or status is invalid")
    _parse_timestamp(value.get("created_at"), field="ready.created_at")
    identity = _validate_run_identity(value.get("run_identity"))
    if value.get("run_identity_sha256") != identity["identity_sha256"]:
        raise LaunchGateError("ready receipt run identity hash differs")
    evidence = _validate_evidence(value.get("evidence"), rehash=rehash)
    return {**value, "run_identity": identity, "evidence": evidence}


def materialize_ready_receipt(
    gate_directory: Path,
    *,
    run_identity: Mapping[str, object],
    evidence_paths: Mapping[str, Path],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Publish an immutable, idempotent readiness receipt."""

    gate_directory = _ensure_gate_directory(gate_directory)
    identity = _validate_run_identity(dict(run_identity))
    if not evidence_paths:
        raise LaunchGateError("at least one evidence path is required")
    evidence = sorted(
        (_artifact_evidence(name, path) for name, path in evidence_paths.items()),
        key=lambda item: item["name"],
    )
    ready_path = gate_directory / "ready.json"
    with _gate_lock(gate_directory):
        if ready_path.exists() or ready_path.is_symlink():
            existing = _validate_ready(
                _load_json(ready_path, label="ready receipt"), rehash=True
            )
            if existing["run_identity"] != identity or existing["evidence"] != evidence:
                raise LaunchGateError(
                    "existing ready receipt is bound to different run identity or evidence"
                )
            return existing
        receipt = {
            "schema_version": READY_SCHEMA,
            "status": "ready",
            "created_at": _isoformat(now or _utc_now()),
            "run_identity": identity,
            "run_identity_sha256": identity["identity_sha256"],
            "evidence": evidence,
        }
        _write_json_exclusive(ready_path, receipt, mode=0o644)
        return receipt


def _validate_authorization(value: object) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "status",
        "token_id",
        "issued_at",
        "expires_at",
        "authorized_by",
        "ready_receipt_sha256",
        "run_identity_sha256",
        "run_id",
        "phase",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise LaunchAuthorizationError(
            "authorization token has an unexpected field set"
        )
    if (
        value.get("schema_version") != AUTHORIZATION_SCHEMA
        or value.get("status") != "authorized"
    ):
        raise LaunchAuthorizationError(
            "authorization token schema or status is invalid"
        )
    token_id = value.get("token_id")
    if not isinstance(token_id, str) or not _TOKEN_ID_RE.fullmatch(token_id):
        raise LaunchAuthorizationError("authorization token_id is malformed")
    issued = _parse_timestamp(value.get("issued_at"), field="authorization.issued_at")
    expires = _parse_timestamp(
        value.get("expires_at"), field="authorization.expires_at"
    )
    if expires <= issued:
        raise LaunchAuthorizationError("authorization expiry must follow issue time")
    if (expires - issued).total_seconds() > MAX_AUTHORIZATION_TTL_SECONDS:
        raise LaunchAuthorizationError("authorization TTL exceeds the maximum")
    for field in ("authorized_by", "run_id", "phase"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise LaunchAuthorizationError(f"authorization {field} is invalid")
    for field in ("ready_receipt_sha256", "run_identity_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise LaunchAuthorizationError(f"authorization {field} is malformed")
    return dict(value)


def issue_launch_authorization(
    gate_directory: Path,
    *,
    ttl_seconds: float,
    authorized_by: str | None = None,
    now: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Explicitly issue one short-lived authorization for the current ready receipt."""

    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, (int, float))
        or not 0 < ttl_seconds <= MAX_AUTHORIZATION_TTL_SECONDS
    ):
        raise LaunchAuthorizationError(
            f"ttl_seconds must be in (0, {MAX_AUTHORIZATION_TTL_SECONDS}]"
        )
    actor = authorized_by or getpass.getuser()
    if not actor or "\x00" in actor:
        raise LaunchAuthorizationError("authorized_by must be non-empty")
    gate_directory = _ensure_gate_directory(gate_directory)
    ready_path = gate_directory / "ready.json"
    with _gate_lock(gate_directory):
        ready = _validate_ready(
            _load_json(ready_path, label="ready receipt"), rehash=True
        )
        issued_at = now or _utc_now()
        token_id = secrets.token_hex(16)
        authorization = {
            "schema_version": AUTHORIZATION_SCHEMA,
            "status": "authorized",
            "token_id": token_id,
            "issued_at": _isoformat(issued_at),
            "expires_at": _isoformat(issued_at + timedelta(seconds=ttl_seconds)),
            "authorized_by": actor,
            "ready_receipt_sha256": _sha256_file(ready_path),
            "run_identity_sha256": ready["run_identity_sha256"],
            "run_id": ready["run_identity"]["run_id"],
            "phase": ready["run_identity"]["phase"],
        }
        token_path = gate_directory / "authorizations" / f"{token_id}.json"
        _write_json_exclusive(token_path, authorization, mode=0o600)
        return token_path, authorization


def _validate_execution_policy(value: object) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "policy_id",
        "revision",
        "execution_mode",
        "reason",
        "freeze_override",
        "runtime_closure",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise LaunchAuthorizationError("execution policy has an unexpected field set")
    if value.get("schema_version") != EXECUTION_POLICY_SCHEMA:
        raise LaunchAuthorizationError("execution policy schema is invalid")
    for field in ("policy_id", "reason"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise LaunchAuthorizationError(f"execution policy {field} is invalid")
    revision = value.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
        raise LaunchAuthorizationError("execution policy revision must be positive")
    if value.get("execution_mode") not in {"frozen", "open"}:
        raise LaunchAuthorizationError("execution policy mode must be frozen or open")
    override = value.get("freeze_override")
    if not isinstance(override, dict) or set(override) != {
        "required_when_frozen",
        "max_ttl_seconds",
        "reason_required",
    }:
        raise LaunchAuthorizationError("execution policy freeze_override is invalid")
    if override.get("required_when_frozen") is not True:
        raise LaunchAuthorizationError(
            "execution policy must require an override while frozen"
        )
    if override.get("reason_required") is not True:
        raise LaunchAuthorizationError("freeze override reasons must be required")
    maximum = override.get("max_ttl_seconds")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 0 < maximum <= MAX_FREEZE_OVERRIDE_TTL_SECONDS
    ):
        raise LaunchAuthorizationError("freeze override max_ttl_seconds is invalid")
    runtime_closure = value.get("runtime_closure")
    if not isinstance(runtime_closure, dict) or set(runtime_closure) != {
        "launch_enabled",
        "blocker_ids",
    }:
        raise LaunchAuthorizationError("execution policy runtime_closure is invalid")
    launch_enabled = runtime_closure.get("launch_enabled")
    blocker_ids = runtime_closure.get("blocker_ids")
    if not isinstance(launch_enabled, bool):
        raise LaunchAuthorizationError("runtime closure launch_enabled must be boolean")
    if (
        not isinstance(blocker_ids, list)
        or any(
            not isinstance(item, str)
            or not item
            or not re.fullmatch(r"[a-z][a-z0-9_]*", item)
            for item in blocker_ids
        )
        or blocker_ids != sorted(set(blocker_ids))
    ):
        raise LaunchAuthorizationError(
            "runtime closure blocker_ids must be sorted unique identifiers"
        )
    if launch_enabled == bool(blocker_ids):
        raise LaunchAuthorizationError(
            "runtime closure must be enabled exactly when no blockers remain"
        )
    return dict(value)


def _load_execution_policy(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve(strict=True)
    policy = _validate_execution_policy(
        _load_json(resolved, label="experiment execution policy")
    )
    return policy, _sha256_file(resolved)


def _validate_freeze_override(value: object) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "status",
        "override_id",
        "issued_at",
        "expires_at",
        "authorized_by",
        "reason",
        "policy_id",
        "policy_revision",
        "policy_sha256",
        "ready_receipt_sha256",
        "run_identity_sha256",
        "run_id",
        "phase",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise LaunchAuthorizationError("freeze override has an unexpected field set")
    if (
        value.get("schema_version") != FREEZE_OVERRIDE_SCHEMA
        or value.get("status") != "freeze_overridden"
    ):
        raise LaunchAuthorizationError("freeze override schema or status is invalid")
    override_id = value.get("override_id")
    if not isinstance(override_id, str) or not _TOKEN_ID_RE.fullmatch(override_id):
        raise LaunchAuthorizationError("freeze override id is malformed")
    issued = _parse_timestamp(value.get("issued_at"), field="freeze_override.issued_at")
    expires = _parse_timestamp(
        value.get("expires_at"), field="freeze_override.expires_at"
    )
    if expires <= issued:
        raise LaunchAuthorizationError("freeze override expiry must follow issue time")
    if (expires - issued).total_seconds() > MAX_FREEZE_OVERRIDE_TTL_SECONDS:
        raise LaunchAuthorizationError("freeze override TTL exceeds the hard maximum")
    for field in ("authorized_by", "reason", "policy_id", "run_id", "phase"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise LaunchAuthorizationError(f"freeze override {field} is invalid")
    revision = value.get("policy_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
        raise LaunchAuthorizationError("freeze override policy revision is invalid")
    for field in (
        "policy_sha256",
        "ready_receipt_sha256",
        "run_identity_sha256",
    ):
        digest = value.get(field)
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise LaunchAuthorizationError(f"freeze override {field} is malformed")
    return dict(value)


def issue_freeze_override(
    gate_directory: Path,
    execution_policy_path: Path,
    *,
    reason: str,
    ttl_seconds: float,
    authorized_by: str | None = None,
    now: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Explicitly record one short-lived exception to a frozen repository."""

    if not isinstance(reason, str) or not reason.strip() or "\x00" in reason:
        raise LaunchAuthorizationError("freeze override reason must be non-empty")
    actor = authorized_by or getpass.getuser()
    if not actor or "\x00" in actor:
        raise LaunchAuthorizationError("authorized_by must be non-empty")
    policy, policy_sha256 = _load_execution_policy(execution_policy_path)
    if policy["execution_mode"] != "frozen":
        raise LaunchAuthorizationError(
            "a freeze override can only be issued while execution is frozen"
        )
    maximum = min(
        policy["freeze_override"]["max_ttl_seconds"],
        MAX_FREEZE_OVERRIDE_TTL_SECONDS,
    )
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, (int, float))
        or not 0 < ttl_seconds <= maximum
    ):
        raise LaunchAuthorizationError(
            f"freeze override ttl_seconds must be in (0, {maximum}]"
        )
    gate_directory = _ensure_gate_directory(gate_directory)
    ready_path = gate_directory / "ready.json"
    with _gate_lock(gate_directory):
        ready = _validate_ready(
            _load_json(ready_path, label="ready receipt"), rehash=True
        )
        issued_at = now or _utc_now()
        override_id = secrets.token_hex(16)
        override = {
            "schema_version": FREEZE_OVERRIDE_SCHEMA,
            "status": "freeze_overridden",
            "override_id": override_id,
            "issued_at": _isoformat(issued_at),
            "expires_at": _isoformat(issued_at + timedelta(seconds=ttl_seconds)),
            "authorized_by": actor,
            "reason": reason.strip(),
            "policy_id": policy["policy_id"],
            "policy_revision": policy["revision"],
            "policy_sha256": policy_sha256,
            "ready_receipt_sha256": _sha256_file(ready_path),
            "run_identity_sha256": ready["run_identity_sha256"],
            "run_id": ready["run_identity"]["run_id"],
            "phase": ready["run_identity"]["phase"],
        }
        override_path = gate_directory / "freeze-overrides" / f"{override_id}.json"
        _write_json_exclusive(override_path, override, mode=0o600)
        return override_path, override


def consume_launch_authorization(
    gate_directory: Path,
    token_path: Path,
    execution_policy_path: Path,
    *,
    expected_run_id: str,
    expected_phase: str,
    freeze_override_path: Path | None = None,
    consumed_by: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Atomically consume one run-bound token and record durable proof."""

    gate_directory = _ensure_gate_directory(gate_directory)
    ready_path = gate_directory / "ready.json"
    policy, policy_sha256 = _load_execution_policy(execution_policy_path)
    actor = consumed_by or getpass.getuser()
    if not actor or "\x00" in actor:
        raise LaunchAuthorizationError("consumed_by must be non-empty")
    with _gate_lock(gate_directory):
        ready = _validate_ready(
            _load_json(ready_path, label="ready receipt"), rehash=True
        )
        authorization = _validate_authorization(
            _load_json(token_path.expanduser().absolute(), label="authorization token")
        )
        token_id = authorization["token_id"]
        consumption_path = gate_directory / "consumptions" / f"{token_id}.json"
        if consumption_path.exists() or consumption_path.is_symlink():
            raise LaunchAuthorizationError(
                f"authorization token {token_id} was already consumed"
            )
        observed_now = now or _utc_now()
        issued_at = _parse_timestamp(
            authorization["issued_at"], field="authorization.issued_at"
        )
        expires_at = _parse_timestamp(
            authorization["expires_at"], field="authorization.expires_at"
        )
        if observed_now < issued_at - timedelta(seconds=1):
            raise LaunchAuthorizationError("authorization token is not active yet")
        if observed_now >= expires_at:
            raise LaunchAuthorizationError("authorization token expired")
        identity = ready["run_identity"]
        if identity["run_id"] != expected_run_id or identity["phase"] != expected_phase:
            raise LaunchAuthorizationError(
                "ready receipt differs from the controller's expected run identity"
            )
        if (
            authorization["run_id"] != expected_run_id
            or authorization["phase"] != expected_phase
            or authorization["run_identity_sha256"] != ready["run_identity_sha256"]
            or authorization["ready_receipt_sha256"] != _sha256_file(ready_path)
        ):
            raise LaunchAuthorizationError(
                "authorization token is not bound to the current ready receipt"
            )
        freeze_override: dict[str, Any] | None = None
        freeze_override_consumption_path: Path | None = None
        if policy["execution_mode"] == "frozen":
            if freeze_override_path is None:
                raise LaunchAuthorizationError(
                    "repository execution is frozen; an explicit freeze override is required"
                )
            freeze_override = _validate_freeze_override(
                _load_json(
                    freeze_override_path.expanduser().absolute(),
                    label="freeze override",
                )
            )
            override_id = freeze_override["override_id"]
            freeze_override_consumption_path = (
                gate_directory / "freeze-override-consumptions" / f"{override_id}.json"
            )
            if (
                freeze_override_consumption_path.exists()
                or freeze_override_consumption_path.is_symlink()
            ):
                raise LaunchAuthorizationError(
                    f"freeze override {override_id} was already consumed"
                )
            override_issued_at = _parse_timestamp(
                freeze_override["issued_at"], field="freeze_override.issued_at"
            )
            override_expires_at = _parse_timestamp(
                freeze_override["expires_at"], field="freeze_override.expires_at"
            )
            if observed_now < override_issued_at - timedelta(seconds=1):
                raise LaunchAuthorizationError("freeze override is not active yet")
            if observed_now >= override_expires_at:
                raise LaunchAuthorizationError("freeze override expired")
            if (
                freeze_override["policy_id"] != policy["policy_id"]
                or freeze_override["policy_revision"] != policy["revision"]
                or freeze_override["policy_sha256"] != policy_sha256
                or freeze_override["ready_receipt_sha256"]
                != authorization["ready_receipt_sha256"]
                or freeze_override["run_identity_sha256"]
                != authorization["run_identity_sha256"]
                or freeze_override["run_id"] != expected_run_id
                or freeze_override["phase"] != expected_phase
            ):
                raise LaunchAuthorizationError(
                    "freeze override is not bound to the current policy and run"
                )
        consumption = {
            "schema_version": CONSUMPTION_SCHEMA,
            "status": "consumed",
            "token_id": token_id,
            "consumed_at": _isoformat(observed_now),
            "consumed_by": actor,
            "consumer_pid": os.getpid(),
            "consumer_host": socket.gethostname(),
            "authorization_sha256": _sha256_file(token_path.expanduser().absolute()),
            "ready_receipt_sha256": authorization["ready_receipt_sha256"],
            "run_identity_sha256": authorization["run_identity_sha256"],
            "run_id": expected_run_id,
            "phase": expected_phase,
            "execution_policy_id": policy["policy_id"],
            "execution_policy_revision": policy["revision"],
            "execution_policy_sha256": policy_sha256,
            "execution_mode": policy["execution_mode"],
            "freeze_override_id": (
                None if freeze_override is None else freeze_override["override_id"]
            ),
            "freeze_override_sha256": (
                None
                if freeze_override_path is None
                else _sha256_file(freeze_override_path.expanduser().absolute())
            ),
        }
        if freeze_override is not None and freeze_override_consumption_path is not None:
            override_consumption = {
                "schema_version": FREEZE_OVERRIDE_CONSUMPTION_SCHEMA,
                "status": "consumed",
                "override_id": freeze_override["override_id"],
                "consumed_at": _isoformat(observed_now),
                "consumed_by": actor,
                "authorization_token_id": token_id,
                "run_identity_sha256": authorization["run_identity_sha256"],
                "policy_sha256": policy_sha256,
            }
            try:
                _write_json_exclusive(
                    freeze_override_consumption_path,
                    override_consumption,
                    mode=0o644,
                )
            except FileExistsError as error:
                raise LaunchAuthorizationError(
                    f"freeze override {freeze_override['override_id']} was concurrently consumed"
                ) from error
        try:
            _write_json_exclusive(consumption_path, consumption, mode=0o644)
        except FileExistsError as error:
            raise LaunchAuthorizationError(
                f"authorization token {token_id} was concurrently consumed"
            ) from error
        return {**consumption, "consumption_path": str(consumption_path)}


def _process_start_ticks(pid: int) -> int:
    try:
        stat_text = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError) as error:
        raise LaunchLivenessError(f"process {pid} is not live") from error
    except (OSError, UnicodeDecodeError) as error:
        raise LaunchLivenessError(f"cannot inspect process {pid}") from error
    closing = stat_text.rfind(")")
    if closing < 0:
        raise LaunchLivenessError(f"process {pid} stat is malformed")
    fields = stat_text[closing + 2 :].split()
    try:
        return int(fields[19])
    except (IndexError, ValueError) as error:
        raise LaunchLivenessError(f"process {pid} stat has no start time") from error


def write_process_liveness_receipt(
    path: Path,
    *,
    run_identity: Mapping[str, object],
    pid: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Atomically publish the exact OS identity of a live producer process."""

    identity = _validate_run_identity(dict(run_identity))
    observed_pid = os.getpid() if pid is None else pid
    if (
        isinstance(observed_pid, bool)
        or not isinstance(observed_pid, int)
        or observed_pid <= 0
    ):
        raise LaunchLivenessError("pid must be a positive integer")
    receipt = {
        "schema_version": LIVENESS_SCHEMA,
        "status": "live",
        "recorded_at": _isoformat(now or _utc_now()),
        "pid": observed_pid,
        "process_start_ticks": _process_start_ticks(observed_pid),
        "run_identity_sha256": identity["identity_sha256"],
        "run_id": identity["run_id"],
        "phase": identity["phase"],
    }
    _write_json_atomic(path.expanduser().absolute(), receipt, mode=0o644)
    return receipt


def assert_process_liveness(
    path: Path,
    *,
    expected_run_id: str | None = None,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    value = _load_json(path.expanduser().absolute(), label="process liveness receipt")
    expected_fields = {
        "schema_version",
        "status",
        "recorded_at",
        "pid",
        "process_start_ticks",
        "run_identity_sha256",
        "run_id",
        "phase",
    }
    if set(value) != expected_fields:
        raise LaunchLivenessError(
            "process liveness receipt has an unexpected field set"
        )
    if value.get("schema_version") != LIVENESS_SCHEMA or value.get("status") != "live":
        raise LaunchLivenessError(
            "process liveness receipt schema or status is invalid"
        )
    _parse_timestamp(value.get("recorded_at"), field="liveness.recorded_at")
    pid = value.get("pid")
    ticks = value.get("process_start_ticks")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise LaunchLivenessError("process liveness pid is invalid")
    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks <= 0:
        raise LaunchLivenessError("process liveness start time is invalid")
    if expected_run_id is not None and value.get("run_id") != expected_run_id:
        raise LaunchLivenessError("process liveness run_id differs")
    if expected_phase is not None and value.get("phase") != expected_phase:
        raise LaunchLivenessError("process liveness phase differs")
    if _process_start_ticks(pid) != ticks:
        raise LaunchLivenessError(
            f"process {pid} was replaced after liveness publication"
        )
    return value


def wait_for_artifact(
    path: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float = 5.0,
    liveness_receipt_path: Path | None = None,
    expected_run_id: str | None = None,
    expected_phase: str | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Wait for a non-empty artifact with a deadline and optional producer proof."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < timeout_seconds <= MAX_WAIT_TIMEOUT_SECONDS
    ):
        raise LaunchTimeoutError(
            f"timeout_seconds must be in (0, {MAX_WAIT_TIMEOUT_SECONDS}]"
        )
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, (int, float))
        or poll_seconds <= 0
        or poll_seconds > timeout_seconds
    ):
        raise LaunchTimeoutError(
            "poll_seconds must be positive and no larger than timeout"
        )
    artifact_path = path.expanduser().absolute()
    deadline = monotonic() + timeout_seconds
    while True:
        if artifact_path.is_symlink():
            raise LaunchGateError(
                f"waited artifact must not be a symlink: {artifact_path}"
            )
        try:
            if artifact_path.is_file() and artifact_path.stat().st_size > 0:
                return artifact_path
        except FileNotFoundError:
            pass
        if liveness_receipt_path is not None:
            assert_process_liveness(
                liveness_receipt_path,
                expected_run_id=expected_run_id,
                expected_phase=expected_phase,
            )
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise LaunchTimeoutError(
                f"artifact did not become ready within {timeout_seconds:g}s: {artifact_path}"
            )
        sleep(min(poll_seconds, remaining))


def gate_status(gate_directory: Path) -> dict[str, Any]:
    """Return a read-only summary without exposing authorization token content."""

    gate_directory = gate_directory.expanduser().absolute()
    ready_path = gate_directory / "ready.json"
    ready: dict[str, Any] | None = None
    error: str | None = None
    if ready_path.exists() or ready_path.is_symlink():
        try:
            ready = _validate_ready(
                _load_json(ready_path, label="ready receipt"), rehash=True
            )
        except LaunchGateError as caught:
            error = str(caught)
    authorizations = sorted((gate_directory / "authorizations").glob("*.json"))
    consumptions = sorted((gate_directory / "consumptions").glob("*.json"))
    return {
        "gate_directory": str(gate_directory),
        "ready": ready,
        "ready_error": error,
        "authorization_count": len(authorizations),
        "consumption_count": len(consumptions),
    }
