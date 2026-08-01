#!/usr/bin/env python3
"""Run the fixed Stage1/validation/Crop overnight chain fail-closed.

The controller is deliberately ignorant of training implementation details.  A
JSON plan declares one argv-only command and one or more durable artifact
predicates for each required stage.  A stage is accepted only after its command
returns zero *and* every predicate passes.  The next command is never started
until the complete accepted prefix has been revalidated.

The required order first proves the Crop runtime, then trains and evaluates the
new Stage1 structure, and only then promotes the Crop pilot::

    crop_rl_smoke_1step
    crop_rl_auto_resume_proof
    stage1_smoke
    stage1_resume_2000
    int_diag
    acc_val
    crop_rl_80step

State is bound to the exact config bytes and atomically replaced.  Re-running
the same plan skips a still-valid accepted prefix.  A stale ``running`` record
is recovered only when its recorded process is no longer the same live process;
an actually live process causes a fail-closed refusal instead of a duplicate
launch.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, NoReturn, Sequence


SCHEMA_VERSION = "tgvf-overnight-pipeline-v1"
STATE_SCHEMA_VERSION = "tgvf-overnight-pipeline-state-v1"
REQUIRED_STAGE_IDS = (
    "crop_rl_smoke_1step",
    "crop_rl_auto_resume_proof",
    "stage1_smoke",
    "stage1_resume_2000",
    "int_diag",
    "acc_val",
    "crop_rl_80step",
)
PREDICATE_TYPES = frozenset({"exists", "json_field", "jsonl_last_step"})
MAX_EXPLICIT_RETRIES = 5


class PipelineError(RuntimeError):
    """Base class for a controlled pipeline refusal."""


class ConfigError(PipelineError):
    """Raised when the JSON plan is incomplete or ambiguous."""


class PipelineBlockedError(PipelineError):
    """Raised when persisted state or a live process makes launch unsafe."""


class PredicateFailed(PipelineError):
    """Raised when a stage's durable acceptance predicate does not hold."""


class StageFailed(PipelineError):
    """Raised after a stage exhausts its explicitly configured attempts."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        os.fsync(handle.fileno())


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], *, location: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{location} contains unknown keys: {', '.join(unknown)}")


def _require_mapping(value: object, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a JSON object")
    return value


def _require_nonempty_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be a non-empty string")
    if "\x00" in value:
        raise ConfigError(f"{location} may not contain NUL")
    return value


def _resolve_path(value: object, *, base: Path, location: str) -> Path:
    raw = _require_nonempty_string(value, location=location)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _validate_field_path(value: object, *, location: str) -> str:
    field = _require_nonempty_string(value, location=location)
    if any(not component for component in field.split(".")):
        raise ConfigError(f"{location} must be a dotted field path without empty parts")
    return field


def _validate_predicate(
    raw: object, *, config_dir: Path, stage_location: str, index: int
) -> dict[str, Any]:
    location = f"{stage_location}.acceptance[{index}]"
    value = _require_mapping(raw, location=location)
    kind = value.get("type")
    if kind not in PREDICATE_TYPES:
        raise ConfigError(
            f"{location}.type must be one of {', '.join(sorted(PREDICATE_TYPES))}"
        )
    common = {"type", "path"}
    if kind == "exists":
        _reject_unknown_keys(value, common | {"kind", "nonempty"}, location=location)
        artifact_kind = value.get("kind", "file")
        if artifact_kind not in {"file", "directory", "any"}:
            raise ConfigError(f"{location}.kind must be file, directory, or any")
        nonempty = value.get("nonempty", True)
        if not isinstance(nonempty, bool):
            raise ConfigError(f"{location}.nonempty must be boolean")
        return {
            "type": kind,
            "path": str(
                _resolve_path(value.get("path"), base=config_dir, location=f"{location}.path")
            ),
            "kind": artifact_kind,
            "nonempty": nonempty,
        }
    if kind == "json_field":
        _reject_unknown_keys(value, common | {"field", "equals"}, location=location)
        if "equals" not in value:
            raise ConfigError(f"{location}.equals is required")
        return {
            "type": kind,
            "path": str(
                _resolve_path(value.get("path"), base=config_dir, location=f"{location}.path")
            ),
            "field": _validate_field_path(
                value.get("field"), location=f"{location}.field"
            ),
            "equals": value["equals"],
        }
    _reject_unknown_keys(value, common | {"field", "equals"}, location=location)
    if "equals" not in value:
        raise ConfigError(f"{location}.equals is required")
    expected = value["equals"]
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        raise ConfigError(f"{location}.equals must be a non-negative integer step")
    return {
        "type": kind,
        "path": str(
            _resolve_path(value.get("path"), base=config_dir, location=f"{location}.path")
        ),
        "field": _validate_field_path(value.get("field"), location=f"{location}.field"),
        "equals": expected,
    }


def _validate_command(
    raw: object, *, config_dir: Path, stage_location: str
) -> dict[str, Any]:
    location = f"{stage_location}.command"
    value = _require_mapping(raw, location=location)
    _reject_unknown_keys(
        value,
        {"argv", "cwd", "env", "timeout_seconds", "terminate_grace_seconds"},
        location=location,
    )
    argv = value.get("argv")
    if not isinstance(argv, list) or not argv:
        raise ConfigError(f"{location}.argv must be a non-empty JSON array")
    normalized_argv = [
        _require_nonempty_string(item, location=f"{location}.argv[{index}]")
        for index, item in enumerate(argv)
    ]
    cwd = _resolve_path(value.get("cwd", "."), base=config_dir, location=f"{location}.cwd")
    if not cwd.is_dir():
        raise ConfigError(f"{location}.cwd is not an existing directory: {cwd}")
    environment = value.get("env", {})
    if not isinstance(environment, dict):
        raise ConfigError(f"{location}.env must be a JSON object")
    normalized_environment: dict[str, str] = {}
    for key, item in environment.items():
        name = _require_nonempty_string(key, location=f"{location}.env key")
        normalized_environment[name] = _require_nonempty_string(
            item, location=f"{location}.env[{name!r}]"
        )
        if "=" in name:
            raise ConfigError(f"{location}.env keys may not contain '='")
    timeout = value.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigError(f"{location}.timeout_seconds must be positive")
    grace = value.get("terminate_grace_seconds", 30)
    if isinstance(grace, bool) or not isinstance(grace, (int, float)) or grace < 0:
        raise ConfigError(f"{location}.terminate_grace_seconds must be non-negative")
    return {
        "argv": normalized_argv,
        "cwd": str(cwd),
        "env": normalized_environment,
        "timeout_seconds": float(timeout),
        "terminate_grace_seconds": float(grace),
    }


def _validate_retry(raw: object, *, stage_location: str) -> dict[str, Any]:
    location = f"{stage_location}.retry"
    if raw is None:
        return {"max_retries": 0, "delay_seconds": 0.0, "explicit": False}
    value = _require_mapping(raw, location=location)
    _reject_unknown_keys(value, {"max_retries", "delay_seconds"}, location=location)
    retries = value.get("max_retries")
    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or not 0 <= retries <= MAX_EXPLICIT_RETRIES
    ):
        raise ConfigError(
            f"{location}.max_retries must be an integer from 0 to "
            f"{MAX_EXPLICIT_RETRIES}"
        )
    delay = value.get("delay_seconds", 0)
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not 0 <= delay <= 3600:
        raise ConfigError(f"{location}.delay_seconds must be between 0 and 3600")
    return {
        "max_retries": retries,
        "delay_seconds": float(delay),
        "explicit": True,
    }


def load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load and strictly normalize one pipeline config."""

    config_path = path.resolve(strict=True)
    if not config_path.is_file():
        raise ConfigError(f"config is not a regular file: {config_path}")
    payload = config_path.read_bytes()
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"config is not valid UTF-8 JSON: {error}") from error
    value = _require_mapping(decoded, location="config")
    _reject_unknown_keys(
        value,
        {"schema_version", "pipeline_id", "runtime_root", "stages"},
        location="config",
    )
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"config.schema_version must equal {SCHEMA_VERSION!r}")
    pipeline_id = _require_nonempty_string(value.get("pipeline_id"), location="config.pipeline_id")
    config_dir = config_path.parent
    runtime_root = _resolve_path(
        value.get("runtime_root"), base=config_dir, location="config.runtime_root"
    )
    raw_stages = value.get("stages")
    if not isinstance(raw_stages, list):
        raise ConfigError("config.stages must be a JSON array")
    observed_ids: list[str] = []
    stages: list[dict[str, Any]] = []
    for index, raw_stage in enumerate(raw_stages):
        location = f"config.stages[{index}]"
        stage = _require_mapping(raw_stage, location=location)
        _reject_unknown_keys(
            stage, {"id", "command", "acceptance", "retry"}, location=location
        )
        stage_id = _require_nonempty_string(stage.get("id"), location=f"{location}.id")
        observed_ids.append(stage_id)
        acceptance = stage.get("acceptance")
        if not isinstance(acceptance, list) or not acceptance:
            raise ConfigError(f"{location}.acceptance must be a non-empty JSON array")
        stages.append(
            {
                "id": stage_id,
                "command": _validate_command(
                    stage.get("command"), config_dir=config_dir, stage_location=location
                ),
                "acceptance": [
                    _validate_predicate(
                        item,
                        config_dir=config_dir,
                        stage_location=location,
                        index=predicate_index,
                    )
                    for predicate_index, item in enumerate(acceptance)
                ],
                "retry": _validate_retry(stage.get("retry"), stage_location=location),
            }
        )
    if tuple(observed_ids) != REQUIRED_STAGE_IDS:
        raise ConfigError(
            "config.stages must contain the exact fail-closed order: "
            + ", ".join(REQUIRED_STAGE_IDS)
        )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "pipeline_id": pipeline_id,
            "runtime_root": str(runtime_root),
            "config_path": str(config_path),
            "config_sha256": _sha256_bytes(payload),
            "stages": stages,
        },
        payload,
    )


def _field(value: object, path: str) -> object:
    current = value
    for component in path.split("."):
        if isinstance(current, dict):
            if component not in current:
                raise PredicateFailed(f"JSON field {path!r} is absent at {component!r}")
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            index = int(component)
            if index >= len(current):
                raise PredicateFailed(f"JSON field {path!r} index {index} is out of range")
            current = current[index]
        else:
            raise PredicateFailed(f"JSON field {path!r} cannot traverse {component!r}")
    return current


def _json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""

    return _canonical_bytes(left) == _canonical_bytes(right)


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise PredicateFailed(f"{label} is not a non-symlink regular file: {path}")
    return path.stat()


def _last_nonempty_line(path: Path) -> bytes:
    _regular_file(path, label="JSONL artifact")
    size = path.stat().st_size
    if size == 0:
        raise PredicateFailed(f"JSONL artifact is empty: {path}")
    with path.open("rb") as handle:
        position = size
        suffix = b""
        while position > 0:
            count = min(64 * 1024, position)
            position -= count
            handle.seek(position)
            suffix = handle.read(count) + suffix
            lines = suffix.splitlines()
            if position == 0 or len(lines) > 1:
                for line in reversed(lines):
                    if line.strip():
                        return line
        raise PredicateFailed(f"JSONL artifact has no non-empty records: {path}")


def evaluate_predicate(predicate: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one normalized artifact predicate and return its observation."""

    kind = str(predicate["type"])
    path = Path(str(predicate["path"]))
    if kind == "exists":
        if path.is_symlink() or not path.exists():
            raise PredicateFailed(f"required artifact does not exist: {path}")
        if not path.is_file() and not path.is_dir():
            raise PredicateFailed(f"required artifact is not a regular file or directory: {path}")
        expected_kind = predicate["kind"]
        if expected_kind == "file" and not path.is_file():
            raise PredicateFailed(f"required artifact is not a file: {path}")
        if expected_kind == "directory" and not path.is_dir():
            raise PredicateFailed(f"required artifact is not a directory: {path}")
        stat = path.stat()
        if predicate["nonempty"]:
            if path.is_file() and stat.st_size == 0:
                raise PredicateFailed(f"required artifact file is empty: {path}")
            if path.is_dir():
                try:
                    next(path.iterdir())
                except StopIteration as error:
                    raise PredicateFailed(f"required artifact directory is empty: {path}") from error
        observation: dict[str, Any] = {
            "type": kind,
            "path": str(path),
            "kind": "file" if path.is_file() else "directory",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if path.is_file():
            observation["sha256"] = _sha256_file(path)
        return observation
    stat = _regular_file(path, label=f"{kind} artifact")
    if kind == "json_field":
        try:
            document = json.loads(path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PredicateFailed(f"JSON artifact is invalid: {path}: {error}") from error
        actual = _field(document, str(predicate["field"]))
        if not _json_equal(actual, predicate["equals"]):
            raise PredicateFailed(
                f"JSON field {predicate['field']!r} in {path} is {actual!r}, "
                f"expected {predicate['equals']!r}"
            )
        return {
            "type": kind,
            "path": str(path),
            "field": predicate["field"],
            "actual": actual,
            "size": stat.st_size,
            "sha256": _sha256_file(path),
        }
    if kind != "jsonl_last_step":
        raise ConfigError(f"unknown normalized predicate type: {kind}")
    line = _last_nonempty_line(path)
    try:
        record = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredicateFailed(f"last JSONL record is invalid in {path}: {error}") from error
    actual = _field(record, str(predicate["field"]))
    if (
        isinstance(actual, bool)
        or not isinstance(actual, int)
        or actual != predicate["equals"]
    ):
        raise PredicateFailed(
            f"last JSONL step {predicate['field']!r} in {path} is {actual!r}, "
            f"expected {predicate['equals']!r}"
        )
    return {
        "type": kind,
        "path": str(path),
        "field": predicate["field"],
        "actual": actual,
        "size": stat.st_size,
        "sha256": _sha256_file(path),
        "last_record_sha256": _sha256_bytes(line),
    }


def evaluate_acceptance(predicates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_predicate(predicate) for predicate in predicates]


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return "unavailable"


def _proc_identity(pid: int) -> dict[str, Any] | None:
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text(encoding="ascii")
        argv_bytes = (proc / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    closing = stat_text.rfind(")")
    fields = stat_text[closing + 2 :].split()
    if closing < 0 or len(fields) < 20:
        raise PipelineBlockedError(f"cannot parse process identity for PID {pid}")
    return {
        "pid": pid,
        "pgrp": int(fields[2]),
        "session": int(fields[3]),
        "starttime_ticks": int(fields[19]),
        "argv_sha256": _sha256_bytes(argv_bytes),
        "boot_id": _boot_id(),
    }


def _same_live_process(record: Mapping[str, Any]) -> bool:
    pid = record.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    observed = _proc_identity(pid)
    if observed is None:
        return False
    required = ("pid", "pgrp", "session", "starttime_ticks", "argv_sha256", "boot_id")
    return all(observed.get(key) == record.get(key) for key in required)


def _new_state(config: Mapping[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "pipeline_id": config["pipeline_id"],
        "config_path": config["config_path"],
        "config_sha256": config["config_sha256"],
        "created_at": now,
        "updated_at": now,
        "status": "pending",
        "current_stage": None,
        "stages": {
            stage["id"]: {"status": "pending", "attempts": []}
            for stage in config["stages"]
        },
    }


def _load_state(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return _new_state(config)
    if path.is_symlink() or not path.is_file():
        raise PipelineBlockedError(f"state path is not a regular file: {path}")
    try:
        state = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PipelineBlockedError(f"state file is invalid: {error}") from error
    if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise PipelineBlockedError("state schema is absent or incompatible")
    for key in ("pipeline_id", "config_path", "config_sha256"):
        if state.get(key) != config.get(key):
            raise PipelineBlockedError(f"state {key} does not match the exact config")
    stages = state.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(REQUIRED_STAGE_IDS):
        raise PipelineBlockedError("state stage set/order is incompatible")
    for stage_id in REQUIRED_STAGE_IDS:
        stage_state = stages.get(stage_id)
        if not isinstance(stage_state, dict) or stage_state.get("status") not in {
            "pending",
            "running",
            "accepted",
            "failed",
            "interrupted",
        }:
            raise PipelineBlockedError(f"state for {stage_id} is malformed")
        if not isinstance(stage_state.get("attempts"), list):
            raise PipelineBlockedError(f"attempt ledger for {stage_id} is malformed")
    return state


@contextmanager
def _singleton_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PipelineBlockedError("another controller owns the pipeline lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_log_record(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(_canonical_bytes(value) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())


def _terminate_group(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    pgid = process.pid

    def group_alive() -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        return True

    if not group_alive():
        process.wait()
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace_seconds
    while group_alive() and time.monotonic() < deadline:
        process.poll()
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    if not group_alive():
        process.wait()
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class PipelineRunner:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.runtime_root = Path(str(config["runtime_root"]))
        self.state_path = self.runtime_root / "state.json"
        self.events_path = self.runtime_root / "events.jsonl"
        self.lock_path = self.runtime_root / "controller.lock"
        self.logs_root = self.runtime_root / "logs"
        self.state: dict[str, Any] = {}
        self._active_process: subprocess.Popen[bytes] | None = None
        self._active_grace_seconds = 0.0
        self._received_signal: int | None = None
        self._previous_handlers: dict[int, Any] = {}

    def _save(self) -> None:
        self.state["updated_at"] = _utc_now()
        _atomic_json(self.state_path, self.state)

    def _event(self, event: str, **fields: object) -> None:
        _append_jsonl(
            self.events_path,
            {"schema_version": STATE_SCHEMA_VERSION, "at": _utc_now(), "event": event, **fields},
        )

    def _signal_handler(self, signum: int, _frame: object) -> None:
        self._received_signal = signum
        process = self._active_process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._signal_handler)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def _recover_stale_running_record(self) -> None:
        running = [
            (stage_id, stage_state)
            for stage_id, stage_state in self.state["stages"].items()
            if stage_state["status"] == "running"
        ]
        if len(running) > 1:
            raise PipelineBlockedError("state contains more than one running stage")
        if not running:
            return
        stage_id, stage_state = running[0]
        attempts = stage_state["attempts"]
        if not attempts or not isinstance(attempts[-1], dict):
            raise PipelineBlockedError(f"running stage {stage_id} lacks an attempt identity")
        attempt = attempts[-1]
        process_identity = attempt.get("process_identity")
        if isinstance(process_identity, dict) and _same_live_process(process_identity):
            raise PipelineBlockedError(
                f"stage {stage_id} still has its exact process group alive; refusing duplicate launch"
            )
        attempt["ended_at"] = _utc_now()
        attempt["outcome"] = "controller_lost_process"
        attempt["exit_code"] = None
        stage_state["status"] = "interrupted"
        self.state["status"] = "interrupted"
        self.state["current_stage"] = None
        self._save()
        self._event("stale_running_stage_recovered", stage_id=stage_id)

    def _revalidate_accepted_prefix(self) -> int:
        first_unaccepted = len(REQUIRED_STAGE_IDS)
        seen_unaccepted = False
        for index, stage in enumerate(self.config["stages"]):
            stage_id = stage["id"]
            stage_state = self.state["stages"][stage_id]
            if stage_state["status"] == "accepted":
                if seen_unaccepted:
                    raise PipelineBlockedError("state contains a non-prefix accepted stage")
                try:
                    observations = evaluate_acceptance(stage["acceptance"])
                except PredicateFailed as error:
                    raise PipelineBlockedError(
                        f"accepted stage {stage_id} no longer satisfies acceptance: {error}"
                    ) from error
                recorded = stage_state.get("accepted_artifacts")
                if not isinstance(recorded, list) or not _json_equal(recorded, observations):
                    raise PipelineBlockedError(
                        f"accepted stage {stage_id} artifact identity changed since acceptance"
                    )
                stage_state["last_revalidated_at"] = _utc_now()
                self._event("stage_revalidated", stage_id=stage_id)
            else:
                if not seen_unaccepted:
                    first_unaccepted = index
                seen_unaccepted = True
        self._save()
        return first_unaccepted

    def _wait_retry_delay(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._received_signal is not None:
                raise KeyboardInterrupt
            time.sleep(min(0.2, deadline - time.monotonic()))

    def _run_attempt(self, stage: Mapping[str, Any]) -> tuple[bool, str | None]:
        stage_id = str(stage["id"])
        stage_state = self.state["stages"][stage_id]
        ordinal = len(stage_state["attempts"]) + 1
        command = stage["command"]
        log_path = self.logs_root / f"{REQUIRED_STAGE_IDS.index(stage_id) + 1:02d}-{stage_id}.attempt-{ordinal}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = _utc_now()
        attempt: dict[str, Any] = {
            "attempt": ordinal,
            "started_at": started_at,
            "ended_at": None,
            "exit_code": None,
            "outcome": "running",
            "timed_out": False,
            "interrupted_by_signal": None,
            "log_path": str(log_path),
            "command_sha256": _sha256_bytes(_canonical_bytes(command)),
            "process_identity": None,
            "acceptance": None,
            "acceptance_error": None,
        }
        stage_state["status"] = "running"
        stage_state["attempts"].append(attempt)
        self.state["status"] = "running"
        self.state["current_stage"] = stage_id
        self._save()
        self._event("stage_attempt_start", stage_id=stage_id, attempt=ordinal, log_path=str(log_path))
        environment = os.environ.copy()
        environment.update(command["env"])
        process: subprocess.Popen[bytes] | None = None
        return_code: int | None = None
        timed_out = False
        interrupted = None
        with log_path.open("ab", buffering=0) as log_handle:
            _write_log_record(
                log_handle,
                {
                    "controller_event": "command_start",
                    "stage_id": stage_id,
                    "attempt": ordinal,
                    "at": started_at,
                    "argv": command["argv"],
                    "cwd": command["cwd"],
                },
            )
            try:
                process = subprocess.Popen(
                    command["argv"],
                    cwd=command["cwd"],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as error:
                attempt["outcome"] = "launch_error"
                attempt["acceptance_error"] = f"{type(error).__name__}: {error}"
                _write_log_record(
                    log_handle,
                    {
                        "controller_event": "command_end",
                        "stage_id": stage_id,
                        "attempt": ordinal,
                        "at": _utc_now(),
                        "exit_code": None,
                        "outcome": "launch_error",
                        "error": str(error),
                    },
                )
            else:
                self._active_process = process
                self._active_grace_seconds = command["terminate_grace_seconds"]
                identity = _proc_identity(process.pid)
                if identity is None:
                    identity = {
                        "pid": process.pid,
                        "pgrp": process.pid,
                        "session": process.pid,
                        "starttime_ticks": None,
                        "argv_sha256": None,
                        "boot_id": _boot_id(),
                    }
                attempt["process_identity"] = identity
                self._save()
                deadline = time.monotonic() + command["timeout_seconds"]
                while process.poll() is None:
                    if self._received_signal is not None:
                        interrupted = self._received_signal
                        _terminate_group(process, command["terminate_grace_seconds"])
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        _terminate_group(process, command["terminate_grace_seconds"])
                        break
                    time.sleep(0.1)
                return_code = process.wait()
                self._active_process = None
                self._active_grace_seconds = 0.0
                _write_log_record(
                    log_handle,
                    {
                        "controller_event": "command_end",
                        "stage_id": stage_id,
                        "attempt": ordinal,
                        "at": _utc_now(),
                        "exit_code": return_code,
                        "timed_out": timed_out,
                        "interrupted_by_signal": interrupted,
                    },
                )
        attempt["ended_at"] = _utc_now()
        attempt["exit_code"] = return_code
        attempt["timed_out"] = timed_out
        attempt["interrupted_by_signal"] = interrupted
        error_message: str | None = None
        if process is None:
            attempt["outcome"] = "launch_error"
            error_message = attempt["acceptance_error"] or "command launch failed"
            stage_state["status"] = "failed"
            self.state["status"] = "failed"
        elif interrupted is not None:
            attempt["outcome"] = "interrupted"
            error_message = f"controller received signal {interrupted}"
            stage_state["status"] = "interrupted"
            self.state["status"] = "interrupted"
        elif timed_out:
            attempt["outcome"] = "timeout"
            error_message = f"command exceeded {command['timeout_seconds']} seconds"
            stage_state["status"] = "failed"
            self.state["status"] = "failed"
        elif return_code != 0:
            attempt["outcome"] = "nonzero_exit"
            error_message = f"command exited with {return_code!r}"
            stage_state["status"] = "failed"
            self.state["status"] = "failed"
        else:
            try:
                observations = evaluate_acceptance(stage["acceptance"])
            except PredicateFailed as error:
                attempt["outcome"] = "acceptance_failed"
                attempt["acceptance_error"] = str(error)
                error_message = str(error)
                stage_state["status"] = "failed"
                self.state["status"] = "failed"
            else:
                attempt["outcome"] = "accepted"
                attempt["acceptance"] = observations
                stage_state["status"] = "accepted"
                stage_state["accepted_at"] = _utc_now()
                stage_state["accepted_artifacts"] = observations
                self.state["status"] = "running"
        self.state["current_stage"] = None
        self._save()
        self._event(
            "stage_attempt_end",
            stage_id=stage_id,
            attempt=ordinal,
            outcome=attempt["outcome"],
            exit_code=return_code,
            error=error_message,
        )
        return attempt["outcome"] == "accepted", error_message

    def run(self) -> dict[str, Any]:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        with _singleton_lock(self.lock_path):
            self.state = _load_state(self.state_path, self.config)
            self._install_signal_handlers()
            try:
                self._recover_stale_running_record()
                start_index = self._revalidate_accepted_prefix()
                if start_index == len(REQUIRED_STAGE_IDS):
                    self.state["status"] = "complete"
                    self.state["completed_at"] = self.state.get("completed_at", _utc_now())
                    self._save()
                    self._event("pipeline_already_complete")
                    return self.state
                for index in range(start_index, len(self.config["stages"])):
                    stage = self.config["stages"][index]
                    stage_id = stage["id"]
                    # Re-check the whole prefix immediately before every launch.
                    observed_start = self._revalidate_accepted_prefix()
                    if observed_start != index:
                        raise PipelineBlockedError(
                            f"refusing {stage_id}: accepted prefix ends at index {observed_start}, "
                            f"expected {index}"
                        )
                    retry = stage["retry"]
                    last_error: str | None = None
                    for local_attempt in range(retry["max_retries"] + 1):
                        if self._received_signal is not None:
                            raise KeyboardInterrupt
                        accepted, last_error = self._run_attempt(stage)
                        if accepted:
                            break
                        if self._received_signal is not None:
                            raise KeyboardInterrupt
                        if local_attempt < retry["max_retries"]:
                            self._event(
                                "stage_retry_scheduled",
                                stage_id=stage_id,
                                local_attempt=local_attempt + 1,
                                delay_seconds=retry["delay_seconds"],
                            )
                            self._wait_retry_delay(retry["delay_seconds"])
                    else:
                        raise StageFailed(
                            f"stage {stage_id} failed after "
                            f"{retry['max_retries'] + 1} attempt(s): {last_error}"
                        )
                self.state["status"] = "complete"
                self.state["current_stage"] = None
                self.state["completed_at"] = _utc_now()
                self._save()
                self._event("pipeline_complete")
                return self.state
            except KeyboardInterrupt as error:
                self.state["status"] = "interrupted"
                self.state["current_stage"] = None
                self._save()
                self._event("pipeline_interrupted", signal=self._received_signal)
                raise StageFailed("pipeline interrupted") from error
            except PipelineError:
                if self.state:
                    if self.state.get("status") == "running":
                        self.state["status"] = "failed"
                    self.state["current_stage"] = None
                    self._save()
                raise
            finally:
                if self._active_process is not None:
                    _terminate_group(self._active_process, self._active_grace_seconds)
                    self._active_process = None
                self._restore_signal_handlers()

    def status(self) -> dict[str, Any]:
        self.state = _load_state(self.state_path, self.config)
        accepted_valid: dict[str, bool] = {}
        for stage in self.config["stages"]:
            stage_id = stage["id"]
            if self.state["stages"][stage_id]["status"] != "accepted":
                accepted_valid[stage_id] = False
                continue
            try:
                observations = evaluate_acceptance(stage["acceptance"])
            except PredicateFailed:
                accepted_valid[stage_id] = False
            else:
                accepted_valid[stage_id] = _json_equal(
                    observations,
                    self.state["stages"][stage_id].get("accepted_artifacts"),
                )
        return {
            "pipeline_id": self.config["pipeline_id"],
            "state_path": str(self.state_path),
            "status": self.state["status"],
            "current_stage": self.state["current_stage"],
            "accepted_artifacts_valid": accepted_valid,
            "state": self.state,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("validate", "status", "run"):
        child = subparsers.add_parser(action)
        child.add_argument("--config", type=Path, required=True)
    return parser


def _print_error(error: BaseException) -> NoReturn:
    print(
        json.dumps(
            {"status": "blocked", "error_type": type(error).__name__, "error": str(error)},
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config, _payload = load_config(args.config)
        if args.action == "validate":
            result: object = {
                "status": "valid",
                "pipeline_id": config["pipeline_id"],
                "config_sha256": config["config_sha256"],
                "runtime_root": config["runtime_root"],
                "stage_ids": [stage["id"] for stage in config["stages"]],
            }
        else:
            runner = PipelineRunner(config)
            result = runner.status() if args.action == "status" else runner.run()
    except PipelineError as error:
        _print_error(error)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
