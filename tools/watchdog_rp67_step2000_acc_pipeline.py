#!/usr/bin/env python3
"""Bounded, fail-closed watchdog for the restartable RP67 ACC controller.

The watchdog never signals a controller.  It adopts one exact, identity-bound
controller and otherwise waits.  A replacement is started only after the ACC
completion marker is still absent, no controller is live, the controller lock
is available, and GPU0/1 have no compute processes.  The controller remains
the authority for strict RP67 handoff, artifact validation, and idempotent
resume of completed ACC stages.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
from functools import cache
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPOSITORY_ROOT / ".venv312/bin/python"
CONTROLLER = REPOSITORY_ROOT / "tools/run_rp67_step2000_acc_pipeline.py"
PIPELINE_ROOT = REPOSITORY_ROOT / (
    "artifacts/representation_experiments/image_axis_grounding/evaluation/"
    "rp67_step2000_acc_pipeline_20260801"
)
COMPLETE_MARKER = REPOSITORY_ROOT / (
    "artifacts/representation_experiments/image_axis_grounding/evaluation/"
    "rp67_step2000_all_validations_complete_v2.json"
)
WATCHDOG_ROOT = PIPELINE_ROOT / "watchdog"
PIPELINE_LOCK = PIPELINE_ROOT / "pipeline.lock"
WATCHDOG_SCHEMA = "rp67-step2000-acc-watchdog-v1"
MAX_CONSECUTIVE_FAILURES = 3
SIGNIFICANT_PROGRESS_EVENTS = frozenset(
    {"generation_complete", "semantic_complete", "pipeline_complete"}
)


class WatchdogBlockedError(RuntimeError):
    """Raised when an identity or ownership condition cannot be proved."""


@dataclass(frozen=True)
class ControllerIdentity:
    pid: int
    uid: int
    boot_id: str
    starttime_ticks: int
    argv_sha256: str


@dataclass(frozen=True)
class RestartGate:
    controller_pids: tuple[int, ...]
    gpu_compute_pids: Mapping[int, tuple[int, ...]]
    pipeline_lock_available: bool
    judge_endpoint_open: bool

    @property
    def wait_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.controller_pids:
            reasons.append(f"controller processes are live: {self.controller_pids}")
        busy = {
            gpu: pids for gpu, pids in self.gpu_compute_pids.items() if pids
        }
        if busy:
            reasons.append(f"GPU0/1 have compute processes: {busy}")
        if not self.pipeline_lock_available:
            reasons.append("the ACC pipeline lock is held")
        if self.judge_endpoint_open:
            reasons.append("the semantic-judge endpoint on port 8013 is open")
        return tuple(reasons)

    @property
    def safe(self) -> bool:
        return not self.wait_reasons


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(
            (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_event(path: Path, event: str, **fields: object) -> None:
    value = {"event": event, "at": _utc_now(), **fields}
    with path.open("ab", buffering=0) as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        os.fsync(handle.fileno())


@cache
def _controller_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rp67_acc_controller", CONTROLLER)
    if spec is None or spec.loader is None:
        raise WatchdogBlockedError("cannot load the pinned ACC controller")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def _proc_state_and_starttime(pid: int) -> tuple[str, int] | None:
    try:
        value = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    closing = value.rfind(")")
    fields = value[closing + 2 :].split()
    return fields[0], int(fields[19])


def _proc_starttime(pid: int) -> int | None:
    observed = _proc_state_and_starttime(pid)
    return None if observed is None else observed[1]


def _process_has_exited(pid: int, *, expected_starttime: int) -> bool:
    observed = _proc_state_and_starttime(pid)
    return (
        observed is None
        or observed[0] in {"Z", "X"}
        or observed[1] != expected_starttime
    )


def _lexical_absolute(value: str, *, cwd: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = cwd / path
    return Path(os.path.abspath(path))


def _controller_argv_is_valid(argv: Sequence[str], *, cwd: Path) -> bool:
    if len(argv) < 5:
        return False
    if _lexical_absolute(argv[0], cwd=cwd) != PYTHON.absolute():
        return False
    try:
        script_index = argv.index(next(item for item in argv if item.endswith(CONTROLLER.name)))
    except (StopIteration, ValueError):
        return False
    if tuple(argv[1:script_index]) != ("-u",):
        return False
    if _lexical_absolute(argv[script_index], cwd=cwd) != CONTROLLER.absolute():
        return False
    controller_args = tuple(argv[script_index + 1 :])
    if controller_args == ("run", "--execute"):
        return True
    if len(controller_args) != 4:
        return False
    if controller_args[0] != "--poll-seconds" or controller_args[2:] != (
        "run",
        "--execute",
    ):
        return False
    try:
        return 0 < float(controller_args[1]) <= 60
    except ValueError:
        return False


def _inspect_controller(pid: int) -> ControllerIdentity | None:
    proc = Path("/proc") / str(pid)
    proc_state = _proc_state_and_starttime(pid)
    if proc_state is None or proc_state[0] in {"Z", "X"}:
        return None
    starttime = proc_state[1]
    try:
        uid = proc.stat().st_uid
        cwd = Path(os.readlink(proc / "cwd"))
        executable = Path(os.readlink(proc / "exe")).resolve()
        argv_bytes = (proc / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    if not argv_bytes:
        if _process_has_exited(pid, expected_starttime=starttime):
            return None
        raise WatchdogBlockedError(f"live controller candidate PID {pid} has empty argv")
    if uid != os.getuid():
        raise WatchdogBlockedError(f"controller candidate PID {pid} has another uid")
    if cwd.resolve() != REPOSITORY_ROOT:
        raise WatchdogBlockedError(f"controller candidate PID {pid} has another cwd")
    if executable != PYTHON.resolve():
        raise WatchdogBlockedError(f"controller candidate PID {pid} has another executable")
    argv = tuple(
        item.decode("utf-8", errors="surrogateescape")
        for item in argv_bytes.rstrip(b"\0").split(b"\0")
    )
    if not _controller_argv_is_valid(argv, cwd=cwd):
        if _process_has_exited(pid, expected_starttime=starttime):
            return None
        raise WatchdogBlockedError(f"controller candidate PID {pid} has unexpected argv")
    return ControllerIdentity(
        pid=pid,
        uid=uid,
        boot_id=_boot_id(),
        starttime_ticks=starttime,
        argv_sha256=sha256(argv_bytes).hexdigest(),
    )


def _identity_is_live(identity: ControllerIdentity) -> bool:
    observed = _proc_state_and_starttime(identity.pid)
    if (
        observed is None
        or observed[0] in {"Z", "X"}
        or observed[1] != identity.starttime_ticks
    ):
        return False
    inspected = _inspect_controller(identity.pid)
    if inspected is None:
        return False
    if inspected != identity:
        raise WatchdogBlockedError("live controller identity drifted")
    return True


def _scan_controllers() -> tuple[ControllerIdentity, ...]:
    matches: list[ControllerIdentity] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError):
            continue
        if CONTROLLER.name.encode() not in raw:
            continue
        try:
            observed = _inspect_controller(int(entry.name))
        except WatchdogBlockedError:
            # A command mentioning the filename is not necessarily an ACC
            # controller (for example an editor or test command).  Only exact
            # controller argv is adoptable.
            continue
        if observed is not None:
            matches.append(observed)
    return tuple(sorted(matches, key=lambda item: item.pid))


def _marker_is_complete() -> bool:
    controller = _controller_module()
    try:
        return bool(controller._existing_complete_marker_is_valid())
    except controller.PipelineBlockedError as error:
        raise WatchdogBlockedError(f"ACC marker validation failed: {error}") from error


def _assert_controller_preflight() -> None:
    controller = _controller_module()
    try:
        controller._assert_pinned_files()
    except controller.PipelineBlockedError as error:
        raise WatchdogBlockedError(f"ACC controller preflight failed: {error}") from error


def _pipeline_lock_is_available() -> bool:
    if PIPELINE_LOCK.exists() and (PIPELINE_LOCK.is_symlink() or not PIPELINE_LOCK.is_file()):
        raise WatchdogBlockedError("ACC pipeline lock is not a regular file")
    PIPELINE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = PIPELINE_LOCK.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return True
    finally:
        handle.close()


def _restart_gate() -> RestartGate:
    controller = _controller_module()
    identities = _scan_controllers()
    try:
        gpu_pids = controller._gpu_compute_pids((0, 1))
        endpoint_open = bool(controller._judge_endpoint_is_open())
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise WatchdogBlockedError(f"cannot audit GPU0/1 before restart: {error}") from error
    return RestartGate(
        controller_pids=tuple(item.pid for item in identities),
        gpu_compute_pids=gpu_pids,
        pipeline_lock_available=_pipeline_lock_is_available(),
        judge_endpoint_open=endpoint_open,
    )


def _progress_token(events_path: Path) -> str:
    significant: list[dict[str, Any]] = []
    if events_path.exists():
        if events_path.is_symlink() or not events_path.is_file():
            raise WatchdogBlockedError("ACC event ledger is not a regular file")
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise WatchdogBlockedError("ACC event ledger contains invalid JSON") from error
            if value.get("event") in SIGNIFICANT_PROGRESS_EVENTS:
                significant.append(value)
    return sha256(_canonical_bytes(significant)).hexdigest()


def _backoff_seconds(failures: int, *, base: float, maximum: float) -> float:
    if failures <= 0 or base <= 0 or maximum <= 0:
        raise ValueError("backoff inputs must be positive")
    return min(maximum, base * (2 ** (failures - 1)))


def _record_controller_failure(
    state: dict[str, Any],
    *,
    now_epoch: float,
    base_backoff: float,
    maximum_backoff: float,
) -> int:
    """Advance the persisted failure budget exactly once for one departed PID."""
    if state.get("controller") is None:
        raise WatchdogBlockedError("cannot count a controller failure twice")
    failures = int(state["consecutive_failures"]) + 1
    if failures > MAX_CONSECUTIVE_FAILURES:
        raise WatchdogBlockedError("controller failure budget was already exhausted")
    state["controller"] = None
    state["consecutive_failures"] = failures
    state["next_restart_not_before_epoch"] = now_epoch + _backoff_seconds(
        failures, base=base_backoff, maximum=maximum_backoff
    )
    return failures


def _controller_command(poll_seconds: float) -> list[str]:
    return [
        str(PYTHON),
        "-u",
        str(CONTROLLER),
        "--poll-seconds",
        str(poll_seconds),
        "run",
        "--execute",
    ]


def _start_controller(*, poll_seconds: float, log_path: Path) -> tuple[subprocess.Popen[bytes], ControllerIdentity]:
    log = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            _controller_command(poll_seconds),
            cwd=REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        observed = _inspect_controller(process.pid)
        if observed is not None:
            return process, observed
        if process.poll() is not None:
            break
        time.sleep(0.05)
    raise WatchdogBlockedError(
        f"replacement controller PID {process.pid} exited before identity capture"
    )


def _new_state(*, identity: ControllerIdentity | None, progress_token: str) -> dict[str, Any]:
    return {
        "schema_version": WATCHDOG_SCHEMA,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "controller": asdict(identity) if identity is not None else None,
        "consecutive_failures": 0,
        "total_restarts": 0,
        "last_progress_token": progress_token,
        "next_restart_not_before_epoch": None,
        "completed": False,
    }


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise WatchdogBlockedError("watchdog state is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "controller",
        "consecutive_failures",
        "total_restarts",
        "last_progress_token",
        "next_restart_not_before_epoch",
        "completed",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise WatchdogBlockedError("watchdog state is malformed")
    if value["schema_version"] != WATCHDOG_SCHEMA:
        raise WatchdogBlockedError("watchdog state schema mismatch")
    failures = value["consecutive_failures"]
    if not isinstance(failures, int) or not 0 <= failures <= MAX_CONSECUTIVE_FAILURES:
        raise WatchdogBlockedError("watchdog failure counter is invalid")
    return value


def _identity_from_state(value: object) -> ControllerIdentity | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise WatchdogBlockedError("watchdog controller identity is malformed")
    try:
        return ControllerIdentity(
            pid=int(value["pid"]),
            uid=int(value["uid"]),
            boot_id=str(value["boot_id"]),
            starttime_ticks=int(value["starttime_ticks"]),
            argv_sha256=str(value["argv_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise WatchdogBlockedError("watchdog controller identity is malformed") from error


def _validate_initial_identity(args: argparse.Namespace) -> ControllerIdentity | None:
    if args.initial_controller_pid is None:
        return None
    starttime = _proc_starttime(args.initial_controller_pid)
    if starttime is None:
        return None
    if starttime != args.initial_controller_starttime_ticks or _boot_id() != args.initial_controller_boot_id:
        raise WatchdogBlockedError("initial controller PID identity no longer matches")
    observed = _inspect_controller(args.initial_controller_pid)
    if observed is None:
        return None
    if (
        observed.starttime_ticks != args.initial_controller_starttime_ticks
        or observed.boot_id != args.initial_controller_boot_id
    ):
        raise WatchdogBlockedError("initial controller identity failed its binding")
    return observed


def _heartbeat(
    path: Path,
    *,
    status: str,
    state: Mapping[str, Any],
    wait_reasons: Sequence[str] = (),
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": WATCHDOG_SCHEMA,
            "status": status,
            "pid": os.getpid(),
            "updated_at": _utc_now(),
            "controller": state.get("controller"),
            "consecutive_failures": state["consecutive_failures"],
            "total_restarts": state["total_restarts"],
            "next_restart_not_before_epoch": state["next_restart_not_before_epoch"],
            "wait_reasons": list(wait_reasons),
            "complete_marker": str(COMPLETE_MARKER),
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    run = subparsers.add_parser("run")
    run.add_argument("--execute", action="store_true", required=True)
    run.add_argument("--initial-controller-pid", type=int)
    run.add_argument("--initial-controller-starttime-ticks", type=int)
    run.add_argument("--initial-controller-boot-id")
    run.add_argument("--restart-backoff-seconds", type=float, default=30.0)
    run.add_argument("--maximum-backoff-seconds", type=float, default=120.0)
    return parser


def _status() -> dict[str, object]:
    identities = _scan_controllers()
    state = _load_state(WATCHDOG_ROOT / "state.json")
    return {
        "complete": _marker_is_complete(),
        "controllers": [asdict(item) for item in identities],
        "pipeline_lock_available": _pipeline_lock_is_available(),
        "state": state,
    }


def _run(args: argparse.Namespace) -> int:
    WATCHDOG_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = WATCHDOG_ROOT / "watchdog.lock"
    if lock_path.exists() and (lock_path.is_symlink() or not lock_path.is_file()):
        raise WatchdogBlockedError("watchdog lock is not a regular file")
    lock_handle = lock_path.open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise WatchdogBlockedError("another ACC watchdog is active") from error

    events_path = WATCHDOG_ROOT / "events.jsonl"
    heartbeat_path = WATCHDOG_ROOT / "heartbeat.json"
    state_path = WATCHDOG_ROOT / "state.json"
    controller_log = WATCHDOG_ROOT / "controller-restarts.log"
    owned_children: dict[int, subprocess.Popen[bytes]] = {}

    if _marker_is_complete():
        state = _load_state(state_path) or _new_state(
            identity=None, progress_token=_progress_token(PIPELINE_ROOT / "events.jsonl")
        )
        state["completed"] = True
        state["updated_at"] = _utc_now()
        _atomic_json(state_path, state)
        _heartbeat(heartbeat_path, status="complete", state=state)
        return 0

    state = _load_state(state_path)
    initial = _validate_initial_identity(args)
    if state is None:
        identities = _scan_controllers()
        if initial is not None:
            if identities != (initial,):
                raise WatchdogBlockedError(
                    "initial controller is not the sole live ACC controller"
                )
            identity = initial
        elif len(identities) == 1:
            identity = identities[0]
        elif len(identities) > 1:
            raise WatchdogBlockedError("multiple ACC controllers are live")
        else:
            identity = None
        state = _new_state(
            identity=identity,
            progress_token=_progress_token(PIPELINE_ROOT / "events.jsonl"),
        )
        _atomic_json(state_path, state)
    elif initial is not None:
        persisted = _identity_from_state(state["controller"])
        if persisted is not None and persisted != initial and _identity_is_live(persisted):
            raise WatchdogBlockedError("initial controller conflicts with watchdog state")
        state["controller"] = asdict(initial)
        state["updated_at"] = _utc_now()
        _atomic_json(state_path, state)

    _append_event(
        events_path,
        "watchdog_started",
        pid=os.getpid(),
        controller=state["controller"],
        consecutive_failures=state["consecutive_failures"],
    )

    while True:
        if _marker_is_complete():
            state["completed"] = True
            state["updated_at"] = _utc_now()
            _atomic_json(state_path, state)
            _heartbeat(heartbeat_path, status="complete", state=state)
            _append_event(events_path, "watchdog_complete", marker=str(COMPLETE_MARKER))
            return 0

        progress = _progress_token(PIPELINE_ROOT / "events.jsonl")
        if progress != state["last_progress_token"]:
            state["last_progress_token"] = progress
            state["consecutive_failures"] = 0
            state["updated_at"] = _utc_now()
            _atomic_json(state_path, state)
            _append_event(events_path, "durable_acc_progress_observed")

        identity = _identity_from_state(state["controller"])
        if identity is not None and _identity_is_live(identity):
            _heartbeat(heartbeat_path, status="monitoring_controller", state=state)
            time.sleep(args.poll_seconds)
            continue

        if identity is not None:
            child = owned_children.pop(identity.pid, None)
            returncode = child.poll() if child is not None else None
            failures = _record_controller_failure(
                state,
                now_epoch=time.time(),
                base_backoff=args.restart_backoff_seconds,
                maximum_backoff=args.maximum_backoff_seconds,
            )
            state["updated_at"] = _utc_now()
            _atomic_json(state_path, state)
            _append_event(
                events_path,
                "controller_exited_before_marker",
                controller=asdict(identity),
                returncode=returncode,
                consecutive_failures=failures,
                next_restart_not_before_epoch=state["next_restart_not_before_epoch"],
            )

        if int(state["consecutive_failures"]) >= MAX_CONSECUTIVE_FAILURES:
            _heartbeat(heartbeat_path, status="failure_limit_reached", state=state)
            _append_event(
                events_path,
                "watchdog_failure_limit_reached",
                consecutive_failures=state["consecutive_failures"],
            )
            return 4

        not_before = state["next_restart_not_before_epoch"]
        if isinstance(not_before, (int, float)) and time.time() < not_before:
            remaining = max(0.0, not_before - time.time())
            _heartbeat(
                heartbeat_path,
                status="restart_backoff",
                state=state,
                wait_reasons=(f"backoff has {remaining:.1f}s remaining",),
            )
            time.sleep(min(args.poll_seconds, remaining))
            continue

        gate = _restart_gate()
        if gate.controller_pids:
            identities = _scan_controllers()
            if len(identities) != 1:
                raise WatchdogBlockedError("cannot uniquely adopt a live ACC controller")
            state["controller"] = asdict(identities[0])
            state["next_restart_not_before_epoch"] = None
            state["updated_at"] = _utc_now()
            _atomic_json(state_path, state)
            _append_event(
                events_path,
                "external_controller_adopted",
                controller=state["controller"],
            )
            continue
        if not gate.safe:
            _heartbeat(
                heartbeat_path,
                status="waiting_for_safe_restart",
                state=state,
                wait_reasons=gate.wait_reasons,
            )
            time.sleep(args.poll_seconds)
            continue

        _assert_controller_preflight()
        process, replacement = _start_controller(
            poll_seconds=args.poll_seconds, log_path=controller_log
        )
        owned_children[process.pid] = process
        state["controller"] = asdict(replacement)
        state["total_restarts"] = int(state["total_restarts"]) + 1
        state["next_restart_not_before_epoch"] = None
        state["updated_at"] = _utc_now()
        _atomic_json(state_path, state)
        _append_event(
            events_path,
            "controller_restarted",
            controller=state["controller"],
            total_restarts=state["total_restarts"],
        )


def main() -> int:
    args = _parser().parse_args()
    if not 0 < args.poll_seconds <= 60:
        raise ValueError("--poll-seconds must be in (0, 60]")
    if args.command == "status":
        print(json.dumps(_status(), indent=2, sort_keys=True))
        return 0
    identity_values = (
        args.initial_controller_pid,
        args.initial_controller_starttime_ticks,
        args.initial_controller_boot_id,
    )
    if any(value is not None for value in identity_values) and not all(
        value is not None for value in identity_values
    ):
        raise ValueError("all initial-controller identity arguments are required together")
    if args.restart_backoff_seconds <= 0 or args.maximum_backoff_seconds <= 0:
        raise ValueError("restart backoff values must be positive")
    if args.maximum_backoff_seconds < args.restart_backoff_seconds:
        raise ValueError("maximum backoff must be at least the initial backoff")
    return _run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WatchdogBlockedError as error:
        print(f"ACC_WATCHDOG_BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(3) from error
