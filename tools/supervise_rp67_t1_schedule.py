#!/usr/bin/python3 -I
"""Safely supervise the one-off RP67/T1 GPU handoff schedule.

The supervisor never signals a process discovered only from GPU utilization.
It adopts and fingerprints the exact T1 worker session leaders recorded under
the T1 runtime directory, and it consumes an explicit RP67 validation-complete
marker before assigning GPUs 0--1 to T1.

``status`` is read-only.  ``run --execute`` owns a durable singleton lock and
reconciles the live machine every few seconds; all transitions are resumable.
"""

from __future__ import annotations
# ruff: noqa: E402

# Direct script execution is stopped before legacy path/environment mutation or
# heavyweight runtime imports. Importing the module for read-only compatibility
# tests remains possible; its public ``main`` retains a second fail-closed guard.
if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range(2):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            "tools/supervise_rp67_t1_schedule.py",
        ),
    )

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.ops.launch_gate import (  # noqa: E402
    LaunchGateError,
    consume_launch_authorization,
    make_run_identity,
    materialize_ready_receipt,
)
from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)


DEFAULT_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/data_selection/"
    "qwen3_instruct_t1_512_vstar170k_arxiv32k_thinklite69842_v1.json"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "artifacts/data/policy_selection/t1/"
    "T1-04-QWEN3-INSTRUCT-512-FULLIMAGE-271842-GPU0123"
)
DEFAULT_ACC_MARKER = REPOSITORY_ROOT / (
    "artifacts/representation_experiments/image_axis_grounding/evaluation/"
    "rp67_step2000_all_validations_complete_v2.json"
)
DEFAULT_CUTOFF = "2026-08-01T07:40:00+09:00"
DEFAULT_EXPECTED_CHUNKS = (17046, 16918, 17045, 16954)
SUPERVISOR_RUNTIME_NAME = "supervisor-rp67-t1-v2-20260801"
EXPECTED_T1_RUN_ID = "T1-04-QWEN3-INSTRUCT-512-FULLIMAGE-271842-GPU0123"
EXPECTED_T1_RUN_MANIFEST_SHA256 = (
    "bdc49eba27ff16aec58ac1116b7eda2a9148f62c334a6fbdd6385502fdf2141f"
)
EXPECTED_RP67_RUN_ID = (
    "RP-67-QWEN3-INSTRUCT-REP-BALANCED-T1-IMAGE-AXIS-GROUNDED-2000-GPU01"
)
STATE_SCHEMA = "rp67-t1-gpu-supervisor-state-v1"
MARKER_SCHEMA = "rp67-all-validations-complete-v2"
SEMANTIC_SCHEMA = "answer-utility-semantic-rescore-v2"
HEARTBEAT_SCHEMA = "rp67-t1-gpu-supervisor-heartbeat-v1"
COMPLETE_SCHEMA = "t1-revision0-supervisor-complete-v1"
EXECUTION_POLICY = REPOSITORY_ROOT / "configs/ops/experiment_execution_policy.json"
WORKER_FILE_RE = re.compile(
    r"^rank-(?P<rank>[0-3])(?:-subshard-(?P<index>\d+)-of-(?P<count>\d+))?\.pgid$"
)
MANIFEST_RE = re.compile(r"^rank-(?P<rank>\d{2})-chunk-(?P<index>\d{6})\.json$")


class SupervisorBlockedError(RuntimeError):
    """Raised when continuing could target the wrong process or GPU."""


@dataclass(frozen=True, slots=True)
class Worker:
    tag: str
    pgid_path: str
    pgid: int
    rank: int
    gpu: int
    subshard_count: int
    subshard_index: int
    broad: bool
    uid: int
    boot_id: str
    starttime_ticks: int
    argv_sha256: str

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.rank, self.gpu, self.subshard_count, self.subshard_index)


@dataclass(frozen=True, slots=True)
class DesiredWorker:
    rank: int
    gpu: int
    subshard_count: int
    subshard_index: int

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.rank, self.gpu, self.subshard_count, self.subshard_index)


@dataclass(frozen=True, slots=True)
class Snapshot:
    observed_at: str
    manifest_counts: tuple[int, int, int, int]
    ranks_complete: tuple[bool, bool, bool, bool]
    workers: tuple[Worker, ...]
    gpu_compute_pids: Mapping[int, tuple[int, ...]]
    acc_complete: bool
    cutoff_reached: bool


@dataclass(frozen=True, slots=True)
class Plan:
    stage: str
    stop_tags: tuple[str, ...]
    desired_workers: tuple[DesiredWorker, ...]
    launch_workers: tuple[DesiredWorker, ...]
    wait_reasons: tuple[str, ...]
    should_latch_cutoff: bool
    complete: bool


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    payload = _canonical_bytes(value) + b"\n"
    with path.open("ab", buffering=0) as handle:
        handle.write(payload)
        os.fsync(handle.fileno())


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("deadline must include an explicit UTC offset")
    return parsed


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f-]{36}", value):
        raise SupervisorBlockedError("kernel boot_id is malformed")
    return value


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise SupervisorBlockedError(f"cannot inspect process group {pgid}") from error
    return True


def _proc_identity(pid: int) -> tuple[int, int, int, int, list[str]]:
    proc = Path("/proc") / str(pid)
    stat_text = (proc / "stat").read_text(encoding="ascii")
    closing = stat_text.rfind(")")
    if closing < 0:
        raise SupervisorBlockedError(f"cannot parse /proc/{pid}/stat")
    fields = stat_text[closing + 2 :].split()
    if len(fields) < 20:
        raise SupervisorBlockedError(f"short /proc/{pid}/stat")
    pgrp = int(fields[2])
    session = int(fields[3])
    starttime = int(fields[19])
    uid = proc.stat().st_uid
    raw_argv = (proc / "cmdline").read_bytes().split(b"\0")
    argv = [item.decode("utf-8", errors="strict") for item in raw_argv if item]
    return pgrp, session, starttime, uid, argv


def _argument(argv: Sequence[str], name: str, *, required: bool = True) -> str | None:
    positions = [index for index, value in enumerate(argv) if value == name]
    if not positions:
        if required:
            raise SupervisorBlockedError(f"worker argv is missing {name}")
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise SupervisorBlockedError(f"worker argv has malformed {name}")
    return argv[positions[0] + 1]


def _validate_worker(
    pgid_path: Path,
    *,
    config_path: Path,
    expected_boot_id: str,
) -> Worker | None:
    match = WORKER_FILE_RE.fullmatch(pgid_path.name)
    if match is None:
        raise SupervisorBlockedError(f"unrecognized PGID record {pgid_path}")
    if pgid_path.is_symlink() or not pgid_path.is_file():
        raise SupervisorBlockedError(f"PGID record must be a regular file: {pgid_path}")
    raw = pgid_path.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[1-9]\d*", raw):
        raise SupervisorBlockedError(f"invalid PGID in {pgid_path}")
    pgid = int(raw)
    if not _group_alive(pgid):
        return None
    try:
        pgrp, session, starttime, uid, argv = _proc_identity(pgid)
    except FileNotFoundError as error:
        if _group_alive(pgid):
            raise SupervisorBlockedError(
                f"session leader {pgid} vanished while its group remains"
            ) from error
        return None
    if pgid != pgrp or pgid != session:
        raise SupervisorBlockedError(
            f"recorded worker {pgid} is not its own process-group/session leader"
        )
    if uid != os.getuid():
        raise SupervisorBlockedError(f"worker {pgid} belongs to another uid")
    if not argv:
        raise SupervisorBlockedError(f"worker {pgid} has an empty argv")
    expected_script = (
        REPOSITORY_ROOT / "tools/run_policy_data_selection_t1.py"
    ).resolve()
    script_positions = [
        index
        for index, value in enumerate(argv)
        if Path(value).resolve() == expected_script
    ]
    if len(script_positions) != 1 or "worker" not in argv[script_positions[0] + 1 :]:
        raise SupervisorBlockedError(f"PGID {pgid} is not the accepted T1 worker")
    observed_config = _argument(argv, "--config")
    if (
        observed_config is None
        or Path(observed_config).resolve() != config_path.resolve()
    ):
        raise SupervisorBlockedError(f"worker {pgid} uses another T1 config")
    rank = int(_argument(argv, "--rank") or "-1")
    gpu = int(_argument(argv, "--cuda-visible-device") or "-1")
    file_rank = int(match.group("rank"))
    if rank != file_rank or rank not in range(4) or gpu not in range(8):
        raise SupervisorBlockedError(f"worker {pgid} rank/GPU binding is invalid")
    count_text = match.group("count")
    index_text = match.group("index")
    broad = count_text is None
    if broad:
        if _argument(argv, "--chunk-subshard-count", required=False) is not None:
            raise SupervisorBlockedError(f"broad worker {pgid} carries subshard args")
        count, index = 1, 0
    else:
        count, index = int(count_text), int(index_text)
        if int(_argument(argv, "--chunk-subshard-count") or "-1") != count:
            raise SupervisorBlockedError(
                f"worker {pgid} subshard count differs from PGID file"
            )
        if int(_argument(argv, "--chunk-subshard-index") or "-1") != index:
            raise SupervisorBlockedError(
                f"worker {pgid} subshard index differs from PGID file"
            )
        if not 0 <= index < count:
            raise SupervisorBlockedError(
                f"worker {pgid} has invalid subshard coordinates"
            )
    return Worker(
        tag=pgid_path.stem,
        pgid_path=str(pgid_path),
        pgid=pgid,
        rank=rank,
        gpu=gpu,
        subshard_count=count,
        subshard_index=index,
        broad=broad,
        uid=uid,
        boot_id=expected_boot_id,
        starttime_ticks=starttime,
        argv_sha256=sha256(b"\0".join(item.encode() for item in argv)).hexdigest(),
    )


def _discover_workers(runtime_root: Path, config_path: Path) -> tuple[Worker, ...]:
    boot = _boot_id()
    workers: list[Worker] = []
    for path in sorted(runtime_root.glob("rank-*.pgid")):
        if WORKER_FILE_RE.fullmatch(path.name) is None:
            continue
        worker = _validate_worker(path, config_path=config_path, expected_boot_id=boot)
        if worker is not None:
            workers.append(worker)
    seen_groups: set[int] = set()
    seen_slots: set[tuple[int, int, int]] = set()
    for worker in workers:
        slot = (worker.rank, worker.subshard_count, worker.subshard_index)
        if worker.pgid in seen_groups or slot in seen_slots:
            raise SupervisorBlockedError("duplicate live T1 worker ownership")
        seen_groups.add(worker.pgid)
        seen_slots.add(slot)
    live_counts_by_rank: dict[int, set[int]] = {}
    for worker in workers:
        live_counts_by_rank.setdefault(worker.rank, set()).add(worker.subshard_count)
    if any(len(counts) != 1 for counts in live_counts_by_rank.values()):
        raise SupervisorBlockedError(
            "one logical rank has incompatible live topologies"
        )
    return tuple(workers)


def _manifest_coverage(
    manifest_root: Path, expected: Sequence[int]
) -> tuple[tuple[int, int, int, int], tuple[bool, bool, bool, bool]]:
    indices: list[set[int]] = [set() for _ in range(4)]
    for path in manifest_root.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        match = MANIFEST_RE.fullmatch(path.name)
        if match is None:
            continue
        rank = int(match.group("rank"))
        index = int(match.group("index"))
        if rank not in range(4):
            raise SupervisorBlockedError(f"unexpected manifest rank in {path}")
        if not 0 <= index < expected[rank]:
            raise SupervisorBlockedError(f"unexpected revision-0 chunk index in {path}")
        indices[rank].add(index)
    counts = tuple(len(values) for values in indices)
    complete = tuple(
        len(values) == expected[rank]
        and min(values, default=0) == 0
        and max(values, default=-1) == expected[rank] - 1
        for rank, values in enumerate(indices)
    )
    return counts, complete  # type: ignore[return-value]


def _gpu_compute_pids() -> dict[int, tuple[int, ...]]:
    gpu_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.splitlines()
    uuid_to_index: dict[str, int] = {}
    for row in gpu_rows:
        index_text, uuid = (part.strip() for part in row.split(",", 1))
        uuid_to_index[uuid] = int(index_text)
    result: dict[int, list[int]] = {index: [] for index in uuid_to_index.values()}
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.splitlines()
    for row in rows:
        if not row.strip():
            continue
        pid_text, uuid = (part.strip() for part in row.split(",", 1))
        if uuid not in uuid_to_index:
            raise SupervisorBlockedError(f"nvidia-smi returned unknown GPU UUID {uuid}")
        result[uuid_to_index[uuid]].append(int(pid_text))
    return {index: tuple(sorted(pids)) for index, pids in result.items()}


def _validated_acc_marker(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise SupervisorBlockedError("ACC completion marker is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupervisorBlockedError("ACC completion marker is malformed") from error
    if not isinstance(value, dict):
        raise SupervisorBlockedError("ACC completion marker must be an object")
    if (
        value.get("schema_version") != MARKER_SCHEMA
        or value.get("status") != "complete"
    ):
        raise SupervisorBlockedError("ACC completion marker schema/status mismatch")
    if value.get("rp67_run_id") != EXPECTED_RP67_RUN_ID:
        raise SupervisorBlockedError("ACC completion marker identifies another RP run")
    artifacts = value.get("artifacts")
    required = (
        "int_diag",
        "acc_first200",
        "acc_full867",
        "diag_first200_sixarm",
    )
    if not isinstance(artifacts, dict) or set(artifacts) != set(required):
        raise SupervisorBlockedError("ACC marker artifact set is incomplete")

    def validate_file_binding(
        record: object, *, expected_path: Path | None = None
    ) -> Path:
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise SupervisorBlockedError("ACC marker file binding is incomplete")
        artifact_path = Path(str(record.get("path", "")))
        expected_sha = str(record.get("sha256", ""))
        if (
            not artifact_path.is_absolute()
            or (expected_path is not None and artifact_path != expected_path)
            or artifact_path.is_symlink()
            or not artifact_path.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or _sha256_file(artifact_path) != expected_sha
        ):
            raise SupervisorBlockedError("ACC marker file binding drifted")
        return artifact_path

    validate_file_binding(artifacts["int_diag"])
    for name in required[1:]:
        record = artifacts[name]
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise SupervisorBlockedError(f"ACC marker artifact {name} is incomplete")
        root = Path(str(record.get("root", "")))
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise SupervisorBlockedError(f"ACC marker artifact {name} root drifted")
        summary_path = validate_file_binding(
            record.get("summary"), expected_path=root / "summary.json"
        )
        manifest_path = validate_file_binding(
            record.get("manifest"), expected_path=root / "manifest.json"
        )
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SupervisorBlockedError(
                f"ACC marker artifact {name} publication is malformed"
            ) from error
        run_identity = record.get("run_identity_sha256")
        manifest_files = manifest.get("files")
        manifest_summary = (
            manifest_files.get("summary") if isinstance(manifest_files, dict) else None
        )
        if (
            summary.get("schema_version") != SEMANTIC_SCHEMA
            or manifest.get("schema_version") != SEMANTIC_SCHEMA
            or summary.get("status") != "complete"
            or manifest.get("status") != "complete"
            or not isinstance(run_identity, str)
            or not re.fullmatch(r"[0-9a-f]{64}", run_identity)
            or summary.get("run_identity_sha256") != run_identity
            or manifest.get("run_identity_sha256") != run_identity
            or not isinstance(manifest_summary, dict)
            or manifest_summary.get("path") != "summary.json"
            or manifest_summary.get("sha256") != _sha256_file(summary_path)
        ):
            raise SupervisorBlockedError(
                f"ACC marker artifact {name} semantic binding drifted"
            )
    return True


def _snapshot(
    *,
    config_path: Path,
    output_root: Path,
    expected_chunks: Sequence[int],
    acc_marker: Path,
    cutoff: datetime,
    cutoff_latched: bool,
) -> Snapshot:
    counts, complete = _manifest_coverage(output_root / "manifests", expected_chunks)
    now = datetime.now(timezone.utc)
    return Snapshot(
        observed_at=now.isoformat(),
        manifest_counts=counts,
        ranks_complete=complete,
        workers=_discover_workers(output_root / "runtime", config_path),
        gpu_compute_pids=_gpu_compute_pids(),
        acc_complete=_validated_acc_marker(acc_marker),
        cutoff_reached=cutoff_latched or now >= cutoff.astimezone(timezone.utc),
    )


def _desired(count: int, gpus: Sequence[int]) -> tuple[DesiredWorker, ...]:
    if len(gpus) != count:
        raise ValueError("one GPU is required per desired subshard")
    return tuple(
        DesiredWorker(rank=2, gpu=gpu, subshard_count=count, subshard_index=index)
        for index, gpu in enumerate(gpus)
    )


def build_plan(snapshot: Snapshot, *, post_cutoff_count: int | None) -> Plan:
    workers = snapshot.workers
    rank2 = tuple(worker for worker in workers if worker.rank == 2)
    if snapshot.ranks_complete[2]:
        return Plan(
            stage="t1-complete" if all(snapshot.ranks_complete) else "rank2-complete",
            stop_tags=(),
            desired_workers=(),
            launch_workers=(),
            wait_reasons=(),
            should_latch_cutoff=snapshot.cutoff_reached,
            complete=all(snapshot.ranks_complete) and not workers,
        )

    if snapshot.acc_complete and snapshot.ranks_complete[1]:
        stage = "final-gpu0123"
        desired = _desired(4, (0, 1, 2, 3))
    elif snapshot.cutoff_reached:
        stage = "post-cutoff-gpu23"
        if not snapshot.ranks_complete[1]:
            desired = ()
        else:
            selected_count = post_cutoff_count
            if selected_count is None:
                selected_count = 6 if any(w.subshard_count == 6 for w in rank2) else 2
            if selected_count == 6:
                desired = (
                    DesiredWorker(2, 2, 6, 0),
                    DesiredWorker(2, 3, 6, 1),
                )
            elif selected_count == 2:
                desired = _desired(2, (2, 3))
            else:
                raise SupervisorBlockedError("invalid persisted post-cutoff topology")
    elif snapshot.ranks_complete[1] and snapshot.ranks_complete[3]:
        stage = "six-way-gpu234567"
        desired = _desired(6, (2, 3, 4, 5, 6, 7))
    elif snapshot.ranks_complete[3]:
        stage = "four-way-gpu4567"
        desired = _desired(4, (4, 5, 6, 7))
    else:
        return Plan(
            stage="initial",
            stop_tags=(),
            desired_workers=(),
            launch_workers=(),
            wait_reasons=("rank3 is not complete",),
            should_latch_cutoff=False,
            complete=False,
        )

    desired_keys = {worker.key for worker in desired}
    current_keys = {worker.key for worker in rank2}
    stop: set[str] = set()
    if snapshot.cutoff_reached:
        stop.update(worker.tag for worker in workers if worker.gpu >= 4)
    stop.update(worker.tag for worker in rank2 if worker.key not in desired_keys)
    if stop:
        return Plan(
            stage=stage,
            stop_tags=tuple(sorted(stop)),
            desired_workers=desired,
            launch_workers=(),
            wait_reasons=("stop incompatible topology before launching",),
            should_latch_cutoff=snapshot.cutoff_reached,
            complete=False,
        )

    missing = tuple(worker for worker in desired if worker.key not in current_keys)
    waits: list[str] = []
    for worker in missing:
        if snapshot.gpu_compute_pids.get(worker.gpu, ()):
            waits.append(
                f"GPU{worker.gpu} has compute PIDs "
                f"{snapshot.gpu_compute_pids[worker.gpu]}"
            )
    return Plan(
        stage=stage,
        stop_tags=(),
        desired_workers=desired,
        launch_workers=() if waits else missing,
        wait_reasons=tuple(waits),
        should_latch_cutoff=snapshot.cutoff_reached,
        complete=False,
    )


def _schedule_identity(
    *, config_path: Path, output_root: Path, cutoff: datetime, acc_marker: Path
) -> tuple[str, dict[str, object]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("run_id") != EXPECTED_T1_RUN_ID:
        raise SupervisorBlockedError("T1 config run_id mismatch")
    if Path(str(config.get("output_root", ""))).resolve() != output_root.resolve():
        raise SupervisorBlockedError("T1 config output_root mismatch")
    run_identity_path = output_root / "run-identity.json"
    run_identity = json.loads(run_identity_path.read_text(encoding="utf-8"))
    observed_manifest = run_identity.get("run_manifest_sha256")
    if observed_manifest is None:
        observed_manifest = run_identity.get("manifest_sha256")
    if observed_manifest is None and isinstance(run_identity.get("identity"), dict):
        observed_manifest = run_identity["identity"].get("manifest_sha256")
    if observed_manifest != EXPECTED_T1_RUN_MANIFEST_SHA256:
        raise SupervisorBlockedError("T1 run manifest identity mismatch")
    value: dict[str, object] = {
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "output_root": str(output_root),
        "run_manifest_sha256": EXPECTED_T1_RUN_MANIFEST_SHA256,
        "cutoff": cutoff.isoformat(),
        "acc_complete_marker": str(acc_marker),
        "expected_chunks": list(DEFAULT_EXPECTED_CHUNKS),
    }
    return sha256(_canonical_bytes(value)).hexdigest(), value


def _load_or_initialize_state(
    state_path: Path, *, schedule_sha256: str, schedule: Mapping[str, object]
) -> dict[str, Any]:
    if state_path.exists():
        if state_path.is_symlink() or not state_path.is_file():
            raise SupervisorBlockedError("supervisor state is not a regular file")
        value = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != STATE_SCHEMA
            or value.get("schedule_sha256") != schedule_sha256
            or value.get("schedule") != schedule
        ):
            raise SupervisorBlockedError("supervisor state identity mismatch")
        return value
    return {
        "schema_version": STATE_SCHEMA,
        "schedule_sha256": schedule_sha256,
        "schedule": dict(schedule),
        "cutoff_latched": False,
        "post_cutoff_count": None,
        "gpu47_cleared_at": None,
        "adopted_workers": {},
        "last_stage": None,
        "completed": False,
    }


def _worker_fingerprint(worker: Worker) -> dict[str, object]:
    return {
        "pgid": worker.pgid,
        "uid": worker.uid,
        "boot_id": worker.boot_id,
        "starttime_ticks": worker.starttime_ticks,
        "argv_sha256": worker.argv_sha256,
        "rank": worker.rank,
        "gpu": worker.gpu,
        "subshard_count": worker.subshard_count,
        "subshard_index": worker.subshard_index,
    }


def _assert_live_worker_identity(worker: Worker, *, allow_missing_leader: bool) -> None:
    if _boot_id() != worker.boot_id:
        raise SupervisorBlockedError("host rebooted after worker adoption")
    try:
        pgrp, session, starttime, uid, argv = _proc_identity(worker.pgid)
    except FileNotFoundError as error:
        if allow_missing_leader and _group_alive(worker.pgid):
            # A child can briefly outlive the adopted session leader during
            # escalation.  The process-group ID cannot be reused while that
            # child remains a member of the group.
            return
        raise SupervisorBlockedError(
            f"cannot revalidate session leader for {worker.tag}"
        ) from error
    argv_digest = sha256(b"\0".join(item.encode() for item in argv)).hexdigest()
    if (
        pgrp != worker.pgid
        or session != worker.pgid
        or starttime != worker.starttime_ticks
        or uid != worker.uid
        or argv_digest != worker.argv_sha256
    ):
        raise SupervisorBlockedError(f"process identity drift for {worker.tag}")


def _adopt_workers(
    state: dict[str, Any], workers: Iterable[Worker], journal_path: Path
) -> bool:
    changed = False
    adopted = state.setdefault("adopted_workers", {})
    for worker in workers:
        observed = _worker_fingerprint(worker)
        previous = adopted.get(worker.tag)
        if previous is not None and previous.get("pgid") == worker.pgid:
            if previous != observed:
                raise SupervisorBlockedError(
                    f"live worker fingerprint changed for {worker.tag}"
                )
            continue
        adopted[worker.tag] = observed
        _append_jsonl(
            journal_path,
            {
                "event": "worker_adopted",
                "at": datetime.now(timezone.utc).isoformat(),
                "tag": worker.tag,
                "fingerprint": observed,
            },
        )
        changed = True
    return changed


def _signal_groups(
    workers: Sequence[Worker],
    sig: signal.Signals,
    *,
    allow_missing_leader: bool,
) -> None:
    for worker in workers:
        if _group_alive(worker.pgid):
            _assert_live_worker_identity(
                worker, allow_missing_leader=allow_missing_leader
            )
            os.killpg(worker.pgid, sig)


def _wait_groups(workers: Sequence[Worker], seconds: float) -> tuple[Worker, ...]:
    deadline = time.monotonic() + seconds
    remaining = tuple(worker for worker in workers if _group_alive(worker.pgid))
    while remaining and time.monotonic() < deadline:
        time.sleep(1.0)
        remaining = tuple(worker for worker in workers if _group_alive(worker.pgid))
    return remaining


def _stop_workers(
    workers: Sequence[Worker], *, reason: str, journal_path: Path
) -> None:
    if not workers:
        return
    _append_jsonl(
        journal_path,
        {
            "event": "stop_intent",
            "at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "workers": [asdict(worker) for worker in workers],
        },
    )
    _signal_groups(workers, signal.SIGINT, allow_missing_leader=False)
    remaining = _wait_groups(workers, 60.0)
    if remaining:
        _append_jsonl(
            journal_path,
            {
                "event": "stop_escalated",
                "at": datetime.now(timezone.utc).isoformat(),
                "signal": "SIGTERM",
                "tags": [worker.tag for worker in remaining],
            },
        )
        _signal_groups(remaining, signal.SIGTERM, allow_missing_leader=True)
        remaining = _wait_groups(remaining, 30.0)
    if remaining:
        _append_jsonl(
            journal_path,
            {
                "event": "stop_escalated",
                "at": datetime.now(timezone.utc).isoformat(),
                "signal": "SIGKILL",
                "tags": [worker.tag for worker in remaining],
            },
        )
        _signal_groups(remaining, signal.SIGKILL, allow_missing_leader=True)
        remaining = _wait_groups(remaining, 15.0)
    if remaining:
        raise SupervisorBlockedError(
            "owned process groups survived SIGKILL: "
            + ", ".join(worker.tag for worker in remaining)
        )
    _append_jsonl(
        journal_path,
        {
            "event": "stop_complete",
            "at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "tags": [worker.tag for worker in workers],
        },
    )


def _launch_worker(
    desired: DesiredWorker,
    *,
    config_path: Path,
    output_root: Path,
    journal_path: Path,
) -> None:
    command = [
        str(REPOSITORY_ROOT / "tools/launch_policy_data_selection_t1_subshard.sh"),
        str(REPOSITORY_ROOT),
        str(config_path),
        str(output_root),
        str(desired.rank),
        str(desired.gpu),
        str(desired.subshard_count),
        str(desired.subshard_index),
    ]
    _append_jsonl(
        journal_path,
        {
            "event": "launch_intent",
            "at": datetime.now(timezone.utc).isoformat(),
            "worker": asdict(desired),
            "command": command,
        },
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30.0,
    )
    if completed.returncode != 0:
        raise SupervisorBlockedError(
            f"launcher failed for {desired}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    _append_jsonl(
        journal_path,
        {
            "event": "launcher_returned",
            "at": datetime.now(timezone.utc).isoformat(),
            "worker": asdict(desired),
            "stdout": completed.stdout.strip(),
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--acc-complete-marker", type=Path, default=DEFAULT_ACC_MARKER)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    run = subparsers.add_parser("run")
    run.add_argument("--execute", action="store_true", required=True)
    run.add_argument("--once", action="store_true")
    run.add_argument("--authorization-token", type=Path)
    run.add_argument("--freeze-override", type=Path)
    return parser


def _public_snapshot(snapshot: Snapshot) -> dict[str, object]:
    return {
        "observed_at": snapshot.observed_at,
        "manifest_counts": list(snapshot.manifest_counts),
        "expected_chunks": list(DEFAULT_EXPECTED_CHUNKS),
        "ranks_complete": list(snapshot.ranks_complete),
        "workers": [
            {
                "tag": worker.tag,
                "pgid": worker.pgid,
                "rank": worker.rank,
                "gpu": worker.gpu,
                "subshard_count": worker.subshard_count,
                "subshard_index": worker.subshard_index,
            }
            for worker in snapshot.workers
        ],
        "gpu_compute_pids": {
            str(index): list(pids) for index, pids in snapshot.gpu_compute_pids.items()
        },
        "acc_complete": snapshot.acc_complete,
        "cutoff_reached": snapshot.cutoff_reached,
    }


def _public_plan(plan: Plan) -> dict[str, object]:
    return {
        "stage": plan.stage,
        "stop_tags": list(plan.stop_tags),
        "desired_workers": [asdict(worker) for worker in plan.desired_workers],
        "launch_workers": [asdict(worker) for worker in plan.launch_workers],
        "wait_reasons": list(plan.wait_reasons),
        "should_latch_cutoff": plan.should_latch_cutoff,
        "complete": plan.complete,
    }


def main() -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/supervise_rp67_t1_schedule.py"
    )
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    acc_marker = args.acc_complete_marker.expanduser().resolve()
    cutoff = _aware_datetime(args.cutoff)
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    schedule_sha, schedule = _schedule_identity(
        config_path=config_path,
        output_root=output_root,
        cutoff=cutoff,
        acc_marker=acc_marker,
    )

    if args.command == "status":
        snapshot = _snapshot(
            config_path=config_path,
            output_root=output_root,
            expected_chunks=DEFAULT_EXPECTED_CHUNKS,
            acc_marker=acc_marker,
            cutoff=cutoff,
            cutoff_latched=False,
        )
        plan = build_plan(snapshot, post_cutoff_count=None)
        print(
            json.dumps(
                {
                    "schedule_sha256": schedule_sha,
                    "snapshot": _public_snapshot(snapshot),
                    "plan": _public_plan(plan),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    supervisor_root = output_root / "runtime" / SUPERVISOR_RUNTIME_NAME
    if supervisor_root.exists() and supervisor_root.is_symlink():
        raise SupervisorBlockedError("supervisor runtime root must not be a symlink")
    gate_root = supervisor_root / "launch-gate"
    run_identity = make_run_identity(
        run_id=EXPECTED_T1_RUN_ID,
        phase="t1-worker-schedule",
        command_id="tools/supervise_rp67_t1_schedule.py",
        parameters={
            "schedule_sha256": schedule_sha,
            "output_root": str(output_root),
            "expected_run_manifest_sha256": EXPECTED_T1_RUN_MANIFEST_SHA256,
        },
    )
    try:
        materialize_ready_receipt(
            gate_root,
            run_identity=run_identity,
            evidence_paths={"selection_config": config_path},
        )
        if args.authorization_token is None:
            raise LaunchGateError(
                f"schedule is ready but launch is denied by default; authorize {gate_root}"
            )
        consume_launch_authorization(
            gate_root,
            args.authorization_token,
            EXECUTION_POLICY,
            expected_run_id=EXPECTED_T1_RUN_ID,
            expected_phase="t1-worker-schedule",
            freeze_override_path=args.freeze_override,
        )
    except LaunchGateError as error:
        raise SupervisorBlockedError(str(error)) from error

    supervisor_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (supervisor_root / "supervisor.lock").open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SupervisorBlockedError(
            "another supervisor instance owns the schedule"
        ) from error
    state_path = supervisor_root / "state.json"
    journal_path = supervisor_root / "events.jsonl"
    heartbeat_path = supervisor_root / "heartbeat.json"
    state = _load_or_initialize_state(
        state_path, schedule_sha256=schedule_sha, schedule=schedule
    )
    _atomic_json(state_path, state)
    _append_jsonl(
        journal_path,
        {
            "event": "supervisor_started",
            "at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "schedule_sha256": schedule_sha,
        },
    )

    while True:
        snapshot = _snapshot(
            config_path=config_path,
            output_root=output_root,
            expected_chunks=DEFAULT_EXPECTED_CHUNKS,
            acc_marker=acc_marker,
            cutoff=cutoff,
            cutoff_latched=bool(state["cutoff_latched"]),
        )
        changed = _adopt_workers(state, snapshot.workers, journal_path)
        if snapshot.cutoff_reached and not state["cutoff_latched"]:
            live_rank2_counts = {
                worker.subshard_count for worker in snapshot.workers if worker.rank == 2
            }
            state["cutoff_latched"] = True
            state["post_cutoff_count"] = 6 if 6 in live_rank2_counts else 2
            _append_jsonl(
                journal_path,
                {
                    "event": "gpu47_cutoff_latched",
                    "at": datetime.now(timezone.utc).isoformat(),
                    "post_cutoff_count": state["post_cutoff_count"],
                },
            )
            changed = True
        plan = build_plan(snapshot, post_cutoff_count=state["post_cutoff_count"])
        if state.get("last_stage") != plan.stage:
            state["last_stage"] = plan.stage
            _append_jsonl(
                journal_path,
                {
                    "event": "stage_changed",
                    "at": datetime.now(timezone.utc).isoformat(),
                    "stage": plan.stage,
                },
            )
            changed = True
        if state["cutoff_latched"] and state.get("gpu47_cleared_at") is None:
            gpu47_pids = {
                str(gpu): list(snapshot.gpu_compute_pids.get(gpu, ()))
                for gpu in range(4, 8)
                if snapshot.gpu_compute_pids.get(gpu, ())
            }
            if not gpu47_pids and not any(
                worker.gpu >= 4 for worker in snapshot.workers
            ):
                state["gpu47_cleared_at"] = datetime.now(timezone.utc).isoformat()
                _append_jsonl(
                    journal_path,
                    {
                        "event": "gpu47_clear_verified",
                        "at": state["gpu47_cleared_at"],
                    },
                )
                changed = True
        if changed:
            _atomic_json(state_path, state)
        _atomic_json(
            heartbeat_path,
            {
                "schema_version": HEARTBEAT_SCHEMA,
                "pid": os.getpid(),
                "schedule_sha256": schedule_sha,
                "snapshot": _public_snapshot(snapshot),
                "plan": _public_plan(plan),
            },
        )

        if plan.complete:
            status_command = [
                str(REPOSITORY_ROOT / ".venv312/bin/python"),
                str(REPOSITORY_ROOT / "tools/run_policy_data_selection_t1.py"),
                "status",
                "--config",
                str(config_path),
            ]
            completed = subprocess.run(
                status_command,
                cwd=REPOSITORY_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=1800,
            )
            if completed.returncode != 0:
                raise SupervisorBlockedError(
                    "final T1 status audit failed: "
                    + (completed.stderr.strip() or completed.stdout.strip())
                )
            status_value = json.loads(completed.stdout)
            observed_counts = tuple(
                int(row["all_revision_complete_manifests"])
                for row in status_value["ranks"]
            )
            if observed_counts != DEFAULT_EXPECTED_CHUNKS:
                raise SupervisorBlockedError("final T1 status counts mismatch")
            complete_record = {
                "schema_version": COMPLETE_SCHEMA,
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "schedule_sha256": schedule_sha,
                "run_id": EXPECTED_T1_RUN_ID,
                "run_manifest_sha256": EXPECTED_T1_RUN_MANIFEST_SHA256,
                "manifest_counts": list(observed_counts),
                "validated_t1_status": status_value,
            }
            complete_path = supervisor_root / "complete.json"
            _atomic_json(complete_path, complete_record)
            state["completed"] = True
            _atomic_json(state_path, state)
            _append_jsonl(
                journal_path,
                {
                    "event": "supervisor_complete",
                    "at": datetime.now(timezone.utc).isoformat(),
                    "complete_path": str(complete_path),
                },
            )
            return 0

        if plan.stop_tags:
            by_tag = {worker.tag: worker for worker in snapshot.workers}
            selected = tuple(by_tag[tag] for tag in plan.stop_tags)
            _stop_workers(selected, reason=plan.stage, journal_path=journal_path)
        elif plan.launch_workers:
            for worker in plan.launch_workers:
                _launch_worker(
                    worker,
                    config_path=config_path,
                    output_root=output_root,
                    journal_path=journal_path,
                )
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
