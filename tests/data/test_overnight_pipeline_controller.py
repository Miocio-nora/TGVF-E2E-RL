from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


_TOOL_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "run_overnight_pipeline.py"
)
_SPEC = importlib.util.spec_from_file_location("overnight_pipeline", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
pipeline = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pipeline
_SPEC.loader.exec_module(pipeline)


_WORKER = r'''from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time

stage, mode, artifact, trace, counter = sys.argv[1:]
artifact_path = Path(artifact)
trace_path = Path(trace)
counter_path = Path(counter)
trace_path.parent.mkdir(parents=True, exist_ok=True)
with trace_path.open("a", encoding="utf-8") as handle:
    handle.write(stage + "\n")

if mode == "fail":
    raise SystemExit(7)
if mode == "flaky":
    observed = int(counter_path.read_text()) if counter_path.exists() else 0
    counter_path.write_text(str(observed + 1))
    if observed == 0:
        raise SystemExit(9)
    mode = "file"
if mode == "spawn_sleep":
    child = subprocess.Popen([
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
    ])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(str(child.pid))
    time.sleep(60)
if mode == "file":
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(stage + "\n")
elif mode == "json_complete":
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"status": "complete", "resume": {"proven": True}}))
elif mode == "json_wrong":
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"status": "wrong"}))
elif mode == "jsonl_2000":
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"event": "complete", "global_step": 2000}) + "\n")
elif mode == "jsonl_1":
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"global_step": 1}) + "\n")
elif mode == "jsonl_80":
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"global_step": 40}) + "\n" + json.dumps({"global_step": 80}) + "\n")
elif mode == "directory":
    artifact_path.mkdir(parents=True, exist_ok=True)
    (artifact_path / "complete.json").write_text("{}\n")
'''


def _write_worker(root: Path) -> Path:
    path = root / "worker.py"
    path.write_text(_WORKER)
    return path


def _stage_specs(root: Path, worker: Path) -> list[dict[str, object]]:
    artifacts = root / "artifacts"
    trace = root / "trace.txt"
    modes_and_predicates: list[tuple[str, str, dict[str, object]]] = [
        (
            "crop_rl_smoke_1step",
            "jsonl_1",
            {
                "type": "jsonl_last_step",
                "path": str(artifacts / "crop-smoke.jsonl"),
                "field": "global_step",
                "equals": 1,
            },
        ),
        (
            "crop_rl_auto_resume_proof",
            "json_complete",
            {
                "type": "json_field",
                "path": str(artifacts / "resume-proof.json"),
                "field": "resume.proven",
                "equals": True,
            },
        ),
        (
            "stage1_smoke",
            "file",
            {"type": "exists", "path": str(artifacts / "smoke.pt"), "kind": "file"},
        ),
        (
            "stage1_resume_2000",
            "jsonl_2000",
            {
                "type": "jsonl_last_step",
                "path": str(artifacts / "stage1-metrics.jsonl"),
                "field": "global_step",
                "equals": 2000,
            },
        ),
        (
            "int_diag",
            "json_complete",
            {
                "type": "json_field",
                "path": str(artifacts / "int-diag.json"),
                "field": "status",
                "equals": "complete",
            },
        ),
        (
            "acc_val",
            "directory",
            {
                "type": "exists",
                "path": str(artifacts / "acc-val"),
                "kind": "directory",
                "nonempty": True,
            },
        ),
        (
            "crop_rl_80step",
            "jsonl_80",
            {
                "type": "jsonl_last_step",
                "path": str(artifacts / "crop-80.jsonl"),
                "field": "global_step",
                "equals": 80,
            },
        ),
    ]
    stages: list[dict[str, object]] = []
    for stage_id, mode, predicate in modes_and_predicates:
        artifact = Path(str(predicate["path"]))
        stages.append(
            {
                "id": stage_id,
                "command": {
                    "argv": [
                        sys.executable,
                        str(worker),
                        stage_id,
                        mode,
                        str(artifact),
                        str(trace),
                        str(root / f"{stage_id}.counter"),
                    ],
                    "cwd": str(root),
                    "timeout_seconds": 5,
                    "terminate_grace_seconds": 0.2,
                },
                "acceptance": [predicate],
            }
        )
    return stages


def _write_config(root: Path, stages: list[dict[str, object]]) -> Path:
    path = root / "pipeline.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": pipeline.SCHEMA_VERSION,
                "pipeline_id": "cpu-test-pipeline",
                "runtime_root": str(root / "runtime"),
                "stages": stages,
            }
        )
        + "\n"
    )
    return path


def _loaded(root: Path) -> tuple[dict[str, object], Path, list[dict[str, object]]]:
    worker = _write_worker(root)
    stages = _stage_specs(root, worker)
    config_path = _write_config(root, stages)
    config, _ = pipeline.load_config(config_path)
    return config, config_path, stages


def test_success_records_all_predicate_types_and_repeat_only_revalidates(tmp_path: Path) -> None:
    config, config_path, _stages = _loaded(tmp_path)
    result = pipeline.PipelineRunner(config).run()

    assert result["status"] == "complete"
    assert (tmp_path / "trace.txt").read_text().splitlines() == list(
        pipeline.REQUIRED_STAGE_IDS
    )
    state_path = tmp_path / "runtime" / "state.json"
    durable = json.loads(state_path.read_text())
    assert durable["config_sha256"] == pipeline.load_config(config_path)[0]["config_sha256"]
    assert all(
        durable["stages"][stage_id]["status"] == "accepted"
        for stage_id in pipeline.REQUIRED_STAGE_IDS
    )
    assert all(
        durable["stages"][stage_id]["attempts"][0]["exit_code"] == 0
        for stage_id in pipeline.REQUIRED_STAGE_IDS
    )
    logs = sorted((tmp_path / "runtime" / "logs").glob("*.log"))
    assert len(logs) == len(pipeline.REQUIRED_STAGE_IDS)
    assert all(b'"controller_event":"command_start"' in path.read_bytes() for path in logs)
    assert all(b'"controller_event":"command_end"' in path.read_bytes() for path in logs)

    second = pipeline.PipelineRunner(config).run()
    assert second["status"] == "complete"
    assert (tmp_path / "trace.txt").read_text().splitlines() == list(
        pipeline.REQUIRED_STAGE_IDS
    )


def test_stage_order_is_fixed_so_validation_cannot_precede_training(tmp_path: Path) -> None:
    _config, _path, stages = _loaded(tmp_path)
    stages[1], stages[2] = stages[2], stages[1]
    bad_path = _write_config(tmp_path, stages)

    with pytest.raises(pipeline.ConfigError, match="exact fail-closed order"):
        pipeline.load_config(bad_path)


def test_no_retry_by_default_and_downstream_never_starts(tmp_path: Path) -> None:
    config, _path, stages = _loaded(tmp_path)
    stages[3]["command"]["argv"][3] = "fail"  # type: ignore[index]
    config_path = _write_config(tmp_path, stages)
    config, _ = pipeline.load_config(config_path)

    with pytest.raises(pipeline.StageFailed, match="1 attempt"):
        pipeline.PipelineRunner(config).run()

    assert (tmp_path / "trace.txt").read_text().splitlines() == [
        "crop_rl_smoke_1step",
        "crop_rl_auto_resume_proof",
        "stage1_smoke",
        "stage1_resume_2000",
    ]
    state = json.loads((tmp_path / "runtime" / "state.json").read_text())
    assert state["stages"]["stage1_resume_2000"]["status"] == "failed"
    assert len(state["stages"]["stage1_resume_2000"]["attempts"]) == 1
    assert state["stages"]["int_diag"]["status"] == "pending"


def test_retry_occurs_only_when_explicitly_bounded(tmp_path: Path) -> None:
    _config, _path, stages = _loaded(tmp_path)
    stages[2]["command"]["argv"][3] = "flaky"  # type: ignore[index]
    stages[2]["retry"] = {"max_retries": 1, "delay_seconds": 0}
    config_path = _write_config(tmp_path, stages)
    config, _ = pipeline.load_config(config_path)

    state = pipeline.PipelineRunner(config).run()

    assert state["status"] == "complete"
    attempts = state["stages"]["stage1_smoke"]["attempts"]
    assert [attempt["outcome"] for attempt in attempts] == ["nonzero_exit", "accepted"]


def test_zero_exit_with_bad_artifact_stops_before_downstream(tmp_path: Path) -> None:
    _config, _path, stages = _loaded(tmp_path)
    stages[4]["command"]["argv"][3] = "json_wrong"  # type: ignore[index]
    config_path = _write_config(tmp_path, stages)
    config, _ = pipeline.load_config(config_path)

    with pytest.raises(pipeline.StageFailed, match="int_diag"):
        pipeline.PipelineRunner(config).run()

    trace = (tmp_path / "trace.txt").read_text().splitlines()
    assert trace == [
        "crop_rl_smoke_1step",
        "crop_rl_auto_resume_proof",
        "stage1_smoke",
        "stage1_resume_2000",
        "int_diag",
    ]
    state = json.loads((tmp_path / "runtime" / "state.json").read_text())
    attempt = state["stages"]["int_diag"]["attempts"][0]
    assert attempt["exit_code"] == 0
    assert attempt["outcome"] == "acceptance_failed"
    assert state["stages"]["acc_val"]["status"] == "pending"


def test_corrupted_accepted_artifact_blocks_resume_instead_of_rerunning(tmp_path: Path) -> None:
    config, _path, _stages = _loaded(tmp_path)
    pipeline.PipelineRunner(config).run()
    trace_before = (tmp_path / "trace.txt").read_text()
    (tmp_path / "artifacts" / "stage1-metrics.jsonl").write_text(
        json.dumps({"global_step": 1999}) + "\n"
    )

    with pytest.raises(pipeline.PipelineBlockedError, match="no longer satisfies"):
        pipeline.PipelineRunner(config).run()

    assert (tmp_path / "trace.txt").read_text() == trace_before


def test_timeout_terminates_the_whole_new_process_group(tmp_path: Path) -> None:
    _config, _path, stages = _loaded(tmp_path)
    child_pid_path = tmp_path / "artifacts" / "child.pid"
    stages[0]["command"]["argv"][3] = "spawn_sleep"  # type: ignore[index]
    stages[0]["command"]["argv"][4] = str(child_pid_path)  # type: ignore[index]
    stages[0]["command"]["timeout_seconds"] = 0.3  # type: ignore[index]
    stages[0]["command"]["terminate_grace_seconds"] = 0.1  # type: ignore[index]
    config_path = _write_config(tmp_path, stages)
    config, _ = pipeline.load_config(config_path)

    with pytest.raises(pipeline.StageFailed, match="exceeded"):
        pipeline.PipelineRunner(config).run()

    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        stat_path = Path("/proc") / str(child_pid) / "stat"
        if not stat_path.exists():
            break
        fields = stat_path.read_text().split()
        if len(fields) > 2 and fields[2] == "Z":
            break
        time.sleep(0.05)
    else:
        pytest.fail("spawned child survived process-group timeout termination")
    state = json.loads((tmp_path / "runtime" / "state.json").read_text())
    attempt = state["stages"]["crop_rl_smoke_1step"]["attempts"][0]
    assert attempt["timed_out"] is True
    assert attempt["outcome"] == "timeout"


def test_live_record_is_not_started_twice(tmp_path: Path) -> None:
    config, _path, _stages = _loaded(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        identity = pipeline._proc_identity(process.pid)
        assert identity is not None
        state = pipeline._new_state(config)
        state["status"] = "running"
        state["current_stage"] = "crop_rl_smoke_1step"
        state["stages"]["crop_rl_smoke_1step"] = {
            "status": "running",
            "attempts": [{"process_identity": identity}],
        }
        pipeline._atomic_json(runtime / "state.json", state)

        with pytest.raises(pipeline.PipelineBlockedError, match="refusing duplicate"):
            pipeline.PipelineRunner(config).run()
        assert not (tmp_path / "trace.txt").exists()
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
