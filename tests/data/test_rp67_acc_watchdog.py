from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "watchdog_rp67_step2000_acc_pipeline.py"
)
_SPEC = importlib.util.spec_from_file_location("rp67_acc_watchdog", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
watchdog = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = watchdog
_SPEC.loader.exec_module(watchdog)


def test_watchdog_observes_only_the_v2_completion_marker() -> None:
    assert watchdog.COMPLETE_MARKER.name == (
        "rp67_step2000_all_validations_complete_v2.json"
    )


def test_controller_argv_requires_exact_executable_script_and_run_mode() -> None:
    cwd = watchdog.REPOSITORY_ROOT
    valid = (
        ".venv312/bin/python",
        "-u",
        "tools/run_rp67_step2000_acc_pipeline.py",
        "--poll-seconds",
        "15",
        "run",
        "--execute",
    )
    assert watchdog._controller_argv_is_valid(valid, cwd=cwd)
    assert not watchdog._controller_argv_is_valid(
        (*valid[:-2], "status", "--execute"), cwd=cwd
    )
    assert not watchdog._controller_argv_is_valid(
        ("/usr/bin/python3.12", *valid[1:]), cwd=cwd
    )
    assert not watchdog._controller_argv_is_valid(
        (*valid[:4], "61", *valid[5:]), cwd=cwd
    )


def test_restart_gate_requires_identity_gpu_lock_and_endpoint_clear() -> None:
    safe = watchdog.RestartGate(
        controller_pids=(),
        gpu_compute_pids={0: (), 1: ()},
        pipeline_lock_available=True,
        judge_endpoint_open=False,
    )
    assert safe.safe
    assert safe.wait_reasons == ()

    blocked = watchdog.RestartGate(
        controller_pids=(123,),
        gpu_compute_pids={0: (456,), 1: ()},
        pipeline_lock_available=False,
        judge_endpoint_open=True,
    )
    assert not blocked.safe
    assert len(blocked.wait_reasons) == 4


def test_failure_budget_is_bounded_and_backoff_is_exponential() -> None:
    state = watchdog._new_state(
        identity=watchdog.ControllerIdentity(1, 2, "boot", 3, "a" * 64),
        progress_token="b" * 64,
    )
    failure = watchdog._record_controller_failure(
        state, now_epoch=100, base_backoff=30, maximum_backoff=120
    )
    assert failure == 1
    assert state["next_restart_not_before_epoch"] == 130
    with pytest.raises(watchdog.WatchdogBlockedError, match="twice"):
        watchdog._record_controller_failure(
            state, now_epoch=100, base_backoff=30, maximum_backoff=120
        )

    assert watchdog._backoff_seconds(1, base=30, maximum=120) == 30
    assert watchdog._backoff_seconds(2, base=30, maximum=120) == 60
    assert watchdog._backoff_seconds(3, base=30, maximum=120) == 120
    assert watchdog._backoff_seconds(4, base=30, maximum=120) == 120


def test_progress_token_only_changes_for_durable_stage_completion(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"event": "pipeline_started", "pid": 1}) + "\n")
    baseline = watchdog._progress_token(events)
    with events.open("a") as handle:
        handle.write(json.dumps({"event": "judge_started", "pgid": 2}) + "\n")
    assert watchdog._progress_token(events) == baseline
    with events.open("a") as handle:
        handle.write(
            json.dumps({"event": "semantic_complete", "output_root": "/result"})
            + "\n"
        )
    assert watchdog._progress_token(events) != baseline


def test_marker_validation_delegates_to_strict_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBlockedError(RuntimeError):
        pass

    monkeypatch.setattr(
        watchdog,
        "_controller_module",
        lambda: SimpleNamespace(
            PipelineBlockedError=FakeBlockedError,
            _existing_complete_marker_is_valid=lambda: True,
        ),
    )
    assert watchdog._marker_is_complete()

    def invalid_marker() -> bool:
        raise FakeBlockedError("artifact drifted")

    monkeypatch.setattr(
        watchdog,
        "_controller_module",
        lambda: SimpleNamespace(
            PipelineBlockedError=FakeBlockedError,
            _existing_complete_marker_is_valid=invalid_marker,
        ),
    )
    with pytest.raises(watchdog.WatchdogBlockedError, match="artifact drifted"):
        watchdog._marker_is_complete()


@pytest.mark.parametrize("state", ("Z", "X"))
def test_zombie_or_dead_controller_is_not_live(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    identity = watchdog.ControllerIdentity(123, 456, "boot", 789, "a" * 64)
    monkeypatch.setattr(
        watchdog, "_proc_state_and_starttime", lambda _pid: (state, 789)
    )

    def should_not_inspect(_pid: int) -> object:
        raise AssertionError("zombie identity must not reach live argv inspection")

    monkeypatch.setattr(watchdog, "_inspect_controller", should_not_inspect)
    assert watchdog._identity_is_live(identity) is False


def test_real_live_identity_drift_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = watchdog.ControllerIdentity(123, 456, "boot", 789, "a" * 64)
    drifted = watchdog.ControllerIdentity(123, 456, "boot", 789, "b" * 64)
    monkeypatch.setattr(
        watchdog, "_proc_state_and_starttime", lambda _pid: ("S", 789)
    )
    monkeypatch.setattr(watchdog, "_inspect_controller", lambda _pid: drifted)
    with pytest.raises(watchdog.WatchdogBlockedError, match="identity drifted"):
        watchdog._identity_is_live(identity)
