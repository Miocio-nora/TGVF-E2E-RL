from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_HANDOFF = _ROOT / "tools/handoff_prl26_a_notool_to_b_crop_train512_s32.sh"
_VALIDATOR = _ROOT / "tools/validate_prl26_train512_training_handoff.py"


def _validator_module() -> object:
    spec = importlib.util.spec_from_file_location("prl26_handoff_validator", _VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric(step: int) -> dict[str, object]:
    return {
        "schema_version": "policy-pilot-v1-metrics-event-v1",
        "optimizer_step": step,
        "step": {"reward": 0.5},
        "cumulative": {"optimizer_steps": step, "reward": 0.5},
        "timing": {"end_to_end_step_seconds": 1.0},
    }


def _attempt_events(
    log_path: Path,
    *,
    attempt: int,
    before: int,
    after: int,
    return_code: int,
    decision: str,
) -> list[dict[str, object]]:
    log_path.write_text("trainer output\n", encoding="utf-8")
    return [
        {
            "event": "attempt_started",
            "attempt": attempt,
            "checkpoint_step": before,
            "log_path": str(log_path),
        },
        {
            "event": "attempt_finished",
            "attempt": attempt,
            "checkpoint_step_before": before,
            "checkpoint_step_after": after,
            "return_code": return_code,
            "decision": decision,
            "log_path": str(log_path),
        },
    ]


def test_handoff_is_shell_valid_and_keeps_crop_outputs_external() -> None:
    subprocess.run(["bash", "-n", str(_HANDOFF)], check=True)
    source = _HANDOFF.read_text(encoding="utf-8")

    assert "expected_training_head=40f1728a69e0a3f868117776c80c45ad6de70b8c" in source
    assert "status --porcelain=v1 --untracked-files=all" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "crop-handoff" in source and "artifacts/control" in source
    assert 'mkdir -p "$control_root" "$crop_log_root"' in source
    assert 'mkdir -p "$crop_root"' not in source
    assert "crop-fresh-root-authorization.json" in source
    assert 'target-ready' in source
    assert 'trainable_tgvf_supervisor' in source
    assert '--run-config "$crop_config"' in source
    assert '--target-step 32' in source


def test_handoff_requires_exact_source_exit_and_integrity_before_resources() -> None:
    source = _HANDOFF.read_text(encoding="utf-8")

    assert "remain-on-exit" in source
    assert "#{pane_dead}" in source
    assert "#{pane_dead_status}" in source
    assert '[[ "$source_status" != 0 ]]' in source
    assert "source-complete" in source
    assert "resources-free" in source
    assert source.index("source-complete") < source.index("resources-free")
    assert source.index("resources-free") < source.index("target-ready")
    assert source.index("target-ready") < source.rindex("trainable_tgvf_supervisor")
    assert "release_stable_polls" in source
    assert "release_maximum_polls" in source


def test_handoff_pins_judge_retries_and_unique_offline_wandb_identity() -> None:
    source = _HANDOFF.read_text(encoding="utf-8")

    expected = {
        "TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP": "8",
        "TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS": "8",
        "TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS": "2",
        "TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS": "30",
        "TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION": "0",
        "WANDB_RUN_ID": "prl26btrain512crops32pixel512",
        "WANDB_RESUME": "allow",
        "WANDB_MODE": "offline",
    }
    for name, value in expected.items():
        assert f"export {name}={value}" in source
    assert "unset RAY_ADDRESS" in source


def test_metrics_validator_requires_exact_contiguous_finite_s32() -> None:
    module = _validator_module()
    rows = [_metric(step) for step in range(1, 33)]
    module._validate_metrics_rows(rows, expected_step=32, owner="test metrics")

    missing = rows[:-1]
    with pytest.raises(RuntimeError, match="exactly 32"):
        module._validate_metrics_rows(
            missing, expected_step=32, owner="test metrics"
        )

    nonfinite = json.loads(json.dumps(rows))
    nonfinite[-1]["timing"]["end_to_end_step_seconds"] = math.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        module._validate_metrics_rows(
            nonfinite, expected_step=32, owner="test metrics"
        )


def test_supervisor_event_validator_rejects_nonzero_terminal_attempt(
    tmp_path: Path,
) -> None:
    module = _validator_module()
    rows = _attempt_events(
        tmp_path / "attempt.log",
        attempt=1,
        before=0,
        after=32,
        return_code=0,
        decision="complete",
    )
    audit = module._validate_supervisor_events(
        rows, event_directory=tmp_path, target_step=32
    )
    assert audit["final_return_code"] == 0

    rows[-1]["return_code"] = 3
    with pytest.raises(RuntimeError, match="completion boundary is malformed"):
        module._validate_supervisor_events(
            rows, event_directory=tmp_path, target_step=32
        )

    failed = _attempt_events(
        tmp_path / "failed.log",
        attempt=1,
        before=0,
        after=26,
        return_code=1,
        decision="fail",
    )
    with pytest.raises(RuntimeError, match="did not finish S32"):
        module._validate_supervisor_events(
            failed, event_directory=tmp_path, target_step=32
        )


def test_supervisor_event_validator_accepts_recovered_independent_invocation(
    tmp_path: Path,
) -> None:
    module = _validator_module()
    rows = _attempt_events(
        tmp_path / "attempt-01-from-step-0.log",
        attempt=1,
        before=0,
        after=26,
        return_code=1,
        decision="fail",
    )
    rows += _attempt_events(
        tmp_path / "attempt-01-from-step-26.log",
        attempt=1,
        before=26,
        after=32,
        return_code=0,
        decision="complete",
    )

    audit = module._validate_supervisor_events(
        rows, event_directory=tmp_path, target_step=32
    )

    assert audit == {
        "invocation_count": 2,
        "attempts": 2,
        "invocations": [
            {
                "invocation": 1,
                "event_record_start": 1,
                "event_record_end": 2,
                "checkpoint_step_before": 0,
                "checkpoint_step_after": 26,
                "attempts": 1,
                "terminal_return_code": 1,
                "terminal_decision": "fail",
            },
            {
                "invocation": 2,
                "event_record_start": 3,
                "event_record_end": 4,
                "checkpoint_step_before": 26,
                "checkpoint_step_after": 32,
                "attempts": 1,
                "terminal_return_code": 0,
                "terminal_decision": "complete",
            },
        ],
        "final_return_code": 0,
        "final_decision": "complete",
        "final_checkpoint_step": 32,
    }


def test_supervisor_event_validator_distinguishes_retry_from_new_invocation(
    tmp_path: Path,
) -> None:
    module = _validator_module()
    rows = _attempt_events(
        tmp_path / "attempt-01.log",
        attempt=1,
        before=0,
        after=8,
        return_code=1,
        decision="retry_weight_wake_oom",
    )
    rows += _attempt_events(
        tmp_path / "attempt-02.log",
        attempt=2,
        before=8,
        after=32,
        return_code=0,
        decision="complete",
    )

    audit = module._validate_supervisor_events(
        rows, event_directory=tmp_path, target_step=32
    )

    assert audit["invocation_count"] == 1
    assert audit["attempts"] == 2
    assert audit["invocations"][0]["terminal_decision"] == "complete"


def test_supervisor_event_validator_rejects_attempt_after_terminal_fail(
    tmp_path: Path,
) -> None:
    module = _validator_module()
    rows = _attempt_events(
        tmp_path / "attempt-01.log",
        attempt=1,
        before=0,
        after=8,
        return_code=1,
        decision="fail",
    )
    rows += _attempt_events(
        tmp_path / "attempt-02.log",
        attempt=2,
        before=8,
        after=32,
        return_code=0,
        decision="complete",
    )

    with pytest.raises(RuntimeError, match="after a terminal decision"):
        module._validate_supervisor_events(
            rows, event_directory=tmp_path, target_step=32
        )


def test_supervisor_event_validator_rejects_reset_after_retry(tmp_path: Path) -> None:
    module = _validator_module()
    rows = _attempt_events(
        tmp_path / "first.log",
        attempt=1,
        before=0,
        after=8,
        return_code=1,
        decision="retry_weight_wake_oom",
    )
    rows += _attempt_events(
        tmp_path / "reset.log",
        attempt=1,
        before=8,
        after=32,
        return_code=0,
        decision="complete",
    )

    with pytest.raises(RuntimeError, match="without a prior terminal failure"):
        module._validate_supervisor_events(
            rows, event_directory=tmp_path, target_step=32
        )


@pytest.mark.parametrize("next_step", [25, 27])
def test_supervisor_event_validator_rejects_recovery_overlap_or_gap(
    tmp_path: Path, next_step: int
) -> None:
    module = _validator_module()
    rows = _attempt_events(
        tmp_path / "first.log",
        attempt=1,
        before=0,
        after=26,
        return_code=1,
        decision="fail",
    )
    rows += _attempt_events(
        tmp_path / "second.log",
        attempt=1,
        before=next_step,
        after=32,
        return_code=0,
        decision="complete",
    )

    with pytest.raises(RuntimeError, match="gap or overlap"):
        module._validate_supervisor_events(
            rows, event_directory=tmp_path, target_step=32
        )


def test_supervisor_event_validator_rejects_run_after_success(tmp_path: Path) -> None:
    module = _validator_module()
    rows = _attempt_events(
        tmp_path / "successful.log",
        attempt=1,
        before=0,
        after=32,
        return_code=0,
        decision="complete",
    )
    rows += _attempt_events(
        tmp_path / "unexpected.log",
        attempt=1,
        before=32,
        after=32,
        return_code=0,
        decision="complete",
    )

    with pytest.raises(RuntimeError, match="successful invocation"):
        module._validate_supervisor_events(
            rows, event_directory=tmp_path, target_step=32
        )


def test_supervisor_event_validator_rejects_unknown_decision(tmp_path: Path) -> None:
    module = _validator_module()
    rows = _attempt_events(
        tmp_path / "attempt.log",
        attempt=1,
        before=0,
        after=26,
        return_code=1,
        decision="manual_retry",
    )

    with pytest.raises(RuntimeError, match="decision is malformed"):
        module._validate_supervisor_events(
            rows, event_directory=tmp_path, target_step=32
        )


def test_supervisor_event_validator_rejects_unfinished_final_invocation(
    tmp_path: Path,
) -> None:
    module = _validator_module()
    log_path = tmp_path / "attempt.log"
    log_path.write_text("trainer output\n", encoding="utf-8")
    rows = [
        {
            "event": "attempt_started",
            "attempt": 1,
            "checkpoint_step": 0,
            "log_path": str(log_path),
        }
    ]

    with pytest.raises(RuntimeError, match="unfinished attempt"):
        module._validate_supervisor_events(
            rows, event_directory=tmp_path, target_step=32
        )


def test_crop_first_launch_authorizes_absent_root_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _validator_module()
    root = tmp_path / "crop-output"
    config = SimpleNamespace(
        run_id="crop-run",
        identity_sha256="1" * 64,
        source_sha256="2" * 64,
        output=SimpleNamespace(
            root=root,
            checkpoint_directory=root / "checkpoints",
            metrics_path=root / "metrics.jsonl",
        ),
        training=SimpleNamespace(maximum_optimizer_steps=32),
    )
    monkeypatch.setattr(module, "_load_config", lambda _: config)
    authorization = tmp_path / "control" / "authorization.json"

    result = module.target_launch_ready(
        config_path=tmp_path / "config.toml",
        authorization_path=authorization,
        training_head="a" * 40,
    )

    assert result["launch_mode"] == "fresh"
    assert authorization.is_file()
    assert not root.exists()


def test_crop_existing_root_requires_authorization_and_exact_tracker_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _validator_module()
    root = tmp_path / "crop-output"
    config = SimpleNamespace(
        run_id="crop-run",
        identity_sha256="1" * 64,
        source_sha256="2" * 64,
        output=SimpleNamespace(
            root=root,
            checkpoint_directory=root / "checkpoints",
            metrics_path=root / "metrics.jsonl",
        ),
        training=SimpleNamespace(maximum_optimizer_steps=32),
    )
    monkeypatch.setattr(module, "_load_config", lambda _: config)
    authorization = tmp_path / "control" / "authorization.json"
    root.mkdir()
    with pytest.raises(RuntimeError, match="without this handoff"):
        module.target_launch_ready(
            config_path=tmp_path / "config.toml",
            authorization_path=authorization,
            training_head="a" * 40,
        )

    root.rmdir()
    module.target_launch_ready(
        config_path=tmp_path / "config.toml",
        authorization_path=authorization,
        training_head="a" * 40,
    )
    (root / "checkpoints/global_step_2").mkdir(parents=True)
    (root / "checkpoints/latest_checkpointed_iteration.txt").write_text(
        "2\n", encoding="utf-8"
    )
    (root / "metrics.jsonl").write_text(
        "\n".join(json.dumps(_metric(step)) for step in (1, 2)) + "\n",
        encoding="utf-8",
    )
    validated: list[tuple[Path, int]] = []

    def accept_generation(
        _config: object, generation: Path, *, optimizer_step: int
    ) -> tuple[object, object, list[Path]]:
        validated.append((generation, optimizer_step))
        return object(), object(), []

    monkeypatch.setattr(module, "_validate_generation", accept_generation)
    result = module.target_launch_ready(
        config_path=tmp_path / "config.toml",
        authorization_path=authorization,
        training_head="a" * 40,
    )
    assert result["launch_mode"] == "resume"
    assert result["checkpointed_step"] == 2
    assert validated == [(root / "checkpoints/global_step_2", 2)]
