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
    attempt_log = tmp_path / "attempt.log"
    attempt_log.write_text("trainer output\n", encoding="utf-8")
    started = {
        "event": "attempt_started",
        "attempt": 1,
        "checkpoint_step": 0,
        "log_path": str(attempt_log),
    }
    finished = {
        "event": "attempt_finished",
        "attempt": 1,
        "checkpoint_step_before": 0,
        "checkpoint_step_after": 32,
        "return_code": 0,
        "decision": "complete",
        "log_path": str(attempt_log),
    }
    audit = module._validate_supervisor_events(
        [started, finished], event_directory=tmp_path, target_step=32
    )
    assert audit["final_return_code"] == 0

    failed = dict(finished, return_code=3)
    with pytest.raises(RuntimeError, match="return code zero"):
        module._validate_supervisor_events(
            [started, failed], event_directory=tmp_path, target_step=32
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
