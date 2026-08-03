from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "run_rp69_post_training_evaluations.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "rp69_post_training_evaluations", _TOOL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
evaluation = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluation
_SPEC.loader.exec_module(evaluation)


def _layout(training_root: Path) -> dict[str, Path]:
    return {
        "adapter_path": training_root / "adapter.pt",
        "metrics_path": training_root / "metrics.jsonl",
        "checkpoint_directory": training_root / "checkpoints",
    }


def test_default_output_root_is_the_isolated_visual_barycentric_tree() -> None:
    expected = evaluation.REPOSITORY_ROOT / (
        "artifacts/representation_experiments/visual_barycentric/evaluation/"
        "rp69_step0500_gpu0123"
    )

    assert evaluation._default_output_root() == expected
    evaluation._assert_output_root_isolated(
        expected,
        _layout(evaluation.EXPECTED_ARTIFACT_ROOT),
    )


@pytest.mark.parametrize(
    "relative_output",
    (
        ".",
        "post-training-evaluation",
        "checkpoints",
        "checkpoints/evaluation",
    ),
)
def test_output_isolation_rejects_training_tree_and_descendants(
    tmp_path: Path, relative_output: str
) -> None:
    training_root = tmp_path / "artifacts" / "representation" / "rp69"
    output = training_root / relative_output

    with pytest.raises(
        evaluation.ControllerError,
        match="evaluation output root overlaps RP69 training/artifact paths",
    ):
        evaluation._assert_output_root_isolated(output, _layout(training_root))


def test_output_isolation_rejects_parent_of_training_tree(tmp_path: Path) -> None:
    training_root = tmp_path / "artifacts" / "representation" / "rp69"

    with pytest.raises(evaluation.ControllerError, match="overlaps"):
        evaluation._assert_output_root_isolated(
            training_root.parent,
            _layout(training_root),
        )


def test_run_rejects_overlap_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training_root = tmp_path / "training"
    output = training_root / "post-training-evaluation"
    monkeypatch.setattr(
        evaluation,
        "_static_training_layout",
        lambda _path: _layout(training_root),
    )

    with pytest.raises(evaluation.ControllerError, match="overlaps"):
        evaluation._run(
            SimpleNamespace(),
            tmp_path / "training.toml",
            evaluation._paths(output),
        )

    assert not output.exists()


def test_dry_run_reports_passed_output_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training_config = tmp_path / "training.toml"
    training_config.write_text("run_id = 'rp69'\n", encoding="utf-8")
    training_root = tmp_path / "training"
    output = tmp_path / "evaluation" / "rp69"
    monkeypatch.setattr(
        evaluation,
        "_assert_static_inputs",
        lambda _path: {
            **_layout(training_root),
            "target_optimizer_steps": evaluation.EXPECTED_STEP,
        },
    )

    result = evaluation._dry_run(
        training_config_path=training_config,
        output_root=output,
        int_gpu=0,
        first_gpus=(1,),
        full_gpus=(2, 3),
        judge_gpus=(2, 3),
    )

    assert result["status"] == "dry_run_complete"
    assert result["files_written"] is False
    assert result["gpu_processes_started"] is False
    assert result["output_isolation_preflight"] == {
        "status": "passed",
        "protected_paths": [
            str(training_root.resolve()),
            str((training_root / "checkpoints").resolve()),
        ],
    }
