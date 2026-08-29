from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "tools/supervise_prl27_b_corrected_crop_s32_evaluation.sh"
SUMMARIZER = ROOT / "tools/summarize_prl27_b_corrected_crop_s32_evaluation.py"


def _source() -> str:
    return SUPERVISOR.read_text(encoding="utf-8")


def _load_summarizer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prl27_b_corrected_crop_summarizer_under_test", SUMMARIZER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prl27_b_evaluation_supervisor_is_valid_executable_bash() -> None:
    assert SUPERVISOR.is_file()
    assert SUPERVISOR.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(SUPERVISOR)], check=True)


def test_prl27_b_waiter_is_training_admission_s32_and_release_gated() -> None:
    source = _source()
    training_head = source.index('while [[ ! -s "$training_admitted_head" ]]')
    receipt_wait = source.index(
        'while [[ ! -f "$training_complete" || ! -s "$receipt" ]]'
    )
    release_gate = source.index("while (( quiet < release_stable_polls ))")
    bind = source.index('"$python_bin" "$binder"', release_gate)
    prepare = source.index("--mode prepare", bind)
    validate = source.index("--mode validate --world-size 4", prepare)
    proof = source.index("--arm crop", validate)
    infer = source.index("--mode infer", proof)
    score = source.index("--mode score", infer)
    summarize = source.index('"$python_bin" "$summarizer"', score)
    complete = source.index('touch "$supervisor_complete"', summarize)

    assert (
        training_head
        < receipt_wait
        < release_gate
        < bind
        < prepare
        < validate
        < proof
        < infer
        < score
        < summarize
        < complete
    )
    assert 'training_complete="$training_control_root/state/s32-accepted"' in source
    assert 'training_failed="$training_control_root/state/failed"' in source
    assert 'training_admitted_head="$training_control_root/admitted-head.txt"' in source
    assert "release_stable_polls < 2" in source
    assert "resources-free" in source
    assert "PRL27_B_EVAL_RELEASE_STABLE_POLLS:-3" in source


def test_prl27_b_waiter_rejects_failure_at_every_training_boundary() -> None:
    source = _source()
    training_admission = source.index('while [[ ! -s "$training_admitted_head" ]]')
    receipt_wait = source.index(
        'while [[ ! -f "$training_complete" || ! -s "$receipt" ]]'
    )
    post_wait = source.index(
        '|| -e "$training_failed" || -L "$training_failed"', receipt_wait
    )
    resource_wait = source.index("while (( quiet < release_stable_polls ))")
    resource_failure = source.index(
        'if [[ -e "$training_failed" || -L "$training_failed" ]]',
        resource_wait,
    )
    binder = source.index('"$python_bin" "$binder"', resource_failure)

    assert (
        source.index("training failed before admission", training_admission)
        < receipt_wait
    )
    assert receipt_wait < post_wait < resource_wait < resource_failure < binder
    assert source.count('[[ -e "$training_failed" || -L "$training_failed" ]]') >= 3


def test_prl27_b_waiter_binds_one_clean_head_without_root_redirection() -> None:
    source = _source()
    assert "admitted_head=${PRL27_B_EVAL_ADMITTED_HEAD:-}" in source
    assert 'admitted_head=$(git -C "$repo_root" rev-parse HEAD)' not in source
    assert "status --porcelain=v1 --untracked-files=all" in source
    assert '"$observed_head" != "$admitted_head"' in source
    assert "evaluation HEAD differs from training admission" in source
    assert source.count("validate_worktree") >= 6
    assert "PRL27_B_EVAL_ROOT" not in source
    assert "PRL27_B_TRAINING_ROOT" not in source
    assert "PRL27_A" not in source
    assert "PRL-27-A" not in source


def test_prl27_b_eval_identity_is_new_training_run_pixel512_single_arm() -> None:
    source = _source()
    assert (
        "PRL27-B-CROP-REPLAY-BYTE-PARITY-TRAIN512-S32-TRAINING-RUN-"
        "COREDEV2511-PIXEL512-V1"
    ) in source
    assert "PRL-27-B-train512-s32-crop-replay-byte-parity" in source
    assert "bind_prl27_b_corrected_crop_training_run_evaluation.py" in source
    assert "summarize_prl27_b_corrected_crop_s32_evaluation.py" in source
    assert source.count("--mode prepare") == 1
    assert source.count("--mode infer") == 1
    assert source.count("--mode score") == 1
    assert source.count("--gpu-ids 0 1 2 3") == 3


def test_prl27_b_summarizer_covers_seven_subsets_macro_tool_use_and_length() -> None:
    source = SUMMARIZER.read_text(encoding="utf-8")
    for dataset in (
        "VStarBench",
        "HRBench4K",
        "BLINK",
        "OCRBench_v2",
        "MMMU_Pro_10c",
        "MathVista_MINI",
        "MathVerse_MINI",
    ):
        assert dataset in source
    assert '"macro_star_percent"' in source
    assert '"seven_subset_statistics"' in source
    assert '"tool_usage_overall"' in source
    assert '"tool_usage_by_subset"' in source
    assert '"total_tool_attempts"' in source
    assert '"successful_tool_call_trajectory_rate"' in source
    assert '"generated_token_mean"' in source
    assert '"generated_token_p95"' in source
    assert '"length_unit"' in source
    assert "coredev2511/prl27-b/crop-replay-byte-parity/training-run/" in source


def test_prl27_b_usage_counts_attempts_calls_errors_and_all_turn_tokens() -> None:
    summarizer = _load_summarizer()
    rows = [
        {
            "tool_calls": [],
            "tool_errors": [],
            "assistant_turns": [{"turn_index": 0, "sampled_token_count": 11}],
            "successful_observation_count": 0,
            "stop": "eos",
        },
        {
            "tool_calls": [
                {"function_name": "image_zoom_in_tool"},
                {"function_name": "image_zoom_in_tool"},
            ],
            "tool_errors": [{"code": "invalid_arguments"}],
            "assistant_turns": [
                {"turn_index": 0, "sampled_token_count": 7},
                {"turn_index": 1, "sampled_token_count": 13},
                {"turn_index": 2, "sampled_token_count": 17},
            ],
            "successful_observation_count": 2,
            "stop": "final",
        },
    ]
    usage = summarizer._usage(rows)
    assert usage["trajectory_count"] == 2
    assert usage["no_tool_trajectory_count"] == 1
    assert usage["total_tool_attempts"] == 3
    assert usage["successful_tool_call_count"] == 2
    assert usage["tool_error_count"] == 1
    assert usage["trajectories_with_repeat_successful_tool_call"] == 1
    assert usage["tool_attempt_trajectory_rate"] == pytest.approx(0.5)
    assert usage["generated_token_mean"] == pytest.approx(24.0)
    assert usage["generated_token_p50"] == 11
    assert usage["generated_token_p95"] == 37
