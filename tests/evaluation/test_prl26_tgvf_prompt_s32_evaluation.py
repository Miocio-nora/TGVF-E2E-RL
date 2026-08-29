from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
PLAN = ROOT / (
    "configs/evaluation/"
    "prl26_cd_tgvf_target_prompt_pair_s32_pixel512_coredev2511_plan.json"
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def paired_runner() -> ModuleType:
    return _load_module(
        "prl26_target_prompt_pair_runner_under_test",
        TOOLS / "run_prl15_paired_evaluation.py",
    )


@pytest.fixture(scope="module")
def summarizer() -> ModuleType:
    return _load_module(
        "prl26_target_prompt_pair_summarizer_under_test",
        TOOLS / "summarize_prl26_tgvf_prompt_s32_evaluation.py",
    )


def test_target_prompt_pair_plan_passes_strict_static_gate(
    paired_runner: ModuleType,
) -> None:
    plan = paired_runner._load_plan(PLAN)

    assert plan["schema_version"] == (
        "tgvf.target-prompt-paired-policy-benchmark-plan.v6"
    )
    assert plan["evaluation_image_max_pixels"] == 262_144
    assert [arm["name"] for arm in plan["arms"]] == ["short", "full"]
    assert plan["paired_rng"]["protocol_projection"] == {
        "kind": "target_prompt_pair_v1",
        "excluded_protocol_field": "prompt_sha256",
        "axis_values": [
            "e74bb5e1253af107ff27badfcfaca747b94574e19677d22cfe42b0b1c0ba5633",
            "77ed3a597d2a58e748b70bafe37882760944e293723a28008818a96aad025d0d",
        ],
    }


def test_evaluation_supervisor_preserves_phase_and_resource_order() -> None:
    source = (
        TOOLS / "supervise_prl26_tgvf_prompt_s32_evaluation.sh"
    ).read_text(encoding="utf-8")

    receipt_gate = 'for receipt in "$short_receipt" "$full_receipt"'
    create_state = 'mkdir -p "$runtime_root" "$log_root"'
    prepare = "phase=preparing_short_full_snapshots"
    infer = "phase=running_short_full_parallel_four_plus_four_inference"
    score = "phase=scoring_short_full_seven_subsets"
    publish = "phase=publishing_headline_and_tool_usage_table"
    assert source.index(receipt_gate) < source.index(create_state)
    assert source.index(prepare) < source.index(infer) < source.index(score)
    assert source.index(score) < source.index(publish)
    assert "--gpu-ids 0 1 2 3 4 5 6 7" in source
    assert "--gpu-ids 0 1 2 3" in source
    assert '--arm "$arm"' in source


def test_tmux_launcher_passes_credentials_without_command_text() -> None:
    source = (
        TOOLS / "launch_prl26_tgvf_prompt_train_and_eval_tmux.sh"
    ).read_text(encoding="utf-8")

    assert 'tmux set-environment -g OPENROUTER_API_KEY "$OPENROUTER_API_KEY"' in source
    assert "OPENROUTER_API_KEY='" not in source
    assert "supervise_prl26_tgvf_prompt_train_and_eval.sh" in source
    assert "remain-on-exit on" in source
    assert "status --porcelain=v1 --untracked-files=all" in source


def test_tool_usage_summary_reports_calls_frequency_and_length(
    summarizer: ModuleType,
) -> None:
    rows = [
        {
            "tool_calls": [
                {"function_name": "tgvf_focus_tool"},
                {"function_name": "tgvf_focus_tool"},
            ],
            "assistant_turns": [
                {"sampled_token_count": 10},
                {"sampled_token_count": 20},
            ],
            "tool_errors": [{"code": "probe"}],
            "successful_observation_count": 2,
            "stop": "final_answer",
        },
        {
            "tool_calls": [],
            "assistant_turns": [{"sampled_token_count": 5}],
            "tool_errors": [],
            "successful_observation_count": 0,
            "stop": "direct_answer",
        },
    ]

    usage = summarizer._usage(rows)

    assert usage["trajectory_count"] == 2
    assert usage["trajectories_using_tool"] == 1
    assert usage["tool_use_rate"] == 0.5
    assert usage["total_tool_calls"] == 2
    assert usage["mean_tool_calls_per_trajectory"] == 1.0
    assert usage["mean_tool_calls_when_used"] == 2.0
    assert usage["successful_observation_count"] == 2
    assert usage["tool_error_count"] == 1
    assert usage["generated_token_mean"] == 17.5
    assert usage["function_call_counts"] == {"tgvf_focus_tool": 2}
