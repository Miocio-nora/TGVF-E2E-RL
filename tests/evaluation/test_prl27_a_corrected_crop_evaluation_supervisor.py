from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "tools/supervise_prl27_a_corrected_crop_s32_evaluation.sh"


def _source() -> str:
    return SUPERVISOR.read_text(encoding="utf-8")


def test_prl27_a_evaluation_supervisor_is_valid_executable_bash() -> None:
    assert SUPERVISOR.is_file()
    assert SUPERVISOR.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(SUPERVISOR)], check=True)


def test_prl27_a_evaluation_chain_is_receipt_and_release_gated() -> None:
    source = _source()
    receipt_wait = source.index(
        'while [[ ! -f "$training_complete" || ! -s "$receipt" ]]'
    )
    release_gate = source.index("while (( quiet < release_stable_polls ))")
    create_evaluation_root = source.index('mkdir -p "$eval_root/logs"')
    bind = source.index('"$python_bin" "$binder"', create_evaluation_root)
    prepare = source.index('--mode prepare')
    static_validate = source.index('--mode validate --world-size 4')
    processor_proof = source.index('--arm crop')
    infer = source.index('--mode infer')
    score = source.index('--mode score')
    aggregate = source.index(
        "phase=publishing_corrected_crop_result_and_tool_usage"
    )
    complete = source.index('touch "$supervisor_complete"')

    assert (
        receipt_wait
        < release_gate
        < create_evaluation_root
        < bind
        < prepare
        < static_validate
        < processor_proof
        < infer
        < score
        < aggregate
        < complete
    )
    assert "release_stable_polls < 2" in source
    assert "resources-free" in source
    assert 'training_complete="$training_control_root/state/s32-accepted"' in source
    assert 'training_failed="$training_control_root/state/failed"' in source
    assert '[[ -e "$training_failed" || -L "$training_failed" ]]' in source


def test_prl27_a_evaluation_rechecks_clean_admitted_head_before_binding() -> None:
    source = _source()
    initial_check = source.index("validate_worktree")
    receipt_wait = source.index(
        'while [[ ! -f "$training_complete" || ! -s "$receipt" ]]'
    )
    release_receipt = source.index('cp "$probe" "$runtime_root/resources-released.json"')
    final_check = source.index("validate_worktree", release_receipt)
    binder = source.index('"$python_bin" "$binder"', final_check)

    assert initial_check < receipt_wait < release_receipt < final_check < binder
    assert 'admitted_head=${PRL27_A_ADMITTED_HEAD:-}' in source
    assert "status --porcelain=v1 --untracked-files=all" in source
    assert '"$observed_head" != "$admitted_head"' in source
    assert "admitted-head.txt" in source
    assert source.count("validate_worktree") >= 7


def test_prl27_a_evaluation_rejects_failed_and_accepted_state_coexistence() -> None:
    source = _source()
    receipt_wait = source.index(
        'while [[ ! -f "$training_complete" || ! -s "$receipt" ]]'
    )
    post_wait_failure_gate = source.index(
        '|| -e "$training_failed" || -L "$training_failed"', receipt_wait
    )
    resource_wait = source.index("while (( quiet < release_stable_polls ))")
    resource_failure_gate = source.index(
        'if [[ -e "$training_failed" || -L "$training_failed" ]]',
        resource_wait,
    )
    binder = source.index('"$python_bin" "$binder"', resource_failure_gate)

    assert (
        receipt_wait
        < post_wait_failure_gate
        < resource_wait
        < resource_failure_gate
        < binder
    )


def test_prl27_a_evaluation_contract_is_single_arm_exact_training_run() -> None:
    source = _source()
    lowered = source.lower()

    assert (
        "PRL27-A-CROP-EXACT-CONTINUATION-TRAIN512-S32-MATCHED-"
        "COREDEV2511-PIXEL512-V1"
    ) in source
    assert "bind_prl27_a_corrected_crop_training_run_evaluation.py" in source
    assert source.count('--mode prepare') == 1
    assert source.count('--mode infer') == 1
    assert source.count('--mode score') == 1
    assert source.count('--gpu-ids 0 1 2 3') == 3
    assert "prl26-a" not in lowered
    assert "no-tool-rl" not in lowered
    assert "supervise_prl26_a" not in lowered
    assert "training_run Eval@512" in source
    assert "continuation_environment_token_count\") != 60" in source
    assert (
        "f745fa6cfcc3ba9eb27125a49581fd823fb5930b7b0a51b28e51982999fa2d0a"
        in source
    )
    assert '"stop_strings": ["</tool_call>"]' in source
    assert '"include_stop_str_in_output": True' in source
    assert '"single_response_max_tokens") != 10240' in source


def test_prl27_a_aggregate_covers_all_subsets_and_tool_usage() -> None:
    source = _source()
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
    assert '"generated_token_p95"' in source
