from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "tools/validate_prl26_tgvf_prompt_handoff.py"
SUPERVISOR = ROOT / "tools/supervise_prl26_tgvf_prompt_train_and_eval.sh"
A_B_LAUNCHER = ROOT / "tools/launch_prl26_train512_s32_coredev2511_tmux.sh"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prl26_tgvf_prompt_handoff_under_test", VALIDATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator() -> ModuleType:
    return _load_module()


def test_supervisor_orders_prerequisite_canaries_formal_arms_and_eval() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    prerequisite = '"$python_bin" "$validator" prerequisite'
    canary = "--run-config \"$short_c0\" --mode canary --target-step 1"
    short = 'run_formal short "$short_formal"'
    full = 'run_formal full "$full_formal"'
    evaluation = 'exec "$post_train_eval" "$admitted_head"'

    assert source.index(prerequisite) < source.index(canary)
    assert source.index(canary) < source.index(short)
    assert source.index(short) < source.index(full)
    assert source.index(full) < source.index(evaluation)
    assert 'export PYTHONPATH="$repo_root/src:' in source
    assert 'export TGVF_REPOSITORY_ROOT="$repo_root"' in source
    assert 'mkdir -p "$short_c0_root"' not in source
    assert 'mkdir -p "$full_c0_root"' not in source
    assert 'mkdir -p "$short_root"' not in source
    assert 'mkdir -p "$full_root"' not in source
    assert 'while [[ ! -f "$prerequisite_complete" ]]' in source
    assert 'while [[ ! -s "$prerequisite_complete" ]]' not in source
    assert 'if [[ "$prerequisite_remain" != on ]]' in source
    assert "A/B evaluator pane disappeared before completion" in source
    assert "A/B evaluator pane disappeared after completion" in source
    assert "A/B evaluator exit status is unavailable" in source


def test_ab_launcher_retains_pane_for_downstream_exit_status_audit() -> None:
    source = A_B_LAUNCHER.read_text(encoding="utf-8")
    launch = 'tmux new-session -d -s "$session"'
    retain = 'tmux set-option -t "$session" remain-on-exit on'
    verify = 'tmux show-options -t "$session" -v remain-on-exit'
    assert source.index(launch) < source.index(retain) < source.index(verify)
    assert 'tmux kill-session -t "$session"' in source


def test_current_tgvf_prompt_config_matrix_is_exact(
    validator: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = "f" * 40

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(ROOT)
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments == ("rev-parse", "HEAD"):
            return head
        raise AssertionError(arguments)

    monkeypatch.setattr(validator, "_git", fake_git)
    monkeypatch.setattr(
        validator.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    result = validator.validate_contracts(
        repository=ROOT,
        short_canary_path=ROOT
        / "configs/policy/runs/prl_26_c_c0_qwen3_instruct_short_tgvf_"
        "train512_parity_bs4_n2_teacher25_1step_ws4.toml",
        full_canary_path=ROOT
        / "configs/policy/runs/prl_26_d_c0_qwen3_instruct_target_guide_v2_"
        "tgvf_train512_parity_bs4_n2_teacher25_1step_ws4.toml",
        short_formal_path=ROOT
        / "configs/policy/runs/prl_26_c_qwen3_instruct_short_tgvf_"
        "train512_parity_s32_bs16_n16_teacher25_ws8.toml",
        full_formal_path=ROOT
        / "configs/policy/runs/prl_26_d_qwen3_instruct_target_guide_v2_"
        "tgvf_train512_parity_s32_bs16_n16_teacher25_ws8.toml",
    )

    assert result["status"] == "accepted"
    assert result["repository_head"] == head
    assert result["configs"]["short_canary"]["output_exists"] is False
    assert result["configs"]["full_canary"]["output_exists"] is False


def test_prerequisite_requires_complete_passed_two_arm_result(
    validator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    control = tmp_path / "control"
    runtime = control / "runtime"
    runtime.mkdir(parents=True)
    marker = runtime / "evaluation-complete"
    marker.touch()
    failed = runtime / "failed"
    result_path = control / "train512-s32-pixel512-results.json"
    handoff_path = runtime / "bound-handoff.json"
    summaries = {
        name: tmp_path / name / "coredev-2511-eval-summary.json"
        for name in ("no_tool", "crop")
    }
    for summary in summaries.values():
        summary.parent.mkdir(parents=True)
    monkeypatch.setattr(validator, "_EXPECTED_RESULT_PATH", result_path)
    monkeypatch.setattr(validator, "_EXPECTED_COMPLETE_MARKER", marker)
    monkeypatch.setattr(validator, "_EXPECTED_FAILED_MARKER", failed)
    monkeypatch.setattr(validator, "_EXPECTED_HANDOFF_PATH", handoff_path)
    monkeypatch.setattr(validator, "_EXPECTED_SUMMARY_PATHS", summaries)

    headline = {
        "schema_version": "tgvf.coredev-2511-macro-star.v1",
        "macro_star_percent": 61.25,
    }
    monkeypatch.setattr(
        validator, "extract_coredev_macro_star", lambda _summary: headline
    )
    slices = [{"dataset": str(index)} for index in range(7)]
    arms: dict[str, object] = {}
    for name in ("no_tool", "crop"):
        summary = summaries[name]
        summary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "pass",
                    "phase": "eval",
                    "model": "Qwen3-VL-8B-Instruct",
                    "sample_count": 2511,
                    "slice_count": 7,
                    "slices": slices,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        arms[name] = {
            "method": validator._EXPECTED_METHODS[name],
            "train_image_max_pixels": 262_144,
            "evaluation_image_max_pixels": 262_144,
            "optimizer_step": 32,
            "macro_star_percent": 61.25,
            "headline": headline,
            "seven_subset_statistics": slices,
            "summary_path": str(summary),
        }

    handoff_arms: dict[str, object] = {}
    for name in ("no_tool", "crop"):
        config = tmp_path / f"{name}.toml"
        completion = tmp_path / f"{name}-completion.json"
        config.write_text(f"name = \"{name}\"\n", encoding="utf-8")
        completion.write_text('{"optimizer_step":32}\n', encoding="utf-8")
        handoff_arms[name] = {
            "evaluation_id": validator._EXPECTED_EVALUATION_IDS[name],
            "run_id": f"RUN-{name}",
            "run_identity_sha256": "a" * 64,
            "config_path": str(config),
            "config_file_sha256": validator._sha256(config),
            "completion_path": str(completion),
            "completion_file_sha256": validator._sha256(completion),
            "checkpoint_pair_integrity_sha256": "b" * 64,
        }
    plan = tmp_path / "bound-plan.json"
    plan.write_text('{"status":"ready"}\n', encoding="utf-8")
    crop_handoff = handoff_arms["crop"]
    assert isinstance(crop_handoff, dict)
    crop_handoff.update(
        {
            "bound_plan_path": str(plan),
            "bound_plan_file_sha256": validator._sha256(plan),
            "tool_action_boundary": {
                "stop_strings": ["</tool_call>"],
                "include_stop_str_in_output": True,
            },
        }
    )
    handoff_content = {
        "schema_version": "tgvf.prl26-train512-s32-evaluation-handoff.v1",
        "status": "ready",
        "train_image_max_pixels": 262_144,
        "evaluation_image_max_pixels": 262_144,
        "optimizer_step": 32,
        "coverage": validator._EXPECTED_HANDOFF_COVERAGE,
        **handoff_arms,
    }
    handoff = {
        **handoff_content,
        "identity_sha256": validator._canonical_sha256(handoff_content),
    }
    handoff_path.write_text(json.dumps(handoff) + "\n", encoding="utf-8")

    result_payload = {
        "schema_version": validator._RESULT_SCHEMA,
        "status": "pass",
        "contract": "fresh-S0 Train@512 S32; matched Eval@512",
        "coverage": validator._EXPECTED_COVERAGE,
        "handoff_identity_sha256": handoff["identity_sha256"],
        "arms": arms,
    }
    result_path.write_text(
        json.dumps(result_payload),
        encoding="utf-8",
    )

    accepted = validator.validate_prerequisite(
        result_path=result_path,
        complete_marker=marker,
        failed_marker=failed,
    )
    assert accepted["status"] == "accepted"
    assert set(accepted["arms"]) == {"no_tool", "crop"}
    assert accepted["handoff_identity_sha256"] == handoff["identity_sha256"]
    assert accepted["coverage"] == validator._EXPECTED_COVERAGE

    failed.symlink_to(tmp_path / "missing-failure-target")
    with pytest.raises(RuntimeError, match="completion boundary differs"):
        validator.validate_prerequisite(
            result_path=result_path,
            complete_marker=marker,
            failed_marker=failed,
        )

    failed.unlink()
    drifted_result = dict(result_payload)
    drifted_result["coverage"] = {**validator._EXPECTED_COVERAGE, "subset_count": 6}
    result_path.write_text(json.dumps(drifted_result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="result table contract differs"):
        validator.validate_prerequisite(
            result_path=result_path,
            complete_marker=marker,
            failed_marker=failed,
        )

    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    drifted_headline = json.loads(json.dumps(result_payload))
    drifted_headline["arms"]["no_tool"]["macro_star_percent"] = 60.0
    result_path.write_text(json.dumps(drifted_headline), encoding="utf-8")
    with pytest.raises(RuntimeError, match="published headline differs"):
        validator.validate_prerequisite(
            result_path=result_path,
            complete_marker=marker,
            failed_marker=failed,
        )

    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    no_tool_summary = summaries["no_tool"]
    no_tool_summary_backup = no_tool_summary.with_name("summary-backup.json")
    no_tool_summary.rename(no_tool_summary_backup)
    no_tool_summary.symlink_to(no_tool_summary_backup)
    with pytest.raises(RuntimeError, match="summary is not a regular file"):
        validator.validate_prerequisite(
            result_path=result_path,
            complete_marker=marker,
            failed_marker=failed,
        )
    no_tool_summary.unlink()
    no_tool_summary_backup.rename(no_tool_summary)

    drifted_handoff = dict(handoff)
    drifted_handoff["identity_sha256"] = "c" * 64
    handoff_path.write_text(json.dumps(drifted_handoff), encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical identity differs"):
        validator.validate_prerequisite(
            result_path=result_path,
            complete_marker=marker,
            failed_marker=failed,
        )


def test_canary_completion_binds_clean_target_worktree_provenance(
    validator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    canary = output / "canary"
    checkpoint = canary / "checkpoints/global_step_1"
    checkpoint.mkdir(parents=True)
    (canary / "checkpoints/latest_checkpointed_iteration.txt").write_text(
        "1", encoding="utf-8"
    )
    metrics = {
        "schema_version": "policy-pilot-v1-metrics-event-v1",
        "optimizer_step": 1,
        "step": {},
        "cumulative": {"optimizer_steps": 1},
        "timing": {},
    }
    (canary / "metrics.jsonl").write_text(
        json.dumps(metrics) + "\n", encoding="utf-8"
    )
    head = "a" * 40
    repository = tmp_path / "repository"
    repository.mkdir()
    config = SimpleNamespace(
        output=SimpleNamespace(root=output),
        run_id="CANARY",
        identity_sha256="b" * 64,
        source_sha256="c" * 64,
    )
    provenance = {
        "schema_version": validator._PROVENANCE_SCHEMA,
        "mode": "canary",
        "target_step": 1,
        "run_id": config.run_id,
        "run_identity_sha256": config.identity_sha256,
        "run_config_file_sha256": config.source_sha256,
        "project": {
            "root": str(repository),
            "commit": head,
            "clean": True,
            "changes": [],
        },
    }
    (canary / "launch-provenance.jsonl").write_text(
        json.dumps(provenance) + "\n", encoding="utf-8"
    )
    state = SimpleNamespace(
        progress=SimpleNamespace(optimizer_step=1), integrity_sha256="d" * 64
    )
    pair = SimpleNamespace(optimizer_step=1, integrity_sha256="e" * 64)
    monkeypatch.setattr(
        validator, "load_policy_e2e_smoke_run_config", lambda _path: config
    )
    monkeypatch.setattr(
        validator,
        "_validate_generation",
        lambda *_args, **_kwargs: (state, pair, [tmp_path / str(i) for i in range(14)]),
    )

    result = validator.validate_canary_completion(
        config_path=tmp_path / "config.toml",
        repository=repository,
        expected_head=head,
    )
    assert result["status"] == "accepted"
    assert result["optimizer_step"] == 1
