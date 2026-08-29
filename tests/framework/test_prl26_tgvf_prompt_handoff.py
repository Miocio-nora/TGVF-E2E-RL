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
    validator: ModuleType, tmp_path: Path
) -> None:
    marker = tmp_path / "evaluation-complete"
    marker.touch()
    failed = tmp_path / "failed"
    arms: dict[str, object] = {}
    for name in ("no_tool", "crop"):
        summary = tmp_path / f"{name}-summary.json"
        summary.write_text('{"status":"complete"}\n', encoding="utf-8")
        arms[name] = {
            "train_image_max_pixels": 262_144,
            "evaluation_image_max_pixels": 262_144,
            "optimizer_step": 32,
            "macro_star_percent": 61.25,
            "seven_subset_statistics": [{} for _ in range(7)],
            "summary_path": str(summary),
        }
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": validator._RESULT_SCHEMA,
                "status": "pass",
                "contract": "fresh-S0 Train@512 S32; matched Eval@512",
                "arms": arms,
            }
        ),
        encoding="utf-8",
    )

    accepted = validator.validate_prerequisite(
        result_path=result_path,
        complete_marker=marker,
        failed_marker=failed,
    )
    assert accepted["status"] == "accepted"
    assert set(accepted["arms"]) == {"no_tool", "crop"}

    failed.write_text("failed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="completion boundary differs"):
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
