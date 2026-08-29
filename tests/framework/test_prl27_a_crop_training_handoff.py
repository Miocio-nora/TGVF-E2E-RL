from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
HANDOFF = TOOLS / "handoff_prl26_c_to_prl27_a_crop_train512_s32.sh"
TMUX_LAUNCHER = TOOLS / "launch_prl27_a_crop_train_and_eval_tmux.sh"
VALIDATOR = TOOLS / "validate_prl27_a_crop_training_handoff.py"
SOURCE_CONFIG = ROOT / (
    "configs/policy/runs/"
    "prl_26_c_qwen3_instruct_short_tgvf_train512_parity_"
    "s32_bs16_n16_teacher25_ws8.toml"
)
TARGET_CONFIG = ROOT / (
    "configs/policy/runs/"
    "prl_27_a_qwen3_instruct_full_crop_train512_exact_continuation_"
    "s32_bs16_n16_teacher25_ws8.toml"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prl27_a_training_handoff_under_test", VALIDATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_handoff_is_shell_valid_and_orders_every_fail_closed_gate() -> None:
    subprocess.run(["bash", "-n", str(HANDOFF)], check=True)
    source = HANDOFF.read_text(encoding="utf-8")

    gates = (
        "source-complete",
        'flock -w "$source_lock_timeout_seconds" 8',
        "resources-free",
        "target-ready",
        source.rindex("trainable_tgvf_supervisor"),
        "target-complete",
    )
    positions = [
        source.index(gates[0]),
        source.index(gates[1]),
        source.index(gates[2]),
        source.index(gates[3]),
        gates[4],
        source.index(gates[5]),
    ]
    assert positions == sorted(positions)
    assert "supervise_prl26_tgvf_prompt_train_and_eval.sh" in source
    assert 'exec 8>>"$source_supervisor_lock"' in source
    assert 'mkdir -p "$state_root" "$log_root"' in source
    assert 'mkdir -p "$target_root"' not in source
    assert "crop-fresh-s0-authorization.json" in source
    assert "--mode formal --target-step 32 --compose-only" in source
    assert "setsid" in source and "stop_process_group" in source


def test_handoff_never_kills_or_rewrites_the_prl26_source() -> None:
    source = HANDOFF.read_text(encoding="utf-8")

    assert 'kill -TERM -- "-$pid"' in source
    assert "kill -TERM 749980" not in source
    assert 'stop_process_group "$active_pid"' in source
    assert "source_supervisor_lock" in source
    assert "source-completion-after-lock-audit" in source
    assert "source_config" not in "\n".join(
        line for line in source.splitlines() if line.lstrip().startswith("rm ")
    )
    lock = source.index('flock -w "$source_lock_timeout_seconds" 8')
    exit_130 = source.index('"$released_source_status" != 130', lock)
    resources = source.index("resources-free", exit_130)
    assert lock < exit_130 < resources
    assert "pane_dead_status" in source
    assert "phase=signal" in source and "exit_status=130" in source


def test_tmux_launcher_arms_two_retained_sessions_without_secret_commands() -> None:
    subprocess.run(["bash", "-n", str(TMUX_LAUNCHER)], check=True)
    source = TMUX_LAUNCHER.read_text(encoding="utf-8")

    clean_gate = source.index("status --porcelain=v1 --untracked-files=all")
    first_session = source.index("tmux new-session")
    signal_gate = source.index('tmux wait-for -S "$launch_gate"')
    assert clean_gate < first_session < signal_gate
    assert source.count("tmux new-session") == 2
    assert source.count("remain-on-exit on") == 2
    assert 'tmux has-session -t "=$name"' in source
    assert "handoff_prl26_c_to_prl27_a_crop_train512_s32.sh" in source
    assert "supervise_prl27_a_corrected_crop_s32_evaluation.sh" in source
    assert '-e "OPENROUTER_API_KEY=$OPENROUTER_API_KEY"' in source
    assert '-e "PRL27_A_ADMITTED_HEAD=$admitted_head"' in source
    pane_commands = source[
        source.index("training_command=") : source.index(
            "# Secrets are session-scoped"
        )
    ]
    assert "OPENROUTER_API_KEY" not in pane_commands
    assert "created_sessions" in source and "kill-session" in source


def test_formal_contract_binds_source_core_fix_and_exact_crop_runtime() -> None:
    module = _module()
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = module.validate_contracts(
        repository=ROOT,
        source_config_path=SOURCE_CONFIG,
        target_config_path=TARGET_CONFIG,
        admitted_head=head,
        require_clean=False,
    )

    assert result == {
        "schema_version": "tgvf.prl27-a-training-contract-audit.v1",
        "status": "accepted",
        "admitted_head": head,
        "source_run_id": module.EXPECTED_SOURCE_RUN_ID,
        "source_identity_sha256": module.EXPECTED_SOURCE_IDENTITY_SHA256,
        "target_run_id": module.EXPECTED_TARGET_RUN_ID,
        "target_identity_sha256": module.EXPECTED_TARGET_IDENTITY_SHA256,
        "target_output_root": str(module.EXPECTED_TARGET_ROOT),
        "continuation_sha256": module.EXPECTED_CONTINUATION_SHA256,
        "response_budget_scope": "total_response_tokens",
        "single_response_max_tokens": 10_240,
    }


def test_fresh_s0_authorization_is_immutable_and_head_bound(
    tmp_path: Path,
) -> None:
    module = _module()
    if module.EXPECTED_TARGET_ROOT.exists():
        pytest.skip("formal PRL-27-A root already exists; fresh-S0 path is historical")
    authorization = tmp_path / "authorization.json"
    admitted_head = "a" * 40

    first = module.validate_target_readiness(
        config_path=TARGET_CONFIG,
        authorization_path=authorization,
        admitted_head=admitted_head,
    )
    second = module.validate_target_readiness(
        config_path=TARGET_CONFIG,
        authorization_path=authorization,
        admitted_head=admitted_head,
    )

    assert first["launch_mode"] == second["launch_mode"] == "fresh-s0"
    assert first["checkpointed_step"] == second["checkpointed_step"] == 0
    value = json.loads(authorization.read_text(encoding="utf-8"))
    assert value["admitted_training_head"] == admitted_head
    assert value["continuation_sha256"] == module.EXPECTED_CONTINUATION_SHA256

    with pytest.raises(RuntimeError, match="authorization identity differs"):
        module.validate_target_readiness(
            config_path=TARGET_CONFIG,
            authorization_path=authorization,
            admitted_head="b" * 40,
        )


def test_existing_target_root_without_authorization_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    target = module._load_config(TARGET_CONFIG)
    foreign_root = tmp_path / "foreign-root"
    foreign_root.mkdir()
    monkeypatch.setattr(module, "EXPECTED_TARGET_ROOT", foreign_root)
    altered_output = replace(
        target.output,
        root=foreign_root,
        checkpoint_directory=foreign_root / "checkpoints",
        metrics_path=foreign_root / "metrics.jsonl",
    )
    monkeypatch.setattr(
        module,
        "_load_config",
        lambda _path: replace(target, output=altered_output),
    )

    with pytest.raises(RuntimeError, match="without its fresh-S0 authorization"):
        module.validate_target_readiness(
            config_path=TARGET_CONFIG,
            authorization_path=tmp_path / "missing-authorization.json",
            admitted_head="a" * 40,
        )
