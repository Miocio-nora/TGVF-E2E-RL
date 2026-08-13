from __future__ import annotations

from pathlib import Path
import subprocess


_ROOT = Path(__file__).parents[2]
_SUPERVISOR = _ROOT / "tools/supervise_prl20_r0_crop_tgvf_step8_to16_and_eval.sh"


def test_handoff_waits_for_source_exit_then_continues_exact_step8_boundary() -> None:
    subprocess.run(["bash", "-n", str(_SUPERVISOR)], check=True)
    source = _SUPERVISOR.read_text(encoding="utf-8")

    assert 'tmux has-session -t "$source_session"' in source
    assert "#{pane_dead}" in source
    assert "while source_training_is_running" in source
    assert "checkpoint_is_closed 8" in source
    assert "materialize_policy_horizon_extension.py" in source
    assert "--target-step 16" in source
    assert "TGVF_POLICY_HORIZON_EXTENSION_PATH" in source
    assert "WANDB_RUN_ID=prl20r0" in source
    assert "WANDB_RESUME=must" in source


def test_handoff_defers_evaluation_until_step16_and_requires_it() -> None:
    source = _SUPERVISOR.read_text(encoding="utf-8")

    assert source.index('touch "$control_root/step16-accepted"') < source.index(
        'exec "$post_train_eval"'
    )
    assert "supervise_prl20_r0_frozen_rp67_tfree_crop_tgvf_step8_step16_paired_evaluation.sh" in source
    assert "absent or not executable" in source
