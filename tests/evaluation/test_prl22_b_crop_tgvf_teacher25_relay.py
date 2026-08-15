from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).parents[2]
_PLAN = (
    _ROOT
    / "configs/evaluation/"
    "prl22_b_r0_frozen_rp67_tfree_crop_tgvf_teacher25_"
    "step8_step16_paired_seed_coredev2511_plan.json"
)
_RUN = (
    _ROOT
    / "configs/policy/runs/"
    "prl_22_b_r0_qwen3_instruct_full_frozen_rp67_bs16_n16_"
    "tfree_crop_tgvf_teacher25_8step_ws8.toml"
)
_EXECUTOR = _ROOT / "tools/run_prl15_paired_evaluation.py"
_TRAIN_SUPERVISOR = (
    _ROOT / "tools/supervise_prl22_b_crop_tgvf_teacher25_step16_and_eval.sh"
)
_EVAL_SUPERVISOR = (
    _ROOT
    / "tools/"
    "supervise_prl22_b_crop_tgvf_teacher25_"
    "step8_step16_paired_evaluation.sh"
)
_RELAY = _ROOT / "tools/relay_prl22_a_complete_to_prl22_b_crop_tgvf_teacher25.sh"
_SPEC = importlib.util.spec_from_file_location("prl22_b_paired_evaluation", _EXECUTOR)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_plan_binds_teacher_quarter_atomic_crop_tgvf_and_prl20_rng() -> None:
    plan = _MODULE._load_plan(_PLAN)
    run = load_policy_e2e_smoke_run_config(
        _RUN.resolve(), allow_external_agent_loop_config=True
    )
    _MODULE._validate_plan_run(plan, run)

    assert hashlib.sha256(_RUN.read_bytes()).hexdigest() == (
        "4ef4eba5b46704b19b5a727263f9d7a448a207ff207f581dc8692ebd560bd312"
    )
    assert run.dataset.kind == "policy_t1_teacher_quarter_mix"
    assert run.protocol.tool_profile.value == "crop_tgvf"
    assert run.protocol.enabled_tool_names == ("tgvf_crop_tool",)
    assert run.representation.adapter_update_mode.value == "frozen_adapter"
    assert run.training.permanent_checkpoint_steps == (8,)
    assert [(arm["name"], arm["optimizer_step"]) for arm in plan["arms"]] == [
        ("step8", 8),
        ("step16", 16),
    ]
    assert plan["paired_rng"]["seed_namespace"] == (
        "coredev2511-official-v1/rp67-tfree-crop-tgvf/"
        "step8-step16/temp1/seed42/v1"
    )
    assert plan["paired_rng"]["protocol_sha256"] == (
        "576beb9a1b77148249f87ff86c118acb7003efe1012ca651495cf908c536c656"
    )


def test_supervisors_bind_step8_step16_wandb_and_evaluation() -> None:
    for script in (_TRAIN_SUPERVISOR, _EVAL_SUPERVISOR, _RELAY):
        subprocess.run(["bash", "-n", str(script)], check=True)

    train = _TRAIN_SUPERVISOR.read_text(encoding="utf-8")
    evaluation = _EVAL_SUPERVISOR.read_text(encoding="utf-8")
    assert "WANDB_RUN_ID=prl22bt25" in train
    assert "--target-step 8" in train
    assert "--target-step 16" in train
    assert "PRL-22-B-CROP-TGVF-TEACHER25-STEP8-TO16" in train
    assert 'exec "$post_train_eval"' in train
    assert "--gpu-ids 0 1 2 3 4 5 6 7" in evaluation
    assert "--wait-for-final-arm" in evaluation
    assert "--wait-for-gpus" in evaluation


def test_relay_requires_canonical_a_receipt_hash_and_gpu_release() -> None:
    relay = _RELAY.read_text(encoding="utf-8")
    assert "tgvf.paired-coredev-evaluation-complete.v1" in relay
    assert "paired_summary_sha256" in relay
    assert "source evaluation receipt identity differs" in relay
    assert 'while source_is_running; do sleep "$poll_seconds"; done' in relay
    assert "quiet_polls < 2" in relay
    assert "nvidia-smi --query-compute-apps=pid" in relay
    assert 'exec "$target_supervisor"' in relay
