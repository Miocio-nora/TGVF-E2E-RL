from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).parents[2]
_PLAN = (
    _ROOT / "configs/evaluation/"
    "prl19_r0_frozen_rp67_tfree_visual_api_step8_step16_"
    "paired_seed_coredev2511_plan.json"
)
_COMMON = (
    _ROOT / "configs/evaluation/"
    "prl17_r2_frozen_rp67_tfree_step0_step8_step16_"
    "paired_seed_coredev2511_plan.json"
)
_RUN = (
    _ROOT / "configs/policy/runs/"
    "prl_19_r0_qwen3_instruct_full_frozen_rp67_bs16_n16_"
    "tfree_visual_api_8step_ws8.toml"
)
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_EVAL_SUPERVISOR = (
    _ROOT
    / "tools/supervise_prl19_r0_frozen_rp67_tfree_visual_api_"
    "step8_step16_paired_evaluation.sh"
)
_TRAIN_SUPERVISOR = (
    _ROOT
    / "tools/supervise_prl19_r0_frozen_rp67_tfree_visual_api_"
    "step16_and_eval.sh"
)
_SPEC = importlib.util.spec_from_file_location("prl19_r0_paired_evaluation", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_two_arm_plan_reuses_common_step0_rng_without_rerunning_it() -> None:
    plan = _MODULE._load_plan(_PLAN)
    run = load_policy_e2e_smoke_run_config(_RUN.resolve())
    _MODULE._validate_plan_run(plan, run)

    assert hashlib.sha256(_RUN.read_bytes()).hexdigest() == (
        "e8f1d45d7568b86d522dd2fa70a622bc23c695c3e8bb9107c04531da256721ff"
    )
    assert [(arm["name"], arm["optimizer_step"]) for arm in plan["arms"]] == [
        ("step8", 8),
        ("step16", 16),
    ]
    common = json.loads(_COMMON.read_text(encoding="utf-8"))
    assert plan["paired_rng"] == common["paired_rng"]
    assert all(arm["name"] != "step0" for arm in plan["arms"])


def test_pairing_requires_frozen_rp67_at_both_checkpoints() -> None:
    plan = _MODULE._load_plan(_PLAN)
    required = plan["required_pairing"]

    assert required["adapter_update_mode"] == "frozen_adapter"
    assert required["rp66_state_must_remain_constant"] is True
    assert required["adapter_manifest_and_tensor_sha256_must_be_frozen"] is True
    assert required["expected_runtime_rp66_weights_sha256"] == (
        "3f60f36589a3c0f3549c12b949eaabb140f6edfac849aa2b25a623bbcde53a14"
    )


def test_supervisors_chain_smoke_train_continue_and_two_arm_evaluation() -> None:
    subprocess.run(["bash", "-n", str(_TRAIN_SUPERVISOR)], check=True)
    subprocess.run(["bash", "-n", str(_EVAL_SUPERVISOR)], check=True)
    training = _TRAIN_SUPERVISOR.read_text(encoding="utf-8")
    evaluation = _EVAL_SUPERVISOR.read_text(encoding="utf-8")

    assert "--mode smoke" in training
    assert "--target-step 8" in training
    assert "--target-step 16" in training
    assert "materialize_policy_horizon_extension.py" in training
    assert "WANDB_RUN_ID=prl19r0v" in training
    assert "tracker < step" in training
    assert "coverage < 0.99" in training
    assert "exec \"$post_train_eval\"" in training
    assert "--gpu-ids 0 1 2 3 4 5 6 7" in evaluation
    assert "--wait-for-final-arm" in evaluation
    assert "--wait-for-gpus" in evaluation
