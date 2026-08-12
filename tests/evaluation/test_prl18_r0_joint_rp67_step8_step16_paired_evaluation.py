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
    "prl18_r0_joint_rp67_tfree_step8_step16_"
    "paired_seed_coredev2511_plan.json"
)
_FROZEN_PLAN = (
    _ROOT / "configs/evaluation/"
    "prl17_r2_frozen_rp67_tfree_step0_step8_step16_"
    "paired_seed_coredev2511_plan.json"
)
_RUN = (
    _ROOT / "configs/policy/runs/"
    "prl_18_r0_qwen3_instruct_full_joint_rp67_bs16_n16_"
    "tfree_novisual_8step_ws8.toml"
)
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_SUPERVISOR = (
    _ROOT
    / "tools/supervise_prl18_r0_joint_rp67_tfree_step8_step16_paired_evaluation.sh"
)
_SPEC = importlib.util.spec_from_file_location("prl18_r0_paired_evaluation", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_two_arm_plan_binds_joint_rp67_and_shared_paired_rng() -> None:
    plan = _MODULE._load_plan(_PLAN)
    run = load_policy_e2e_smoke_run_config(
        _RUN.resolve(), allow_external_agent_loop_config=True
    )
    _MODULE._validate_plan_run(plan, run)

    assert hashlib.sha256(_RUN.read_bytes()).hexdigest() == (
        "3ae96a2c8c194dd4bd2d1e1d7f2bdbb04b60e77831eaee923195156e18f459c7"
    )
    assert [(arm["name"], arm["optimizer_step"]) for arm in plan["arms"]] == [
        ("step8", 8),
        ("step16", 16),
    ]
    assert run.representation.adapter_update_mode.value == "joint"
    assert run.representation.adapter_trainable is True
    assert run.training.permanent_checkpoint_steps == (8,)

    frozen = json.loads(_FROZEN_PLAN.read_text(encoding="utf-8"))
    assert plan["paired_rng"]["seed_namespace"] == (
        frozen["paired_rng"]["seed_namespace"]
    )
    assert plan["paired_rng"]["temperature"] == 1.0
    assert plan["paired_rng"]["do_sample"] is True


def test_joint_pairing_contract_has_no_frozen_only_state_equality_fields() -> None:
    plan = _MODULE._load_plan(_PLAN)
    required = plan["required_pairing"]

    assert required == {
        "qwen_and_rp66_optimizer_step_must_match": True,
        "full_qwen_checkpoint_must_be_materialized_without_adapter_keys": True,
        "do_not_use_crop_native_pixels_backend": True,
        "do_not_treat_rp66_as_policy_lora": True,
        "adapter_update_mode": "joint",
        "rp66_state_must_remain_constant": False,
    }
    assert "adapter_manifest_and_tensor_sha256_must_be_frozen" not in required
    assert "expected_runtime_rp66_weights_sha256" not in required
    _MODULE._validate_materialized_frozen_pairing(plan, {})


def test_supervisor_waits_for_step16_and_evaluates_only_step8_and_step16() -> None:
    subprocess.run(["bash", "-n", str(_SUPERVISOR)], check=True)
    source = _SUPERVISOR.read_text(encoding="utf-8")

    assert "--gpu-ids 0 1 2 3 4 5 6 7" in source
    assert "--wait-for-final-arm" in source
    assert "--wait-for-gpus" in source
    assert "STEP8-STEP16-PAIRED-SEED-V1" in source
    assert "STEP0" not in source
    assert "PRL18_R0_PAIRED_EVAL_MAX_RESTARTS" in source
