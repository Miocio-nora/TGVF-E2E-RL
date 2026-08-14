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
    "prl20_r0_frozen_rp67_tfree_crop_tgvf_step8_step16_"
    "paired_seed_coredev2511_plan.json"
)
_RUN = (
    _ROOT
    / "configs/policy/runs/"
    "prl_20_r0_qwen3_instruct_full_frozen_rp67_bs16_n16_"
    "tfree_crop_tgvf_8step_ws8.toml"
)
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_SUPERVISOR = (
    _ROOT
    / "tools/"
    "supervise_prl20_r0_frozen_rp67_tfree_crop_tgvf_"
    "step8_step16_paired_evaluation.sh"
)
_SPEC = importlib.util.spec_from_file_location("prl20_r0_paired_evaluation", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_two_arm_plan_binds_atomic_crop_tgvf_and_paired_rng() -> None:
    plan = _MODULE._load_plan(_PLAN)
    run = load_policy_e2e_smoke_run_config(
        _RUN.resolve(), allow_external_agent_loop_config=True
    )
    _MODULE._validate_plan_run(plan, run)

    assert hashlib.sha256(_RUN.read_bytes()).hexdigest() == (
        "645690f7d2fc26cd98cd919907ecae3750fceff73b7fae38b3669c421393d3fe"
    )
    assert [(arm["name"], arm["optimizer_step"]) for arm in plan["arms"]] == [
        ("step8", 8),
        ("step16", 16),
    ]
    assert plan["protocol"] == {
        "evaluation_protocol": "training_run",
        "prompt_sha256": run.protocol.prompt_sha256,
        "tool_profile": "crop_tgvf",
        "tool_schema_sha256": run.protocol.tool_schema_sha256,
        "maximum_tool_calls": 6,
        "sampling_source": "bound_policy_run_config",
        "same_tasks_and_rank_partition": True,
    }
    assert plan["paired_rng"]["protocol_sha256"] == (
        "576beb9a1b77148249f87ff86c118acb7003efe1012ca651495cf908c536c656"
    )
    assert plan["paired_rng"]["temperature"] == 1.0
    assert plan["paired_rng"]["do_sample"] is True


def test_pairing_keeps_frozen_rp67_and_forbids_crop_pixels_backend() -> None:
    plan = _MODULE._load_plan(_PLAN)
    required = plan["required_pairing"]

    assert required["adapter_update_mode"] == "frozen_adapter"
    assert required["rp66_state_must_remain_constant"] is True
    assert required["adapter_manifest_and_tensor_sha256_must_be_frozen"] is True
    assert required["full_qwen_checkpoint_must_be_materialized_without_adapter_keys"] is True
    assert required["do_not_use_crop_native_pixels_backend"] is True
    assert required["do_not_treat_rp66_as_policy_lora"] is True


def test_supervisor_waits_for_step16_then_splits_two_arms_across_eight_gpus() -> None:
    subprocess.run(["bash", "-n", str(_SUPERVISOR)], check=True)
    source = _SUPERVISOR.read_text(encoding="utf-8")

    assert "--gpu-ids 0 1 2 3 4 5 6 7" in source
    assert "--wait-for-final-arm" in source
    assert "--wait-for-gpus" in source
    assert "STEP8-STEP16-PAIRED-SEED-V1" in source
    assert "STEP0" not in source
    assert "PRL20_R0_PAIRED_EVAL_MAX_RESTARTS" in source
