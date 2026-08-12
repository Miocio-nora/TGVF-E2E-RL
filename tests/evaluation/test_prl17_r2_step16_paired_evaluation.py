from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).parents[2]
_PLAN = (
    _ROOT / "configs/evaluation/"
    "prl17_r2_frozen_rp67_tfree_step0_step8_step16_"
    "paired_seed_coredev2511_plan.json"
)
_RUN = (
    _ROOT / "configs/policy/runs/"
    "prl_17_r2_qwen3_instruct_full_frozen_rp67_bs16_n16_"
    "tfree_novisual_8step_ws8.toml"
)
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_SUPERVISOR = (
    _ROOT / "tools/supervise_prl17_r2_tfree_step0_step8_step16_paired_evaluation.sh"
)
_SPEC = importlib.util.spec_from_file_location("prl17_r2_step16_evaluation", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_three_arm_plan_keeps_temp1_and_excludes_arm_identity_from_rng() -> None:
    plan = _MODULE._load_plan(_PLAN)
    run = load_policy_e2e_smoke_run_config(
        _RUN.resolve(), allow_external_agent_loop_config=True
    )
    _MODULE._validate_plan_run(plan, run)

    assert hashlib.sha256(_RUN.read_bytes()).hexdigest() == (
        "3820cf64ddf5e7cc825f6596f5a5ca02f4234fa65d0e67e3f8ec0e906e78593d"
    )
    assert [(arm["name"], arm["optimizer_step"]) for arm in plan["arms"]] == [
        ("step0", 0),
        ("step8", 8),
        ("step16", 16),
    ]
    paired = plan["paired_rng"]
    assert paired["temperature"] == 1.0
    assert paired["do_sample"] is True
    assert tuple(paired["excluded_arm_components"]) == (
        "evaluation_id",
        "arm_name",
        "optimizer_step",
        "checkpoint_hash",
        "policy_weights_sha256",
        "prompt_token_ids_sha256",
    )
    assert run.policy.sampling.temperature == 1.0
    assert run.policy.sampling.do_sample is True


def test_single_supervisor_waits_for_step16_and_uses_fresh_output_root() -> None:
    subprocess.run(["bash", "-n", str(_SUPERVISOR)], check=True)
    source = _SUPERVISOR.read_text(encoding="utf-8")

    assert "--gpu-ids 0 1 2 3 4 5 6 7" in source
    assert "--wait-for-final-arm" in source
    assert "--wait-for-gpus" in source
    assert "STEP0-STEP8-STEP16-PAIRED-SEED-V1" in source
    assert "STEP0-STEP8-SAME-PROTOCOL-V1" not in source
    assert "PRL17_R2_PAIRED_EVAL_MAX_RESTARTS" in source
