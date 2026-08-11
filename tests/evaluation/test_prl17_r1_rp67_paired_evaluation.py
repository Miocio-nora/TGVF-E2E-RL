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
    "prl17_r1_frozen_rp67_step0_step8_coredev2511_plan.json"
)
_RUN = (
    _ROOT
    / "configs/policy/runs/"
    "prl_17_r1_qwen3_instruct_full_frozen_rp67_bs16_n16_t1_shaped_novisual_8step_ws8.toml"
)
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_SUPERVISOR = _ROOT / "tools/supervise_prl17_r1_step0_step8_evaluation.sh"
_SPEC = importlib.util.spec_from_file_location("prl17_r1_evaluation", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_prl17_r1_plan_binds_exact_frozen_rp67_step0_step8_pair() -> None:
    plan = _MODULE._load_plan(_PLAN)
    run = load_policy_e2e_smoke_run_config(
        _RUN.resolve(), allow_external_agent_loop_config=True
    )

    _MODULE._validate_plan_run(plan, run)

    assert plan["policy_config_sha256"] == hashlib.sha256(_RUN.read_bytes()).hexdigest()
    assert plan["evaluation_id"] == (
        "PRL17-R1-FROZEN-RP67-COREDEV2511-STEP0-STEP8-SAME-PROTOCOL-V1"
    )
    assert plan["arms"] == [
        {
            "name": "step0",
            "optimizer_step": 0,
            "qwen_source": "model.path",
            "rp66_source": "representation.artifact_path",
        },
        {
            "name": "step8",
            "optimizer_step": 8,
            "qwen_source": "output.root/permanent-checkpoints/global_step_8",
            "rp66_source": (
                "output.root/runtime-policy-state/"
                "lora-manifests/step-00000008-*.json"
            ),
        },
    ]
    required = plan["required_pairing"]
    assert run.representation.adapter_update_mode.value == "frozen_adapter"
    assert required["adapter_update_mode"] == "frozen_adapter"
    assert required["rp66_state_must_remain_constant"] is True
    assert required["expected_runtime_rp66_weights_sha256"] == (
        "3f60f36589a3c0f3549c12b949eaabb140f6edfac849aa2b25a623bbcde53a14"
    )
    assert run.representation.artifact_file_sha256 == (
        "13332865eb30a2b04ce2ee90a9228e490c718e87fa57bc758078cdd28b6f0f68"
    )


def test_prl17_r1_supervisor_waits_and_uses_all_eight_gpus() -> None:
    subprocess.run(["bash", "-n", str(_SUPERVISOR)], check=True)
    source = _SUPERVISOR.read_text(encoding="utf-8")

    assert "--gpu-ids 0 1 2 3 4 5 6 7" in source
    assert "--wait-for-step8" in source
    assert "--wait-for-gpus" in source
    assert "PRL17_R1_EVAL_MAX_RESTARTS" in source
    assert "deterministic evaluation contract failure; refusing retry" in source
