from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).parents[2]
_PLAN_PATH = (
    _ROOT
    / "configs/evaluation/prl15_rp66_step0_step8_coredev2511_plan.json"
)
_RUN_PATH = (
    _ROOT
    / "configs/policy/runs/"
    "prl_15_r0_qwen3_instruct_full_rp66_bs16_n16_t1_matched_8step_gpu0123.toml"
)


def test_paired_benchmark_plan_binds_the_training_protocol_and_both_states() -> None:
    plan = json.loads(_PLAN_PATH.read_text(encoding="utf-8"))
    run = load_policy_e2e_smoke_run_config(_RUN_PATH)

    assert plan["status"] == "awaiting_formal_step8_paired_snapshot"
    assert plan["expected_task_count"] == 2511
    assert plan["protocol"] == {
        "evaluation_protocol": "training_run",
        "prompt_sha256": run.protocol.prompt_sha256,
        "tool_profile": run.protocol.tool_profile.value,
        "tool_schema_sha256": run.protocol.tool_schema_sha256,
        "maximum_tool_calls": run.protocol.maximum_tool_calls,
        "sampling_source": "bound_policy_run_config",
        "same_tasks_and_rank_partition": True,
    }
    assert [arm["optimizer_step"] for arm in plan["arms"]] == [0, 8]
    assert plan["required_pairing"]["qwen_and_rp66_optimizer_step_must_match"]
    assert plan["required_pairing"]["do_not_treat_rp66_as_policy_lora"]
    assert plan["executor"] == {
        "path": "tools/run_prl15_paired_evaluation.py",
        "snapshot_backend": "full_model_trainable_rp66",
        "supports_wait_before_training": True,
        "supports_resume": True,
        "four_gpu_schedule": "step0_then_step8",
        "eight_gpu_schedule": "step0_and_step8_parallel",
    }


def test_paired_benchmark_task_manifest_bytes_are_still_pinned() -> None:
    plan = json.loads(_PLAN_PATH.read_text(encoding="utf-8"))
    task_path = Path(plan["task_manifest_path"])
    digest = hashlib.sha256(task_path.read_bytes()).hexdigest()

    assert digest == plan["task_manifest_sha256"]
    assert sum(1 for _ in task_path.open("rb")) == plan["expected_task_count"]
