from __future__ import annotations

import json
from pathlib import Path

from tgvf_rl.evaluation.policy_coredev import (
    CoreDevTask,
    _termination_contract,
    load_coredev_tasks,
    load_policy_coredev_config,
    policy_version_from_pointer,
)
from tgvf_rl.framework.vllm import VLLMTerminationOutcome
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_formal_policy_configs_bind_exact_step80_snapshots() -> None:
    expected = {
        "coredev_2511_tgvf_step80_v1.json": (
            "PRL-02-R5-QWEN3-GRPO-BS16-TGVF-T1-FORMAL-PILOT-80STEP-GPU0123",
            "561132e49848fd43f8e7f352ef54782249aff59b2a5d331027a0e5e0f78be321",
        ),
        "coredev_2511_crop_step80_v1.json": (
            "PRL-03-R2-QWEN3-GRPO-BS16-CROP-ONLY-FORMAL-COMPARISON-80STEP-GPU0123",
            "eed4ffeaf5b77277a41dafeba428a20d5f3c8bce73049c02e63f63292d78b0b0",
        ),
    }
    for name, (run_id, weights_sha256) in expected.items():
        config = load_policy_coredev_config(
            REPOSITORY_ROOT / "configs/evaluation" / name
        )
        assert config.inference_concurrency_per_gpu == 8
        version = policy_version_from_pointer(config)
        assert version.run_id == run_id
        assert version.optimizer_step == 80
        assert version.weights_sha256 == weights_sha256


def test_policy_evaluation_accepts_native_vllm_eos_identity() -> None:
    run = load_policy_e2e_smoke_run_config(
        REPOSITORY_ROOT
        / "configs/policy/runs/prl_02_r5_qwen3_grpo_bs16_tgvf_t1_formal_pilot_80step_gpu0123.toml"
    )

    assert VLLMTerminationOutcome("stop", None) in _termination_contract(
        run
    ).final_turn_outcomes


def test_coredev_task_loader_keeps_order_and_single_image_boundary(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    rows = [
        {
            "ordinal": index,
            "dataset": "fixture",
            "row_number": index,
            "index": str(index),
            "question": "question",
            "image_paths": ["a.jpg"] if index != 3 else ["a.jpg", "b.jpg"],
        }
        for index in range(2511)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    tasks = load_coredev_tasks(path)
    assert isinstance(tasks[0], CoreDevTask)
    assert tasks[0].single_image is True
    assert tasks[3].single_image is False
    assert tasks[-1].ordinal == 2510
