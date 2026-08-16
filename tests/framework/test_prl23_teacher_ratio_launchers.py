from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / (
    "configs/policy/runs/"
    "prl_22_a_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_"
    "teacher25_8step_ws8.toml"
)
BASE_PLAN = ROOT / (
    "configs/evaluation/"
    "prl22_a_frozen_rp67_tfree_teacher25_step8_step16_paired_seed_"
    "coredev2511_plan.json"
)
ARMS = (
    (
        50,
        ROOT
        / (
            "configs/policy/runs/"
            "prl_23_a_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_"
            "teacher50_8step_ws8.toml"
        ),
        ROOT
        / (
            "configs/evaluation/"
            "prl23_a_frozen_rp67_tfree_teacher50_step8_step16_paired_seed_"
            "coredev2511_plan.json"
        ),
    ),
    (
        100,
        ROOT
        / (
            "configs/policy/runs/"
            "prl_23_b_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_"
            "teacher100_8step_ws8.toml"
        ),
        ROOT
        / (
            "configs/evaluation/"
            "prl23_b_frozen_rp67_tfree_teacher100_step8_step16_paired_seed_"
            "coredev2511_plan.json"
        ),
    ),
)


def _config(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _plan(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_config(config: dict[str, object]) -> dict[str, object]:
    normalized = deepcopy(config)
    for key in ("run_id", "code", "dataset", "output"):
        normalized.pop(key)
    return normalized


def _normalized_plan(plan: dict[str, object]) -> dict[str, object]:
    normalized = deepcopy(plan)
    for key in ("evaluation_id", "policy_config", "policy_config_sha256"):
        normalized.pop(key)
    normalized["executor"].pop("supervisor")
    normalized["scoring"].pop("run_id_prefix")
    return normalized


def test_prl23_changes_only_dataset_and_experiment_identity_from_prl22_a() -> None:
    baseline = _normalized_config(_config(BASE_CONFIG))
    for _, config_path, _ in ARMS:
        assert _normalized_config(_config(config_path)) == baseline


def test_teacher50_and_teacher100_are_a_single_variable_ablation() -> None:
    teacher50 = _config(ARMS[0][1])
    teacher100 = _config(ARMS[1][1])
    assert _normalized_config(teacher50) == _normalized_config(teacher100)
    for percentage, config_path, _ in ARMS:
        config = _config(config_path)
        dataset = config["dataset"]
        assert dataset["kind"] == "policy_t1_teacher_ratio_mix"
        assert dataset["teacher_percentage"] == percentage
        assert dataset["sample_count"] == 20_480
        assert dataset["shuffle_seed"] == 42
        assert f"TEACHER{percentage}-MIXED-SCHEDULE-v1" in dataset["root"]
        assert config["representation"]["adapter_update_mode"] == "frozen_adapter"
        assert config["accumulation"]["global_prompt_batch_size"] == 16
        assert config["sampling"]["trajectories_per_prompt"] == 16
        assert config["distributed"]["world_size"] == 8
        assert config["optimizer"]["learning_rate"] == 1e-6


def test_prl23_evaluation_protocol_is_identical_to_prl22_a() -> None:
    baseline = _normalized_plan(_plan(BASE_PLAN))
    for _, config_path, plan_path in ARMS:
        plan = _plan(plan_path)
        assert _normalized_plan(plan) == baseline
        assert plan["policy_config"] == config_path.relative_to(ROOT).as_posix()
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert plan["policy_config_sha256"] in {"0" * 64, config_sha256}
        assert plan["paired_rng"]["temperature"] == 1.0
        assert plan["paired_rng"]["master_seed"] == 42
        assert [arm["optimizer_step"] for arm in plan["arms"]] == [8, 16]


def test_prl23_supervisors_pin_the_matched_runtime_and_unique_wandb_ids() -> None:
    supervisor = (
        ROOT / "tools/supervise_prl23_tgvf_teacher_ratio_step16_and_eval.sh"
    ).read_text(encoding="utf-8")
    assert "TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP=8" in supervisor
    assert "TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS=8" in supervisor
    assert "wandb_run_id=prl23at50" in supervisor
    assert "wandb_run_id=prl23bt100" in supervisor
    assert "--target-step 8" in supervisor
    assert "--target-step 16" in supervisor
    assert "checkpoint_is_complete 8" in supervisor
    assert "checkpoint_is_complete 16" in supervisor
