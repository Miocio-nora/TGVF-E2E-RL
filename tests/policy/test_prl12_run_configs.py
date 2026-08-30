from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tomllib

import pytest

from tgvf_rl.policy.deepeyes_strict_control import (
    DeepEyesSourceToolRoutingMode,
    DeepEyesVisualAnswerVerifierMode,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).resolve().parents[2]
_RUNS = _ROOT / "configs/policy/runs"
_PRL11 = (
    _RUNS
    / "prl_11_r0_qwen3_instruct_grpo_bs256_n16_crop_t1mixed_v2_deepeyes_scaled_20step_gpu0123.toml"
)
_PRL12_A = (
    _RUNS
    / "prl_12_a_qwen3_instruct_grpo_bs256_n16_crop_t1mixed_v2_visual_always72b_20step_gpu0123.toml"
)
_PRL12_B = (
    _RUNS
    / "prl_12_b_qwen3_instruct_grpo_bs256_n16_mixedtools_t1mixed_v2_source_routed_20step_gpu0123.toml"
)
_PRL12_A_SHA256 = "c6edbdfcc168fedaaac19ad75e94d67f6b198a6499ec34cf09722251402e0c9b"
_PRL12_B_SHA256 = "4832e0c66409ca746e59eae89ec65e51688d0deb9641af88405c17d03ae31c50"


def _toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _materialize_historical_read_copy(tmp_path: Path, source_path: Path) -> Path:
    """Rebind deleted-worktree dependencies without changing frozen evidence."""

    payload = _toml(source_path)
    framework = payload["framework"]
    reward = payload["reward"]
    assert isinstance(framework, dict)
    assert isinstance(reward, dict)
    historical_agent_loop = str(framework["agent_loop_config_path"])
    historical_judge = str(reward["judge_config_path"])
    local_agent_loop = (
        _ROOT / "configs/policy/agent_loops" / Path(historical_agent_loop).name
    )
    local_judge = _ROOT / "configs/policy/judges" / Path(historical_judge).name
    assert local_agent_loop.is_file()
    assert local_judge.is_file()
    text = source_path.read_text(encoding="utf-8")
    text = text.replace(historical_agent_loop, str(local_agent_loop), 1)
    text = text.replace(historical_judge, str(local_judge), 1)
    destination = tmp_path / source_path.name
    destination.write_text(text, encoding="utf-8")
    return destination


def test_prl12_arms_keep_all_training_and_runtime_controls_equal_to_prl11() -> None:
    reference = _toml(_PRL11)
    arm_a = _toml(_PRL12_A)
    arm_b = _toml(_PRL12_B)

    protected_tables = (
        "model",
        "dataset",
        "representation",
        "sampling",
        "optimizer",
        "scheduler",
        "precision",
        "accumulation",
        "distributed",
        "capacity",
        "training",
    )
    for table in protected_tables:
        assert arm_a[table] == reference[table]
        assert arm_b[table] == reference[table]

    assert arm_a["protocol"] == reference["protocol"]
    assert {
        key: value for key, value in arm_b["protocol"].items() if key != "prompt_sha256"
    } == {
        key: value
        for key, value in reference["protocol"].items()
        if key != "prompt_sha256"
    }
    for key in ("answer_weight", "format_weight", "conditional_tool_weight"):
        assert arm_a["reward"][key] == reference["reward"][key]
        assert arm_b["reward"][key] == reference["reward"][key]


def test_prl12_v1_is_read_only_by_default() -> None:
    with pytest.raises(ValueError, match="read-only"):
        load_policy_e2e_smoke_run_config(_PRL12_A)


def test_prl12_configs_load_with_separate_content_identities(tmp_path: Path) -> None:
    arm_a_path = _materialize_historical_read_copy(tmp_path, _PRL12_A)
    arm_b_path = _materialize_historical_read_copy(tmp_path, _PRL12_B)
    arm_a = load_policy_e2e_smoke_run_config(
        arm_a_path,
        allow_historical_read_only_contract=True,
    )
    arm_b = load_policy_e2e_smoke_run_config(
        arm_b_path,
        allow_historical_read_only_contract=True,
    )

    assert arm_a.deepeyes_control is not None
    assert arm_b.deepeyes_control is not None
    assert (
        arm_a.deepeyes_control.visual_answer_verifier
        is DeepEyesVisualAnswerVerifierMode.ALWAYS_QWEN25_72B
    )
    assert (
        arm_a.deepeyes_control.source_tool_routing
        is DeepEyesSourceToolRoutingMode.UNIFORM_CROP
    )
    assert (
        arm_b.deepeyes_control.visual_answer_verifier
        is DeepEyesVisualAnswerVerifierMode.RULE_FIRST_QWEN25_72B
    )
    assert (
        arm_b.deepeyes_control.source_tool_routing
        is DeepEyesSourceToolRoutingMode.OFFICIAL_BY_SOURCE
    )
    assert arm_a.identity_sha256 != arm_b.identity_sha256
    assert sha256(_PRL12_A.read_bytes()).hexdigest() == _PRL12_A_SHA256
    assert sha256(_PRL12_B.read_bytes()).hexdigest() == _PRL12_B_SHA256
    assert arm_a.source_sha256 == sha256(arm_a_path.read_bytes()).hexdigest()
    assert arm_b.source_sha256 == sha256(arm_b_path.read_bytes()).hexdigest()
    for arm in (arm_a, arm_b):
        assert arm.accumulation.global_prompt_batch_size == 256
        assert arm.policy.sampling.trajectories_per_prompt == 16
        assert arm.optimizer.learning_rate == 1.0e-6
        assert arm.policy.lora.rank == 64
        assert arm.policy.lora.alpha == 64
        assert arm.policy.lora.expected_target_module_count == 252
        assert (
            arm.accumulation.global_prompt_batch_size
            * arm.policy.sampling.trajectories_per_prompt
            == 4096
        )


def test_prl12_read_contract_binds_full_4096_trajectory_audit(
    tmp_path: Path,
) -> None:
    arm_a = load_policy_e2e_smoke_run_config(
        _materialize_historical_read_copy(tmp_path, _PRL12_A),
        allow_historical_read_only_contract=True,
    )
    assert arm_a.deepeyes_control is not None
    assert arm_a.deepeyes_control.as_config()["trajectory_audit_retention"] == "all"
    assert arm_a.deepeyes_control.as_config()["expected_trajectories_per_step"] == 4096


@pytest.mark.parametrize(
    ("source_path", "old_judge", "new_judge"),
    (
        (
            _PRL12_A,
            "openrouter_qwen25_72b_prl12_visual_always_v1.json",
            "openrouter_qwen25_72b_formal_pilot_judge_v4.json",
        ),
        (
            _PRL12_B,
            "openrouter_qwen25_72b_formal_pilot_judge_v4.json",
            "openrouter_qwen25_72b_prl12_visual_always_v1.json",
        ),
    ),
)
def test_prl12_rejects_judge_scope_swapped_between_arms(
    tmp_path: Path,
    source_path: Path,
    old_judge: str,
    new_judge: str,
) -> None:
    judge_root = _ROOT / "configs/policy/judges"
    old_path = judge_root / old_judge
    new_path = judge_root / new_judge
    rebound_path = _materialize_historical_read_copy(tmp_path, source_path)
    source = rebound_path.read_text(encoding="utf-8")
    source = source.replace(str(old_path), str(new_path)).replace(
        sha256(old_path.read_bytes()).hexdigest(),
        sha256(new_path.read_bytes()).hexdigest(),
    )
    invalid = tmp_path / source_path.name
    invalid.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="MCQ scope"):
        load_policy_e2e_smoke_run_config(
            invalid,
            allow_historical_read_only_contract=True,
        )
