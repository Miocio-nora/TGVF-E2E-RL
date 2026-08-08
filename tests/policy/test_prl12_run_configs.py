from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tomllib

import pytest

from tgvf_rl.framework.verl import policy_task_runner
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


def _toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


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


def test_prl12_configs_load_with_separate_content_identities() -> None:
    arm_a = load_policy_e2e_smoke_run_config(_PRL12_A)
    arm_b = load_policy_e2e_smoke_run_config(_PRL12_B)

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
    assert arm_a.source_sha256 == sha256(_PRL12_A.read_bytes()).hexdigest()
    assert arm_b.source_sha256 == sha256(_PRL12_B.read_bytes()).hexdigest()
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


def test_prl12_actor_boundary_binds_full_4096_trajectory_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm_a = load_policy_e2e_smoke_run_config(_PRL12_A)
    observed: list[tuple[int, int | None, bool]] = []

    def capture(writer, optimizer_step: int) -> None:
        observed.append(
            (
                optimizer_step,
                writer.expected_trajectories_per_step,
                writer.retain_all_trajectories,
            )
        )

    monkeypatch.setattr(
        policy_task_runner.PolicyTrajectoryAuditWriter,
        "assert_step_complete",
        capture,
    )

    policy_task_runner._assert_strict_trajectory_audit_complete(
        arm_a,
        behavior_optimizer_step=7,
    )

    assert observed == [(7, 4096, True)]


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
    source = source_path.read_text()
    source = source.replace(str(old_path), str(new_path)).replace(
        sha256(old_path.read_bytes()).hexdigest(),
        sha256(new_path.read_bytes()).hexdigest(),
    )
    invalid = tmp_path / source_path.name
    invalid.write_text(source)

    with pytest.raises(ValueError, match="MCQ scope"):
        load_policy_e2e_smoke_run_config(invalid)
