from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from tgvf_rl.contracts.identity import CodeIdentity
from tgvf_rl.policy.config import PolicyMethodProfile
from tgvf_rl.policy.method_matrix_validation import (
    DEFAULT_REQUIRED_METHOD_PROFILES,
    MethodMatrixValidationError,
    validate_policy_method_matrix,
)
from tgvf_rl.policy.run_config_schema import (
    POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA,
    PolicyE2ESmokeRunConfig,
    PolicyMethodMatrixBinding,
)


_TOOL_PROFILE = {
    PolicyMethodProfile.NO_TOOL: "no_tool",
    PolicyMethodProfile.CROP: "crop_only",
    PolicyMethodProfile.TGVF_SHORT: "tgvf_only",
    PolicyMethodProfile.TGVF_TARGET_GUIDE_V2: "tgvf_only",
    PolicyMethodProfile.ATOMIC: "crop_tgvf",
}


def _arm(
    profile: PolicyMethodProfile,
    *,
    image_max_pixels: int,
    matrix_id: str = "configurable-resolution-comparison",
) -> PolicyE2ESmokeRunConfig:
    tool_profile = _TOOL_PROFILE[profile]
    tool_arm = profile is not PolicyMethodProfile.NO_TOOL
    adapter_arm = profile in {
        PolicyMethodProfile.TGVF_SHORT,
        PolicyMethodProfile.TGVF_TARGET_GUIDE_V2,
        PolicyMethodProfile.ATOMIC,
    }
    maximum_tool_calls = 6 if tool_arm else 1
    source_path = Path(f"/runs/configs/{profile.value}.toml")
    return PolicyE2ESmokeRunConfig(
        run_id=f"run-{profile.value}",
        code=CodeIdentity(
            repository="example/repository",
            commit="1" * 40,
        ),
        model={
            "family": "qwen3_vl",
            "revision": "qwen3-vl-instruct",
            "image_max_pixels": image_max_pixels,
        },
        dataset={
            "kind": "teacher-quarter-mix",
            "iteration_identity_sha256": "dataset-identity",
        },
        representation={
            "artifact": "shared-adapter-artifact",
            "adapter_update_mode": ("frozen_adapter" if adapter_arm else "joint"),
        },
        protocol={
            "prompt_sha256": f"prompt-{profile.value}",
            "cap_error_sha256": f"cap-{maximum_tool_calls}",
            "tool_profile": tool_profile,
            "tool_schema_sha256": f"schema-{tool_profile}",
            "enabled_tool_names": [tool_profile],
            "maximum_tool_calls": maximum_tool_calls,
            "success_observation_protocol_id": f"observation-{profile.value}",
            "action_boundary_protocol_id": "strict-single-terminal-v2",
        },
        policy={
            "method": profile,
            "tool_profile": tool_profile,
            "enabled_tool_names": [tool_profile],
            "max_tgvf_call_attempts": maximum_tool_calls,
            "image_max_pixels": image_max_pixels,
            "sampling": {
                "trajectories_per_prompt": 16,
                "max_response_length": 20_480,
                "stop_strings": ["</tool_call>"] if tool_arm else [],
                "include_stop_str_in_output": True,
            },
        },
        rollout_rng={"master_seed": 42, "derivation": "content-addressed"},
        reward={
            "profile": "answer-protocol-only",
            "judge_reason": f"reason for {profile.value}",
            "answer_reward_scale": 1.0,
        },
        optimizer={"name": "adamw", "learning_rate": 1.0e-6},
        scheduler={"name": "constant", "total_steps": 32},
        precision={"parameter_dtype": "bf16"},
        accumulation={
            "global_prompt_batch_size": 16,
            "prompt_micro_batch_size_per_rank": 2,
        },
        distributed={
            "world_size": 8,
            "weight_sync_mode": (
                "full-qwen-plus-adapter" if adapter_arm else "full-qwen"
            ),
        },
        capacity={"vllm_max_model_len": 32_768},
        framework={
            "agent_loop": "tgvf-native-policy-v1",
            "agent_loop_config_sha256": "shared-agent-loop-identity",
        },
        training={
            "maximum_optimizer_steps": 32,
            "checkpoint_steps": [16, 32],
            "resume_mode": "disabled",
            "resume_from_path": Path(f"/runs/{profile.value}/checkpoints"),
        },
        output={
            "root": Path(f"/runs/{profile.value}"),
            "metrics": Path(f"/runs/{profile.value}/metrics.jsonl"),
        },
        source_path=source_path,
        source_sha256=f"source-{profile.value}",
        canonical_json=json.dumps({"source": str(source_path)}),
        method=PolicyMethodMatrixBinding(matrix_id=matrix_id, profile=profile),
        schema_version=POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA,
    )


def _matrix(*, image_max_pixels: int) -> tuple[PolicyE2ESmokeRunConfig, ...]:
    return tuple(
        _arm(profile, image_max_pixels=image_max_pixels)
        for profile in DEFAULT_REQUIRED_METHOD_PROFILES
    )


def _replace_mapping_path(
    arm: PolicyE2ESmokeRunConfig,
    path: str,
    value: object,
) -> PolicyE2ESmokeRunConfig:
    root, *children = path.split(".")
    root_value = getattr(arm, root)
    if not isinstance(root_value, dict):
        raise TypeError("test helper expects mapping-backed config sections")

    def update(mapping: dict[str, object], remaining: list[str]) -> dict[str, object]:
        result = dict(mapping)
        key = remaining[0]
        if len(remaining) == 1:
            result[key] = value
        else:
            child = result[key]
            if not isinstance(child, dict):
                raise TypeError("test helper expects nested mappings")
            result[key] = update(child, remaining[1:])
        return result

    return replace(arm, **{root: update(root_value, children)})


def test_valid_matrix_returns_order_independent_auditable_fingerprint() -> None:
    arms = _matrix(image_max_pixels=262_144)

    receipt = validate_policy_method_matrix(arms)
    reordered = validate_policy_method_matrix(reversed(arms))

    assert receipt.matrix_id == "configurable-resolution-comparison"
    assert receipt.required_profiles == DEFAULT_REQUIRED_METHOD_PROFILES
    assert len(receipt.shared_fingerprint_sha256) == 64
    assert receipt.shared_fingerprint_sha256 == reordered.shared_fingerprint_sha256
    assert "model.image_max_pixels" in receipt.shared_leaf_paths
    assert "policy.sampling.trajectories_per_prompt" in receipt.shared_leaf_paths
    assert "training.maximum_optimizer_steps" in receipt.shared_leaf_paths
    assert "rollout_rng.master_seed" in receipt.shared_leaf_paths
    record = json.loads(receipt.shared_canonical_json)
    assert record["shared_controls"]["model"]["image_max_pixels"] == 262_144
    assert "output" not in record["shared_controls"]


def test_uniform_resolution_change_is_config_selected_and_accepted() -> None:
    pixel512 = validate_policy_method_matrix(_matrix(image_max_pixels=262_144))
    another_resolution = validate_policy_method_matrix(
        _matrix(image_max_pixels=589_824)
    )

    assert another_resolution.shared_fingerprint_sha256 != (
        pixel512.shared_fingerprint_sha256
    )
    assert (
        json.loads(another_resolution.shared_canonical_json)["shared_controls"][
            "model"
        ]["image_max_pixels"]
        == 589_824
    )


def test_changing_resolution_on_only_one_arm_is_rejected_with_paths() -> None:
    arms = list(_matrix(image_max_pixels=262_144))
    arms[-1] = _replace_mapping_path(arms[-1], "model.image_max_pixels", 589_824)
    arms[-1] = _replace_mapping_path(arms[-1], "policy.image_max_pixels", 589_824)

    with pytest.raises(MethodMatrixValidationError) as captured:
        validate_policy_method_matrix(arms)

    assert captured.value.mismatch_paths == (
        "model.image_max_pixels",
        "policy.image_max_pixels",
    )
    assert captured.value.mismatches[0].actual_profile is PolicyMethodProfile.ATOMIC


@pytest.mark.parametrize(
    ("path", "changed_value"),
    (
        ("policy.sampling.trajectories_per_prompt", 8),
        ("accumulation.global_prompt_batch_size", 32),
        ("training.maximum_optimizer_steps", 80),
        ("rollout_rng.master_seed", 43),
        ("reward.answer_reward_scale", 2.0),
        ("dataset.iteration_identity_sha256", "different-dataset"),
        ("framework.agent_loop_config_sha256", "different-agent-loop"),
        ("capacity.vllm_max_model_len", 65_536),
        ("precision.parameter_dtype", "fp32"),
        ("optimizer.learning_rate", 3.0e-6),
    ),
)
def test_one_arm_control_drift_is_rejected(
    path: str,
    changed_value: object,
) -> None:
    arms = list(_matrix(image_max_pixels=262_144))
    arms[1] = _replace_mapping_path(arms[1], path, changed_value)

    with pytest.raises(MethodMatrixValidationError) as captured:
        validate_policy_method_matrix(arms)

    assert captured.value.mismatch_paths == (path,)


def test_one_arm_code_commit_drift_is_rejected() -> None:
    arms = list(_matrix(image_max_pixels=262_144))
    arms[2] = replace(
        arms[2],
        code=CodeIdentity(
            repository=arms[2].code.repository,
            commit="2" * 40,
        ),
    )

    with pytest.raises(MethodMatrixValidationError) as captured:
        validate_policy_method_matrix(arms)

    assert captured.value.mismatch_paths == ("code.commit",)


def test_required_profiles_can_select_a_unique_diagnostic_subset() -> None:
    required = (
        PolicyMethodProfile.NO_TOOL,
        PolicyMethodProfile.CROP,
    )
    receipt = validate_policy_method_matrix(
        _matrix(image_max_pixels=262_144)[:2],
        required_profiles=required,
    )

    assert receipt.required_profiles == required


def test_missing_duplicate_and_cross_matrix_arms_are_explicit() -> None:
    arms = list(_matrix(image_max_pixels=262_144))
    missing = arms[:-1]
    with pytest.raises(MethodMatrixValidationError) as captured_missing:
        validate_policy_method_matrix(missing)
    assert captured_missing.value.mismatch_paths == ("$profiles.atomic",)

    duplicate = [*arms, arms[0]]
    with pytest.raises(MethodMatrixValidationError) as captured_duplicate:
        validate_policy_method_matrix(duplicate)
    assert "$profiles.no_tool" in captured_duplicate.value.mismatch_paths

    arms[-1] = replace(
        arms[-1],
        method=PolicyMethodMatrixBinding(
            matrix_id="different-matrix",
            profile=PolicyMethodProfile.ATOMIC,
        ),
    )
    with pytest.raises(MethodMatrixValidationError) as captured_matrix:
        validate_policy_method_matrix(arms)
    assert captured_matrix.value.mismatch_paths == ("arms[atomic].method.matrix_id",)
