"""Strict reward and judge binding for policy run configurations.

All effect-capable loader callables are supplied by the public run-config
facade.  This leaf does not import judges, parse TOML, or launch work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tgvf_rl.data import PolicyT1MixedRuntimeBinding
from tgvf_rl.protocol import NativeToolCapabilityProfile

from .run_config_schema import (
    POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER,
    POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_JUDGE_MODE,
    POLICY_E2E_MIXED_ANSWER_VERIFIER,
    POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_MIXED_ANSWER_VERIFIER_V1_SHA256,
    POLICY_E2E_MIXED_JUDGE_MODE,
    POLICY_E2E_MIXED_REWARD_TASK,
    POLICY_E2E_SMOKE_ANSWER_VERIFIER,
    POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_SMOKE_ANSWER_VERIFIER_V2_SHA256,
    POLICY_E2E_SMOKE_JUDGE_MODE,
    POLICY_E2E_SMOKE_REWARD_TASK,
    SmokeRewardBinding,
)
from .run_config_validation import (
    _existing_file,
    _real,
    _require_exact,
    _sha256,
    _sha256_file,
    _table,
    _text,
    _validate_deepeyes_strict_judge,
)


def bind_policy_reward(
    payload: Mapping[str, object],
    *,
    allow_historical_reward_contract: bool,
    deepeyes_control_present: bool,
    formal_pilot: bool,
    iteration_sha256: str,
    mixed_run: bool,
    runtime_binding: object,
    stage3_shaped_reward_version: str,
    stage3_shaped_run: bool,
    tool_profile: NativeToolCapabilityProfile,
    visual_always_judge: bool,
    pilot_reward_weight_profile_name: Callable[..., Any],
    load_openai_compatible_judge: Callable[..., Any],
    load_tgvf_tool_utility_runtime_binding: Callable[..., Any],
    load_tgvf_visual_quality_judge: Callable[..., Any],
) -> SmokeRewardBinding:
    """Validate and bind reward inputs without enabling external calls."""

    reward_fields = {
        "task_kind",
        "answer_verifier",
        "answer_verifier_sha256",
        "judge_mode",
        "judge_reason",
    }
    if mixed_run:
        reward_fields.update({"judge_config_path", "judge_config_sha256"})
    if stage3_shaped_run:
        reward_fields.update(
            {
                "profile",
                "tool_utility_sidecar_path",
                "tool_utility_sidecar_sha256",
                "tool_utility_manifest_path",
                "tool_utility_manifest_sha256",
                "visual_quality_judge_config_path",
                "visual_quality_judge_config_sha256",
            }
        )
    else:
        reward_fields.update(
            {"answer_weight", "format_weight", "conditional_tool_weight"}
        )
    reward_table = _table(
        payload,
        "reward",
        reward_fields,
    )
    expected_task = (
        POLICY_E2E_MIXED_REWARD_TASK if mixed_run else POLICY_E2E_SMOKE_REWARD_TASK
    )
    if visual_always_judge:
        expected_verifier = POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER
        expected_judge_mode = POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_JUDGE_MODE
    else:
        expected_verifier = (
            POLICY_E2E_MIXED_ANSWER_VERIFIER
            if mixed_run
            else POLICY_E2E_SMOKE_ANSWER_VERIFIER
        )
        expected_judge_mode = (
            POLICY_E2E_MIXED_JUDGE_MODE if mixed_run else POLICY_E2E_SMOKE_JUDGE_MODE
        )
    _require_exact(reward_table["task_kind"], expected_task, "reward.task_kind")
    _require_exact(
        reward_table["answer_verifier"],
        expected_verifier,
        "reward.answer_verifier",
    )
    _require_exact(reward_table["judge_mode"], expected_judge_mode, "reward.judge_mode")
    answer_verifier_sha256 = _sha256(
        reward_table["answer_verifier_sha256"],
        name="reward.answer_verifier_sha256",
    )
    if visual_always_judge:
        current_answer_verifier_sha256 = (
            POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER_SHA256
        )
    else:
        current_answer_verifier_sha256 = (
            POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256
            if mixed_run
            else POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256
        )
    historical_answer_verifier_sha256 = (
        POLICY_E2E_MIXED_ANSWER_VERIFIER_V1_SHA256
        if mixed_run
        else POLICY_E2E_SMOKE_ANSWER_VERIFIER_V2_SHA256
    )
    accepted_answer_verifier_sha256s = {current_answer_verifier_sha256}
    if allow_historical_reward_contract and not deepeyes_control_present:
        accepted_answer_verifier_sha256s.add(historical_answer_verifier_sha256)
    if answer_verifier_sha256 not in accepted_answer_verifier_sha256s:
        raise ValueError(
            "reward.answer_verifier_sha256 differs from the current contract"
            + (
                " and the named historical evaluation contract"
                if allow_historical_reward_contract
                else ""
            )
        )
    if mixed_run:
        judge_config_path = _existing_file(
            reward_table["judge_config_path"], name="reward.judge_config_path"
        )
        if judge_config_path.is_symlink():
            raise ValueError("reward judge config must not be a symlink")
        judge_config_sha256 = _sha256(
            reward_table["judge_config_sha256"],
            name="reward.judge_config_sha256",
        )
        if _sha256_file(judge_config_path) != judge_config_sha256:
            raise ValueError("reward judge config SHA256 mismatch")
        if deepeyes_control_present:
            _validate_deepeyes_strict_judge(
                judge_config_path,
                judge_config_sha256=judge_config_sha256,
                visual_always=visual_always_judge,
            )
        else:
            bound_judge = load_openai_compatible_judge(
                judge_config_path,
                expected_file_sha256=judge_config_sha256,
            )
            if formal_pilot and not bound_judge.formal_pilot_accepted:
                raise ValueError("reward judge is not accepted for the formal Pilot")
    else:
        judge_config_path = None
        judge_config_sha256 = None
    if stage3_shaped_run:
        _require_exact(
            reward_table["profile"],
            stage3_shaped_reward_version,
            "reward.profile",
        )
        if not isinstance(runtime_binding, PolicyT1MixedRuntimeBinding):
            raise ValueError("Stage3-shaped reward requires the mixed-v2 T1 dataset")
        if tool_profile is not NativeToolCapabilityProfile.TGVF_ONLY:
            raise ValueError("Stage3-shaped reward requires the TGVF-only tool profile")
        sidecar_path = _existing_file(
            reward_table["tool_utility_sidecar_path"],
            name="reward.tool_utility_sidecar_path",
        )
        sidecar_sha256 = _sha256(
            reward_table["tool_utility_sidecar_sha256"],
            name="reward.tool_utility_sidecar_sha256",
        )
        sidecar_manifest_path = _existing_file(
            reward_table["tool_utility_manifest_path"],
            name="reward.tool_utility_manifest_path",
        )
        sidecar_manifest_sha256 = _sha256(
            reward_table["tool_utility_manifest_sha256"],
            name="reward.tool_utility_manifest_sha256",
        )
        tool_utility = load_tgvf_tool_utility_runtime_binding(
            sidecar_path,
            expected_sidecar_sha256=sidecar_sha256,
            manifest_path=sidecar_manifest_path,
            expected_manifest_sha256=sidecar_manifest_sha256,
            expected_dataset_iteration_identity_sha256=iteration_sha256,
        )
        visual_quality_config_path = _existing_file(
            reward_table["visual_quality_judge_config_path"],
            name="reward.visual_quality_judge_config_path",
        )
        visual_quality_config_sha256 = _sha256(
            reward_table["visual_quality_judge_config_sha256"],
            name="reward.visual_quality_judge_config_sha256",
        )
        if _sha256_file(visual_quality_config_path) != visual_quality_config_sha256:
            raise ValueError("reward visual-quality judge config SHA256 mismatch")
        bound_visual_quality_judge = load_tgvf_visual_quality_judge(
            visual_quality_config_path,
            expected_file_sha256=visual_quality_config_sha256,
        )
        visual_quality_judge_identity = bound_visual_quality_judge.config_identity
        reward_profile = stage3_shaped_reward_version
        reward_weights: tuple[float, float, float] | None = None
    else:
        reward_weights = (
            _real(reward_table["answer_weight"], name="reward.answer_weight"),
            _real(reward_table["format_weight"], name="reward.format_weight"),
            _real(
                reward_table["conditional_tool_weight"],
                name="reward.conditional_tool_weight",
            ),
        )
        pilot_reward_weight_profile_name(reward_weights)
        reward_profile = "pilot-v1"
        tool_utility = None
        visual_quality_config_path = None
        visual_quality_config_sha256 = None
        visual_quality_judge_identity = None
    reward = SmokeRewardBinding(
        profile=reward_profile,
        task_kind=reward_table["task_kind"],
        answer_verifier=reward_table["answer_verifier"],
        answer_verifier_sha256=answer_verifier_sha256,
        judge_mode=reward_table["judge_mode"],
        judge_reason=_text(reward_table["judge_reason"], name="reward.judge_reason"),
        answer_weight=None if reward_weights is None else reward_weights[0],
        format_weight=None if reward_weights is None else reward_weights[1],
        conditional_tool_weight=None if reward_weights is None else reward_weights[2],
        judge_config_path=judge_config_path,
        judge_config_sha256=judge_config_sha256,
        tool_utility=tool_utility,
        visual_quality_judge_config_path=visual_quality_config_path,
        visual_quality_judge_config_sha256=visual_quality_config_sha256,
        visual_quality_judge_identity=visual_quality_judge_identity,
    )
    return reward


__all__ = ["bind_policy_reward"]
