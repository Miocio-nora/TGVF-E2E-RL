"""Strict reward and judge binding for policy run configurations.

All effect-capable loader callables are supplied by the public run-config
facade.  This leaf does not import judges, parse TOML, or launch work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.data import PolicyT1MixedRuntimeBinding
from tgvf_rl.protocol import NativeToolCapabilityProfile

from .run_config_schema import (
    POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER,
    POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_JUDGE_MODE,
    POLICY_E2E_MIXED_ALTERNATE_ANSWER_VERIFIER,
    POLICY_E2E_MIXED_ALTERNATE_JUDGE_MODE,
    POLICY_E2E_MIXED_ANSWER_VERIFIER,
    POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_MIXED_ANSWER_VERIFIER_V1_SHA256,
    POLICY_E2E_MIXED_JUDGE_MODE,
    POLICY_E2E_MIXED_REWARD_TASK,
    POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA,
    POLICY_E2E_SMOKE_ANSWER_VERIFIER,
    POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_SMOKE_ANSWER_VERIFIER_V2_SHA256,
    POLICY_E2E_SMOKE_JUDGE_MODE,
    POLICY_E2E_SMOKE_REWARD_TASK,
    PolicyMethodMatrixBinding,
    SmokeRewardBinding,
    policy_e2e_mixed_alternate_answer_verifier_sha256,
)
from .run_config_validation import (
    _boolean,
    _existing_file,
    _real,
    _nonnegative_real,
    _require_exact,
    _sha256,
    _sha256_file,
    _table,
    _text,
    _validate_deepeyes_strict_judge,
)


# Read-only compatibility semantics for Stage3 TOMLs that predate explicit
# coefficient fields. New method-matrix runs bind all three values from TOML.
_LEGACY_STAGE3_ANSWER_REWARD_SCALE = 2.0
_LEGACY_STAGE3_REPEATED_CALL_PENALTY = 0.05
_LEGACY_STAGE3_PROTOCOL_ERROR_PENALTY = 1.0
_QWEN25_72B_SERVED_NAMES = frozenset(
    {
        "qwen2.5-72b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen2.5-72b-instruct",
    }
)
_ALTERNATE_JUDGE_FIELDS = frozenset(
    {
        "alternate_judge_model_name",
        "alternate_judge_model_identity_sha256",
        "alternate_semantics_acknowledged",
    }
)


def bind_policy_reward(
    payload: Mapping[str, object],
    *,
    allow_historical_reward_contract: bool,
    deepeyes_control_present: bool,
    formal_pilot: bool,
    iteration_sha256: str,
    method_binding: PolicyMethodMatrixBinding | None,
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

    method_run = method_binding is not None
    explicit_judge_route_contract = payload.get("schema_version") == (
        POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA
    )

    reward_fields = {
        "task_kind",
        "answer_verifier",
        "answer_verifier_sha256",
        "judge_mode",
        "judge_reason",
    }
    if mixed_run:
        reward_fields.update({"judge_config_path", "judge_config_sha256"})
    if explicit_judge_route_contract:
        reward_fields.add("judge_model_route")
        raw_reward = payload.get("reward")
        if isinstance(raw_reward, Mapping):
            reward_fields.update(_ALTERNATE_JUDGE_FIELDS & set(raw_reward))
    if stage3_shaped_run:
        reward_fields.add("profile")
        if method_run:
            reward_fields.update(
                {
                    "answer_reward_scale",
                    "protocol_error_penalty",
                    "repeated_call_penalty",
                    "tool_utility_reward_enabled",
                    "focus_reward_enabled",
                    "grounding_reward_enabled",
                    "visual_quality_judge_mode",
                }
            )
        else:
            reward_fields.update(
                {
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
    judge_model_route = (
        _text(reward_table["judge_model_route"], name="reward.judge_model_route")
        if explicit_judge_route_contract
        else "qwen2.5_72b"
    )
    if judge_model_route not in {"qwen2.5_72b", "explicit_alternate"}:
        raise ValueError(
            "reward.judge_model_route must be qwen2.5_72b or explicit_alternate"
        )
    present_alternate_fields = _ALTERNATE_JUDGE_FIELDS & set(reward_table)
    expected_task = (
        POLICY_E2E_MIXED_REWARD_TASK if mixed_run else POLICY_E2E_SMOKE_REWARD_TASK
    )
    if judge_model_route == "explicit_alternate":
        expected_verifier = POLICY_E2E_MIXED_ALTERNATE_ANSWER_VERIFIER
        expected_judge_mode = POLICY_E2E_MIXED_ALTERNATE_JUDGE_MODE
    elif visual_always_judge:
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
    bound_judge: object | None = None
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
    alternate_judge_model_name: str | None = None
    alternate_judge_model_identity: ArtifactIdentity | None = None
    alternate_semantics_acknowledged = False
    if judge_model_route == "qwen2.5_72b":
        if present_alternate_fields:
            raise ValueError(
                "default Qwen2.5-72B route cannot carry alternate judge fields"
            )
        if bound_judge is not None:
            bound_name = bound_judge.provider.config.model_name
            if (
                not isinstance(bound_name, str)
                or bound_name.strip().casefold() not in _QWEN25_72B_SERVED_NAMES
            ):
                raise ValueError("default reward judge binding must remain Qwen2.5-72B")
    else:
        if present_alternate_fields != _ALTERNATE_JUDGE_FIELDS:
            raise ValueError(
                "explicit alternate judge requires model name, identity SHA256, "
                "and semantic acknowledgement"
            )
        if not mixed_run or deepeyes_control_present or bound_judge is None:
            raise ValueError(
                "explicit alternate judge requires a directly loaded mixed-run judge"
            )
        alternate_judge_model_name = _text(
            reward_table["alternate_judge_model_name"],
            name="reward.alternate_judge_model_name",
        )
        declared_alternate_sha256 = _sha256(
            reward_table["alternate_judge_model_identity_sha256"],
            name="reward.alternate_judge_model_identity_sha256",
        )
        alternate_semantics_acknowledged = _boolean(
            reward_table["alternate_semantics_acknowledged"],
            name="reward.alternate_semantics_acknowledged",
        )
        if not alternate_semantics_acknowledged:
            raise ValueError(
                "explicit alternate judge requires semantic acknowledgement"
            )
        bound_name = bound_judge.provider.config.model_name
        bound_identity = bound_judge.model_identity
        if bound_name != alternate_judge_model_name:
            raise ValueError("explicit alternate judge model name differs")
        if not isinstance(bound_identity, ArtifactIdentity):
            raise TypeError("loaded alternate judge model identity has the wrong type")
        if bound_identity.sha256 != declared_alternate_sha256:
            raise ValueError("explicit alternate judge model identity differs")
        alternate_judge_model_identity = bound_identity
    if judge_model_route == "explicit_alternate":
        assert alternate_judge_model_identity is not None
        current_answer_verifier_sha256 = (
            policy_e2e_mixed_alternate_answer_verifier_sha256(
                alternate_judge_model_identity
            )
        )
        accepted_answer_verifier_sha256s = {current_answer_verifier_sha256}
    else:
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
        mismatch = (
            "reward.answer_verifier_sha256 differs from the selected "
            "alternate model-bound contract"
            if judge_model_route == "explicit_alternate"
            else "reward.answer_verifier_sha256 differs from the current contract"
        )
        raise ValueError(
            mismatch
            + (
                " and the named historical evaluation contract"
                if (
                    allow_historical_reward_contract
                    and judge_model_route == "qwen2.5_72b"
                )
                else ""
            )
        )
    if stage3_shaped_run:
        _require_exact(
            reward_table["profile"],
            stage3_shaped_reward_version,
            "reward.profile",
        )
        if method_run:
            tool_utility_reward_enabled = _boolean(
                reward_table["tool_utility_reward_enabled"],
                name="reward.tool_utility_reward_enabled",
            )
            focus_reward_enabled = _boolean(
                reward_table["focus_reward_enabled"],
                name="reward.focus_reward_enabled",
            )
            grounding_reward_enabled = _boolean(
                reward_table["grounding_reward_enabled"],
                name="reward.grounding_reward_enabled",
            )
            _require_exact(
                (
                    tool_utility_reward_enabled,
                    focus_reward_enabled,
                    grounding_reward_enabled,
                    _text(
                        reward_table["visual_quality_judge_mode"],
                        name="reward.visual_quality_judge_mode",
                    ),
                ),
                (False, False, False, "disabled"),
                "method answer/protocol-only reward",
            )
            answer_reward_scale = _nonnegative_real(
                reward_table["answer_reward_scale"],
                name="reward.answer_reward_scale",
            )
            repeated_call_penalty = _nonnegative_real(
                reward_table["repeated_call_penalty"],
                name="reward.repeated_call_penalty",
            )
            protocol_error_penalty = _nonnegative_real(
                reward_table["protocol_error_penalty"],
                name="reward.protocol_error_penalty",
            )
            visual_quality_judge_mode = "disabled"
            tool_utility = None
            visual_quality_config_path = None
            visual_quality_config_sha256 = None
            visual_quality_judge_identity = None
        else:
            if not isinstance(runtime_binding, PolicyT1MixedRuntimeBinding):
                raise ValueError(
                    "Stage3-shaped reward requires the mixed-v2 T1 dataset"
                )
            if tool_profile is not NativeToolCapabilityProfile.TGVF_ONLY:
                raise ValueError(
                    "Stage3-shaped reward requires the TGVF-only tool profile"
                )
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
            tool_utility_reward_enabled = True
            focus_reward_enabled = True
            grounding_reward_enabled = True
            answer_reward_scale = _LEGACY_STAGE3_ANSWER_REWARD_SCALE
            repeated_call_penalty = _LEGACY_STAGE3_REPEATED_CALL_PENALTY
            protocol_error_penalty = _LEGACY_STAGE3_PROTOCOL_ERROR_PENALTY
            visual_quality_judge_mode = "configured"
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
        tool_utility_reward_enabled = None
        focus_reward_enabled = None
        grounding_reward_enabled = None
        answer_reward_scale = None
        repeated_call_penalty = None
        protocol_error_penalty = None
        visual_quality_judge_mode = None
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
        protocol_error_penalty=protocol_error_penalty,
        answer_reward_scale=answer_reward_scale,
        repeated_call_penalty=repeated_call_penalty,
        judge_config_path=judge_config_path,
        judge_config_sha256=judge_config_sha256,
        tool_utility=tool_utility,
        tool_utility_reward_enabled=tool_utility_reward_enabled,
        focus_reward_enabled=focus_reward_enabled,
        grounding_reward_enabled=grounding_reward_enabled,
        visual_quality_judge_config_path=visual_quality_config_path,
        visual_quality_judge_config_sha256=visual_quality_config_sha256,
        visual_quality_judge_identity=visual_quality_judge_identity,
        visual_quality_judge_mode=visual_quality_judge_mode,
        judge_model_route=judge_model_route,
        alternate_judge_model_name=alternate_judge_model_name,
        alternate_judge_model_identity=alternate_judge_model_identity,
        alternate_semantics_acknowledged=alternate_semantics_acknowledged,
    )
    return reward


__all__ = ["bind_policy_reward"]
