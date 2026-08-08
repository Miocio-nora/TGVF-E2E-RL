from __future__ import annotations

import pytest

from tgvf_rl.policy.deepeyes_strict_control import (
    DeepEyesSourceToolRoutingMode,
    DeepEyesStrictControlBinding,
    DeepEyesVisualAnswerVerifierMode,
)
from tgvf_rl.protocol import NativeAssistantDialect, NativeToolCapabilityProfile
from tgvf_rl.rewards.schema import (
    PILOT_REWARD_EQUATION_DEEPEYES_MATH,
    PILOT_REWARD_EQUATION_DEEPEYES_VISUAL,
)


def _control(*, arm: str) -> DeepEyesStrictControlBinding:
    values = {
        "a": {
            "visual_answer_verifier": "always_qwen25_72b",
            "source_tool_routing": "uniform_crop",
            "trajectory_audit_retention": "all",
            "expected_trajectories_per_step": 4096,
        },
        "b": {
            "visual_answer_verifier": "rule_first_qwen25_72b",
            "source_tool_routing": "official_by_source",
            "trajectory_audit_retention": "all",
            "expected_trajectories_per_step": 4096,
        },
    }
    return DeepEyesStrictControlBinding.from_mapping(values[arm])


def test_prl12_a_changes_only_visual_answer_verification() -> None:
    control = _control(arm="a")

    vstar = control.route("vstar", "open")
    arxiv = control.route("arxivqa", "mcq")
    thinklite = control.route("thinklite", "open")

    assert vstar.official_source == "vstar"
    assert arxiv.official_source == "chart"
    assert thinklite.official_source == "thinklite_eureka"
    assert vstar.reward_equation == PILOT_REWARD_EQUATION_DEEPEYES_VISUAL
    assert arxiv.reward_equation == PILOT_REWARD_EQUATION_DEEPEYES_VISUAL
    assert thinklite.reward_equation == PILOT_REWARD_EQUATION_DEEPEYES_MATH
    assert vstar.always_judge_answer and arxiv.always_judge_answer
    assert not thinklite.always_judge_answer
    assert {
        vstar.tool_profile,
        arxiv.tool_profile,
        thinklite.tool_profile,
    } == {NativeToolCapabilityProfile.CROP_ONLY}


@pytest.mark.parametrize("task_kind", ("math", "open", "mcq"))
def test_prl12_b_routes_every_local_thinklite_kind_explicitly_no_tool(
    task_kind: str,
) -> None:
    control = _control(arm="b")

    route = control.route("thinklite", task_kind)

    assert route.official_source == "thinklite_eureka"
    assert route.reward_equation == PILOT_REWARD_EQUATION_DEEPEYES_MATH
    assert route.tool_profile is None
    assert not route.always_judge_answer


@pytest.mark.parametrize(
    ("source", "task_kind"),
    (
        ("unknown", "open"),
        ("vstar", "mcq"),
        ("arxivqa", "open"),
        ("thinklite", "unknown"),
    ),
)
def test_strict_source_task_routing_fails_closed(source: str, task_kind: str) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _control(arm="b").route(source, task_kind)


@pytest.mark.parametrize(
    "mapping",
    (
        {
            "visual_answer_verifier": "always_qwen25_72b",
            "source_tool_routing": "official_by_source",
            "trajectory_audit_retention": "all",
            "expected_trajectories_per_step": 4096,
        },
        {
            "visual_answer_verifier": "rule_first_qwen25_72b",
            "source_tool_routing": "uniform_crop",
            "trajectory_audit_retention": "all",
            "expected_trajectories_per_step": 4096,
        },
    ),
)
def test_strict_controls_reject_combined_or_noop_arms(
    mapping: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="PRL12-A or source/tool-routing-only PRL12-B"):
        DeepEyesStrictControlBinding.from_mapping(mapping)


def test_source_routed_prompt_identity_separates_visual_and_thinklite() -> None:
    control = _control(arm="b")
    dialect = NativeAssistantDialect.QWEN3_VL_INSTRUCT

    visual = control.prompt_sha256_for_source(
        "vstar", "open", assistant_dialect=dialect
    )
    thinklite = control.prompt_sha256_for_source(
        "thinklite", "open", assistant_dialect=dialect
    )

    assert visual != thinklite
    assert control.prompt_bundle_sha256(dialect) not in {visual, thinklite}


def test_control_enum_values_remain_the_two_pre_registered_axes() -> None:
    assert set(DeepEyesVisualAnswerVerifierMode) == {
        DeepEyesVisualAnswerVerifierMode.RULE_FIRST_QWEN25_72B,
        DeepEyesVisualAnswerVerifierMode.ALWAYS_QWEN25_72B,
    }
    assert set(DeepEyesSourceToolRoutingMode) == {
        DeepEyesSourceToolRoutingMode.UNIFORM_CROP,
        DeepEyesSourceToolRoutingMode.OFFICIAL_BY_SOURCE,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("trajectory_audit_retention", "selected"),
        ("expected_trajectories_per_step", 2837),
    ),
)
def test_strict_control_rejects_incomplete_trajectory_audit_contract(
    field: str,
    value: object,
) -> None:
    config = _control(arm="a").as_config()
    config[field] = value

    with pytest.raises(ValueError, match="trajectory-audit contract"):
        DeepEyesStrictControlBinding.from_mapping(config)
