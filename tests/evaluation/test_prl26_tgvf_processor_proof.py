from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.tgvf_target_guide_v2_protocol import (
    TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY,
)


_ROOT = Path(__file__).parents[2]
_TOOL = _ROOT / "tools/validate_prl26_train512_processor_proof.py"
_SPEC = importlib.util.spec_from_file_location("prl26_tgvf_processor_proof", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VALIDATOR)


def _case(arm: str):
    if arm == "short":
        name = (
            "prl_26_c_qwen3_instruct_short_tgvf_train512_parity_"
            "s32_bs16_n16_teacher25_ws8.toml"
        )
        identity = TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY
        builder = "build_tgvf_visual_messages"
    else:
        name = (
            "prl_26_d_qwen3_instruct_target_guide_v2_tgvf_train512_parity_"
            "s32_bs16_n16_teacher25_ws8.toml"
        )
        identity = TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY
        builder = "build_tgvf_target_guide_v2_visual_messages"
    owner = load_policy_e2e_smoke_run_config(
        (_ROOT / "configs/policy/runs" / name).resolve(),
        allow_external_agent_loop_config=True,
    )
    proof = {
        "schema_version": "tgvf.matched-tgvf-processor-static-proof.v1",
        "run_config_schema": owner.schema_version,
        "message_builder": builder,
        "prompt_bundle_sha256": identity.bundle_sha256,
        "system_prompt_sha256": identity.system_prompt_sha256,
        "template_tools_argument": [],
        "configured_image_max_pixels": 262144,
        "effective_processor_image_size": {
            "shortest_edge": 65536,
            "longest_edge": 262144,
        },
        "synthetic_source_pixel_area": 3_145_728,
        "synthetic_represented_pixel_area": 239_616,
        "synthetic_visual_token_count": 234,
        "success_environment_renderer": (
            "render_qwen_native_matched_tgvf_success_environment_text"
        ),
        "action_boundary": {
            "required_request_stop_strings": ["</tool_call>"],
            "required_request_stop_token_ids": [151_645],
            "include_stop_str_in_output": True,
            "tool_call_terminal_suffixes": [""],
            "tool_call_outcomes": [
                {"finish_reason": "stop", "stop_reason": "</tool_call>"}
            ],
        },
    }
    return owner, proof


@pytest.mark.parametrize("arm", ("short", "full"))
def test_tgvf_processor_proof_validator_accepts_exact_prompt_and_boundary(
    arm: str,
) -> None:
    owner, proof = _case(arm)

    result = _VALIDATOR._validate_tgvf(
        proof, owner, owner.policy.sampling, arm=arm
    )

    assert result["tool_profile"] == "tgvf_only"
    assert result["prompt_bundle_sha256"] == owner.protocol.prompt_sha256
    assert result["stop_strings"] == ["</tool_call>"]


def test_tgvf_processor_proof_validator_rejects_action_boundary_drift() -> None:
    owner, proof = _case("full")
    drifted = copy.deepcopy(proof)
    drifted["action_boundary"]["required_request_stop_strings"] = []

    with pytest.raises(RuntimeError, match="processor/action proof differs"):
        _VALIDATOR._validate_tgvf(
            drifted, owner, owner.policy.sampling, arm="full"
        )

