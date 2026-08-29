from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "tools/validate_prl26_train512_processor_proof.py"
_SPEC = importlib.util.spec_from_file_location("prl26_atomic_processor_proof", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VALIDATOR)
_CONFIG = _ROOT / (
    "configs/policy/runs/prl_26_e_qwen3_instruct_full_atomic_crop_tgvf_"
    "train512_parity_s32_bs16_n16_teacher25_ws8.toml"
)


def _case():
    owner = load_policy_e2e_smoke_run_config(_CONFIG.resolve())
    proof = {
        "schema_version": "tgvf.matched-atomic-processor-static-proof.v1",
        "run_config_schema": owner.schema_version,
        "message_builder": "build_crop_tgvf_visual_messages",
        "prompt_bundle_sha256": (
            CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
        ),
        "system_prompt_sha256": (
            CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.system_prompt_sha256
        ),
        "template_tools_argument": [],
        "configured_image_max_pixels": 262144,
        "effective_processor_image_size": {
            "shortest_edge": 65536,
            "longest_edge": 262144,
        },
        "runtime_mm_processor_kwargs": {
            "size": {"shortest_edge": 65536, "longest_edge": 262144}
        },
        "runtime_override_path": "mm_processor_kwargs.size.longest_edge",
        "synthetic_source_pixel_area": 3_145_728,
        "synthetic_represented_pixel_area": 239_616,
        "synthetic_visual_token_count": 234,
        "success_environment_renderer": (
            "render_qwen_native_matched_crop_tgvf_success_environment_text"
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


def test_atomic_processor_proof_accepts_pixel512_and_action_boundary() -> None:
    owner, proof = _case()

    result = _VALIDATOR._validate_atomic(proof, owner, owner.policy.sampling)

    assert result["tool_profile"] == "crop_tgvf"
    assert result["enabled_tool_names"] == ["tgvf_crop_tool"]
    assert result["stop_strings"] == ["</tool_call>"]


def test_atomic_processor_proof_rejects_missing_action_stop() -> None:
    owner, proof = _case()
    drifted = copy.deepcopy(proof)
    drifted["action_boundary"]["required_request_stop_strings"] = []

    with pytest.raises(RuntimeError, match="action-boundary proof differs"):
        _VALIDATOR._validate_atomic(drifted, owner, owner.policy.sampling)
