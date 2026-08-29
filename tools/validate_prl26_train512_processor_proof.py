#!/usr/bin/env python3
"""Persist a fail-closed real-processor proof for one PRL-26 eval arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_coredev import (  # noqa: E402
    load_frozen_policy_evaluation_snapshot,
    load_policy_coredev_config,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (  # noqa: E402
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (  # noqa: E402
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.tgvf_target_guide_v2_protocol import (  # noqa: E402
    TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY,
)


IMAGE_MAX_PIXELS = 262_144
OPTIMIZER_STEP = 32
PROCESSOR_DEFAULT_MAX_PIXELS = 16_777_216


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"immutable processor proof differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise RuntimeError(f"immutable processor proof differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _common_geometry(proof: dict[str, Any]) -> None:
    runtime = proof.get("runtime_mm_processor_kwargs")
    if (
        proof.get("configured_image_max_pixels") != IMAGE_MAX_PIXELS
        or proof.get("processor_image_size", {}).get("longest_edge")
        != PROCESSOR_DEFAULT_MAX_PIXELS
        or proof.get("effective_processor_image_size", {}).get("longest_edge")
        != IMAGE_MAX_PIXELS
        or runtime
        != {"size": {"shortest_edge": 65_536, "longest_edge": IMAGE_MAX_PIXELS}}
        or proof.get("runtime_override_path") != "mm_processor_kwargs.size.longest_edge"
        or proof.get("vllm_012_shallow_hashable") is not True
        or proof.get("nested_images_kwargs_present") is not False
        or proof.get("max_pixels_kwarg_present") is not False
    ):
        raise RuntimeError("real-Qwen pixel512 runtime override proof differs")


def _validate_no_tool(
    proof: dict[str, Any], owner: Any, runtime_sampling: Any
) -> dict[str, object]:
    _common_geometry(proof)
    represented = proof.get("synthetic_native_represented_pixel_area")
    visual_tokens = proof.get("synthetic_native_visual_token_count")
    if (
        proof.get("synthetic_native_source_pixel_area") != 3_145_728
        or type(represented) is not int
        or represented != 239_616
        or type(visual_tokens) is not int
        or visual_tokens != 234
        or proof.get("tool_schema_visible") is not False
        or proof.get("system_prompt_present") is not False
        or owner.protocol.tool_profile.value != "no_tool"
        or owner.protocol.enabled_tool_names
        or tuple(owner.policy.sampling.stop_strings) != ()
        or owner.policy.sampling.include_stop_str_in_output is not True
        or tuple(runtime_sampling.stop_strings) != ()
        or runtime_sampling.include_stop_str_in_output is not True
    ):
        raise RuntimeError("PRL-26-A NoTool processor/protocol proof differs")
    return {
        "tool_profile": "no_tool",
        "tool_schema_visible": False,
        "stop_strings": [],
        "represented_pixel_areas": [represented],
        "visual_token_counts": [visual_tokens],
    }


def _validate_crop(
    proof: dict[str, Any], owner: Any, runtime_sampling: Any
) -> dict[str, object]:
    _common_geometry(proof)
    sources = proof.get("synthetic_native_source_pixel_areas")
    represented = proof.get("synthetic_native_represented_pixel_areas")
    visual_tokens = proof.get("synthetic_native_visual_token_counts")
    if (
        sources != [3_145_728, 3_145_728]
        or represented != [239_616, 239_616]
        or visual_tokens != [234, 234]
        or proof.get("native_original_image_count") != 1
        or proof.get("native_crop_image_count") != 1
        or proof.get("tools_argument_empty") is not True
        or proof.get("visible_system_schema") is not True
        or proof.get("observation_role") != "user"
        or owner.protocol.tool_profile.value != "crop_only"
        or owner.protocol.enabled_tool_names != ("image_zoom_in_tool",)
        or owner.protocol.maximum_tool_calls != 6
        or tuple(owner.policy.sampling.stop_strings) != ("</tool_call>",)
        or owner.policy.sampling.include_stop_str_in_output is not True
        or tuple(owner.policy.sampling.stop_token_ids) != (151_645,)
        or tuple(runtime_sampling.stop_strings) != ("</tool_call>",)
        or runtime_sampling.include_stop_str_in_output is not True
        or tuple(runtime_sampling.stop_token_ids) != (151_645,)
    ):
        raise RuntimeError("PRL-26-B Crop processor/action-boundary proof differs")
    return {
        "tool_profile": "crop_only",
        "enabled_tool_names": ["image_zoom_in_tool"],
        "maximum_tool_calls": 6,
        "stop_strings": ["</tool_call>"],
        "include_stop_str_in_output": True,
        "represented_pixel_areas": represented,
        "visual_token_counts": visual_tokens,
    }


def _validate_tgvf(
    proof: dict[str, Any], owner: Any, runtime_sampling: Any, *, arm: str
) -> dict[str, object]:
    expected = {
        "short": (
            POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
            "build_tgvf_visual_messages",
        ),
        "full": (
            POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
            TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY,
            "build_tgvf_target_guide_v2_visual_messages",
        ),
    }[arm]
    expected_schema, prompt_identity, builder = expected
    action = proof.get("action_boundary")
    represented = proof.get("synthetic_represented_pixel_area")
    visual_tokens = proof.get("synthetic_visual_token_count")
    effective_size = proof.get("effective_processor_image_size")
    if (
        proof.get("schema_version")
        != "tgvf.matched-tgvf-processor-static-proof.v1"
        or proof.get("run_config_schema") != expected_schema
        or proof.get("message_builder") != builder
        or proof.get("prompt_bundle_sha256") != prompt_identity.bundle_sha256
        or proof.get("system_prompt_sha256")
        != prompt_identity.system_prompt_sha256
        or proof.get("template_tools_argument") != []
        or proof.get("configured_image_max_pixels") != IMAGE_MAX_PIXELS
        or not isinstance(effective_size, dict)
        or effective_size.get("longest_edge") != IMAGE_MAX_PIXELS
        or proof.get("synthetic_source_pixel_area") != 3_145_728
        or type(represented) is not int
        or not 0 < represented <= IMAGE_MAX_PIXELS
        or type(visual_tokens) is not int
        or visual_tokens <= 0
        or proof.get("success_environment_renderer")
        != "render_qwen_native_matched_tgvf_success_environment_text"
        or not isinstance(action, dict)
        or action.get("required_request_stop_strings") != ["</tool_call>"]
        or action.get("required_request_stop_token_ids") != [151_645]
        or action.get("include_stop_str_in_output") is not True
        or action.get("tool_call_terminal_suffixes") != [""]
        or action.get("tool_call_outcomes")
        != [{"finish_reason": "stop", "stop_reason": "</tool_call>"}]
        or owner.protocol.tool_profile.value != "tgvf_only"
        or owner.protocol.enabled_tool_names != ("tgvf_focus_tool",)
        or owner.protocol.maximum_tool_calls != 6
        or tuple(owner.policy.sampling.stop_strings) != ("</tool_call>",)
        or tuple(owner.policy.sampling.stop_token_ids) != (151_645,)
        or owner.policy.sampling.include_stop_str_in_output is not True
        or tuple(runtime_sampling.stop_strings) != ("</tool_call>",)
        or tuple(runtime_sampling.stop_token_ids) != (151_645,)
        or runtime_sampling.include_stop_str_in_output is not True
    ):
        raise RuntimeError(f"PRL-26 {arm} TGVF processor/action proof differs")
    return {
        "tool_profile": "tgvf_only",
        "enabled_tool_names": ["tgvf_focus_tool"],
        "maximum_tool_calls": 6,
        "prompt_bundle_sha256": prompt_identity.bundle_sha256,
        "message_builder": builder,
        "stop_strings": ["</tool_call>"],
        "include_stop_str_in_output": True,
        "represented_pixel_area": represented,
        "visual_token_count": visual_tokens,
    }


def _validate_atomic(
    proof: dict[str, Any], owner: Any, runtime_sampling: Any
) -> dict[str, object]:
    action = proof.get("action_boundary")
    represented = proof.get("synthetic_represented_pixel_area")
    visual_tokens = proof.get("synthetic_visual_token_count")
    effective_size = proof.get("effective_processor_image_size")
    runtime = proof.get("runtime_mm_processor_kwargs")
    if (
        proof.get("schema_version") != "tgvf.matched-atomic-processor-static-proof.v1"
        or proof.get("run_config_schema")
        != POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
        or proof.get("message_builder") != "build_crop_tgvf_visual_messages"
        or proof.get("prompt_bundle_sha256")
        != CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
        or proof.get("system_prompt_sha256")
        != CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.system_prompt_sha256
        or proof.get("template_tools_argument") != []
        or proof.get("configured_image_max_pixels") != IMAGE_MAX_PIXELS
        or not isinstance(effective_size, dict)
        or effective_size.get("longest_edge") != IMAGE_MAX_PIXELS
        or runtime
        != {
            "size": {
                "shortest_edge": 65_536,
                "longest_edge": IMAGE_MAX_PIXELS,
            }
        }
        or proof.get("runtime_override_path") != "mm_processor_kwargs.size.longest_edge"
        or proof.get("synthetic_source_pixel_area") != 3_145_728
        or type(represented) is not int
        or not 0 < represented <= IMAGE_MAX_PIXELS
        or type(visual_tokens) is not int
        or visual_tokens <= 0
        or proof.get("success_environment_renderer")
        != "render_qwen_native_matched_crop_tgvf_success_environment_text"
        or not isinstance(action, dict)
        or action.get("required_request_stop_strings") != ["</tool_call>"]
        or action.get("required_request_stop_token_ids") != [151_645]
        or action.get("include_stop_str_in_output") is not True
        or action.get("tool_call_terminal_suffixes") != [""]
        or action.get("tool_call_outcomes")
        != [{"finish_reason": "stop", "stop_reason": "</tool_call>"}]
        or owner.protocol.tool_profile.value != "crop_tgvf"
        or owner.protocol.enabled_tool_names != ("tgvf_crop_tool",)
        or owner.protocol.maximum_tool_calls != 6
        or tuple(owner.policy.sampling.stop_strings) != ("</tool_call>",)
        or tuple(owner.policy.sampling.stop_token_ids) != (151_645,)
        or owner.policy.sampling.include_stop_str_in_output is not True
        or tuple(runtime_sampling.stop_strings) != ("</tool_call>",)
        or tuple(runtime_sampling.stop_token_ids) != (151_645,)
        or runtime_sampling.include_stop_str_in_output is not True
    ):
        raise RuntimeError("PRL-26-E Atomic processor/action-boundary proof differs")
    return {
        "tool_profile": "crop_tgvf",
        "enabled_tool_names": ["tgvf_crop_tool"],
        "maximum_tool_calls": 6,
        "prompt_bundle_sha256": (
            CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
        ),
        "message_builder": "build_crop_tgvf_visual_messages",
        "stop_strings": ["</tool_call>"],
        "include_stop_str_in_output": True,
        "represented_pixel_area": represented,
        "visual_token_count": visual_tokens,
    }


def validate(
    *, arm: str, config_path: Path, validation_path: Path, output_path: Path
) -> dict[str, object]:
    config = load_policy_coredev_config(config_path.resolve())
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    owner = load_policy_e2e_smoke_run_config(
        config.policy_config_path.resolve(), allow_external_agent_loop_config=True
    )
    snapshot = load_frozen_policy_evaluation_snapshot(config)
    runtime_sampling = snapshot.run.policy.sampling
    expected_schema = {
        "no-tool": POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        "crop": POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        "short": POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        "full": (
            POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
        ),
        "atomic": POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    }[arm]
    if (
        owner.schema_version != expected_schema
        or owner.policy.image_max_pixels != IMAGE_MAX_PIXELS
        or config.evaluation_image_max_pixels != IMAGE_MAX_PIXELS
        or config.expected_optimizer_step != OPTIMIZER_STEP
        or validation.get("evaluation_id") != config.evaluation_id
        or validation.get("evaluation_identity_sha256") is None
        or validation.get("task_count") != 2511
        or validation.get("single_image_count") != 2240
        or validation.get("optimizer_step") != OPTIMIZER_STEP
        or validation.get("gpu_or_api_used") is not False
        or validation.get("vllm_engine_constructed") is not False
    ):
        raise RuntimeError("PRL-26 Train@512 validation envelope differs")
    proof_key = {
        "no-tool": "no_tool_matched_processor_proof",
        "crop": "official_visible_processor_proof",
        "short": "matched_tgvf_processor_proof",
        "full": "matched_tgvf_processor_proof",
        "atomic": "matched_atomic_processor_proof",
    }[arm]
    proof = validation.get(proof_key)
    if not isinstance(proof, dict):
        raise RuntimeError(f"PRL-26 {arm} validation omitted real-processor proof")
    if arm == "no-tool":
        protocol = _validate_no_tool(proof, owner, runtime_sampling)
    elif arm == "crop":
        protocol = _validate_crop(proof, owner, runtime_sampling)
    elif arm in {"short", "full"}:
        protocol = _validate_tgvf(proof, owner, runtime_sampling, arm=arm)
    else:
        protocol = _validate_atomic(proof, owner, runtime_sampling)
    content: dict[str, object] = {
        "schema_version": "tgvf.prl26-train512-processor-proof.v1",
        "arm": arm,
        "evaluation_id": config.evaluation_id,
        "evaluation_identity_sha256": validation["evaluation_identity_sha256"],
        "optimizer_step": OPTIMIZER_STEP,
        "train_image_max_pixels": owner.policy.image_max_pixels,
        "evaluation_image_max_pixels": config.evaluation_image_max_pixels,
        "runtime_override_path": "mm_processor_kwargs.size.longest_edge",
        "runtime_longest_edge": proof["runtime_mm_processor_kwargs"]["size"][
            "longest_edge"
        ],
        "dynamic_real_processor_validation": True,
        "gpu_or_api_used": False,
        "vllm_engine_constructed": False,
        "policy_config_path": str(config.policy_config_path.resolve()),
        "policy_config_file_sha256": _sha256(config.policy_config_path),
        "evaluation_config_path": str(config_path.resolve()),
        "evaluation_config_file_sha256": _sha256(config_path),
        "validation_file_sha256": _sha256(validation_path),
        "protocol": protocol,
        "proof": proof,
    }
    payload = {**content, "identity_sha256": _canonical_sha256(content)}
    _write_immutable_json(output_path.resolve(), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=("no-tool", "crop", "short", "full", "atomic"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        arm=args.arm,
        config_path=args.config,
        validation_path=args.validation_json,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
