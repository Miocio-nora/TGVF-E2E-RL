"""DeepEyes-matched, clean-final protocol for atomic Crop+TGVF."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from tgvf_rl.protocol.schema import (
    TGVF_CROP_TOOL_NAME,
    TGVF_CROP_TOOL_SCHEMA_SHA256,
    build_tgvf_crop_tool_schema,
)
from tgvf_rl.protocol.observation_contract import (
    NativeSuccessObservationProtocolId,
)

from .deepeyes_official_protocol import (
    DEEPEYES_MAX_ACTIVE_PERCEPTION,
    USER_PROMPT_V2,
)
from .tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
)


CROP_TGVF_DEEPEYES_MATCHED_PROTOCOL_SCHEMA = (
    "tgvf.crop-tgvf-deepeyes-matched-clean-final-protocol.v1"
)
CROP_TGVF_DEEPEYES_MATCHED_PROMPT_VERSION = (
    "crop-tgvf-deepeyes-system-v2-clean-final-v1"
)
CROP_TGVF_DEEPEYES_MATCHED_TOOL_NAME = TGVF_CROP_TOOL_NAME
CROP_TGVF_DEEPEYES_MATCHED_TOOL_PARSER = "hermes"
CROP_TGVF_DEEPEYES_MATCHED_MAXIMUM_TOOL_CALLS = DEEPEYES_MAX_ACTIVE_PERCEPTION


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _replace_deepeyes_tool_contract(
    *, base: str, visible_schema: str, example_call: str
) -> str:
    """Change only the tool-specific regions of the matched DeepEyes frame."""

    tools_open = "<tools>\n"
    tools_close = "\n</tools>"
    example_open = "**Example**:  \n<tool_call>  \n"
    example_close = "  \n</tool_call>"
    before_tools, separator, after_tools_open = base.partition(tools_open)
    if not separator:
        raise RuntimeError("matched DeepEyes system prompt lacks <tools> frame")
    _old_schema, separator, after_tools = after_tools_open.partition(tools_close)
    if not separator:
        raise RuntimeError("matched DeepEyes system prompt lacks </tools> frame")
    before_example, separator, after_example_open = after_tools.partition(example_open)
    if not separator:
        raise RuntimeError("matched DeepEyes system prompt lacks example frame")
    _old_example, separator, after_example = after_example_open.partition(example_close)
    if not separator:
        raise RuntimeError("matched DeepEyes system prompt lacks example closer")
    return (
        before_tools
        + tools_open
        + visible_schema
        + tools_close
        + before_example
        + example_open
        + example_call
        + example_close
        + after_example
    )


_VISIBLE_TOOL_SCHEMA = _compact_json(build_tgvf_crop_tool_schema())
_VISIBLE_EXAMPLE_CALL = _compact_json(
    {
        "name": CROP_TGVF_DEEPEYES_MATCHED_TOOL_NAME,
        "arguments": {
            "bbox_2d": [120, 180, 460, 620],
            "target": (
                "the small circular gauge's needle position for reading its value"
            ),
        },
    }
)

# The prose, XML framing, whitespace, and clean-final dialect are inherited
# byte-for-byte from the current matched TGVF protocol.  Only the visible
# function schema and example call are replaced with the one atomic contract.
CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT = _replace_deepeyes_tool_contract(
    base=TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
    visible_schema=_VISIBLE_TOOL_SCHEMA,
    example_call=_VISIBLE_EXAMPLE_CALL,
)
CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT = USER_PROMPT_V2.replace(
    "image_zoom_in_tool", CROP_TGVF_DEEPEYES_MATCHED_TOOL_NAME
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT_SHA256 = _sha256_text(
    CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
)
CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT_SHA256 = _sha256_text(
    CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT
)


@dataclass(frozen=True, slots=True)
class CropTGVFDeepEyesMatchedPromptIdentity:
    system_prompt_sha256: str
    user_instruction_sha256: str
    tool_schema_sha256: str
    bundle_sha256: str
    version: str
    tool_parser: str
    maximum_tool_calls: int
    success_observation_protocol_id: NativeSuccessObservationProtocolId


CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY = CropTGVFDeepEyesMatchedPromptIdentity(
    system_prompt_sha256=CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT_SHA256,
    user_instruction_sha256=CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT_SHA256,
    tool_schema_sha256=TGVF_CROP_TOOL_SCHEMA_SHA256,
    bundle_sha256=_sha256_json(
        {
            "schema": CROP_TGVF_DEEPEYES_MATCHED_PROTOCOL_SCHEMA,
            "source_family": "visual",
            "system_prompt": CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
            "user_suffix": CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT,
            "tool_name": CROP_TGVF_DEEPEYES_MATCHED_TOOL_NAME,
            "tool_parser": CROP_TGVF_DEEPEYES_MATCHED_TOOL_PARSER,
            "tool_schema_sha256": TGVF_CROP_TOOL_SCHEMA_SHA256,
            "maximum_tool_calls": (CROP_TGVF_DEEPEYES_MATCHED_MAXIMUM_TOOL_CALLS),
            "observation_text_echoes_target": False,
            "final_answer_wrapper": None,
        }
    ),
    version=CROP_TGVF_DEEPEYES_MATCHED_PROMPT_VERSION,
    tool_parser=CROP_TGVF_DEEPEYES_MATCHED_TOOL_PARSER,
    maximum_tool_calls=CROP_TGVF_DEEPEYES_MATCHED_MAXIMUM_TOOL_CALLS,
    success_observation_protocol_id=(
        NativeSuccessObservationProtocolId.DEEPEYES_ATOMIC_MATCHED_V1
    ),
)


def build_crop_tgvf_visual_messages(
    question: str, *, image: object = "<image>"
) -> tuple[dict[str, Any], ...]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    return (
        {"role": "system", "content": CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": question + CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT,
                },
            ],
        },
    )


def build_crop_tgvf_tool_response_message(*, observation: object) -> dict[str, Any]:
    """Return only the crop-conditioned latent visual and matched suffix."""

    return {
        "role": "tool",
        "name": CROP_TGVF_DEEPEYES_MATCHED_TOOL_NAME,
        "content": [
            {"type": "image", "image": observation},
            {"type": "text", "text": CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT},
        ],
    }


__all__ = [
    "CROP_TGVF_DEEPEYES_MATCHED_MAXIMUM_TOOL_CALLS",
    "CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY",
    "CROP_TGVF_DEEPEYES_MATCHED_PROMPT_VERSION",
    "CROP_TGVF_DEEPEYES_MATCHED_PROTOCOL_SCHEMA",
    "CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT",
    "CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT_SHA256",
    "CROP_TGVF_DEEPEYES_MATCHED_TOOL_NAME",
    "CROP_TGVF_DEEPEYES_MATCHED_TOOL_PARSER",
    "CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT",
    "CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT_SHA256",
    "CropTGVFDeepEyesMatchedPromptIdentity",
    "build_crop_tgvf_tool_response_message",
    "build_crop_tgvf_visual_messages",
]
