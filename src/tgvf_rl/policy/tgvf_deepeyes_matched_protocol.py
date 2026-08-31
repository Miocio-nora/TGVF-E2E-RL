"""DeepEyes-matched, clean-final protocol for the TGVF visual tool."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from tgvf_rl.protocol.observation_contract import (
    NativeSuccessObservationProtocolId,
)

from .deepeyes_official_protocol import USER_PROMPT_V2


TGVF_DEEPEYES_MATCHED_PROTOCOL_SCHEMA = "tgvf.deepeyes-matched-clean-final-protocol.v1"
TGVF_DEEPEYES_MATCHED_PROMPT_VERSION = "tgvf-deepeyes-system-v2-clean-final-v1"
TGVF_DEEPEYES_MATCHED_TOOL_NAME = "tgvf_focus_tool"
TGVF_DEEPEYES_MATCHED_TOOL_PARSER = "hermes"

TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"tgvf_focus_tool","description":"Extract target-conditioned visual evidence from the original image for a specific visual query.","parameters":{"type":"object","properties":{"target":{"type":"string","description":"A concise, self-contained visual query specifying what to inspect and what visual evidence, attribute, text, count, comparison, or spatial relation to obtain. Do not include a guessed final answer or answer-option value."}},"required":["target"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "tgvf_focus_tool", "arguments": {"target": "the small circular gauge's needle position for reading its value"}}  
</tool_call>"""

TGVF_DEEPEYES_MATCHED_USER_PROMPT = USER_PROMPT_V2.replace(
    "image_zoom_in_tool", TGVF_DEEPEYES_MATCHED_TOOL_NAME
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


TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT_SHA256 = _sha256_text(
    TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
)
TGVF_DEEPEYES_MATCHED_USER_PROMPT_SHA256 = _sha256_text(
    TGVF_DEEPEYES_MATCHED_USER_PROMPT
)


@dataclass(frozen=True, slots=True)
class TGVFDeepEyesMatchedPromptIdentity:
    system_prompt_sha256: str
    user_instruction_sha256: str
    bundle_sha256: str
    version: str
    tool_parser: str
    success_observation_protocol_id: NativeSuccessObservationProtocolId


TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY = TGVFDeepEyesMatchedPromptIdentity(
    system_prompt_sha256=TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT_SHA256,
    user_instruction_sha256=TGVF_DEEPEYES_MATCHED_USER_PROMPT_SHA256,
    bundle_sha256=_sha256_json(
        {
            "schema": TGVF_DEEPEYES_MATCHED_PROTOCOL_SCHEMA,
            "source_family": "visual",
            "system_prompt": TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
            "user_suffix": TGVF_DEEPEYES_MATCHED_USER_PROMPT,
            "tool_name": TGVF_DEEPEYES_MATCHED_TOOL_NAME,
            "tool_parser": TGVF_DEEPEYES_MATCHED_TOOL_PARSER,
            "observation_text_echoes_target": False,
            "final_answer_wrapper": None,
        }
    ),
    version=TGVF_DEEPEYES_MATCHED_PROMPT_VERSION,
    tool_parser=TGVF_DEEPEYES_MATCHED_TOOL_PARSER,
    success_observation_protocol_id=(
        NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1
    ),
)


def build_tgvf_visual_messages(
    question: str, *, image: object = "<image>"
) -> tuple[dict[str, Any], ...]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    return (
        {"role": "system", "content": TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": question + TGVF_DEEPEYES_MATCHED_USER_PROMPT,
                },
            ],
        },
    )


def build_tgvf_tool_response_message(*, observation: object) -> dict[str, Any]:
    """Return native latent visual evidence without echoing target as text."""

    return {
        "role": "tool",
        "name": TGVF_DEEPEYES_MATCHED_TOOL_NAME,
        "content": [
            {"type": "image", "image": observation},
            {"type": "text", "text": TGVF_DEEPEYES_MATCHED_USER_PROMPT},
        ],
    }


__all__ = [
    "TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY",
    "TGVF_DEEPEYES_MATCHED_PROMPT_VERSION",
    "TGVF_DEEPEYES_MATCHED_PROTOCOL_SCHEMA",
    "TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT",
    "TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT_SHA256",
    "TGVF_DEEPEYES_MATCHED_TOOL_NAME",
    "TGVF_DEEPEYES_MATCHED_TOOL_PARSER",
    "TGVF_DEEPEYES_MATCHED_USER_PROMPT",
    "TGVF_DEEPEYES_MATCHED_USER_PROMPT_SHA256",
    "TGVFDeepEyesMatchedPromptIdentity",
    "build_tgvf_tool_response_message",
    "build_tgvf_visual_messages",
]
