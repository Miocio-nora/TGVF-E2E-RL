"""Source-facing protocol used by the native DeepEyes Crop control.

The tool system literal is copied byte-for-byte from DeepEyes' public
``mm_process_engine/prompt.py``.  The user suffix intentionally differs in one
project-owned respect: the final answer is plain text, with no answer-wrapper
dialect.  This module deliberately does not know how a crop is executed; the
native runtime owns pixels and tool state.  Keeping the prompt contract
independent makes it possible to prove that a standard veRL ``ToolAgentLoop``
sees the intended protocol without importing CUDA or veRL in unit tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any


DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA = "tgvf.deepeyes-native-clean-final-protocol.v2"
DEEPEYES_VISUAL_PROMPT_VERSION = "deepeyes-system-v2-clean-final-v1"
DEEPEYES_THINKLITE_PROMPT_VERSION = "deepeyes-thinklite-image-boxed-v3"
DEEPEYES_TOOL_NAME = "image_zoom_in_tool"
DEEPEYES_TOOL_PARSER = "hermes"
# The paper's successful run allows at most six active perceptions.  One final
# assistant turn is reserved for the answer after the sixth Crop observation.
DEEPEYES_MAX_ACTIVE_PERCEPTION = 6
DEEPEYES_VISUAL_AGENT_NAME = "prl13_native_deepeyes_visual"
DEEPEYES_THINKLITE_AGENT_NAME = "single_turn_agent"

# Keep whitespace, newlines, the public example, and even the public schema's
# ``required=[\"bbox\"]`` typo intact.  The executable call contract below is
# explicit that the actual argument is ``bbox_2d``; changing the visible
# system prompt would stop this from being an exact DeepEyes protocol control.
SYSTEM_PROMPT_V2 = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:\x20\x20
<tool_call>\x20\x20
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}\x20\x20
</tool_call>"""

USER_PROMPT_V2 = (
    "\nThink first, call **image_zoom_in_tool** if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> "
    "(if tools needed), followed by the final answer directly as plain text. "
)

THINKLITE_BOXED_INSTRUCTION = (
    "Let's think step by step and output the final answer within \\boxed{}."
)

VISUAL_SOURCES = frozenset({"vstar", "arxivqa"})
THINKLITE_SOURCE = "thinklite"
OFFICIAL_SOURCE_ALIASES: Mapping[str, str] = MappingProxyType(
    {"vstar": "vstar", "arxivqa": "chart", "thinklite": "thinklite_eureka"}
)

_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


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


SYSTEM_PROMPT_V2_SHA256 = _sha256_text(SYSTEM_PROMPT_V2)
USER_PROMPT_V2_SHA256 = _sha256_text(USER_PROMPT_V2)
THINKLITE_BOXED_INSTRUCTION_SHA256 = _sha256_text(THINKLITE_BOXED_INSTRUCTION)


@dataclass(frozen=True, slots=True)
class DeepEyesPromptIdentity:
    source_family: str
    system_prompt_sha256: str | None
    user_instruction_sha256: str
    bundle_sha256: str
    version: str
    tool_parser: str


VISUAL_PROMPT_IDENTITY = DeepEyesPromptIdentity(
    source_family="visual",
    system_prompt_sha256=SYSTEM_PROMPT_V2_SHA256,
    user_instruction_sha256=USER_PROMPT_V2_SHA256,
    bundle_sha256=_sha256_json(
        {
            "schema": DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA,
            "source_family": "visual",
            "system_prompt": SYSTEM_PROMPT_V2,
            "user_suffix": USER_PROMPT_V2,
            "tool_name": DEEPEYES_TOOL_NAME,
            "tool_parser": DEEPEYES_TOOL_PARSER,
        }
    ),
    version=DEEPEYES_VISUAL_PROMPT_VERSION,
    tool_parser=DEEPEYES_TOOL_PARSER,
)

THINKLITE_PROMPT_IDENTITY = DeepEyesPromptIdentity(
    source_family="thinklite",
    system_prompt_sha256=None,
    user_instruction_sha256=THINKLITE_BOXED_INSTRUCTION_SHA256,
    bundle_sha256=_sha256_json(
        {
            "schema": DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA,
            "source_family": "thinklite",
            "image_bearing": True,
            "task_instruction": THINKLITE_BOXED_INSTRUCTION,
            "tools": [],
        }
    ),
    version=DEEPEYES_THINKLITE_PROMPT_VERSION,
    tool_parser=DEEPEYES_TOOL_PARSER,
)


def source_family(data_source: object) -> str:
    """Map a local T1 source to the only accepted DeepEyes protocol family."""

    if data_source in VISUAL_SOURCES:
        return "visual"
    if data_source == THINKLITE_SOURCE:
        return "thinklite"
    raise ValueError(f"unsupported DeepEyes source: {data_source!r}")


def prompt_identity_for_source(data_source: object) -> DeepEyesPromptIdentity:
    return (
        VISUAL_PROMPT_IDENTITY
        if source_family(data_source) == "visual"
        else THINKLITE_PROMPT_IDENTITY
    )


def agent_name_for_source(data_source: object) -> str:
    return (
        DEEPEYES_VISUAL_AGENT_NAME
        if source_family(data_source) == "visual"
        else DEEPEYES_THINKLITE_AGENT_NAME
    )


def tools_kwargs_for_visual_row(
    gt_regions: object,
) -> dict[str, dict[str, dict[str, tuple[tuple[int, int, int, int], ...]]]]:
    """Build the exact per-row state injection consumed by the native tool.

    ArxivQA must call this with an empty sequence.  ThinkLite must not call it
    at all because it is routed through ``single_turn_agent``.
    """

    if not isinstance(gt_regions, (list, tuple)):
        raise TypeError("gt_regions must be a list/tuple of source-pixel boxes")
    normalized: list[tuple[int, int, int, int]] = []
    for index, region in enumerate(gt_regions):
        if (
            not isinstance(region, (list, tuple))
            or len(region) != 4
            or any(type(coordinate) is not int for coordinate in region)
        ):
            raise ValueError(f"gt_regions[{index}] must contain four integers")
        left, top, right, bottom = region
        if not 0 <= left < right or not 0 <= top < bottom:
            raise ValueError(f"gt_regions[{index}] must be a non-empty box")
        normalized.append((left, top, right, bottom))
    return {
        DEEPEYES_TOOL_NAME: {
            "create_kwargs": {"gt_regions": tuple(normalized)},
        }
    }


def build_visual_messages(
    question: str, *, image: object = "<image>"
) -> tuple[dict[str, Any], ...]:
    """Build the initial native multimodal message pair for V*/ArxivQA."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    return (
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question + USER_PROMPT_V2},
            ],
        },
    )


def build_visual_tool_response_message(*, image: object) -> dict[str, Any]:
    """Return the official V2 continuation prompt carrying one native crop."""

    return {
        "role": "tool",
        "name": DEEPEYES_TOOL_NAME,
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT_V2},
        ],
    }


def build_thinklite_messages(
    question: str,
    *,
    image: object = "<image>",
    task_kind: str = "math",
) -> tuple[dict[str, Any], ...]:
    """Build an image-bearing, single-turn, no-tool ThinkLite prompt."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    if task_kind not in {"math", "open", "mcq"}:
        raise ValueError("ThinkLite task_kind must be math, open, or mcq")
    return (
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": question.rstrip() + " " + THINKLITE_BOXED_INSTRUCTION,
                },
            ],
        },
    )


def direct_answer_after_last_tool_call(text: str) -> str | None:
    """Return trailing direct-final text from a mixed action/final turn.

    The clean dialect has no final-answer wrapper.  This tail rule preserves
    the previous answer-over-action precedence without depending on one: a
    tool call followed by non-empty plain text is final, while a turn ending
    at ``</tool_call>`` remains an action.
    """

    if not isinstance(text, str):
        raise TypeError("assistant turn must be text")
    if "</tool_call>" not in text:
        return None
    tail = text.rsplit("</tool_call>", 1)[-1]
    for terminal in ("<|im_end|>", "<|endoftext|>"):
        tail = tail.replace(terminal, "")
    tail = tail.strip()
    return tail or None


def parse_hermes_crop_call(text: str) -> dict[str, object]:
    """Validate the JSON payload expected from veRL's Hermes tool parser.

    This is a dependency-free contract probe, not a replacement for veRL's
    parser.  It intentionally rejects the historical ``qwen3_coder`` shape and
    the typoed ``bbox`` field so a smoke cannot pass with an incompatible
    parser/tool pairing.
    """

    if not isinstance(text, str):
        raise TypeError("tool call must be text")
    matches = _TOOL_CALL.findall(text)
    if len(matches) != 1:
        raise ValueError("exactly one <tool_call> JSON object is required")
    try:
        value = json.loads(matches[0].strip())
    except json.JSONDecodeError as error:
        raise ValueError("tool call payload must be strict JSON") from error
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        raise ValueError("Hermes tool call must contain name and arguments")
    if value["name"] != DEEPEYES_TOOL_NAME:
        raise ValueError("tool call name differs from image_zoom_in_tool")
    arguments = value["arguments"]
    if (
        not isinstance(arguments, dict)
        or not set(arguments)
        <= {
            "bbox_2d",
            "label",
        }
        or "bbox_2d" not in arguments
    ):
        raise ValueError("tool arguments require bbox_2d and optional label")
    bbox = arguments["bbox_2d"]
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(type(coordinate) not in {int, float} for coordinate in bbox)
    ):
        raise ValueError("bbox_2d must contain four JSON numbers")
    left, top, right, bottom = (float(value) for value in bbox)
    if not left < right or not top < bottom:
        raise ValueError("bbox_2d must be non-empty")
    label = arguments.get("label")
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ValueError("optional label must be non-empty text")
    return {"name": DEEPEYES_TOOL_NAME, "arguments": dict(arguments)}


__all__ = [
    "DEEPEYES_MAX_ACTIVE_PERCEPTION",
    "DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA",
    "DEEPEYES_TOOL_NAME",
    "DEEPEYES_TOOL_PARSER",
    "DEEPEYES_THINKLITE_AGENT_NAME",
    "DEEPEYES_VISUAL_AGENT_NAME",
    "OFFICIAL_SOURCE_ALIASES",
    "SYSTEM_PROMPT_V2",
    "SYSTEM_PROMPT_V2_SHA256",
    "THINKLITE_BOXED_INSTRUCTION",
    "THINKLITE_BOXED_INSTRUCTION_SHA256",
    "THINKLITE_PROMPT_IDENTITY",
    "USER_PROMPT_V2",
    "USER_PROMPT_V2_SHA256",
    "VISUAL_PROMPT_IDENTITY",
    "agent_name_for_source",
    "build_thinklite_messages",
    "build_visual_messages",
    "build_visual_tool_response_message",
    "direct_answer_after_last_tool_call",
    "parse_hermes_crop_call",
    "prompt_identity_for_source",
    "source_family",
    "tools_kwargs_for_visual_row",
]
