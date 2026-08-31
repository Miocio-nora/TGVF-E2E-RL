"""Teacher-aligned target-guide-only variant of the matched TGVF prompt."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
    TGVF_DEEPEYES_MATCHED_TOOL_NAME,
    TGVF_DEEPEYES_MATCHED_TOOL_PARSER,
    TGVF_DEEPEYES_MATCHED_USER_PROMPT,
    TGVFDeepEyesMatchedPromptIdentity,
)


TGVF_TARGET_GUIDE_V2_PROTOCOL_SCHEMA = (
    "tgvf.deepeyes-matched-target-guide-only-protocol.v2"
)
TGVF_TARGET_GUIDE_V2_PROMPT_VERSION = "tgvf-deepeyes-matched-target-guide-only-v2"

# This guide and the teacher-aligned Target replacement below are the complete
# treatment.  They say nothing about reasoning turns, final-answer formatting,
# observation rendering, or call count; those remain owned by the Short arm.
TGVF_TARGET_GUIDE_V2_INSERTION = """

# Target definition and examples
The target is an answer-neutral, visually grounded descriptor of the evidence to re-encode. Use a noun phrase or short visual description, usually 6 to 24 words. Include the visual region or anchors, the visual aspect that matters, and enough surrounding context when needed. Avoid bare object names, task commands such as inspect, read, count, compare, determine, judge, answer, verify, decide, or infer, whether-questions, and guessed answers or option values.

Valid targets include:
- "small circular gauge, its needle position, and surrounding scale markings"
- "printed text below the red warning symbol"
- "wide shared view containing the bicycle, the parked car, and the space between them"
"""

_TARGET_GUIDE_INSERTION_ANCHOR = "\n\n# How to call a tool"
_LEGACY_TOOL_CALL_EXAMPLE_TARGET = (
    "the small circular gauge's needle position for reading its value"
)
_TEACHER_ALIGNED_TOOL_CALL_EXAMPLE_TARGET = (
    "small circular gauge, its needle position, and surrounding scale markings"
)
if TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT.count(_TARGET_GUIDE_INSERTION_ANCHOR) != 1:
    raise RuntimeError("matched TGVF target-guide insertion anchor differs")
if TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT.count(_LEGACY_TOOL_CALL_EXAMPLE_TARGET) != 1:
    raise RuntimeError("matched TGVF legacy target example differs")

TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT = TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT.replace(
    _LEGACY_TOOL_CALL_EXAMPLE_TARGET,
    _TEACHER_ALIGNED_TOOL_CALL_EXAMPLE_TARGET,
    1,
).replace(
    _TARGET_GUIDE_INSERTION_ANCHOR,
    TGVF_TARGET_GUIDE_V2_INSERTION + _TARGET_GUIDE_INSERTION_ANCHOR,
    1,
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


TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT_SHA256 = _sha256_text(
    TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT
)
TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY = TGVFDeepEyesMatchedPromptIdentity(
    system_prompt_sha256=TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT_SHA256,
    user_instruction_sha256=_sha256_text(TGVF_DEEPEYES_MATCHED_USER_PROMPT),
    bundle_sha256=_sha256_json(
        {
            "schema": TGVF_TARGET_GUIDE_V2_PROTOCOL_SCHEMA,
            "source_family": "visual",
            "system_prompt": TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT,
            "user_suffix": TGVF_DEEPEYES_MATCHED_USER_PROMPT,
            "tool_name": TGVF_DEEPEYES_MATCHED_TOOL_NAME,
            "tool_parser": TGVF_DEEPEYES_MATCHED_TOOL_PARSER,
            "observation_text_echoes_target": False,
            "final_answer_wrapper": None,
        }
    ),
    version=TGVF_TARGET_GUIDE_V2_PROMPT_VERSION,
    tool_parser=TGVF_DEEPEYES_MATCHED_TOOL_PARSER,
    success_observation_protocol_id=(
        TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.success_observation_protocol_id
    ),
)


def remove_tgvf_target_guide_v2(system_prompt: str) -> str:
    """Normalize the Target-only treatment back to the frozen Short prompt."""

    if system_prompt.count(TGVF_TARGET_GUIDE_V2_INSERTION) != 1:
        raise ValueError("target-guide v2 insertion count differs")
    normalized = system_prompt.replace(TGVF_TARGET_GUIDE_V2_INSERTION, "", 1)
    if normalized.count(_TEACHER_ALIGNED_TOOL_CALL_EXAMPLE_TARGET) != 1:
        raise ValueError("teacher-aligned tool-call target example count differs")
    return normalized.replace(
        _TEACHER_ALIGNED_TOOL_CALL_EXAMPLE_TARGET,
        _LEGACY_TOOL_CALL_EXAMPLE_TARGET,
        1,
    )


def build_tgvf_target_guide_v2_visual_messages(
    question: str, *, image: object = "<image>"
) -> tuple[dict[str, Any], ...]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    return (
        {"role": "system", "content": TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT},
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


__all__ = [
    "TGVF_TARGET_GUIDE_V2_INSERTION",
    "TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY",
    "TGVF_TARGET_GUIDE_V2_PROMPT_VERSION",
    "TGVF_TARGET_GUIDE_V2_PROTOCOL_SCHEMA",
    "TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT",
    "TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT_SHA256",
    "build_tgvf_target_guide_v2_visual_messages",
    "remove_tgvf_target_guide_v2",
]
