"""Matched visual prompt for the full-Qwen, no-tool RL control.

The control retains the reasoning/final-answer surface shared by the PRL25
visual arms while removing every tool-facing instruction and schema. It is a
single user message containing the immutable source image and canonical
question; the runtime must execute it in direct-only mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


NO_TOOL_RL_PROTOCOL_SCHEMA = "tgvf.no-tool-rl-clean-final-protocol.v1"
NO_TOOL_RL_PROMPT_VERSION = "no-tool-reasoning-clean-final-v1"
NO_TOOL_RL_TOOL_PARSER = "none"
NO_TOOL_RL_USER_PROMPT = (
    "\nThink first, then answer. Format strictly as:  <think>...</think>  "
    "followed by the final answer directly as plain text. "
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


NO_TOOL_RL_USER_PROMPT_SHA256 = _sha256_text(NO_TOOL_RL_USER_PROMPT)


@dataclass(frozen=True, slots=True)
class NoToolRLPromptIdentity:
    system_prompt_sha256: None
    user_instruction_sha256: str
    bundle_sha256: str
    version: str
    tool_parser: str


NO_TOOL_RL_PROMPT_IDENTITY = NoToolRLPromptIdentity(
    system_prompt_sha256=None,
    user_instruction_sha256=NO_TOOL_RL_USER_PROMPT_SHA256,
    bundle_sha256=_sha256_json(
        {
            "schema": NO_TOOL_RL_PROTOCOL_SCHEMA,
            "source_family": "visual",
            "system_prompt": None,
            "user_suffix": NO_TOOL_RL_USER_PROMPT,
            "tools": [],
            "tool_parser": NO_TOOL_RL_TOOL_PARSER,
            "direct_only": True,
            "final_answer_wrapper": None,
        }
    ),
    version=NO_TOOL_RL_PROMPT_VERSION,
    tool_parser=NO_TOOL_RL_TOOL_PARSER,
)


def build_no_tool_visual_messages(
    question: str, *, image: object = "<image>"
) -> tuple[dict[str, Any], ...]:
    """Return the exact image-bearing, user-only no-tool transcript."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    return (
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question + NO_TOOL_RL_USER_PROMPT},
            ],
        },
    )


__all__ = [
    "NO_TOOL_RL_PROMPT_IDENTITY",
    "NO_TOOL_RL_PROMPT_VERSION",
    "NO_TOOL_RL_PROTOCOL_SCHEMA",
    "NO_TOOL_RL_TOOL_PARSER",
    "NO_TOOL_RL_USER_PROMPT",
    "NO_TOOL_RL_USER_PROMPT_SHA256",
    "NoToolRLPromptIdentity",
    "build_no_tool_visual_messages",
]
