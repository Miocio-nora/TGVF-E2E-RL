"""Accepted Policy RL visual-tool prompts and their exact identities.

The literal ``<image>`` in the human-readable shared template is represented
by a native Qwen image content item.  The following text content therefore
starts with the newline that immediately follows ``<image>``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .schema import (
    IMAGE_ZOOM_IN_TOOL_NAME,
    TGVF_CROP_TOOL_NAME,
    TGVF_FOCUS_TOOL_NAME,
    NativeToolCapabilityProfile,
    validate_native_tool_target_for_echo,
)


TGVF_VISUAL_TOOL_PROMPTS_VERSION = "tgvf-visual-tool-prompts-v2"
TGVF_VISUAL_TOOL_RESPONSES_VERSION = "tgvf-visual-tool-responses-v1"

SHARED_USER_PROMPT_TEMPLATE = """<image>
{question}

Use the available visual tool if additional visual evidence is needed.

After completing your reasoning, give only the final answer without explanation:
- For multiple-choice questions, give only the option letter.
- For mathematics questions, give only the final value or expression.
- For other questions, give only a concise answer."""

NATIVE_SHARED_USER_TEXT_TEMPLATE = """
{question}

Use the available visual tool if additional visual evidence is needed.

After completing your reasoning, give only the final answer without explanation:
- For multiple-choice questions, give only the option letter.
- For mathematics questions, give only the final value or expression.
- For other questions, give only a concise answer."""

TGVF_ONLY_SYSTEM_PROMPT = """You are a visual reasoning assistant.

Use tgvf_focus_tool when additional target-conditioned visual evidence
is needed to answer the question.

The target must be a concise, self-contained visual query specifying
both what to inspect and what visual evidence or relation to obtain.
It may request an attribute, text reading, texture, state, count,
comparison, or spatial relation.

Do not provide only an object name, and do not include a guessed final
answer or answer-option value.

After receiving a tool result, continue reasoning. You may call the
tool again if more visual evidence is needed, up to four times. When
sufficient evidence is available, provide a concise final answer.

Example valid tool call:

<tool_call>
{"name":"tgvf_focus_tool","arguments":{"target":"the small circular gauge's needle position for reading its value"}}
</tool_call>"""

CROP_ONLY_SYSTEM_PROMPT = """You are a visual reasoning assistant.

Use image_zoom_in_tool when a relevant object, text, or region is too
small, distant, or visually unclear in the original image.

Select a bounding box that is focused enough to enlarge the relevant
content, but large enough to preserve the context needed to answer the
question.

After receiving the zoomed-in image, continue reasoning. You may call
the tool again if more visual evidence is needed, up to four times.
When sufficient evidence is available, provide a concise final answer.

Example valid tool call:

<tool_call>
{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[120,180,460,620],"label":"the small circular gauge"}}
</tool_call>"""

TGVF_CROP_SYSTEM_PROMPT = """You are a visual reasoning assistant.

Use tgvf_crop_tool when answering the question requires both localized
inspection and target-conditioned visual evidence.

The tool first crops a selected region from the original image and then
adapts the crop's visual representation according to the specified
target.

The bbox_2d argument specifies where to look.

The target argument specifies both what to inspect inside the crop and
what visual evidence or relation to obtain. It may request an attribute,
text reading, texture, state, count, comparison, or spatial relation.

Choose a bounding box that is focused enough to emphasize the relevant
region, but large enough to preserve the context required by the
target. For comparison or relation tasks, include all required entities
inside the crop when possible.

Do not include a guessed final answer or answer-option value in the
target.

After receiving the tool result, continue reasoning. You may call the
tool again if more visual evidence is needed, up to four times. When
sufficient evidence is available, provide a concise final answer.

Example valid tool call:

<tool_call>
{"name":"tgvf_crop_tool","arguments":{"bbox_2d":[120,180,460,620],"target":"the small circular gauge's needle position for reading its value"}}
</tool_call>"""

TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE = (
    'Focused visual observation for target:\n"{target}"'
)
IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT = "Zoomed-in visual observation:"
TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE = 'Target-conditioned crop for:\n"{target}"'


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


SHARED_USER_PROMPT_TEMPLATE_SHA256 = _sha256_text(SHARED_USER_PROMPT_TEMPLATE)
NATIVE_SHARED_USER_TEXT_TEMPLATE_SHA256 = _sha256_text(NATIVE_SHARED_USER_TEXT_TEMPLATE)
TGVF_ONLY_SYSTEM_PROMPT_SHA256 = _sha256_text(TGVF_ONLY_SYSTEM_PROMPT)
CROP_ONLY_SYSTEM_PROMPT_SHA256 = _sha256_text(CROP_ONLY_SYSTEM_PROMPT)
TGVF_CROP_SYSTEM_PROMPT_SHA256 = _sha256_text(TGVF_CROP_SYSTEM_PROMPT)
TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE_SHA256 = _sha256_text(
    TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE
)
IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT_SHA256 = _sha256_text(
    IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT
)
TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE_SHA256 = _sha256_text(
    TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE
)


def _system_prompt(profile: NativeToolCapabilityProfile) -> str:
    if not isinstance(profile, NativeToolCapabilityProfile):
        raise TypeError("tool_profile must be NativeToolCapabilityProfile")
    prompts = {
        NativeToolCapabilityProfile.TGVF_ONLY: TGVF_ONLY_SYSTEM_PROMPT,
        NativeToolCapabilityProfile.CROP_ONLY: CROP_ONLY_SYSTEM_PROMPT,
        NativeToolCapabilityProfile.CROP_TGVF: TGVF_CROP_SYSTEM_PROMPT,
    }
    try:
        return prompts[profile]
    except KeyError as error:  # pragma: no cover - enum expansion guard
        raise ValueError(f"no accepted visual-tool prompt for {profile!r}") from error


def _success_response_template(profile: NativeToolCapabilityProfile) -> str:
    if not isinstance(profile, NativeToolCapabilityProfile):
        raise TypeError("tool_profile must be NativeToolCapabilityProfile")
    responses = {
        NativeToolCapabilityProfile.TGVF_ONLY: TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE,
        NativeToolCapabilityProfile.CROP_ONLY: IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT,
        NativeToolCapabilityProfile.CROP_TGVF: TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE,
    }
    try:
        return responses[profile]
    except KeyError as error:  # pragma: no cover - enum expansion guard
        raise ValueError(f"no accepted visual-tool response for {profile!r}") from error


@dataclass(frozen=True, slots=True)
class VisualToolPromptIdentity:
    """Exact profile-specific system/shared-user prompt bundle identity."""

    tool_profile: NativeToolCapabilityProfile
    system_prompt_sha256: str
    shared_user_prompt_template_sha256: str
    native_user_text_template_sha256: str
    success_response_template_sha256: str
    bundle_sha256: str
    version: str = TGVF_VISUAL_TOOL_PROMPTS_VERSION
    response_version: str = TGVF_VISUAL_TOOL_RESPONSES_VERSION


def visual_tool_prompt_identity(
    profile: NativeToolCapabilityProfile,
) -> VisualToolPromptIdentity:
    """Return hashes for the exact accepted prompt bundle of one tool profile."""

    system_prompt = _system_prompt(profile)
    success_response_template = _success_response_template(profile)
    payload = {
        "version": TGVF_VISUAL_TOOL_PROMPTS_VERSION,
        "response_version": TGVF_VISUAL_TOOL_RESPONSES_VERSION,
        "tool_profile": profile.value,
        "system_prompt": system_prompt,
        "shared_user_prompt_template": SHARED_USER_PROMPT_TEMPLATE,
        "native_shared_user_text_template": NATIVE_SHARED_USER_TEXT_TEMPLATE,
        "success_response_template": success_response_template,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return VisualToolPromptIdentity(
        tool_profile=profile,
        system_prompt_sha256=_sha256_text(system_prompt),
        shared_user_prompt_template_sha256=(SHARED_USER_PROMPT_TEMPLATE_SHA256),
        native_user_text_template_sha256=(NATIVE_SHARED_USER_TEXT_TEMPLATE_SHA256),
        success_response_template_sha256=_sha256_text(success_response_template),
        bundle_sha256=_sha256_text(canonical),
    )


def build_visual_tool_prompt_messages(
    question: str,
    *,
    tool_profile: NativeToolCapabilityProfile,
) -> tuple[Mapping[str, Any], ...]:
    """Build the exact native system/user messages for one policy prompt."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return (
        {"role": "system", "content": _system_prompt(tool_profile)},
        {
            "role": "user",
            "content": (
                {"type": "image"},
                {
                    "type": "text",
                    "text": NATIVE_SHARED_USER_TEXT_TEMPLATE.format(question=question),
                },
            ),
        },
    )


def native_policy_messages_sha256(
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Hash JSON-compatible native message objects without rendering them."""

    canonical = json.dumps(
        list(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_text(canonical)


def render_successful_visual_tool_response(
    tool_name: str,
    arguments: Mapping[str, Any],
) -> str:
    """Render the sole accepted text attached to a successful observation.

    The image or latent observation is appended by the environment boundary;
    this function intentionally emits no image or vision placeholder.
    """

    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("tool_name must be non-empty")
    if not isinstance(arguments, Mapping):
        raise TypeError("tool arguments must be a mapping")
    if tool_name == IMAGE_ZOOM_IN_TOOL_NAME:
        return IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT
    templates = {
        TGVF_FOCUS_TOOL_NAME: TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE,
        TGVF_CROP_TOOL_NAME: TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE,
    }
    try:
        template = templates[tool_name]
    except KeyError as error:
        raise ValueError(f"unsupported visual tool name: {tool_name!r}") from error
    target = arguments.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("successful target-conditioned response requires target")
    validate_native_tool_target_for_echo(target)
    return template.format(target=target)


__all__ = [
    "CROP_ONLY_SYSTEM_PROMPT",
    "CROP_ONLY_SYSTEM_PROMPT_SHA256",
    "IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT",
    "IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT_SHA256",
    "NATIVE_SHARED_USER_TEXT_TEMPLATE",
    "NATIVE_SHARED_USER_TEXT_TEMPLATE_SHA256",
    "SHARED_USER_PROMPT_TEMPLATE",
    "SHARED_USER_PROMPT_TEMPLATE_SHA256",
    "TGVF_CROP_SYSTEM_PROMPT",
    "TGVF_CROP_SYSTEM_PROMPT_SHA256",
    "TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE",
    "TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE_SHA256",
    "TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE",
    "TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE_SHA256",
    "TGVF_ONLY_SYSTEM_PROMPT",
    "TGVF_ONLY_SYSTEM_PROMPT_SHA256",
    "TGVF_VISUAL_TOOL_PROMPTS_VERSION",
    "TGVF_VISUAL_TOOL_RESPONSES_VERSION",
    "VisualToolPromptIdentity",
    "build_visual_tool_prompt_messages",
    "native_policy_messages_sha256",
    "render_successful_visual_tool_response",
    "visual_tool_prompt_identity",
]
