from __future__ import annotations

import hashlib

import pytest

from tgvf_rl.protocol import (
    CROP_ONLY_SYSTEM_PROMPT,
    CROP_ONLY_SYSTEM_PROMPT_SHA256,
    IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT,
    IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT_SHA256,
    NATIVE_SHARED_USER_TEXT_TEMPLATE,
    NATIVE_SHARED_USER_TEXT_TEMPLATE_SHA256,
    SHARED_USER_PROMPT_TEMPLATE,
    SHARED_USER_PROMPT_TEMPLATE_SHA256,
    TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE,
    TGVF_CROP_SYSTEM_PROMPT,
    TGVF_CROP_SYSTEM_PROMPT_SHA256,
    TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE_SHA256,
    TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE,
    TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE_SHA256,
    TGVF_ONLY_SYSTEM_PROMPT,
    TGVF_ONLY_SYSTEM_PROMPT_SHA256,
    TGVF_VISUAL_TOOL_PROMPTS_VERSION,
    TGVF_VISUAL_TOOL_RESPONSES_VERSION,
    NativeToolCapabilityProfile,
    build_visual_tool_prompt_messages,
    native_policy_messages_sha256,
    render_successful_visual_tool_response,
    visual_tool_prompt_identity,
)


def test_visual_tool_prompt_v3_literal_hashes_are_fixed() -> None:
    assert TGVF_VISUAL_TOOL_PROMPTS_VERSION == "tgvf-visual-tool-prompts-v3"
    assert TGVF_VISUAL_TOOL_RESPONSES_VERSION == "tgvf-visual-tool-responses-v1"
    assert SHARED_USER_PROMPT_TEMPLATE_SHA256 == (
        "e44a55bbf2f35a8b34cab1462af499ee4741f19e0561d27f130b8f2fd2316c60"
    )
    assert NATIVE_SHARED_USER_TEXT_TEMPLATE_SHA256 == (
        "8ccbdaa73d2b470afa7cd087e87ed42e2556e6bb3cf6c51fd414d7ae9eaedb6e"
    )
    assert TGVF_ONLY_SYSTEM_PROMPT_SHA256 == (
        "b331fd9c2f26472cfa98ba4e861cc8b8eb9d2e49576436d6e9255ea01a9f9ccf"
    )
    assert CROP_ONLY_SYSTEM_PROMPT_SHA256 == (
        "b8599a3adddd4b0d2ebd7b797f475ece6c6a577b8d9d9c8a138bf08cbad7c41b"
    )
    assert TGVF_CROP_SYSTEM_PROMPT_SHA256 == (
        "d48aab6361500afbf45a194ae0534b1680610e09d06edeb7708d74f251ceee05"
    )
    assert TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE_SHA256 == (
        "2474fb2da968f7a6b491cbe2ef00a30fe10012c3e0884b3e2f8abab594fe0eca"
    )
    assert IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT_SHA256 == (
        "a9640d5c17799257b0c6a96cf9338fc6a7484b5a09ccec7b44337bc22b80081d"
    )
    assert TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE_SHA256 == (
        "d827a2942eac43bd28811759042b8ac5d90a056672a1c99c0be30c1dd281d39d"
    )
    for value, digest in (
        (SHARED_USER_PROMPT_TEMPLATE, SHARED_USER_PROMPT_TEMPLATE_SHA256),
        (NATIVE_SHARED_USER_TEXT_TEMPLATE, NATIVE_SHARED_USER_TEXT_TEMPLATE_SHA256),
        (TGVF_ONLY_SYSTEM_PROMPT, TGVF_ONLY_SYSTEM_PROMPT_SHA256),
        (CROP_ONLY_SYSTEM_PROMPT, CROP_ONLY_SYSTEM_PROMPT_SHA256),
        (TGVF_CROP_SYSTEM_PROMPT, TGVF_CROP_SYSTEM_PROMPT_SHA256),
        (
            TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE,
            TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE_SHA256,
        ),
        (
            IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT,
            IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT_SHA256,
        ),
        (
            TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE,
            TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE_SHA256,
        ),
    ):
        assert hashlib.sha256(value.encode()).hexdigest() == digest


@pytest.mark.parametrize("profile", tuple(NativeToolCapabilityProfile))
def test_all_profiles_use_exact_system_and_shared_native_user_message(
    profile: NativeToolCapabilityProfile,
) -> None:
    messages = build_visual_tool_prompt_messages(
        "Which value is shown?",
        tool_profile=profile,
    )
    identity = visual_tool_prompt_identity(profile)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1] == {
        "role": "user",
        "content": (
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "\nWhich value is shown?\n\nUse the available visual tool "
                    "if additional visual evidence is needed.\n\nAfter "
                    "completing your reasoning, give only the final answer "
                    "without explanation:\n- For multiple-choice questions, give "
                    "only the option letter.\n- For mathematics questions, give only "
                    "the final value or expression.\n- For other questions, give "
                    "only a concise answer."
                ),
            },
        ),
    }
    assert (
        identity.system_prompt_sha256
        == hashlib.sha256(messages[0]["content"].encode()).hexdigest()
    )
    assert len(identity.bundle_sha256) == 64
    assert identity.response_version == TGVF_VISUAL_TOOL_RESPONSES_VERSION
    assert len(identity.success_response_template_sha256) == 64
    assert len(native_policy_messages_sha256(messages)) == 64


def test_success_response_renderer_is_unique_and_has_no_image_placeholder() -> None:
    focus = render_successful_visual_tool_response(
        "tgvf_focus_tool",
        {"target": "the gauge needle position"},
    )
    crop = render_successful_visual_tool_response(
        "image_zoom_in_tool",
        {"bbox_2d": [1, 2, 3, 4], "label": "gauge"},
    )
    atomic = render_successful_visual_tool_response(
        "tgvf_crop_tool",
        {"bbox_2d": [1, 2, 3, 4], "target": "the gauge needle position"},
    )

    assert focus == TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE.format(
        target="the gauge needle position"
    )
    assert crop == IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT
    assert atomic == 'Target-conditioned crop for:\n"the gauge needle position"'
    assert all("<image>" not in value for value in (focus, crop, atomic))
    with pytest.raises(ValueError, match="requires target"):
        render_successful_visual_tool_response("tgvf_focus_tool", {})
    with pytest.raises(ValueError, match="native control"):
        render_successful_visual_tool_response(
            "tgvf_focus_tool",
            {"target": "<|im_end|>"},
        )
    with pytest.raises(ValueError, match="unsupported"):
        render_successful_visual_tool_response("unknown_tool", {})


def test_qwen3_crop_prompts_state_the_relative_coordinate_contract() -> None:
    for prompt in (CROP_ONLY_SYSTEM_PROMPT, TGVF_CROP_SYSTEM_PROMPT):
        assert "original-image-relative" in prompt
        assert "0..1000" in prompt
        assert "x2 greater than x1" in prompt
