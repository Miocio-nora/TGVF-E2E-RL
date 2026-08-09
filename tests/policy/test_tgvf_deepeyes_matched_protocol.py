from __future__ import annotations

from tgvf_rl.policy.deepeyes_official_protocol import USER_PROMPT_V2
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
    TGVF_DEEPEYES_MATCHED_TOOL_NAME,
    TGVF_DEEPEYES_MATCHED_USER_PROMPT,
    build_tgvf_tool_response_message,
    build_tgvf_visual_messages,
)


def test_matched_user_suffix_changes_only_tool_name() -> None:
    assert TGVF_DEEPEYES_MATCHED_USER_PROMPT == USER_PROMPT_V2.replace(
        "image_zoom_in_tool", "tgvf_focus_tool"
    )
    assert "<answer>" not in TGVF_DEEPEYES_MATCHED_USER_PROMPT


def test_matched_system_prompt_has_only_tgvf_specific_contract() -> None:
    assert TGVF_DEEPEYES_MATCHED_TOOL_NAME in TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
    assert "target" in TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
    assert "guessed final answer or answer-option value" in TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
    assert "up to four" not in TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
    assert "<answer>" not in TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT


def test_visual_messages_use_plain_final_matched_prompt() -> None:
    messages = build_tgvf_visual_messages("Which option is correct?", image="img")
    assert messages[0] == {
        "role": "system",
        "content": TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
    }
    assert messages[1]["content"] == [
        {"type": "image", "image": "img"},
        {
            "type": "text",
            "text": "Which option is correct?" + TGVF_DEEPEYES_MATCHED_USER_PROMPT,
        },
    ]


def test_tool_observation_does_not_echo_target_into_text_channel() -> None:
    message = build_tgvf_tool_response_message(observation="latent-d")
    assert message == {
        "role": "tool",
        "name": "tgvf_focus_tool",
        "content": [
            {"type": "image", "image": "latent-d"},
            {"type": "text", "text": TGVF_DEEPEYES_MATCHED_USER_PROMPT},
        ],
    }
    assert "target" not in message["content"][1]["text"]


def test_prompt_identity_is_content_addressed() -> None:
    assert len(TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256) == 64
    assert TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.tool_parser == "hermes"
