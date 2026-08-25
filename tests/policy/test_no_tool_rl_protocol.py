from __future__ import annotations

from tgvf_rl.policy.no_tool_rl_protocol import (
    NO_TOOL_RL_PROMPT_IDENTITY,
    NO_TOOL_RL_PROMPT_VERSION,
    NO_TOOL_RL_USER_PROMPT,
    build_no_tool_visual_messages,
)


def test_no_tool_visual_prompt_is_user_only_image_bearing_and_direct() -> None:
    messages = build_no_tool_visual_messages("What is shown?", image="source.png")

    assert messages == (
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "source.png"},
                {
                    "type": "text",
                    "text": "What is shown?" + NO_TOOL_RL_USER_PROMPT,
                },
            ],
        },
    )
    encoded = repr(messages)
    assert "<think>...</think>" in encoded
    assert "tool_call" not in encoded
    assert "target" not in encoded
    assert "bbox" not in encoded
    assert NO_TOOL_RL_PROMPT_IDENTITY.system_prompt_sha256 is None
    assert NO_TOOL_RL_PROMPT_IDENTITY.version == NO_TOOL_RL_PROMPT_VERSION
    assert NO_TOOL_RL_PROMPT_IDENTITY.tool_parser == "none"


def test_no_tool_visual_prompt_rejects_empty_questions() -> None:
    try:
        build_no_tool_visual_messages("  ")
    except ValueError as error:
        assert "non-empty" in str(error)
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("empty question was accepted")
