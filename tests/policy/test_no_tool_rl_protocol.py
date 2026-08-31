from __future__ import annotations

import pytest

from tgvf_rl.policy.no_tool_rl_protocol import (
    NO_TOOL_RL_PROMPT_IDENTITY,
    NO_TOOL_RL_PROMPT_VERSION,
    NO_TOOL_RL_USER_PROMPT,
    NO_TOOL_RL_USER_PROMPT_SHA256,
    build_no_tool_visual_messages,
)


def test_no_tool_visual_prompt_is_exact_user_only_image_bearing_and_direct() -> None:
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
    assert NO_TOOL_RL_USER_PROMPT_SHA256 == (
        "e8251aee2ed7e51de02c03f2591279c3ae568dc6a34b920c66dd5f1b559b3410"
    )
    assert NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256 == (
        "8010ec191b5b9147baaf8363ed9826250151568e55bcbb80693bd759d5bb8593"
    )
    assert NO_TOOL_RL_PROMPT_IDENTITY.system_prompt_sha256 is None
    assert NO_TOOL_RL_PROMPT_IDENTITY.version == NO_TOOL_RL_PROMPT_VERSION
    assert NO_TOOL_RL_PROMPT_IDENTITY.tool_parser == "none"


@pytest.mark.parametrize("question", ("", "  ", 3))
def test_no_tool_visual_prompt_rejects_invalid_questions(question: object) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_no_tool_visual_messages(question)  # type: ignore[arg-type]
