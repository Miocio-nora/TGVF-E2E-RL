from __future__ import annotations

import pytest

from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
    TGVF_DEEPEYES_MATCHED_USER_PROMPT,
    build_tgvf_visual_messages,
)
from tgvf_rl.policy.tgvf_target_guide_v2_protocol import (
    TGVF_TARGET_GUIDE_V2_INSERTION,
    TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY,
    TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT,
    build_tgvf_target_guide_v2_visual_messages,
    remove_tgvf_target_guide_v2,
)


def test_target_guide_v2_normalizes_byte_exactly_to_frozen_short_prompt() -> None:
    assert TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256 == (
        "77ed3a597d2a58e748b70bafe37882760944e293723a28008818a96aad025d0d"
    )
    assert TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256 != (
        TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    )
    assert remove_tgvf_target_guide_v2(TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT) == (
        TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
    )


def test_target_guide_v2_is_teacher_aligned_and_target_only() -> None:
    assert "for reading" not in TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT
    assert "read the gauge" not in TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT
    assert "option C" not in TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT
    assert "close-up of" not in TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT
    assert (
        '"target": "small circular gauge, its needle position, and surrounding '
        'scale markings"'
    ) in TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT
    assert '"printed text below the red warning symbol"' in (
        TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT
    )
    assert TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT.count("# How to call a tool") == 1
    for tag in ("<tool_call>", "</tool_call>"):
        assert TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT.count(tag) == (
            TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT.count(tag)
        ) == 3

    lowered_insertion = TGVF_TARGET_GUIDE_V2_INSERTION.lower()
    for broad_protocol_phrase in (
        "begin every assistant turn",
        "final-only",
        "after any tool observation",
        "observation text",
        "tool-call count",
        "</tool_call>",
        "<think>",
    ):
        assert broad_protocol_phrase not in lowered_insertion


def test_target_guide_v2_changes_only_system_target_conditioning() -> None:
    short = build_tgvf_visual_messages("Which option is correct?", image="image")
    full = build_tgvf_target_guide_v2_visual_messages(
        "Which option is correct?", image="image"
    )

    assert short[1] == full[1]
    assert short[1]["content"][1]["text"].endswith(TGVF_DEEPEYES_MATCHED_USER_PROMPT)
    assert remove_tgvf_target_guide_v2(full[0]["content"]) == short[0]["content"]
    assert TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.user_instruction_sha256 == (
        TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.user_instruction_sha256
    )
    assert TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.tool_parser == (
        TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.tool_parser
    )


def test_target_guide_v2_rejects_ambiguous_normalization_and_empty_question() -> None:
    with pytest.raises(ValueError, match="insertion count differs"):
        remove_tgvf_target_guide_v2(TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT)
    with pytest.raises(ValueError, match="question must be non-empty"):
        build_tgvf_target_guide_v2_visual_messages("  ")
