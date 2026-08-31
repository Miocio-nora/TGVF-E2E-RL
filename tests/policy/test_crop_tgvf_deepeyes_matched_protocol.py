from __future__ import annotations

import json

from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_MAXIMUM_TOOL_CALLS,
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
    CROP_TGVF_DEEPEYES_MATCHED_TOOL_NAME,
    CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT,
    build_crop_tgvf_tool_response_message,
    build_crop_tgvf_visual_messages,
)
from tgvf_rl.policy.deepeyes_official_protocol import (
    DEEPEYES_MAX_ACTIVE_PERCEPTION,
    USER_PROMPT_V2,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
)
from tgvf_rl.protocol.schema import (
    NativeToolCapabilityProfile,
    build_tgvf_crop_tool_schema,
)


def _without_tool_specific_regions(value: str) -> str:
    before_tools, after_tools_open = value.split("<tools>\n", 1)
    _tool_schema, after_tools = after_tools_open.split("\n</tools>", 1)
    before_example, after_example_open = after_tools.split(
        "**Example**:  \n<tool_call>  \n", 1
    )
    _example, after_example = after_example_open.split("  \n</tool_call>", 1)
    return before_tools + "<TOOL_SCHEMA>" + before_example + "<EXAMPLE>" + after_example


def test_atomic_system_prompt_changes_only_deepeyes_tool_contract() -> None:
    assert _without_tool_specific_regions(
        CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
    ) == _without_tool_specific_regions(TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT)
    visible_schema = CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT.split("<tools>\n", 1)[
        1
    ].split("\n</tools>", 1)[0]
    assert json.loads(visible_schema) == build_tgvf_crop_tool_schema()
    assert CROP_TGVF_DEEPEYES_MATCHED_TOOL_NAME == "tgvf_crop_tool"
    assert "bbox_2d" in CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
    assert "target" in CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
    assert "up to four" not in CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT
    assert "<answer>" not in CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT


def test_atomic_user_suffix_is_the_mechanical_deepeyes_tool_swap() -> None:
    assert CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT == USER_PROMPT_V2.replace(
        "image_zoom_in_tool", "tgvf_crop_tool"
    )
    assert "<answer>" not in CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT


def test_atomic_visual_messages_use_one_clean_final_tool() -> None:
    messages = build_crop_tgvf_visual_messages(
        "Which option is correct?", image="source-rgb"
    )
    assert messages == (
        {
            "role": "system",
            "content": CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "source-rgb"},
                {
                    "type": "text",
                    "text": (
                        "Which option is correct?"
                        + CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT
                    ),
                },
            ],
        },
    )


def test_atomic_observation_is_d_only_and_does_not_echo_target() -> None:
    message = build_crop_tgvf_tool_response_message(observation="crop-latent-d")
    assert message == {
        "role": "tool",
        "name": "tgvf_crop_tool",
        "content": [
            {"type": "image", "image": "crop-latent-d"},
            {"type": "text", "text": CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT},
        ],
    }
    assert "target" not in message["content"][1]["text"]
    assert "Target-conditioned crop for" not in message["content"][1]["text"]


def test_atomic_prompt_identity_binds_schema_and_six_call_budget() -> None:
    identity = CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY
    assert len(identity.bundle_sha256) == 64
    assert identity.tool_parser == "hermes"
    assert identity.tool_schema_sha256 == (
        NativeToolCapabilityProfile.CROP_TGVF.tool_set_sha256
    )
    assert (
        identity.maximum_tool_calls
        == CROP_TGVF_DEEPEYES_MATCHED_MAXIMUM_TOOL_CALLS
        == DEEPEYES_MAX_ACTIVE_PERCEPTION
        == 6
    )
