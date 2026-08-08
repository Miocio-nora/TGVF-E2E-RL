from __future__ import annotations

import pytest

from tgvf_rl.policy.deepeyes_official_protocol import (
    DEEPEYES_THINKLITE_AGENT_NAME,
    DEEPEYES_VISUAL_AGENT_NAME,
    SYSTEM_PROMPT_V2,
    SYSTEM_PROMPT_V2_SHA256,
    THINKLITE_BOXED_INSTRUCTION,
    THINKLITE_PROMPT_IDENTITY,
    USER_PROMPT_V2,
    USER_PROMPT_V2_SHA256,
    VISUAL_PROMPT_IDENTITY,
    agent_name_for_source,
    build_thinklite_messages,
    build_visual_messages,
    build_visual_tool_response_message,
    direct_answer_after_last_tool_call,
    parse_hermes_crop_call,
    tools_kwargs_for_visual_row,
)


def test_official_v2_prompt_exact_identities_are_frozen() -> None:
    assert SYSTEM_PROMPT_V2.startswith("You are a helpful assistant.\n\n# Tools")
    assert '"required":["bbox"]' in SYSTEM_PROMPT_V2
    assert SYSTEM_PROMPT_V2.endswith("</tool_call>")
    assert USER_PROMPT_V2 == (
        "\nThink first, call **image_zoom_in_tool** if needed, then answer. "
        "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> "
        "(if tools needed), followed by the final answer directly as plain text. "
    )
    assert SYSTEM_PROMPT_V2_SHA256 == (
        "1fc5b8b5ebdc9b24d6a9281071222872c8542dd65a4a4be1e70d9760c3a7f99f"
    )
    assert USER_PROMPT_V2_SHA256 == (
        "eac6399e048fc406c5a10fc44dd2f8d0c43c252e6f305b38844519ac71dbcfb0"
    )
    assert VISUAL_PROMPT_IDENTITY.bundle_sha256 == (
        "2b8b6d799ebe4bbfd6b3830344850575141b2293750f857c031a2031426c0dd2"
    )
    assert THINKLITE_PROMPT_IDENTITY.bundle_sha256 == (
        "72171d0d540de99ffecf75fd6d5820fc91f312c2079296bf5a270c2e7a8eea0b"
    )


def test_visual_messages_and_tool_observation_keep_official_prompts_visible() -> None:
    marker = object()
    messages = build_visual_messages("Where is the cup?", image=marker)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT_V2}
    assert messages[1]["content"] == [
        {"type": "image", "image": marker},
        {"type": "text", "text": "Where is the cup?" + USER_PROMPT_V2},
    ]
    crop = object()
    assert build_visual_tool_response_message(image=crop) == {
        "role": "tool",
        "name": "image_zoom_in_tool",
        "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": USER_PROMPT_V2},
        ],
    }


def test_thinklite_is_image_bearing_single_turn_task_routed_and_has_no_tool() -> None:
    marker = object()
    assert build_thinklite_messages("2+2?", image=marker, task_kind="math") == (
        {
            "role": "user",
            "content": [
                {"type": "image", "image": marker},
                {"type": "text", "text": "2+2? " + THINKLITE_BOXED_INSTRUCTION},
            ],
        },
    )
    direct = build_thinklite_messages("Color?", image=marker, task_kind="open")
    assert direct[0]["content"][0] == {"type": "image", "image": marker}
    assert THINKLITE_BOXED_INSTRUCTION in direct[0]["content"][1]["text"]
    assert "<answer>" not in direct[0]["content"][1]["text"]
    assert agent_name_for_source("thinklite") == DEEPEYES_THINKLITE_AGENT_NAME
    assert agent_name_for_source("vstar") == DEEPEYES_VISUAL_AGENT_NAME
    assert agent_name_for_source("arxivqa") == DEEPEYES_VISUAL_AGENT_NAME


def test_hermes_json_contract_accepts_bbox_2d_and_rejects_qwen3_coder_shape() -> None:
    call = (
        '<think>inspect</think><tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[10,20,100,200],"label":"cup"}}'
        "</tool_call>"
    )
    assert parse_hermes_crop_call(call) == {
        "name": "image_zoom_in_tool",
        "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "cup"},
    }
    with pytest.raises(ValueError, match="strict JSON"):
        parse_hermes_crop_call(
            "<tool_call><function=image_zoom_in_tool>"
            "<parameter=bbox_2d>[10,20,100,200]</parameter></function></tool_call>"
        )
    with pytest.raises(ValueError, match="bbox_2d"):
        parse_hermes_crop_call(
            '<tool_call>{"name":"image_zoom_in_tool",'
            '"arguments":{"bbox":[10,20,100,200]}}</tool_call>'
        )


def test_clean_direct_final_tail_preserves_answer_over_action_precedence() -> None:
    action = '<think>zoom</think><tool_call>{"name":"image_zoom_in_tool"}</tool_call>'
    assert direct_answer_after_last_tool_call(action) is None
    assert direct_answer_after_last_tool_call(action + " blue") == "blue"
    assert direct_answer_after_last_tool_call("blue") is None


def test_per_row_tools_kwargs_shape_is_exact() -> None:
    assert tools_kwargs_for_visual_row(()) == {
        "image_zoom_in_tool": {"create_kwargs": {"gt_regions": ()}}
    }
    assert tools_kwargs_for_visual_row([[1, 2, 10, 20]]) == {
        "image_zoom_in_tool": {"create_kwargs": {"gt_regions": ((1, 2, 10, 20),)}}
    }
    with pytest.raises(ValueError, match="four integers"):
        tools_kwargs_for_visual_row([[1, 2, 3]])
