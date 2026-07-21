from __future__ import annotations

import hashlib
import json

import pytest

from tgvf_rl.protocol import (
    CROP_TGVF_TOOL_NAME,
    CROP_TGVF_TOOL_SCHEMA,
    CROP_TGVF_TOOL_SCHEMA_CANONICAL_JSON,
    CROP_TGVF_TOOL_SCHEMA_SHA256,
    IMAGE_ZOOM_IN_TOOL_NAME,
    IMAGE_ZOOM_IN_TOOL_SCHEMA,
    NativeToolCapabilityProfile,
    ParseErrorCode,
    SampledAssistantTurn,
    StrictToolCallParser,
    TGVF_FOCUS_TOOL_NAME,
    TGVF_FOCUS_TOOL_SCHEMA,
    TokenByteSpan,
    ToolCallParseError,
    build_crop_tgvf_tool_schema,
    build_tgvf_focus_tool_schema,
    build_image_zoom_in_tool_schema,
    build_native_tool_schemas,
)


def _character_token_turn(text: str) -> SampledAssistantTurn:
    token_ids: list[int] = []
    spans: list[TokenByteSpan] = []
    byte_cursor = 0
    for index, character in enumerate(text):
        width = len(character.encode("utf-8"))
        token_id = 10_000 + index
        token_ids.append(token_id)
        spans.append(TokenByteSpan(index, token_id, byte_cursor, byte_cursor + width))
        byte_cursor += width
    return SampledAssistantTurn(text, tuple(token_ids), tuple(spans))


def _byte_chunk_turn(text: str, boundaries: list[int]) -> SampledAssistantTurn:
    encoded_length = len(text.encode("utf-8"))
    assert boundaries[0] == 0 and boundaries[-1] == encoded_length
    token_ids = tuple(20_000 + index for index in range(len(boundaries) - 1))
    spans = tuple(
        TokenByteSpan(index, token_ids[index], start, end)
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    )
    return SampledAssistantTurn(text, token_ids, spans)


def _call_text(target: str, *, ensure_ascii: bool = False) -> str:
    payload = json.dumps(
        {"name": TGVF_FOCUS_TOOL_NAME, "arguments": {"target": target}},
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
    )
    return f"</think>\n<tool_call>\n{payload}\n</tool_call>"


def test_fixed_schema_is_exact_and_returned_as_a_fresh_json_object() -> None:
    first = build_tgvf_focus_tool_schema()
    second = build_tgvf_focus_tool_schema()

    assert TGVF_FOCUS_TOOL_SCHEMA["function"]["name"] == TGVF_FOCUS_TOOL_NAME
    assert TGVF_FOCUS_TOOL_SCHEMA["function"]["parameters"]["required"] == ("target",)
    assert first["function"]["name"] == TGVF_FOCUS_TOOL_NAME
    assert first["function"]["parameters"]["required"] == ["target"]
    assert first["function"]["parameters"]["additionalProperties"] is False
    first["function"]["name"] = "mutated"
    assert second["function"]["name"] == TGVF_FOCUS_TOOL_NAME


def test_crop_schema_and_policy_tool_set_are_explicit() -> None:
    first = build_image_zoom_in_tool_schema()
    second = build_image_zoom_in_tool_schema()
    assert IMAGE_ZOOM_IN_TOOL_SCHEMA["function"]["name"] == IMAGE_ZOOM_IN_TOOL_NAME
    assert first["function"]["parameters"]["required"] == ["bbox_2d"]
    assert first["function"]["parameters"]["additionalProperties"] is False
    first["function"]["name"] = "mutated"
    assert second["function"]["name"] == IMAGE_ZOOM_IN_TOOL_NAME
    assert [item["function"]["name"] for item in build_native_tool_schemas()] == [
        TGVF_FOCUS_TOOL_NAME,
        IMAGE_ZOOM_IN_TOOL_NAME,
        CROP_TGVF_TOOL_NAME,
    ]


def test_atomic_crop_tgvf_schema_hash_and_capability_profiles_are_exact() -> None:
    first = build_crop_tgvf_tool_schema()
    second = build_crop_tgvf_tool_schema()
    assert CROP_TGVF_TOOL_SCHEMA["function"]["name"] == CROP_TGVF_TOOL_NAME
    assert first["function"]["parameters"]["required"] == ["bbox_2d", "target"]
    assert first["function"]["parameters"]["additionalProperties"] is False
    assert "exact crop" in first["function"]["description"]
    assert (
        hashlib.sha256(CROP_TGVF_TOOL_SCHEMA_CANONICAL_JSON.encode("utf-8")).hexdigest()
        == CROP_TGVF_TOOL_SCHEMA_SHA256
        == "41f6f99f34b0d3e9fb5b7a4166af5c367cef78214285bc56f12c6ca45e02ceb9"
    )
    first["function"]["name"] = "mutated"
    assert second["function"]["name"] == CROP_TGVF_TOOL_NAME
    assert NativeToolCapabilityProfile.CROP_ONLY.tool_names == (
        IMAGE_ZOOM_IN_TOOL_NAME,
    )
    assert NativeToolCapabilityProfile.TGVF_ONLY.tool_names == (
        TGVF_FOCUS_TOOL_NAME,
    )
    assert NativeToolCapabilityProfile.CROP_TGVF.tool_names == (
        CROP_TGVF_TOOL_NAME,
    )
    assert len(NativeToolCapabilityProfile.CROP_TGVF.tool_set_sha256) == 64


def test_parse_atomic_crop_tgvf_preserves_bbox_and_exact_target_span() -> None:
    target = '红色 "标签" \\ shelf'
    payload = json.dumps(
        {
            "name": CROP_TGVF_TOOL_NAME,
            "arguments": {"bbox_2d": [-3, 2, 40, 31], "target": target},
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    text = f"</think>\n<tool_call>{payload}</tool_call>"
    turn = _character_token_turn(text)
    parsed = StrictToolCallParser(
        enabled_tool_names=NativeToolCapabilityProfile.CROP_TGVF.tool_names
    ).parse(turn)

    assert parsed.name == CROP_TGVF_TOOL_NAME
    assert parsed.bbox_2d == (-3, 2, 40, 31)
    assert parsed.target == target
    assert parsed.raw_tool_call == text[text.index("<tool_call>") :]
    assert parsed.target_span.target_text == target
    assert parsed.target_span.raw_json_value != target
    assert parsed.target_span.token_ids == turn.token_ids[
        parsed.target_span.token_start : parsed.target_span.token_end
    ]


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({"bbox_2d": [0, 0, 2, 2]}, ParseErrorCode.INVALID_ARGUMENTS),
        (
            {"bbox_2d": [0, 0, 2, 2], "target": "x", "extra": 1},
            ParseErrorCode.INVALID_ARGUMENTS,
        ),
        ({"bbox_2d": [0, 0, 0, 2], "target": "x"}, ParseErrorCode.INVALID_BBOX),
        ({"bbox_2d": [0, 0, 2, True], "target": "x"}, ParseErrorCode.INVALID_BBOX),
        ({"bbox_2d": [0, 0, 2, 2], "target": "  "}, ParseErrorCode.EMPTY_TARGET),
    ],
)
def test_invalid_atomic_crop_tgvf_arguments_fail_closed(
    arguments: dict[str, object], code: ParseErrorCode
) -> None:
    payload = json.dumps(
        {"name": CROP_TGVF_TOOL_NAME, "arguments": arguments},
        separators=(",", ":"),
    )
    parser = StrictToolCallParser(
        enabled_tool_names=NativeToolCapabilityProfile.CROP_TGVF.tool_names
    )
    with pytest.raises(ToolCallParseError) as error:
        parser.parse(_character_token_turn(f"<tool_call>{payload}</tool_call>"))
    assert error.value.code is code


def test_atomic_crop_tgvf_call_fails_closed_under_tgvf_only_profile() -> None:
    payload = json.dumps(
        {
            "name": CROP_TGVF_TOOL_NAME,
            "arguments": {"bbox_2d": [0, 0, 2, 2], "target": "label"},
        },
        separators=(",", ":"),
    )
    parser = StrictToolCallParser(
        enabled_tool_names=NativeToolCapabilityProfile.TGVF_ONLY.tool_names
    )
    with pytest.raises(ToolCallParseError) as error:
        parser.parse(_character_token_turn(f"<tool_call>{payload}</tool_call>"))
    assert error.value.code is ParseErrorCode.INVALID_TOOL_NAME


def test_parse_crop_preserves_exact_call_and_integer_bbox() -> None:
    text = (
        "</think>\n<tool_call>"
        '{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[-3,2,40,31]}}'
        "</tool_call>"
    )
    turn = _character_token_turn(text)
    parsed = StrictToolCallParser().parse(turn)
    assert parsed.name == IMAGE_ZOOM_IN_TOOL_NAME
    assert parsed.bbox_2d == (-3, 2, 40, 31)
    assert parsed.raw_tool_call == text[text.index("<tool_call>") :]
    assert parsed.sampled_token_ids == turn.token_ids


@pytest.mark.parametrize(
    "bbox_json",
    ("[1,2,3]", "[1,2,3,4.0]", "[1,2,true,4]", "[3,2,1,4]"),
)
def test_invalid_crop_bbox_fails_closed(bbox_json: str) -> None:
    text = (
        '<tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":'
        f"{bbox_json}}}}}</tool_call>"
    )
    with pytest.raises(ToolCallParseError) as error:
        StrictToolCallParser().parse(_character_token_turn(text))
    assert error.value.code is ParseErrorCode.INVALID_BBOX


def test_parser_can_fail_closed_when_crop_is_disabled() -> None:
    text = (
        '<tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[0,0,2,2]}}</tool_call>'
    )
    parser = StrictToolCallParser(enabled_tool_names=(TGVF_FOCUS_TOOL_NAME,))
    with pytest.raises(ToolCallParseError) as error:
        parser.parse(_character_token_turn(text))
    assert error.value.code is ParseErrorCode.INVALID_TOOL_NAME


def test_parse_preserves_exact_sampled_text_tokens_and_unicode_offsets() -> None:
    text = _call_text('红色 "猫" \\ shelf')
    turn = _character_token_turn(text)

    parsed = StrictToolCallParser().parse(turn)

    assert parsed.sampled_text == text
    assert parsed.sampled_token_ids == turn.token_ids
    assert parsed.sampled_token_byte_spans == turn.token_byte_spans
    assert parsed.raw_tool_call == text[text.index("<tool_call>") :]
    assert (
        parsed.raw_json
        == text[parsed.json_offsets.char_start : parsed.json_offsets.char_end]
    )
    span = parsed.target_span
    assert span.target_text == '红色 "猫" \\ shelf'
    assert span.raw_json_value == text[span.offsets.char_start : span.offsets.char_end]
    assert span.offsets.byte_start == len(
        text[: span.offsets.char_start].encode("utf-8")
    )
    assert span.offsets.byte_end == len(text[: span.offsets.char_end].encode("utf-8"))
    assert span.token_ids == turn.token_ids[span.token_start : span.token_end]


def test_escaped_json_target_span_uses_raw_escaped_bytes_not_fuzzy_text_search() -> (
    None
):
    target = 'same repeated value: "same"'
    text = _call_text(target, ensure_ascii=True)
    parsed = StrictToolCallParser().parse(_character_token_turn(text))

    assert parsed.target == target
    assert '\\"same\\"' in parsed.target_span.raw_json_value
    assert parsed.target_span.raw_json_value != target


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("plain final answer", ParseErrorCode.MISSING_TOOL_CALL),
        ("<tool_call>{}", ParseErrorCode.INCOMPLETE_TOOL_CALL),
        (
            '<tool_call>{"name":"tgvf_focus_tool","arguments":{"target":"a"}}</tool_call>'
            '<tool_call>{"name":"tgvf_focus_tool","arguments":{"target":"b"}}</tool_call>',
            ParseErrorCode.MULTIPLE_TOOL_CALLS,
        ),
        (
            '<tool_call>{"name":"tgvf_focus_tool","arguments":{"target":"a"}}</tool_call>answer',
            ParseErrorCode.TRAILING_ASSISTANT_TEXT,
        ),
        (
            '<tool_call>{"name":"wrong","arguments":{"target":"a"}}</tool_call>',
            ParseErrorCode.INVALID_TOOL_NAME,
        ),
        (
            '<tool_call>{"name":"tgvf_focus_tool","arguments":{"target":"a","extra":1}}</tool_call>',
            ParseErrorCode.INVALID_ARGUMENTS,
        ),
        (
            '<tool_call>{"name":"tgvf_focus_tool","arguments":{"target":"   "}}</tool_call>',
            ParseErrorCode.EMPTY_TARGET,
        ),
        (
            '<tool_call>{"name":"tgvf_focus_tool","name":"tgvf_focus_tool","arguments":{"target":"a"}}</tool_call>',
            ParseErrorCode.INVALID_JSON,
        ),
    ],
)
def test_invalid_calls_fail_closed(text: str, code: ParseErrorCode) -> None:
    with pytest.raises(ToolCallParseError) as error:
        StrictToolCallParser().parse(_character_token_turn(text))
    assert error.value.code is code


def test_explicit_terminal_suffix_is_allowed_but_trailing_answer_is_not() -> None:
    base = _call_text("left sign")
    text = f"{base}\n<|im_end|>\n"
    turn = _character_token_turn(text)

    with pytest.raises(ToolCallParseError, match="trailing_assistant_text"):
        StrictToolCallParser().parse(turn)
    parsed = StrictToolCallParser(allowed_terminal_suffixes=("<|im_end|>",)).parse(turn)
    assert parsed.sampled_text == text


@pytest.mark.parametrize("crossing", ("left", "right", "both"))
def test_target_span_uses_minimal_overlapping_sampled_token_cover(
    crossing: str,
) -> None:
    text = _call_text("target")
    exact = (
        StrictToolCallParser().parse(_character_token_turn(text)).target_span.offsets
    )
    total = len(text.encode("utf-8"))
    if crossing == "left":
        boundaries = [
            0,
            exact.byte_start - 1,
            exact.byte_start + 1,
            exact.byte_end,
            total,
        ]
    elif crossing == "right":
        boundaries = [
            0,
            exact.byte_start,
            exact.byte_end - 1,
            exact.byte_end + 1,
            total,
        ]
    else:
        boundaries = [
            0,
            exact.byte_start - 1,
            exact.byte_start + 1,
            exact.byte_end - 1,
            exact.byte_end + 1,
            total,
        ]
    turn = _byte_chunk_turn(text, boundaries)
    parsed = StrictToolCallParser().parse(turn)
    overlapping = tuple(
        span
        for span in turn.token_byte_spans
        if span.byte_end > exact.byte_start and span.byte_start < exact.byte_end
    )

    assert parsed.target_span.offsets == exact
    assert parsed.target_span.raw_json_value == "target"
    assert parsed.target_span.token_start == overlapping[0].token_index
    assert parsed.target_span.token_end == overlapping[-1].token_index + 1
    assert parsed.target_span.token_ids == tuple(span.token_id for span in overlapping)


def test_sampled_turn_requires_exact_byte_coverage_without_tokenizer_calls() -> None:
    with pytest.raises(ValueError, match="cover sampled_text exactly"):
        SampledAssistantTurn(
            sampled_text="abc",
            token_ids=(1,),
            token_byte_spans=(TokenByteSpan(0, 1, 0, 2),),
        )
