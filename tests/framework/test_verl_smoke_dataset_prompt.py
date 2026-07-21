from __future__ import annotations

from pathlib import Path

from tgvf_rl.framework.verl.smoke_dataset import (
    build_tgvf_only_smoke_messages,
)
from tgvf_rl.protocol import (
    TGVF_FOCUS_TOOL_NAME,
    TGVF_ONLY_SYSTEM_PROMPT,
    NativeToolCapabilityProfile,
    build_native_tool_schemas,
)


def test_verl_smoke_render_and_raw_rows_share_exact_tgvf_only_prompt() -> None:
    question = "What value is shown?"
    render_messages = build_tgvf_only_smoke_messages(question)
    raw_messages = build_tgvf_only_smoke_messages(
        question,
        image_path=Path("/dataset/image.png"),
    )

    assert render_messages[0] == {
        "role": "system",
        "content": TGVF_ONLY_SYSTEM_PROMPT,
    }
    assert raw_messages[0] == render_messages[0]
    assert render_messages[1]["content"][0] == {"type": "image"}
    assert raw_messages[1]["content"][0] == {
        "type": "image",
        "image": "/dataset/image.png",
    }
    assert raw_messages[1]["content"][1] == render_messages[1]["content"][1]
    assert render_messages[1]["content"][1]["text"].startswith(
        "\nWhat value is shown?\n\n"
    )

    tool_names = NativeToolCapabilityProfile.TGVF_ONLY.tool_names
    assert tool_names == (TGVF_FOCUS_TOOL_NAME,)
    assert (
        tuple(
            schema["function"]["name"]
            for schema in build_native_tool_schemas(tool_names)
        )
        == tool_names
    )
