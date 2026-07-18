from __future__ import annotations

import json
from pathlib import Path

import pytest

from tgvf_rl.protocol.native import NativeProtocolRenderer


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "qwen3_native_v1.json"


def _messages():
    user = {
        "role": "user",
        "content": [{"type": "text", "text": "What color is the small label?"}],
    }
    call1 = {
        "role": "assistant",
        "reasoning_content": "I should inspect the label.",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "tgvf_focus_tool",
                    "arguments": {"target": "the small label"},
                },
            }
        ],
    }
    tool1 = {"role": "tool", "content": [{"type": "image"}]}
    call2 = {
        "role": "assistant",
        "reasoning_content": "The first view is unclear; inspect its lower half.",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "tgvf_focus_tool",
                    "arguments": {"target": "the lower half of the small label"},
                },
            }
        ],
    }
    tool2 = {"role": "tool", "content": [{"type": "image"}]}
    answer = {
        "role": "assistant",
        "reasoning_content": "The focused evidence shows blue.",
        "content": "The label is blue.",
    }
    return {
        "prompt": ([user], True),
        "direct": ([user, answer], False),
        "one_call": ([user, call1, tool1, answer], False),
        "two_call": ([user, call1, tool1, call2, tool2, answer], False),
    }


def test_qwen3_local_native_transcript_golden_digests() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    model_path = Path(fixture["model_path"])
    if not model_path.is_dir():
        pytest.skip("accepted local Qwen3 processor is unavailable")
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    renderer = NativeProtocolRenderer(
        processor, expected_tokenizer_length=fixture["tokenizer_length"]
    )
    assert renderer.chat_template_sha256 == fixture["chat_template_sha256"]
    for name, (messages, add_generation_prompt) in _messages().items():
        result = renderer.render(messages, add_generation_prompt=add_generation_prompt)
        expected = fixture["cases"][name]
        assert len(result.token_ids) == expected["length"]
        assert result.token_ids_sha256 == expected["token_ids_sha256"]
        assert result.text_sha256 == expected["text_sha256"]
        assert result.tokenizer_length == fixture["tokenizer_length"]
        if name == "prompt":
            renderer.assert_generation_prefill(result, processor.tokenizer)
