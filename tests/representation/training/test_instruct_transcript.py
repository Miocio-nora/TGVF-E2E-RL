from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from tgvf_rl.protocol.native import NativeAssistantDialect, NativeProtocolRenderer
from tgvf_rl.protocol.schema import build_representation_tgvf_focus_tool_schema
from tgvf_rl.representation.training.native_pipeline import (
    RepresentationPromptConfig,
    build_native_representation_messages,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.transcript import render_native_evidence_labels
from tgvf_rl.tokenizer_invariants import effective_tokenizer_length


_MODEL_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct")


@pytest.mark.skipif(not _MODEL_PATH.is_dir(), reason="pinned Instruct model is absent")
def test_real_instruct_representation_transcript_has_no_synthetic_think() -> None:
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        _MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False,
    )
    renderer = NativeProtocolRenderer(
        processor,
        expected_tokenizer_length=151_669,
        tool_schemas=(build_representation_tgvf_focus_tool_schema(),),
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    prompt = RepresentationPromptConfig(
        identity="qwen3-representation-image-question-v1",
        template="{question}",
        expected_sha256=sha256(b"{question}").hexdigest(),
    )
    sample = RepresentationTrainingSample(
        sample_id="instruct-golden",
        image="/fixture/image.png",
        question="What word appears on the status label?",
        target="status label text",
        evidence_description="The status label reads OPEN.",
        short_answer="OPEN",
    )
    messages = build_native_representation_messages(
        sample,
        prompt,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )

    initial = renderer.render(messages[:1], add_generation_prompt=True)
    renderer.assert_generation_prefill(initial, renderer.tokenizer)
    evidence = render_native_evidence_labels(
        renderer,
        messages,
        evidence_description=sample.evidence_description,
    )

    assert renderer.chat_template_sha256 == (
        "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
    )
    assert initial.text.endswith("<|im_start|>assistant\n")
    assert "<think>" not in evidence.transcript.text
    assert "</think>" not in evidence.transcript.text
    assert messages[1]["content"] == ""
    assert "reasoning_content" not in messages[1]
    assert messages[3]["content"] == "The status label reads OPEN.\n\nOPEN"
    assert evidence.transcript.text_sha256 == (
        "ec2a7a44ec39d40de1ac16193cc7e5e21f3433a25d91e4ff6b83f0d6df3e5415"
    )
    assert evidence.transcript.token_ids_sha256 == (
        "a70bc198d5b592022ce0e1a06c1c0c3651787b738e0cad33d5f890e3b0cf9e04"
    )
    assert evidence.evidence_token_positions == tuple(range(221, 227))
    assert tuple(
        evidence.transcript.token_ids[position]
        for position in evidence.evidence_token_positions
    ) == (785, 2639, 2383, 15804, 29841, 382)
    assert effective_tokenizer_length(renderer.tokenizer) == 151_669
