from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.qwen.base import InjectedForwardRequest, InjectedVisualBlock
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.oracle_d_utility import (
    DEFAULT_THINKING_EOS_TOKEN_IDS,
    ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION,
    OracleDUtilityArm,
    OracleDUtilityGroundTruth,
    OracleDUtilityModelInput,
    _OracleRunLedger,
    _run_identity_payload,
    build_image_only_messages,
    build_oracle_target_messages,
    greedy_oracle_answer,
    greedy_oracle_answers_batched,
    prepare_oracle_arm_context,
    score_oracle_generated_answer,
    split_oracle_d_utility_sample,
)
from tgvf_rl.representation.training.readout import RepresentationVisualTensorBundle
from tgvf_rl.representation.training.schema import (
    RepresentationChoice,
    RepresentationTrainingSample,
)


_EVIDENCE_SENTINEL = "NEVER_RENDER_EVIDENCE_7ad1"
_ANSWER_SENTINEL = "NEVER_RENDER_ANSWER_91b2"


def _sample() -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id="sample-1",
        image="/tmp/image.png",
        image_id="image-1",
        question="What color is the object?",
        target="the object in the center",
        evidence_description=_EVIDENCE_SENTINEL,
        short_answer=_ANSWER_SENTINEL,
        choices=(
            RepresentationChoice(label="A", text="red"),
            RepresentationChoice(label="B", text=_ANSWER_SENTINEL),
        ),
    )


def test_prompt_types_cannot_receive_teacher_post_focus_fields() -> None:
    model_input, ground_truth = split_oracle_d_utility_sample(_sample())

    assert isinstance(model_input, OracleDUtilityModelInput)
    assert isinstance(ground_truth, OracleDUtilityGroundTruth)
    image_only = build_image_only_messages(model_input)
    d_only = build_oracle_target_messages(
        model_input,
        include_source_image=False,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_THINKING,
    )
    image_d = build_oracle_target_messages(
        model_input,
        include_source_image=True,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_THINKING,
    )
    rendered_inputs = json.dumps(
        (image_only, d_only, image_d), ensure_ascii=False, sort_keys=True
    )

    assert _EVIDENCE_SENTINEL not in rendered_inputs
    assert _ANSWER_SENTINEL not in rendered_inputs
    assert model_input.target in rendered_inputs
    assert len(image_only) == 1
    assert len(d_only) == len(image_d) == 3
    assert image_only[0]["content"][0] == {"type": "image"}
    assert d_only[0]["content"] == ({"type": "text", "text": model_input.question},)


def test_scoring_uses_last_thinking_suffix_and_short_answer_only() -> None:
    truth = OracleDUtilityGroundTruth(
        sample_id="s",
        short_answer="white",
        choices=(
            RepresentationChoice(label="A", text="white"),
            RepresentationChoice(label="B", text="black"),
        ),
    )

    exact = score_oracle_generated_answer(
        "old </think> wrong <think>new</think> WHITE<|im_end|>", truth
    )
    choice_only = score_oracle_generated_answer("</think>A. white", truth)
    missing = score_oracle_generated_answer("white", truth)
    capped = score_oracle_generated_answer(
        "</think>white", truth, generation_stop_reason="length_cap"
    )

    assert exact.correct is True
    assert exact.final_answer == "WHITE"
    # The primary gold remains short_answer; choices only map a native label
    # back to that same answer after generation.
    assert choice_only.correct is True
    assert choice_only.candidate_choice_label == "A"
    assert choice_only.route == "multiple_choice_label"
    assert missing.correct is False and missing.route == "missing_final_answer"
    assert capped.correct is False and capped.route == "length_cap"


def test_default_thinking_eos_ids_include_both_native_stops() -> None:
    assert DEFAULT_THINKING_EOS_TOKEN_IDS == (151645, 151643)


def test_run_identity_payload_is_canonical_json_serializable() -> None:
    source = SimpleNamespace(
        source_path=Path("/tmp/eval.toml"),
        source_sha256="a" * 64,
        training_config_sha256="b" * 64,
        artifact_file_sha256="c" * 64,
        artifact_manifest_sha256="d" * 64,
        expected_run_identity_sha256="e" * 64,
        expected_global_step=2000,
        evaluation=SimpleNamespace(eos_token_ids=(151645,), random_seed=42),
    )
    training = SimpleNamespace(
        model=SimpleNamespace(
            model_name="Qwen3-VL-8B-Thinking",
            local_path=Path("/models/Qwen3-VL-8B-Thinking"),
        )
    )
    model_input, _truth = split_oracle_d_utility_sample(_sample())

    payload = _run_identity_payload(
        source_config=source,
        training=training,  # type: ignore[arg-type]
        data_manifest_sha256="f" * 64,
        model_inputs=(model_input,),
        arms=(OracleDUtilityArm.IMAGE_ONLY,),
        max_new_tokens=8,
        eos_token_ids=DEFAULT_THINKING_EOS_TOKEN_IDS,
        decode_mode="cached",
        group_start=0,
        group_limit=1,
        shard_index=0,
        shard_count=1,
    )

    assert payload["model_path"] == "/models/Qwen3-VL-8B-Thinking"
    json.dumps(payload, ensure_ascii=False, sort_keys=True)


@dataclass
class _FakeContext:
    forbidden_multimodal_token_ids: frozenset[int] = frozenset()

    def materialize(self, suffix: tuple[int, ...], runtime: object) -> object:
        return SimpleNamespace(
            input_ids=torch.tensor(((1, 2, *suffix),), dtype=torch.long),
            attention_mask=torch.ones((1, 2 + len(suffix)), dtype=torch.long),
            position_ids=torch.arange(2 + len(suffix)).reshape(1, 1, -1),
        )


class _FakeTokenizer:
    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return "</think>done<|im_end|>" if token_ids == [5] else "unexpected"


class _FakeRenderer:
    def assert_tokenizer_length(self) -> None:
        return None


class _FakeFamily:
    capabilities = SimpleNamespace(native_injected_kv_cache=True)

    def prefill_injected_cache(self, model: object, request: object) -> object:
        logits = torch.zeros((1, 1, 6))
        logits[0, 0, 5] = 10
        return SimpleNamespace(logits=logits, past_key_values=object())


def test_cached_greedy_stops_on_the_second_declared_eos() -> None:
    runtime = SimpleNamespace(
        model=object(), tokenizer=_FakeTokenizer(), renderer=_FakeRenderer()
    )

    output = greedy_oracle_answer(
        context=_FakeContext(),  # type: ignore[arg-type]
        runtime=runtime,  # type: ignore[arg-type]
        family_adapter=_FakeFamily(),  # type: ignore[arg-type]
        eos_token_ids=(4, 5),
        max_new_tokens=8,
        decode_mode="cached",
    )

    assert output.token_ids == (5,)
    assert output.stop_reason == "natural_stop"


def _record(
    identity: str, sample_id: str, arm: str, correct: bool | None
) -> dict[str, object]:
    return {
        "schema_version": ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION,
        "run_identity_sha256": identity,
        "sample_id": sample_id,
        "arm": arm,
        "score": {"correct": correct, "route": "fixture"},
    }


def test_atomic_records_resume_and_rebuild_torn_jsonl(tmp_path: Path) -> None:
    keys = (("s", "control"), ("s", "treatment"))
    identity = {"fixture": 1, "arms": ["control", "treatment"]}
    first = _OracleRunLedger(tmp_path, identity_payload=identity, expected_keys=keys)
    with first.locked():
        first.prepare()
        first.commit(_record(first.identity_sha256, "s", "control", False))
    with (tmp_path / "records.jsonl").open("ab") as handle:
        handle.write(b'{"torn":')

    resumed = _OracleRunLedger(tmp_path, identity_payload=identity, expected_keys=keys)
    with resumed.locked():
        resumed.prepare()
        assert resumed.has("s", "control")
        assert not resumed.has("s", "treatment")
        assert (tmp_path / "records.jsonl").read_bytes().endswith(b"\n")
        resumed.commit(_record(resumed.identity_sha256, "s", "treatment", None))
        summary = resumed.summary()

    assert summary["record_count"] == 2
    assert summary["status"] == "complete"
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["completed_records"] == progress["total_records"] == 2


def test_matched_wrong_arm_contract_keeps_current_target_visible() -> None:
    model_input = OracleDUtilityModelInput(
        sample_id="current",
        image_group_key="image",
        image="/tmp/image.png",
        question="question",
        target="CURRENT_ROW_TARGET",
        sample_content_sha256="a" * 64,
    )
    messages = build_oracle_target_messages(
        model_input,
        include_source_image=False,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_THINKING,
    )

    assert messages[1]["tool_calls"][0]["function"]["arguments"]["target"] == (
        "CURRENT_ROW_TARGET"
    )
    assert OracleDUtilityArm.MATCHED_WRONG_D.value == "matched_wrong_D"


def _visual_bundle(
    fill: float, *, active: bool = True
) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.full((1, 2, 4), fill),
        deepstack=tuple(torch.full((1, 2, 4), fill + index + 1) for index in range(3)),
        branch_layers=(8, 16, 24),
        d_deepstack_active=active,
    )


def test_direct_D_replacement_preserves_native_image_prompt_and_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tgvf_rl.representation.training import oracle_d_utility as module

    expansion = SimpleNamespace(
        canonical_token_ids=(99,),
        canonical_to_model_positions=((0, 1),),
    )
    monkeypatch.setattr(
        module,
        "_render_direct_without_tools",
        lambda _runtime, _messages: ("identical native image prompt", (99,)),
    )
    monkeypatch.setattr(
        module,
        "_expand_native_visual_placeholders",
        lambda _runtime, _ids, *, visual_token_counts: ((99, 99), expansion),
    )
    monkeypatch.setattr(
        module,
        "_qwen3_position_ids",
        lambda _model, *, input_ids, attention_mask, image_grid_thw: torch.zeros(
            (3, 1, input_ids.shape[1]), dtype=torch.long
        ),
    )
    monkeypatch.setattr(
        module, "_qwen3_multimodal_token_ids", lambda _runtime: frozenset()
    )
    runtime = SimpleNamespace(
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _token: 99),
        model=object(),
    )
    model_input, _truth = split_oracle_d_utility_sample(_sample())
    source = _visual_bundle(1.0)
    correct_d = _visual_bundle(10.0)

    image = prepare_oracle_arm_context(
        model_input=model_input,
        arm=OracleDUtilityArm.IMAGE_ONLY,
        runtime=runtime,  # type: ignore[arg-type]
        source=source,
        correct_d=correct_d,
        image_grid_thw=(1, 2, 2),
    )
    replacement = prepare_oracle_arm_context(
        model_input=model_input,
        arm=OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT,
        runtime=runtime,  # type: ignore[arg-type]
        source=source,
        correct_d=correct_d,
        image_grid_thw=(1, 2, 2),
    )

    assert torch.equal(image.prefix_input_ids, replacement.prefix_input_ids)
    assert image.canonical_token_ids_sha256 == replacement.canonical_token_ids_sha256
    assert image.source_positions == replacement.d_positions == (0, 1)
    assert replacement.source_positions == ()
    assert torch.equal(replacement.visual_blocks[0].embeddings, correct_d.main)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            replacement.visual_blocks[0].deepstack,
            correct_d.deepstack,
            strict=True,
        )
    )


def test_direct_D_replacement_fails_closed_without_active_deepstack() -> None:
    model_input, _truth = split_oracle_d_utility_sample(_sample())

    with pytest.raises(ValueError, match="all three DeepStack"):
        prepare_oracle_arm_context(
            model_input=model_input,
            arm=OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT,
            runtime=SimpleNamespace(),  # type: ignore[arg-type]
            source=_visual_bundle(1.0),
            correct_d=_visual_bundle(10.0, active=False),
            image_grid_thw=(1, 2, 2),
        )


@dataclass
class _BatchContext:
    visual_value: float
    prefix_token_ids: tuple[int, ...] = (1, 2, 3, 4)
    forbidden_multimodal_token_ids: frozenset[int] = frozenset()

    def materialize(self, suffix: tuple[int, ...], runtime: object) -> object:
        token_ids = (*self.prefix_token_ids, *suffix)
        sequence = len(token_ids)
        visual = torch.full((1, 2, 6), self.visual_value)
        return InjectedForwardRequest(
            input_ids=torch.tensor((token_ids,), dtype=torch.long),
            attention_mask=torch.ones((1, sequence), dtype=torch.long),
            position_ids=torch.arange(sequence).view(1, 1, sequence).expand(3, -1, -1),
            visual_blocks=(
                InjectedVisualBlock(
                    kind="focused_d",
                    positions=(1, 2),
                    embeddings=visual,
                    deepstack=tuple(
                        visual * float(index + 1) / 10 for index in range(3)
                    ),
                    deepstack_positions=((1, 2), (1, 2), (1, 2)),
                ),
            ),
        )


class _BatchTokenizer:
    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return ",".join(str(token_id) for token_id in token_ids)


class _ScriptedBatchFamily:
    capabilities = SimpleNamespace(native_injected_kv_cache=True)

    def __init__(
        self,
        initial_tokens: tuple[int, ...],
        cached_tokens: tuple[tuple[int, ...], ...],
    ) -> None:
        self.initial_tokens = initial_tokens
        self.cached_tokens = list(cached_tokens)
        self.prefill_batch_sizes: list[int] = []
        self.cached_batch_sizes: list[int] = []

    @staticmethod
    def _result(token_ids: tuple[int, ...]) -> object:
        logits = torch.zeros((len(token_ids), 1, 8))
        for lane, token_id in enumerate(token_ids):
            logits[lane, 0, token_id] = 10
        return SimpleNamespace(logits=logits, past_key_values=object())

    def prefill_injected_cache(self, model: object, request: object) -> object:
        self.prefill_batch_sizes.append(int(request.input_ids.shape[0]))
        return self._result(self.initial_tokens)

    def forward_cached_token(self, model: object, request: object) -> object:
        self.cached_batch_sizes.append(int(request.input_ids.shape[0]))
        return self._result(self.cached_tokens.pop(0))


def test_batched_cached_greedy_tracks_lane_eos_and_uses_one_decode_wave() -> None:
    runtime = SimpleNamespace(
        model=object(), tokenizer=_BatchTokenizer(), renderer=_FakeRenderer()
    )
    family = _ScriptedBatchFamily(
        initial_tokens=(5, 3),
        cached_tokens=((4, 5),),
    )

    outputs = greedy_oracle_answers_batched(
        contexts=(_BatchContext(1.0), _BatchContext(2.0)),  # type: ignore[arg-type]
        runtime=runtime,  # type: ignore[arg-type]
        family_adapter=family,  # type: ignore[arg-type]
        eos_token_ids=(5,),
        max_new_tokens=4,
    )

    assert tuple(output.token_ids for output in outputs) == ((5,), (3, 5))
    assert tuple(output.stop_reason for output in outputs) == (
        "natural_stop",
        "natural_stop",
    )
    assert family.prefill_batch_sizes == [2]
    assert family.cached_batch_sizes == [2]


def test_batched_cached_greedy_rejects_nonfinite_next_token_logits() -> None:
    runtime = SimpleNamespace(
        model=object(), tokenizer=_BatchTokenizer(), renderer=_FakeRenderer()
    )
    family = _ScriptedBatchFamily(initial_tokens=(1, 2), cached_tokens=())

    def nonfinite_result(_model: object, request: object) -> object:
        family.prefill_batch_sizes.append(int(request.input_ids.shape[0]))
        return SimpleNamespace(
            logits=torch.full((2, 1, 8), float("nan")),
            past_key_values=object(),
        )

    family.prefill_injected_cache = nonfinite_result  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="invalid logits"):
        greedy_oracle_answers_batched(
            contexts=(_BatchContext(1.0), _BatchContext(2.0)),  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            family_adapter=family,  # type: ignore[arg-type]
            eos_token_ids=(5,),
            max_new_tokens=4,
        )


class _TinyCache:
    def __init__(self, cumulative: torch.Tensor, sequence_length: int) -> None:
        self.cumulative = cumulative
        self.sequence_length = sequence_length

    def get_seq_length(self) -> int:
        return self.sequence_length


class _TinyCachedLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 6)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        past_key_values: _TinyCache | None = None,
        use_cache: bool = False,
        **_kwargs: object,
    ) -> object:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)
        hidden = inputs_embeds.clone()
        if deepstack_visual_embeds is not None:
            assert visual_pos_masks is not None
            for branch in deepstack_visual_embeds:
                hidden = hidden.clone()
                hidden[visual_pos_masks] += branch
        cumulative = hidden.cumsum(dim=1)
        previous_length = 0
        if past_key_values is not None:
            cumulative = cumulative + past_key_values.cumulative
            previous_length = past_key_values.get_seq_length()
        output = hidden + cumulative * 0.03
        cache = (
            _TinyCache(
                cumulative[:, -1:].clone(),
                previous_length + hidden.shape[1],
            )
            if use_cache
            else None
        )
        return SimpleNamespace(last_hidden_state=output, past_key_values=cache)


class _TinyCachedQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(language_model=_TinyCachedLanguageModel())
        self.lm_head = nn.Linear(6, 64, bias=False)


class _RecordingQwen3Adapter(Qwen3VLAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.prefill_batch_sizes: list[int] = []
        self.cached_batch_sizes: list[int] = []

    def prefill_injected_cache(self, model: object, request: object) -> object:
        self.prefill_batch_sizes.append(int(request.input_ids.shape[0]))
        return super().prefill_injected_cache(model, request)  # type: ignore[arg-type]

    def forward_cached_token(self, model: object, request: object) -> object:
        self.cached_batch_sizes.append(int(request.input_ids.shape[0]))
        return super().forward_cached_token(model, request)  # type: ignore[arg-type]


def test_tiny_qwen3_cached_batch_matches_scalar_greedy_tokens() -> None:
    torch.manual_seed(501)
    model = _TinyCachedQwen().eval()
    model.requires_grad_(False)
    runtime = SimpleNamespace(
        model=model, tokenizer=_BatchTokenizer(), renderer=_FakeRenderer()
    )
    contexts = (_BatchContext(0.2), _BatchContext(0.7))
    scalar_adapter = _RecordingQwen3Adapter()
    scalar = tuple(
        greedy_oracle_answer(
            context=context,  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            family_adapter=scalar_adapter,
            eos_token_ids=(63,),
            max_new_tokens=3,
            decode_mode="cached",
        )
        for context in contexts
    )
    batched_adapter = _RecordingQwen3Adapter()

    batched = greedy_oracle_answers_batched(
        contexts=contexts,  # type: ignore[arg-type]
        runtime=runtime,  # type: ignore[arg-type]
        family_adapter=batched_adapter,
        eos_token_ids=(63,),
        max_new_tokens=3,
    )

    assert tuple(output.token_ids for output in batched) == tuple(
        output.token_ids for output in scalar
    )
    assert tuple(output.stop_reason for output in batched) == tuple(
        output.stop_reason for output in scalar
    )
    assert scalar_adapter.prefill_batch_sizes == [1, 1]
    assert batched_adapter.prefill_batch_sizes == [2]
    assert all(size == 2 for size in batched_adapter.cached_batch_sizes)
