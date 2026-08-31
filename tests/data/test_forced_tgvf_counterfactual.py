from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.data.forced_tgvf_counterfactual import (
    FORCED_TGVF_RUN_SCHEMA,
    ForcedTGVFRunPlan,
    ForcedTGVFSample,
    _arxiv_option_count,
    attempt_seed,
    build_forced_policy_action_target,
    build_forced_policy_prefix,
    finalize_forced_tgvf_attempts,
    load_forced_tgvf_schedule,
    question_target_proxy,
    sample_injected_answers_batched,
    score_forced_tgvf_answer,
)
from tgvf_rl.data.forced_tgvf_prefix import finalize_forced_tgvf_prefix
from tgvf_rl.data.tgvf_tool_utility import (
    TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA,
    TGVF_TOOL_UTILITY_SCHEDULE_ROW_SCHEMA,
    TGVF_TOOL_UTILITY_SCHEDULE_SCHEMA,
)
from tgvf_rl.protocol import (
    NativeAssistantDialect,
    NativeProtocolRenderer,
    NativeToolCapabilityProfile,
    build_native_tool_schemas,
)
from tgvf_rl.qwen.base import InjectedForwardRequest, InjectedVisualBlock
from tgvf_rl.representation.training.oracle_d_utility import OracleGeneratedAnswer


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _line(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _schedule(
    tmp_path: Path, count: int = 4
) -> tuple[Path, list[dict[str, object]], str]:
    root = tmp_path / "schedule"
    root.mkdir()
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.append(
            {
                "schema_version": TGVF_TOOL_UTILITY_SCHEDULE_ROW_SCHEMA,
                "training_index": index,
                "optimizer_step": index // 2 + 1,
                "prompt_index_in_step": index % 2,
                "is_canary": index < 2,
                "sample_id": f"sample:{index}",
                "candidate_sha256": _sha(f"candidate:{index}".encode()),
                "data_source": "vstar",
                "task_kind": "open",
                "image": {
                    "path": str((tmp_path / f"image-{index}.png").resolve()),
                    "sha256": _sha(f"image:{index}".encode()),
                },
                "question": f"Question {index}?",
                "ground_truth": f"answer {index}",
                "p_full": 0.5,
                "full_attempt_counts": {
                    "correct": 4,
                    "expected": 8,
                    "observed": 8,
                    "scoreable": 8,
                },
            }
        )
    schedule_payload = b"".join(_line(row) for row in rows)
    (root / "schedule.jsonl").write_bytes(schedule_payload)
    identity = {
        "schema_version": TGVF_TOOL_UTILITY_SCHEDULE_SCHEMA,
        "dataset": {
            "iteration_identity_sha256": "a" * 64,
            "sample_count": count,
        },
        "schedule": {"sample_count": count},
        "files": {
            "schedule": {
                "path": "schedule.jsonl",
                "rows": count,
                "sha256": _sha(schedule_payload),
            }
        },
    }
    manifest_sha = _sha(_canonical(identity))
    (root / "schedule-manifest.json").write_bytes(
        _line({**identity, "manifest_sha256": manifest_sha})
    )
    return root, rows, manifest_sha


def _sample(
    index: int = 0, *, source: str = "vstar", question: str = "Question?"
) -> ForcedTGVFSample:
    return ForcedTGVFSample(
        training_index=index,
        sample_id=f"sample:{index}",
        candidate_sha256=_sha(f"candidate:{index}".encode()),
        data_source=source,
        task_kind="mcq" if source == "arxivqa" else "open",
        image_path=Path(f"/tmp/image-{index}.png"),
        image_sha256=_sha(f"image:{index}".encode()),
        question=question,
        ground_truth="B" if source == "arxivqa" else "answer",
    )


def _plan(sample: ForcedTGVFSample, *, attempts: int = 8) -> ForcedTGVFRunPlan:
    identity = {
        "run_id": "forced-fixture",
        "sampling": {"attempts_per_sample": attempts, "master_seed": 42},
        "execution": {"shard_count": 1},
    }
    return ForcedTGVFRunPlan(
        "forced-fixture",
        _sha(_canonical(identity)),
        identity,
        (sample,),
    )


def test_schedule_loads_exact_first_n_in_declared_order(tmp_path: Path) -> None:
    root, rows, manifest_sha = _schedule(tmp_path)
    samples, _manifest, observed_sha = load_forced_tgvf_schedule(root, sample_count=3)

    assert [sample.sample_id for sample in samples] == [
        row["sample_id"] for row in rows[:3]
    ]
    assert [sample.training_index for sample in samples] == [0, 1, 2]
    assert observed_sha == manifest_sha


def test_question_target_proxy_is_explicitly_newline_free() -> None:
    assert question_target_proxy("  Which one?\nA. first\nB. second ") == (
        "Which one? A. first B. second"
    )
    with pytest.raises(ValueError, match="native control"):
        question_target_proxy("What is <|im_end|> shown?")


def test_attempt_seed_is_shard_independent_and_attempt_specific() -> None:
    sample = _sample()
    plan = _plan(sample)
    observed = [attempt_seed(plan, sample, index) for index in range(8)]

    assert len(set(observed)) == 8
    assert observed == [attempt_seed(plan, sample, index) for index in range(8)]


def test_arxiv_option_count_reconstructs_the_original_row_bound() -> None:
    question = "Which?\nChoices:\nA. alpha\nB. beta\nC. gamma"
    assert _arxiv_option_count(question) == 3
    with pytest.raises(ValueError, match="contiguous"):
        _arxiv_option_count("Which?\nChoices:\nA. alpha\nC. gamma")


class _UnusedJudge:
    provider = SimpleNamespace(
        judge=lambda _request: (_ for _ in ()).throw(AssertionError)
    )


def test_rule_first_scoring_uses_arxiv_row_bound_without_judge() -> None:
    sample = _sample(
        source="arxivqa",
        question="Which?\nChoices:\nA. alpha\nB. beta\nC. gamma",
    )
    result = score_forced_tgvf_answer(
        plan=_plan(sample),
        sample=sample,
        attempt_index=0,
        generated=OracleGeneratedAnswer((7,), "B<|im_end|>", "natural_stop"),
        bound_judge=_UnusedJudge(),
    )

    assert result["status"] == "scored"
    assert result["correct"] is True
    assert result["verification"]["route"] == "arxivqa_rule"
    assert result["verification"]["judge_used"] is False


def test_length_cap_is_audited_and_conservatively_scored_incorrect() -> None:
    sample = _sample()
    result = score_forced_tgvf_answer(
        plan=_plan(sample),
        sample=sample,
        attempt_index=0,
        generated=OracleGeneratedAnswer((7,), "unfinished", "length_cap"),
        bound_judge=_UnusedJudge(),
    )

    assert result["status"] == "scored"
    assert result["correct"] is False
    assert result["verification"] == {
        "route": "max_new_tokens_exhausted_scored_incorrect_v1",
        "evidence": (
            "counterfactual answer did not reach a configured EOS; "
            "the length-capped attempt is conservatively scored incorrect"
        ),
        "judge_used": False,
        "generation_stop_reason": "length_cap",
    }


@dataclass
class _BatchContext:
    forbidden_multimodal_token_ids: frozenset[int] = frozenset()

    def materialize(
        self, suffix: tuple[int, ...], _runtime: object
    ) -> InjectedForwardRequest:
        token_ids = (1, 2, *suffix)
        sequence = len(token_ids)
        visual = torch.ones((1, 1, 4))
        return InjectedForwardRequest(
            input_ids=torch.tensor((token_ids,), dtype=torch.long),
            attention_mask=torch.ones((1, sequence), dtype=torch.long),
            position_ids=torch.arange(sequence).view(1, 1, sequence).expand(3, -1, -1),
            visual_blocks=(
                InjectedVisualBlock(
                    kind="focused_d",
                    positions=(1,),
                    embeddings=visual,
                    deepstack=(visual, visual, visual),
                    deepstack_positions=((1,), (1,), (1,)),
                ),
            ),
        )


class _BatchTokenizer:
    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return ",".join(str(value) for value in token_ids)


class _OneTokenFamily:
    capabilities = SimpleNamespace(native_injected_kv_cache=True)

    def __init__(self) -> None:
        self.prefill_batch_size = 0

    def prefill_injected_cache(self, _model: object, request: object) -> object:
        self.prefill_batch_size = int(request.input_ids.shape[0])
        logits = torch.full((self.prefill_batch_size, 1, 8), -1000.0)
        logits[:, 0, 5] = 1000.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def test_eight_attempt_sampler_shares_one_prefill_and_keeps_lane_seeds() -> None:
    family = _OneTokenFamily()
    outputs = sample_injected_answers_batched(
        context=_BatchContext(),  # type: ignore[arg-type]
        attempt_seeds=tuple(range(8)),
        runtime=SimpleNamespace(model=object(), tokenizer=_BatchTokenizer()),
        family_adapter=family,  # type: ignore[arg-type]
        eos_token_ids=(5,),
        max_new_tokens=2,
    )

    assert family.prefill_batch_size == 8
    assert [output.token_ids for output in outputs] == [(5,)] * 8
    assert all(output.stop_reason == "natural_stop" for output in outputs)


def test_finalize_merges_modulo_shards_into_sidecar_attempt_order(
    tmp_path: Path,
) -> None:
    schedule, rows, schedule_sha = _schedule(tmp_path, count=4)
    output = tmp_path / "output"
    output.mkdir()
    identity = {
        "schema_version": FORCED_TGVF_RUN_SCHEMA,
        "run_id": "forced-fixture",
        "schedule": {"sample_count": 4, "manifest_sha256": schedule_sha},
        "sampling": {"attempts_per_sample": 2},
        "execution": {"shard_count": 2},
    }
    run_sha = _sha(_canonical(identity))
    (output / "run-identity.json").write_bytes(
        _line(
            {
                "schema_version": FORCED_TGVF_RUN_SCHEMA,
                "run_id": "forced-fixture",
                "run_identity_sha256": run_sha,
                "identity": identity,
            }
        )
    )
    for shard in range(2):
        ledger = output / "shards" / f"shard-{shard:02d}" / "ledger.jsonl"
        ledger.parent.mkdir(parents=True)
        records = []
        for row in rows:
            if int(row["training_index"]) % 2 != shard:
                continue
            for attempt_index in range(2):
                records.append(
                    {
                        "schema_version": TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA,
                        "run_id": "forced-fixture",
                        "run_identity_sha256": run_sha,
                        "sample_id": row["sample_id"],
                        "training_index": row["training_index"],
                        "attempt_index": attempt_index,
                        "status": "scored",
                        "correct": attempt_index == 0,
                    }
                )
        ledger.write_bytes(b"".join(_line(record) for record in records))

    result = finalize_forced_tgvf_attempts(
        schedule,
        output,
        run_id="forced-fixture",
        sample_count=4,
        attempts_per_sample=2,
        shard_count=2,
    )
    attempts = [
        json.loads(line)
        for line in Path(result["attempts_path"]).read_text().splitlines()
    ]

    assert [(row["training_index"], row["attempt_index"]) for row in attempts] == [
        (index, attempt) for index in range(4) for attempt in range(2)
    ]
    assert result["attempt_count"] == 8


def _full_run_with_ledgers(
    tmp_path: Path,
    *,
    sample_count: int = 6,
    attempts_per_sample: int = 2,
    shard_count: int = 2,
) -> tuple[Path, Path, str]:
    schedule, rows, schedule_sha = _schedule(tmp_path, count=sample_count)
    output = tmp_path / "full-output"
    output.mkdir()
    identity = {
        "schema_version": FORCED_TGVF_RUN_SCHEMA,
        "run_id": "forced-full-fixture",
        "schedule": {
            "sample_count": sample_count,
            "manifest_sha256": schedule_sha,
            "ordered_samples": [
                {
                    "training_index": row["training_index"],
                    "sample_id": row["sample_id"],
                    "candidate_sha256": row["candidate_sha256"],
                    "image_sha256": row["image"]["sha256"],  # type: ignore[index]
                }
                for row in rows
            ],
        },
        "sampling": {"attempts_per_sample": attempts_per_sample},
        "execution": {
            "shard_assignment": "training_index_mod_shard_count_v1",
            "shard_count": shard_count,
            "attempt_count": sample_count * attempts_per_sample,
        },
    }
    run_sha = _sha(_canonical(identity))
    (output / "run-identity.json").write_bytes(
        _line(
            {
                "schema_version": FORCED_TGVF_RUN_SCHEMA,
                "run_id": "forced-full-fixture",
                "run_identity_sha256": run_sha,
                "identity": identity,
            }
        )
    )
    for shard in range(shard_count):
        ledger = output / "shards" / f"shard-{shard:02d}" / "ledger.jsonl"
        ledger.parent.mkdir(parents=True)
        records = []
        for row in rows:
            if int(row["training_index"]) % shard_count != shard:
                continue
            for attempt_index in range(attempts_per_sample):
                records.append(
                    {
                        "schema_version": TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA,
                        "run_id": "forced-full-fixture",
                        "run_identity_sha256": run_sha,
                        "sample_id": row["sample_id"],
                        "training_index": row["training_index"],
                        "attempt_index": attempt_index,
                        "status": "scored",
                        "correct": attempt_index == 0,
                    }
                )
        ledger.write_bytes(b"".join(_line(record) for record in records))
    return schedule, output, run_sha


def test_prefix_finalize_extracts_canonical_complete_prefix_from_full_ledgers(
    tmp_path: Path,
) -> None:
    schedule, output, run_sha = _full_run_with_ledgers(tmp_path)

    result = finalize_forced_tgvf_prefix(
        schedule,
        output,
        run_id="forced-full-fixture",
        prefix_sample_count=4,
        attempts_per_sample=2,
        shard_count=2,
    )
    attempts_path = Path(result["attempts_path"])
    attempts = [json.loads(line) for line in attempts_path.read_text().splitlines()]

    assert [(row["training_index"], row["attempt_index"]) for row in attempts] == [
        (index, attempt) for index in range(4) for attempt in range(2)
    ]
    assert all(row["run_identity_sha256"] == run_sha for row in attempts)
    assert result["parent_sample_count"] == 6
    assert result["attempt_count"] == 8
    assert attempts_path == output / "prefixes/first-000004/attempts.jsonl"

    repeated = finalize_forced_tgvf_prefix(
        schedule,
        output,
        run_id="forced-full-fixture",
        prefix_sample_count=4,
        attempts_per_sample=2,
        shard_count=2,
    )
    assert repeated["manifest_sha256"] == result["manifest_sha256"]


def test_prefix_finalize_rejects_missing_or_unscored_prefix_attempt(
    tmp_path: Path,
) -> None:
    schedule, output, _run_sha = _full_run_with_ledgers(tmp_path)
    ledger = output / "shards/shard-01/ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    records = [
        row
        for row in records
        if not (row["training_index"] == 3 and row["attempt_index"] == 1)
    ]
    ledger.write_bytes(b"".join(_line(record) for record in records))

    with pytest.raises(ValueError, match="prefix attempts are incomplete"):
        finalize_forced_tgvf_prefix(
            schedule,
            output,
            run_id="forced-full-fixture",
            prefix_sample_count=4,
            attempts_per_sample=2,
            shard_count=2,
        )

    records.append(
        {
            "schema_version": TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA,
            "run_id": "forced-full-fixture",
            "run_identity_sha256": _run_sha,
            "sample_id": "sample:3",
            "training_index": 3,
            "attempt_index": 1,
            "status": "judge_failed",
            "correct": None,
        }
    )
    ledger.write_bytes(b"".join(_line(record) for record in records))
    with pytest.raises(ValueError, match="contains an unscored attempt"):
        finalize_forced_tgvf_prefix(
            schedule,
            output,
            run_id="forced-full-fixture",
            prefix_sample_count=4,
            attempts_per_sample=2,
            shard_count=2,
        )


def test_prefix_finalize_rejects_tampered_parent_identity(tmp_path: Path) -> None:
    schedule, output, _run_sha = _full_run_with_ledgers(tmp_path)
    identity_path = output / "run-identity.json"
    record = json.loads(identity_path.read_text())
    record["identity"]["execution"]["attempt_count"] += 1
    identity_path.write_bytes(_line(record))

    with pytest.raises(ValueError, match="parent identity differs"):
        finalize_forced_tgvf_prefix(
            schedule,
            output,
            run_id="forced-full-fixture",
            prefix_sample_count=4,
            attempts_per_sample=2,
            shard_count=2,
        )


@pytest.mark.skipif(
    not Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct").is_dir(),
    reason="pinned Qwen3-VL-Instruct processor is absent",
)
def test_real_instruct_forced_prefix_is_the_formal_policy_transcript() -> None:
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct",
        local_files_only=True,
        trust_remote_code=False,
    )
    renderer = NativeProtocolRenderer(
        processor,
        expected_tokenizer_length=151_669,
        tool_names=NativeToolCapabilityProfile.TGVF_ONLY.tool_names,
        tool_schemas=build_native_tool_schemas(
            NativeToolCapabilityProfile.TGVF_ONLY.tool_names
        ),
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    sample = _sample(question="What color is it?")
    action = build_forced_policy_action_target(
        sample=sample,
        target=sample.question,
        policy_renderer=renderer,
    )
    token_ids, text, metadata = build_forced_policy_prefix(
        sample=sample,
        target=sample.question,
        policy_renderer=renderer,
    )

    assert token_ids
    forced_turns = text[text.index("<|im_start|>assistant\n") :]
    assert forced_turns.count("<tool_call>") == 1
    assert forced_turns.count("<tool_response>") == 1
    assert forced_turns.count("<|image_pad|>") == 1
    assert text.count("<|image_pad|>") == 2
    assert text.endswith("<|im_start|>assistant\n")
    assert '"target": "What color is it?"' in text
    assert metadata["target_strategy"] == "question_target_proxy_v1"
    assert action.target_text == "What color is it?"
    assert action.transcript.text.startswith("<|im_start|>system\n")
    assert action.transcript.text.endswith("<|im_end|>\n")
    assert action.sampled_turn.sampled_text.startswith("<think>")
    assert (
        action.canonical_target_token_ids
        == action.transcript.token_ids[
            action.canonical_target_span.start : action.canonical_target_span.end
        ]
    )

    unicode_sample = _sample(question="How does η change with Lmin?")
    unicode_action = build_forced_policy_action_target(
        sample=unicode_sample,
        target=unicode_sample.question,
        policy_renderer=renderer,
    )
    assert unicode_action.target_text == unicode_sample.question
    assert (
        unicode_action.canonical_target_token_ids
        == unicode_action.transcript.token_ids[
            unicode_action.canonical_target_span.start : unicode_action.canonical_target_span.end
        ]
    )
