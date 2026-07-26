from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tgvf_rl.data.policy_selection import (
    AttemptStatus,
    SelectionBranch,
    SelectionCandidate,
    stable_selection_request_id,
)
from tgvf_rl.data.policy_selection_runtime import (
    GenerationDisposition,
    T1_MAX_PIXELS,
    T1_PROMPT_SCHEMA,
    T1_RAW_GENERATION_SCHEMA,
    T1_RUN_CONFIG_SCHEMA,
    T1_SHARD_COUNT,
    T1RawGenerationEvidence,
    VerificationOutcome,
    candidate_rank,
    classify_generation_finish,
    derive_t1_attempt_seed,
    evidence_to_attempt_record,
    extract_direct_completion,
    extract_final_answer,
    load_resumable_chunk,
    load_t1_run_config,
    native_prompt_identity_sha256,
    native_user_message_descriptor,
    sampled_token_ids_sha256,
    source_rgb_sha256,
    verify_arxivqa_answer,
    verify_thinklite_answer,
    verify_vstar_answer,
    write_content_addressed_chunk,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_record(tmp_path: Path) -> dict[str, object]:
    sources = []
    for source in ("vstar", "arxivqa", "thinklite"):
        path = tmp_path / f"{source}.jsonl"
        payload = (json.dumps({"source": source}) + "\n").encode()
        path.write_bytes(payload)
        sources.append(
            {
                "source": source,
                "path": str(path),
                "sha256": _sha256(payload),
                "rows": 1,
            }
        )

    selected_records = []
    selected_manifest_rows = []
    for source in ("vstar", "arxivqa", "thinklite"):
        for index in range(64):
            candidate_record = {
                "schema_version": "tgvf.policy-selection.candidate.v1",
                "sample_id": f"fixture:{source}:{index}",
                "source": source,
                "question": f"Question {source} {index}?",
                "ground_truth": "A",
                "image": {
                    "path": str(tmp_path / f"image-{source}-{index}.png"),
                    "sha256": f"{index + 1:064x}",
                    "width": 32,
                    "height": 32,
                },
                "gt_regions": [],
                "provenance": {"fixture": True, "index": index},
            }
            candidate = SelectionCandidate.from_record(candidate_record)
            selected_records.append(candidate_record)
            selected_manifest_rows.append(
                {"candidate_sha256": candidate.identity_sha256}
            )
    selection_payload = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for record in selected_records
    )
    selection_path = tmp_path / "selection.jsonl"
    selection_path.write_bytes(selection_payload)
    selection_manifest = {
        "schema_version": "tgvf.policy-selection.t1-canary-manifest.v1",
        "selection_algorithm_version": "t1-canary-content-hash-v1",
        "selection_is_outcome_independent": True,
        "selected": selected_manifest_rows,
    }
    selection_manifest_path = tmp_path / "selection-manifest.json"
    selection_manifest_payload = json.dumps(
        selection_manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    selection_manifest_path.write_bytes(selection_manifest_payload)
    return {
        "schema_version": T1_RUN_CONFIG_SCHEMA,
        "run_id": "t1-canary-fixture",
        "model": {
            "repository": "Qwen/Qwen3-VL-8B-Thinking",
            "path": "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking",
            "dtype": "bfloat16",
            "config_sha256": "1" * 64,
            "generation_config_sha256": "9" * 64,
            "tokenizer_config_sha256": "2" * 64,
            "tokenizer_json_sha256": "a" * 64,
            "preprocessor_config_sha256": "3" * 64,
            "chat_template_sha256": "4" * 64,
            "tokenizer_length": 151_669,
            "eos_token_id": 151_645,
            "generation_eos_token_ids": [151_645, 151_643],
            "trust_remote_code": True,
            "quantization": None,
        },
        "prompt": {
            "schema": T1_PROMPT_SCHEMA,
            "user_content_order": ["image", "question"],
            "no_system": True,
            "no_tools": True,
            "add_generation_prompt": True,
        },
        "image": {
            "min_pixels": None,
            "max_pixels": T1_MAX_PIXELS,
            "resize_factor": 32,
            "resample": "transformers-fast-bicubic",
            "pre_resize": False,
            "processor_do_resize": True,
            "preserve_aspect_ratio": True,
            "limit_image_per_prompt": 1,
            "color_mode": "RGB",
            "alpha_handling": "pil-convert-rgb-discard-alpha-v1",
            "source_pixel_hash_schema": "tgvf.policy-selection.source-rgb-pixels.v1",
        },
        "sampling": {
            "attempts": 8,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_processors": [],
            "seed_root": 17,
            "seed_namespace": "t1-canary-fixture-v1",
            "do_sample": True,
            "ignore_eos": False,
            "stop_token_ids": [151_645],
            "effective_stop_token_ids": [151_645, 151_643],
            "stop_strings": [],
            "include_stop_str_in_output": False,
            "detokenize": True,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        },
        "response_budgets": [
            {"revision": 0, "max_model_len": 65_536, "max_new_tokens": 40_960},
            {"revision": 1, "max_model_len": 131_072, "max_new_tokens": 98_304},
            {"revision": 2, "max_model_len": 262_144, "max_new_tokens": 196_608},
        ],
        "runtime": {
            "backend": "vllm",
            "version": "0.12.0",
            "python": "3.12.11",
            "torch": "2.8.0",
            "transformers": "4.57.3",
            "pillow": "12.3.0",
            "flashinfer": "0.5.3",
            "world_size": 4,
            "tensor_parallel_size": 1,
            "max_num_seqs": 32,
            "gpu_memory_utilization": 0.9,
            "mm_encoder_attn_backend": "TORCH_SDPA",
            "decoder_attn_backend": "FLASHINFER",
            "max_num_batched_tokens": 65_536,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "mm_processor_cache_gb": 4.0,
            "engine_seed": 42,
            "chunk_candidates": 4,
            "max_inflight": 64,
            "retain_token_ids": True,
            "generation_config_mode": "auto",
        },
        "data": {"sources": sources},
        "selection": {
            "kind": "stratified_canary",
            "algorithm_version": "t1-canary-content-hash-v1",
            "candidates_path": str(selection_path),
            "candidates_sha256": _sha256(selection_payload),
            "rows": 192,
            "manifest_path": str(selection_manifest_path),
            "manifest_sha256": _sha256(selection_manifest_payload),
        },
        "verifier": {
            "schema": "t1-source-verifier-v1",
            "answer_parser": "last-think-suffix-v1",
            "arxivqa_rule": "row-bounded-a-z-v1",
            "thinklite_rule": "normalized-exact-numeric-v1",
            "vstar_rule": "normalized-exact-v1",
            "semantic_judge": {
                "provider": "local-openai-compatible",
                "repository": "Qwen/Qwen2.5-72B-Instruct",
                "path": str(tmp_path / "Qwen2.5-72B-Instruct"),
                "served_name": "Qwen2.5-72B-Instruct",
                "prompt_sha256": "5" * 64,
                "config_sha256": "6" * 64,
                "temperature": 0.0,
                "max_tokens": 256,
                "remote": False,
            },
        },
        "output_root": str(tmp_path / "output"),
    }


def _load_run(tmp_path: Path, record: dict[str, object] | None = None):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(record or _run_record(tmp_path)), encoding="utf-8")
    return load_t1_run_config(path, verify_data_files=True)


def _candidate_for_rank(rank: int) -> str:
    value = rank
    candidate = f"{value:064x}"
    assert candidate_rank(candidate, world_size=T1_SHARD_COUNT) == rank
    return candidate


def _raw_record(
    run,
    *,
    rank: int = 1,
    attempt_index: int = 2,
    budget_revision: int = 0,
    raw_text: str = "reasoning</think> B",
    finish_reason: str = "stop",
) -> dict[str, object]:
    candidate_sha256 = _candidate_for_rank(rank)
    budget = run.budget(budget_revision)
    token_ids = [] if finish_reason == "error" else [101, 102, 103]
    record: dict[str, object] = {
        "schema_version": T1_RAW_GENERATION_SCHEMA,
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "request_id": stable_selection_request_id(
            candidate_sha256=candidate_sha256,
            branch=SelectionBranch.FULL_IMAGE,
            attempt_index=attempt_index,
        ),
        "sample_id": f"sample-{rank}",
        "candidate_sha256": candidate_sha256,
        "source": "arxivqa",
        "branch": "full_image",
        "attempt_index": attempt_index,
        "attempt_seed": run.attempt_seed(
            candidate_sha256=candidate_sha256, attempt_index=attempt_index
        ),
        "budget_revision": budget_revision,
        "max_model_len": budget.max_model_len,
        "max_new_tokens": budget.max_new_tokens,
        "prompt_sha256": "7" * 64,
        "rendered_prompt_token_ids_sha256": "8" * 64,
        "prompt_token_count": 23,
        "image_sha256": "8" * 64,
        "image_evidence": {
            "source_width": 640,
            "source_height": 448,
            "source_mode": "RGBA",
            "source_rgb_sha256": "9" * 64,
            "processed_width": 448,
            "processed_height": 448,
        },
        "sampled_token_ids_sha256": sampled_token_ids_sha256(token_ids),
        "sampled_token_count": len(token_ids),
        "sampled_token_ids": token_ids,
        "raw_text": "" if finish_reason == "error" else raw_text,
        "finish_reason": finish_reason,
        "stop_reason": None if finish_reason != "stop" else run.model["eos_token_id"],
        "backend": {
            "name": run.runtime["backend"],
            "version": run.runtime["version"],
            "runtime_sha256": run.runtime_identity_sha256,
            "model_sha256": run.model_identity_sha256,
            "processor_sha256": run.processor_identity_sha256,
        },
    }
    if finish_reason == "error":
        record["generation_error"] = "backend request failed"
    return record


def test_strict_run_config_is_content_addressed_and_checks_sources(
    tmp_path: Path,
) -> None:
    record = _run_record(tmp_path)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(record), encoding="utf-8")
    second_path.write_text(
        json.dumps(record, sort_keys=True, indent=2), encoding="utf-8"
    )

    first = load_t1_run_config(first_path, verify_data_files=True)
    second = load_t1_run_config(second_path, verify_data_files=True)

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.response_budgets[-1].max_new_tokens == 196_608
    assert first.model_identity_sha256 != first.processor_identity_sha256
    assert first.as_record() == record


def test_run_config_accepts_exact_qwen3_vl_instruct_repository_path_pair(
    tmp_path: Path,
) -> None:
    record = _run_record(tmp_path)
    record["model"]["repository"] = "Qwen/Qwen3-VL-8B-Instruct"
    record["model"]["path"] = (
        "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct"
    )
    record["verifier"]["answer_parser"] = "direct-completion-v1"

    run = _load_run(tmp_path, record)

    assert run.model["repository"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert run.model["path"] == (
        "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct"
    )


@pytest.mark.parametrize(
    ("repository", "path", "message"),
    [
        (
            "Qwen/Qwen3-VL-8B-Instruct",
            "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking",
            "model.path must be.*Qwen3-VL-8B-Instruct",
        ),
        (
            "Qwen/Qwen3-VL-8B-Unknown",
            "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Unknown",
            "model.repository is not an accepted Qwen3-VL edition",
        ),
    ],
)
def test_run_config_rejects_cross_wired_or_unknown_qwen3_vl_editions(
    tmp_path: Path, repository: str, path: str, message: str
) -> None:
    record = _run_record(tmp_path)
    record["model"]["repository"] = repository
    record["model"]["path"] = path
    config_path = tmp_path / "bad-model.json"
    config_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_t1_run_config(config_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.update({"unknown": True}), "fields differ"),
        (
            lambda record: record["sampling"].update({"top_p": 0.95}),
            "top_p must be 1.0",
        ),
        (
            lambda record: record["image"].update({"max_pixels": 512}),
            "max_pixels must be 262144",
        ),
        (
            lambda record: record["prompt"].update({"no_tools": False}),
            "no_tools must be true",
        ),
    ],
)
def test_run_config_rejects_unknown_or_changed_scientific_inputs(
    tmp_path: Path, mutation, message: str
) -> None:
    record = _run_record(tmp_path)
    mutation(record)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_t1_run_config(path)


def test_run_config_rejects_duplicate_json_keys_and_source_drift(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_t1_run_config(duplicate)

    record = _run_record(tmp_path)
    source_path = Path(record["data"]["sources"][0]["path"])
    source_path.write_text("changed\n", encoding="utf-8")
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate SHA-256 mismatch"):
        load_t1_run_config(path, verify_data_files=True)


def test_attempt_seed_and_candidate_rank_are_stable_and_order_independent() -> None:
    run_sha = "a" * 64
    candidate_sha = "b" * 64
    first = derive_t1_attempt_seed(
        run_manifest_sha256=run_sha,
        candidate_sha256=candidate_sha,
        attempt_index=3,
        seed_root=9,
        seed_namespace="fixture",
    )
    assert first == derive_t1_attempt_seed(
        candidate_sha256=candidate_sha,
        run_manifest_sha256=run_sha,
        seed_namespace="fixture",
        seed_root=9,
        attempt_index=3,
    )
    assert 0 <= first < 2**31 - 1
    assert first != derive_t1_attempt_seed(
        run_manifest_sha256=run_sha,
        candidate_sha256=candidate_sha,
        attempt_index=4,
        seed_root=9,
        seed_namespace="fixture",
    )
    assert candidate_rank(candidate_sha, world_size=4) == int(candidate_sha, 16) % 4
    with pytest.raises(ValueError, match="attempt_index must be <= 7"):
        derive_t1_attempt_seed(
            run_manifest_sha256=run_sha,
            candidate_sha256=candidate_sha,
            attempt_index=8,
        )


def test_native_prompt_is_user_only_image_then_question_and_path_independent() -> None:
    messages = native_user_message_descriptor(
        image="/data/image.png", question="What is shown?"
    )
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/data/image.png"},
                {"type": "text", "text": "What is shown?"},
            ],
        }
    ]
    identity = native_prompt_identity_sha256(
        question="What is shown?",
        image_sha256="a" * 64,
        chat_template_sha256="b" * 64,
    )
    assert len(identity) == 64
    assert identity != native_prompt_identity_sha256(
        question="What else is shown?",
        image_sha256="a" * 64,
        chat_template_sha256="b" * 64,
    )


def test_source_rgb_hash_binds_dimensions_and_exact_pixels() -> None:
    pixels = bytes(range(12))
    identity = source_rgb_sha256(width=2, height=2, pixel_bytes=pixels)
    assert len(identity) == 64
    assert identity != source_rgb_sha256(
        width=2, height=2, pixel_bytes=bytes(reversed(pixels))
    )
    with pytest.raises(ValueError, match="RGB pixel byte length"):
        source_rgb_sha256(width=2, height=2, pixel_bytes=b"short")


def test_raw_evidence_retains_tokens_text_finish_resize_and_backend(
    tmp_path: Path,
) -> None:
    run = _load_run(tmp_path)
    record = _raw_record(run)
    evidence = T1RawGenerationEvidence.from_record(record)
    evidence.validate_against_run(run)

    assert evidence.sampled_token_ids == (101, 102, 103)
    assert evidence.raw_text.endswith(" B")
    assert (evidence.processed_width, evidence.processed_height) == (448, 448)
    assert evidence.source_mode == "RGBA"
    assert evidence.disposition is GenerationDisposition.COMPLETED
    assert evidence.as_record() == record
    assert len(evidence.evidence_sha256) == 64

    without_ids = deepcopy(record)
    del without_ids["sampled_token_ids"]
    assert T1RawGenerationEvidence.from_record(without_ids).sampled_token_ids is None


def test_raw_evidence_fails_closed_on_token_seed_budget_or_backend_drift(
    tmp_path: Path,
) -> None:
    run = _load_run(tmp_path)
    token_drift = _raw_record(run)
    token_drift["sampled_token_ids"] = [999]
    with pytest.raises(ValueError, match="token IDs SHA-256 mismatch"):
        T1RawGenerationEvidence.from_record(token_drift)

    seed_drift = T1RawGenerationEvidence.from_record(_raw_record(run))
    changed = deepcopy(seed_drift.as_record())
    changed["attempt_seed"] += 1
    with pytest.raises(ValueError, match="attempt seed mismatch"):
        T1RawGenerationEvidence.from_record(changed).validate_against_run(run)

    changed = _raw_record(run)
    changed["backend"]["version"] = "other"
    with pytest.raises(ValueError, match="backend identity mismatch"):
        T1RawGenerationEvidence.from_record(changed).validate_against_run(run)


def test_final_answer_uses_last_think_closer_and_length_is_never_wrong() -> None:
    assert extract_final_answer("a</think> old b</think>  final <|im_end|>") == "final"
    assert extract_final_answer("a</think> final <|endoftext|>") == "final"
    assert extract_final_answer("no closer") is None
    assert extract_final_answer("thought</think>  ") is None
    assert classify_generation_finish("stop") is GenerationDisposition.COMPLETED
    assert classify_generation_finish("length") is GenerationDisposition.TRUNCATED


def test_instruct_direct_completion_does_not_require_think_markers() -> None:
    assert extract_direct_completion("  The answer is B. <|im_end|>  ") == (
        "The answer is B."
    )
    assert extract_direct_completion("  ") is None


def test_arxivqa_rule_is_a_to_z_and_row_bounded() -> None:
    assert verify_arxivqa_answer(
        "The final answer is J.", "J", option_count=10
    ).outcome is (VerificationOutcome.CORRECT)
    assert verify_arxivqa_answer("K", "J", option_count=10).outcome is (
        VerificationOutcome.INCORRECT
    )
    assert verify_arxivqa_answer(
        "because this is ambiguous", "B", option_count=4
    ).outcome is (VerificationOutcome.INCORRECT)
    with pytest.raises(ValueError, match="outside the row option range"):
        verify_arxivqa_answer("B", "J", option_count=4)


def test_thinklite_exact_numeric_and_semantic_boundaries() -> None:
    assert verify_thinklite_answer("<answer>  BLUE </answer>", "blue").outcome is (
        VerificationOutcome.CORRECT
    )
    assert verify_thinklite_answer("0.5", r"\frac{1}{2}").outcome is (
        VerificationOutcome.CORRECT
    )
    assert verify_thinklite_answer("25%", "0.5").outcome is (
        VerificationOutcome.INCORRECT
    )
    assert verify_thinklite_answer(
        "two blue objects", "there are two blue objects"
    ).outcome is (VerificationOutcome.SEMANTIC_REQUIRED)


def test_vstar_exact_mismatch_requires_semantics_instead_of_becoming_wrong() -> None:
    assert verify_vstar_answer("LEFT", "left").outcome is VerificationOutcome.CORRECT
    assert verify_vstar_answer("on the left", "left").outcome is (
        VerificationOutcome.SEMANTIC_REQUIRED
    )
    assert verify_vstar_answer(None, "left").outcome is VerificationOutcome.INCORRECT


def test_attempt_conversion_retries_length_and_preserves_unresolved_semantics(
    tmp_path: Path,
) -> None:
    run = _load_run(tmp_path)
    length = T1RawGenerationEvidence.from_record(
        _raw_record(run, raw_text="thinking", finish_reason="length")
    )
    assert (
        evidence_to_attempt_record(length, expected_answer="B", option_count=4) is None
    )
    exhausted = evidence_to_attempt_record(
        length, expected_answer="B", option_count=4, budget_exhausted=True
    )
    assert exhausted["status"] == AttemptStatus.TRUNCATED.value
    assert exhausted["correct"] is None

    vstar_record = _raw_record(run, raw_text="thought</think> on the left")
    vstar_record["source"] = "vstar"
    vstar = T1RawGenerationEvidence.from_record(vstar_record)
    unresolved = evidence_to_attempt_record(vstar, expected_answer="left")
    assert unresolved["status"] == AttemptStatus.VERIFIER_ERROR.value
    assert unresolved["semantic_required"] is True
    with pytest.raises(ValueError, match="requires semantic_judge_evidence_sha256"):
        evidence_to_attempt_record(vstar, expected_answer="left", semantic_verdict=True)
    judged = evidence_to_attempt_record(
        vstar,
        expected_answer="left",
        semantic_verdict=True,
        semantic_judge_evidence_sha256="c" * 64,
    )
    assert judged["status"] == AttemptStatus.SCORED.value
    assert judged["correct"] is True


def test_missing_final_answer_is_scoreable_incorrect_not_truncated(
    tmp_path: Path,
) -> None:
    run = _load_run(tmp_path)
    raw = T1RawGenerationEvidence.from_record(
        _raw_record(run, raw_text="thinking without a closer")
    )
    attempt = evidence_to_attempt_record(raw, expected_answer="B", option_count=4)
    assert attempt["status"] == AttemptStatus.SCORED.value
    assert attempt["correct"] is False
    assert attempt["verification_route"] == "arxivqa_missing_final_answer"


def test_generation_error_never_becomes_incorrect(tmp_path: Path) -> None:
    run = _load_run(tmp_path)
    raw = T1RawGenerationEvidence.from_record(_raw_record(run, finish_reason="error"))
    attempt = evidence_to_attempt_record(raw, expected_answer="B", option_count=4)
    assert attempt["status"] == AttemptStatus.GENERATION_ERROR.value
    assert attempt["correct"] is None


def test_content_addressed_chunk_is_atomic_resumable_and_validated(
    tmp_path: Path,
) -> None:
    run = _load_run(tmp_path)
    first = T1RawGenerationEvidence.from_record(
        _raw_record(run, rank=1, attempt_index=1)
    )
    second = T1RawGenerationEvidence.from_record(
        _raw_record(run, rank=1, attempt_index=0)
    )
    manifest = write_content_addressed_chunk(
        run.output_root,
        [first, second],
        run=run,
        shard_rank=1,
        chunk_index=3,
    )
    same = write_content_addressed_chunk(
        run.output_root,
        [second, first],
        run=run,
        shard_rank=1,
        chunk_index=3,
    )
    assert same.manifest_sha256 == manifest.manifest_sha256
    assert manifest.evidence_file.name == f"{manifest.evidence_sha256}.jsonl"

    manifest_path = run.output_root / "manifests" / "rank-01-chunk-000003.json"
    resumed = load_resumable_chunk(
        manifest_path,
        output_root=run.output_root,
        run=run,
        expected_rank=1,
        expected_chunk_index=3,
    )
    assert resumed is not None
    assert resumed.record_count == 2
    assert (
        load_resumable_chunk(
            run.output_root / "manifests" / "missing.json",
            output_root=run.output_root,
            run=run,
            expected_rank=1,
            expected_chunk_index=4,
        )
        is None
    )

    evidence_path = run.output_root / manifest.evidence_file
    evidence_path.write_bytes(evidence_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        load_resumable_chunk(
            manifest_path,
            output_root=run.output_root,
            run=run,
            expected_rank=1,
            expected_chunk_index=3,
        )


def test_chunk_rejects_wrong_rank_duplicate_or_logical_overwrite(
    tmp_path: Path,
) -> None:
    run = _load_run(tmp_path)
    raw = T1RawGenerationEvidence.from_record(_raw_record(run, rank=2))
    with pytest.raises(ValueError, match="different shard"):
        write_content_addressed_chunk(
            run.output_root, [raw], run=run, shard_rank=1, chunk_index=0
        )
    with pytest.raises(ValueError, match="duplicate logical evidence"):
        write_content_addressed_chunk(
            run.output_root, [raw, raw], run=run, shard_rank=2, chunk_index=0
        )

    write_content_addressed_chunk(
        run.output_root, [raw], run=run, shard_rank=2, chunk_index=0
    )
    changed_record = raw.as_record()
    changed_record["raw_text"] = "different</think> B"
    changed = T1RawGenerationEvidence.from_record(changed_record)
    with pytest.raises(ValueError, match="existing immutable artifact differs"):
        write_content_addressed_chunk(
            run.output_root, [changed], run=run, shard_rank=2, chunk_index=0
        )


def test_runtime_module_imports_without_torch_or_gpu_runtime() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    script = f"""
import sys
sys.path.insert(0, {str(source_root)!r})
import tgvf_rl.data.policy_selection_runtime
for name in ('torch', 'vllm', 'transformers'):
    assert name not in sys.modules, (name, sorted(sys.modules))
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
