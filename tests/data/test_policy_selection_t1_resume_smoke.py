from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.data.test_policy_selection_runtime import _run_record
from tgvf_rl.data.policy_selection import (
    SelectionBranch,
    stable_selection_request_id,
)
from tgvf_rl.data.policy_selection_runtime import (
    T1_ATTEMPTS,
    T1_RAW_GENERATION_SCHEMA,
    T1RawGenerationEvidence,
    native_prompt_identity_sha256,
    rendered_prompt_token_ids_sha256,
    sampled_token_ids_sha256,
    write_content_addressed_chunk,
)
from tgvf_rl.data import policy_selection_t1_resume_smoke as resume_smoke
from tgvf_rl.data import policy_selection_vllm
from tgvf_rl.data.policy_selection_t1_resume_smoke import (
    archive_t1_continuous_baseline,
    build_t1_resume_smoke_plan,
    compare_t1_resume_with_continuous,
    t1_resume_smoke_core_digest,
    validate_t1_resume_smoke_prefix,
)
from tgvf_rl.data.policy_selection_vllm import (
    load_t1_candidates,
    prepare_output_root,
    rank_candidate_chunks,
    run_t1_worker,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_CONFIG = (
    _REPO_ROOT / "configs/policy/data_selection/"
    "qwen3_instruct_t1_512_filter_resume_smoke_gpu3_v1.json"
)


def _fixture_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    record = _run_record(tmp_path)
    record["run_id"] = "t1-instruct-resume-smoke-fixture"
    record["model"]["repository"] = "Qwen/Qwen3-VL-8B-Instruct"
    record["model"]["path"] = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct"
    record["verifier"]["answer_parser"] = "direct-completion-v1"
    artifact_root = tmp_path / "artifacts/data/policy_selection/t1"
    record["output_root"] = str(artifact_root / "active")
    monkeypatch.setattr(resume_smoke, "_ARTIFACT_ROOT", artifact_root)
    config = tmp_path / "resume-smoke.json"
    config.write_text(json.dumps(record), encoding="utf-8")
    return config


def _chunk_records(plan, chunk_index: int) -> list[T1RawGenerationEvidence]:
    candidates = load_t1_candidates(plan.run)
    chunks = rank_candidate_chunks(
        candidates,
        rank=plan.rank,
        world_size=int(plan.run.runtime["world_size"]),
        chunk_candidates=int(plan.run.runtime["chunk_candidates"]),
    )
    budget = plan.run.budget(0)
    records = []
    for candidate in chunks[chunk_index]:
        for attempt_index in range(T1_ATTEMPTS):
            token_ids = [101, 102]
            record = {
                "schema_version": T1_RAW_GENERATION_SCHEMA,
                "run_id": plan.run.run_id,
                "run_manifest_sha256": plan.run.manifest_sha256,
                "request_id": stable_selection_request_id(
                    candidate_sha256=candidate.identity_sha256,
                    branch=SelectionBranch.FULL_IMAGE,
                    attempt_index=attempt_index,
                ),
                "sample_id": candidate.sample_id,
                "candidate_sha256": candidate.identity_sha256,
                "source": candidate.source.value,
                "branch": SelectionBranch.FULL_IMAGE.value,
                "attempt_index": attempt_index,
                "attempt_seed": plan.run.attempt_seed(
                    candidate_sha256=candidate.identity_sha256,
                    attempt_index=attempt_index,
                ),
                "budget_revision": 0,
                "max_model_len": budget.max_model_len,
                "max_new_tokens": budget.max_new_tokens,
                "prompt_sha256": native_prompt_identity_sha256(
                    question=candidate.question,
                    image_sha256=str(candidate.image["sha256"]),
                    chat_template_sha256=str(plan.run.model["chat_template_sha256"]),
                ),
                "rendered_prompt_token_ids_sha256": (
                    rendered_prompt_token_ids_sha256([11, 12])
                ),
                "prompt_token_count": 2,
                "image_sha256": candidate.image["sha256"],
                "image_evidence": {
                    "source_width": 32,
                    "source_height": 32,
                    "source_mode": "RGB",
                    "source_rgb_sha256": "b" * 64,
                    "processed_width": 32,
                    "processed_height": 32,
                },
                "sampled_token_ids_sha256": sampled_token_ids_sha256(token_ids),
                "sampled_token_count": len(token_ids),
                "sampled_token_ids": token_ids,
                "raw_text": "A",
                "finish_reason": "stop",
                "stop_reason": plan.run.model["eos_token_id"],
                "backend": {
                    "name": plan.run.runtime["backend"],
                    "version": plan.run.runtime["version"],
                    "runtime_sha256": plan.run.runtime_identity_sha256,
                    "model_sha256": plan.run.model_identity_sha256,
                    "processor_sha256": plan.run.processor_identity_sha256,
                },
            }
            records.append(T1RawGenerationEvidence.from_record(record))
    return records


def _write_chunk(plan, chunk_index: int) -> None:
    write_content_addressed_chunk(
        plan.output_root,
        _chunk_records(plan, chunk_index),
        run=plan.run,
        shard_rank=plan.rank,
        chunk_index=chunk_index,
    )


def test_real_gpu3_plan_is_exact_instruct_tool_free_request_set() -> None:
    plan = build_t1_resume_smoke_plan(_REAL_CONFIG)
    candidates = {
        candidate.identity_sha256: candidate
        for candidate in load_t1_candidates(plan.run)
    }

    assert plan.run.model["repository"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert dict(plan.run.prompt) == {
        "schema": "qwen-native-user-image-question-v1",
        "user_content_order": ("image", "question"),
        "no_system": True,
        "no_tools": True,
        "add_generation_prompt": True,
    }
    assert plan.rank == 3
    assert plan.max_chunks == 2
    assert len(plan.candidate_sha256s) == 8
    assert plan.expected_records == 64
    assert {candidates[value].source.value for value in plan.candidate_sha256s} == {
        "vstar"
    }


def test_cpu_stop_resume_outputs_match_continuous_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    plan = build_t1_resume_smoke_plan(config)

    prepare_output_root(config)
    _write_chunk(plan, 0)
    _write_chunk(plan, 1)
    baseline = archive_t1_continuous_baseline(plan)
    assert baseline["committed_chunks"] == 2
    assert baseline["records"] == 64

    prepare_output_root(config)
    _write_chunk(plan, 0)
    assert validate_t1_resume_smoke_prefix(plan, committed_chunks=1)["records"] == 32
    _write_chunk(plan, 1)
    digest_before = t1_resume_smoke_core_digest(plan.output_root, plan)
    _write_chunk(plan, 0)
    _write_chunk(plan, 1)
    assert t1_resume_smoke_core_digest(plan.output_root, plan) == digest_before
    comparison = compare_t1_resume_with_continuous(plan)

    assert comparison["result"] == "PASS"
    assert comparison["continuous_core_sha256"] == comparison["resumed_core_sha256"]
    assert all(row["byte_identical"] for row in comparison["compared"])

    monkeypatch.setattr(
        policy_selection_vllm, "_validate_runtime_versions", lambda _: None
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    result = asyncio.run(run_t1_worker(config, rank=3, max_chunks=2))
    assert result == {
        "run_id": plan.run.run_id,
        "rank": 3,
        "budget_revision": 0,
        "chunks_written": 0,
        "records_written": 0,
        "records_resumed": 64,
    }


def test_plan_cli_is_cpu_only_and_reports_no_tools() -> None:
    script = f"""
import json
import subprocess
import sys
result = subprocess.run(
    [sys.executable, {str(_REPO_ROOT / "tools/smoke_policy_data_selection_t1_resume.py")!r},
     'plan', '--config', {str(_REAL_CONFIG)!r}],
    check=True, capture_output=True, text=True,
)
record = json.loads(result.stdout)
assert record['expected_records'] == 64
assert record['prompt']['no_tools'] is True
assert record['prompt']['no_system'] is True
for name in ('torch', 'vllm', 'transformers'):
    assert name not in sys.modules, name
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
