from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tgvf_rl.data.policy_selection import (
    POLICY_SELECTION_PRIMARY_SOURCES,
    SelectionCandidate,
    SelectionSource,
)
from tgvf_rl.data.policy_selection_runtime import native_prompt_identity_sha256
from tgvf_rl.data.policy_selection_t1_replay_audit import (
    LocatedT1Evidence,
    _build_histories,
    _validate_replay_environment,
    _write_audit_report,
    compare_replayed_evidence,
    select_replay_audit_candidates,
)


def _candidate(source: str, index: int) -> SelectionCandidate:
    return SelectionCandidate.from_record(
        {
            "schema_version": "tgvf.policy-selection.candidate.v1",
            "sample_id": f"fixture:{source}:{index}",
            "source": source,
            "question": f"Question {source} {index}?",
            "ground_truth": "answer",
            "image": {
                "sha256": f"{index + 1:064x}",
                "width": 32,
                "height": 32,
            },
            "gt_regions": [],
            "provenance": {"fixture": True, "index": index},
        }
    )


def test_content_hash_selection_is_order_independent_and_covers_sources() -> None:
    candidates = tuple(
        _candidate(source, source_index * 10 + index)
        for source_index, source in enumerate(("vstar", "arxivqa", "thinklite"))
        for index in range(3)
    )
    first = select_replay_audit_candidates(
        candidates,
        run_manifest_sha256="a" * 64,
        rank=0,
        world_size=1,
    )
    second = select_replay_audit_candidates(
        tuple(reversed(candidates)),
        run_manifest_sha256="a" * 64,
        rank=0,
        world_size=1,
    )
    assert [(item.identity_sha256, score) for item, score in first] == [
        (item.identity_sha256, score) for item, score in second
    ]
    assert {candidate.source for candidate, _ in first} == set(
        POLICY_SELECTION_PRIMARY_SOURCES
    )
    assert len(first) == 3

    with pytest.raises(ValueError, match="no replay-audit candidate"):
        select_replay_audit_candidates(
            tuple(
                item for item in candidates if item.source is not SelectionSource.VSTAR
            ),
            run_manifest_sha256="a" * 64,
            rank=0,
            world_size=1,
        )


def _run() -> SimpleNamespace:
    return SimpleNamespace(
        run_id="run",
        manifest_sha256="a" * 64,
        runtime={
            "world_size": 1,
            "chunk_candidates": 4,
            "retain_token_ids": True,
            "engine_seed": 42,
        },
        response_budgets=(SimpleNamespace(revision=0),),
        model={"chat_template_sha256": "b" * 64},
        sampling={"attempts": 8, "temperature": 1.0},
        model_identity_sha256="c" * 64,
        processor_identity_sha256="d" * 64,
        runtime_identity_sha256="e" * 64,
    )


def _evidence(
    candidate: SelectionCandidate,
    *,
    attempt_index: int,
    token_ids: tuple[int, ...] = (1, 2, 3),
) -> SimpleNamespace:
    prompt_sha = native_prompt_identity_sha256(
        question=candidate.question,
        image_sha256=str(candidate.image["sha256"]),
        chat_template_sha256="b" * 64,
    )
    token_sha = hashlib.sha256(
        json.dumps(list(token_ids), separators=(",", ":")).encode()
    ).hexdigest()
    return SimpleNamespace(
        run_id="run",
        run_manifest_sha256="a" * 64,
        request_id=f"request-{candidate.identity_sha256}-{attempt_index}",
        sample_id=candidate.sample_id,
        candidate_sha256=candidate.identity_sha256,
        source=candidate.source,
        attempt_index=attempt_index,
        attempt_seed=100 + attempt_index,
        budget_revision=0,
        max_model_len=65_536,
        max_new_tokens=40_960,
        prompt_sha256=prompt_sha,
        rendered_prompt_token_ids_sha256="f" * 64,
        prompt_token_count=25,
        image_sha256=candidate.image["sha256"],
        source_width=32,
        source_height=32,
        source_mode="RGB",
        source_rgb_sha256="1" * 64,
        processed_width=256,
        processed_height=256,
        sampled_token_ids_sha256=token_sha,
        sampled_token_count=len(token_ids),
        sampled_token_ids=token_ids,
        raw_text="reasoning</think> answer",
        finish_reason="stop",
        stop_reason=151_645,
        backend={
            "name": "vllm",
            "version": "0.12.0",
            "runtime_sha256": "e" * 64,
            "model_sha256": "c" * 64,
            "processor_sha256": "d" * 64,
        },
        evidence_sha256=f"{attempt_index + 2:064x}",
    )


def test_complete_histories_require_all_eight_attempts() -> None:
    run = _run()
    candidates = tuple(
        _candidate(source, index)
        for index, source in enumerate(("vstar", "arxivqa", "thinklite"))
    )
    manifest = SimpleNamespace(
        shard_rank=0,
        chunk_index=0,
        manifest_sha256="2" * 64,
        evidence_sha256="3" * 64,
    )
    located = [
        (manifest, _evidence(candidate, attempt_index=attempt_index))
        for candidate in candidates
        for attempt_index in range(8)
    ]
    histories, locations = _build_histories(
        run=run,
        candidates=candidates,
        located_records=located,
    )
    assert len(histories) == 24
    assert len(locations) == 3

    with pytest.raises(ValueError, match="revision-0 evidence is incomplete"):
        _build_histories(
            run=run,
            candidates=candidates,
            located_records=located[:-1],
        )
    with pytest.raises(ValueError, match="duplicate logical request"):
        _build_histories(
            run=run,
            candidates=candidates,
            located_records=[*located, located[0]],
        )


def test_comparison_checks_exact_identity_tokens_and_finish_reason() -> None:
    run = _run()
    candidate = _candidate("vstar", 0)
    evidence = _evidence(candidate, attempt_index=0)
    located = LocatedT1Evidence(
        evidence=evidence,
        chunk_manifest_sha256="2" * 64,
        chunk_evidence_sha256="3" * 64,
        shard_rank=0,
        chunk_index=0,
    )
    exact = compare_replayed_evidence(run=run, expected=located, actual=evidence)
    assert exact["passed"] is True
    assert all(exact["checks"].values())

    changed = _evidence(candidate, attempt_index=0, token_ids=(1, 9, 3))
    changed.prompt_sha256 = "9" * 64
    changed.finish_reason = "length"
    changed.stop_reason = None
    mismatch = compare_replayed_evidence(run=run, expected=located, actual=changed)
    assert mismatch["passed"] is False
    assert mismatch["checks"]["sampled_token_ids_exact"] is False
    assert mismatch["checks"]["finish_reason_exact"] is False
    assert mismatch["mismatched_identity_fields"] == ["prompt_sha256"]


def test_audit_report_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    run = _run()
    run.output_root = tmp_path
    selections = tuple(
        SimpleNamespace(
            candidate=_candidate(source, index),
            selector_sha256=f"{index + 4:064x}",
            local_chunk_index=index,
        )
        for index, source in enumerate(("vstar", "arxivqa", "thinklite"))
    )
    plan = SimpleNamespace(
        run=run,
        rank=0,
        selections=selections,
        raw_evidence_count=24,
        logical_attempt_count=24,
        audited_attempt_count=24,
    )
    comparisons = [{"passed": True, "attempt_index": index} for index in range(24)]
    first = _write_audit_report(plan, comparisons)
    second = _write_audit_report(plan, comparisons)
    assert first == second
    path = Path(first["report_path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first["report_sha256"]

    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable artifact differs"):
        _write_audit_report(plan, comparisons)


def test_replay_environment_is_bound_to_original_physical_rank(monkeypatch) -> None:
    plan = SimpleNamespace(rank=2, run=SimpleNamespace(runtime={"engine_seed": 42}))
    required = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "2",
        "VLLM_USE_V1": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "42",
    }
    for name, value in required.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    _validate_replay_environment(plan)

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    with pytest.raises(ValueError, match="must be unset"):
        _validate_replay_environment(plan)


def test_replay_audit_import_and_cli_help_are_cpu_safe() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    repository_root = Path(__file__).resolve().parents[2]
    script = f"""
import sys
sys.path.insert(0, {str(source_root)!r})
import tgvf_rl.data.policy_selection_t1_replay_audit
for name in ('torch', 'vllm', 'transformers', 'PIL'):
    assert name not in sys.modules, (name, sorted(sys.modules))
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools" / "audit_policy_data_selection_t1_replay.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "plan" in completed.stdout
    assert "run" in completed.stdout
