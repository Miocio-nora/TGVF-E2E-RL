from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tgvf_rl.data.policy_selection import SelectionCandidate
from tgvf_rl.data.policy_selection_runtime import candidate_rank
from tgvf_rl.data.policy_selection_vllm import (
    _EXPECTED_PROMPT_SUFFIX_BY_REPOSITORY,
    T1_BUDGET_CHUNK_STRIDE,
    budget_chunk_index,
    qwen_smart_resize_dimensions,
    rank_candidate_chunks,
)
from tgvf_rl.data.policy_selection_vllm_retry import (
    length_retry_prefix_audit,
    select_length_retry_evidence,
)


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (427, 640, (416, 608)),
        (226, 220, (288, 256)),
        (2016, 2016, (512, 512)),
        (29, 104, (160, 512)),
    ],
)
def test_smart_resize_matches_qwen_geometry(
    height: int, width: int, expected: tuple[int, int]
) -> None:
    actual = qwen_smart_resize_dimensions(
        height=height,
        width=width,
        factor=32,
        min_pixels=65_536,
        max_pixels=262_144,
    )
    assert actual == expected
    resized_height, resized_width = actual
    assert resized_height % 32 == resized_width % 32 == 0
    assert 65_536 <= resized_height * resized_width <= 262_144


def test_smart_resize_rejects_extreme_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="aspect ratio"):
        qwen_smart_resize_dimensions(
            height=1,
            width=201,
            factor=32,
            min_pixels=65_536,
            max_pixels=262_144,
        )


def test_native_prompt_suffix_is_exact_for_each_accepted_qwen3_vl_edition() -> None:
    assert _EXPECTED_PROMPT_SUFFIX_BY_REPOSITORY == {
        "Qwen/Qwen3-VL-8B-Thinking": "<|im_start|>assistant\n<think>\n",
        "Qwen/Qwen3-VL-8B-Instruct": "<|im_start|>assistant\n",
    }


def _candidate(index: int) -> SelectionCandidate:
    return SelectionCandidate.from_record(
        {
            "schema_version": "tgvf.policy-selection.candidate.v1",
            "sample_id": f"fixture:{index}",
            "source": "thinklite",
            "question": f"Question {index}?",
            "ground_truth": str(index),
            "image": {
                "sha256": f"{index + 1:064x}",
                "width": 32,
                "height": 32,
            },
            "provenance": {"index": index},
        }
    )


def test_rank_chunks_cover_every_candidate_once_and_keep_candidate_attempts_local() -> (
    None
):
    candidates = tuple(_candidate(index) for index in range(29))
    chunks_by_rank = [
        rank_candidate_chunks(candidates, rank=rank, world_size=4, chunk_candidates=4)
        for rank in range(4)
    ]
    flattened = [
        candidate
        for rank_chunks in chunks_by_rank
        for chunk in rank_chunks
        for candidate in chunk
    ]
    assert {item.identity_sha256 for item in flattened} == {
        item.identity_sha256 for item in candidates
    }
    assert len(flattened) == len(candidates)
    for rank, chunks in enumerate(chunks_by_rank):
        assert all(1 <= len(chunk) <= 4 for chunk in chunks)
        assert all(
            candidate_rank(candidate.identity_sha256, world_size=4) == rank
            for chunk in chunks
            for candidate in chunk
        )


def test_budget_chunk_namespaces_do_not_collide() -> None:
    assert budget_chunk_index(budget_revision=0, local_chunk_index=7) == 7
    assert (
        budget_chunk_index(budget_revision=1, local_chunk_index=7)
        == T1_BUDGET_CHUNK_STRIDE + 7
    )
    assert (
        budget_chunk_index(budget_revision=2, local_chunk_index=7)
        == 2 * T1_BUDGET_CHUNK_STRIDE + 7
    )


def test_runner_module_and_cli_help_do_not_import_gpu_libraries() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    repository_root = Path(__file__).resolve().parents[2]
    script = f"""
import sys
sys.path.insert(0, {str(source_root)!r})
import tgvf_rl.data.policy_selection_vllm
import tgvf_rl.data.policy_selection_vllm_retry
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
            str(repository_root / "tools" / "run_policy_data_selection_t1.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prepare" in completed.stdout
    assert "worker" in completed.stdout

    rejected = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools" / "run_policy_data_selection_t1.py"),
            "worker",
            "--config",
            "does-not-matter.json",
            "--rank",
            "0",
            "--budget-revision",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "invalid choice" in rejected.stderr

    retry_help = subprocess.run(
        [
            sys.executable,
            str(
                repository_root
                / "tools"
                / "run_policy_data_selection_t1_retry.py"
            ),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "plan" in retry_help.stdout
    assert "worker" in retry_help.stdout


def _retry_evidence(
    *, request_id: str, candidate: str, attempt: int, revision: int, finish: str
) -> SimpleNamespace:
    token_count = 4 + revision
    return SimpleNamespace(
        run_id="run",
        run_manifest_sha256="f" * 64,
        request_id=request_id,
        sample_id=f"sample:{candidate}",
        candidate_sha256=candidate,
        source="arxivqa",
        attempt_index=attempt,
        attempt_seed=17,
        budget_revision=revision,
        max_model_len=64 * (revision + 1),
        max_new_tokens=32 * (revision + 1),
        prompt_sha256="a" * 64,
        rendered_prompt_token_ids_sha256="b" * 64,
        prompt_token_count=7,
        image_sha256="c" * 64,
        source_width=16,
        source_height=16,
        source_mode="RGB",
        source_rgb_sha256="d" * 64,
        processed_width=16,
        processed_height=16,
        sampled_token_ids=tuple(range(token_count)),
        sampled_token_count=token_count,
        finish_reason=finish,
        backend={"runtime": "fixture"},
        evidence_sha256=f"{revision + 5:064x}",
    )


def test_length_retry_selects_only_length_and_preserves_exact_prefix() -> None:
    length = _retry_evidence(
        request_id="length", candidate="1" * 64, attempt=5, revision=0, finish="length"
    )
    stopped = _retry_evidence(
        request_id="stop", candidate="2" * 64, attempt=2, revision=0, finish="stop"
    )
    expected = {
        "length": ("1" * 64, 5),
        "stop": ("2" * 64, 2),
    }
    selected = select_length_retry_evidence(
        [length, stopped], expected_requests=expected, budget_revision=1
    )
    assert selected.pending_previous == (length,)
    assert selected.completed_current == ()

    retried = _retry_evidence(
        request_id="length", candidate="1" * 64, attempt=5, revision=1, finish="stop"
    )
    audit = length_retry_prefix_audit(length, retried)
    assert audit["previous_is_exact_prefix"] is True
    assert audit["common_prefix_token_count"] == 4
    resumed = select_length_retry_evidence(
        [length, stopped, retried],
        expected_requests=expected,
        budget_revision=1,
    )
    assert resumed.pending_previous == ()
    assert resumed.completed_current == (retried,)


def test_length_retry_rejects_replay_after_stop_and_audits_prefix_drift() -> None:
    stopped = _retry_evidence(
        request_id="request", candidate="3" * 64, attempt=0, revision=0, finish="stop"
    )
    invalid_retry = _retry_evidence(
        request_id="request", candidate="3" * 64, attempt=0, revision=1, finish="stop"
    )
    with pytest.raises(ValueError, match="only a length finish"):
        select_length_retry_evidence(
            [stopped, invalid_retry],
            expected_requests={"request": ("3" * 64, 0)},
            budget_revision=1,
        )

    length = _retry_evidence(
        request_id="request", candidate="3" * 64, attempt=0, revision=0, finish="length"
    )
    invalid_retry.sampled_token_ids = (9, 1, 2, 3, 4)
    audit = length_retry_prefix_audit(length, invalid_retry)
    assert audit["previous_is_exact_prefix"] is False
    assert audit["common_prefix_token_count"] == 0
    assert audit["first_divergence_index"] == 0


def test_length_retry_accepts_shorter_normal_completion_as_audited_replay() -> None:
    length = _retry_evidence(
        request_id="request", candidate="4" * 64, attempt=1, revision=0, finish="length"
    )
    completed = _retry_evidence(
        request_id="request", candidate="4" * 64, attempt=1, revision=1, finish="stop"
    )
    completed.sampled_token_ids = (0, 1)
    completed.sampled_token_count = 2
    audit = length_retry_prefix_audit(length, completed)
    assert audit["previous_is_exact_prefix"] is False
    assert audit["common_prefix_token_count"] == 2
    assert audit["first_divergence_index"] == 2
