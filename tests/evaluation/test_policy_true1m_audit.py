from __future__ import annotations

import hashlib
import fcntl
import json
from pathlib import Path

import pytest

from tgvf_rl.evaluation import policy_true1m_audit as audit


def _processor_proof() -> dict[str, object]:
    return {
        "configured_image_max_pixels": audit.TRUE1M_IMAGE_MAX_PIXELS,
        "effective_processor_image_size": {
            "shortest_edge": 65_536,
            "longest_edge": audit.TRUE1M_IMAGE_MAX_PIXELS,
        },
        "processor_patch_size": 16,
        "processor_merge_size": 2,
        "runtime_mm_processor_kwargs": {
            "size": {
                "shortest_edge": 65_536,
                "longest_edge": audit.TRUE1M_IMAGE_MAX_PIXELS,
            }
        },
        "runtime_override_path": "mm_processor_kwargs.size.longest_edge",
        "vllm_012_shallow_hashable": True,
        "nested_images_kwargs_present": False,
        "max_pixels_kwarg_present": False,
    }


def _result(ordinal: int, *, first_count: int = 980) -> dict[str, object]:
    return {
        "result_identity_sha256": hashlib.sha256(str(ordinal).encode()).hexdigest(),
        "successful_observation_count": 1,
        "native_original_image_count": 1,
        "native_crop_image_count": 1,
        "native_total_image_count": 2,
        "native_image_sha256s": ["a" * 64, "b" * 64],
        "tool_calls": [{"assistant_turn_index": 0}],
        "assistant_turns": [
            {"turn_index": 0, "native_visual_token_counts": [first_count]},
            {"turn_index": 1, "native_visual_token_counts": [100, 200]},
        ],
    }


def test_true1m_row_audit_binds_all_visual_counts_and_result_identities() -> None:
    records = {
        ordinal: _result(ordinal)
        for ordinal in range(audit.TRUE1M_COREDEV_SINGLE_IMAGE_ROWS)
    }

    result = audit.audit_official_visible_true1m_records(
        records, processor_proof=_processor_proof()
    )

    assert result["accepted_row_count"] == 2240
    assert result["assistant_turn_count"] == 4480
    assert result["encoded_image_instance_count"] == 6720
    assert result["visual_token_pixel_quantum"] == 1024
    assert result["maximum_allowed_visual_token_count"] == 980
    assert result["maximum_observed_visual_token_count"] == 980
    assert result["maximum_observed_represented_pixel_area"] == 1_003_520
    assert result["all_native_images_within_true1m"] is True
    assert len(result["result_identity_sequence_sha256"]) == 64
    assert len(result["native_visual_count_evidence_sha256"]) == 64


def test_true1m_row_audit_rejects_one_grid_above_cap() -> None:
    records = {
        ordinal: _result(ordinal)
        for ordinal in range(audit.TRUE1M_COREDEV_SINGLE_IMAGE_ROWS)
    }
    records[17] = _result(17, first_count=981)

    with pytest.raises(RuntimeError, match="exceeds true1M"):
        audit.audit_official_visible_true1m_records(
            records, processor_proof=_processor_proof()
        )


def test_incomplete_preflight_fails_before_rank_lock_phase(tmp_path: Path) -> None:
    inference = tmp_path / "inference"
    inference.mkdir()
    for rank in range(4):
        (inference / f"rank-{rank}.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="before rank locks"):
        audit._preflight_complete_rank_files(inference, world_size=4)

    assert not (tmp_path / "runtime").exists()


def test_completed_rank_locks_use_existing_noncreating_writable_files(
    tmp_path: Path,
) -> None:
    lock_root = tmp_path / "runtime/locks"
    lock_root.mkdir(parents=True)
    paths = []
    for rank in range(4):
        path = lock_root / f"rank-{rank}.lock"
        path.write_bytes(b"")
        paths.append(path)

    with audit._completed_rank_locks(tmp_path, world_size=4):
        assert all(path.read_bytes() == b"" for path in paths)

    assert all(path.read_bytes() == b"" for path in paths)


def test_completed_rank_locks_fail_closed_when_one_worker_is_active(
    tmp_path: Path,
) -> None:
    lock_root = tmp_path / "runtime/locks"
    lock_root.mkdir(parents=True)
    for rank in range(4):
        (lock_root / f"rank-{rank}.lock").write_bytes(b"")
    active_path = lock_root / "rank-2.lock"
    with active_path.open("r+b") as active:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(RuntimeError, match="rank 2 is still active"):
                with audit._completed_rank_locks(tmp_path, world_size=4):
                    raise AssertionError("unreachable")
        finally:
            fcntl.flock(active.fileno(), fcntl.LOCK_UN)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_receipt_loader_revalidates_bound_inference_bytes(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    config = tmp_path / "benchmark-config.json"
    identity = tmp_path / "evaluation-identity.json"
    tasks = tmp_path / "tasks.jsonl"
    snapshot = tmp_path / "snapshot.json"
    for path in (config, identity, tasks, snapshot):
        path.write_text("{}\n", encoding="utf-8")
    plan_payload = {
        "schema_version": "tgvf.resolution-paired-policy-benchmark-plan.v4",
        "evaluation_id": "PAIR",
    }
    _write_json(plan, plan_payload)
    proof_content = {
        "schema_version": audit.TRUE1M_PROCESSOR_PROOF_SCHEMA,
        "attestation_scope": "posthoc_static_processor_contract",
        "evaluation_id": "ARM",
        "proof": _processor_proof(),
    }
    proof = {
        **proof_content,
        "proof_identity_sha256": audit.canonical_sha256(proof_content),
    }
    proof_path = tmp_path / "processor-proof.json"
    _write_json(proof_path, proof)
    rank_records = []
    for rank in range(4):
        rank_path = tmp_path / f"rank-{rank}.jsonl"
        rank_path.write_text("{}\n" * 560, encoding="utf-8")
        record = audit._file_record(rank_path)
        record.update({"rank": rank, "line_count": 560})
        rank_records.append(record)
    rows = {
        "accepted_row_count": 2240,
        "all_native_images_within_true1m": True,
        "turn_image_sequence_verified": True,
        "result_identity_sequence_sha256": "a" * 64,
        "native_visual_count_evidence_sha256": "b" * 64,
    }
    content = {
        "schema_version": audit.TRUE1M_AUDIT_RECEIPT_SCHEMA,
        "status": "accepted",
        "attestation_scope": (
            "posthoc_effective_visual_grid_and_static_processor_contract"
        ),
        "generation_identity_extended": False,
        "plan": audit._plan_receipt_record(plan, plan_payload),
        "arm": {
            "name": "pixel1003520",
            "evaluation_id": "ARM",
            "evaluation_image_max_pixels": audit.TRUE1M_IMAGE_MAX_PIXELS,
            "output_root": str(tmp_path),
            "benchmark_config": audit._file_record(config),
        },
        "evaluation_identity": audit._file_record(identity),
        "task_manifest": audit._file_record(tasks),
        "snapshot_files": [audit._file_record(snapshot)],
        "processor_proof": {
            **audit._file_record(proof_path),
            "proof_identity_sha256": proof["proof_identity_sha256"],
        },
        "inference": {
            "world_size": 4,
            "files": rank_records,
            "tree_identity_sha256": audit.canonical_sha256(rank_records),
        },
        "rows": rows,
    }
    receipt = {
        **content,
        "receipt_identity_sha256": audit.canonical_sha256(content),
    }
    assert receipt["plan"]["sha256"] == audit.file_sha256(plan)
    assert receipt["plan"]["file_sha256"] == receipt["plan"]["sha256"]
    receipt_path = tmp_path / audit.TRUE1M_AUDIT_RECEIPT_FILENAME
    _write_json(receipt_path, receipt)

    assert (
        audit.load_official_visible_true1m_audit_receipt(receipt_path)[
            "receipt_identity_sha256"
        ]
        == receipt["receipt_identity_sha256"]
    )

    with (tmp_path / "rank-3.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="rank 3 bytes differ"):
        audit.load_official_visible_true1m_audit_receipt(receipt_path)


def test_v5_reference_resolves_the_real_v4_s80_arm() -> None:
    plan_path = (
        Path(__file__).parents[2] / "configs/evaluation/"
        "prl25_b_crop_exact_step32_true1m_resolution_rng_extension_"
        "v5_coredev2511_plan.json"
    )
    plan = audit._load_plan(plan_path)

    resolved = audit._referenced_receipt_path(plan, explicit_path=None)

    assert resolved is not None
    receipt_path, reference_plan, reference_arm = resolved
    assert reference_plan["schema_version"].endswith("plan.v4")
    assert reference_arm["name"] == "pixel1003520"
    assert reference_arm["evaluation_image_max_pixels"] == 1_003_520
    assert receipt_path.as_posix().endswith(
        "PRL25-B-CROP-EXACT-COREDEV2511-STEP80-TRUE1M-TRUE512-"
        "RESOLUTION-PAIR-V1/pixel1003520/runtime/true1m-audit-receipt.json"
    )
