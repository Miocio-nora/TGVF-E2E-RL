from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "run_rp67_step2000_acc_pipeline.py"
)
_SPEC = importlib.util.spec_from_file_location("rp67_acc_pipeline", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
pipeline = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pipeline
_SPEC.loader.exec_module(pipeline)


def _write_semantic_publication(
    root: Path, *, arms: tuple[str, ...], samples: int
) -> None:
    root.mkdir(parents=True)
    run_identity = "a" * 64
    summary = {
        "schema_version": pipeline.SEMANTIC_SCHEMA,
        "status": "complete",
        "run_identity_sha256": run_identity,
        "overall": {"total": samples * len(arms)},
        "by_arm": {arm: {"total": samples} for arm in arms},
    }
    summary_payload = (json.dumps(summary, sort_keys=True) + "\n").encode()
    (root / "summary.json").write_bytes(summary_payload)
    manifest = {
        "schema_version": pipeline.SEMANTIC_SCHEMA,
        "status": "complete",
        "run_identity_sha256": run_identity,
        "files": {
            "summary": {
                "path": "summary.json",
                "sha256": sha256(summary_payload).hexdigest(),
            },
            "overlay_records": {"path": "records.jsonl", "rows": samples * len(arms)},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")


def test_v2_uses_new_semantic_roots_and_marker_without_reusing_v1_paths() -> None:
    assert pipeline.FIRST_MAIN_SEMANTIC.name == (
        "rp67_step2000_first200_acc_main_semantic_v2_20260801"
    )
    assert pipeline.FULL_MAIN_SEMANTIC.name == (
        "rp67_step2000_full867_acc_main_semantic_v2_20260801"
    )
    assert pipeline.COMPLETE_MARKER.name == (
        "rp67_step2000_all_validations_complete_v2.json"
    )
    assert pipeline.PIPELINE_COMPLETE.name == "complete-v2.json"
    assert pipeline.MARKER_SCHEMA == "rp67-all-validations-complete-v2"


def test_v2_marker_binds_each_semantic_summary_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    full = tmp_path / "full"
    six = tmp_path / "six"
    _write_semantic_publication(first, arms=pipeline.MAIN_ARMS, samples=200)
    _write_semantic_publication(full, arms=pipeline.MAIN_ARMS, samples=867)
    _write_semantic_publication(six, arms=pipeline.SIX_ARMS, samples=200)
    int_diag = tmp_path / "int-diag.json"
    int_diag.write_text("{}\n")
    marker_path = tmp_path / "complete-v2.json"
    monkeypatch.setattr(pipeline, "FIRST_MAIN_SEMANTIC", first)
    monkeypatch.setattr(pipeline, "FULL_MAIN_SEMANTIC", full)
    monkeypatch.setattr(pipeline, "FIRST_SIX_SEMANTIC", six)
    monkeypatch.setattr(pipeline, "INT_DIAG", int_diag)
    monkeypatch.setattr(pipeline, "COMPLETE_MARKER", marker_path)
    marker = {
        "schema_version": pipeline.MARKER_SCHEMA,
        "status": "complete",
        "rp67_run_id": pipeline.RUN_ID,
        "artifacts": {
            "int_diag": pipeline._artifact_record(int_diag),
            "acc_first200": pipeline._semantic_artifact_record(
                first, arms=pipeline.MAIN_ARMS, samples=200
            ),
            "acc_full867": pipeline._semantic_artifact_record(
                full, arms=pipeline.MAIN_ARMS, samples=867
            ),
            "diag_first200_sixarm": pipeline._semantic_artifact_record(
                six, arms=pipeline.SIX_ARMS, samples=200
            ),
        },
    }
    marker_path.write_text(json.dumps(marker) + "\n")

    assert pipeline._existing_complete_marker_is_valid()
    (full / "manifest.json").write_text("{}\n")
    with pytest.raises(pipeline.PipelineBlockedError, match="publication mismatch"):
        pipeline._existing_complete_marker_is_valid()


def test_semantic_completion_rejects_legacy_v1_publication(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    _write_semantic_publication(root, arms=pipeline.MAIN_ARMS, samples=200)
    for name in ("summary.json", "manifest.json"):
        value = json.loads((root / name).read_text())
        value["schema_version"] = "answer-utility-semantic-rescore-v1"
        (root / name).write_text(json.dumps(value) + "\n")
    with pytest.raises(pipeline.PipelineBlockedError, match="publication mismatch"):
        pipeline._semantic_complete(root, arms=pipeline.MAIN_ARMS, samples=200)
