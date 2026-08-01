from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools/prove_policy_auto_resume.py"
_SPEC = importlib.util.spec_from_file_location("policy_auto_resume_proof", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
proof = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = proof
_SPEC.loader.exec_module(proof)


def test_metrics_snapshot_binds_exact_bytes_count_and_step(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"optimizer_step": 1, "value": 3}) + "\n", encoding="utf-8"
    )

    snapshot = proof._metrics_snapshot(path, expected_step=1)

    assert snapshot["records"] == 1
    assert snapshot["last_optimizer_step"] == 1
    assert snapshot["bytes"] == len(path.read_bytes())
    assert len(snapshot["sha256"]) == 64


def test_metrics_snapshot_rejects_wrong_or_malformed_terminal_step(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"optimizer_step": 2}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected 1"):
        proof._metrics_snapshot(path, expected_step=1)

    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=":1"):
        proof._metrics_snapshot(path, expected_step=1)


def test_checkpoint_tree_digest_is_content_and_path_bound(tmp_path: Path) -> None:
    checkpoint = tmp_path / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)
    state = checkpoint / "actor/state.bin"
    state.write_bytes(b"one")
    first = proof._tree_sha256(checkpoint)
    assert first == proof._tree_sha256(checkpoint)

    state.write_bytes(b"two")
    assert proof._tree_sha256(checkpoint) != first


def test_atomic_proof_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "proof.json"
    proof._atomic_json(path, {"resume": {"proven": False}})
    proof._atomic_json(path, {"resume": {"proven": True}})

    assert json.loads(path.read_text()) == {"resume": {"proven": True}}
    assert not tuple(tmp_path.glob(".proof.json.tmp.*"))
