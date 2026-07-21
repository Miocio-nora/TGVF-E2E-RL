from __future__ import annotations

import json
from pathlib import Path

import pytest

from tgvf_rl.representation.training.qwen3_grounding import (
    TARGET_PRESENCE_QUESTION,
    load_qwen3_grounding_manifest,
)


_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST = (
    _ROOT / "configs/representation/internal_evaluation/"
    "qwen3_v4_clean_imend_audited_grounding_v1.json"
)


def test_fixed_grounding_manifest_binds_audited_pair_families() -> None:
    manifest = load_qwen3_grounding_manifest(_MANIFEST)

    assert manifest.identity == "qwen3-v4-clean-imend-audited-grounding-v1"
    assert len(manifest.cross_image_probes) == 8
    assert len(manifest.target_presence_probes) == 36
    assert all(
        probe.question == TARGET_PRESENCE_QUESTION
        and probe.present_value == "PRESENT"
        and probe.not_present_value == "NOT_PRESENT"
        and probe.positive_target != probe.negative_target
        and probe.positive_audit_rationale
        and probe.negative_audit_rationale
        for probe in manifest.target_presence_probes
    )
    assert all(
        probe.source_image_a_sha256 != probe.source_image_b_sha256
        and probe.expected_value_a != probe.expected_value_b
        and probe.audit_rationale_a
        and probe.audit_rationale_b
        for probe in manifest.cross_image_probes
    )


def test_grounding_manifest_rejects_unbound_or_unreviewed_fields(
    tmp_path: Path,
) -> None:
    payload = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    del payload["target_presence_probes"][0]["negative_audit_rationale"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields differ"):
        load_qwen3_grounding_manifest(path)
