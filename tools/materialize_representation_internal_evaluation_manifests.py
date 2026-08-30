#!/usr/bin/python3 -I
"""Materialize the Golden-matched Qwen3 representation evaluation population."""

from __future__ import annotations
# ruff: noqa: E402

# Direct script execution is stopped before legacy path/environment mutation or
# heavyweight runtime imports. Importing the module for read-only compatibility
# tests remains possible; its public ``main`` retains a second fail-closed guard.
if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range(2):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            "tools/materialize_representation_internal_evaluation_manifests.py",
        ),
    )

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from tgvf_rl.representation.training.data import load_retained_representation_jsonl
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)


VALIDATION_PATH = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/data/tgvf_teacher/generated/"
    "runs/tgvf_v4_teacher_50k_clean_imend/splits/"
    "tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl"
)
VALIDATION_SOURCE_SHA256 = (
    "de61c731eb961825a77df587cd76c00eabfea75b5c6003096f3cc7f1a51dd82d"
)
RETAINED_MANIFEST_SHA256 = (
    "534f5b1e648d0bca2b1ea2ff02f81e1fb7abbb456f16faacbb118ca94f7306b0"
)
GOLDEN_ROW_COUNT = 200
GOLDEN_GROUP_COUNT = 46
GOLDEN_K_COUNTS = {3: 6, 4: 19, 5: 20, 6: 1}
FULL_ROW_COUNT = 867
FULL_GROUP_COUNT = 203
FULL_K_COUNTS = {3: 26, 4: 98, 5: 77, 6: 2}
FIRST_SAMPLE_ID = "tgvf_v4_teacher_50k:visual_genome:2410492:0::focus1"
LAST_SAMPLE_ID = "tgvf_v4_teacher_50k:textocr:e08ccd92443c5924:3::focus1"
FULL_LAST_SAMPLE_ID = "tgvf_v4_teacher_50k:textocr:b10c2a9cf285fde5:3::focus1"
PAIR_A = "tgvf_v4_teacher_50k:chartqa:train_017485:0::focus1"
PAIR_B = "tgvf_v4_teacher_50k:chartqa:train_015867:0::focus1"
OUTPUT_DIRECTORY = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/representation/"
    "internal_evaluation"
)


def main() -> None:
    assert_legacy_standalone_execution_quarantined(
        "tools/materialize_representation_internal_evaluation_manifests.py"
    )
    dataset = load_retained_representation_jsonl(
        VALIDATION_PATH,
        expected_source_sha256=VALIDATION_SOURCE_SHA256,
        warn_on_leakage=False,
    )
    if dataset.manifest.manifest_sha256 != RETAINED_MANIFEST_SHA256:
        raise RuntimeError("retained v4 clean-imend test manifest changed")

    selected = dataset.samples[:GOLDEN_ROW_COUNT]
    if (
        len(selected) != GOLDEN_ROW_COUNT
        or selected[0].sample_id != FIRST_SAMPLE_ID
        or selected[-1].sample_id != LAST_SAMPLE_ID
    ):
        raise RuntimeError("Golden first-200 row population changed")
    groups: list[tuple[str, list[object]]] = []
    for sample in selected:
        if not groups or groups[-1][0] != sample.image_group_key:
            groups.append((sample.image_group_key, []))
        groups[-1][1].append(sample)
    observed_k_counts = Counter(len(samples) for _, samples in groups)
    if len(groups) != GOLDEN_GROUP_COUNT or dict(observed_k_counts) != GOLDEN_K_COUNTS:
        raise RuntimeError(
            "Golden group population changed: "
            f"groups={len(groups)}, K-counts={dict(observed_k_counts)}"
        )

    full_selected = dataset.samples
    if (
        len(full_selected) != FULL_ROW_COUNT
        or full_selected[0].sample_id != FIRST_SAMPLE_ID
        or full_selected[-1].sample_id != FULL_LAST_SAMPLE_ID
    ):
        raise RuntimeError("full-867 row population changed")
    full_groups: list[tuple[str, list[object]]] = []
    for sample in full_selected:
        if not full_groups or full_groups[-1][0] != sample.image_group_key:
            full_groups.append((sample.image_group_key, []))
        full_groups[-1][1].append(sample)
    observed_full_k_counts = Counter(len(samples) for _, samples in full_groups)
    if (
        len(full_groups) != FULL_GROUP_COUNT
        or dict(observed_full_k_counts) != FULL_K_COUNTS
        or len({key for key, _samples in full_groups}) != FULL_GROUP_COUNT
    ):
        raise RuntimeError(
            "full-867 group population changed: "
            f"groups={len(full_groups)}, K-counts={dict(observed_full_k_counts)}"
        )

    selected_by_id = {sample.sample_id: sample for sample in selected}
    if not {PAIR_A, PAIR_B}.issubset(selected_by_id):
        raise RuntimeError("fixed v4 counterfactual pair is absent")
    if selected_by_id[PAIR_A].short_answer != "2019" or (
        selected_by_id[PAIR_B].short_answer != "2017"
    ):
        raise RuntimeError("fixed v4 counterfactual values changed")

    ordered_payload = {
        "schema_version": "representation_internal_evaluation_group_manifest_v2",
        "identity": "qwen3-v4-clean-imend-test-golden-first200-variable-k-v1",
        "source_data_manifest_sha256": RETAINED_MANIFEST_SHA256,
        "groups": [
            {
                "image_group_key": key,
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "content_sha256": sample.content_sha256,
                    }
                    for sample in samples
                ],
            }
            for key, samples in groups
        ],
    }
    full_ordered_payload = {
        "schema_version": "representation_internal_evaluation_group_manifest_v2",
        "identity": "qwen3-v4-clean-imend-test-full867-variable-k-v1",
        "source_data_manifest_sha256": RETAINED_MANIFEST_SHA256,
        "groups": [
            {
                "image_group_key": key,
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "content_sha256": sample.content_sha256,
                    }
                    for sample in samples
                ],
            }
            for key, samples in full_groups
        ],
    }
    counterfactual_payload = {
        "schema_version": "qwen3_counterfactual_manifest_v1",
        "identity": "qwen3-v4-clean-imend-test-golden-value-pair-v1",
        "source_data_manifest_sha256": RETAINED_MANIFEST_SHA256,
        "pairs": [
            {
                "schema_version": "qwen3_counterfactual_pair_v1",
                "pair_id": "chartqa-tallest-bar-year-2019-vs-2017",
                "sample_a_id": PAIR_A,
                "sample_b_id": PAIR_B,
                "expected_value_a": "2019",
                "expected_value_b": "2017",
                "pair_audit_identity": (
                    "v4-clean-imend-golden-first200-exact-question-target-"
                    "distinct-image-matched-qwen3-grid-v1"
                ),
            }
        ],
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outputs = {
        OUTPUT_DIRECTORY
        / "qwen3_v4_clean_imend_test_golden_first200_variable_k_v1.json": (
            ordered_payload
        ),
        OUTPUT_DIRECTORY / "qwen3_v4_clean_imend_test_full867_variable_k_v1.json": (
            full_ordered_payload
        ),
        OUTPUT_DIRECTORY / "qwen3_v4_clean_imend_test_golden_counterfactual_v1.json": (
            counterfactual_payload
        ),
    }
    for path, payload in outputs.items():
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite changed manifest: {path}")
        path.write_text(encoded, encoding="utf-8")
        print(f"{sha256(encoded.encode('utf-8')).hexdigest()}  {path}")


if __name__ == "__main__":
    main()
