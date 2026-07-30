from __future__ import annotations

import json
from pathlib import Path
import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test lane
    import tomli as tomllib

    sys.modules.setdefault("tomllib", tomllib)

from PIL import Image

from tgvf_rl.representation.experiments.image_axis_grounding.matching import (
    ImageAxisDonorManifest,
    ImageAxisDonorSourceBinding,
    QwenImageGridContract,
    build_image_axis_donor_manifest,
    load_image_axis_donor_manifest,
    load_qwen_image_grid_contract,
    materialize_image_axis_donor_manifest,
    qwen_image_grid_thw,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 64), color=color).save(path)


def _sample(
    index: int,
    *,
    image: Path,
    image_id: str,
    answer: str,
    source_dataset: str = "source-a",
    source_profile: str = "profile-a",
    stable_image_uid: str | None = None,
) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=f"sample-{index}",
        image=str(image.resolve()),
        image_id=image_id,
        question=f"question {index}",
        target=f"target {index}",
        evidence_description=f"evidence {index}",
        short_answer=answer,
        source_dataset=source_dataset,
        source_profile=source_profile,
        stable_image_uid=stable_image_uid,
    )


def _grid_contract() -> QwenImageGridContract:
    return QwenImageGridContract(
        patch_size=16,
        merge_size=2,
        min_pixels=32 * 32,
        max_pixels=64 * 64,
    )


def _source_binding() -> ImageAxisDonorSourceBinding:
    return ImageAxisDonorSourceBinding(
        train_source_sha256="1" * 64,
        retained_manifest_sha256="2" * 64,
        raw_image_manifest_sha256="3" * 64,
        preprocessor_config_sha256="4" * 64,
    )


def test_manifest_is_exact_grid_answer_disjoint_and_order_invariant(
    tmp_path: Path,
) -> None:
    anchor_path = tmp_path / "anchor.png"
    donor_path = tmp_path / "donor.png"
    rejected_path = tmp_path / "rejected.png"
    _write_image(anchor_path, (1, 2, 3))
    _write_image(donor_path, (4, 5, 6))
    _write_image(rejected_path, (7, 8, 9))
    anchors = (
        _sample(
            0,
            image=anchor_path,
            image_id="anchor",
            answer="Red",
            stable_image_uid="anchor-uid",
        ),
        _sample(
            1,
            image=anchor_path,
            image_id="anchor",
            answer="Blue",
            stable_image_uid="anchor-uid",
        ),
    )
    donors = (
        _sample(
            2,
            image=donor_path,
            image_id="donor",
            answer="Green",
            stable_image_uid="donor-uid",
        ),
        _sample(
            3,
            image=donor_path,
            image_id="donor",
            answer="Yellow",
            stable_image_uid="donor-uid",
        ),
        _sample(
            4,
            image=rejected_path,
            image_id="answer-overlap",
            answer="  RED ",
            stable_image_uid="other-uid",
        ),
    )

    first = build_image_axis_donor_manifest(
        anchors,
        donors,
        grid_contract=_grid_contract(),
        source_binding=_source_binding(),
        random_seed=42,
    )
    reordered = build_image_axis_donor_manifest(
        tuple(reversed(anchors)),
        tuple(reversed(donors)),
        grid_contract=_grid_contract(),
        source_binding=_source_binding(),
        random_seed=42,
    )

    assert isinstance(first, ImageAxisDonorManifest)
    assert first == reordered
    assert first.identity_sha256 == reordered.identity_sha256
    assert first.matched_count == 1
    assert first.unmatched_count == 0
    assignment = first.assignment_for("anchor")
    assert assignment.matched
    assert assignment.donor_image_group_key == "donor"
    assert assignment.match_tier == "exact_grid_same_source_dataset"
    assert assignment.image_grid_thw == (1, 4, 4)
    assert assignment.anchor_image_sha256 != assignment.donor_image_sha256
    assert first.to_payload()["matched_count"] == 1


def test_manifest_retains_explicit_unmatched_anchor_for_masking(tmp_path: Path) -> None:
    anchor_path = tmp_path / "anchor.png"
    same_answer_path = tmp_path / "same-answer.png"
    duplicate_path = tmp_path / "duplicate.png"
    _write_image(anchor_path, (1, 2, 3))
    _write_image(same_answer_path, (4, 5, 6))
    duplicate_path.write_bytes(anchor_path.read_bytes())
    anchor = _sample(
        0,
        image=anchor_path,
        image_id="anchor",
        answer="500",
    )
    donors = (
        _sample(
            1,
            image=same_answer_path,
            image_id="same-answer",
            answer=" 500 ",
        ),
        _sample(
            2,
            image=duplicate_path,
            image_id="duplicate-bytes",
            answer="600",
        ),
    )

    manifest = build_image_axis_donor_manifest(
        (anchor,),
        donors,
        grid_contract=_grid_contract(),
        source_binding=_source_binding(),
        random_seed=42,
    )

    assignment = manifest.assignment_for("anchor")
    assert not assignment.matched
    assert assignment.unmatched_reason == (
        "no_exact_grid_answer_disjoint_distinct_image"
    )
    assert assignment.donor_image is None
    assert manifest.matched_count == 0
    assert manifest.unmatched_count == 1


def test_manifest_identity_binds_seed_and_complete_donor_population(
    tmp_path: Path,
) -> None:
    anchor_path = tmp_path / "anchor.png"
    donor_path = tmp_path / "donor.png"
    extra_path = tmp_path / "extra.png"
    _write_image(anchor_path, (1, 2, 3))
    _write_image(donor_path, (4, 5, 6))
    _write_image(extra_path, (7, 8, 9))
    anchor = _sample(0, image=anchor_path, image_id="anchor", answer="a")
    donor = _sample(1, image=donor_path, image_id="donor", answer="b")
    extra = _sample(2, image=extra_path, image_id="extra", answer="c")

    baseline = build_image_axis_donor_manifest(
        (anchor,),
        (donor,),
        grid_contract=_grid_contract(),
        source_binding=_source_binding(),
        random_seed=42,
    )
    changed_seed = build_image_axis_donor_manifest(
        (anchor,),
        (donor,),
        grid_contract=_grid_contract(),
        source_binding=_source_binding(),
        random_seed=43,
    )
    changed_pool = build_image_axis_donor_manifest(
        (anchor,),
        (donor, extra),
        grid_contract=_grid_contract(),
        source_binding=_source_binding(),
        random_seed=42,
    )

    assert baseline.identity_sha256 != changed_seed.identity_sha256
    assert baseline.donor_population_sha256 != changed_pool.donor_population_sha256
    assert baseline.identity_sha256 != changed_pool.identity_sha256


def test_grid_contract_loads_preprocessor_and_honors_run_pixel_cap(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "preprocessor_config.json"
    config_path.write_text(
        json.dumps(
            {
                "patch_size": 16,
                "merge_size": 2,
                "size": {"shortest_edge": 1024, "longest_edge": 999999},
            }
        ),
        encoding="utf-8",
    )
    image_path = tmp_path / "image.png"
    _write_image(image_path, (1, 2, 3))

    contract = load_qwen_image_grid_contract(
        config_path,
        image_max_pixels=4096,
    )

    assert contract == QwenImageGridContract(16, 2, 1024, 4096)
    assert qwen_image_grid_thw(image_path, contract) == (1, 4, 4)
    assert len(contract.identity_sha256) == 64


def test_persisted_manifest_loader_rederives_identity_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    anchor_path = tmp_path / "anchor.png"
    donor_path = tmp_path / "donor.png"
    _write_image(anchor_path, (1, 2, 3))
    _write_image(donor_path, (4, 5, 6))
    manifest = build_image_axis_donor_manifest(
        (_sample(0, image=anchor_path, image_id="anchor", answer="a"),),
        (_sample(1, image=donor_path, image_id="donor", answer="b"),),
        grid_contract=_grid_contract(),
        source_binding=_source_binding(),
        random_seed=42,
    )
    output = tmp_path / "donors.json"

    assert materialize_image_axis_donor_manifest(manifest, output) == output.resolve()
    assert materialize_image_axis_donor_manifest(manifest, output) == output.resolve()
    assert load_image_axis_donor_manifest(output) == manifest

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["random_seed"] = 43
    output.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_image_axis_donor_manifest(output)
    except ValueError as error:
        assert "identity or derived fields" in str(error)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("tampered manifest was accepted")
