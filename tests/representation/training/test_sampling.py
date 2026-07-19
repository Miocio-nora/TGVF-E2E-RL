from __future__ import annotations

import json
from pathlib import Path

import pytest

from tgvf_rl.representation.training.sampling import (
    SameImageBatchSampler,
    partition_same_image_group,
    same_image_group_owner,
)
from tgvf_rl.representation.training.data import load_retained_representation_jsonl
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


DATA_MANIFEST_SHA256 = "1" * 64
K4_TRAIN_SOURCE_SHA256 = (
    "beb1b8a7c3f97811e8a8f9b0734d7484cc5de4d31861fe09b61342b3c88b61f2"
)


def _samples(group_sizes: dict[str, int]) -> tuple[RepresentationTrainingSample, ...]:
    samples = []
    for group_key, size in group_sizes.items():
        for member in range(size):
            samples.append(
                RepresentationTrainingSample(
                    sample_id=f"{group_key}-{member}",
                    image=f"/images/{group_key}.png",
                    image_id=group_key,
                    question=f"question {member}",
                    target=f"target {member}",
                    evidence_description=f"evidence {member}",
                )
            )
    return tuple(samples)


def _batch_group_keys(
    samples: tuple[RepresentationTrainingSample, ...], batch: tuple[int, ...]
) -> set[str]:
    return {samples[index].image_group_key for index in batch}


def test_k4_two_rank_fixture_has_one_complete_group_per_rank_and_ga4_geometry() -> None:
    fixture = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "representation_smoke"
        / "train_k4.jsonl"
    )
    dataset = load_retained_representation_jsonl(
        fixture,
        expected_source_sha256=K4_TRAIN_SOURCE_SHA256,
        warn_on_leakage=False,
    )
    assert len(dataset.samples) == 8
    assert same_image_group_owner("repr-smoke-train-image", world_size=2) == 0
    assert same_image_group_owner("repr-smoke-train-image-2", world_size=2) == 1

    sample_ids_by_rank: list[tuple[tuple[str, ...], ...]] = []
    for rank in range(2):
        sampler = SameImageBatchSampler(
            dataset.samples,
            batch_size=4,
            seed=71,
            data_manifest_sha256=dataset.manifest.manifest_sha256,
            rank=rank,
            world_size=2,
        )
        assert sampler.local_epoch_batch_count == 1
        groups = tuple(
            tuple(dataset.samples[index].sample_id for index in sampler.next_batch())
            for _ in range(4)
        )
        assert all(len(group) == 4 and len(set(group)) == 4 for group in groups)
        sample_ids_by_rank.append(groups)

    assert sum(len(group) for rank in sample_ids_by_rank for group in rank) == 32
    assert sum(len(rank) for rank in sample_ids_by_rank) == 8


@pytest.mark.parametrize(
    ("group_size", "expected"),
    [
        (0, ()),
        (3, ()),
        (4, (4,)),
        (5, (5,)),
        (6, (5,)),
        (8, (4, 4)),
        (9, (4, 5)),
        (10, (5, 5)),
        (11, (5, 5)),
        (12, (4, 4, 4)),
    ],
)
def test_batch_size_five_matches_golden_dynamic_partition(
    group_size: int, expected: tuple[int, ...]
) -> None:
    assert partition_same_image_group(group_size=group_size, batch_size=5) == expected


def test_other_batch_sizes_require_full_batches_and_drop_remainder() -> None:
    assert partition_same_image_group(group_size=11, batch_size=4) == (4, 4)
    assert partition_same_image_group(group_size=3, batch_size=4) == ()


def test_batches_never_mix_images_and_epoch_shuffle_is_deterministic() -> None:
    samples = _samples({"image-a": 10, "image-b": 10, "image-c": 10})
    first = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=17,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )
    twin = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=17,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )

    epoch_zero = list(first)
    assert epoch_zero == [
        (6, 3, 0, 2, 7),
        (8, 1, 9, 5, 4),
        (11, 15, 10, 19, 17),
        (14, 12, 18, 16, 13),
        (27, 20, 28, 21, 26),
        (24, 25, 29, 23, 22),
    ]
    assert epoch_zero == list(twin)
    assert all(len(_batch_group_keys(samples, batch)) == 1 for batch in epoch_zero)
    assert sorted(index for batch in epoch_zero for index in batch) == list(
        range(len(samples))
    )

    epoch_one = list(first)
    assert epoch_one != epoch_zero
    assert sorted(index for batch in epoch_one for index in batch) == list(
        range(len(samples))
    )


def test_sha1_owner_keeps_whole_groups_on_one_data_parallel_rank() -> None:
    keys_by_rank: dict[int, list[str]] = {0: [], 1: []}
    candidate = 0
    while not all(keys_by_rank.values()):
        key = f"image-{candidate}"
        keys_by_rank[same_image_group_owner(key, world_size=2)].append(key)
        candidate += 1
    selected_keys = tuple(keys_by_rank[0][:1] + keys_by_rank[1][:1])
    samples = _samples({key: 5 for key in selected_keys})

    rank_zero = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=3,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
        rank=0,
        world_size=2,
    )
    rank_one = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=3,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
        rank=1,
        world_size=2,
    )

    owned_zero = set(rank_zero.owned_group_keys)
    owned_one = set(rank_one.owned_group_keys)
    assert owned_zero.isdisjoint(owned_one)
    assert owned_zero | owned_one == set(selected_keys)
    for rank, sampler in enumerate((rank_zero, rank_one)):
        batches = tuple(
            sampler.next_batch() for _ in range(sampler.local_epoch_batch_count)
        )
        for batch in batches:
            (group_key,) = _batch_group_keys(samples, batch)
            assert same_image_group_owner(group_key, world_size=2) == rank
        with pytest.raises(RuntimeError, match="globally fixed steps"):
            iter(sampler).__next__()
        with pytest.raises(RuntimeError, match="globally fixed training steps"):
            len(sampler)


def test_unusable_groups_and_remainders_are_dropped_without_random_fallback() -> None:
    samples = _samples({"too-small": 3, "usable": 6})
    sampler = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=0,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )

    batches = list(sampler)
    assert tuple(map(len, batches)) == (5,)
    assert _batch_group_keys(samples, batches[0]) == {"usable"}
    assert "too-small" not in sampler.owned_group_keys
    with pytest.raises(TypeError):
        SameImageBatchSampler(  # type: ignore[call-arg]
            samples,
            batch_size=5,
            seed=0,
            data_manifest_sha256=DATA_MANIFEST_SHA256,
            shuffle=True,
        )


def test_state_round_trip_resumes_at_exact_next_batch() -> None:
    samples = _samples({"image-a": 15, "image-b": 10, "image-c": 9})
    uninterrupted = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=91,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )
    assert uninterrupted.next_batch()
    assert uninterrupted.next_batch()
    serialized_state = json.loads(json.dumps(uninterrupted.state_dict()))
    expected = tuple(uninterrupted.next_batch() for _ in range(7))

    resumed = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=91,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )
    resumed.load_state_dict(serialized_state)

    assert tuple(resumed.next_batch() for _ in range(7)) == expected


def test_state_load_fails_closed_for_identity_or_materialized_order_mismatch() -> None:
    samples = _samples({"image-a": 10, "image-b": 10})
    source = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=11,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )
    source.next_batch()
    state = source.state_dict()

    different_seed = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=12,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )
    with pytest.raises(ValueError, match="identity"):
        different_seed.load_state_dict(state)

    corrupted = json.loads(json.dumps(state))
    corrupted["epoch_batches"][0].reverse()
    target = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=11,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )
    with pytest.raises(ValueError, match="materialized epoch batches"):
        target.load_state_dict(corrupted)


def test_state_identity_binds_manifest_and_full_sample_content() -> None:
    samples = _samples({"image-a": 5})
    source = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=11,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )
    source.next_batch()
    state = source.state_dict()

    different_manifest = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=11,
        data_manifest_sha256="2" * 64,
    )
    with pytest.raises(ValueError, match="identity"):
        different_manifest.load_state_dict(state)

    changed = list(samples)
    original = changed[0]
    changed[0] = RepresentationTrainingSample(
        sample_id=original.sample_id,
        image=original.image,
        image_id=original.image_id,
        question=original.question,
        target="changed target with the same row and group IDs",
        evidence_description=original.evidence_description,
    )
    changed_content = SameImageBatchSampler(
        changed,
        batch_size=5,
        seed=11,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )
    with pytest.raises(ValueError, match="identity"):
        changed_content.load_state_dict(state)


def test_distributed_preflight_rejects_any_rank_without_a_usable_group() -> None:
    owned_by_zero = []
    candidate = 0
    while len(owned_by_zero) < 2:
        key = f"only-rank-zero-{candidate}"
        if same_image_group_owner(key, world_size=2) == 0:
            owned_by_zero.append(key)
        candidate += 1
    samples = _samples({key: 5 for key in owned_by_zero})

    with pytest.raises(ValueError, match=r"ranks with no permitted.*\(1,\)"):
        SameImageBatchSampler(
            samples,
            batch_size=5,
            seed=0,
            data_manifest_sha256=DATA_MANIFEST_SHA256,
            rank=0,
            world_size=2,
        )


def test_distributed_unequal_local_lengths_use_fixed_step_and_rank_local_resume() -> (
    None
):
    keys_by_rank: dict[int, list[str]] = {0: [], 1: []}
    candidate = 0
    while len(keys_by_rank[0]) < 2 or len(keys_by_rank[1]) < 1:
        key = f"unequal-rank-{candidate}"
        keys_by_rank[same_image_group_owner(key, world_size=2)].append(key)
        candidate += 1
    selected = keys_by_rank[0][:2] + keys_by_rank[1][:1]
    samples = _samples({key: 5 for key in selected})
    rank_zero = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=7,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
        rank=0,
        world_size=2,
    )
    rank_one = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=7,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
        rank=1,
        world_size=2,
    )

    assert rank_zero.usable_group_counts_by_rank == (2, 1)
    assert rank_one.usable_group_counts_by_rank == (2, 1)
    assert rank_zero.local_epoch_batch_count == 2
    assert rank_one.local_epoch_batch_count == 1
    rank_zero.next_batch()
    saved = json.loads(json.dumps(rank_zero.state_dict()))
    expected_next = rank_zero.next_batch()
    resumed_rank_zero = SameImageBatchSampler(
        samples,
        batch_size=5,
        seed=7,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
        rank=0,
        world_size=2,
    )
    resumed_rank_zero.load_state_dict(saved)
    assert resumed_rank_zero.next_batch() == expected_next
    with pytest.raises(ValueError, match="identity"):
        rank_one.load_state_dict(saved)


def test_manifest_sha256_is_required_and_validated() -> None:
    samples = _samples({"image-a": 5})
    with pytest.raises(ValueError, match="data_manifest_sha256"):
        SameImageBatchSampler(
            samples,
            batch_size=5,
            seed=0,
            data_manifest_sha256="not-a-sha",
        )


def test_duplicate_sample_identity_is_rejected() -> None:
    samples = list(_samples({"image-a": 5}))
    samples[-1] = RepresentationTrainingSample(
        sample_id=samples[0].sample_id,
        image="/images/image-a.png",
        image_id="image-a",
        question="another question",
        target="another target",
        evidence_description="another evidence",
    )

    with pytest.raises(ValueError, match="sample_id values must be unique"):
        SameImageBatchSampler(
            samples,
            batch_size=5,
            seed=0,
            data_manifest_sha256=DATA_MANIFEST_SHA256,
        )


def test_duplicate_target_within_one_image_group_is_rejected() -> None:
    samples = list(_samples({"image-a": 5}))
    original = samples[-1]
    samples[-1] = RepresentationTrainingSample(
        sample_id=original.sample_id,
        image=original.image,
        image_id=original.image_id,
        question=original.question,
        target=samples[0].target,
        evidence_description=original.evidence_description,
    )

    with pytest.raises(ValueError, match="exact distinct target strings"):
        SameImageBatchSampler(
            samples,
            batch_size=5,
            seed=0,
            data_manifest_sha256=DATA_MANIFEST_SHA256,
        )
