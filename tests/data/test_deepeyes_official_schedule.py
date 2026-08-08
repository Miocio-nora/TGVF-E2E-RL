from __future__ import annotations

from collections import Counter
from pathlib import Path

from tgvf_rl.data.deepeyes_official_schedule import (
    DEEPEYES_BATCH_COUNTS,
    DEEPEYES_PROBE_SEED,
    DEEPEYES_TRAIN_SEED,
    DeepEyesOfficialSample,
    assert_verl_route_contract,
    build_deepeyes_schedule,
)


def synthetic_official_pool() -> tuple[DeepEyesOfficialSample, ...]:
    required = {
        # One extra source-balanced block remains available for the isolated
        # four-row launcher smoke split after formal train + probe selection.
        source: count * 82 for source, count in DEEPEYES_BATCH_COUNTS.items()
    }
    samples: list[DeepEyesOfficialSample] = []
    for source, count in required.items():
        for source_index in range(count):
            samples.append(
                DeepEyesOfficialSample(
                    index=len(samples),
                    sample_id=f"{source}-{source_index:05d}",
                    candidate_sha256="a" * 64,
                    data_source=source,
                    task_kind="math" if source == "thinklite" else "open",
                    question=f"question {source_index}",
                    ground_truth="answer",
                    image_path=Path(f"/images/{source}-{source_index}.png"),
                    image_sha256="b" * 64,
                    image_width=100,
                    image_height=80,
                    gt_regions=((1, 2, 30, 40),) if source == "vstar" else None,
                )
            )
    return tuple(samples)


def test_exact_stratified_schedule_probe_and_routes() -> None:
    samples = synthetic_official_pool()
    schedule = build_deepeyes_schedule(samples, mode="stratified")
    assert len(schedule.batches) == 80
    assert len({index for batch in schedule.batches for index in batch}) == 20_480
    assert not set(schedule.probe_indices).intersection(
        index for batch in schedule.batches for index in batch
    )
    for batch in schedule.batches:
        assert Counter(samples[index].data_source for index in batch) == Counter(
            DEEPEYES_BATCH_COUNTS
        )
    assert schedule.probe_manifest["source_counts"] == dict(DEEPEYES_BATCH_COUNTS)
    for source in ("vstar", "arxivqa", "thinklite"):
        assert_verl_route_contract(
            next(sample for sample in samples if sample.data_source == source)
        )


def test_schedule_is_deterministic_and_natural_arm_is_nonrepeating() -> None:
    samples = synthetic_official_pool()
    first = build_deepeyes_schedule(
        samples,
        mode="natural",
        seed=DEEPEYES_TRAIN_SEED,
        probe_seed=DEEPEYES_PROBE_SEED,
    )
    second = build_deepeyes_schedule(samples, mode="natural")
    assert first.identity_sha256 == second.identity_sha256
    assert first.batches == second.batches
    assert first.probe_indices == second.probe_indices
    flattened = [index for batch in first.batches for index in batch]
    assert len(flattened) == len(set(flattened)) == 20_480
