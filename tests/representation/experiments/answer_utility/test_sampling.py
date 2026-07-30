from __future__ import annotations

from tgvf_rl.representation.experiments.answer_utility.sampling import (
    AnswerSafeSameImageBatchSampler,
    is_answer_safe_batch,
)
from tgvf_rl.representation.training.sampling import SameImageBatchSampler
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


_MANIFEST_SHA256 = "a" * 64


def _samples() -> tuple[RepresentationTrainingSample, ...]:
    rows = []
    for group, answers in (
        ("unsafe", ("same", "same", "same", "same")),
        ("safe", ("same", "same", "same", "different")),
    ):
        rows.extend(
            RepresentationTrainingSample(
                sample_id=f"{group}-{index}",
                image=f"/fixture/{group}.png",
                image_id=group,
                question=f"question {group} {index}",
                target=f"target {group} {index}",
                evidence_description=f"evidence {group} {index}",
                short_answer=answer,
            )
            for index, answer in enumerate(answers)
        )
    return tuple(rows)


def _sampler(
    cls: type[SameImageBatchSampler],
    samples: tuple[RepresentationTrainingSample, ...],
    *,
    seed: int,
) -> SameImageBatchSampler:
    return cls(
        samples,
        batch_size=4,
        seed=seed,
        data_manifest_sha256=_MANIFEST_SHA256,
    )


def test_answer_safe_sampler_skips_unsafe_group_and_resumes_exactly() -> None:
    samples = _samples()
    seed = next(
        candidate
        for candidate in range(100)
        if not is_answer_safe_batch(
            samples,
            _sampler(SameImageBatchSampler, samples, seed=candidate).next_batch(),
        )
    )
    sampler = _sampler(AnswerSafeSameImageBatchSampler, samples, seed=seed)

    first = sampler.next_batch()

    assert is_answer_safe_batch(samples, first)
    assert sampler.skipped_batch_count == 1
    state = sampler.state_dict()
    expected_next = sampler.next_batch()
    restored = _sampler(AnswerSafeSameImageBatchSampler, samples, seed=seed)
    restored.load_state_dict(state)
    assert restored.next_batch() == expected_next
