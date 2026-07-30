"""Deterministic matched-data sampler for answer-utility ablations."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

from tgvf_rl.representation.training.sampling import SameImageBatchSampler
from tgvf_rl.representation.training.schema import RepresentationTrainingSample

from .controls import _normalized_answer_identity


ANSWER_SAFE_SAMPLER_RULE = "answer-safe-same-image-batch-v1"


def is_answer_safe_batch(
    samples: Sequence[RepresentationTrainingSample],
    indices: tuple[int, ...],
) -> bool:
    """Whether every row has another same-image D with a different answer."""

    if not indices:
        raise ValueError("answer-safe batch cannot be empty")
    selected = tuple(samples[index] for index in indices)
    if len({sample.image_group_key for sample in selected}) != 1:
        raise ValueError("answer-safe filtering requires one same-image group")
    return (
        len({_normalized_answer_identity(sample.short_answer) for sample in selected})
        >= 2
    )


class AnswerSafeSameImageBatchSampler(SameImageBatchSampler):
    """Skip batches that cannot support an answer-safe wrong-D intervention.

    The same sampler is used by every matched-budget cell, including cells that
    do not instantiate a wrong-D branch. The inherited cursor therefore makes
    exact resume and cross-cell sample identity agree after every skipped raw
    batch.
    """

    def __init__(
        self,
        samples: Sequence[RepresentationTrainingSample],
        *,
        batch_size: int,
        seed: int,
        data_manifest_sha256: str,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self._answer_safe_samples = tuple(samples)
        self.skipped_batch_count = 0
        super().__init__(
            samples,
            batch_size=batch_size,
            seed=seed,
            data_manifest_sha256=data_manifest_sha256,
            rank=rank,
            world_size=world_size,
        )
        self._identity_sha256 = sha256(
            f"{self._identity_sha256}:{ANSWER_SAFE_SAMPLER_RULE}".encode("utf-8")
        ).hexdigest()
        if not any(
            is_answer_safe_batch(self._answer_safe_samples, batch)
            for batch in self._materialize_epoch(0)
        ):
            raise ValueError("sampler epoch 0 has no answer-safe same-image batch")

    def next_batch(self) -> tuple[int, ...]:
        maximum_attempts = max(1, self.local_epoch_batch_count * 2)
        for _ in range(maximum_attempts):
            batch = super().next_batch()
            if is_answer_safe_batch(self._answer_safe_samples, batch):
                return batch
            self.skipped_batch_count += 1
        raise RuntimeError("two sampler epochs produced no answer-safe batch")


__all__ = [
    "ANSWER_SAFE_SAMPLER_RULE",
    "AnswerSafeSameImageBatchSampler",
    "is_answer_safe_batch",
]
