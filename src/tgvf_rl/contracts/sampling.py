"""Exact tensor oracle for vLLM v1 ``processed_logprobs``.

The oracle intentionally implements only the sampling transforms represented by
``SamplingIdentity``.  It is independent of vLLM at runtime so replay checks do
not silently change when an inference backend is upgraded.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .tokens import LogProbMeasurement, SamplingIdentity


VLLM_V1_ORACLE_VERSION = "0.12.0"
_SAMPLING_EPS = 1e-5
_MIN_RANDOM_TEMPERATURE = 1e-2


class UnsupportedVLLMSamplingTransformError(ValueError):
    """A recorded transform cannot be reproduced by this exact oracle."""


def vllm_v1_processed_logprobs(
    raw_logits: torch.Tensor,
    sampling: SamplingIdentity,
    *,
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
) -> torch.Tensor:
    """Return vLLM 0.12 v1 processed log probabilities for one request.

    ``prompt_token_ids`` is the complete request prompt and
    ``output_token_ids`` is the generated history *before* the token whose
    logits are supplied.  This distinction is required because repetition
    penalty sees both histories, while frequency and presence penalties see
    only generated output history.

    The implemented vLLM v1 order is repetition/frequency/presence penalties,
    temperature, min-p, joint top-k/top-p filtering, then float32 log-softmax.
    Custom processors, backend versions other than the audited vLLM 0.12.0,
    raw-logprob measurement identities, and greedy sampling fail closed.
    vLLM's reported greedy ``processed_logprobs`` is not the actual argmax
    sampling measure, so accepting it as behavior would be incorrect.  Built-in
    min-p is represented by ``SamplingIdentity.min_p`` and must not be repeated
    in ``SamplingIdentity.logit_processors``.
    """

    _validate_request(raw_logits, sampling, prompt_token_ids, output_token_ids)

    # vLLM casts sampling logits to float32 before applying any transform.
    logits = raw_logits.to(dtype=torch.float32).clone()
    vocab_size = logits.shape[0]
    prompt_counts = _token_counts(prompt_token_ids, vocab_size, logits.device)
    output_counts = _token_counts(output_token_ids, vocab_size, logits.device)

    repetition_mask = (prompt_counts > 0) | (output_counts > 0)
    repetition_scale = torch.where(
        logits > 0,
        1.0 / sampling.repetition_penalty,
        sampling.repetition_penalty,
    )
    logits = torch.where(repetition_mask, logits * repetition_scale, logits)
    logits = logits - sampling.frequency_penalty * output_counts
    logits = logits - sampling.presence_penalty * (output_counts > 0)

    logits = logits / sampling.temperature
    logits = _apply_min_p(logits, sampling.min_p)
    logits = _apply_top_k_top_p(logits, sampling.top_k, sampling.top_p)
    return torch.log_softmax(logits, dim=-1, dtype=torch.float32)


def _validate_request(
    raw_logits: torch.Tensor,
    sampling: SamplingIdentity,
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
) -> None:
    if not isinstance(raw_logits, torch.Tensor):
        raise TypeError("raw_logits must be a torch.Tensor")
    if raw_logits.ndim != 1 or raw_logits.numel() == 0:
        raise ValueError("raw_logits must have shape [vocab] with non-zero vocab")
    if not raw_logits.is_floating_point():
        raise TypeError("raw_logits must use a floating-point dtype")
    if not torch.isfinite(raw_logits).all():
        raise ValueError("raw model logits must be finite before sampling transforms")
    if not isinstance(sampling, SamplingIdentity):
        raise TypeError("sampling must be a SamplingIdentity")
    if sampling.backend_version != VLLM_V1_ORACLE_VERSION:
        raise UnsupportedVLLMSamplingTransformError(
            "processed-logprob oracle is pinned to vLLM 0.12.0; "
            f"recorded backend is {sampling.backend_version!r}"
        )
    if sampling.measurement is not LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS:
        raise UnsupportedVLLMSamplingTransformError(
            "sampling identity is not measured after sampling transforms"
        )
    if sampling.temperature < _SAMPLING_EPS:
        raise UnsupportedVLLMSamplingTransformError(
            "greedy vLLM processed_logprobs are not the actual argmax behavior measure"
        )
    if sampling.temperature < _MIN_RANDOM_TEMPERATURE:
        raise UnsupportedVLLMSamplingTransformError(
            "vLLM 0.12 clamps random temperatures below 0.01; record the effective "
            "backend temperature"
        )
    if not -2.0 <= sampling.presence_penalty <= 2.0:
        raise UnsupportedVLLMSamplingTransformError(
            "vLLM presence_penalty must be in [-2, 2]"
        )
    if not -2.0 <= sampling.frequency_penalty <= 2.0:
        raise UnsupportedVLLMSamplingTransformError(
            "vLLM frequency_penalty must be in [-2, 2]"
        )
    if sampling.logit_processors:
        names = ", ".join(sampling.logit_processors)
        raise UnsupportedVLLMSamplingTransformError(
            f"custom or otherwise unmodeled logit processors are unsupported: {names}"
        )

    _validate_history(prompt_token_ids, raw_logits.shape[0], "prompt_token_ids")
    _validate_history(output_token_ids, raw_logits.shape[0], "output_token_ids")


def _validate_history(
    token_ids: Sequence[int], vocab_size: int, field_name: str
) -> None:
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError(f"{field_name} must be a sequence of integer token IDs")
    invalid = [
        token_id
        for token_id in token_ids
        if type(token_id) is not int or not 0 <= token_id < vocab_size
    ]
    if invalid:
        raise ValueError(f"{field_name} contains invalid token IDs: {invalid}")


def _token_counts(
    token_ids: Sequence[int], vocab_size: int, device: torch.device
) -> torch.Tensor:
    counts = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    if token_ids:
        indices = torch.tensor(token_ids, dtype=torch.int64, device=device)
        counts.scatter_add_(0, indices, torch.ones_like(indices))
    return counts


def _apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p == 0.0:
        return logits
    probabilities = torch.softmax(logits, dim=-1)
    threshold = probabilities.amax(dim=-1, keepdim=True) * min_p
    return logits.masked_fill(probabilities < threshold, -torch.inf)


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    k_is_active = 0 < top_k < vocab_size
    p_is_active = top_p < 1.0
    if not k_is_active and not p_is_active:
        return logits

    # This follows vLLM's joint path: ascending sort, top-k boundary masking,
    # ascending cumulative probability masking, then scatter to vocabulary
    # order.  Strict comparisons deliberately preserve boundary ties.
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=False)
    if k_is_active:
        boundary = sorted_logits[vocab_size - top_k]
        sorted_logits = sorted_logits.masked_fill(sorted_logits < boundary, -torch.inf)
    if p_is_active:
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # vLLM metadata stores p as float32.  Materializing it at logits dtype
        # before subtraction preserves its boundary behavior (notably for 0.8).
        p = torch.as_tensor(top_p, dtype=logits.dtype, device=logits.device)
        top_p_mask = cumulative <= 1.0 - p
        top_p_mask[-1] = False
        sorted_logits = sorted_logits.masked_fill(top_p_mask, -torch.inf)

    return torch.empty_like(sorted_logits).scatter(
        dim=-1, index=sorted_indices, src=sorted_logits
    )


__all__ = [
    "UnsupportedVLLMSamplingTransformError",
    "VLLM_V1_ORACLE_VERSION",
    "vllm_v1_processed_logprobs",
]
