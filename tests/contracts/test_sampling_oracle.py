from __future__ import annotations

from dataclasses import replace
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.sampling import (
    UnsupportedVLLMSamplingTransformError,
    VLLM_V1_ORACLE_VERSION,
    vllm_v1_processed_logprobs,
)
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity


SHA0 = "0" * 64
SHA1 = "1" * 64


def _sampling(**updates: object) -> SamplingIdentity:
    base = SamplingIdentity(
        policy_version=PolicyVersion("sampling-oracle", 0, SHA0),
        backend="vllm",
        backend_version=VLLM_V1_ORACLE_VERSION,
        seed=17,
        rng_state_sha256=SHA1,
        temperature=0.7,
        top_p=0.8,
        top_k=4,
        min_p=0.1,
        repetition_penalty=1.2,
        presence_penalty=0.2,
        frequency_penalty=0.1,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    return replace(base, **updates)


def test_nontrivial_processed_distribution_matches_independent_golden() -> None:
    result = vllm_v1_processed_logprobs(
        torch.tensor([2.0, 1.0, -0.5, 0.3, 1.5, -1.0], dtype=torch.float64),
        _sampling(),
        prompt_token_ids=(0, 2, 4),
        output_token_ids=(1, 1, 3),
    )

    assert result.dtype is torch.float32
    assert torch.isneginf(result[[1, 2, 3, 5]]).all()
    torch.testing.assert_close(
        result[[0, 4]],
        torch.tensor([-0.43917784, -1.03441596]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(result.exp().sum(), torch.tensor(1.0))


def test_output_history_controls_stateful_frequency_penalty() -> None:
    sampling = _sampling(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.4,
    )
    raw_logits = torch.tensor([0.0, 2.0, 0.0])
    after_one = vllm_v1_processed_logprobs(
        raw_logits,
        sampling,
        prompt_token_ids=(2,),
        output_token_ids=(1,),
    )
    after_three = vllm_v1_processed_logprobs(
        raw_logits,
        sampling,
        prompt_token_ids=(2,),
        output_token_ids=(1, 1, 1),
    )

    assert after_three[1] < after_one[1]
    assert after_three[0] > after_one[0]


def test_prompt_history_affects_repetition_but_not_frequency_or_presence() -> None:
    sampling = _sampling(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=2.0,
        presence_penalty=0.7,
        frequency_penalty=0.4,
    )
    raw_logits = torch.tensor([2.0, 1.0, 0.0])
    prompt_only = vllm_v1_processed_logprobs(
        raw_logits,
        sampling,
        prompt_token_ids=(0,),
        output_token_ids=(),
    )
    output_only = vllm_v1_processed_logprobs(
        raw_logits,
        sampling,
        prompt_token_ids=(),
        output_token_ids=(0,),
    )

    # Both paths apply repetition to token zero; only generated history also
    # receives presence and frequency subtraction.
    assert output_only[0] < prompt_only[0]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"logit_processors": ("project.module:CustomProcessor",)}, "processors"),
        ({"backend_version": "0.12.1"}, "pinned"),
        ({"temperature": 0.0}, "argmax behavior"),
        (
            {"measurement": LogProbMeasurement.RAW_MODEL},
            "not measured after sampling transforms",
        ),
    ],
)
def test_unmodeled_probability_measures_fail_closed(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(UnsupportedVLLMSamplingTransformError, match=message):
        vllm_v1_processed_logprobs(
            torch.tensor([1.0, 0.0]),
            _sampling(**updates),
            prompt_token_ids=(),
            output_token_ids=(),
        )


def test_invalid_explicit_history_fails_closed() -> None:
    with pytest.raises(ValueError, match="output_token_ids"):
        vllm_v1_processed_logprobs(
            torch.tensor([1.0, 0.0]),
            _sampling(top_k=0, min_p=0.0, top_p=1.0),
            prompt_token_ids=(),
            output_token_ids=(2,),
        )


def test_matches_installed_vllm_012_cpu_source_path() -> None:
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM is not installed; independent golden remains active")

    import vllm

    if vllm.__version__ != VLLM_V1_ORACLE_VERSION:
        pytest.skip(f"vLLM {VLLM_V1_ORACLE_VERSION} is not installed")

    from vllm import SamplingParams
    from vllm.model_executor.layers.utils import apply_penalties
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor
    from vllm.v1.sample.logits_processor.interface import BatchUpdate
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.sample.sampler import Sampler

    raw_logits = torch.tensor([2.0, 1.0, -0.5, 0.3, 1.5, -1.0], dtype=torch.float64)
    prompt_token_ids = [0, 2, 4]
    output_token_ids = [1, 1, 3]
    sampling = _sampling()

    expected = raw_logits.float().unsqueeze(0)
    apply_penalties(
        expected,
        torch.tensor([prompt_token_ids]),
        torch.tensor([output_token_ids]),
        torch.tensor([sampling.presence_penalty]),
        torch.tensor([sampling.frequency_penalty]),
        torch.tensor([sampling.repetition_penalty]),
    )
    expected = Sampler.apply_temperature(
        expected, torch.tensor([sampling.temperature]), all_random=True
    )

    processor = MinPLogitsProcessor(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=1)),
        torch.device("cpu"),
        False,
    )
    params = SamplingParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        repetition_penalty=sampling.repetition_penalty,
        presence_penalty=sampling.presence_penalty,
        frequency_penalty=sampling.frequency_penalty,
    )
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=((0, params, prompt_token_ids, output_token_ids),),
            moved=(),
        )
    )
    expected = processor.apply(expected)
    expected = apply_top_k_top_p(
        expected,
        torch.tensor([sampling.top_k], dtype=torch.int32),
        torch.tensor([sampling.top_p]),
    ).log_softmax(dim=-1, dtype=torch.float32)[0]

    actual = vllm_v1_processed_logprobs(
        raw_logits,
        sampling,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
