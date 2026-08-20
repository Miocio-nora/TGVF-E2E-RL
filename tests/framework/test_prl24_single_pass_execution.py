from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from tgvf_rl.framework.verl.prl24_single_pass_execution import (
    _cpu_bilinear_coefficients,
    cached_fast_pos_embed_interpolate,
    crop_aware_micro_block_schedule,
    estimate_qwen3_training_costs,
    install_prl24_single_pass_rollout_bypass,
    target_only_linear_for_ppo,
)


class _VisionPositionFixture:
    def __init__(self, *, dtype: torch.dtype) -> None:
        self.num_grid_per_side = 12
        self.pos_embed = torch.nn.Embedding(12 * 12, 16).to(dtype=dtype)
        self.config = SimpleNamespace(spatial_merge_size=2)


def test_crop_aware_schedule_preserves_every_loss_micro() -> None:
    block_costs = [100, 90, 80, 70, 4, 3, 2, 1]
    sample_costs = [cost // 2 for cost in block_costs for _ in range(2)]
    indices, metrics = crop_aware_micro_block_schedule(
        sample_costs, dp_size=4, micro_batch_size=2
    )

    original_blocks = {frozenset(range(start, start + 2)) for start in range(0, 16, 2)}
    scheduled_blocks = {
        frozenset(indices[start : start + 2]) for start in range(0, 16, 2)
    }
    assert scheduled_blocks == original_blocks
    assert sorted(indices) == list(range(16))
    assert metrics["actor/crop_aware_block_schedule_old_critical_cost"] == 190.0
    assert metrics["actor/crop_aware_block_schedule_new_critical_cost"] == 104.0
    assert metrics["actor/crop_aware_block_schedule_ratio"] == 104 / 190


def test_qwen3_cost_model_counts_crop_vision_work() -> None:
    config = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct/config.json"
    costs = estimate_qwen3_training_costs(
        [1000, 1000, 2000],
        [
            {"images_seqlens": torch.tensor([400, 900])},
            {},
            {},
        ],
        model_config_path=config,
    )
    assert costs[0] > costs[1]
    assert costs[2] > costs[1]


def test_cached_vision_position_interpolation_is_bitwise_upstream() -> None:
    from verl.models.transformers.qwen3_vl import fast_pos_embed_interpolate

    torch.manual_seed(19)
    fixture = _VisionPositionFixture(dtype=torch.bfloat16)
    grid = torch.tensor([[1, 6, 8], [2, 10, 12], [1, 6, 8]])
    _cpu_bilinear_coefficients.cache_clear()

    expected = fast_pos_embed_interpolate(fixture, grid)
    actual = cached_fast_pos_embed_interpolate(fixture, grid)
    repeated = cached_fast_pos_embed_interpolate(fixture, grid)

    assert torch.equal(actual, expected)
    assert torch.equal(repeated, expected)
    cache = _cpu_bilinear_coefficients.cache_info()
    assert cache.misses == 2
    assert cache.hits >= 4


def test_target_only_projection_matches_upstream_value_and_gradient() -> None:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    torch.manual_seed(23)
    hidden = torch.randn(2, 5, 11, dtype=torch.float64, requires_grad=True)
    vocab = torch.randn(17, 11, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(17, (2, 5))
    coefficients = torch.linspace(-1.0, 1.0, 10, dtype=torch.float64).view(2, 5)

    upstream_log_probs, _unused_entropy = FusedLinearForPPO(chunk_size=3)(
        hidden, vocab, labels, temperature=0.7
    )
    upstream_loss = (upstream_log_probs * coefficients).sum()
    upstream_gradients = torch.autograd.grad(upstream_loss, (hidden, vocab))

    target_hidden = hidden.detach().clone().requires_grad_(True)
    target_vocab = vocab.detach().clone().requires_grad_(True)
    target_log_probs = target_only_linear_for_ppo(
        target_hidden,
        target_vocab,
        labels,
        temperature=0.7,
        chunk_size=3,
    )
    target_loss = (target_log_probs * coefficients).sum()
    target_gradients = torch.autograd.grad(target_loss, (target_hidden, target_vocab))

    torch.testing.assert_close(target_log_probs, upstream_log_probs, rtol=0, atol=0)
    torch.testing.assert_close(target_loss, upstream_loss, rtol=0, atol=0)
    for actual, expected in zip(target_gradients, upstream_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rollout_bypass_preserves_single_pass_registry_identity() -> None:
    from verl.trainer.ppo import rollout_corr_helper

    original = rollout_corr_helper.apply_bypass_mode
    marker = "_tgvf_prl24_single_pass_v1"
    previous_marker = getattr(rollout_corr_helper, marker, None)
    batch = SimpleNamespace(batch={"rollout_log_probs": torch.tensor([[1.0, 2.0]])})
    correction = OmegaConf.create(
        {"bypass_mode": True, "rollout_is": None, "rollout_rs": None}
    )
    policy_loss = OmegaConf.create(
        {"loss_mode": "deepeyes_single_pass_micro_token_mean"}
    )
    try:
        install_prl24_single_pass_rollout_bypass()
        rollout_corr_helper.apply_bypass_mode(
            batch=batch,
            rollout_corr_config=correction,
            policy_loss_config=policy_loss,
        )
        assert batch.batch["old_log_probs"] is batch.batch["rollout_log_probs"]
        assert policy_loss.loss_mode == ("deepeyes_single_pass_micro_token_mean")
        assert policy_loss.rollout_correction.bypass_mode is True
    finally:
        rollout_corr_helper.apply_bypass_mode = original
        if previous_marker is None:
            delattr(rollout_corr_helper, marker)
        else:
            setattr(rollout_corr_helper, marker, previous_marker)
