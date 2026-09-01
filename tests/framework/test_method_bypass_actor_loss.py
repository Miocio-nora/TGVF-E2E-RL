from __future__ import annotations

import importlib
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

_verl_core_algos = pytest.importorskip(
    "verl.trainer.ppo.core_algos",
    reason="method bypass registration requires the optional pinned veRL",
)
get_policy_loss_fn = _verl_core_algos.get_policy_loss_fn

from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode  # noqa: E402

from tgvf_rl.framework.verl.dynamic_token_loss_contract import (  # noqa: E402
    DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE,
    METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME,
)
from tgvf_rl.framework.verl.method_bypass_actor_loss import (  # noqa: E402
    compute_method_matrix_bypass_loss,
)
from tgvf_rl.framework.verl.native_deepeyes_runtime import (  # noqa: E402
    NATIVE_DEEPEYES_POLICY_LOSS_MODE,
)


def _actor_config(*, dynamic: bool) -> SimpleNamespace:
    semantic_loss_mode = (
        DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE
        if dynamic
        else NATIVE_DEEPEYES_POLICY_LOSS_MODE
    )
    return SimpleNamespace(
        policy_loss=OmegaConf.create(
            {
                "loss_mode": semantic_loss_mode,
                "rollout_correction": {
                    "bypass_mode": True,
                    "loss_type": "ppo_clip",
                    "rollout_is": None,
                    "rollout_rs": None,
                    "rollout_is_batch_normalize": False,
                },
            }
        ),
        use_dynamic_bsz=dynamic,
        ppo_micro_batch_size_per_gpu=None if dynamic else 2,
        ppo_epochs=1,
        entropy_coeff=0.0,
        use_kl_loss=False,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        global_batch_info={
            "dp_size": 1,
            "batch_num_tokens": 5,
            "global_batch_size": 4,
        },
    )


def _dual_clip_per_token(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    ratio = torch.exp(log_prob - old_log_prob)
    raw = -advantages * ratio
    clipped = -advantages * ratio.clamp(min=0.8, max=1.2)
    upper = torch.maximum(raw, clipped)
    return torch.where(advantages < 0, torch.minimum(-advantages * 3.0, upper), upper)


@pytest.mark.parametrize("dynamic", (False, True))
def test_apply_bypass_overwrite_dispatches_to_bound_reduction(dynamic: bool) -> None:
    # The worker external-lib hook must reinstall our dispatcher even if another
    # test or integration imported veRL's default ``bypass_mode`` afterward.
    external = importlib.import_module("tgvf_rl.framework.verl.trainable_crop_external")
    importlib.reload(external)
    assert get_policy_loss_fn(METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME) is (
        compute_method_matrix_bypass_loss
    )

    config = _actor_config(dynamic=dynamic)
    old_log_prob = torch.zeros((2, 3), dtype=torch.float64)
    log_prob = torch.tensor(
        [[-0.1, 0.0, 0.2], [0.05, -0.05, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    advantages = torch.tensor(
        [[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    batch = SimpleNamespace(batch={"rollout_log_probs": old_log_prob})

    apply_bypass_mode(
        batch,
        rollout_corr_config=config.policy_loss.rollout_correction,
        policy_loss_config=config.policy_loss,
    )

    assert batch.batch["old_log_probs"] is old_log_prob
    assert config.policy_loss.loss_mode == METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME
    loss, metrics = get_policy_loss_fn(config.policy_loss.loss_mode)(
        old_log_prob=batch.batch["old_log_probs"],
        log_prob=log_prob,
        advantages=advantages,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=config,
    )

    per_token = _dual_clip_per_token(old_log_prob, log_prob, advantages)
    token_sum = per_token.masked_select(mask).sum()
    expected = token_sum / 5 if dynamic else token_sum / 5 / 2
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-12)
    assert metrics["actor/tgvf_dynamic_token_batching"] == float(dynamic)


def test_method_bypass_rejects_ambiguous_batching_identity() -> None:
    config = _actor_config(dynamic=True)
    config.use_dynamic_bsz = "true"
    config.policy_loss.loss_mode = METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME
    zeros = torch.zeros((2, 3), dtype=torch.float64)

    with pytest.raises(TypeError, match="explicit actor.use_dynamic_bsz"):
        compute_method_matrix_bypass_loss(
            zeros,
            zeros,
            torch.ones_like(zeros),
            torch.ones_like(zeros, dtype=torch.bool),
            config=config,
        )
