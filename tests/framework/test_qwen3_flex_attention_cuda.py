from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() == 0,
    reason="Qwen3-VL FlexAttention parity requires CUDA",
)
def test_qwen3_vl_packed_text_flex_matches_sdpa_forward_and_backward() -> None:
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    from tgvf_rl.framework.verl.qwen3_flex_attention_compat import (
        install_qwen3_vl_text_flex_attention_compat,
    )

    install_qwen3_vl_text_flex_attention_compat()
    common = {
        "vocab_size": 64,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 128,
        "attention_dropout": 0.0,
        "use_cache": False,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [4, 2, 2],
            "rope_type": "default",
        },
    }
    sdpa_config = Qwen3VLTextConfig(**common)
    sdpa_config._attn_implementation = "sdpa"
    flex_config = Qwen3VLTextConfig(**common)
    flex_config._attn_implementation = "flex_attention"

    torch.manual_seed(17)
    sdpa = Qwen3VLTextModel(sdpa_config).to(device="cuda:0", dtype=torch.bfloat16)
    flex = Qwen3VLTextModel(flex_config).to(device="cuda:0", dtype=torch.bfloat16)
    flex.load_state_dict(sdpa.state_dict())
    sdpa.train()
    flex.train()

    # Two packed documents of lengths 3 and 2.  The position reset is the
    # contract used by veRL's remove-padding FSDP engine.
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], device="cuda:0")
    base = torch.randn(
        1, 5, common["hidden_size"], device="cuda:0", dtype=torch.bfloat16
    )
    sdpa_input = base.detach().clone().requires_grad_(True)
    flex_input = base.detach().clone().requires_grad_(True)

    sdpa_output = sdpa(
        inputs_embeds=sdpa_input,
        position_ids=position_ids,
        use_cache=False,
    ).last_hidden_state
    flex_output = flex(
        inputs_embeds=flex_input,
        position_ids=position_ids,
        use_cache=False,
    ).last_hidden_state
    torch.testing.assert_close(flex_output, sdpa_output, rtol=0.04, atol=0.04)

    sdpa_output.float().square().mean().backward()
    flex_output.float().square().mean().backward()
    torch.testing.assert_close(flex_input.grad, sdpa_input.grad, rtol=0.08, atol=0.08)
    torch.testing.assert_close(
        flex.layers[0].self_attn.q_proj.weight.grad,
        sdpa.layers[0].self_attn.q_proj.weight.grad,
        rtol=0.08,
        atol=0.08,
    )

    # Changing document 1 cannot alter document 2 under the packed BlockMask.
    perturbed = base.detach().clone()
    perturbed[:, :3].add_(4)
    with torch.no_grad():
        flex_perturbed = flex(
            inputs_embeds=perturbed,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
    torch.testing.assert_close(
        flex_perturbed[:, 3:], flex_output.detach()[:, 3:], rtol=0, atol=0
    )


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() == 0,
    reason="variable-length FlexAttention checkpoint regression requires CUDA",
)
def test_qwen3_vl_variable_length_nonreentrant_checkpoint_backward() -> None:
    """One exact-replay micro may contain several decoder sequence lengths.

    Before compat v2, the first four lengths below reliably failed with the
    production error: checkpoint tensor positions 31--34 were permuted between
    scalar/padding-mask and the two int32 BlockMask tensors.  Eight lengths
    model the treatment's per-trajectory replay while keeping this a one-layer
    CUDA regression test.
    """

    import torch
    from transformers.integrations import flex_attention as flex_runtime
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    from tgvf_rl.framework.verl.qwen3_flex_attention_compat import (
        install_qwen3_vl_text_flex_attention_compat,
    )

    install_qwen3_vl_text_flex_attention_compat()
    flex_runtime.WrappedFlexAttention._instance = None
    flex_runtime.WrappedFlexAttention._is_flex_compiled = False
    flex_runtime.WrappedFlexAttention._compiled_flex_attention = None

    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=512,
        attention_dropout=0.0,
        use_cache=False,
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": [8, 4, 4],
            "rope_type": "default",
        },
    )
    config._attn_implementation = "flex_attention"
    model = Qwen3VLTextModel(config).to(device="cuda:0", dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    checkpoint_fn = model.layers[0]._gradient_checkpointing_func
    assert isinstance(checkpoint_fn, partial)
    assert checkpoint_fn.keywords["use_reentrant"] is False

    policy_log_probs: list[torch.Tensor] = []
    inputs: list[torch.Tensor] = []
    for sequence_length in (282, 310, 314, 308, 290, 330, 278, 300):
        values = torch.randn(
            1,
            sequence_length,
            config.hidden_size,
            device="cuda:0",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        inputs.append(values)
        attention_mask = torch.ones(
            1, sequence_length, dtype=torch.bool, device="cuda:0"
        )
        position_ids = (
            torch.arange(sequence_length, device="cuda:0")
            .view(1, 1, -1)
            .expand(3, 1, -1)
        )
        output = model(
            inputs_embeds=values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        # One differentiable scalar per generated token is sufficient for the
        # exact DeepEyes policy-loss regression.  The actor kernel only needs
        # current/old token log probabilities with their real replay shapes;
        # the tiny text model deliberately omits the production LM head.
        policy_log_probs.append(output[..., 0].float().squeeze(0))

    from tgvf_rl.framework.verl.deepeyes_actor_loss import (
        DEEPEYES_OFFICIAL_LOSS_AGG_MODE,
        DEEPEYES_OFFICIAL_POLICY_LOSS_MODE,
        compute_deepeyes_official_micro_token_mean_loss,
    )

    maximum_length = max(value.shape[0] for value in policy_log_probs)
    log_prob = torch.stack(
        [
            torch.nn.functional.pad(value, (0, maximum_length - value.shape[0]))
            for value in policy_log_probs
        ]
    )
    response_mask = torch.stack(
        [
            torch.nn.functional.pad(
                torch.ones_like(value, dtype=torch.bool),
                (0, maximum_length - value.shape[0]),
            )
            for value in policy_log_probs
        ]
    )
    # Four deterministic n=2 groups provide genuine non-zero GRPO-style
    # advantages.  This is an engineering regression, not a sampled RL result.
    sequence_advantages = torch.tensor(
        [-1.0, 1.0] * 4, device="cuda:0", dtype=log_prob.dtype
    )
    advantages = sequence_advantages[:, None] * response_mask
    actor_config = SimpleNamespace(
        policy_loss=SimpleNamespace(loss_mode=DEEPEYES_OFFICIAL_POLICY_LOSS_MODE),
        use_dynamic_bsz=False,
        ppo_epochs=1,
        entropy_coeff=0.0,
        use_kl_loss=False,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        ppo_micro_batch_size_per_gpu=8,
        global_batch_info={"dp_size": 1, "global_batch_size": 8},
    )
    policy_loss, _ = compute_deepeyes_official_micro_token_mean_loss(
        old_log_prob=log_prob.detach().clone(),
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=DEEPEYES_OFFICIAL_LOSS_AGG_MODE,
        config=actor_config,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)
    q_weight_before = model.layers[0].self_attn.q_proj.weight.detach().clone()
    policy_loss.backward()
    for values in inputs:
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) > 0
    q_gradient = model.layers[0].self_attn.q_proj.weight.grad
    assert q_gradient is not None
    assert torch.isfinite(q_gradient).all()
    assert torch.count_nonzero(q_gradient) > 0
    optimizer.step()
    q_weight_after = model.layers[0].self_attn.q_proj.weight.detach()
    assert not torch.equal(q_weight_after, q_weight_before)
    q_optimizer_state = optimizer.state[model.layers[0].self_attn.q_proj.weight]
    assert q_optimizer_state["step"].item() == 1
    assert torch.count_nonzero(q_optimizer_state["exp_avg"]) > 0
    assert torch.count_nonzero(q_optimizer_state["exp_avg_sq"]) > 0
