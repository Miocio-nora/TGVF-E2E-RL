from __future__ import annotations

from functools import partial

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
    sdpa = Qwen3VLTextModel(sdpa_config).to(
        device="cuda:0", dtype=torch.bfloat16
    )
    flex = Qwen3VLTextModel(flex_config).to(
        device="cuda:0", dtype=torch.bfloat16
    )
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
    torch.testing.assert_close(
        flex_input.grad, sdpa_input.grad, rtol=0.08, atol=0.08
    )
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
    model = Qwen3VLTextModel(config).to(
        device="cuda:0", dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    checkpoint_fn = model.layers[0]._gradient_checkpointing_func
    assert isinstance(checkpoint_fn, partial)
    assert checkpoint_fn.keywords["use_reentrant"] is False

    losses: list[torch.Tensor] = []
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
        losses.append(output.float().square().mean())

    torch.stack(losses).sum().backward()
    for values in inputs:
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) > 0
    q_gradient = model.layers[0].self_attn.q_proj.weight.grad
    assert q_gradient is not None
    assert torch.isfinite(q_gradient).all()
    assert torch.count_nonzero(q_gradient) > 0
