from __future__ import annotations

from tgvf_rl.framework.verl.qwen3_flex_attention_compat import (
    QWEN3_VL_TEXT_FLEX_COMPAT_SCHEMA,
    install_qwen3_vl_text_flex_attention_compat,
)


def test_qwen3_vl_text_flex_compat_is_narrow_and_idempotent() -> None:
    from transformers.integrations import flex_attention as flex_runtime
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLTextModel,
        Qwen3VLVisionModel,
    )

    conditional_before = Qwen3VLForConditionalGeneration._supports_flex_attn
    vision_before = Qwen3VLVisionModel._supports_flex_attn

    assert (
        install_qwen3_vl_text_flex_attention_compat()
        == QWEN3_VL_TEXT_FLEX_COMPAT_SCHEMA
    )
    assert (
        install_qwen3_vl_text_flex_attention_compat()
        == QWEN3_VL_TEXT_FLEX_COMPAT_SCHEMA
    )
    assert Qwen3VLTextModel._supports_flex_attn is True
    assert Qwen3VLForConditionalGeneration._supports_flex_attn is conditional_before
    assert Qwen3VLVisionModel._supports_flex_attn is vision_before
    assert flex_runtime.WrappedFlexAttention._tgvf_dynamic_training_compile is True


def test_qwen3_vl_config_routes_flex_only_to_text_subconfig() -> None:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from verl.utils.model import update_model_config

    config = Qwen3VLConfig()
    update_model_config(
        config,
        {
            "text_config": {
                "_attn_implementation_internal": "flex_attention"
            },
            "vision_config": {"_attn_implementation_internal": "sdpa"},
        },
    )

    assert config._attn_implementation is None
    assert config.text_config._attn_implementation == "flex_attention"
    assert config.vision_config._attn_implementation == "sdpa"
