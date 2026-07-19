"""Optional exact-latent vLLM bridge; importing it does not require vLLM."""

from .packer import (
    QWEN3_DEEPSTACK_BRANCH_COUNT,
    QWEN3_DEEPSTACK_BRANCH_LAYERS,
    PackedQwen3ImageItem,
    PackedQwen3Replay,
    pack_qwen3_vllm_replay,
)
from .registration import (
    SUPPORTED_VLLM_VERSION,
    SUPPORTED_VLLM_VERSIONS,
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    TGVF_VLLM_ATTENTION_BACKEND,
    TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
    VLLMCompatibilityError,
    VLLMPluginError,
    VLLMPluginRegistration,
    VLLMPublicPluginAPI,
    VLLMUnavailableError,
    load_vllm_public_plugin_api,
    register_tgvf_qwen3_vllm_plugin,
    vllm_is_available,
)

__all__ = [
    "QWEN3_DEEPSTACK_BRANCH_COUNT",
    "QWEN3_DEEPSTACK_BRANCH_LAYERS",
    "SUPPORTED_VLLM_VERSION",
    "SUPPORTED_VLLM_VERSIONS",
    "TGVF_QWEN3_VLLM_ARCHITECTURE",
    "TGVF_VLLM_ATTENTION_BACKEND",
    "TGVF_VLLM_MM_ENCODER_ATTN_BACKEND",
    "PackedQwen3ImageItem",
    "PackedQwen3Replay",
    "VLLMCompatibilityError",
    "VLLMPluginError",
    "VLLMPluginRegistration",
    "VLLMPublicPluginAPI",
    "VLLMUnavailableError",
    "load_vllm_public_plugin_api",
    "pack_qwen3_vllm_replay",
    "register_tgvf_qwen3_vllm_plugin",
    "vllm_is_available",
]
