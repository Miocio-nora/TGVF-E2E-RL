"""Fail-closed public vLLM plugin registration for audited runtime builds."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Any, Callable


# Keep the original singular name as the accepted-control identity for callers
# that already import it. Live registration uses the exact set below and does
# not accept an un-audited patch, local-build suffix, or neighboring release.
SUPPORTED_VLLM_VERSION = "0.12.0"
SUPPORTED_VLLM_VERSIONS = frozenset(
    {
        SUPPORTED_VLLM_VERSION,
        "0.23.0+cu129",
    }
)
TGVF_QWEN3_VLLM_ARCHITECTURE = "TGVFQwen3VLForConditionalGeneration"
TGVF_VLLM_ATTENTION_BACKEND = "TRITON_ATTN"
TGVF_VLLM_MM_ENCODER_ATTN_BACKEND = "TORCH_SDPA"
VLLM_012_TRITON_VERSION = "3.5.0"
VLLM_012_LORA_PDL_MODE = "vllm-0.12-triton-3.5-lora-pdl-disabled-v1"
VLLM_UPSTREAM_LORA_PDL_MODE = "upstream"
_VLLM_LORA_PDL_MODULES = (
    "vllm.lora.ops.triton_ops.utils",
    "vllm.lora.ops.triton_ops.lora_expand_op",
    "vllm.lora.ops.triton_ops.lora_shrink_op",
    "vllm.lora.ops.triton_ops.fused_moe_lora_op",
)


class VLLMPluginError(RuntimeError):
    """Base error for the optional vLLM plugin."""


class VLLMUnavailableError(VLLMPluginError):
    """Raised when a live plugin operation needs an unavailable vLLM."""


class VLLMCompatibilityError(VLLMPluginError):
    """Raised when the installed public API differs from the spike candidate."""


@dataclass(frozen=True, slots=True)
class VLLMPublicPluginAPI:
    model_registry: Any
    multimodal_registry: Any
    model_cls: type[Any]
    processor_cls: type[Any]
    processing_info_cls: type[Any]
    dummy_inputs_cls: type[Any]
    version: str


@dataclass(frozen=True, slots=True)
class VLLMPluginRegistration:
    architecture: str
    model_cls: type[Any]
    processor_cls: type[Any]
    version: str
    lora_pdl_mode: str


def _lora_pdl_disabled(_device: object | None = None) -> bool:
    return False


def install_vllm_lora_pdl_compatibility(
    *,
    vllm_version: str,
    triton_version: str,
    importer: Callable[[str], Any] = import_module,
) -> str:
    """Disable optional Punica PDL on the one audited failing stack.

    vLLM 0.12 enables PDL/GDC on every SM90+ device.  Triton 3.5 cannot
    predicate the resulting inline assembly while compiling the LoRA expand
    kernel for SM100.  PDL is a launch optimization, so replacing only the
    imported capability aliases leaves the Punica LoRA computation unchanged.
    """

    if vllm_version != SUPPORTED_VLLM_VERSION:
        return VLLM_UPSTREAM_LORA_PDL_MODE
    if triton_version != VLLM_012_TRITON_VERSION:
        raise VLLMCompatibilityError(
            "vLLM 0.12 LoRA PDL compatibility requires exact Triton 3.5.0"
        )
    for module_name in _VLLM_LORA_PDL_MODULES:
        module = importer(module_name)
        if not callable(getattr(module, "supports_pdl", None)):
            raise VLLMCompatibilityError(
                f"vLLM LoRA PDL surface is missing in {module_name}"
            )
        module.supports_pdl = _lora_pdl_disabled
        if module.supports_pdl(None) is not False:
            raise VLLMCompatibilityError(
                f"vLLM LoRA PDL patch did not bind {module_name}"
            )
    return VLLM_012_LORA_PDL_MODE


def vllm_is_available() -> bool:
    try:
        metadata.version("vllm")
    except metadata.PackageNotFoundError:
        return False
    return True


def load_vllm_public_plugin_api(
    importer: Callable[[str], Any] = import_module,
) -> VLLMPublicPluginAPI:
    """Load only the general model and multimodal registration surfaces."""

    try:
        root = importer("vllm")
        multimodal = importer("vllm.multimodal")
        plugin = importer("tgvf_rl.framework.vllm.qwen3_plugin")
    except (ImportError, ModuleNotFoundError) as error:
        raise VLLMUnavailableError(
            "vLLM is optional; install an exact audited compatibility environment"
        ) from error

    if importer is import_module:
        try:
            version = metadata.version("vllm")
        except metadata.PackageNotFoundError as error:
            raise VLLMUnavailableError(
                "vLLM distribution metadata is absent"
            ) from error
    else:
        version = getattr(root, "__version__", "")
    if version not in SUPPORTED_VLLM_VERSIONS:
        accepted = ", ".join(sorted(SUPPORTED_VLLM_VERSIONS))
        raise VLLMCompatibilityError(
            f"vLLM must be one of the exact audited builds ({accepted}); "
            f"found {version or 'unknown'}"
        )

    import_error = getattr(plugin, "VLLM_IMPORT_ERROR", None)
    if import_error is not None:
        raise VLLMCompatibilityError(
            f"TGVF Qwen3 plugin dependencies failed to import: {import_error}"
        )
    try:
        model_registry = root.ModelRegistry
        multimodal_registry = multimodal.MULTIMODAL_REGISTRY
        model_cls = plugin.TGVFQwen3VLForConditionalGeneration
        processor_cls = plugin.TGVFQwen3VLMultiModalProcessor
        processing_info_cls = plugin.TGVFQwen3VLProcessingInfo
        dummy_inputs_cls = plugin.Qwen3VLDummyInputsBuilder
    except AttributeError as error:
        raise VLLMCompatibilityError(
            "the audited vLLM public model/processor registry surface is incomplete"
        ) from error
    if not callable(getattr(model_registry, "register_model", None)):
        raise VLLMCompatibilityError("ModelRegistry.register_model is unavailable")
    if not callable(getattr(multimodal_registry, "register_processor", None)):
        raise VLLMCompatibilityError(
            "MULTIMODAL_REGISTRY.register_processor is unavailable"
        )
    return VLLMPublicPluginAPI(
        model_registry=model_registry,
        multimodal_registry=multimodal_registry,
        model_cls=model_cls,
        processor_cls=processor_cls,
        processing_info_cls=processing_info_cls,
        dummy_inputs_cls=dummy_inputs_cls,
        version=version,
    )


def register_tgvf_qwen3_vllm_plugin(
    *,
    api: VLLMPublicPluginAPI | None = None,
) -> VLLMPluginRegistration:
    """Register the repo-owned class through vLLM's public registries.

    This deliberately performs no model construction, CUDA initialization, or
    site-package mutation.  Any missing or changed API aborts registration.
    """

    live_registration = api is None
    public = load_vllm_public_plugin_api() if live_registration else api
    if public.version not in SUPPORTED_VLLM_VERSIONS:
        raise VLLMCompatibilityError("injected vLLM API has an unsupported version")
    register_processor = getattr(public.multimodal_registry, "register_processor", None)
    register_model = getattr(public.model_registry, "register_model", None)
    if not callable(register_processor) or not callable(register_model):
        raise VLLMCompatibilityError("public vLLM registration methods are unavailable")

    if live_registration:
        try:
            triton_version = metadata.version("triton")
        except metadata.PackageNotFoundError as error:
            raise VLLMCompatibilityError(
                "the audited vLLM LoRA runtime requires Triton metadata"
            ) from error
        lora_pdl_mode = install_vllm_lora_pdl_compatibility(
            vllm_version=public.version,
            triton_version=triton_version,
        )
    else:
        # Injected APIs are registration fixtures and must not mutate an
        # unrelated installed vLLM process. The patch helper is tested
        # independently with an injected importer.
        lora_pdl_mode = VLLM_UPSTREAM_LORA_PDL_MODE

    decorator = register_processor(
        public.processor_cls,
        info=public.processing_info_cls,
        dummy_inputs=public.dummy_inputs_cls,
    )
    if not callable(decorator):
        raise VLLMCompatibilityError("register_processor did not return a decorator")
    registered_cls = decorator(public.model_cls)
    if registered_cls is not public.model_cls:
        raise VLLMCompatibilityError("register_processor replaced the model class")
    register_model(TGVF_QWEN3_VLLM_ARCHITECTURE, public.model_cls)
    return VLLMPluginRegistration(
        architecture=TGVF_QWEN3_VLLM_ARCHITECTURE,
        model_cls=public.model_cls,
        processor_cls=public.processor_cls,
        version=public.version,
        lora_pdl_mode=lora_pdl_mode,
    )


__all__ = [
    "SUPPORTED_VLLM_VERSION",
    "SUPPORTED_VLLM_VERSIONS",
    "TGVF_QWEN3_VLLM_ARCHITECTURE",
    "TGVF_VLLM_ATTENTION_BACKEND",
    "TGVF_VLLM_MM_ENCODER_ATTN_BACKEND",
    "VLLM_012_LORA_PDL_MODE",
    "VLLM_012_TRITON_VERSION",
    "VLLM_UPSTREAM_LORA_PDL_MODE",
    "VLLMCompatibilityError",
    "VLLMPluginError",
    "VLLMPluginRegistration",
    "VLLMPublicPluginAPI",
    "VLLMUnavailableError",
    "load_vllm_public_plugin_api",
    "install_vllm_lora_pdl_compatibility",
    "register_tgvf_qwen3_vllm_plugin",
    "vllm_is_available",
]
