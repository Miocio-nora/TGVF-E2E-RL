"""Audited vLLM Qwen3-VL processor/model extension for recorded TGVF latents.

This module is imported only by the live registration function.  The project
package therefore remains importable when vLLM is not installed.  All vLLM
extension points used here are normal model/processor subclass and registry
interfaces; no installed vLLM file is modified.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tgvf_rl.qwen.deepstack_control import (
    apply_native_deepstack_tensor_control,
    native_deepstack_enabled_from_config,
)

from .preexpanded_prompt import split_preexpanded_prompt_contract


VLLM_IMPORT_ERROR: BaseException | None = None

try:
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3VLDummyInputsBuilder,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMultiModalProcessor,
        Qwen3VLProcessingInfo as _Qwen3VLProcessingInfo,
    )
    from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalInputs
    from vllm.multimodal.parse import (
        DictEmbeddingItems,
        ImageItem,
        ModalityData,
        ModalityDataItems,
        MultiModalDataParser,
    )
except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - env-specific
    VLLM_IMPORT_ERROR = error


def validate_precomputed_qwen3_image_dict(
    data: Mapping[str, torch.Tensor],
    *,
    spatial_merge_size: int,
    deepstack_levels: int = 3,
) -> tuple[int, ...]:
    """Validate merged embedding rows against exact Qwen image grids.

    Returns the per-item feature sizes used by vLLM's flat field splitter.
    The embedding width is ``H * (1 + deepstack_levels)``; this function does
    not guess ``H`` or a missing grid.
    """

    required = {"image_embeds", "image_grid_thw"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"precomputed Qwen3 image data is missing {sorted(missing)}")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")
    if deepstack_levels != 3:
        raise ValueError("TGVF Qwen3 transport requires exactly three DeepStack levels")

    embeds = data["image_embeds"]
    grid = data["image_grid_thw"]
    if not isinstance(embeds, torch.Tensor) or embeds.ndim != 2:
        raise ValueError("image_embeds must have flat shape [sum(N),H*(1+3)]")
    if not embeds.is_floating_point() or embeds.shape[0] <= 0:
        raise ValueError("image_embeds must be a non-empty floating tensor")
    width_factor = 1 + deepstack_levels
    if embeds.shape[1] <= 0 or embeds.shape[1] % width_factor:
        raise ValueError("image_embeds feature width must be divisible by main+3")
    if not isinstance(grid, torch.Tensor) or grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [num_items,3]")
    if grid.dtype == torch.bool or grid.is_floating_point():
        raise TypeError("image_grid_thw must use an integer dtype")
    if grid.shape[0] <= 0 or bool((grid <= 0).any().item()):
        raise ValueError("image_grid_thw must contain positive values")
    if bool((grid[:, 0] != 1).any().item()):
        raise ValueError("precomputed image embeddings require temporal grid size 1")
    if bool((grid[:, 1:] % spatial_merge_size != 0).any().item()):
        raise ValueError("image grids must be divisible by spatial_merge_size")

    sizes_tensor = grid.prod(-1) // (spatial_merge_size**2)
    sizes = tuple(int(value) for value in sizes_tensor.tolist())
    if sum(sizes) != embeds.shape[0]:
        raise ValueError(
            "image_embeds rows do not match merged image_grid_thw feature sizes"
        )
    return sizes


if VLLM_IMPORT_ERROR is None:
    # Preserve the previously exported upstream identity for import
    # compatibility; live TGVF registration uses the subclass below.
    Qwen3VLProcessingInfo = _Qwen3VLProcessingInfo

    def _merged_field_config(
        data: Mapping[str, torch.Tensor], spatial_merge_size: int
    ) -> Mapping[str, Any]:
        image_grid_thw = data.get("image_grid_thw", torch.empty((0, 3)))
        pixel_sizes = image_grid_thw.prod(-1)
        embed_sizes = pixel_sizes // (spatial_merge_size**2)
        return {
            "image_embeds": MultiModalFieldConfig.flat_from_sizes("image", embed_sizes),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }

    class TGVFQwen3VLDataParser(MultiModalDataParser):
        """Qwen3 parser that keeps precomputed embeddings and grids together."""

        def __init__(self, spatial_merge_size: int, *args: Any, **kwargs: Any) -> None:
            self.spatial_merge_size = int(spatial_merge_size)
            if self.spatial_merge_size <= 0:
                raise ValueError("spatial_merge_size must be positive")
            super().__init__(*args, **kwargs)

        def _parse_image_data(
            self,
            data: Mapping[str, torch.Tensor] | ModalityData[ImageItem],
        ) -> ModalityDataItems[Any, Any] | None:
            if (
                isinstance(data, list)
                and data
                and all(isinstance(item, Mapping) for item in data)
            ):
                item_dicts = list(data)
                for item in item_dicts:
                    validate_precomputed_qwen3_image_dict(
                        item,
                        spatial_merge_size=self.spatial_merge_size,
                    )
                data = {
                    "image_embeds": torch.cat(
                        [item["image_embeds"] for item in item_dicts], dim=0
                    ),
                    "image_grid_thw": torch.cat(
                        [item["image_grid_thw"] for item in item_dicts], dim=0
                    ),
                }
            if isinstance(data, Mapping):
                validate_precomputed_qwen3_image_dict(
                    data,
                    spatial_merge_size=self.spatial_merge_size,
                )
                return DictEmbeddingItems(
                    data,
                    modality="image",
                    required_fields={"image_embeds", "image_grid_thw"},
                    fields_factory=lambda fields: _merged_field_config(
                        fields, self.spatial_merge_size
                    ),
                )
            return super()._parse_image_data(data)

    class TGVFQwen3VLProcessingInfo(_Qwen3VLProcessingInfo):
        """Select the TGVF latent parser on every audited vLLM generation.

        vLLM 0.12 asks the multimodal processor for its parser, while vLLM
        0.23 asks the registered processing-info object. Defining this hook and
        retaining the processor hook below keeps the transport explicit across
        both public lifecycles.
        """

        def get_data_parser(self) -> TGVFQwen3VLDataParser:
            parser_kwargs: dict[str, Any] = {"video_needs_metadata": True}
            expected_hidden_size = getattr(self, "_get_expected_hidden_size", None)
            if callable(expected_hidden_size):
                parser_kwargs["expected_hidden_size"] = expected_hidden_size()
            merge_size = int(self.get_hf_config().vision_config.spatial_merge_size)
            return TGVFQwen3VLDataParser(merge_size, **parser_kwargs)

    class TGVFQwen3VLMultiModalProcessor(Qwen3VLMultiModalProcessor):
        """Qwen3 processor with one fail-closed expanded-token coordinate.

        Production rollout submits token IDs whose image-pad runs already have
        the merged visual feature length.  Calling the stock token-prompt path
        would replace the first token of each run and can produce ``2N-1``
        tokens.  The reserved per-turn contract instead makes this path locate
        the already-expanded placeholders and prove that vLLM retained the
        submitted hash, length, ranges, and item order exactly.
        """

        def apply(
            self,
            prompt: str | list[int],
            mm_data: Mapping[str, Any],
            hf_processor_mm_kwargs: Mapping[str, object],
            tokenization_kwargs: Mapping[str, object] | None = None,
            *,
            mm_uuids: Mapping[str, list[str | None] | str] | None = None,
        ) -> MultiModalInputs:
            # Keep stock string processing for vLLM's dummy/profile lifecycle.
            # Every production policy turn uses token IDs and must carry the
            # contract; absence or mismatch fails before model execution.
            if isinstance(prompt, str):
                return super().apply(
                    prompt,
                    mm_data,
                    hf_processor_mm_kwargs,
                    tokenization_kwargs,
                    mm_uuids=mm_uuids,
                )

            contract, clean_mm_kwargs = split_preexpanded_prompt_contract(
                hf_processor_mm_kwargs
            )
            mm_items = self._to_mm_items(mm_data)
            item_counts = mm_items.get_all_counts()
            if set(item_counts) != {"image"}:
                raise ValueError(
                    "TGVF pre-expanded processor requires exactly image items"
                )
            contract.validate_submitted_prompt(
                prompt,
                expected_image_items=item_counts["image"],
            )
            actual_image_token_id = self.info.get_hf_processor(
                **clean_mm_kwargs
            ).image_token_id
            if contract.image_token_id != actual_image_token_id:
                raise ValueError(
                    "pre-expanded contract image token differs from Qwen processor"
                )

            if tokenization_kwargs is None:
                tokenization_kwargs = {}
            prompt_ids, mm_info, is_update_applied = self._cached_apply_hf_processor(
                prompt,
                mm_items,
                clean_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
            if is_update_applied:
                raise RuntimeError(
                    "token-ID pre-expanded prompt unexpectedly reports an HF update"
                )

            # Intentionally do not call _maybe_apply_prompt_updates().  Locate
            # each complete N-token replacement already present in the input.
            mm_placeholders = self._find_mm_placeholders(
                prompt_ids,
                mm_info.prompt_updates,
            )
            self._validate_mm_placeholders(mm_placeholders, item_counts)
            mm_placeholder_ranges = {
                modality: [item.to_range() for item in placeholders]
                for modality, placeholders in mm_placeholders.items()
            }
            contract.validate_processed_prompt(prompt_ids, mm_placeholder_ranges)
            return MultiModalInputs(
                type="multimodal",
                prompt_token_ids=prompt_ids,
                mm_kwargs=mm_info.kwargs,
                mm_hashes=mm_info.hashes,
                mm_placeholders=mm_placeholder_ranges,
            )

        def _get_data_parser(self) -> TGVFQwen3VLDataParser:
            merge_size = int(self.info.get_hf_config().vision_config.spatial_merge_size)
            return TGVFQwen3VLDataParser(
                merge_size,
                video_needs_metadata=True,
            )

        def _get_mm_fields_config(
            self,
            hf_inputs: Any,
            hf_processor_mm_kwargs: Mapping[str, object],
        ) -> Mapping[str, Any]:
            fields = dict(
                super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
            )
            image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
            merge_size = int(self.info.get_hf_config().vision_config.spatial_merge_size)
            merged_sizes = image_grid_thw.prod(-1) // (merge_size**2)
            # vLLM 0.12's stock Qwen3 processor incorrectly uses pre-merge
            # grid sizes for image_embeds.  The TGVF packer transports outputs
            # of the frozen merger, so its rows are the merged token count.
            fields["image_embeds"] = MultiModalFieldConfig.flat_from_sizes(
                "image", merged_sizes
            )
            return fields

    class TGVFQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
        """Qwen3-vLLM model accepting main+DeepStack precomputed embeddings."""

        def __init__(self, *, vllm_config: Any, prefix: str = "model") -> None:
            # Preserve vLLM's public new-style model signature. A variadic
            # wrapper is classified as an old-style model and receives legacy
            # constructor arguments before this class can validate anything.
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self._tgvf_native_deepstack_enabled = native_deepstack_enabled_from_config(
                self.config
            )
            if not self.use_deepstack or self.deepstack_num_level != 3:
                raise ValueError(
                    "TGVF Qwen3 vLLM plugin requires exactly three DeepStack levels"
                )

        def _compute_deepstack_embeds(
            self,
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: Any,
            is_multimodal: torch.Tensor,
        ) -> tuple[torch.Tensor, Any]:
            deepstack, main = super()._compute_deepstack_embeds(
                inputs_embeds,
                multimodal_embeddings,
                is_multimodal,
            )
            return (
                apply_native_deepstack_tensor_control(
                    deepstack,
                    enabled=self._tgvf_native_deepstack_enabled,
                ),
                main,
            )

        def _process_image_input(self, image_input: Any) -> tuple[torch.Tensor, ...]:
            if image_input["type"] != "image_embeds":
                return super()._process_image_input(image_input)

            image_embeds = image_input["image_embeds"]
            image_grid_thw = image_input["image_grid_thw"]
            merge_size = int(self.config.vision_config.spatial_merge_size)
            sizes = validate_precomputed_qwen3_image_dict(
                {
                    "image_embeds": image_embeds,
                    "image_grid_thw": image_grid_thw,
                },
                spatial_merge_size=merge_size,
                deepstack_levels=self.deepstack_num_level,
            )
            expected_width = self.visual_dim * (1 + self.deepstack_num_level)
            if image_embeds.shape[-1] != expected_width:
                raise ValueError(
                    "precomputed Qwen3 image width differs from main+DeepStack model width"
                )
            visual_dtype = getattr(self.visual, "dtype", None)
            if visual_dtype is None:
                parameter = next(self.language_model.parameters(), None)
                if parameter is None:
                    raise TypeError("Qwen3 model has no dtype-bearing parameters")
                visual_dtype = parameter.dtype
            return image_embeds.to(dtype=visual_dtype).split(sizes)


else:

    class _VLLMRequired:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ModuleNotFoundError(
                "an audited vLLM build is required to construct the TGVF Qwen3 plugin"
            ) from VLLM_IMPORT_ERROR

    TGVFQwen3VLDataParser = _VLLMRequired
    TGVFQwen3VLProcessingInfo = _VLLMRequired
    TGVFQwen3VLMultiModalProcessor = _VLLMRequired
    TGVFQwen3VLForConditionalGeneration = _VLLMRequired
    Qwen3VLProcessingInfo = _VLLMRequired
    Qwen3VLDummyInputsBuilder = _VLLMRequired


__all__ = [
    "Qwen3VLDummyInputsBuilder",
    "Qwen3VLProcessingInfo",
    "TGVFQwen3VLDataParser",
    "TGVFQwen3VLForConditionalGeneration",
    "TGVFQwen3VLMultiModalProcessor",
    "TGVFQwen3VLProcessingInfo",
    "VLLM_IMPORT_ERROR",
    "validate_precomputed_qwen3_image_dict",
]
