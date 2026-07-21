"""Concrete live-vLLM resolver for exact recorded visual tool observations."""

from __future__ import annotations

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.observations.store import ObservationHandle, ObservationStore

from .packer import pack_qwen3_vllm_observation
from .turn_runtime import VLLMResolvedObservationPayload


class Qwen3VLLMObservationPayloadResolver:
    """Append crop, TGVF, or atomic crop+TGVF without recomputation."""

    def __init__(
        self,
        *,
        store: ObservationStore,
        include_multi_modal_uuid: bool,
    ) -> None:
        if not isinstance(store, ObservationStore):
            raise TypeError("Qwen3 observation resolver requires an ObservationStore")
        if not isinstance(include_multi_modal_uuid, bool):
            raise TypeError("include_multi_modal_uuid must be bool")
        self.store = store
        self.include_multi_modal_uuid = include_multi_modal_uuid

    def resolve(
        self,
        observation: ObservationHandle,
        *,
        call_index: int,
    ) -> VLLMResolvedObservationPayload:
        if not isinstance(observation, ObservationHandle):
            raise TypeError("observation must be an ObservationHandle")
        if type(call_index) is not int or call_index < 0:
            raise ValueError("call_index must be non-negative")
        record = self.store.resolve_record(observation)
        if record.call_index != call_index:
            raise IdentityMismatchError(
                "live vLLM call index differs from recorded observation"
            )
        item = pack_qwen3_vllm_observation(self.store, observation)
        if item.call_index != call_index:
            raise IdentityMismatchError("packed Qwen3 observation changed call identity")
        item.verify_integrity()
        payload = {
            "image_embeds": item.image_embeds.clone(),
            "image_grid_thw": torch.tensor(
                (item.image_grid_thw,), dtype=torch.long
            ),
        }
        return VLLMResolvedObservationPayload(
            observation=observation,
            call_index=call_index,
            modality="image",
            multi_modal_data_item=payload,
            payload_sha256=item.item_sha256,
            multi_modal_uuid=(
                item.item_sha256 if self.include_multi_modal_uuid else None
            ),
        )


__all__ = ["Qwen3VLLMObservationPayloadResolver"]
