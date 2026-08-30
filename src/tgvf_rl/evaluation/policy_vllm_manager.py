"""Backend-neutral standalone vLLM manager used by policy evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch

from tgvf_rl.public_api_compat import (
    rebind_public_class,
    rebind_public_function,
)
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVFFocusMaterializationResult,
    _focus_from_utility_wire,
    _source_from_utility_wire,
    _tensor_to_utility_wire,
)


class AdapterIntegrityVerifier(ABC):
    """Nominal boundary required when the manager receives a LoRA request."""

    @abstractmethod
    def verify(self, *, phase: str) -> None:
        """Verify the adapter closure at one named consumption boundary."""

    @abstractmethod
    def assert_lora_request_binding(self, lora_request: object) -> None:
        """Verify that a request names this verifier's adapter closure."""


def _single_collective(value: object, *, operation: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 1
    ):
        raise RuntimeError(f"{operation} requires one vLLM worker result")
    result = value[0]
    if not isinstance(result, Mapping):
        raise TypeError(f"{operation} returned a non-mapping utility result")
    return result


@dataclass(frozen=True, slots=True)
class _TurnRoute:
    backend_request_id: str
    output_ids: tuple[int, ...]
    optimizer_step: int


class StandaloneTGVFVLLMManager:
    """Small AsyncLLM adapter matching the already-audited training client ABI."""

    def __init__(
        self,
        engine: object,
        lora_request: object,
        *,
        capture_hidden: bool,
        native_pixels: bool = False,
        adapter_integrity_verifier: AdapterIntegrityVerifier | None = None,
    ) -> None:
        if lora_request is None:
            if adapter_integrity_verifier is not None:
                raise ValueError(
                    "full-model manager cannot receive a LoRA integrity verifier"
                )
        else:
            if not isinstance(adapter_integrity_verifier, AdapterIntegrityVerifier):
                raise TypeError(
                    "LoRA manager requires VLLMLoRAAdapterIntegrityVerifier"
                )
            adapter_integrity_verifier.assert_lora_request_binding(lora_request)
            adapter_integrity_verifier.verify(phase="manager construction")
        self.engine = engine
        self.lora_request = lora_request
        self.adapter_integrity_verifier = adapter_integrity_verifier
        self.capture_hidden = capture_hidden
        self.native_pixels = native_pixels
        self.turns: dict[str, _TurnRoute] = {}
        self.backend_ids: dict[str, list[str]] = {}

    async def materialize_source(
        self,
        *,
        request_id: str,
        expected_step: int,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_sha256: str,
    ) -> object:
        del expected_step
        result = await self.engine.collective_rpc(
            "tgvf_materialize_source",
            kwargs={
                "trajectory_id": request_id,
                "pixel_values_wire": _tensor_to_utility_wire(pixel_values),
                "image_grid_thw": tuple(int(v) for v in image_grid_thw[0].tolist()),
                "image_sha256": image_sha256,
            },
        )
        return _source_from_utility_wire(
            _single_collective(result, operation="source materialization")
        )

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        audio_data: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> object:
        del video_data, audio_data
        from vllm import SamplingParams

        step = int(kwargs.pop("tgvf_expected_step"))
        if kwargs:
            raise TypeError(f"unsupported standalone vLLM arguments: {sorted(kwargs)}")
        if self.adapter_integrity_verifier is not None:
            self.adapter_integrity_verifier.assert_lora_request_binding(
                self.lora_request
            )
            self.adapter_integrity_verifier.verify(phase="before engine.generate")
        backend_id = f"eval-{uuid4().hex}"
        maximum = sampling_params.get("max_tokens")
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("generation requires positive max_tokens")
        if self.capture_hidden:
            await self.engine.collective_rpc(
                "tgvf_register_behavior_trace",
                kwargs={
                    "request_id": backend_id,
                    "prompt_length": len(prompt_ids),
                    "maximum_output_tokens": maximum,
                },
            )
            self.backend_ids.setdefault(request_id, []).append(backend_id)
        prompt = {
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": {"image": image_data},
            "mm_processor_kwargs": mm_processor_kwargs,
        }
        parameters = dict(sampling_params)
        if parameters.get("logprobs") is True:
            parameters["logprobs"] = 0
        final = None
        adapter_arguments = (
            {} if self.lora_request is None else {"lora_request": self.lora_request}
        )
        async for output in self.engine.generate(
            prompt,
            SamplingParams(**parameters),
            backend_id,
            **adapter_arguments,
        ):
            final = output
        if self.adapter_integrity_verifier is not None:
            self.adapter_integrity_verifier.verify(phase="after engine.generate")
        if final is None or not final.finished or len(final.outputs) != 1:
            raise RuntimeError("standalone vLLM generation did not finish exactly once")
        completion = final.outputs[0]
        token_ids = tuple(int(value) for value in completion.token_ids)
        logprobs = []
        for token_id, position in zip(token_ids, completion.logprobs, strict=True):
            entry = position.get(token_id)
            if entry is None:
                raise RuntimeError("sampled token is absent from vLLM logprobs")
            logprobs.append(float(entry.logprob))
        self.turns[request_id] = _TurnRoute(backend_id, token_ids, step)
        return SimpleNamespace(
            token_ids=list(token_ids),
            log_probs=logprobs,
            stop_reason="completed",
            extra_fields={
                "global_steps": step,
                "min_global_steps": step,
                "max_global_steps": step,
                "logprobs_mode": "processed_logprobs",
                "tgvf_vllm_finish_reason": completion.finish_reason,
                "tgvf_vllm_stop_reason": completion.stop_reason,
            },
        )

    async def materialize_focus(
        self,
        *,
        request_id: str,
        expected_step: int,
        sampled_output_ids: tuple[int, ...],
        target_start: int,
        target_end: int,
        expected_target_token_ids: tuple[int, ...],
        provider: str,
    ) -> tuple[torch.Tensor, object]:
        turn = self._validated_turn(request_id, expected_step, sampled_output_ids)
        if turn.output_ids[target_start:target_end] != expected_target_token_ids:
            raise RuntimeError("focus target differs from sampled output")
        result = await self.engine.collective_rpc(
            "tgvf_materialize_focus",
            kwargs={
                "trajectory_id": request_id,
                "backend_request_id": turn.backend_request_id,
                "target_start": target_start,
                "target_end": target_end,
                "expected_target_token_ids": expected_target_token_ids,
                "provider": provider,
            },
        )
        typed = _focus_from_utility_wire(
            _single_collective(result, operation="focus materialization")
        )
        if not isinstance(typed, TGVFFocusMaterializationResult):
            raise TypeError("focus RPC returned an invalid result")
        return typed.hq, typed.observation

    async def materialize_crop(
        self,
        *,
        request_id: str,
        expected_step: int,
        sampled_output_ids: tuple[int, ...],
        call_index: int,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        crop_sha256: str,
    ) -> object:
        self._validated_turn(request_id, expected_step, sampled_output_ids)
        result = await self.engine.collective_rpc(
            "tgvf_materialize_crop",
            kwargs={
                "trajectory_id": request_id,
                "call_index": call_index,
                "pixel_values_wire": _tensor_to_utility_wire(pixel_values),
                "image_grid_thw": tuple(int(v) for v in image_grid_thw[0].tolist()),
                "crop_sha256": crop_sha256,
            },
        )
        return _source_from_utility_wire(
            _single_collective(result, operation="crop materialization")
        )

    def _validated_turn(
        self, request_id: str, expected_step: int, output_ids: tuple[int, ...]
    ) -> _TurnRoute:
        turn = self.turns.get(request_id)
        if turn is None or turn.output_ids != tuple(output_ids):
            raise RuntimeError("tool call differs from the last vLLM turn")
        if turn.optimizer_step != expected_step:
            raise RuntimeError("tool call policy step changed")
        return turn

    async def release_trajectory(self, request_id: str) -> None:
        backend_ids = tuple(self.backend_ids.pop(request_id, ()))
        self.turns.pop(request_id, None)
        if self.native_pixels:
            if backend_ids:
                raise RuntimeError(
                    "native-pixel evaluator unexpectedly registered hidden traces"
                )
            return
        await self.engine.collective_rpc(
            "tgvf_release_trajectory", args=(request_id, backend_ids)
        )


# These objects were historically defined in policy_coredev.  Keep their
# import/pickle coordinates stable while making their implementation reusable.
_LEGACY_PUBLIC_MODULE = "tgvf_rl.evaluation.policy_coredev"
rebind_public_function(
    _single_collective,
    implementation_module=__name__,
    public_module=_LEGACY_PUBLIC_MODULE,
)
for _legacy_class in (_TurnRoute, StandaloneTGVFVLLMManager):
    rebind_public_class(
        _legacy_class,
        implementation_module=__name__,
        public_module=_LEGACY_PUBLIC_MODULE,
    )
del _legacy_class


__all__ = ["AdapterIntegrityVerifier", "StandaloneTGVFVLLMManager"]
