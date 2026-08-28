"""Native-pixel matched evaluation for the PRL25-F no-tool full model."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
import hashlib
import json
import threading

from PIL import Image

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.tokens import OwnedTokenSequence, TokenOwnership
from tgvf_rl.environment.agent_loop import (
    _extract_final_answer,
    _final_format_valid,
    _terminated_by_length,
)
from tgvf_rl.framework.verl.native_agent_loop import (
    _recover_termination,
    _selected_processed_logprobs,
    _token_ids,
    _validate_policy_step_evidence,
    _verl_server_sampling_parameters,
)
from tgvf_rl.framework.vllm import (
    ContentAddressedVLLMTurnRNG,
    FastTokenizerTokenByteSpanDecoder,
    VLLMPolicySampler,
    VLLMPolicyTurnRequest,
    VLLMPolicyTurnResponse,
    VLLMTokenLogprob,
)
from tgvf_rl.framework.vllm.registration import SUPPORTED_VLLM_VERSION
from tgvf_rl.framework.vllm.sampler import VLLM_PROCESSED_LOGPROBS_MODE
from tgvf_rl.observations.store import ObservationStore
from tgvf_rl.policy.no_tool_rl_protocol import build_no_tool_visual_messages
from tgvf_rl.policy.no_tool_rl_protocol import NO_TOOL_RL_PROMPT_IDENTITY
from tgvf_rl.policy.run_config import (
    POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMAS,
    PolicyE2ESmokeRunConfig,
)
from tgvf_rl.protocol import (
    NativeToolCapabilityProfile,
    native_assistant_dialect_for_model,
)
from tgvf_rl.environment import Qwen3NativeToolLayoutBuilder
from tgvf_rl.trajectories.behavior import BehaviorTraceStore, VLLMBehaviorRecorder
from tgvf_rl.trajectories.schema import (
    AssistantTurnRecord,
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)

from .policy_coredev import (
    TRAINING_RUN_EVALUATION_PROTOCOL,
    CoreDevTask,
    PolicyCoreDevConfig,
    StandaloneTGVFVLLMManager,
    _decoding_contract,
    _termination_contract,
    load_verified_task_image,
    evaluation_image_max_pixels,
    paired_evaluation_rng_for_task,
    policy_evaluation_identity,
)
from .policy_full_model_snapshot import FullModelEvaluationSnapshot
from .policy_official_visible import (
    _official_visible_mm_processor_kwargs,
    _official_visible_processor_geometry,
    _render_native_prompt,
    _rgb_tensor_to_pil,
)


_VLLM_CONTEXT_SAFETY_TOKENS = 16


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class _NativePixelRequestContext:
    """One single-use identity for a pre-rendered native-pixel prompt."""

    def __init__(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        image_sha256: str,
        image_max_pixels: int,
    ) -> None:
        self.prompt_token_ids = tuple(prompt_token_ids)
        self.identity_sha256 = _canonical_sha256(
            {
                "schema": "tgvf-no-tool-native-pixel-prompt-v1",
                "prompt_token_ids": self.prompt_token_ids,
                "image_sha256": image_sha256,
                "image_max_pixels": image_max_pixels,
            }
        )
        self._claimed = False

    def sha256_for_turn(
        self, prompt_token_ids: tuple[int, ...], *, turn_index: int
    ) -> str:
        if turn_index != 0 or tuple(prompt_token_ids) != self.prompt_token_ids:
            raise ReplayMismatchError("no-tool native-pixel prompt identity differs")
        if self._claimed:
            raise ReplayMismatchError("no-tool native-pixel prompt was already sampled")
        self._claimed = True
        return self.identity_sha256


class _NativePixelAsyncPolicyTurnClient:
    """Audited synchronous sampler bridge over one stock-vLLM native image."""

    backend_version = SUPPORTED_VLLM_VERSION
    logprobs_mode = VLLM_PROCESSED_LOGPROBS_MODE

    def __init__(
        self,
        *,
        manager: StandaloneTGVFVLLMManager,
        event_loop: asyncio.AbstractEventLoop,
        processor: object,
        tokenizer: object,
        image: Image.Image,
        image_max_pixels: int,
        sticky_request_id: str,
        max_model_len: int,
        timeout_seconds: float = 2400.0,
    ) -> None:
        if not event_loop.is_running():
            raise RuntimeError("no-tool native-pixel client requires a running loop")
        if not callable(getattr(tokenizer, "decode", None)):
            raise TypeError("no-tool native-pixel client requires tokenizer.decode")
        self.manager = manager
        self.event_loop = event_loop
        self.processor = processor
        self.tokenizer = tokenizer
        self.image = image
        self.image_max_pixels = image_max_pixels
        self.sticky_request_id = sticky_request_id
        self.max_model_len = max_model_len
        self.timeout_seconds = timeout_seconds
        self.decoder = FastTokenizerTokenByteSpanDecoder(tokenizer)
        self._event_loop_thread_id = threading.get_ident()

    def generate(self, request: VLLMPolicyTurnRequest) -> VLLMPolicyTurnResponse:
        if threading.get_ident() == self._event_loop_thread_id:
            raise RuntimeError("native-pixel sampling must run outside the owner loop")
        if not isinstance(request, VLLMPolicyTurnRequest):
            raise TypeError(
                "no-tool native-pixel client requires VLLMPolicyTurnRequest"
            )
        if (
            request.backend_version != self.backend_version
            or request.logprobs_mode != self.logprobs_mode
        ):
            raise IdentityMismatchError("no-tool native-pixel backend identity differs")
        parameters = _verl_server_sampling_parameters(
            request, max_model_len=self.max_model_len
        )
        awaitable = self.manager.generate(
            request_id=self.sticky_request_id,
            prompt_ids=list(request.prompt_token_ids),
            sampling_params=parameters,
            image_data=[self.image],
            video_data=None,
            audio_data=None,
            mm_processor_kwargs=_official_visible_mm_processor_kwargs(
                self.processor, self.image_max_pixels
            ),
            tgvf_expected_step=request.behavior_policy.optimizer_step,
        )
        future = asyncio.run_coroutine_threadsafe(awaitable, self.event_loop)
        try:
            output = future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError("no-tool native-pixel generation timed out") from error

        token_ids = _token_ids(getattr(output, "token_ids", None))
        logprobs = _selected_processed_logprobs(
            getattr(output, "log_probs", None), token_ids=token_ids
        )
        extra = getattr(output, "extra_fields", None)
        _validate_policy_step_evidence(extra, request=request)
        text = self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=request.decoding.skip_special_tokens,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=(
                request.decoding.spaces_between_special_tokens
            ),
        )
        if not isinstance(text, str):
            raise TypeError("no-tool tokenizer decode did not return text")
        spans = self.decoder.spans_for_output(
            text=text, token_ids=token_ids, decoding=request.decoding
        )
        finish_reason, stop_reason = _recover_termination(
            request=request,
            token_ids=token_ids,
            text=text,
            upstream_stop_reason=getattr(output, "stop_reason", None),
            exact_finish_reason=(
                extra.get("tgvf_vllm_finish_reason")
                if isinstance(extra, Mapping)
                else None
            ),
            exact_stop_reason=(
                extra.get("tgvf_vllm_stop_reason")
                if isinstance(extra, Mapping)
                else None
            ),
        )
        return VLLMPolicyTurnResponse(
            request_id=request.request_id,
            backend_request_sha256=request.backend_request_sha256,
            prompt_token_ids=request.prompt_token_ids,
            text=text,
            token_ids=token_ids,
            token_byte_spans=tuple(spans),
            token_logprobs=tuple(
                (VLLMTokenLogprob(token_id=token_id, logprob=logprob),)
                for token_id, logprob in zip(token_ids, logprobs, strict=True)
            ),
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )


class NoToolMatchedPolicyEvaluator:
    """One-turn, direct-only evaluator matching PRL25-F visual training rows."""

    def __init__(
        self,
        *,
        config: PolicyCoreDevConfig,
        run: PolicyE2ESmokeRunConfig,
        manager: StandaloneTGVFVLLMManager,
        processor: object,
        snapshot: FullModelEvaluationSnapshot,
        evaluation_identity: Mapping[str, object],
    ) -> None:
        if (
            config.evaluation_protocol != TRAINING_RUN_EVALUATION_PROTOCOL
            or run.schema_version
            not in POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMAS
            or run.protocol.tool_profile is not NativeToolCapabilityProfile.NO_TOOL
            or run.protocol.enabled_tool_names
        ):
            raise ValueError("no-tool evaluator received another protocol")
        if not isinstance(snapshot, FullModelEvaluationSnapshot) or snapshot.run != run:
            raise ValueError("no-tool evaluator requires its bound full-model snapshot")
        if not manager.native_pixels or manager.capture_hidden or manager.lora_request:
            raise ValueError(
                "no-tool evaluator requires stock full-model native pixels"
            )
        expected_identity = policy_evaluation_identity(config, snapshot)
        if dict(evaluation_identity) != expected_identity:
            raise ValueError("no-tool matched evaluation identity differs")
        protocol = expected_identity.get("protocol")
        if not isinstance(protocol, Mapping) or (
            protocol.get("profile") != TRAINING_RUN_EVALUATION_PROTOCOL
            or protocol.get("tool_profile") != NativeToolCapabilityProfile.NO_TOOL.value
            or protocol.get("enabled_tool_names") != []
            or protocol.get("native_pixels") is not True
        ):
            raise ValueError("no-tool matched protocol identity is malformed")
        self.config = config
        self.run = run
        self.manager = manager
        self.processor = processor
        self.snapshot = snapshot
        self.evaluation_identity = expected_identity
        self.policy_version = snapshot.policy_version
        self.image_max_pixels = evaluation_image_max_pixels(config, snapshot)
        self.assistant_dialect = native_assistant_dialect_for_model(
            run.model.model_name
        )
        self.tokenizer = getattr(processor, "tokenizer", None)
        self.layout_builder = Qwen3NativeToolLayoutBuilder.from_processor_config(
            processor=processor,
            model_identity=run.model,
            observation_store=ObservationStore(),
        )

    async def evaluate(self, task: CoreDevTask) -> TrajectoryRecord:
        if not task.single_image:
            raise ValueError("matched no-tool evaluation requires one image")
        source_rgb = load_verified_task_image(task)
        image = _rgb_tensor_to_pil(source_rgb)
        identity = TrajectoryIdentity(
            self.config.evaluation_id,
            task.bound_sample_id,
            0,
            (
                f"coredev:{task.ordinal}"
                if self.config.uses_legacy_coredev_manifest
                else f"benchmark:{task.ordinal}"
            ),
        )
        trajectory_id = identity.canonical_id
        behavior_store = BehaviorTraceStore()
        try:
            _text, prompt_ids, _visual_counts = _render_native_prompt(
                self.processor,
                build_no_tool_visual_messages(task.question, image="<image>"),
                images=[image],
                image_max_pixels=self.image_max_pixels,
            )
            image_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
            context = _NativePixelRequestContext(
                prompt_ids,
                image_sha256=image_sha256,
                image_max_pixels=self.image_max_pixels,
            )
            client = _NativePixelAsyncPolicyTurnClient(
                manager=self.manager,
                event_loop=asyncio.get_running_loop(),
                processor=self.processor,
                tokenizer=self.tokenizer,
                image=image,
                image_max_pixels=self.image_max_pixels,
                sticky_request_id=trajectory_id,
                max_model_len=self.config.max_model_len,
            )
            rng = (
                paired_evaluation_rng_for_task(
                    self.evaluation_identity,
                    sample_id=identity.sample_id,
                    rollout_index=identity.rollout_index,
                )
                if self.config.paired_seed_namespace is not None
                else ContentAddressedVLLMTurnRNG(
                    master_seed=self.run.rollout_rng.master_seed,
                    stream_identity=trajectory_id,
                )
            )
            sampler = VLLMPolicySampler(
                client=client,
                behavior_policy=self.policy_version,
                rng=rng,
                request_context=context,
                decoding=_decoding_contract(),
                termination=_termination_contract(self.run),
                assistant_dialect=self.assistant_dialect,
            )
            maximum = min(
                self.run.policy.sampling.max_response_length,
                self.config.max_model_len
                - len(prompt_ids)
                - _VLLM_CONTEXT_SAFETY_TOKENS,
            )
            if maximum <= 0:
                raise ValueError("matched no-tool prompt exhausted max_model_len")
            parameters = self.run.policy.sampling.as_vllm_parameters(max_tokens=maximum)
            sampled = await asyncio.to_thread(
                sampler.sample, prompt_ids, parameters, turn_index=0
            )
            self.run.policy.sampling.validate_sampling_identity(
                sampled.sampling, expected_max_tokens=maximum
            )
            tokens = OwnedTokenSequence(
                sampled.token_ids,
                tuple(TokenOwnership.POLICY_SAMPLED for _ in sampled.token_ids),
            )
            behavior_trace = VLLMBehaviorRecorder(behavior_store).record(
                trajectory_id=trajectory_id,
                assistant_turn_index=0,
                tokens=tokens,
                actual_sampled_logprobs=sampled.behavior_logprobs,
                sampling=sampled.sampling,
                behavior_policy=self.policy_version,
                backend_request_sha256=sampled.backend_request_sha256,
                backend_response_sha256=sampled.backend_response_sha256,
            )
            has_tool_marker = (
                "<tool_call>" in sampled.text or "</tool_call>" in sampled.text
            )
            final_answer = _extract_final_answer(
                sampled.text, assistant_dialect=self.assistant_dialect
            )
            if self.layout_builder.forbidden_policy_visual_token_ids and set(
                sampled.token_ids
            ).intersection(self.layout_builder.forbidden_policy_visual_token_ids):
                stop = TrajectoryStop.INVALID_FORMAT
                final_answer = None
            elif has_tool_marker:
                stop = TrajectoryStop.INVALID_FORMAT
                final_answer = None
            elif _terminated_by_length(sampled):
                stop = TrajectoryStop.MAX_TOKENS
                final_answer = None
            elif (
                not _final_format_valid(
                    sampled, assistant_dialect=self.assistant_dialect
                )
                or final_answer is None
            ):
                stop = TrajectoryStop.INVALID_FORMAT
                final_answer = None
            else:
                stop = TrajectoryStop.DIRECT_ANSWER
            return TrajectoryRecord(
                schema_version="trajectory-v1",
                identity=identity,
                model=self.run.model,
                behavior_policy=self.policy_version,
                assistant_turns=(
                    AssistantTurnRecord(
                        turn_index=0,
                        raw_text=sampled.text,
                        tokens=tokens,
                        behavior_trace=behavior_trace,
                        think_span=sampled.think_token_span,
                        is_tool_call=has_tool_marker,
                        stop_reason=sampled.stop_reason,
                    ),
                ),
                tool_calls=(),
                observations=(),
                final_answer=final_answer,
                stop=stop,
            )
        finally:
            image.close()
            try:
                await self.manager.release_trajectory(trajectory_id)
            finally:
                behavior_store.release_trajectories((trajectory_id,))


def validate_no_tool_matched_processor(
    processor: object,
    *,
    tokenizer_length: int,
    image_max_pixels: int,
) -> dict[str, object]:
    """CPU proof that both no-tool prompt expansion and decode obey the cap."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or len(tokenizer) != tokenizer_length:
        raise ValueError("no-tool matched tokenizer length differs")
    image = Image.new("RGB", (2048, 1536), (10, 20, 30))
    source_pixel_area = image.width * image.height
    try:
        _text, prompt_ids, visual_counts = _render_native_prompt(
            processor,
            build_no_tool_visual_messages("STATIC_PROTOCOL_PROBE", image="<image>"),
            images=[image],
            image_max_pixels=image_max_pixels,
        )
    finally:
        image.close()
    if len(visual_counts) != 1 or source_pixel_area <= image_max_pixels:
        raise ValueError("no-tool matched pixel probe is malformed")
    processor_size, patch_size, merge_size = _official_visible_processor_geometry(
        processor
    )
    represented_pixel_area = visual_counts[0] * merge_size**2 * patch_size**2
    if represented_pixel_area > image_max_pixels:
        raise ValueError("no-tool matched represented pixels exceed the cap")
    runtime_mm_processor_kwargs = _official_visible_mm_processor_kwargs(
        processor, image_max_pixels
    )
    runtime_size = runtime_mm_processor_kwargs.get("size")
    if (
        set(runtime_mm_processor_kwargs) != {"size"}
        or not isinstance(runtime_size, Mapping)
        or runtime_size
        != {
            "shortest_edge": processor_size["shortest_edge"],
            "longest_edge": image_max_pixels,
        }
    ):
        raise ValueError("no-tool matched vLLM processor override differs")
    # vLLM 0.12 shallow-wraps this one top-level dict before lru_cache use.
    # Prove its contents are hashable; nesting under images_kwargs would leave
    # a plain dict as one of the outer values and fail on the first request.
    hash(frozenset(runtime_size.items()))
    return {
        "schema_version": "tgvf.no-tool-matched-processor-static-proof.v1",
        "prompt_bundle_sha256": NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256,
        "prompt_token_ids_sha256": _canonical_sha256(prompt_ids),
        "prompt_token_count": len(prompt_ids),
        "configured_image_max_pixels": image_max_pixels,
        "processor_image_size": dict(processor_size),
        "effective_processor_image_size": {
            "shortest_edge": processor_size["shortest_edge"],
            "longest_edge": image_max_pixels,
        },
        "processor_patch_size": patch_size,
        "processor_merge_size": merge_size,
        "synthetic_native_source_pixel_area": source_pixel_area,
        "synthetic_native_represented_pixel_area": represented_pixel_area,
        "synthetic_native_visual_token_count": visual_counts[0],
        "runtime_mm_processor_kwargs": runtime_mm_processor_kwargs,
        "runtime_override_path": "mm_processor_kwargs.size.longest_edge",
        "vllm_012_shallow_hashable": True,
        "nested_images_kwargs_present": False,
        "max_pixels_kwarg_present": False,
        "tool_schema_visible": False,
        "system_prompt_present": False,
    }


__all__ = ["NoToolMatchedPolicyEvaluator", "validate_no_tool_matched_processor"]
