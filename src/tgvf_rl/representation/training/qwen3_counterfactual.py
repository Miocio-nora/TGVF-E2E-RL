"""Qwen3-native fresh-D counterfactual materialization.

The processor sees one geometry-carrier image solely to expand Qwen's native
tool-response vision placeholder.  Pixel values are discarded.  Every model
forward receives only the already-materialized main ``D`` and all ordered
D-DeepStack branches, with no source-image block and no reused KV cache.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import struct
from typing import Any

import torch

from tgvf_rl.protocol.native import NativeProtocolRenderer, RenderedTranscript
from tgvf_rl.qwen.base import (
    InjectedForwardRequest,
    InjectedVisualBlock,
    resolve_language_model,
)

from .internal_evaluation import (
    NativeCounterfactualCase,
    NativeDOnlyContext,
    NativeGenerationForward,
    NativeInjectedRequestMaterializer,
    NativeTeacherForcedForward,
)
from .native_pipeline import (
    RepresentationPromptConfig,
    _processor_batch,
    _qwen3_position_ids,
    build_native_representation_messages,
)
from .readout import (
    RepresentationCandidateObservation,
    RepresentationVisualTensorBundle,
)
from .runtime import Qwen3RepresentationRuntime
from .schema import RepresentationTrainingSample
from .transcript import _build_visual_token_expansion


QWEN3_COUNTERFACTUAL_PAIR_SCHEMA_VERSION = "qwen3_counterfactual_pair_v1"
QWEN3_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION = "qwen3_counterfactual_manifest_v1"
QWEN3_D_ONLY_CONTEXT_SCHEMA_VERSION = "qwen3_native_d_only_context_v1"
QWEN3_COUNTERFACTUAL_BUILD_SCHEMA_VERSION = "qwen3_counterfactual_build_v1"
QWEN3_D_ONLY_TOOL_REASONING = "I will inspect only the requested local field."
_D_ONLY_CONTEXT_ID_SCHEMA = "qwen3_native_d_only_context_identity_v1"
_OBSERVATION_ID_SCHEMA = "qwen3_counterfactual_observation_identity_v1"
_FORBIDDEN_VALUE_FRAGMENTS = (
    "<|im_",
    "<|vision_",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
)


@dataclass(frozen=True, slots=True)
class Qwen3CounterfactualPairSpec:
    """One externally audited pair differing in one declared local value."""

    pair_id: str
    sample_a_id: str
    sample_b_id: str
    expected_value_a: str
    expected_value_b: str
    pair_audit_identity: str
    schema_version: str = QWEN3_COUNTERFACTUAL_PAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "sample_a_id",
            "sample_b_id",
            "expected_value_a",
            "expected_value_b",
            "pair_audit_identity",
        ):
            _non_empty_text(getattr(self, name), name=name)
        if self.sample_a_id == self.sample_b_id:
            raise ValueError("counterfactual pair samples must be distinct")
        if self.expected_value_a == self.expected_value_b:
            raise ValueError("counterfactual pair values must be distinct")
        if self.schema_version != QWEN3_COUNTERFACTUAL_PAIR_SCHEMA_VERSION:
            raise ValueError("Qwen3 counterfactual pair schema mismatch")

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "pair_id": self.pair_id,
                "sample_a_id": self.sample_a_id,
                "sample_b_id": self.sample_b_id,
                "expected_value_a": self.expected_value_a,
                "expected_value_b": self.expected_value_b,
                "pair_audit_identity": self.pair_audit_identity,
            }
        )


@dataclass(frozen=True, slots=True)
class Qwen3CounterfactualManifest:
    """Ordered, content-identified counterfactual-pair manifest."""

    identity: str
    source_data_manifest_sha256: str
    pairs: tuple[Qwen3CounterfactualPairSpec, ...]
    schema_version: str = QWEN3_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_text(self.identity, name="counterfactual manifest identity")
        _lowercase_sha256(
            self.source_data_manifest_sha256,
            name="source_data_manifest_sha256",
        )
        if not isinstance(self.pairs, tuple) or not self.pairs:
            raise ValueError("counterfactual manifest requires ordered pairs")
        if any(
            not isinstance(pair, Qwen3CounterfactualPairSpec) for pair in self.pairs
        ):
            raise TypeError("counterfactual manifest contains an invalid pair")
        if len({pair.pair_id for pair in self.pairs}) != len(self.pairs):
            raise ValueError("counterfactual manifest pair IDs must be unique")
        sample_ids = tuple(
            sample_id
            for pair in self.pairs
            for sample_id in (pair.sample_a_id, pair.sample_b_id)
        )
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("one sample cannot appear in multiple manifest pair slots")
        if self.schema_version != QWEN3_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Qwen3 counterfactual manifest schema mismatch")

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "identity": self.identity,
                "source_data_manifest_sha256": self.source_data_manifest_sha256,
                "pairs": tuple(
                    {
                        "content_sha256": pair.content_sha256,
                        "pair_id": pair.pair_id,
                    }
                    for pair in self.pairs
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class Qwen3DOnlyProcessorPrefix:
    """Real processor output before model-owned M-RoPE construction."""

    transcript: RenderedTranscript
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_grid_thw: torch.Tensor
    d_positions: tuple[int, ...]
    geometry_pixel_values_shape: tuple[int, ...]
    prompt_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.transcript, RenderedTranscript):
            raise TypeError("D-only prefix requires a rendered transcript")
        if (
            self.input_ids.dtype != torch.long
            or self.input_ids.ndim != 2
            or (self.input_ids.shape[0] != 1)
        ):
            raise ValueError("D-only processor input_ids must be long [1,S]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("D-only processor attention mask must match input IDs")
        if self.attention_mask.dtype.is_floating_point or not bool(
            self.attention_mask.bool().all().item()
        ):
            raise ValueError("D-only processor attention mask must be integer/all-one")
        if self.image_grid_thw.shape != (1, 3):
            raise ValueError("D-only processor prefix requires one image grid")
        if self.image_grid_thw.dtype.is_floating_point or bool(
            (self.image_grid_thw <= 0).any().item()
        ):
            raise ValueError("D-only processor grid must be positive integers")
        if not self.d_positions or len(set(self.d_positions)) != len(self.d_positions):
            raise ValueError("D-only processor prefix requires unique D positions")
        if self.d_positions != tuple(
            range(self.d_positions[0], self.d_positions[-1] + 1)
        ):
            raise ValueError("D-only processor D positions must be contiguous")
        if any(
            position < 0 or position >= self.input_ids.shape[1]
            for position in self.d_positions
        ):
            raise ValueError("D-only processor D position lies outside input IDs")
        if not self.geometry_pixel_values_shape or any(
            dimension <= 0 for dimension in self.geometry_pixel_values_shape
        ):
            raise ValueError("geometry-carrier pixel tensor shape must be non-empty")
        _non_empty_text(self.prompt_identity, name="prompt_identity")


@dataclass(frozen=True, slots=True)
class Qwen3NativeDOnlyContextRecord:
    """Bound native transcript/grid state used for every no-cache extension."""

    pair_id: str
    prompt_identity: str
    transcript: RenderedTranscript
    context: NativeDOnlyContext
    schema_version: str = QWEN3_D_ONLY_CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_text(self.pair_id, name="pair_id")
        _non_empty_text(self.prompt_identity, name="prompt_identity")
        if not isinstance(self.transcript, RenderedTranscript):
            raise TypeError("Qwen3 D-only record requires a rendered transcript")
        if not isinstance(self.context, NativeDOnlyContext):
            raise TypeError("Qwen3 D-only record requires a typed context")
        if self.context.family != "qwen3_vl":
            raise ValueError("Qwen3 D-only record has another family")
        if self.context.transcript_identity != self.transcript.token_ids_sha256:
            raise ValueError("Qwen3 D-only record transcript identity drifted")
        if self.schema_version != QWEN3_D_ONLY_CONTEXT_SCHEMA_VERSION:
            raise ValueError("Qwen3 D-only context record schema mismatch")


@dataclass(frozen=True, slots=True)
class Qwen3CounterfactualBuild:
    manifest_identity: str
    manifest_sha256: str
    cases: tuple[NativeCounterfactualCase, ...]
    contexts: tuple[Qwen3NativeDOnlyContextRecord, ...]
    materializer: Qwen3NativeInjectedRequestMaterializer
    schema_version: str = QWEN3_COUNTERFACTUAL_BUILD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_text(self.manifest_identity, name="manifest_identity")
        _lowercase_sha256(self.manifest_sha256, name="manifest_sha256")
        if not self.cases or len(self.cases) != len(self.contexts):
            raise ValueError("counterfactual cases and contexts must align")
        if not isinstance(self.materializer, Qwen3NativeInjectedRequestMaterializer):
            raise TypeError("counterfactual build requires its bound materializer")
        if self.schema_version != QWEN3_COUNTERFACTUAL_BUILD_SCHEMA_VERSION:
            raise ValueError("Qwen3 counterfactual build schema mismatch")


def build_qwen3_d_only_messages(
    sample: RepresentationTrainingSample,
    prompt: RepresentationPromptConfig,
) -> tuple[dict[str, Any], ...]:
    """Render user text, native call, and one D-backed tool response only."""

    messages = build_native_representation_messages(sample, prompt)
    user_content = messages[0]["content"]
    text_items = tuple(
        dict(item)
        for item in user_content
        if isinstance(item, Mapping) and item.get("type") == "text"
    )
    if len(text_items) != 1:
        raise ValueError("D-only user context requires exactly one text item")
    tool_call = dict(messages[1])
    tool_call["reasoning_content"] = QWEN3_D_ONLY_TOOL_REASONING
    return (
        {"role": "user", "content": text_items},
        tool_call,
        dict(messages[2]),
    )


def materialize_qwen3_d_only_processor_prefix(
    *,
    processor: Any,
    renderer: NativeProtocolRenderer,
    sample: RepresentationTrainingSample,
    prompt: RepresentationPromptConfig,
    geometry_image: Any,
) -> Qwen3DOnlyProcessorPrefix:
    """Use the real Qwen processor for one tool-response geometry block."""

    if not isinstance(renderer, NativeProtocolRenderer):
        raise TypeError("renderer must be NativeProtocolRenderer")
    messages = build_qwen3_d_only_messages(sample, prompt)
    transcript = renderer.render(messages, add_generation_prompt=True)
    renderer.assert_generation_prefill(transcript, renderer.tokenizer)
    visual_token_id = renderer.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if isinstance(visual_token_id, bool) or not isinstance(visual_token_id, int):
        raise TypeError("Qwen3 image placeholder must resolve to an integer")
    if renderer.tokenizer.convert_ids_to_tokens(visual_token_id) != "<|image_pad|>":
        raise ValueError("Qwen3 image placeholder token does not round trip")
    if transcript.token_ids.count(visual_token_id) != 1:
        raise ValueError("fresh D-only transcript must contain one vision placeholder")
    batch = _processor_batch(
        processor,
        text=transcript.text,
        images=(geometry_image,),
    )
    input_ids = batch["input_ids"].to(dtype=torch.long)
    attention_mask = batch["attention_mask"]
    image_grid_thw = batch["image_grid_thw"].to(dtype=torch.long)
    expansion = _build_visual_token_expansion(
        family="qwen3_vl",
        canonical_token_ids=transcript.token_ids,
        model_token_ids=tuple(int(value) for value in input_ids[0].tolist()),
        visual_placeholder_token_id=visual_token_id,
    )
    d_positions = expansion.visual_model_positions
    if image_grid_thw.shape != (1, 3):
        raise ValueError("fresh D-only processor output must contain one grid")
    merge_size = _processor_merge_size(processor)
    if any(int(value) % merge_size for value in image_grid_thw[0, 1:].tolist()):
        raise ValueError("D-only image grid is not spatial-merge divisible")
    expected_d_tokens = int(image_grid_thw.prod().item()) // (merge_size**2)
    if expected_d_tokens != len(d_positions):
        raise ValueError("D-only visual expansion differs from the Qwen image grid")
    pixel_values = batch["pixel_values"]
    renderer.assert_tokenizer_length()
    renderer.assert_chat_template_identity()
    return Qwen3DOnlyProcessorPrefix(
        transcript=transcript,
        input_ids=input_ids.detach().clone(),
        attention_mask=attention_mask.detach().clone(),
        image_grid_thw=image_grid_thw.detach().clone(),
        d_positions=d_positions,
        geometry_pixel_values_shape=tuple(int(value) for value in pixel_values.shape),
        prompt_identity=prompt.identity,
    )


def load_qwen3_counterfactual_manifest(
    path: str | Path,
) -> Qwen3CounterfactualManifest:
    """Strictly load one deterministic JSON manifest without path inference."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("counterfactual manifest root must be an object")
    expected = {
        "schema_version",
        "identity",
        "source_data_manifest_sha256",
        "pairs",
    }
    if set(payload) != expected:
        raise ValueError("counterfactual manifest fields differ from schema v1")
    raw_pairs = payload["pairs"]
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        raise TypeError("counterfactual manifest pairs must be an array")
    pair_fields = {
        "schema_version",
        "pair_id",
        "sample_a_id",
        "sample_b_id",
        "expected_value_a",
        "expected_value_b",
        "pair_audit_identity",
    }
    pairs: list[Qwen3CounterfactualPairSpec] = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, Mapping) or set(raw_pair) != pair_fields:
            raise ValueError("counterfactual pair fields differ from schema v1")
        pairs.append(Qwen3CounterfactualPairSpec(**dict(raw_pair)))
    return Qwen3CounterfactualManifest(
        identity=payload["identity"],
        source_data_manifest_sha256=payload["source_data_manifest_sha256"],
        pairs=tuple(pairs),
        schema_version=payload["schema_version"],
    )


class Qwen3NativeInjectedRequestMaterializer(NativeInjectedRequestMaterializer):
    """Recompute exact Qwen3 M-RoPE for every teacher/greedy extension."""

    def __init__(
        self,
        *,
        runtime: Qwen3RepresentationRuntime,
        contexts: Sequence[Qwen3NativeDOnlyContextRecord],
    ) -> None:
        if not isinstance(runtime, Qwen3RepresentationRuntime):
            raise TypeError("Qwen3 materializer requires Qwen3RepresentationRuntime")
        records = tuple(contexts)
        if not records or any(
            not isinstance(record, Qwen3NativeDOnlyContextRecord) for record in records
        ):
            raise TypeError("Qwen3 materializer requires typed context records")
        if len({record.context.context_id for record in records}) != len(records):
            raise ValueError("Qwen3 materializer context IDs must be unique")
        runtime.assert_bound_invariants()
        merge_size = runtime.architecture.spatial_merge_size
        for record in records:
            temporal, height, width = record.context.image_grid_thw[0]
            if height % merge_size or width % merge_size:
                raise ValueError("Qwen3 D-only context grid is not merge-divisible")
            expected_tokens = temporal * height * width // (merge_size**2)
            if len(record.context.d_positions) != expected_tokens:
                raise ValueError("Qwen3 D-only context grid differs from D positions")
        self.runtime = runtime
        self._records = {record.context.context_id: record for record in records}
        self._context_content_sha256 = {
            record.context.context_id: _native_context_content_sha256(record.context)
            for record in records
        }
        self._forbidden_multimodal_ids = _qwen3_multimodal_token_ids(runtime)

    def value_token_ids(
        self, context: NativeDOnlyContext, value: str
    ) -> tuple[int, ...]:
        self._record(context)
        _non_empty_text(value, name="counterfactual value")
        if any(fragment in value for fragment in _FORBIDDEN_VALUE_FRAGMENTS):
            raise ValueError("counterfactual value contains native control text")
        tokenizer = self.runtime.tokenizer
        expanded_text = tokenizer.decode(
            context.input_ids[0].tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        expanded_ids = tuple(
            int(value_id)
            for value_id in tokenizer.encode(
                expanded_text,
                add_special_tokens=False,
            )
        )
        prefix = tuple(int(value_id) for value_id in context.input_ids[0].tolist())
        if expanded_ids != prefix:
            raise ValueError("Qwen3 tokenizer cannot round-trip expanded D context")
        combined_ids = tuple(
            int(value_id)
            for value_id in tokenizer.encode(
                expanded_text + value,
                add_special_tokens=False,
            )
        )
        self.runtime.renderer.assert_tokenizer_length()
        if len(combined_ids) <= len(prefix) or combined_ids[: len(prefix)] != prefix:
            raise ValueError(
                "counterfactual value retokenized the native generation prefix"
            )
        suffix = combined_ids[len(prefix) :]
        self._validate_suffix_ids(suffix)
        return suffix

    def teacher_forced(
        self,
        *,
        context: NativeDOnlyContext,
        observation: RepresentationVisualTensorBundle,
        continuation_token_ids: tuple[int, ...],
    ) -> NativeTeacherForcedForward:
        request = self._materialize_request(
            context=context,
            observation=observation,
            suffix=continuation_token_ids,
        )
        start = context.input_ids.shape[1]
        return NativeTeacherForcedForward(
            request=request,
            continuation_positions=tuple(
                range(start, start + len(continuation_token_ids))
            ),
        )

    def generation_step(
        self,
        *,
        context: NativeDOnlyContext,
        observation: RepresentationVisualTensorBundle,
        generated_token_ids: tuple[int, ...],
    ) -> NativeGenerationForward:
        request = self._materialize_request(
            context=context,
            observation=observation,
            suffix=generated_token_ids,
        )
        return NativeGenerationForward(
            request=request,
            next_token_logit_position=request.input_ids.shape[1] - 1,
        )

    def decode_generated(self, token_ids: tuple[int, ...]) -> str:
        _validate_token_ids(token_ids, name="generated token IDs", allow_empty=False)
        self._validate_suffix_ids(token_ids)
        try:
            text = self.runtime.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError as error:
            raise TypeError("Qwen3 tokenizer rejected exact generated IDs") from error
        self.runtime.renderer.assert_tokenizer_length()
        if not isinstance(text, str) or not text:
            raise ValueError("Qwen3 generated IDs decode to empty text")
        return text

    def extract_expected_value(
        self, generated_text: str, expected_value: str
    ) -> str | None:
        _non_empty_text(generated_text, name="generated_text")
        _non_empty_text(expected_value, name="expected_value")
        return expected_value if expected_value in generated_text else None

    def _materialize_request(
        self,
        *,
        context: NativeDOnlyContext,
        observation: RepresentationVisualTensorBundle,
        suffix: tuple[int, ...],
    ) -> InjectedForwardRequest:
        self._record(context)
        _validate_token_ids(suffix, name="native continuation IDs", allow_empty=True)
        self._validate_suffix_ids(suffix)
        if len(context.d_positions) != observation.main.shape[1]:
            raise ValueError("Qwen3 D-only positions differ from observation tokens")
        if observation.branch_layers != self.runtime.architecture.branch_layers:
            raise ValueError("Qwen3 D-only observation branch order drifted")
        if len(observation.deepstack) != len(self.runtime.architecture.branch_layers):
            raise ValueError("Qwen3 D-only observation is missing DeepStack branches")
        if context.attention_mask.ndim != 2 or not bool(
            context.attention_mask.bool().all().item()
        ):
            raise ValueError("Qwen3 D-only context requires an unpadded 2D mask")
        device = context.input_ids.device
        suffix_tensor = torch.tensor((suffix,), dtype=torch.long, device=device)
        input_ids = torch.cat((context.input_ids, suffix_tensor), dim=1)
        attention_mask = torch.ones_like(input_ids, dtype=context.attention_mask.dtype)
        grid = torch.tensor(
            context.image_grid_thw,
            dtype=torch.long,
            device=device,
        )
        position_ids = _qwen3_position_ids(
            self.runtime.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=grid,
        )
        if not torch.equal(
            position_ids[..., : context.input_ids.shape[1]],
            context.position_ids,
        ):
            raise RuntimeError("Qwen3 M-RoPE recompute changed the D-only prefix")
        block = InjectedVisualBlock(
            kind="focused_d",
            positions=context.d_positions,
            embeddings=observation.main,
            deepstack=observation.deepstack,
            deepstack_positions=tuple(
                context.d_positions for _ in observation.deepstack
            ),
        )
        self.runtime.assert_bound_invariants()
        return InjectedForwardRequest(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_blocks=(block,),
            use_cache=False,
        )

    def _record(self, context: NativeDOnlyContext) -> Qwen3NativeDOnlyContextRecord:
        if not isinstance(context, NativeDOnlyContext):
            raise TypeError("Qwen3 materializer context must be NativeDOnlyContext")
        try:
            record = self._records[context.context_id]
        except KeyError as error:
            raise ValueError("Qwen3 materializer does not own this context") from error
        if (
            _native_context_content_sha256(context)
            != self._context_content_sha256[context.context_id]
        ):
            raise ValueError("Qwen3 materializer context identity/content drifted")
        expected = record.context
        if context is not expected and (
            context.transcript_identity != expected.transcript_identity
            or context.family != expected.family
            or context.d_positions != expected.d_positions
            or context.image_grid_thw != expected.image_grid_thw
            or context.source_image_positions != expected.source_image_positions
            or context.pre_d_text_kv_reused != expected.pre_d_text_kv_reused
            or context.cache_mode != expected.cache_mode
            or not torch.equal(context.input_ids, expected.input_ids)
            or not torch.equal(context.attention_mask, expected.attention_mask)
            or not torch.equal(context.position_ids, expected.position_ids)
        ):
            raise ValueError("Qwen3 materializer context identity/content drifted")
        return record

    def _validate_suffix_ids(self, token_ids: tuple[int, ...]) -> None:
        # Even generated continuations cannot introduce another image/video
        # placeholder because this materializer owns exactly one recorded grid.
        if any(token_id in self._forbidden_multimodal_ids for token_id in token_ids):
            raise ValueError("D-only continuation introduced another visual block")


class Qwen3CounterfactualCaseBuilder:
    """Bind audited manifest rows to detached exact Adapter observations."""

    def __init__(
        self,
        *,
        runtime: Qwen3RepresentationRuntime,
        prompt: RepresentationPromptConfig,
    ) -> None:
        if not isinstance(runtime, Qwen3RepresentationRuntime):
            raise TypeError("counterfactual builder requires Qwen3 runtime")
        if not isinstance(prompt, RepresentationPromptConfig):
            raise TypeError("counterfactual builder requires explicit prompt")
        runtime.assert_bound_invariants()
        self.runtime = runtime
        self.prompt = prompt

    def build(
        self,
        *,
        manifest: Qwen3CounterfactualManifest,
        data_manifest_sha256: str,
        samples: Sequence[RepresentationTrainingSample],
        observations: Mapping[str, RepresentationCandidateObservation],
    ) -> Qwen3CounterfactualBuild:
        if not isinstance(manifest, Qwen3CounterfactualManifest):
            raise TypeError("manifest must be Qwen3CounterfactualManifest")
        _lowercase_sha256(data_manifest_sha256, name="data_manifest_sha256")
        if data_manifest_sha256 != manifest.source_data_manifest_sha256:
            raise ValueError(
                "counterfactual manifest does not identify the supplied dataset"
            )
        sample_map = _sample_map(samples)
        if not isinstance(observations, Mapping):
            raise TypeError("counterfactual observations must be a mapping")
        contexts: list[Qwen3NativeDOnlyContextRecord] = []
        cases: list[NativeCounterfactualCase] = []
        for pair in manifest.pairs:
            sample_a = _required_sample(sample_map, pair.sample_a_id)
            sample_b = _required_sample(sample_map, pair.sample_b_id)
            candidate_a = _required_candidate(observations, pair.sample_a_id)
            candidate_b = _required_candidate(observations, pair.sample_b_id)
            _validate_counterfactual_pair_inputs(
                pair,
                sample_a=sample_a,
                sample_b=sample_b,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
                prompt=self.prompt,
            )
            observation_a = _detached_visual(candidate_a.visual)
            observation_b = _detached_visual(candidate_b.visual)
            geometry_a = _qwen3_geometry_carrier(
                self.runtime.processor,
                candidate_a.image_grid_thw,
            )
            geometry_b = _qwen3_geometry_carrier(
                self.runtime.processor,
                candidate_b.image_grid_thw,
            )
            prefix_a = materialize_qwen3_d_only_processor_prefix(
                processor=self.runtime.processor,
                renderer=self.runtime.renderer,
                sample=sample_a,
                prompt=self.prompt,
                geometry_image=geometry_a,
            )
            prefix_b = materialize_qwen3_d_only_processor_prefix(
                processor=self.runtime.processor,
                renderer=self.runtime.renderer,
                sample=sample_a,
                prompt=self.prompt,
                geometry_image=geometry_b,
            )
            _validate_matched_geometry(prefix_a, prefix_b)
            if len(prefix_a.d_positions) != observation_a.main.shape[1]:
                raise ValueError("counterfactual D shape differs from native geometry")
            device = _runtime_language_device(self.runtime)
            input_ids = prefix_a.input_ids.to(device=device)
            attention_mask = prefix_a.attention_mask.to(device=device)
            grid = prefix_a.image_grid_thw.to(device=device)
            position_ids = _qwen3_position_ids(
                self.runtime.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=grid,
            )
            context_id = _d_only_context_identity(
                pair=pair,
                prompt=self.prompt,
                prefix=prefix_a,
                position_ids=position_ids,
            )
            context = NativeDOnlyContext(
                context_id=context_id,
                transcript_identity=prefix_a.transcript.token_ids_sha256,
                family="qwen3_vl",
                input_ids=input_ids.detach().clone(),
                attention_mask=attention_mask.detach().clone(),
                position_ids=position_ids.detach().clone(),
                d_positions=prefix_a.d_positions,
                image_grid_thw=tuple(
                    tuple(int(value) for value in row.tolist())
                    for row in prefix_a.image_grid_thw
                ),
            )
            record = Qwen3NativeDOnlyContextRecord(
                pair_id=pair.pair_id,
                prompt_identity=self.prompt.identity,
                transcript=prefix_a.transcript,
                context=context,
            )
            contexts.append(record)
            cases.append(
                NativeCounterfactualCase(
                    case_id=f"{manifest.identity}:{pair.pair_id}",
                    pair_identity=pair.content_sha256,
                    sample_a_id=pair.sample_a_id,
                    sample_b_id=pair.sample_b_id,
                    expected_value_a=pair.expected_value_a,
                    expected_value_b=pair.expected_value_b,
                    observation_a_identity=_observation_identity(candidate_a),
                    observation_b_identity=_observation_identity(candidate_b),
                    context=context,
                    observation_a=observation_a,
                    observation_b=observation_b,
                )
            )
        materializer = Qwen3NativeInjectedRequestMaterializer(
            runtime=self.runtime,
            contexts=tuple(contexts),
        )
        self.runtime.assert_bound_invariants()
        return Qwen3CounterfactualBuild(
            manifest_identity=manifest.identity,
            manifest_sha256=manifest.content_sha256,
            cases=tuple(cases),
            contexts=tuple(contexts),
            materializer=materializer,
        )


def _validate_counterfactual_pair_inputs(
    pair: Qwen3CounterfactualPairSpec,
    *,
    sample_a: RepresentationTrainingSample,
    sample_b: RepresentationTrainingSample,
    candidate_a: RepresentationCandidateObservation,
    candidate_b: RepresentationCandidateObservation,
    prompt: RepresentationPromptConfig,
) -> None:
    if sample_a.image == sample_b.image:
        raise ValueError("counterfactual pair must reference distinct images")
    if sample_a.question != sample_b.question or sample_a.target != sample_b.target:
        raise ValueError("counterfactual pair must share exact question and target")
    if prompt.render(sample_a) != prompt.render(sample_b):
        raise ValueError("counterfactual pair must render one identical fresh prompt")
    if candidate_a.sample_id != sample_a.sample_id or (
        candidate_b.sample_id != sample_b.sample_id
    ):
        raise ValueError("counterfactual candidate/sample identities differ")
    if candidate_a.source_visual_identity == candidate_b.source_visual_identity:
        raise ValueError("counterfactual observations must come from distinct images")
    if candidate_a.target_conditioning_provider is not (
        candidate_b.target_conditioning_provider
    ):
        raise ValueError("counterfactual observations mix conditioning providers")
    if candidate_a.projection_identities != candidate_b.projection_identities:
        raise ValueError("counterfactual observations mix projection identities")
    if candidate_a.image_grid_thw is None or candidate_b.image_grid_thw is None:
        raise ValueError("counterfactual observations must retain exact image grids")
    if candidate_a.image_grid_thw != candidate_b.image_grid_thw:
        raise ValueError("counterfactual observations must share exact image geometry")
    _assert_visual_contract(candidate_b.visual, candidate_a.visual)
    if pair.expected_value_a not in sample_a.evidence_description or (
        pair.expected_value_b not in sample_b.evidence_description
    ):
        raise ValueError(
            "manifest values must appear exactly in paired evidence labels"
        )


def _validate_matched_geometry(
    first: Qwen3DOnlyProcessorPrefix,
    second: Qwen3DOnlyProcessorPrefix,
) -> None:
    if first.transcript.token_ids != second.transcript.token_ids or not torch.equal(
        first.input_ids, second.input_ids
    ):
        raise ValueError("counterfactual geometry changed native transcript IDs")
    if not torch.equal(first.attention_mask, second.attention_mask) or not torch.equal(
        first.image_grid_thw, second.image_grid_thw
    ):
        raise ValueError("counterfactual images do not share attention/grid geometry")
    if first.d_positions != second.d_positions or (
        first.geometry_pixel_values_shape != second.geometry_pixel_values_shape
    ):
        raise ValueError("counterfactual images do not share processor geometry")


def _detached_visual(
    visual: RepresentationVisualTensorBundle,
) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=visual.main.detach().clone(),
        deepstack=tuple(branch.detach().clone() for branch in visual.deepstack),
        branch_layers=visual.branch_layers,
    )


def _assert_visual_contract(
    actual: RepresentationVisualTensorBundle,
    expected: RepresentationVisualTensorBundle,
) -> None:
    if actual.branch_layers != expected.branch_layers:
        raise ValueError("counterfactual observation branch order differs")
    for actual_tensor, expected_tensor in zip(
        (actual.main, *actual.deepstack),
        (expected.main, *expected.deepstack),
        strict=True,
    ):
        if (
            actual_tensor.shape != expected_tensor.shape
            or actual_tensor.dtype != expected_tensor.dtype
            or actual_tensor.device != expected_tensor.device
        ):
            raise ValueError("counterfactual observation tensor contract differs")


def _sample_map(
    samples: Sequence[RepresentationTrainingSample],
) -> dict[str, RepresentationTrainingSample]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("counterfactual samples must be a sequence")
    materialized = tuple(samples)
    if not materialized or any(
        not isinstance(sample, RepresentationTrainingSample) for sample in materialized
    ):
        raise TypeError("counterfactual samples must be typed and non-empty")
    result = {sample.sample_id: sample for sample in materialized}
    if len(result) != len(materialized):
        raise ValueError("counterfactual sample IDs must be unique")
    return result


def _required_sample(
    samples: Mapping[str, RepresentationTrainingSample], sample_id: str
) -> RepresentationTrainingSample:
    try:
        return samples[sample_id]
    except KeyError as error:
        raise ValueError(
            f"counterfactual sample {sample_id!r} is unavailable"
        ) from error


def _required_candidate(
    observations: Mapping[str, RepresentationCandidateObservation], sample_id: str
) -> RepresentationCandidateObservation:
    try:
        candidate = observations[sample_id]
    except KeyError as error:
        raise ValueError(
            f"counterfactual observation {sample_id!r} is unavailable"
        ) from error
    if not isinstance(candidate, RepresentationCandidateObservation):
        raise TypeError("counterfactual observation mapping contains another type")
    return candidate


def _processor_merge_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if (
        isinstance(merge_size, bool)
        or not isinstance(merge_size, int)
        or (merge_size <= 0)
    ):
        raise TypeError("Qwen3 image processor must expose a positive merge_size")
    return merge_size


def _qwen3_geometry_carrier(
    processor: Any,
    image_grid_thw: tuple[int, int, int] | None,
) -> Any:
    """Create content-free pixels that reproduce one recorded Qwen image grid."""

    if image_grid_thw is None:
        raise ValueError("Qwen3 geometry carrier requires a recorded image grid")
    temporal, grid_height, grid_width = image_grid_thw
    if temporal != 1:
        raise ValueError("Qwen3 static-image counterfactual requires temporal grid 1")
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    if (
        isinstance(patch_size, bool)
        or not isinstance(patch_size, int)
        or (patch_size <= 0)
    ):
        raise TypeError("Qwen3 image processor must expose a positive patch_size")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - processor dependency boundary
        raise RuntimeError("Qwen3 geometry carrier requires Pillow") from error
    return Image.new(
        "RGB",
        (grid_width * patch_size, grid_height * patch_size),
        color=(0, 0, 0),
    )


def _runtime_language_device(runtime: Qwen3RepresentationRuntime) -> torch.device:
    embedding = resolve_language_model(runtime.model).get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("Qwen3 input embedding must expose a tensor weight")
    return weight.device


def _qwen3_multimodal_token_ids(
    runtime: Qwen3RepresentationRuntime,
) -> frozenset[int]:
    tokens = (
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    )
    ids: list[int] = []
    for token in tokens:
        token_id = runtime.tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"Qwen3 control token {token!r} has no integer ID")
        if runtime.tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(f"Qwen3 control token {token!r} does not round trip")
        ids.append(token_id)
    return frozenset(ids)


def _d_only_context_identity(
    *,
    pair: Qwen3CounterfactualPairSpec,
    prompt: RepresentationPromptConfig,
    prefix: Qwen3DOnlyProcessorPrefix,
    position_ids: torch.Tensor,
) -> str:
    return _canonical_sha256(
        {
            "schema": _D_ONLY_CONTEXT_ID_SCHEMA,
            "pair_sha256": pair.content_sha256,
            "prompt_identity": prompt.identity,
            "prompt_sha256": prompt.expected_sha256,
            "transcript_text_sha256": prefix.transcript.text_sha256,
            "transcript_token_ids_sha256": prefix.transcript.token_ids_sha256,
            "model_input_ids_sha256": _integer_tensor_sha256(prefix.input_ids),
            "position_ids_sha256": _integer_tensor_sha256(position_ids),
            "image_grid_thw": tuple(
                int(value) for value in prefix.image_grid_thw[0].tolist()
            ),
            "d_positions": prefix.d_positions,
        }
    )


def _observation_identity(candidate: RepresentationCandidateObservation) -> str:
    tensors = (candidate.visual.main, *candidate.visual.deepstack)
    return _canonical_sha256(
        {
            "schema": _OBSERVATION_ID_SCHEMA,
            "sample_id": candidate.sample_id,
            "image_group_key": candidate.image_group_key,
            "source_visual_identity": candidate.source_visual_identity,
            "provider": candidate.target_conditioning_provider.value,
            "projection_identities": candidate.projection_identities,
            "image_grid_thw": candidate.image_grid_thw,
            "branch_layers": candidate.visual.branch_layers,
            "tensor_sha256": tuple(
                _floating_tensor_sha256(tensor) for tensor in tensors
            ),
        }
    )


def _native_context_content_sha256(context: NativeDOnlyContext) -> str:
    return _canonical_sha256(
        {
            "context_id": context.context_id,
            "transcript_identity": context.transcript_identity,
            "family": context.family,
            "input_ids": _integer_tensor_identity(context.input_ids),
            "attention_mask": _integer_tensor_identity(context.attention_mask),
            "position_ids": _integer_tensor_identity(context.position_ids),
            "d_positions": context.d_positions,
            "image_grid_thw": context.image_grid_thw,
            "source_image_positions": context.source_image_positions,
            "pre_d_text_kv_reused": context.pre_d_text_kv_reused,
            "cache_mode": context.cache_mode,
        }
    )


def _integer_tensor_identity(tensor: torch.Tensor) -> Mapping[str, Any]:
    return {
        "sha256": _integer_tensor_sha256(tensor),
        "dtype": str(tensor.dtype),
        "shape": tuple(int(value) for value in tensor.shape),
        "device": str(tensor.device),
    }


def _integer_tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().to(device="cpu", dtype=torch.int64).contiguous().view(-1)
    raw = b"".join(struct.pack("<q", int(value)) for value in values.tolist())
    return sha256(raw).hexdigest()


def _floating_tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().to(device="cpu").contiguous()
    raw = values.view(torch.uint8).numpy().tobytes()
    header = json.dumps(
        {"dtype": str(values.dtype), "shape": tuple(values.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(header + b"\0" + raw).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_token_ids(
    token_ids: object,
    *,
    name: str,
    allow_empty: bool,
) -> None:
    if (
        not isinstance(token_ids, tuple)
        or (not token_ids and not allow_empty)
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in token_ids
        )
    ):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise ValueError(f"{name} must be {qualifier} non-negative integer IDs")


def _non_empty_text(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _lowercase_sha256(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


__all__ = [
    "QWEN3_COUNTERFACTUAL_BUILD_SCHEMA_VERSION",
    "QWEN3_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION",
    "QWEN3_COUNTERFACTUAL_PAIR_SCHEMA_VERSION",
    "QWEN3_D_ONLY_CONTEXT_SCHEMA_VERSION",
    "QWEN3_D_ONLY_TOOL_REASONING",
    "Qwen3CounterfactualBuild",
    "Qwen3CounterfactualCaseBuilder",
    "Qwen3CounterfactualManifest",
    "Qwen3CounterfactualPairSpec",
    "Qwen3DOnlyProcessorPrefix",
    "Qwen3NativeDOnlyContextRecord",
    "Qwen3NativeInjectedRequestMaterializer",
    "build_qwen3_d_only_messages",
    "load_qwen3_counterfactual_manifest",
    "materialize_qwen3_d_only_processor_prefix",
]
