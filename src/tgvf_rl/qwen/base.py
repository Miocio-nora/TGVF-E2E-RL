"""Family-neutral recorded-observation forward contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import torch
from torch.nn import functional as F

from tgvf_rl.contracts.identity import SupportLevel
from tgvf_rl.contracts.tokens import SamplingIdentity
from tgvf_rl.observations.store import (
    ObservationHandle,
    ObservationStore,
    TrajectoryReplayHandle,
)


@dataclass(frozen=True, slots=True)
class FamilyCapabilities:
    family: str
    support_level: SupportLevel
    native_thinking_prefill: bool
    deepstack_branch_count: int
    recorded_d_forward: bool
    native_tool_template: bool


class ReplayConsumer(str, Enum):
    POLICY = "policy"
    REFERENCE = "reference"
    TEACHER = "teacher"


_NATIVE_STREAMING_PROOF_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class _NativeStreamingForwardConstructionProof:
    """Identity-bound proof for tensors built by the native streaming path.

    This is deliberately private: public and generic request validation always
    performs content checks.  The proof only removes redundant device-to-host
    reads after the streaming path has constructed an additive mask and exact
    main/DeepStack layouts itself.
    """

    authority: object
    attention_mask: torch.Tensor
    attention_mask_version: int
    visual_layout: tuple[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], ...]


@dataclass(frozen=True, slots=True)
class RecordedVisualBlock:
    """One store-verified source-image or focused-D block."""

    kind: str
    observation_handle: ObservationHandle
    call_index: int
    positions: tuple[int, ...]
    embeddings: torch.Tensor
    deepstack: tuple[torch.Tensor, ...]
    deepstack_positions: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.kind not in {"source_image", "focused_d"}:
            raise ValueError("unknown recorded visual block kind")
        if self.call_index < 0:
            raise ValueError("recorded visual call index must be non-negative")
        if len(self.deepstack) != len(self.deepstack_positions):
            raise ValueError("DeepStack tensors and injection positions must align")


@dataclass(frozen=True, slots=True)
class InjectedVisualBlock:
    """One live, differentiable source-image or focused-D tensor block."""

    kind: str
    positions: tuple[int, ...]
    embeddings: torch.Tensor
    deepstack: tuple[torch.Tensor, ...]
    deepstack_positions: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.kind not in {"source_image", "focused_d"}:
            raise ValueError("unknown injected visual block kind")
        if len(self.deepstack) != len(self.deepstack_positions):
            raise ValueError("DeepStack tensors and injection positions must align")


@dataclass(frozen=True, slots=True)
class RecordedReplayRequest:
    replay_handle: TrajectoryReplayHandle
    consumer: ReplayConsumer
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    visual_blocks: tuple[RecordedVisualBlock, ...]
    token_type_ids: torch.Tensor | None = None
    cache_position: torch.Tensor | None = None
    rope_delta: torch.Tensor | None = None
    use_cache: bool = False


@dataclass(frozen=True, slots=True)
class InjectedForwardRequest:
    """Direct live-tensor request used by differentiable representation readout."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    visual_blocks: tuple[InjectedVisualBlock, ...]
    token_type_ids: torch.Tensor | None = None
    cache_position: torch.Tensor | None = None
    rope_delta: torch.Tensor | None = None
    use_cache: bool = False
    _construction_proof: _NativeStreamingForwardConstructionProof | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.token_type_ids is not None:
            raise ValueError("live injected token_type_ids are not yet accepted")
        if self.cache_position is not None or self.rope_delta is not None:
            raise ValueError("live injected representation forward is no_cache only")
        if self.use_cache:
            raise ValueError("live injected representation forward is no_cache only")


@dataclass(frozen=True, slots=True)
class RecordedReplayResult:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    past_key_values: Any | None
    visual_position_mask: torch.Tensor


class QwenVLMFamilyAdapter(ABC):
    capabilities: FamilyCapabilities

    @abstractmethod
    def forward_recorded(
        self,
        model: Any,
        store: ObservationStore,
        replay_handle: TrajectoryReplayHandle,
        consumer: ReplayConsumer,
    ) -> RecordedReplayResult:
        """Forward exact recorded visual tensors without rerunning a vision tower."""

    def forward_injected(
        self,
        model: Any,
        request: InjectedForwardRequest,
    ) -> RecordedReplayResult:
        """Forward live visual tensors while preserving gradients to those tensors."""

        raise NotImplementedError(
            f"{self.capabilities.family} has no accepted live injected forward"
        )

    def materialize_representation_supervision(
        self,
        model: Any,
        tokenizer: Any,
        canonical: Any,
        model_input_ids: torch.Tensor,
    ) -> Any:
        """Map canonical evidence labels to exact family model positions."""

        raise NotImplementedError(
            f"{self.capabilities.family} has no accepted representation supervision"
        )

    def assert_tokenizer_invariant(self, tokenizer: Any, expected_length: int) -> None:
        actual = len(tokenizer)
        if actual != expected_length:
            raise ValueError(
                f"tokenizer length changed: expected={expected_length} actual={actual}"
            )


def assert_model_vocabulary_compatible(
    model: Any,
    tokenizer: Any,
    *,
    expected_tokenizer_length: int,
    token_ids: torch.Tensor,
) -> None:
    """Verify no tokenizer growth and that every native id has a model row.

    Qwen checkpoints may pad embedding/head matrices beyond ``len(tokenizer)``;
    exact equality is therefore not required. The two model matrices must agree,
    and the pinned tokenizer plus every realized model token must fit within
    those unchanged rows.
    """

    actual_length = len(tokenizer)
    if actual_length != expected_tokenizer_length:
        raise ValueError(
            "tokenizer length changed: "
            f"expected={expected_tokenizer_length} actual={actual_length}"
        )
    embedding = resolve_language_model(model).get_input_embeddings()
    head = resolve_lm_head(model)
    embedding_weight = getattr(embedding, "weight", None)
    head_weight = getattr(head, "weight", None)
    if not isinstance(embedding_weight, torch.Tensor) or not isinstance(
        head_weight, torch.Tensor
    ):
        raise TypeError("model embeddings and lm_head must expose tensor weights")
    embedding_rows = int(embedding_weight.shape[0])
    head_rows = int(head_weight.shape[0])
    if embedding_rows != head_rows:
        raise ValueError("input embedding and lm_head vocabulary rows differ")
    if actual_length > embedding_rows:
        raise ValueError("tokenizer vocabulary exceeds model embedding/head rows")
    if not isinstance(token_ids, torch.Tensor) or token_ids.numel() == 0:
        raise ValueError("model token ids must be a non-empty tensor")
    if token_ids.dtype != torch.long:
        raise TypeError("model token ids must have dtype torch.long")
    minimum = int(token_ids.min().item())
    maximum = int(token_ids.max().item())
    if minimum < 0 or maximum >= embedding_rows:
        raise ValueError("model token id is outside embedding/head vocabulary rows")


def resolve_replay_request(
    store: ObservationStore,
    replay_handle: TrajectoryReplayHandle,
    consumer: ReplayConsumer,
) -> RecordedReplayRequest:
    """Resolve the only accepted replay request from content-addressed state."""

    if not isinstance(store, ObservationStore):
        raise TypeError("recorded replay requires an ObservationStore")
    if not isinstance(consumer, ReplayConsumer):
        raise TypeError("recorded replay consumer must be explicit")
    replay = store.resolve_replay(replay_handle)
    if replay.cache_mode != "no_cache":
        raise ValueError(
            "the first executable replay contract is no_cache; recorded KV replay is not yet accepted"
        )
    refs = replay.tensors
    if refs.cache_position is not None or refs.rope_delta is not None:
        raise ValueError("no_cache replay must not carry cache_position or rope_delta")
    if refs.original_image_key_block is not None:
        raise ValueError(
            "original-image key blocking needs a frozen mask convention before replay"
        )
    if refs.token_type_ids is not None:
        raise ValueError(
            "token_type replay is not accepted for the current Qwen family path"
        )
    role_ref = {
        ReplayConsumer.POLICY: refs.policy_attention_mask,
        ReplayConsumer.REFERENCE: refs.reference_attention_mask,
        ReplayConsumer.TEACHER: refs.teacher_attention_mask,
    }[consumer]
    input_ids = store.resolve_verified(refs.input_ids)
    base_attention = store.resolve_verified(refs.attention_mask)
    attention_mask = store.resolve_verified(role_ref)
    if bool((attention_mask.bool() & ~base_attention.bool()).any().item()):
        raise ValueError("role attention mask cannot reveal a base-masked token")
    position_ids = store.resolve_verified(refs.position_ids)

    observations = tuple(
        store.resolve_record(handle) for handle in replay.observation_handles
    )
    blocks: list[RecordedVisualBlock] = []
    if observations:
        first = observations[0]
        source_embeddings = _normalize_recorded_features(
            store.resolve_verified(first.source_visual.merged_main),
            input_ids.shape[0],
            len(first.layout.original_image_positions),
            "source visual embeddings",
        )
        source_deepstack = tuple(
            _normalize_recorded_features(
                store.resolve_verified(ref),
                input_ids.shape[0],
                len(first.layout.original_image_positions),
                f"source DeepStack branch {index}",
            )
            for index, ref in enumerate(first.source_visual.merged_deepstack)
        )
        blocks.append(
            RecordedVisualBlock(
                kind="source_image",
                observation_handle=replay.observation_handles[0],
                call_index=0,
                positions=first.layout.original_image_positions,
                embeddings=source_embeddings,
                deepstack=source_deepstack,
                deepstack_positions=tuple(
                    first.layout.original_image_positions for _ in source_deepstack
                ),
            )
        )
    for handle, record in zip(replay.observation_handles, observations, strict=True):
        focused = _normalize_recorded_features(
            store.resolve_verified(record.payload.main_d),
            input_ids.shape[0],
            len(record.layout.d_positions),
            f"call {record.call_index} main D",
        )
        branches = tuple(
            _normalize_recorded_features(
                store.resolve_verified(branch.d_tensor),
                input_ids.shape[0],
                len(branch.injection_positions),
                f"call {record.call_index} D-DeepStack branch {branch.layer}",
            )
            for branch in record.branches
        )
        blocks.append(
            RecordedVisualBlock(
                kind="focused_d",
                observation_handle=handle,
                call_index=record.call_index,
                positions=record.layout.d_positions,
                embeddings=focused,
                deepstack=branches,
                deepstack_positions=tuple(
                    branch.injection_positions for branch in record.branches
                ),
            )
        )
    request = RecordedReplayRequest(
        replay_handle=replay_handle,
        consumer=consumer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_blocks=tuple(blocks),
        use_cache=False,
    )
    validate_replay_request(request)
    return request


def validate_replay_request(request: RecordedReplayRequest) -> tuple[int, int]:
    return _validate_forward_request(request)


def validate_injected_request(request: InjectedForwardRequest) -> tuple[int, int]:
    if not isinstance(request, InjectedForwardRequest):
        raise TypeError("request must be InjectedForwardRequest")
    return _validate_forward_request(request)


def _prove_native_streaming_injected_request(
    request: InjectedForwardRequest,
) -> InjectedForwardRequest:
    """Bind native construction guarantees to one immutable request.

    The caller is the native streaming batch constructor.  Structural tensor
    checks still run here and on consumption.  Additive-mask contents need no
    device read because that constructor only emits zero or finite non-positive
    bias.  DeepStack masks need no ``torch.equal`` because branch positions are
    proven identical to the corresponding main positions as Python tuples.
    """

    if not isinstance(request, InjectedForwardRequest):
        raise TypeError("request must be InjectedForwardRequest")
    if request._construction_proof is not None:
        raise ValueError("native streaming request is already construction-proven")
    if request.attention_mask.ndim != 4:
        raise ValueError(
            "native streaming construction proof requires a blocked attention mask"
        )
    _validate_forward_request(request, _check_additive_mask_contents=False)
    visual_layout = _request_visual_layout(request)
    if any(
        branch_positions != positions
        for positions, deepstack_positions in visual_layout
        for branch_positions in deepstack_positions
    ):
        raise ValueError(
            "Qwen DeepStack injection positions must equal the main visual positions"
        )
    proven = replace(request)
    object.__setattr__(
        proven,
        "_construction_proof",
        _NativeStreamingForwardConstructionProof(
            authority=_NATIVE_STREAMING_PROOF_AUTHORITY,
            attention_mask=proven.attention_mask,
            attention_mask_version=proven.attention_mask._version,
            visual_layout=visual_layout,
        ),
    )
    return proven


def injected_request_from_recorded(
    request: RecordedReplayRequest,
) -> InjectedForwardRequest:
    """Drop store handles after verification while retaining exact live tensors."""

    if not isinstance(request, RecordedReplayRequest):
        raise TypeError("request must be RecordedReplayRequest")
    validate_replay_request(request)
    return InjectedForwardRequest(
        input_ids=request.input_ids,
        attention_mask=request.attention_mask,
        position_ids=request.position_ids,
        visual_blocks=tuple(
            InjectedVisualBlock(
                kind=block.kind,
                positions=block.positions,
                embeddings=block.embeddings,
                deepstack=block.deepstack,
                deepstack_positions=block.deepstack_positions,
            )
            for block in request.visual_blocks
        ),
        token_type_ids=request.token_type_ids,
        cache_position=request.cache_position,
        rope_delta=request.rope_delta,
        use_cache=request.use_cache,
    )


def _validate_forward_request(
    request: RecordedReplayRequest | InjectedForwardRequest,
    *,
    _check_additive_mask_contents: bool = True,
) -> tuple[int, int]:
    if request.input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, S]")
    batch, sequence = request.input_ids.shape
    standard_attention_shape = (batch, sequence)
    blocked_attention_shape = (batch, 1, sequence, sequence)
    if request.attention_mask.shape != standard_attention_shape:
        if not isinstance(request, InjectedForwardRequest) or (
            request.attention_mask.shape != blocked_attention_shape
        ):
            raise ValueError(
                "attention_mask must have shape [B,S], or [B,1,S,S] for a "
                "live injected forward"
            )
        if not request.attention_mask.dtype.is_floating_point:
            raise TypeError("a four-dimensional attention mask must be additive")
        if _check_additive_mask_contents:
            if not bool(torch.isfinite(request.attention_mask).all().item()):
                raise ValueError("a four-dimensional attention mask must be finite")
            if bool((request.attention_mask > 0).any().item()):
                raise ValueError(
                    "an additive attention mask cannot contain positive bias"
                )
    if request.position_ids.ndim not in {2, 3}:
        raise ValueError("position_ids must have shape [B,S] or [R,B,S]")
    if request.position_ids.shape[-2:] != (batch, sequence):
        raise ValueError("position_ids trailing dimensions must be [B,S]")
    positions = tuple(
        position for block in request.visual_blocks for position in block.positions
    )
    if len(set(positions)) != len(positions):
        raise ValueError("recorded visual blocks must not overlap")
    if any(position < 0 or position >= sequence for position in positions):
        raise ValueError("visual position outside sequence")
    hidden: int | None = None
    for block_index, block in enumerate(request.visual_blocks):
        if hidden is None:
            hidden = block.embeddings.shape[-1]
        _validate_feature_tensor(
            block.embeddings,
            batch,
            len(block.positions),
            hidden,
            f"visual_blocks[{block_index}].embeddings",
        )
        for branch_index, (branch, branch_positions) in enumerate(
            zip(block.deepstack, block.deepstack_positions, strict=True)
        ):
            if any(
                position < 0 or position >= sequence for position in branch_positions
            ):
                raise ValueError("DeepStack injection position outside sequence")
            _validate_feature_tensor(
                branch,
                batch,
                len(branch_positions),
                hidden,
                f"visual_blocks[{block_index}].deepstack[{branch_index}]",
            )
    return batch, sequence


def materialize_inputs_embeds(
    model: Any, request: RecordedReplayRequest | InjectedForwardRequest
) -> tuple[torch.Tensor, torch.Tensor]:
    proof = _resolve_native_streaming_construction_proof(request)
    batch, sequence = _validate_forward_request(
        request,
        _check_additive_mask_contents=proof is None,
    )
    language_model = resolve_language_model(model)
    embedding_layer = language_model.get_input_embeddings()
    embedding_device = embedding_layer.weight.device
    embeddings = embedding_layer(request.input_ids.to(embedding_device)).clone()
    hidden = embeddings.shape[-1]
    if embeddings.shape != (batch, sequence, hidden):
        raise ValueError("language-model embeddings have an unexpected shape")
    visual_mask = torch.zeros(
        (batch, sequence), dtype=torch.bool, device=embeddings.device
    )
    for block in request.visual_blocks:
        if block.embeddings.shape[-1] != hidden:
            raise ValueError(
                "recorded visual hidden size differs from model embeddings"
            )
        _scatter_recorded(embeddings, block.positions, block.embeddings)
        if block.positions:
            visual_mask[:, list(block.positions)] = True
    return embeddings, visual_mask


def materialize_deepstack(
    request: RecordedReplayRequest | InjectedForwardRequest,
    visual_mask: torch.Tensor,
    *,
    target_dtype: torch.dtype,
) -> list[torch.Tensor]:
    proof = _resolve_native_streaming_construction_proof(request)
    if not target_dtype.is_floating_point:
        raise TypeError("DeepStack target dtype must be floating point")
    batch, sequence = visual_mask.shape
    branches: list[torch.Tensor] = []
    branch_count = (
        len(request.visual_blocks[0].deepstack) if request.visual_blocks else 0
    )
    if any(len(block.deepstack) != branch_count for block in request.visual_blocks):
        raise ValueError("all recorded visual blocks must carry the same branch count")
    for branch_index in range(branch_count):
        prototype = request.visual_blocks[0].deepstack[branch_index]
        full = torch.zeros(
            (batch, sequence, prototype.shape[-1]),
            device=visual_mask.device,
            dtype=target_dtype,
        )
        branch_mask = torch.zeros_like(visual_mask) if proof is None else None
        for block in request.visual_blocks:
            positions = block.deepstack_positions[branch_index]
            _scatter_recorded(full, positions, block.deepstack[branch_index])
            if branch_mask is not None and positions:
                branch_mask[:, list(positions)] = True
        if branch_mask is not None and not torch.equal(branch_mask, visual_mask):
            raise ValueError(
                "Qwen DeepStack injection positions must equal the main visual positions"
            )
        branches.append(full[visual_mask])
    return branches


def _request_visual_layout(
    request: InjectedForwardRequest,
) -> tuple[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], ...]:
    return tuple(
        (block.positions, block.deepstack_positions) for block in request.visual_blocks
    )


def _resolve_native_streaming_construction_proof(
    request: RecordedReplayRequest | InjectedForwardRequest,
) -> _NativeStreamingForwardConstructionProof | None:
    if not isinstance(request, InjectedForwardRequest):
        return None
    proof = request._construction_proof
    if proof is None:
        return None
    if (
        not isinstance(proof, _NativeStreamingForwardConstructionProof)
        or proof.authority is not _NATIVE_STREAMING_PROOF_AUTHORITY
    ):
        raise ValueError("invalid native streaming construction proof")
    if (
        request.attention_mask is not proof.attention_mask
        or request.attention_mask._version != proof.attention_mask_version
    ):
        raise ValueError(
            "native streaming attention mask changed after construction proof"
        )
    if _request_visual_layout(request) != proof.visual_layout:
        raise ValueError(
            "native streaming visual layout changed after construction proof"
        )
    return proof


def _normalize_recorded_features(
    tensor: torch.Tensor,
    batch: int,
    count: int,
    name: str,
) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != batch or tensor.shape[1] != count:
        raise ValueError(
            f"{name} must resolve to shape [B,{count},H], got {tuple(tensor.shape)}"
        )
    return tensor


def resolve_language_model(model: Any) -> Any:
    container = getattr(model, "model", model)
    language_model = getattr(container, "language_model", None)
    if language_model is None:
        language_model = getattr(model, "language_model", None)
    if language_model is None or not hasattr(language_model, "get_input_embeddings"):
        raise TypeError(
            "Qwen model does not expose a language_model embedding boundary"
        )
    return language_model


def resolve_lm_head(model: Any) -> Any:
    head = getattr(model, "lm_head", None)
    if head is None:
        raise TypeError("Qwen model does not expose lm_head")
    return head


def gather_next_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    sampled_positions: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Gather raw/temperature-only diagnostic logprobs.

    This function is not a behavior-measure replay when top-k/top-p/min-p,
    penalties, or custom processors are active. Objective code must use
    :func:`gather_behavior_measure_logprobs`.
    """
    if logits.ndim != 3 or token_ids.ndim != 2 or sampled_positions.ndim != 2:
        raise ValueError(
            "expected logits [B,S,V], token_ids [B,S], sampled_positions [B,K]"
        )
    if temperature <= 0:
        raise ValueError("replay temperature must be positive")
    if sampled_positions.numel() and (
        sampled_positions.min() < 1 or sampled_positions.max() >= token_ids.shape[1]
    ):
        raise ValueError("sampled positions must be in [1, sequence_length)")
    batch_indices = torch.arange(token_ids.shape[0], device=token_ids.device).unsqueeze(
        1
    )
    predictive_logits = logits[batch_indices, sampled_positions - 1]
    sampled_ids = token_ids[batch_indices, sampled_positions]
    return (
        F.log_softmax(predictive_logits.float() / temperature, dim=-1)
        .gather(-1, sampled_ids.unsqueeze(-1))
        .squeeze(-1)
    )


def gather_behavior_measure_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    sampled_positions: torch.Tensor,
    sampling: SamplingIdentity,
) -> torch.Tensor:
    """Replay the accepted processed behavior measure, failing closed.

    The current executable oracle permits only identity sampling transforms.
    Non-trivial vLLM transforms remain blocked until their exact operation
    order/history semantics pass the dedicated parity spike.
    """

    if not isinstance(sampling, SamplingIdentity):
        raise TypeError("sampling identity is required for behavior replay")
    if not sampling.has_identity_sampling_transforms:
        raise ValueError(
            "processed-logprob replay for non-trivial vLLM transforms is not yet parity-approved"
        )
    return gather_next_token_logprobs(
        logits,
        token_ids,
        sampled_positions,
        temperature=1.0,
    )


def _validate_feature_tensor(
    tensor: torch.Tensor, batch: int, count: int, hidden: int, name: str
) -> None:
    if tensor.ndim != 3 or tensor.shape != (batch, count, hidden):
        raise ValueError(
            f"{name} must have shape {(batch, count, hidden)}, got {tuple(tensor.shape)}"
        )


def _scatter_recorded(
    destination: torch.Tensor, positions: tuple[int, ...], values: torch.Tensor
) -> None:
    if positions:
        destination[:, list(positions), :] = values.to(
            device=destination.device, dtype=destination.dtype
        )
