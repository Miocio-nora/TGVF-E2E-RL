"""Framework-neutral exact observation records."""

from __future__ import annotations

from dataclasses import dataclass

from tgvf_rl.contracts.identity import (
    ArtifactIdentity,
    ModelIdentity,
    PolicyVersion,
    _validate_sha256,
)
from tgvf_rl.contracts.tensors import TensorArtifactRef, TensorPayloadSet


@dataclass(frozen=True, slots=True)
class ConditionProvenance:
    provider: str
    sampled_target_text_sha256: str
    sampled_target_token_start: int
    sampled_target_token_end: int
    source_input_ids_sha256: str
    trajectory_ids: tuple[str, ...]
    call_indices: tuple[int, ...]
    hidden_layer: int | None
    policy_version: PolicyVersion

    def __post_init__(self) -> None:
        if self.provider not in {"contextual_hidden_state", "target_token_embedding"}:
            raise ValueError(f"unknown target-conditioning provider {self.provider!r}")
        if self.sampled_target_token_start < 0:
            raise ValueError("target token start must be non-negative")
        if self.sampled_target_token_end <= self.sampled_target_token_start:
            raise ValueError("target token span must be non-empty")
        _validate_sha256(self.sampled_target_text_sha256)
        _validate_sha256(self.source_input_ids_sha256)
        if not self.trajectory_ids or len(self.trajectory_ids) != len(
            self.call_indices
        ):
            raise ValueError("conditioning trajectory/call provenance must align")


@dataclass(frozen=True, slots=True)
class SourceVisualState:
    image_sha256: str
    premerge_main: TensorArtifactRef
    premerge_deepstack: tuple[TensorArtifactRef, ...]
    merged_main: TensorArtifactRef
    merged_deepstack: tuple[TensorArtifactRef, ...]
    image_grid_thw: tuple[int, int, int]
    spatial_merge_size: int

    def __post_init__(self) -> None:
        if self.spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if any(value <= 0 for value in self.image_grid_thw):
            raise ValueError("image_grid_thw values must be positive")
        if len(self.premerge_deepstack) != len(self.merged_deepstack):
            raise ValueError(
                "pre-merge and merged original DeepStack branches must align"
            )


@dataclass(frozen=True, slots=True)
class VisualLayout:
    sequence_length: int
    original_image_positions: tuple[int, ...]
    d_positions: tuple[int, ...]
    deepstack_branch_layers: tuple[int, ...]
    deepstack_injection_positions: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if len(self.deepstack_branch_layers) != len(self.deepstack_injection_positions):
            raise ValueError("DeepStack branch layers and positions must align")
        all_positions = self.original_image_positions + self.d_positions
        if any(pos < 0 or pos >= self.sequence_length for pos in all_positions):
            raise ValueError("visual position lies outside sequence")
        if len(set(self.d_positions)) != len(self.d_positions):
            raise ValueError("D positions must be unique")
        if len(set(all_positions)) != len(all_positions):
            raise ValueError("source-image and D positions must not overlap")
        for positions in self.deepstack_injection_positions:
            if len(set(positions)) != len(positions):
                raise ValueError("DeepStack injection positions must be unique")
            if any(pos < 0 or pos >= self.sequence_length for pos in positions):
                raise ValueError("DeepStack injection position lies outside sequence")


@dataclass(frozen=True, slots=True)
class ObservationMasks:
    policy_visible: TensorArtifactRef
    reference_visible: TensorArtifactRef
    teacher_visible: TensorArtifactRef
    original_image_key_block: TensorArtifactRef | None


@dataclass(frozen=True, slots=True)
class CacheContract:
    mode: str
    prefix_length: int
    cache_position: TensorArtifactRef | None
    rope_delta: TensorArtifactRef | None
    deterministic_forward: bool
    adapter_dropout: float

    def __post_init__(self) -> None:
        if self.mode not in {"no_cache", "prefill_decode", "recorded_cache"}:
            raise ValueError(f"unknown cache mode {self.mode!r}")
        if self.prefix_length < 0:
            raise ValueError("prefix_length must be non-negative")
        if not self.deterministic_forward:
            raise ValueError("rollout/replay forward must be deterministic")
        if self.adapter_dropout != 0.0:
            raise ValueError("adapter dropout must be zero without exact mask replay")
        if self.mode == "no_cache" and (
            self.cache_position is not None or self.rope_delta is not None
        ):
            raise ValueError("no_cache observations cannot carry cache artifacts")
        if self.mode == "recorded_cache" and (
            self.cache_position is None or self.rope_delta is None
        ):
            raise ValueError(
                "recorded_cache observations require cache and RoPE artifacts"
            )


@dataclass(frozen=True, slots=True)
class DeepStackBranchRecord:
    layer: int
    d_tensor: TensorArtifactRef
    injection_positions: tuple[int, ...]
    merger_identity: ArtifactIdentity


@dataclass(frozen=True, slots=True)
class FocusedObservationRecord:
    schema_version: str
    observation_id: str
    call_index: int
    model: ModelIdentity
    representation: ArtifactIdentity
    condition: ConditionProvenance
    source_visual: SourceVisualState
    payload: TensorPayloadSet
    branches: tuple[DeepStackBranchRecord, ...]
    layout: VisualLayout
    masks: ObservationMasks
    cache: CacheContract

    def __post_init__(self) -> None:
        if self.schema_version != "focused-observation-v1" or not self.observation_id:
            raise ValueError("observation schema_version and ID must be non-empty")
        if self.call_index < 0:
            raise ValueError("call_index must be non-negative")
        payload_digests = tuple(ref.address.digest for ref in self.payload.deepstack)
        branch_digests = tuple(
            branch.d_tensor.address.digest for branch in self.branches
        )
        if payload_digests != branch_digests:
            raise ValueError("payload and branch DeepStack order/content differ")
        if self.layout.deepstack_branch_layers != tuple(
            branch.layer for branch in self.branches
        ):
            raise ValueError("layout and branch layer order differ")
        if self.layout.deepstack_injection_positions != tuple(
            branch.injection_positions for branch in self.branches
        ):
            raise ValueError("layout and branch injection positions differ")
        if self.condition.call_indices != (self.call_index,):
            raise ValueError("conditioning provenance must identify exactly this call")
        if len(self.condition.trajectory_ids) != 1:
            raise ValueError("an observation must belong to exactly one trajectory")

        sequence = self.layout.sequence_length
        if self.payload.attention_mask.descriptor.shape != (1, sequence):
            raise ValueError("observation attention mask must have shape [1,S]")
        if self.payload.attention_mask.descriptor.dtype != "bool":
            raise TypeError("observation attention mask must use bool dtype")
        position_shape = self.payload.position_ids.descriptor.shape
        if position_shape != (1, sequence) and not (
            len(position_shape) == 3 and position_shape[-2:] == (1, sequence)
        ):
            raise ValueError(
                "observation position IDs must have shape [1,S] or [R,1,S]"
            )
        for name, ref in (
            ("policy", self.masks.policy_visible),
            ("reference", self.masks.reference_visible),
            ("teacher", self.masks.teacher_visible),
        ):
            if ref.descriptor.shape != (1, sequence) or ref.descriptor.dtype != "bool":
                raise ValueError(f"{name} visibility mask must be bool [1,S]")
        if self.payload.token_type_ids is not None and (
            self.payload.token_type_ids.descriptor.shape != (1, sequence)
        ):
            raise ValueError("token type IDs must have shape [1,S]")

        source_count = _feature_count(self.source_visual.merged_main)
        if source_count != len(self.layout.original_image_positions):
            raise ValueError("source merged features and source positions differ")
        if len(self.source_visual.merged_deepstack) != len(self.branches):
            raise ValueError("source and focused DeepStack branch counts differ")
        for index, ref in enumerate(self.source_visual.merged_deepstack):
            if _feature_count(ref) != source_count:
                raise ValueError(
                    f"source DeepStack branch {index} length differs from main source"
                )
        if _feature_count(self.payload.main_d) != len(self.layout.d_positions):
            raise ValueError("main D features and D positions differ")
        for branch in self.branches:
            if _feature_count(branch.d_tensor) != len(branch.injection_positions):
                raise ValueError(
                    f"D-DeepStack branch {branch.layer} features and positions differ"
                )


def _feature_count(ref: TensorArtifactRef) -> int:
    shape = ref.descriptor.shape
    if len(shape) == 2:
        return shape[0]
    if len(shape) == 3 and shape[0] == 1:
        return shape[1]
    raise ValueError(f"visual tensor {ref.name!r} must have shape [N,H] or [1,N,H]")
