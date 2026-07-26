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
    conditioning_target_token_start: int
    conditioning_target_token_end: int
    source_sequence_length: int
    source_input_ids_sha256: str
    trajectory_ids: tuple[str, ...]
    call_indices: tuple[int, ...]
    hidden_layer: int | None
    contextual_forward_identity: ArtifactIdentity | None
    policy_version: PolicyVersion
    embedding_identity: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in {"contextual_hidden_state", "target_token_embedding"}:
            raise ValueError(f"unknown target-conditioning provider {self.provider!r}")
        if self.sampled_target_token_start < 0:
            raise ValueError("target token start must be non-negative")
        if self.sampled_target_token_end <= self.sampled_target_token_start:
            raise ValueError("target token span must be non-empty")
        if self.conditioning_target_token_start < 0 or (
            self.conditioning_target_token_end <= self.conditioning_target_token_start
        ):
            raise ValueError("conditioning target token span must be non-empty")
        if (
            self.conditioning_target_token_end - self.conditioning_target_token_start
            != self.sampled_target_token_end - self.sampled_target_token_start
        ):
            raise ValueError("sampled and conditioning target spans must align")
        if self.source_sequence_length <= 0 or (
            self.conditioning_target_token_end > self.source_sequence_length
        ):
            raise ValueError("conditioning target span lies outside source sequence")
        _validate_sha256(self.sampled_target_text_sha256)
        _validate_sha256(self.source_input_ids_sha256)
        if not self.trajectory_ids or len(self.trajectory_ids) != len(
            self.call_indices
        ):
            raise ValueError("conditioning trajectory/call provenance must align")
        if self.provider == "contextual_hidden_state":
            if (
                self.hidden_layer is None
                or not isinstance(self.contextual_forward_identity, ArtifactIdentity)
                or self.embedding_identity is not None
            ):
                raise ValueError(
                    "contextual provenance requires a hidden layer and forward identity"
                )
        elif (
            self.hidden_layer is not None
            or self.contextual_forward_identity is not None
            or not self.embedding_identity
        ):
            raise ValueError("embedding provenance requires only an embedding identity")


@dataclass(frozen=True, slots=True)
class SourceVisualState:
    image_sha256: str
    premerge_main: TensorArtifactRef
    premerge_deepstack: tuple[TensorArtifactRef, ...]
    merged_main: TensorArtifactRef
    merged_deepstack: tuple[TensorArtifactRef, ...]
    image_grid_thw: tuple[int, int, int]
    spatial_merge_size: int
    decoded_rgb_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_sha256(self.image_sha256)
        if self.decoded_rgb_sha256 is not None:
            _validate_sha256(self.decoded_rgb_sha256)
        if self.spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if any(value <= 0 for value in self.image_grid_thw):
            raise ValueError("image_grid_thw values must be positive")
        if len(self.premerge_deepstack) != len(self.merged_deepstack):
            raise ValueError(
                "pre-merge and merged original DeepStack branches must align"
            )


@dataclass(frozen=True, slots=True)
class TrajectorySourceVisual:
    """Mandatory original-image artifact for one exact trajectory replay.

    The source image exists independently of tool observations so a direct
    answer can be replayed without inventing a tool call or a synthetic D.
    Positions are final-sequence positions materialized at rollout time.
    """

    state: SourceVisualState
    positions: tuple[int, ...]
    deepstack_branch_layers: tuple[int, ...]
    deepstack_injection_positions: tuple[tuple[int, ...], ...]
    source_pixels: TensorArtifactRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, SourceVisualState):
            raise TypeError("trajectory source visual state must be SourceVisualState")
        if self.source_pixels is not None and (
            self.source_pixels.descriptor.dtype != "uint8"
            or len(self.source_pixels.descriptor.shape) != 3
            or self.source_pixels.descriptor.shape[-1] != 3
            or any(value <= 0 for value in self.source_pixels.descriptor.shape[:2])
        ):
            raise ValueError("trajectory source pixels must be RGB uint8 [H,W,3]")
        if self.source_pixels is not None and (
            self.state.decoded_rgb_sha256 is None
            or self.state.decoded_rgb_sha256 != self.source_pixels.address.digest
        ):
            raise ValueError(
                "trajectory source pixels differ from the decoded RGB bound to features"
            )
        if not self.positions:
            raise ValueError("trajectory source visual positions must be non-empty")
        if any(
            type(position) is not int or position < 0 for position in self.positions
        ):
            raise ValueError(
                "trajectory source visual positions must be non-negative integers"
            )
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("trajectory source visual positions must be unique")
        if _feature_count(self.state.merged_main) != len(self.positions):
            raise ValueError("trajectory source merged features and positions differ")
        branch_count = len(self.state.merged_deepstack)
        if branch_count != len(self.deepstack_branch_layers) or branch_count != len(
            self.deepstack_injection_positions
        ):
            raise ValueError(
                "trajectory source DeepStack tensors, layers, and positions must align"
            )
        if len(set(self.deepstack_branch_layers)) != branch_count or any(
            type(layer) is not int or layer < 0
            for layer in self.deepstack_branch_layers
        ):
            raise ValueError(
                "trajectory source DeepStack branch layers must be unique non-negative integers"
            )
        for index, (ref, positions) in enumerate(
            zip(
                self.state.merged_deepstack,
                self.deepstack_injection_positions,
                strict=True,
            )
        ):
            if any(type(position) is not int or position < 0 for position in positions):
                raise ValueError(
                    "trajectory source DeepStack positions must be non-negative integers"
                )
            if len(set(positions)) != len(positions):
                raise ValueError(
                    "trajectory source DeepStack injection positions must be unique"
                )
            if _feature_count(ref) != len(positions):
                raise ValueError(
                    f"trajectory source DeepStack branch {index} features and positions differ"
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


@dataclass(frozen=True, slots=True)
class CropVisualState:
    """Exact rollout-time model visual state for one RGB crop."""

    crop_pixels: TensorArtifactRef
    merged_main: TensorArtifactRef
    merged_deepstack: tuple[TensorArtifactRef, ...]
    image_grid_thw: tuple[int, int, int]
    spatial_merge_size: int
    positions: tuple[int, ...]
    deepstack_branch_layers: tuple[int, ...]
    deepstack_injection_positions: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.crop_pixels.descriptor.dtype != "uint8" or (
            len(self.crop_pixels.descriptor.shape) != 3
            or self.crop_pixels.descriptor.shape[-1] != 3
        ):
            raise ValueError("crop pixels must be RGB uint8 [H,W,3]")
        if any(value <= 0 for value in self.crop_pixels.descriptor.shape[:2]):
            raise ValueError("crop pixels must have positive height and width")
        if self.spatial_merge_size <= 0:
            raise ValueError("crop spatial_merge_size must be positive")
        if len(self.image_grid_thw) != 3 or any(
            type(value) is not int or value <= 0 for value in self.image_grid_thw
        ):
            raise ValueError("crop image_grid_thw must contain three positive ints")
        if len(self.merged_deepstack) != len(self.deepstack_branch_layers) or len(
            self.merged_deepstack
        ) != len(self.deepstack_injection_positions):
            raise ValueError("crop DeepStack tensors, layers, and positions must align")
        if len(set(self.deepstack_branch_layers)) != len(self.deepstack_branch_layers):
            raise ValueError("crop DeepStack branch layers must be unique")
        if any(layer < 0 for layer in self.deepstack_branch_layers):
            raise ValueError("crop DeepStack branch layers must be non-negative")
        if not self.positions:
            raise ValueError("crop visual positions must be non-empty")
        if _feature_count(self.merged_main) != len(self.positions):
            raise ValueError("crop merged features and positions differ")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("crop positions must be unique")
        for index, (ref, positions) in enumerate(
            zip(
                self.merged_deepstack,
                self.deepstack_injection_positions,
                strict=True,
            )
        ):
            if _feature_count(ref) != len(positions):
                raise ValueError(
                    f"crop DeepStack branch {index} features and positions differ"
                )


@dataclass(frozen=True, slots=True)
class CropObservationRecord:
    """Content-identified crop pixels and their exact processed visual state."""

    schema_version: str
    observation_id: str
    call_index: int
    model: ModelIdentity
    policy_version: PolicyVersion
    processor_identity: ArtifactIdentity
    layout_identity: ArtifactIdentity
    trajectory_id: str
    source_pixels_sha256: str
    source_width: int
    source_height: int
    model_coordinate_space: str
    coordinate_conversion_version: str
    coordinate_reference_width: int
    coordinate_reference_height: int
    model_bbox_2d: tuple[int, int, int, int]
    source_bbox_2d: tuple[int, int, int, int]
    effective_bbox_2d: tuple[int, int, int, int]
    source_visual: SourceVisualState
    sequence_length: int
    original_image_positions: tuple[int, ...]
    crop_visual: CropVisualState

    def __post_init__(self) -> None:
        if self.schema_version != "crop-observation-v2" or not self.observation_id:
            raise ValueError("crop observation schema version and ID are required")
        if self.call_index < 0 or not self.trajectory_id:
            raise ValueError("crop observation call/trajectory identity is invalid")
        if not isinstance(self.processor_identity, ArtifactIdentity) or not isinstance(
            self.layout_identity, ArtifactIdentity
        ):
            raise TypeError("crop processor/layout identities must be explicit")
        _validate_sha256(self.source_pixels_sha256)
        if self.source_visual.decoded_rgb_sha256 != self.source_pixels_sha256:
            raise ValueError(
                "crop source pixels differ from source visual decoded RGB identity"
            )
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("crop source dimensions must be positive")
        if not self.model_coordinate_space or not self.coordinate_conversion_version:
            raise ValueError("crop coordinate identities must be explicit")
        if self.coordinate_reference_width <= 0 or self.coordinate_reference_height <= 0:
            raise ValueError("crop coordinate reference dimensions must be positive")
        for name, bbox in (
            ("model", self.model_bbox_2d),
            ("source", self.source_bbox_2d),
            ("effective", self.effective_bbox_2d),
        ):
            if len(bbox) != 4 or any(type(value) is not int for value in bbox):
                raise ValueError(f"{name} bbox must contain exactly four integers")
            left, top, right, bottom = bbox
            if right <= left or bottom <= top:
                raise ValueError(f"{name} bbox must be non-empty")
        left, top, right, bottom = self.effective_bbox_2d
        if (
            left < 0
            or top < 0
            or right > self.source_width
            or bottom > self.source_height
        ):
            raise ValueError("effective crop bbox lies outside the source image")
        crop_height, crop_width, _ = self.crop_visual.crop_pixels.descriptor.shape
        if (crop_width, crop_height) != (right - left, bottom - top):
            raise ValueError("effective bbox dimensions differ from crop pixels")
        if self.sequence_length <= 0:
            raise ValueError("crop observation sequence length must be positive")
        all_positions = self.original_image_positions + self.crop_visual.positions
        if any(
            position < 0 or position >= self.sequence_length
            for position in all_positions
        ):
            raise ValueError("crop observation visual position lies outside sequence")
        if len(set(all_positions)) != len(all_positions):
            raise ValueError("source-image and crop positions must not overlap")
        for positions in self.crop_visual.deepstack_injection_positions:
            if len(set(positions)) != len(positions) or any(
                position < 0 or position >= self.sequence_length
                for position in positions
            ):
                raise ValueError("crop DeepStack positions are invalid")
        if _feature_count(self.source_visual.merged_main) != len(
            self.original_image_positions
        ):
            raise ValueError("source visual features and positions differ")
        if len(self.source_visual.merged_deepstack) != len(
            self.crop_visual.merged_deepstack
        ):
            raise ValueError("source and crop DeepStack branch counts differ")

    @property
    def requested_bbox_2d(self) -> tuple[int, int, int, int]:
        """Compatibility alias for the exact sampled model-space box."""

        return self.model_bbox_2d


@dataclass(frozen=True, slots=True)
class CropTGVFVisualState:
    """Exact crop pixels and processor-owned visual state consumed by TGVF."""

    crop_pixels: TensorArtifactRef
    processor_identity: ArtifactIdentity
    layout_identity: ArtifactIdentity
    source: SourceVisualState

    def __post_init__(self) -> None:
        if self.crop_pixels.descriptor.dtype != "uint8" or (
            len(self.crop_pixels.descriptor.shape) != 3
            or self.crop_pixels.descriptor.shape[-1] != 3
        ):
            raise ValueError("atomic crop pixels must be RGB uint8 [H,W,3]")
        if any(value <= 0 for value in self.crop_pixels.descriptor.shape[:2]):
            raise ValueError("atomic crop pixels must have positive dimensions")
        if not isinstance(self.processor_identity, ArtifactIdentity) or not isinstance(
            self.layout_identity, ArtifactIdentity
        ):
            raise TypeError("atomic crop processor/layout identities must be explicit")
        if self.source.decoded_rgb_sha256 != self.crop_pixels.address.digest:
            raise ValueError("crop visual image identity differs from exact crop pixels")


@dataclass(frozen=True, slots=True)
class CropTGVFObservationRecord:
    """One indivisible crop-and-focus observation recorded during rollout."""

    schema_version: str
    observation_id: str
    call_index: int
    model: ModelIdentity
    representation: ArtifactIdentity
    condition: ConditionProvenance
    source_pixels_sha256: str
    source_width: int
    source_height: int
    model_coordinate_space: str
    coordinate_conversion_version: str
    coordinate_reference_width: int
    coordinate_reference_height: int
    model_bbox_2d: tuple[int, int, int, int]
    source_bbox_2d: tuple[int, int, int, int]
    effective_bbox_2d: tuple[int, int, int, int]
    sampled_target_char_span: tuple[int, int]
    source_visual: SourceVisualState
    crop_visual: CropTGVFVisualState
    payload: TensorPayloadSet
    branches: tuple[DeepStackBranchRecord, ...]
    layout: VisualLayout
    masks: ObservationMasks
    cache: CacheContract

    def __post_init__(self) -> None:
        if self.schema_version != "crop-tgvf-observation-v2" or not self.observation_id:
            raise ValueError("atomic crop+TGVF schema version and ID are required")
        if self.call_index < 0:
            raise ValueError("atomic crop+TGVF call index must be non-negative")
        _validate_sha256(self.source_pixels_sha256)
        if self.source_visual.decoded_rgb_sha256 != self.source_pixels_sha256:
            raise ValueError(
                "atomic source pixels differ from source visual decoded RGB identity"
            )
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("atomic crop+TGVF source dimensions must be positive")
        if not self.model_coordinate_space or not self.coordinate_conversion_version:
            raise ValueError("atomic crop+TGVF coordinate identities must be explicit")
        if self.coordinate_reference_width <= 0 or self.coordinate_reference_height <= 0:
            raise ValueError(
                "atomic crop+TGVF coordinate reference dimensions must be positive"
            )
        for name, bbox in (
            ("model", self.model_bbox_2d),
            ("source", self.source_bbox_2d),
            ("effective", self.effective_bbox_2d),
        ):
            if len(bbox) != 4 or any(type(value) is not int for value in bbox):
                raise ValueError(f"{name} bbox must contain exactly four integers")
            left, top, right, bottom = bbox
            if right <= left or bottom <= top:
                raise ValueError(f"{name} bbox must be non-empty")
        left, top, right, bottom = self.effective_bbox_2d
        if (
            left < 0
            or top < 0
            or right > self.source_width
            or bottom > self.source_height
        ):
            raise ValueError("effective atomic crop lies outside the source image")
        crop_height, crop_width, _ = self.crop_visual.crop_pixels.descriptor.shape
        if (crop_width, crop_height) != (right - left, bottom - top):
            raise ValueError("effective bbox dimensions differ from atomic crop pixels")
        if self.condition.call_indices != (self.call_index,) or len(
            self.condition.trajectory_ids
        ) != 1:
            raise ValueError("atomic conditioning must identify exactly this call")
        if (
            len(self.sampled_target_char_span) != 2
            or self.sampled_target_char_span[0] < 0
            or self.sampled_target_char_span[1] <= self.sampled_target_char_span[0]
        ):
            raise ValueError("atomic sampled target char span must be non-empty")

        payload_digests = tuple(ref.address.digest for ref in self.payload.deepstack)
        branch_digests = tuple(
            branch.d_tensor.address.digest for branch in self.branches
        )
        if payload_digests != branch_digests:
            raise ValueError("atomic payload and D-DeepStack branches differ")
        if self.layout.deepstack_branch_layers != tuple(
            branch.layer for branch in self.branches
        ) or self.layout.deepstack_injection_positions != tuple(
            branch.injection_positions for branch in self.branches
        ):
            raise ValueError("atomic D-DeepStack branches and layout differ")

        sequence = self.layout.sequence_length
        if self.payload.attention_mask.descriptor.shape != (1, sequence) or (
            self.payload.attention_mask.descriptor.dtype != "bool"
        ):
            raise ValueError("atomic observation attention mask must be bool [1,S]")
        position_shape = self.payload.position_ids.descriptor.shape
        if position_shape != (1, sequence) and not (
            len(position_shape) == 3 and position_shape[-2:] == (1, sequence)
        ):
            raise ValueError("atomic position IDs must have shape [1,S] or [R,1,S]")
        for name, ref in (
            ("policy", self.masks.policy_visible),
            ("reference", self.masks.reference_visible),
            ("teacher", self.masks.teacher_visible),
        ):
            if ref.descriptor.shape != (1, sequence) or ref.descriptor.dtype != "bool":
                raise ValueError(f"atomic {name} visibility mask must be bool [1,S]")
        if self.payload.token_type_ids is not None and (
            self.payload.token_type_ids.descriptor.shape != (1, sequence)
        ):
            raise ValueError("atomic token type IDs must have shape [1,S]")

        if _feature_count(self.source_visual.merged_main) != len(
            self.layout.original_image_positions
        ):
            raise ValueError("atomic original source features and positions differ")
        if _feature_count(self.payload.main_d) != len(self.layout.d_positions):
            raise ValueError("atomic main D features and positions differ")
        branch_count = len(self.branches)
        if len(self.source_visual.merged_deepstack) != branch_count or len(
            self.crop_visual.source.premerge_deepstack
        ) != branch_count or len(self.crop_visual.source.merged_deepstack) != branch_count:
            raise ValueError("atomic source/crop/D DeepStack branch counts differ")
        for branch in self.branches:
            if _feature_count(branch.d_tensor) != len(branch.injection_positions):
                raise ValueError(
                    f"atomic D-DeepStack branch {branch.layer} positions differ"
                )

    @property
    def requested_bbox_2d(self) -> tuple[int, int, int, int]:
        """Compatibility alias for the exact sampled model-space box."""

        return self.model_bbox_2d


ObservationRecord = (
    FocusedObservationRecord | CropObservationRecord | CropTGVFObservationRecord
)


def _feature_count(ref: TensorArtifactRef) -> int:
    shape = ref.descriptor.shape
    if len(shape) == 2:
        return shape[0]
    if len(shape) == 3 and shape[0] == 1:
        return shape[1]
    raise ValueError(f"visual tensor {ref.name!r} must have shape [N,H] or [1,N,H]")
