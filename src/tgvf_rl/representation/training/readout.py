"""Synthetic K×K layout/swap checks over same-image TGVF observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.qwen.base import (
    InjectedForwardRequest,
    InjectedVisualBlock,
    QwenVLMFamilyAdapter,
    resolve_language_model,
    resolve_lm_head,
)

from .losses import (
    EvidenceReadabilityLossTerms,
    SameImageMatrixCELossTerms,
    causal_evidence_losses,
    same_image_matrix_ce_loss_terms,
)
from .transcript import ModelEvidenceSupervision


@dataclass(frozen=True, slots=True)
class RepresentationVisualTensorBundle:
    """Atomic main tensor plus every ordered DeepStack branch."""

    main: torch.Tensor
    deepstack: tuple[torch.Tensor, ...]
    branch_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_visual_tensor(self.main, name="main visual tensor")
        if not self.deepstack:
            raise ValueError(
                "a representation visual bundle requires DeepStack branches"
            )
        if len(self.deepstack) != len(self.branch_layers):
            raise ValueError("DeepStack tensors and branch layers must align")
        if len(set(self.branch_layers)) != len(self.branch_layers):
            raise ValueError("DeepStack branch layers must be unique")
        for index, branch in enumerate(self.deepstack):
            _validate_visual_tensor(branch, name=f"DeepStack branch {index}")
            if branch.shape != self.main.shape:
                raise ValueError("main and every DeepStack tensor must share shape")
            if branch.device != self.main.device or branch.dtype != self.main.dtype:
                raise ValueError(
                    "main and every DeepStack tensor must share device/dtype"
                )


@dataclass(frozen=True, slots=True)
class RepresentationReadoutRow:
    """Row-fixed query, evidence labels, layout, masks, and positions."""

    sample_id: str
    image_group_key: str
    source_visual_identity: str
    supervision: ModelEvidenceSupervision
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    source_positions: tuple[int, ...]
    d_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in ("sample_id", "image_group_key", "source_visual_identity"):
            _require_non_empty_text(getattr(self, field_name), field_name=field_name)
        if not isinstance(self.supervision, ModelEvidenceSupervision):
            raise TypeError("supervision must be ModelEvidenceSupervision")
        if self.input_ids.dtype != torch.long or self.input_ids.ndim != 2:
            raise ValueError("readout input_ids must have shape [B,S] and dtype long")
        if self.input_ids.shape[0] != 1:
            raise ValueError("synthetic same-image readout requires row batch size one")
        batch, sequence = self.input_ids.shape
        if tuple(int(value) for value in self.input_ids[0].tolist()) != (
            self.supervision.model_token_ids
        ):
            raise ValueError("readout input_ids differ from model evidence supervision")
        if self.attention_mask.shape != (batch, sequence):
            raise ValueError("readout attention_mask must match input_ids")
        if self.position_ids.ndim not in {2, 3} or self.position_ids.shape[-2:] != (
            batch,
            sequence,
        ):
            raise ValueError("readout position_ids must end in [B,S]")
        if not self.source_positions or not self.d_positions:
            raise ValueError("source-image and D positions must both be non-empty")
        visual_blocks = self.supervision.visual_expansion_blocks
        if len(visual_blocks) != 2:
            raise ValueError(
                "representation readout requires exactly source-image and D visual blocks"
            )
        if (
            self.source_positions != visual_blocks[0]
            or self.d_positions != visual_blocks[1]
        ):
            raise ValueError(
                "source-image/D positions must preserve native placeholder order"
            )
        combined = (*self.source_positions, *self.d_positions)
        if tuple(sorted(set(combined))) != tuple(sorted(combined)):
            raise ValueError("source-image and D positions must be unique")
        if tuple(sorted(combined)) != self.supervision.visual_model_positions:
            raise ValueError(
                "source-image plus D positions must equal supervised visual positions"
            )
        if any(position < 0 or position >= sequence for position in combined):
            raise ValueError("readout visual position is outside the model sequence")


@dataclass(frozen=True, slots=True)
class RepresentationCandidateObservation:
    """One column-swapped, indivisible TGVF Adapter output bundle."""

    sample_id: str
    image_group_key: str
    source_visual_identity: str
    target_conditioning_provider: TargetConditioningProviderKind
    projection_identities: tuple[str, ...]
    visual: RepresentationVisualTensorBundle

    def __post_init__(self) -> None:
        for field_name in (
            "sample_id",
            "image_group_key",
            "source_visual_identity",
        ):
            _require_non_empty_text(getattr(self, field_name), field_name=field_name)
        if not isinstance(
            self.target_conditioning_provider, TargetConditioningProviderKind
        ):
            raise TypeError(
                "candidate target-conditioning provider must be an explicit kind"
            )
        if not self.projection_identities or any(
            not isinstance(identity, str) or not identity.strip()
            for identity in self.projection_identities
        ):
            raise ValueError("candidate projection identities must be non-empty")
        if len(self.projection_identities) != 1 + len(self.visual.deepstack):
            raise ValueError("candidate must identify main and every branch projection")


@dataclass(frozen=True, slots=True)
class SameImageReadoutGroup:
    """One image, K real rows/candidates, plus loss-excluded collective padding.

    ``collective_padding`` contains extra Adapter forwards needed only to keep
    composable-FSDP collective counts identical when data-parallel ranks own
    different permitted local K values.  Padding has no row identity and must
    never enter Matrix CE, L_gen, or metric denominators.
    """

    image_group_key: str
    source_visual_identity: str
    source_visual: RepresentationVisualTensorBundle
    rows: tuple[RepresentationReadoutRow, ...]
    candidates: tuple[RepresentationCandidateObservation, ...]
    collective_padding: tuple[RepresentationVisualTensorBundle, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_text(self.image_group_key, field_name="image_group_key")
        _require_non_empty_text(
            self.source_visual_identity, field_name="source_visual_identity"
        )
        if len(self.rows) < 2 or len(self.candidates) != len(self.rows):
            raise ValueError("same-image readout requires aligned K>=2 rows/candidates")
        if not isinstance(self.collective_padding, tuple):
            raise TypeError("collective_padding must be an immutable tuple")
        row_ids = tuple(row.sample_id for row in self.rows)
        candidate_ids = tuple(candidate.sample_id for candidate in self.candidates)
        if row_ids != candidate_ids or len(set(row_ids)) != len(row_ids):
            raise ValueError(
                "row/candidate order must define a unique diagonal identity"
            )
        providers = {
            candidate.target_conditioning_provider for candidate in self.candidates
        }
        projections = {candidate.projection_identities for candidate in self.candidates}
        if len(providers) != 1 or len(projections) != 1:
            raise ValueError(
                "one readout group cannot mix provider/projection identities"
            )
        for row in self.rows:
            if (
                row.image_group_key != self.image_group_key
                or row.source_visual_identity != self.source_visual_identity
            ):
                raise ValueError("every row must share the group source-image identity")
            if len(row.source_positions) != self.source_visual.main.shape[1]:
                raise ValueError(
                    "row source positions differ from source visual tokens"
                )
        for candidate in self.candidates:
            if (
                candidate.image_group_key != self.image_group_key
                or candidate.source_visual_identity != self.source_visual_identity
            ):
                raise ValueError(
                    "every candidate must share the group source-image identity"
                )
            if candidate.visual.branch_layers != self.source_visual.branch_layers:
                raise ValueError("source and candidate DeepStack layer order differs")
            if (
                candidate.visual.main.device != self.source_visual.main.device
                or candidate.visual.main.dtype != self.source_visual.main.dtype
                or candidate.visual.main.shape[-1] != self.source_visual.main.shape[-1]
            ):
                raise ValueError("source and candidate visual tensor contracts differ")
            for row in self.rows:
                if len(row.d_positions) != candidate.visual.main.shape[1]:
                    raise ValueError("row D positions differ from candidate D tokens")
        for padding in self.collective_padding:
            if not isinstance(padding, RepresentationVisualTensorBundle):
                raise TypeError("collective padding must contain visual bundles")
            if padding.branch_layers != self.source_visual.branch_layers:
                raise ValueError("collective padding DeepStack layer order differs")
            if (
                padding.main.device != self.source_visual.main.device
                or padding.main.dtype != self.source_visual.main.dtype
                or padding.main.shape[-1] != self.source_visual.main.shape[-1]
            ):
                raise ValueError("collective padding visual tensor contract differs")
            if any(len(row.d_positions) != padding.main.shape[1] for row in self.rows):
                raise ValueError("row D positions differ from collective padding tokens")

    @property
    def collective_candidate_count(self) -> int:
        """Number of Adapter forwards/backwards every rank must execute."""

        return len(self.candidates) + len(self.collective_padding)


@dataclass(frozen=True, slots=True)
class SameImageReadoutTerms:
    """Synthetic score matrix plus both independently reduced loss terms."""

    sample_ids: tuple[str, ...]
    score_matrix: torch.Tensor
    evidence_token_counts: torch.Tensor
    matrix_ce: SameImageMatrixCELossTerms
    l_gen: EvidenceReadabilityLossTerms

    def __post_init__(self) -> None:
        size = len(self.sample_ids)
        if size < 2 or self.score_matrix.shape != (size, size):
            raise ValueError("same-image score matrix must have shape [K,K], K>=2")
        if self.evidence_token_counts.shape != (size,):
            raise ValueError("evidence token counts must have shape [K]")
        if self.matrix_ce.valid_row_count != size or self.l_gen.sample_count != size:
            raise ValueError("same-image loss term counts must equal K")


def synthetic_same_image_layout_readout_terms(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: SameImageReadoutGroup,
) -> SameImageReadoutTerms:
    """Exercise K×K row/layout and atomic whole-D swaps on a synthetic mask.

    This function deliberately does not claim accepted representation readout:
    the post-D original-image key-block/Qwen mask contract is still open, so the
    current 2D attention mask lets evidence queries see the source image. It is
    only a typed injection/layout/sensitivity fixture. It also retains every
    cell graph and is not an accepted 8B execution schedule.
    """

    if not isinstance(family_adapter, QwenVLMFamilyAdapter):
        raise TypeError("family_adapter must be QwenVLMFamilyAdapter")
    if not isinstance(group, SameImageReadoutGroup):
        raise TypeError("group must be SameImageReadoutGroup")
    _assert_frozen_deterministic_model(model)
    family = family_adapter.capabilities.family
    if any(row.supervision.family != family for row in group.rows):
        raise ValueError("readout supervision belongs to a different Qwen family")
    if len(group.source_visual.deepstack) != (
        family_adapter.capabilities.deepstack_branch_count
    ):
        raise ValueError("source visual branch count differs from family capability")

    score_rows: list[torch.Tensor] = []
    diagonal_l_gen: list[torch.Tensor] = []
    evidence_counts: list[torch.Tensor] = []
    for row_index, row in enumerate(group.rows):
        cell_scores: list[torch.Tensor] = []
        for column_index, candidate in enumerate(group.candidates):
            request = _cell_request(group.source_visual, row, candidate.visual)
            result = family_adapter.forward_injected(model, request)
            labels = torch.tensor(
                row.supervision.labels,
                dtype=torch.long,
                device=result.logits.device,
            ).unsqueeze(0)
            losses = causal_evidence_losses(result.logits, labels)
            cell_scores.append(losses.per_sample_summed_log_likelihood[0])
            if row_index == column_index:
                diagonal_l_gen.append(losses.per_sample_token_mean_nll[0])
                evidence_counts.append(losses.valid_token_counts[0])
        score_rows.append(torch.stack(cell_scores))

    score_matrix = torch.stack(score_rows)
    l_gen_values = torch.stack(diagonal_l_gen)
    counts = torch.stack(evidence_counts)
    matrix_terms = same_image_matrix_ce_loss_terms((score_matrix,))
    l_gen_terms = EvidenceReadabilityLossTerms(
        numerator=l_gen_values.sum(),
        sample_count=len(group.rows),
    )
    return SameImageReadoutTerms(
        sample_ids=tuple(row.sample_id for row in group.rows),
        score_matrix=score_matrix,
        evidence_token_counts=counts,
        matrix_ce=matrix_terms,
        l_gen=l_gen_terms,
    )


def assert_frozen_deterministic_readout_model(model: object) -> None:
    """Require the base Qwen readout path to be frozen and deterministic."""

    _assert_frozen_deterministic_model(model)


def _cell_request(
    source: RepresentationVisualTensorBundle,
    row: RepresentationReadoutRow,
    candidate: RepresentationVisualTensorBundle,
) -> InjectedForwardRequest:
    source_block = InjectedVisualBlock(
        kind="source_image",
        positions=row.source_positions,
        embeddings=source.main,
        deepstack=source.deepstack,
        deepstack_positions=tuple(row.source_positions for _ in source.deepstack),
    )
    candidate_block = InjectedVisualBlock(
        kind="focused_d",
        positions=row.d_positions,
        embeddings=candidate.main,
        deepstack=candidate.deepstack,
        deepstack_positions=tuple(row.d_positions for _ in candidate.deepstack),
    )
    return InjectedForwardRequest(
        input_ids=row.input_ids,
        attention_mask=row.attention_mask,
        position_ids=row.position_ids,
        visual_blocks=(source_block, candidate_block),
        use_cache=False,
    )


def _assert_frozen_deterministic_model(model: object) -> None:
    if getattr(model, "training", False):
        raise ValueError("frozen Qwen readout model must be in eval mode")
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise TypeError("readout model must expose parameters()")
    language_model = resolve_language_model(model)
    lm_head = resolve_lm_head(model)
    if getattr(language_model, "training", False) or getattr(
        lm_head, "training", False
    ):
        raise ValueError("frozen Qwen language model and lm_head must be in eval mode")
    all_parameters: dict[int, torch.nn.Parameter] = {}
    for owner in (model, language_model, lm_head):
        owner_parameters = getattr(owner, "parameters", None)
        if not callable(owner_parameters):
            raise TypeError("Qwen model components must expose parameters()")
        for parameter in owner_parameters():
            all_parameters[id(parameter)] = parameter
    if any(parameter.requires_grad for parameter in all_parameters.values()):
        raise ValueError("all frozen Qwen readout parameters must disable gradients")


def _validate_visual_tensor(value: object, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 3 or value.shape[0] != 1 or min(value.shape[1:]) <= 0:
        raise ValueError(f"{name} must have shape [1,N,H] with N,H>0")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype")


def _require_non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
