"""Synthetic K×K layout/swap checks over same-image TGVF observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tgvf_rl.conditioning.base import (
    TargetConditioningProviderKind,
    _CanonicalInputIdsProof,
    _validate_canonical_input_ids_proof,
)
from tgvf_rl.qwen.base import (
    InjectedForwardRequest,
    InjectedVisualBlock,
    QwenVLMFamilyAdapter,
    resolve_language_model,
    resolve_lm_head,
)

from .losses import (
    EVIDENCE_IGNORE_INDEX,
    EvidenceReadabilityLossTerms,
    MatrixCEScoreMode,
    SameImageMatrixCELossTerms,
    causal_evidence_losses,
    matrix_ce_cell_scores,
    same_image_matrix_ce_loss_terms,
)
from .transcript import ModelEvidenceSupervision


@dataclass(frozen=True, slots=True)
class RepresentationReadoutLossSupervision:
    """One explicit override of the historical evidence-only loss view.

    ``labels`` remains aligned to the unchanged native readout transcript.  The
    two component position fields make a sparse answer-bearing objective
    auditable without changing the transcript, candidate observations, or the
    historical :class:`ModelEvidenceSupervision`.  The source-image query range
    is half open and describes which causal queries must not attend to original
    image keys while scoring these labels.
    """

    identity: str
    labels: tuple[int, ...]
    supervised_token_positions: tuple[int, ...]
    evidence_value_token_positions: tuple[int, ...]
    answer_token_positions: tuple[int, ...]
    source_image_block_query_start: int
    source_image_block_query_end: int

    def __post_init__(self) -> None:
        _require_non_empty_text(self.identity, field_name="loss supervision identity")
        if not isinstance(self.labels, tuple) or not self.labels:
            raise ValueError("loss supervision labels must be a non-empty tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.labels
        ):
            raise TypeError("loss supervision labels must contain integers")
        if any(value != EVIDENCE_IGNORE_INDEX and value < 0 for value in self.labels):
            raise ValueError(
                "loss supervision labels must be token IDs or the ignore index"
            )
        _validate_ordered_token_positions(
            self.supervised_token_positions,
            allow_empty=False,
            require_contiguous=False,
            name="supervised token",
        )
        _validate_ordered_token_positions(
            self.evidence_value_token_positions,
            allow_empty=True,
            require_contiguous=False,
            name="evidence-value token",
        )
        _validate_ordered_token_positions(
            self.answer_token_positions,
            allow_empty=False,
            require_contiguous=True,
            name="answer token",
        )
        if self.evidence_value_token_positions and (
            self.evidence_value_token_positions[-1] >= self.answer_token_positions[0]
        ):
            raise ValueError("evidence-value tokens must precede answer tokens")
        components = tuple(
            sorted(
                (
                    *self.evidence_value_token_positions,
                    *self.answer_token_positions,
                )
            )
        )
        if len(set(components)) != len(components):
            raise ValueError("loss supervision components must be disjoint")
        if components != self.supervised_token_positions:
            raise ValueError(
                "loss supervision component union must equal supervised positions"
            )
        realized = tuple(
            position
            for position, label in enumerate(self.labels)
            if label != EVIDENCE_IGNORE_INDEX
        )
        if realized != self.supervised_token_positions:
            raise ValueError(
                "loss supervision labels must own exactly the supervised positions"
            )
        if self.supervised_token_positions[0] == 0:
            raise ValueError("the first sequence token cannot receive a causal label")
        for field_name in (
            "source_image_block_query_start",
            "source_image_block_query_end",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if not (
            0
            <= self.source_image_block_query_start
            < self.source_image_block_query_end
            <= len(self.labels)
        ):
            raise ValueError(
                "source-image block query range must be non-empty and within labels"
            )
        if any(
            not (
                self.source_image_block_query_start
                <= position - 1
                < self.source_image_block_query_end
            )
            for position in self.supervised_token_positions
        ):
            raise ValueError(
                "source-image block query range must cover every supervised prediction"
            )


@dataclass(frozen=True, slots=True)
class RepresentationVisualTensorBundle:
    """Atomic main tensor plus every ordered DeepStack branch."""

    main: torch.Tensor
    deepstack: tuple[torch.Tensor, ...]
    branch_layers: tuple[int, ...]
    d_deepstack_active: bool = True

    def __post_init__(self) -> None:
        _validate_visual_tensor(self.main, name="main visual tensor")
        if type(self.d_deepstack_active) is not bool:
            raise TypeError("D-DeepStack activity must be explicit")
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
class RepresentationAttentionTensorBundle:
    """Detached target-to-visual attention for main and ordered branches.

    These tensors are diagnostics emitted by the TGVF Adapter.  They are not
    readout inputs and never contribute to an optimization objective.  The
    native group builder retains detached copies so an internal-evaluation
    runner can reproduce the historical attention-health reductions without
    rerunning the Adapter or guessing an internal module path.
    """

    main: torch.Tensor
    deepstack: tuple[torch.Tensor, ...]
    branch_layers: tuple[int, ...]
    d_deepstack_active: bool = True

    def __post_init__(self) -> None:
        _validate_attention_tensor(self.main, name="main attention tensor")
        if type(self.d_deepstack_active) is not bool:
            raise TypeError("attention D-DeepStack activity must be explicit")
        if len(self.deepstack) != len(self.branch_layers):
            raise ValueError("attention tensors and branch layers must align")
        if self.d_deepstack_active and not self.deepstack:
            raise ValueError("attention diagnostics require DeepStack branches")
        if not self.d_deepstack_active and (self.deepstack or self.branch_layers):
            raise ValueError("main-D-only attention cannot contain branch diagnostics")
        if len(set(self.branch_layers)) != len(self.branch_layers):
            raise ValueError("attention branch layers must be unique")
        for index, branch in enumerate(self.deepstack):
            _validate_attention_tensor(
                branch, name=f"DeepStack attention tensor {index}"
            )
            if branch.device != self.main.device or branch.dtype != self.main.dtype:
                raise ValueError(
                    "main and every DeepStack attention tensor must share device/dtype"
                )
        if any(tensor.requires_grad for tensor in (self.main, *self.deepstack)):
            raise ValueError("retained evaluation attention tensors must be detached")


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
    canonical_input_ids_proof: _CanonicalInputIdsProof | None = None
    loss_supervision: RepresentationReadoutLossSupervision | None = None

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
        self.assert_input_ids_authority()
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
        self._validate_loss_supervision(sequence)

    @property
    def loss_labels(self) -> tuple[int, ...]:
        """Effective labels, preserving historical RP66 behavior by default."""

        if self.loss_supervision is None:
            return self.supervision.labels
        return self.loss_supervision.labels

    @property
    def loss_supervised_token_positions(self) -> tuple[int, ...]:
        """Effective causal-label positions for loss accounting and diagnostics."""

        if self.loss_supervision is None:
            return self.supervision.evidence_token_positions
        return self.loss_supervision.supervised_token_positions

    @property
    def source_image_block_query_start(self) -> int:
        """Inclusive first query whose original-image keys are blocked."""

        if self.loss_supervision is None:
            return self.supervision.evidence_token_positions[0] - 1
        return self.loss_supervision.source_image_block_query_start

    @property
    def source_image_block_query_end(self) -> int:
        """Exclusive end of original-image key blocking for the loss view."""

        if self.loss_supervision is None:
            return self.supervision.evidence_token_positions[-1]
        return self.loss_supervision.source_image_block_query_end

    def _validate_loss_supervision(self, sequence: int) -> None:
        override = self.loss_supervision
        if override is None:
            return
        if not isinstance(override, RepresentationReadoutLossSupervision):
            raise TypeError(
                "loss_supervision must be RepresentationReadoutLossSupervision"
            )
        if len(override.labels) != sequence:
            raise ValueError(
                "loss supervision labels must align with readout input IDs"
            )
        supervised = set(override.supervised_token_positions)
        expected_labels = tuple(
            token_id if position in supervised else EVIDENCE_IGNORE_INDEX
            for position, token_id in enumerate(self.supervision.model_token_ids)
        )
        if override.labels != expected_labels:
            raise ValueError(
                "loss supervision labels differ from readout token IDs or positions"
            )
        evidence_positions = set(self.supervision.evidence_token_positions)
        if not set(override.evidence_value_token_positions).issubset(
            evidence_positions
        ):
            raise ValueError(
                "evidence-value supervision must stay inside the complete evidence span"
            )
        final_evidence = self.supervision.evidence_token_positions[-1]
        if override.answer_token_positions[0] <= final_evidence:
            raise ValueError(
                "answer supervision must follow the complete evidence span"
            )
        if set(override.supervised_token_positions).intersection(
            (*self.source_positions, *self.d_positions)
        ):
            raise ValueError("loss supervision cannot own visual model positions")
        expected_block_start = self.supervision.evidence_token_positions[0] - 1
        if override.source_image_block_query_start != expected_block_start:
            raise ValueError(
                "loss supervision must block source-image keys from the complete "
                "evidence prediction boundary"
            )
        if override.source_image_block_query_end != override.answer_token_positions[-1]:
            raise ValueError(
                "source-image block query end must equal the final answer token position"
            )

    def assert_input_ids_authority(self) -> None:
        """Revalidate the exact input tensor without reading bound CUDA content."""

        if self.input_ids.dtype != torch.long or self.input_ids.ndim != 2:
            raise ValueError("readout input_ids must have shape [B,S] and dtype long")
        if self.input_ids.shape[0] != 1:
            raise ValueError("synthetic same-image readout requires row batch size one")
        sequence = int(self.input_ids.shape[1])
        if self.canonical_input_ids_proof is None:
            realized_ids = tuple(int(value) for value in self.input_ids[0].tolist())
        else:
            rows, _digest = _validate_canonical_input_ids_proof(
                self.canonical_input_ids_proof,
                input_ids=self.input_ids,
                batched=True,
                batch_size=1,
                sequence_length=int(sequence),
            )
            realized_ids = rows[0]
        if realized_ids != self.supervision.model_token_ids:
            raise ValueError("readout input_ids differ from model evidence supervision")


@dataclass(frozen=True, slots=True)
class RepresentationCandidateObservation:
    """One column-swapped, indivisible TGVF Adapter output bundle."""

    sample_id: str
    image_group_key: str
    source_visual_identity: str
    target_conditioning_provider: TargetConditioningProviderKind
    projection_identities: tuple[str, ...]
    visual: RepresentationVisualTensorBundle
    image_grid_thw: tuple[int, int, int] | None = None
    attention: RepresentationAttentionTensorBundle | None = None

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
        expected_projection_count = 1 + (
            len(self.visual.deepstack) if self.visual.d_deepstack_active else 0
        )
        if len(self.projection_identities) != expected_projection_count:
            raise ValueError("candidate must identify main and every branch projection")
        if self.image_grid_thw is not None and (
            not isinstance(self.image_grid_thw, tuple)
            or len(self.image_grid_thw) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.image_grid_thw
            )
        ):
            raise ValueError("candidate image_grid_thw must contain positive integers")
        if self.attention is not None:
            if not isinstance(self.attention, RepresentationAttentionTensorBundle):
                raise TypeError("candidate attention must be a typed attention bundle")
            if self.attention.d_deepstack_active != self.visual.d_deepstack_active:
                raise ValueError("candidate visual and attention activity differs")
            if self.visual.d_deepstack_active and (
                self.attention.branch_layers != self.visual.branch_layers
            ):
                raise ValueError(
                    "candidate visual and attention branch layer order differs"
                )


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
        activity = {
            candidate.visual.d_deepstack_active for candidate in self.candidates
        }
        if len(providers) != 1 or len(projections) != 1 or len(activity) != 1:
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
                padding.d_deepstack_active
                != self.candidates[0].visual.d_deepstack_active
            ):
                raise ValueError("collective padding D-DeepStack activity differs")
            if (
                padding.main.device != self.source_visual.main.device
                or padding.main.dtype != self.source_visual.main.dtype
                or padding.main.shape[-1] != self.source_visual.main.shape[-1]
            ):
                raise ValueError("collective padding visual tensor contract differs")
            if any(len(row.d_positions) != padding.main.shape[1] for row in self.rows):
                raise ValueError(
                    "row D positions differ from collective padding tokens"
                )

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
    *,
    matrix_ce_mode: MatrixCEScoreMode = MatrixCEScoreMode.LEGACY_SUMMED_NLL,
    matrix_ce_temperature: float = 1.0,
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
                row.loss_labels,
                dtype=torch.long,
                device=result.logits.device,
            ).unsqueeze(0)
            losses = causal_evidence_losses(result.logits, labels)
            cell_scores.append(
                matrix_ce_cell_scores(
                    losses,
                    mode=matrix_ce_mode,
                    temperature=matrix_ce_temperature,
                )[0]
            )
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


def _validate_attention_tensor(value: object, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim not in {2, 3} or min(value.shape) <= 0:
        raise ValueError(f"{name} must have non-empty rank two or three shape")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype")


def _validate_ordered_token_positions(
    positions: object,
    *,
    allow_empty: bool,
    require_contiguous: bool,
    name: str,
) -> None:
    if not isinstance(positions, tuple):
        raise TypeError(f"{name} positions must be a tuple")
    if not positions and not allow_empty:
        raise ValueError(f"{name} positions must be non-empty")
    if any(
        isinstance(position, bool) or not isinstance(position, int)
        for position in positions
    ):
        raise TypeError(f"{name} positions must contain integers")
    if any(position < 0 for position in positions):
        raise ValueError(f"{name} positions cannot be negative")
    if tuple(sorted(set(positions))) != positions:
        raise ValueError(f"{name} positions must be ordered and unique")
    if (
        require_contiguous
        and positions
        and positions != tuple(range(positions[0], positions[-1] + 1))
    ):
        raise ValueError(f"{name} positions must be contiguous")


def _require_non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
