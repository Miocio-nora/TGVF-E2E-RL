"""Parallel group builder for the removable answer-utility experiment."""

from __future__ import annotations

from dataclasses import dataclass

from tgvf_rl.representation.adapter import TGVFAdapter
from tgvf_rl.representation.training.native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
    RepresentationPromptConfig,
)
from tgvf_rl.representation.training.readout import SameImageReadoutGroup
from tgvf_rl.representation.training.runtime import Qwen3RepresentationRuntime
from tgvf_rl.representation.training.schema import RepresentationTrainingSample

from .config import AnswerSupervisionView, AnswerUtilityExperimentProfile
from .controls import AnswerUtilityControlRow, build_same_image_answer_controls
from .supervision import (
    NativeAnswerSupervision,
    build_qwen3_clean_answer_supervision,
    build_qwen3_gold_evidence_answer_supervision,
)


@dataclass(frozen=True, slots=True)
class AnswerUtilityReadoutGroup:
    """The unchanged legacy group plus optional answer-only experiment views."""

    legacy: SameImageReadoutGroup
    answer_supervisions: tuple[NativeAnswerSupervision, ...]
    controls: tuple[AnswerUtilityControlRow, ...]
    supervision_view: AnswerSupervisionView
    requires_zero_control: bool
    requires_wrong_control: bool

    def __post_init__(self) -> None:
        if not isinstance(self.legacy, SameImageReadoutGroup):
            raise TypeError("answer utility group requires a legacy readout group")
        if not isinstance(self.supervision_view, AnswerSupervisionView):
            raise TypeError("answer supervision view must be explicit")
        if (
            type(self.requires_zero_control) is not bool
            or type(self.requires_wrong_control) is not bool
        ):
            raise TypeError("answer-control requirements must be explicit")
        sample_ids = tuple(row.sample_id for row in self.legacy.rows)
        if self.supervision_view is AnswerSupervisionView.NONE:
            if (
                self.answer_supervisions
                or self.controls
                or self.requires_zero_control
                or self.requires_wrong_control
            ):
                raise ValueError("E0/no-answer group cannot contain answer views")
            return
        if tuple(row.sample_id for row in self.answer_supervisions) != sample_ids:
            raise ValueError("answer supervision order differs from legacy rows")
        if tuple(row.sample_id for row in self.controls) != sample_ids:
            raise ValueError("answer control order differs from legacy rows")
        expected_kind = (
            "clean_d_only"
            if self.supervision_view is AnswerSupervisionView.CLEAN_D_ONLY
            else "gold_evidence"
        )
        if any(row.context_kind != expected_kind for row in self.answer_supervisions):
            raise ValueError("answer supervision kind differs from selected view")
        for row in self.controls:
            if (row.zero is not None) is not self.requires_zero_control:
                raise ValueError("zero-D control differs from selected profile")
            if (row.wrong is not None) is not self.requires_wrong_control:
                raise ValueError("wrong-D control differs from selected profile")


class Qwen3AnswerUtilityGroupBuilder:
    """Wrap the accepted group builder without changing its return contract."""

    def __init__(
        self,
        *,
        base_builder: Qwen3NativeRepresentationGroupBuilder,
        runtime: Qwen3RepresentationRuntime,
        prompt: RepresentationPromptConfig,
        profile: AnswerUtilityExperimentProfile,
    ) -> None:
        if not isinstance(base_builder, Qwen3NativeRepresentationGroupBuilder):
            raise TypeError(
                "base_builder must be Qwen3NativeRepresentationGroupBuilder"
            )
        if not isinstance(runtime, Qwen3RepresentationRuntime):
            raise TypeError("answer utility builder requires Qwen3 runtime")
        if base_builder.runtime is not runtime:
            raise ValueError("base and answer builders must share one runtime")
        if not isinstance(prompt, RepresentationPromptConfig):
            raise TypeError("answer utility builder requires an explicit prompt")
        if base_builder.prompt != prompt:
            raise ValueError("base and answer builders must share one prompt")
        if not isinstance(profile, AnswerUtilityExperimentProfile):
            raise TypeError("answer utility builder requires an experiment profile")
        self.base_builder = base_builder
        self.runtime = runtime
        self.prompt = prompt
        self.profile = profile
        self.supervision_view = profile.answer_supervision_view

    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        adapter: TGVFAdapter,
        *,
        collective_candidate_count: int,
    ) -> AnswerUtilityReadoutGroup:
        legacy = self.base_builder(
            samples,
            adapter,
            collective_candidate_count=collective_candidate_count,
        )
        if self.supervision_view is AnswerSupervisionView.NONE:
            return AnswerUtilityReadoutGroup(
                legacy=legacy,
                answer_supervisions=(),
                controls=(),
                supervision_view=self.supervision_view,
                requires_zero_control=False,
                requires_wrong_control=False,
            )
        controls = build_same_image_answer_controls(
            samples,
            legacy,
            requires_zero_control=self.profile.requires_zero_control,
            requires_wrong_control=self.profile.requires_wrong_control,
        )
        supervisions: list[NativeAnswerSupervision] = []
        for sample, row, candidate in zip(
            samples,
            legacy.rows,
            legacy.candidates,
            strict=True,
        ):
            grid = candidate.image_grid_thw
            if grid is None:
                raise ValueError("answer utility requires exact Qwen image geometry")
            if self.supervision_view is AnswerSupervisionView.CLEAN_D_ONLY:
                supervision = build_qwen3_clean_answer_supervision(
                    self.runtime,
                    sample,
                    self.prompt,
                    d_token_count=int(candidate.visual.main.shape[1]),
                    image_grid_thw=grid,
                    device=candidate.visual.main.device,
                )
            else:
                supervision = build_qwen3_gold_evidence_answer_supervision(
                    self.runtime,
                    sample,
                    self.prompt,
                    row=row,
                    image_grid_thw=grid,
                )
            supervisions.append(supervision)
        return AnswerUtilityReadoutGroup(
            legacy=legacy,
            answer_supervisions=tuple(supervisions),
            controls=controls,
            supervision_view=self.supervision_view,
            requires_zero_control=self.profile.requires_zero_control,
            requires_wrong_control=self.profile.requires_wrong_control,
        )


__all__ = [
    "AnswerUtilityReadoutGroup",
    "Qwen3AnswerUtilityGroupBuilder",
]
