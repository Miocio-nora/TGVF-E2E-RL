"""Pure SDPO contracts, teacher controller, and tensor objective."""

from .objective import SDPOInputs, compute_sdpo_loss
from .schema import (
    FullVocabularyDivergence,
    ImportanceSamplingMode,
    ImportanceSamplingSpec,
    SDPOLossMode,
    SDPOSpec,
    TeacherContextOverflow,
    TeacherContextRecord,
    TeacherContextSpec,
    TeacherControllerSpec,
    TeacherReplayArtifact,
    TeacherRegularization,
    TeacherTurnAlignment,
)
from .teacher import (
    TeacherController,
    build_teacher_context,
    parameter_mapping_sha256,
    replay_teacher_on_recorded_observations,
    validate_teacher_replay,
)

__all__ = [
    "FullVocabularyDivergence",
    "ImportanceSamplingMode",
    "ImportanceSamplingSpec",
    "SDPOInputs",
    "SDPOLossMode",
    "SDPOSpec",
    "TeacherContextOverflow",
    "TeacherContextRecord",
    "TeacherContextSpec",
    "TeacherController",
    "TeacherControllerSpec",
    "TeacherReplayArtifact",
    "TeacherRegularization",
    "TeacherTurnAlignment",
    "build_teacher_context",
    "compute_sdpo_loss",
    "parameter_mapping_sha256",
    "replay_teacher_on_recorded_observations",
    "validate_teacher_replay",
]
