"""Contracts and CPU-side input loading for answer-utility evaluation.

CUDA execution and scoring remain in :mod:`.runner`. This module owns the
integrity-bound candidate/input contracts and common manifest materialization.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.public_api_compat import rebind_public_class, rebind_public_function
from tgvf_rl.representation.training.config import (
    RepresentationTrainingConfig,
    load_representation_training_config,
)
from tgvf_rl.representation.training.data import load_retained_representation_jsonl
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)
from tgvf_rl.representation.training.evaluation_runner import (
    _validate_training_artifact_binding,
    load_representation_internal_evaluation_run_config,
)
from tgvf_rl.representation.training.oracle_d_utility import (
    OracleDUtilityArm,
    _normalize_eos_token_ids,
    _validate_selection,
)
from tgvf_rl.representation.training.post_training_evaluation import (
    file_sha256,
    load_internal_evaluation_group_manifest,
    materialize_internal_evaluation_groups,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample

from ..config import load_answer_utility_experiment_config
from ..run_config import load_answer_utility_run_config
from ..runner import _answer_utility_state_digest
from .input_audit import (
    _audit_completed_training_metrics_impl,
    _validate_private_source_bindings_impl,
)
from .input_artifact import (
    AnswerUtilityAdapterArtifact,
    load_answer_utility_adapter_artifact,
)
from .input_matching import (
    AnswerUtilityWrongImageDonor,
    _load_qwen_image_grid_contract,
    _require_sha256,
    build_answer_safe_wrong_mapping,
    build_same_target_wrong_image_mapping,
)


_PUBLIC_RUNNER_MODULE = (
    "tgvf_rl.representation.experiments.answer_utility.evaluation.runner"
)
ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION = "answer-utility-instruct-evaluation-v2"
ANSWER_UTILITY_EVALUATION_RECORD_SCHEMA_VERSION = (
    "answer-utility-instruct-evaluation-record-v2"
)
DEFAULT_INSTRUCT_EOS_TOKEN_IDS = (151645, 151643)


class AnswerUtilityEvaluationArm(str, Enum):
    """Declared held-out interventions, including image-plus-wrong D."""

    IMAGE_ONLY = OracleDUtilityArm.IMAGE_ONLY.value
    D_ONLY_ZERO = OracleDUtilityArm.TARGET_ZERO_D_ONLY.value
    D_ONLY_CORRECT = OracleDUtilityArm.CORRECT_D_ONLY.value
    D_ONLY_WRONG = OracleDUtilityArm.MATCHED_WRONG_D.value
    IMAGE_PLUS_ZERO = OracleDUtilityArm.IMAGE_TARGET_ZERO_D.value
    IMAGE_PLUS_CORRECT = OracleDUtilityArm.IMAGE_CORRECT_D.value
    IMAGE_PLUS_WRONG = "image_matched_wrong_D"
    IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE = "image_same_target_wrong_image_D"
    DIRECT_ZERO_REPLACEMENT = OracleDUtilityArm.DIRECT_ZERO_D_REPLACEMENT.value
    DIRECT_CORRECT_REPLACEMENT = OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT.value
    DIRECT_WRONG_REPLACEMENT = (
        OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT.value
    )


DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS = (
    AnswerUtilityEvaluationArm.IMAGE_ONLY,
    AnswerUtilityEvaluationArm.D_ONLY_ZERO,
    AnswerUtilityEvaluationArm.D_ONLY_CORRECT,
    AnswerUtilityEvaluationArm.D_ONLY_WRONG,
    AnswerUtilityEvaluationArm.IMAGE_PLUS_ZERO,
    AnswerUtilityEvaluationArm.IMAGE_PLUS_CORRECT,
    AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG,
)


@dataclass(frozen=True, slots=True)
class AnswerUtilityEvaluationCandidate:
    """One integrity-bound Adapter candidate evaluated by the common runner."""

    kind: Literal["private_formal500", "production_source"]
    candidate_id: str
    adapter_path: Path
    adapter_file_sha256: str
    adapter_state_sha256: str
    adapter_state: Mapping[str, torch.Tensor]
    global_step: int
    training_run_identity_sha256: str
    production_source_artifact_path: Path
    production_source_artifact_sha256: str
    production_source_manifest_sha256: str
    production_source_run_identity_sha256: str
    production_source_global_step: int
    protected_paths: tuple[Path, ...]
    private_run_id: str | None = None
    private_run_config_path: Path | None = None
    private_run_config_sha256: str | None = None
    private_experiment_config_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"private_formal500", "production_source"}:
            raise ValueError("unknown answer-utility evaluation candidate kind")
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("evaluation candidate ID must be non-empty text")
        for path, name in (
            (self.adapter_path, "candidate Adapter path"),
            (
                self.production_source_artifact_path,
                "production source Adapter path",
            ),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be absolute")
        for value, name in (
            (self.adapter_file_sha256, "candidate Adapter file SHA256"),
            (self.adapter_state_sha256, "candidate Adapter state SHA256"),
            (
                self.training_run_identity_sha256,
                "candidate training run identity SHA256",
            ),
            (
                self.production_source_artifact_sha256,
                "production source Adapter file SHA256",
            ),
            (
                self.production_source_manifest_sha256,
                "production source manifest SHA256",
            ),
            (
                self.production_source_run_identity_sha256,
                "production source run identity SHA256",
            ),
        ):
            _require_sha256(value, name=name)
        for value, name in (
            (self.global_step, "candidate global step"),
            (self.production_source_global_step, "production source global step"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.adapter_state or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in self.adapter_state.items()
        ):
            raise TypeError("evaluation candidate Adapter state is invalid")
        if not self.protected_paths or any(
            not isinstance(path, Path) or not path.is_absolute()
            for path in self.protected_paths
        ):
            raise ValueError("candidate protected paths must be absolute")
        private_fields = (
            self.private_run_id,
            self.private_run_config_path,
            self.private_run_config_sha256,
            self.private_experiment_config_sha256,
        )
        if self.kind == "private_formal500":
            if any(value is None for value in private_fields):
                raise ValueError("private candidate identity fields are incomplete")
            assert self.private_run_config_path is not None
            if not self.private_run_config_path.is_absolute():
                raise ValueError("private run config path must be absolute")
            _require_sha256(
                self.private_run_config_sha256,
                name="private run config SHA256",
            )
            _require_sha256(
                self.private_experiment_config_sha256,
                name="private experiment config SHA256",
            )
        elif any(value is not None for value in private_fields):
            raise ValueError(
                "production-source candidate cannot carry private identity"
            )


@dataclass(frozen=True, slots=True)
class AnswerUtilityEvaluationInputs:
    training: RepresentationTrainingConfig
    source_evaluation: Any
    candidate: AnswerUtilityEvaluationCandidate
    selected_groups: tuple[tuple[int, tuple[RepresentationTrainingSample, ...]], ...]
    wrong_source_by_sample_id: Mapping[str, str]
    same_target_wrong_image_by_group_key: Mapping[str, "AnswerUtilityWrongImageDonor"]
    wrong_image_pool_manifest_sha256: str | None
    data_manifest_sha256: str
    ordered_group_manifest_identity: str
    arms: tuple[AnswerUtilityEvaluationArm, ...]
    max_new_tokens: int
    eos_token_ids: tuple[int, ...]
    decode_mode: Literal["cached", "no_cache"]
    arm_batch_size: int
    group_start: int
    group_limit: int | None
    shard_index: int
    shard_count: int


@dataclass(frozen=True, slots=True)
class _InputLoaderBindings:
    normalize_evaluation_arms: Callable[
        [Sequence[AnswerUtilityEvaluationArm | str]],
        tuple[AnswerUtilityEvaluationArm, ...],
    ]
    load_run_config: Callable[[str | Path], Any]
    load_experiment_config: Callable[[str | Path], Any]
    load_training_config: Callable[..., Any]
    load_source_evaluation_config: Callable[[str | Path], Any]
    require_instruct_training: Callable[[RepresentationTrainingConfig], None]
    validate_private_source_bindings: Callable[[Any, Any, Any], None]
    load_validated_production_export: Callable[[Any, Any], Any]
    load_adapter_artifact: Callable[[str | Path], AnswerUtilityAdapterArtifact]
    audit_completed_training_metrics: Callable[[Any, str, Any], None]
    materialize_common_inputs: Callable[..., AnswerUtilityEvaluationInputs]


def _load_private_inputs_impl(
    run_config_path: str | Path,
    source_evaluation_config_path: str | Path,
    *,
    arms: Sequence[AnswerUtilityEvaluationArm | str],
    max_new_tokens: int | None,
    eos_token_ids: Sequence[int] | None,
    decode_mode: Literal["cached", "no_cache"],
    arm_batch_size: int = 1,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
    bindings: _InputLoaderBindings,
) -> AnswerUtilityEvaluationInputs:
    selected_arms = bindings.normalize_evaluation_arms(arms)
    run = bindings.load_run_config(run_config_path)
    experiment = bindings.load_experiment_config(run.experiment_config_path)
    if run.run_id != experiment.run_id:
        raise ValueError("run and experiment run_id values differ")
    if run.experiment_config_sha256 != experiment.source_toml_sha256:
        raise ValueError("run sidecar points at another experiment config")
    training = bindings.load_training_config(experiment.base_training_config_path)
    bindings.require_instruct_training(training)
    source_evaluation = bindings.load_source_evaluation_config(
        source_evaluation_config_path
    )
    bindings.validate_private_source_bindings(run, experiment, source_evaluation)
    bindings.load_validated_production_export(training, source_evaluation)
    artifact = bindings.load_adapter_artifact(run.final_artifact_path)
    bindings.audit_completed_training_metrics(run, experiment.variant.value, artifact)
    if artifact.run_identity_sha256 == run.source_artifact.expected_run_identity_sha256:
        raise ValueError(
            "private artifact incorrectly identifies the production source run"
        )
    if artifact.global_step != run.target_optimizer_steps:
        raise ValueError("private artifact is not at the formal target step")
    if artifact.source_artifact_sha256 != run.source_artifact.file_sha256:
        raise ValueError("private artifact source production Adapter differs")
    if artifact.experiment_config_sha256 != run.experiment_config_sha256:
        raise ValueError("private artifact experiment config differs")
    candidate = AnswerUtilityEvaluationCandidate(
        kind="private_formal500",
        candidate_id=run.run_id,
        adapter_path=artifact.path,
        adapter_file_sha256=artifact.file_sha256,
        adapter_state_sha256=artifact.adapter_state_sha256,
        adapter_state=artifact.adapter_state,
        global_step=artifact.global_step,
        training_run_identity_sha256=artifact.run_identity_sha256,
        production_source_artifact_path=source_evaluation.artifact_path,
        production_source_artifact_sha256=source_evaluation.artifact_file_sha256,
        production_source_manifest_sha256=source_evaluation.artifact_manifest_sha256,
        production_source_run_identity_sha256=(
            source_evaluation.expected_run_identity_sha256
        ),
        production_source_global_step=source_evaluation.expected_global_step,
        protected_paths=(
            run.output_directory.resolve(),
            source_evaluation.artifact_path.parent.resolve(),
        ),
        private_run_id=run.run_id,
        private_run_config_path=run.source_path,
        private_run_config_sha256=run.source_toml_sha256,
        private_experiment_config_sha256=run.experiment_config_sha256,
    )
    return bindings.materialize_common_inputs(
        training=training,
        source_evaluation=source_evaluation,
        candidate=candidate,
        selected_arms=selected_arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def _load_production_source_inputs_impl(
    source_evaluation_config_path: str | Path,
    *,
    arms: Sequence[AnswerUtilityEvaluationArm | str],
    max_new_tokens: int | None,
    eos_token_ids: Sequence[int] | None,
    decode_mode: Literal["cached", "no_cache"],
    arm_batch_size: int = 1,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
    bindings: _InputLoaderBindings,
) -> AnswerUtilityEvaluationInputs:
    selected_arms = bindings.normalize_evaluation_arms(arms)
    source_evaluation = bindings.load_source_evaluation_config(
        source_evaluation_config_path
    )
    training = bindings.load_training_config(
        source_evaluation.training_config_path,
        allow_existing_post_training_report=True,
    )
    bindings.require_instruct_training(training)
    export = bindings.load_validated_production_export(training, source_evaluation)
    if export.state is None:  # Defensive: the loader already requires writer state.
        raise ValueError("production source Adapter export has no owned state")
    adapter_state_sha256 = _answer_utility_state_digest(export.state)
    candidate = AnswerUtilityEvaluationCandidate(
        kind="production_source",
        candidate_id=(
            f"{training.run_id}-PRODUCTION-SOURCE-STEP"
            f"{source_evaluation.expected_global_step}"
        ),
        adapter_path=source_evaluation.artifact_path,
        adapter_file_sha256=source_evaluation.artifact_file_sha256,
        adapter_state_sha256=adapter_state_sha256,
        adapter_state=dict(export.state),
        global_step=source_evaluation.expected_global_step,
        training_run_identity_sha256=source_evaluation.expected_run_identity_sha256,
        production_source_artifact_path=source_evaluation.artifact_path,
        production_source_artifact_sha256=source_evaluation.artifact_file_sha256,
        production_source_manifest_sha256=source_evaluation.artifact_manifest_sha256,
        production_source_run_identity_sha256=(
            source_evaluation.expected_run_identity_sha256
        ),
        production_source_global_step=source_evaluation.expected_global_step,
        protected_paths=(source_evaluation.artifact_path.parent.resolve(),),
    )
    return bindings.materialize_common_inputs(
        training=training,
        source_evaluation=source_evaluation,
        candidate=candidate,
        selected_arms=selected_arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def _materialize_common_inputs_impl(
    *,
    training: RepresentationTrainingConfig,
    source_evaluation: Any,
    candidate: AnswerUtilityEvaluationCandidate,
    selected_arms: tuple[AnswerUtilityEvaluationArm, ...],
    max_new_tokens: int | None,
    eos_token_ids: Sequence[int] | None,
    decode_mode: Literal["cached", "no_cache"],
    arm_batch_size: int,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> AnswerUtilityEvaluationInputs:
    selected_max_new_tokens = (
        source_evaluation.evaluation.max_new_tokens
        if max_new_tokens is None
        else max_new_tokens
    )
    selected_eos = _normalize_eos_token_ids(
        DEFAULT_INSTRUCT_EOS_TOKEN_IDS if eos_token_ids is None else eos_token_ids
    )
    _validate_selection(
        max_new_tokens=selected_max_new_tokens,
        eos_token_ids=selected_eos,
        decode_mode=decode_mode,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    selected_arm_batch_size = _normalize_arm_batch_size(arm_batch_size)
    data = load_retained_representation_jsonl(
        source_evaluation.evaluation_data_path,
        expected_source_sha256=source_evaluation.evaluation_data_source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    ordered_manifest_path = source_evaluation.evaluation.ordered_group_manifest_path
    ordered_manifest_sha256 = source_evaluation.evaluation.ordered_group_manifest_sha256
    assert ordered_manifest_path is not None and ordered_manifest_sha256 is not None
    if file_sha256(ordered_manifest_path) != ordered_manifest_sha256:
        raise ValueError("ordered first200 group manifest SHA256 mismatch")
    manifest = load_internal_evaluation_group_manifest(ordered_manifest_path)
    all_groups = materialize_internal_evaluation_groups(
        manifest,
        data_manifest_sha256=data.manifest.manifest_sha256,
        samples=data.samples,
    )
    enumerated = tuple(enumerate(all_groups))
    after_start = enumerated[group_start:]
    sharded = tuple(
        group
        for index, group in enumerate(after_start)
        if index % shard_count == shard_index
    )
    selected_groups = sharded if group_limit is None else sharded[:group_limit]
    if not selected_groups:
        raise ValueError("answer-utility evaluation selection contains no image group")
    wrong_mapping = build_answer_safe_wrong_mapping(selected_groups)
    wrong_image_mapping: Mapping[str, AnswerUtilityWrongImageDonor] = {}
    wrong_image_pool_manifest_sha256: str | None = None
    if _has_same_target_wrong_image_arm(selected_arms):
        donor_data = load_retained_representation_jsonl(
            training.data.train.jsonl_path,
            expected_source_sha256=training.data.train.source_sha256,
            warn_on_leakage=training.data.warn_on_target_leakage,
        )
        wrong_image_mapping = build_same_target_wrong_image_mapping(
            selected_groups,
            donor_data.samples,
            grid_contract=_load_qwen_image_grid_contract(training),
            random_seed=source_evaluation.evaluation.random_seed,
        )
        wrong_image_pool_manifest_sha256 = donor_data.manifest.manifest_sha256
    return AnswerUtilityEvaluationInputs(
        training=training,
        source_evaluation=source_evaluation,
        candidate=candidate,
        selected_groups=selected_groups,
        wrong_source_by_sample_id=wrong_mapping,
        same_target_wrong_image_by_group_key=wrong_image_mapping,
        wrong_image_pool_manifest_sha256=wrong_image_pool_manifest_sha256,
        data_manifest_sha256=data.manifest.manifest_sha256,
        ordered_group_manifest_identity=manifest.identity,
        arms=selected_arms,
        max_new_tokens=selected_max_new_tokens,
        eos_token_ids=selected_eos,
        decode_mode=decode_mode,
        arm_batch_size=selected_arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def _load_validated_production_export_impl(
    training: RepresentationTrainingConfig,
    source_evaluation: Any,
    *,
    file_sha256_fn: Callable[[str | Path], str] = file_sha256,
    load_export_fn: Callable[[str | Path], Any] = (
        load_rank_zero_adapter_owned_state_export
    ),
    state_digest_fn: Callable[[Any], str] = state_digest,
    validate_training_artifact_binding_fn: Callable[[Any, Any], None] = (
        _validate_training_artifact_binding
    ),
) -> Any:
    """Load the source export after checking every evaluation-side binding."""

    if source_evaluation.evaluation_data_path is None or (
        source_evaluation.evaluation_data_source_sha256 is None
    ):
        raise ValueError("held-out evaluation requires an explicit test split")
    if source_evaluation.training_config_path != training.source_path:
        raise ValueError("evaluation source identifies another training config")
    if source_evaluation.training_config_sha256 != training.source_toml_sha256:
        raise ValueError("evaluation/training config SHA256 mismatch")
    if (
        source_evaluation.evaluation_data_path != training.data.validation.jsonl_path
        or (
            source_evaluation.evaluation_data_source_sha256
            != training.data.validation.source_sha256
        )
    ):
        raise ValueError("evaluation source is not the bound RP66 held-out split")
    for path, expected, name in (
        (
            source_evaluation.training_config_path,
            source_evaluation.training_config_sha256,
            "base training config",
        ),
        (
            source_evaluation.artifact_path,
            source_evaluation.artifact_file_sha256,
            "production source Adapter",
        ),
    ):
        if file_sha256_fn(path) != expected:
            raise ValueError(f"{name} SHA256 mismatch")
    export = load_export_fn(source_evaluation.artifact_path)
    if state_digest_fn(export.manifest) != source_evaluation.artifact_manifest_sha256:
        raise ValueError("production source Adapter manifest SHA256 mismatch")
    if (
        export.manifest.run_identity_sha256
        != source_evaluation.expected_run_identity_sha256
        or export.manifest.run_identity.identity_sha256
        != source_evaluation.expected_run_identity_sha256
        or export.manifest.global_step != source_evaluation.expected_global_step
    ):
        raise ValueError("production source Adapter identity/step mismatch")
    if export.state is None:
        raise ValueError("production source Adapter export has no owned state")
    validate_training_artifact_binding_fn(training, export.manifest.run_identity)
    return export


def _require_instruct_training_impl(
    training: RepresentationTrainingConfig,
) -> None:
    if training.model.model_name != "Qwen3-VL-8B-Instruct" or (
        training.model.local_path.name != "Qwen3-VL-8B-Instruct"
    ):
        raise ValueError(
            "answer-utility held-out evaluation is pinned to Qwen3-VL-8B-Instruct"
        )


def _default_input_loader_bindings() -> _InputLoaderBindings:
    return _InputLoaderBindings(
        normalize_evaluation_arms=_normalize_evaluation_arms,
        load_run_config=load_answer_utility_run_config,
        load_experiment_config=load_answer_utility_experiment_config,
        load_training_config=load_representation_training_config,
        load_source_evaluation_config=(
            load_representation_internal_evaluation_run_config
        ),
        require_instruct_training=_require_instruct_training_impl,
        validate_private_source_bindings=_validate_private_source_bindings_impl,
        load_validated_production_export=_load_validated_production_export_impl,
        load_adapter_artifact=load_answer_utility_adapter_artifact,
        audit_completed_training_metrics=_audit_completed_training_metrics_impl,
        materialize_common_inputs=_materialize_common_inputs_impl,
    )


def _normalize_evaluation_arms(
    arms: Sequence[AnswerUtilityEvaluationArm | str],
) -> tuple[AnswerUtilityEvaluationArm, ...]:
    if isinstance(arms, (str, bytes)):
        raise TypeError("arms must be a sequence")
    try:
        selected = tuple(
            arm
            if isinstance(arm, AnswerUtilityEvaluationArm)
            else AnswerUtilityEvaluationArm(arm)
            for arm in arms
        )
    except ValueError as error:
        raise ValueError("unknown answer-utility evaluation arm") from error
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("evaluation arms must be non-empty and unique")
    return selected


def _normalize_arm_batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("arm_batch_size must be a positive integer")
    return value


def _has_same_target_wrong_image_arm(
    arms: Sequence[AnswerUtilityEvaluationArm],
) -> bool:
    return AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE in arms


def _assert_evaluation_output_isolated(
    output: Path,
    inputs: AnswerUtilityEvaluationInputs,
) -> None:
    protected = (
        *inputs.candidate.protected_paths,
        inputs.training.checkpoint.directory.resolve(),
        inputs.training.output.metrics_jsonl_path.parent.resolve(),
    )
    for directory in protected:
        if (
            output == directory
            or directory in output.parents
            or output in directory.parents
        ):
            raise ValueError(
                "evaluation output must not overlap training/artifact directories"
            )


for _contract_type in (
    AnswerUtilityEvaluationArm,
    AnswerUtilityAdapterArtifact,
    AnswerUtilityEvaluationCandidate,
    AnswerUtilityEvaluationInputs,
):
    rebind_public_class(
        _contract_type,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNNER_MODULE,
    )
for _helper in (
    load_answer_utility_adapter_artifact,
    _normalize_evaluation_arms,
    _normalize_arm_batch_size,
    _has_same_target_wrong_image_arm,
    _assert_evaluation_output_isolated,
):
    rebind_public_function(
        _helper,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNNER_MODULE,
    )
del _contract_type, _helper


__all__ = [
    "ANSWER_UTILITY_EVALUATION_RECORD_SCHEMA_VERSION",
    "ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION",
    "DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS",
    "DEFAULT_INSTRUCT_EOS_TOKEN_IDS",
    "AnswerUtilityAdapterArtifact",
    "AnswerUtilityEvaluationArm",
    "AnswerUtilityEvaluationCandidate",
    "AnswerUtilityEvaluationInputs",
    "_InputLoaderBindings",
    "_assert_evaluation_output_isolated",
    "_audit_completed_training_metrics_impl",
    "_default_input_loader_bindings",
    "_has_same_target_wrong_image_arm",
    "_load_private_inputs_impl",
    "_load_production_source_inputs_impl",
    "_load_validated_production_export_impl",
    "_materialize_common_inputs_impl",
    "_normalize_evaluation_arms",
    "_require_instruct_training_impl",
    "_validate_private_source_bindings_impl",
    "load_answer_utility_adapter_artifact",
]
