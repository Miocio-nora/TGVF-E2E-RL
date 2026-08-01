"""Strict, removable Instruct evaluation of answer-utility Adapter artifacts.

The production oracle-D evaluator owns the native prompt rendering, visual
injection, greedy generation, and durable ledger.  This module supplies the
experiment-private artifact loader, reader prompt, scorer, and RP66 Instruct
bindings that the production evaluator deliberately does not accept.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Literal
import unicodedata

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.config import (
    RepresentationTrainingConfig,
    load_representation_training_config,
)
from tgvf_rl.representation.training.data import load_retained_representation_jsonl
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)
from tgvf_rl.representation.training.evaluation_runner import (
    _enable_determinism,
    _load_qwen,
    _load_rgb_image,
    _seed_current_process,
    _torch_dtype,
    _validate_training_artifact_binding,
    load_representation_internal_evaluation_run_config,
)
from tgvf_rl.representation.training.native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
)
from tgvf_rl.representation.training.oracle_d_utility import (
    ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION,
    OracleArmContext,
    OracleBatchCompatibilityError,
    OracleDUtilityArm,
    OracleGeneratedAnswer,
    _OracleRunLedger,
    _arm_contract,
    _atomic_write_json,
    _injected_block,
    _model_eos_token_ids,
    _normalize_eos_token_ids,
    _paired_summary,
    _require_single_gpu_environment,
    _validate_selection,
    greedy_oracle_answer,
    greedy_oracle_answers_batched,
    materialize_oracle_group_visuals,
    prepare_oracle_arm_context,
    split_oracle_d_utility_sample,
    verify_image_only_injected_native_parity,
)
from tgvf_rl.representation.training.post_training_evaluation import (
    file_sha256,
    load_internal_evaluation_group_manifest,
    materialize_internal_evaluation_groups,
)
from tgvf_rl.representation.training.runtime import (
    create_qwen3_representation_runtime,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample

from ..config import load_answer_utility_experiment_config
from ..controls import _normalized_answer_identity
from ..run_config import AnswerUtilityRunConfig, load_answer_utility_run_config
from ..runner import (
    ANSWER_UTILITY_ARTIFACT_SCHEMA_VERSION,
    ANSWER_UTILITY_METRICS_SCHEMA_VERSION,
    _answer_utility_state_digest,
)
from .scoring import (
    INSTRUCT_READER_CONTRACT_VERSION,
    INSTRUCT_READER_INSTRUCTION,
    INSTRUCT_SCORING_CONTRACT_VERSION,
    reader_question,
    score_instruct_generated_answer,
)


ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION = "answer-utility-instruct-evaluation-v2"
ANSWER_UTILITY_EVALUATION_RECORD_SCHEMA_VERSION = (
    "answer-utility-instruct-evaluation-record-v2"
)
DEFAULT_INSTRUCT_EOS_TOKEN_IDS = (151645, 151643)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "run_identity_sha256",
        "global_step",
        "source_artifact_sha256",
        "experiment_config_sha256",
        "adapter_state_sha256",
        "adapter_state",
    }
)
_SPACE = re.compile(r"\s+")


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
class AnswerUtilityAdapterArtifact:
    path: Path
    file_sha256: str
    run_identity_sha256: str
    global_step: int
    source_artifact_sha256: str
    experiment_config_sha256: str
    adapter_state_sha256: str
    adapter_state: Mapping[str, torch.Tensor]


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
class AnswerUtilityWrongImageDonor:
    """One identity-bound image donor for a same-target wrong-image D arm."""

    anchor_image_group_key: str
    anchor_image_sha256: str
    donor_sample_id: str
    donor_sample_content_sha256: str
    donor_image_group_key: str
    donor_image: str
    donor_image_sha256: str
    image_grid_thw: tuple[int, int, int]
    match_tier: Literal[
        "exact_grid_same_source_dataset",
        "exact_grid_same_source_profile",
        "exact_grid_cross_domain",
    ]

    def __post_init__(self) -> None:
        for value, name in (
            (self.anchor_image_group_key, "anchor image group key"),
            (self.donor_sample_id, "donor sample ID"),
            (self.donor_image_group_key, "donor image group key"),
            (self.donor_image, "donor image path"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        for value, name in (
            (self.anchor_image_sha256, "anchor image SHA256"),
            (self.donor_sample_content_sha256, "donor sample content SHA256"),
            (self.donor_image_sha256, "donor image SHA256"),
        ):
            _require_sha256(value, name=name)
        if self.anchor_image_group_key == self.donor_image_group_key:
            raise ValueError("wrong-image donor must use a distinct image group")
        if self.anchor_image_sha256 == self.donor_image_sha256:
            raise ValueError("wrong-image donor must use distinct image bytes")
        if (
            not isinstance(self.image_grid_thw, tuple)
            or len(self.image_grid_thw) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.image_grid_thw
            )
        ):
            raise ValueError("wrong-image donor grid must be positive integer THW")
        if self.match_tier not in {
            "exact_grid_same_source_dataset",
            "exact_grid_same_source_profile",
            "exact_grid_cross_domain",
        }:
            raise ValueError("unknown wrong-image donor match tier")


@dataclass(frozen=True, slots=True)
class _QwenImageGridContract:
    patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.patch_size, "patch size"),
            (self.merge_size, "merge size"),
            (self.min_pixels, "minimum pixels"),
            (self.max_pixels, "maximum pixels"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Qwen image-grid {name} must be positive")
        if self.max_pixels < self.min_pixels:
            raise ValueError("Qwen image-grid maximum is below its minimum")


def load_answer_utility_adapter_artifact(
    path: str | Path,
) -> AnswerUtilityAdapterArtifact:
    """Load exactly the private seven-key final-artifact schema."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"answer-utility artifact is missing: {source}")
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError("answer-utility artifact must be a mapping")
    if set(value) != _ARTIFACT_FIELDS:
        raise ValueError("answer-utility artifact must have exactly seven fields")
    if value["schema_version"] != ANSWER_UTILITY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("answer-utility artifact schema mismatch")
    state = value["adapter_state"]
    if (
        not isinstance(state, Mapping)
        or not state
        or any(
            not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
            for name, tensor in state.items()
        )
    ):
        raise TypeError("answer-utility artifact Adapter state is invalid")
    observed_state_sha256 = _answer_utility_state_digest(state)
    if value["adapter_state_sha256"] != observed_state_sha256:
        raise ValueError("answer-utility artifact state digest mismatch")
    for name in (
        "run_identity_sha256",
        "source_artifact_sha256",
        "experiment_config_sha256",
        "adapter_state_sha256",
    ):
        _require_sha256(value[name], name=f"artifact {name}")
    global_step = value["global_step"]
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("answer-utility artifact global_step must be an integer")
    if global_step <= 0:
        raise ValueError("answer-utility artifact global_step must be positive")
    return AnswerUtilityAdapterArtifact(
        path=source,
        file_sha256=file_sha256(source),
        run_identity_sha256=value["run_identity_sha256"],
        global_step=global_step,
        source_artifact_sha256=value["source_artifact_sha256"],
        experiment_config_sha256=value["experiment_config_sha256"],
        adapter_state_sha256=value["adapter_state_sha256"],
        adapter_state=dict(state),
    )


def validate_answer_utility_evaluation(
    run_config_path: str | Path,
    source_evaluation_config_path: str | Path,
    *,
    arms: Sequence[AnswerUtilityEvaluationArm | str] = (
        DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS
    ),
    max_new_tokens: int | None = None,
    eos_token_ids: Sequence[int] | None = None,
    decode_mode: Literal["cached", "no_cache"] = "cached",
    arm_batch_size: int = 1,
    group_start: int = 0,
    group_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    """Perform the complete binding/materialization preflight without CUDA."""

    inputs = _load_private_inputs(
        run_config_path,
        source_evaluation_config_path,
        arms=arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return _validated_payload(inputs)


def validate_production_source_answer_utility_evaluation(
    source_evaluation_config_path: str | Path,
    *,
    arms: Sequence[AnswerUtilityEvaluationArm | str] = (
        DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS
    ),
    max_new_tokens: int | None = None,
    eos_token_ids: Sequence[int] | None = None,
    decode_mode: Literal["cached", "no_cache"] = "cached",
    arm_batch_size: int = 1,
    group_start: int = 0,
    group_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    """Preflight the exact production Adapter as the E0 comparison candidate."""

    inputs = _load_production_source_inputs(
        source_evaluation_config_path,
        arms=arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return _validated_payload(inputs)


def _validated_payload(inputs: AnswerUtilityEvaluationInputs) -> dict[str, object]:
    samples = tuple(
        sample for _ordinal, group in inputs.selected_groups for sample in group
    )
    candidate = inputs.candidate
    payload: dict[str, object] = {
        "schema_version": ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION,
        "status": "validated",
        "candidate_kind": candidate.kind,
        "candidate_id": candidate.candidate_id,
        "training_model_name": inputs.training.model.model_name,
        "training_model_path": str(inputs.training.model.local_path),
        "candidate_training_run_identity_sha256": (
            candidate.training_run_identity_sha256
        ),
        "candidate_artifact_file_sha256": candidate.adapter_file_sha256,
        "candidate_adapter_state_sha256": candidate.adapter_state_sha256,
        "candidate_global_step": candidate.global_step,
        "private_run_id": candidate.private_run_id,
        "production_source_artifact_sha256": (
            candidate.production_source_artifact_sha256
        ),
        "production_source_manifest_sha256": (
            candidate.production_source_manifest_sha256
        ),
        "production_source_run_identity_sha256": (
            candidate.production_source_run_identity_sha256
        ),
        "production_source_global_step": candidate.production_source_global_step,
        "evaluation_data_manifest_sha256": inputs.data_manifest_sha256,
        "ordered_group_manifest_identity": inputs.ordered_group_manifest_identity,
        "selected_group_ordinals": [
            ordinal for ordinal, _group in inputs.selected_groups
        ],
        "selected_group_count": len(inputs.selected_groups),
        "selected_sample_count": len(samples),
        "arms": [arm.value for arm in inputs.arms],
        "answer_safe_wrong_mapping_count": len(inputs.wrong_source_by_sample_id),
        "max_new_tokens": inputs.max_new_tokens,
        "eos_token_ids": list(inputs.eos_token_ids),
        "decode_mode": inputs.decode_mode,
        "arm_batch_size": inputs.arm_batch_size,
        "decoder_implementation": _decoder_implementation(inputs),
        "shard_index": inputs.shard_index,
        "shard_count": inputs.shard_count,
    }
    if _has_same_target_wrong_image_arm(inputs.arms):
        payload.update(
            {
                "same_target_wrong_image_mapping_count": len(
                    inputs.same_target_wrong_image_by_group_key
                ),
                "wrong_image_pool_manifest_sha256": (
                    inputs.wrong_image_pool_manifest_sha256
                ),
                "wrong_image_pool_source_sha256": (
                    inputs.training.data.train.source_sha256
                ),
                "wrong_image_match_tiers": dict(
                    _wrong_image_match_tier_counts(
                        inputs.same_target_wrong_image_by_group_key
                    )
                ),
            }
        )
    return payload


def run_answer_utility_evaluation(
    run_config_path: str | Path,
    source_evaluation_config_path: str | Path,
    *,
    output_root: str | Path,
    arms: Sequence[AnswerUtilityEvaluationArm | str] = (
        DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS
    ),
    max_new_tokens: int | None = None,
    eos_token_ids: Sequence[int] | None = None,
    decode_mode: Literal["cached", "no_cache"] = "cached",
    arm_batch_size: int = 1,
    group_start: int = 0,
    group_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    """Run or exactly resume one manifest-ordered single-GPU evaluation shard."""

    inputs = _load_private_inputs(
        run_config_path,
        source_evaluation_config_path,
        arms=arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return _run_loaded_inputs(inputs, output_root=output_root)


def run_production_source_answer_utility_evaluation(
    source_evaluation_config_path: str | Path,
    *,
    output_root: str | Path,
    arms: Sequence[AnswerUtilityEvaluationArm | str] = (
        DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS
    ),
    max_new_tokens: int | None = None,
    eos_token_ids: Sequence[int] | None = None,
    decode_mode: Literal["cached", "no_cache"] = "cached",
    arm_batch_size: int = 1,
    group_start: int = 0,
    group_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    """Run the exact production Adapter through the common E0 evaluation path."""

    inputs = _load_production_source_inputs(
        source_evaluation_config_path,
        arms=arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return _run_loaded_inputs(inputs, output_root=output_root)


def _run_loaded_inputs(
    inputs: AnswerUtilityEvaluationInputs, *, output_root: str | Path
) -> dict[str, object]:
    output = Path(output_root).expanduser().resolve()
    _assert_evaluation_output_isolated(output, inputs)
    model_truth_rows = tuple(
        split_oracle_d_utility_sample(sample)
        for _ordinal, group in inputs.selected_groups
        for sample in group
    )
    model_inputs = tuple(model for model, _truth in model_truth_rows)
    identity_payload = _evaluation_identity_payload(inputs, model_inputs)
    ledger = _OracleRunLedger(
        output,
        identity_payload=identity_payload,
        expected_keys=tuple(
            (row.sample_id, arm.value) for row in model_inputs for arm in inputs.arms
        ),
    )
    with ledger.locked():
        ledger.prepare()
        if ledger.complete:
            return _extended_summary(ledger)
        _require_single_gpu_environment()
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        _enable_determinism()
        _seed_current_process(inputs.source_evaluation.evaluation.random_seed)
        processor, model = _load_qwen(inputs.training, device=device)
        tokenizer_length_before = len(processor.tokenizer)
        runtime = create_qwen3_representation_runtime(
            model=model,
            processor=processor,
            model_identity=inputs.training.model_identity,
            conditioning_config=inputs.training.provider,
            adapter_dtype=_torch_dtype(inputs.training.model.dtype),
            adapter_variant=inputs.training.adapter_variant,
            fixture_mode=False,
        )
        if runtime.renderer.assistant_dialect is not (
            NativeAssistantDialect.QWEN3_VL_INSTRUCT
        ):
            raise ValueError("answer-utility evaluation requires Qwen3-VL Instruct")
        observed_eos = _model_eos_token_ids(model)
        if set(observed_eos) != set(inputs.eos_token_ids):
            raise ValueError(
                "configured EOS IDs differ from local Instruct generation config: "
                f"configured={inputs.eos_token_ids}, model={observed_eos}"
            )
        if len(processor.tokenizer) != tokenizer_length_before:
            raise RuntimeError("answer-utility evaluation changed tokenizer length")
        runtime.adapter.load_artifact_state_dict(inputs.candidate.adapter_state)
        runtime.adapter.requires_grad_(False)
        runtime.adapter.eval()
        model.requires_grad_(False)
        model.eval()
        family_adapter = Qwen3VLAdapter()
        group_builder = Qwen3NativeRepresentationGroupBuilder(
            runtime=runtime,
            family_adapter=family_adapter,
            prompt=inputs.training.prompt,
            image_loader=_load_rgb_image,
            image_max_pixels=inputs.training.model.image_max_pixels,
        )
        truth_by_id = {truth.sample_id: truth for _model, truth in model_truth_rows}
        started = time.monotonic()
        parity_path = output / "image_only_native_parity.json"
        parity_checked = AnswerUtilityEvaluationArm.IMAGE_ONLY not in inputs.arms
        if parity_path.exists():
            parity_payload = json.loads(parity_path.read_text(encoding="utf-8"))
            if (
                parity_payload.get("run_identity_sha256") != ledger.identity_sha256
                or parity_payload.get("top1_match") is not True
            ):
                raise ValueError(
                    "existing image-only native parity artifact is invalid"
                )
            parity_checked = True
        for group_ordinal, samples in inputs.selected_groups:
            group_models = tuple(
                split_oracle_d_utility_sample(sample)[0] for sample in samples
            )
            if all(
                ledger.has(row.sample_id, arm.value)
                for row in group_models
                for arm in inputs.arms
            ):
                continue
            visuals = materialize_oracle_group_visuals(
                model_inputs=group_models,
                runtime=runtime,
                group_builder=group_builder,
            )
            wrong_image_visuals = None
            wrong_image_donor = None
            if (
                AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE
                in inputs.arms
                and any(
                    not ledger.has(
                        row.sample_id,
                        AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE.value,
                    )
                    for row in group_models
                )
            ):
                wrong_image_donor = inputs.same_target_wrong_image_by_group_key.get(
                    group_models[0].image_group_key
                )
                if wrong_image_donor is None:
                    raise RuntimeError("same-target wrong-image donor is missing")
                if (
                    file_sha256(group_models[0].image)
                    != wrong_image_donor.anchor_image_sha256
                    or file_sha256(wrong_image_donor.donor_image)
                    != wrong_image_donor.donor_image_sha256
                ):
                    raise RuntimeError(
                        "anchor or wrong-image donor bytes changed after identity binding"
                    )
                wrong_image_models = _same_target_wrong_image_model_inputs(
                    group_models, wrong_image_donor
                )
                wrong_image_visuals = materialize_oracle_group_visuals(
                    model_inputs=wrong_image_models,
                    runtime=runtime,
                    group_builder=group_builder,
                )
                if wrong_image_visuals.image_grid_thw != visuals.image_grid_thw or (
                    wrong_image_visuals.image_grid_thw
                    != wrong_image_donor.image_grid_thw
                ):
                    raise RuntimeError(
                        "runtime Qwen grid differs from the bound wrong-image mapping"
                    )
            if not parity_checked:
                first = group_models[0]
                first_reader = _reader_model_input(first)
                parity_context = prepare_oracle_arm_context(
                    model_input=first_reader,
                    arm=OracleDUtilityArm.IMAGE_ONLY,
                    runtime=runtime,
                    source=visuals.source,
                    correct_d=visuals.correct_d_by_sample_id[first.sample_id],
                    image_grid_thw=visuals.image_grid_thw,
                )
                parity = verify_image_only_injected_native_parity(
                    model_input=first_reader,
                    context=parity_context,
                    runtime=runtime,
                    family_adapter=family_adapter,
                    image_loader=group_builder.image_loader,
                    image_max_pixels=group_builder.image_max_pixels,
                )
                _atomic_write_json(
                    parity_path,
                    {
                        "schema_version": "answer_utility_image_only_parity_v1",
                        "run_identity_sha256": ledger.identity_sha256,
                        **asdict(parity),
                    },
                )
                parity_checked = True
            by_id = {row.sample_id: row for row in group_models}
            for row in group_models:
                reader_row = _reader_model_input(row)
                correct_d = visuals.correct_d_by_sample_id[row.sample_id]
                wrong_source_id = inputs.wrong_source_by_sample_id[row.sample_id]
                wrong_d = visuals.correct_d_by_sample_id[wrong_source_id]
                same_target_wrong_image_d = (
                    None
                    if wrong_image_visuals is None
                    else wrong_image_visuals.correct_d_by_sample_id[row.sample_id]
                )
                pending: list[tuple[AnswerUtilityEvaluationArm, OracleArmContext]] = []
                for arm in inputs.arms:
                    if ledger.has(row.sample_id, arm.value):
                        continue
                    context = _prepare_context(
                        arm=arm,
                        model_input=reader_row,
                        runtime=runtime,
                        source=visuals.source,
                        correct_d=correct_d,
                        wrong_d=wrong_d,
                        same_target_wrong_image_d=same_target_wrong_image_d,
                        image_grid_thw=visuals.image_grid_thw,
                    )
                    pending.append((arm, context))
                generated_answers = _generate_pending_answers(
                    contexts=tuple(context for _arm, context in pending),
                    runtime=runtime,
                    family_adapter=family_adapter,
                    eos_token_ids=inputs.eos_token_ids,
                    max_new_tokens=inputs.max_new_tokens,
                    decode_mode=inputs.decode_mode,
                    arm_batch_size=inputs.arm_batch_size,
                )
                for (arm, context), generated in zip(
                    pending,
                    generated_answers,
                    strict=True,
                ):
                    score = score_instruct_generated_answer(
                        generated.text,
                        truth_by_id[row.sample_id],
                        generation_stop_reason=generated.stop_reason,
                    )
                    record: dict[str, Any] = {
                        "schema_version": ANSWER_UTILITY_EVALUATION_RECORD_SCHEMA_VERSION,
                        "oracle_record_schema_compatibility": (
                            ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION
                        ),
                        "run_identity_sha256": ledger.identity_sha256,
                        "candidate_kind": inputs.candidate.kind,
                        "candidate_id": inputs.candidate.candidate_id,
                        "candidate_training_run_identity_sha256": (
                            inputs.candidate.training_run_identity_sha256
                        ),
                        "sample_id": row.sample_id,
                        "sample_content_sha256": row.sample_content_sha256,
                        "reader_question_sha256": sha256(
                            reader_row.question.encode("utf-8")
                        ).hexdigest(),
                        "reader_contract_version": (INSTRUCT_READER_CONTRACT_VERSION),
                        "scoring_contract_version": (INSTRUCT_SCORING_CONTRACT_VERSION),
                        "image_group_key": row.image_group_key,
                        "group_ordinal": group_ordinal,
                        "arm": arm.value,
                        "arm_contract": _evaluation_arm_contract(arm),
                        "wrong_d_source_sample_id": (
                            wrong_source_id if _uses_wrong_d(arm) else None
                        ),
                        "wrong_d_source_target_sha256": (
                            sha256(
                                by_id[wrong_source_id].target.encode("utf-8")
                            ).hexdigest()
                            if _uses_wrong_d(arm)
                            else None
                        ),
                        "rendered_prefix_text_sha256": context.rendered_text_sha256,
                        "canonical_prefix_token_ids_sha256": (
                            context.canonical_token_ids_sha256
                        ),
                        "prefix_token_count": int(context.prefix_input_ids.shape[1]),
                        "source_visual_token_count": len(context.source_positions),
                        "d_visual_token_count": len(context.d_positions),
                        "generated_token_ids": list(generated.token_ids),
                        "generated_text": generated.text,
                        "generation_stop_reason": generated.stop_reason,
                        "score": asdict(score),
                        "expected_short_answer": truth_by_id[
                            row.sample_id
                        ].short_answer,
                        "elapsed_seconds_since_process_start": (
                            time.monotonic() - started
                        ),
                    }
                    if _uses_same_target_wrong_image_d(arm):
                        if wrong_image_donor is None:
                            raise RuntimeError(
                                "wrong-image record lost its bound donor"
                            )
                        record.update(
                            {
                                "wrong_image_source_sample_id": (
                                    wrong_image_donor.donor_sample_id
                                ),
                                "wrong_image_source_sample_content_sha256": (
                                    wrong_image_donor.donor_sample_content_sha256
                                ),
                                "wrong_image_source_group_key": (
                                    wrong_image_donor.donor_image_group_key
                                ),
                                "wrong_image_source_file_sha256": (
                                    wrong_image_donor.donor_image_sha256
                                ),
                                "anchor_image_file_sha256": (
                                    wrong_image_donor.anchor_image_sha256
                                ),
                                "wrong_image_grid_thw": list(
                                    wrong_image_donor.image_grid_thw
                                ),
                                "wrong_image_match_tier": (
                                    wrong_image_donor.match_tier
                                ),
                                "d_materialization_question_sha256": sha256(
                                    row.question.encode("utf-8")
                                ).hexdigest(),
                                "d_materialization_target_sha256": sha256(
                                    row.target.encode("utf-8")
                                ).hexdigest(),
                                "d_materialization_contextual_hidden_state": (
                                    "recomputed_on_wrong_image_with_anchor_question_and_target"
                                ),
                            }
                        )
                    ledger.commit(record)
            del visuals
            if wrong_image_visuals is not None:
                del wrong_image_visuals
        if len(processor.tokenizer) != tokenizer_length_before:
            raise RuntimeError("answer-utility evaluation changed tokenizer length")
        return _extended_summary(ledger)


def build_answer_safe_wrong_mapping(
    groups: Sequence[tuple[int, Sequence[RepresentationTrainingSample]]],
) -> dict[str, str]:
    """Choose a deterministic same-image, different-target/different-answer D."""

    mapping: dict[str, str] = {}
    for _ordinal, raw_group in groups:
        group = tuple(raw_group)
        if len(group) < 2:
            raise ValueError("answer-safe wrong D requires K>=2")
        if len({sample.image_group_key for sample in group}) != 1:
            raise ValueError("wrong-D candidates must share one image group")
        for index, sample in enumerate(group):
            answer = _normalized_answer_identity(sample.short_answer)
            target = _normalized_target_identity(sample.target)
            candidate = next(
                (
                    group[(index + offset) % len(group)]
                    for offset in range(1, len(group))
                    if _normalized_answer_identity(
                        group[(index + offset) % len(group)].short_answer
                    )
                    != answer
                    and _normalized_target_identity(
                        group[(index + offset) % len(group)].target
                    )
                    != target
                ),
                None,
            )
            if candidate is None:
                raise ValueError(
                    "image group has no same-image different-target/different-answer "
                    f"wrong D for sample {sample.sample_id}"
                )
            mapping[sample.sample_id] = candidate.sample_id
    if len(mapping) != sum(len(tuple(group)) for _ordinal, group in groups):
        raise ValueError("wrong-D mapping contains duplicate sample IDs")
    return mapping


def build_same_target_wrong_image_mapping(
    groups: Sequence[tuple[int, Sequence[RepresentationTrainingSample]]],
    donor_samples: Sequence[RepresentationTrainingSample],
    *,
    grid_contract: _QwenImageGridContract,
    random_seed: int,
) -> dict[str, AnswerUtilityWrongImageDonor]:
    """Bind each anchor group to a deterministic, exact-grid, distinct image.

    Donor labels are used only to reject accidental answer-equivalent images;
    the returned object contains no donor answer, target, or evidence and only
    its image is permitted to reach D materialization.
    """

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("wrong-image mapping seed must be an integer")
    if isinstance(donor_samples, (str, bytes)) or not isinstance(
        donor_samples, Sequence
    ):
        raise TypeError("wrong-image donor samples must be a sequence")
    donors_by_group: dict[str, list[RepresentationTrainingSample]] = {}
    for sample in donor_samples:
        if not isinstance(sample, RepresentationTrainingSample):
            raise TypeError("wrong-image donor pool must contain typed samples")
        donors_by_group.setdefault(sample.image_group_key, []).append(sample)
    if not donors_by_group:
        raise ValueError("wrong-image donor pool is empty")

    grid_by_path: dict[str, tuple[int, int, int]] = {}

    def image_grid(path: str) -> tuple[int, int, int]:
        observed = grid_by_path.get(path)
        if observed is None:
            observed = _qwen_image_grid_thw(path, grid_contract)
            grid_by_path[path] = observed
        return observed

    donor_descriptors: list[
        tuple[
            RepresentationTrainingSample,
            frozenset[str],
            frozenset[str],
            tuple[int, int, int],
        ]
    ] = []
    for raw_group in donors_by_group.values():
        donor_group = tuple(raw_group)
        paths = frozenset(sample.image for sample in donor_group)
        if len(paths) != 1:
            raise ValueError("one donor image group contains multiple image paths")
        representative = min(donor_group, key=lambda sample: sample.sample_id)
        donor_descriptors.append(
            (
                representative,
                frozenset(
                    _normalized_answer_identity(sample.short_answer)
                    for sample in donor_group
                ),
                frozenset(
                    sample.stable_image_uid
                    for sample in donor_group
                    if sample.stable_image_uid is not None
                ),
                image_grid(representative.image),
            )
        )

    by_grid_source: dict[
        tuple[tuple[int, int, int], str | None],
        list[
            tuple[
                RepresentationTrainingSample,
                frozenset[str],
                frozenset[str],
                tuple[int, int, int],
            ]
        ],
    ] = {}
    by_grid_profile: dict[
        tuple[tuple[int, int, int], str | None],
        list[
            tuple[
                RepresentationTrainingSample,
                frozenset[str],
                frozenset[str],
                tuple[int, int, int],
            ]
        ],
    ] = {}
    by_grid: dict[
        tuple[int, int, int],
        list[
            tuple[
                RepresentationTrainingSample,
                frozenset[str],
                frozenset[str],
                tuple[int, int, int],
            ]
        ],
    ] = {}
    for descriptor in donor_descriptors:
        representative, _answers, _uids, grid = descriptor
        by_grid_source.setdefault((grid, representative.source_dataset), []).append(
            descriptor
        )
        by_grid_profile.setdefault((grid, representative.source_profile), []).append(
            descriptor
        )
        by_grid.setdefault(grid, []).append(descriptor)

    mapping: dict[str, AnswerUtilityWrongImageDonor] = {}
    for _ordinal, raw_group in groups:
        anchor_group = tuple(raw_group)
        if not anchor_group or any(
            not isinstance(sample, RepresentationTrainingSample)
            for sample in anchor_group
        ):
            raise ValueError("wrong-image anchors must be non-empty typed groups")
        if (
            len({sample.image_group_key for sample in anchor_group}) != 1
            or len({sample.image for sample in anchor_group}) != 1
        ):
            raise ValueError("one wrong-image anchor group must share one image")
        anchor = min(anchor_group, key=lambda sample: sample.sample_id)
        anchor_answers = frozenset(
            _normalized_answer_identity(sample.short_answer) for sample in anchor_group
        )
        anchor_uids = frozenset(
            sample.stable_image_uid
            for sample in anchor_group
            if sample.stable_image_uid is not None
        )
        anchor_grid = image_grid(anchor.image)
        anchor_image_sha256 = file_sha256(anchor.image)
        tiers: tuple[
            tuple[
                Literal[
                    "exact_grid_same_source_dataset",
                    "exact_grid_same_source_profile",
                    "exact_grid_cross_domain",
                ],
                Sequence[
                    tuple[
                        RepresentationTrainingSample,
                        frozenset[str],
                        frozenset[str],
                        tuple[int, int, int],
                    ]
                ],
            ],
            ...,
        ] = (
            (
                "exact_grid_same_source_dataset",
                by_grid_source.get((anchor_grid, anchor.source_dataset), ()),
            ),
            (
                "exact_grid_same_source_profile",
                by_grid_profile.get((anchor_grid, anchor.source_profile), ()),
            ),
            ("exact_grid_cross_domain", by_grid.get(anchor_grid, ())),
        )
        chosen: AnswerUtilityWrongImageDonor | None = None
        for match_tier, raw_candidates in tiers:
            candidates = tuple(
                descriptor
                for descriptor in raw_candidates
                if descriptor[0].image_group_key != anchor.image_group_key
                and descriptor[0].image != anchor.image
                and anchor_answers.isdisjoint(descriptor[1])
                and anchor_uids.isdisjoint(descriptor[2])
            )
            ordered = sorted(
                candidates,
                key=lambda descriptor: sha256(
                    (
                        f"{random_seed}\0{anchor.image_group_key}\0"
                        f"{descriptor[0].image_group_key}"
                    ).encode("utf-8")
                ).digest(),
            )
            for representative, _answers, _uids, donor_grid in ordered:
                donor_image_sha256 = file_sha256(representative.image)
                if donor_image_sha256 == anchor_image_sha256:
                    continue
                if donor_grid != anchor_grid:
                    raise RuntimeError("wrong-image mapping admitted a grid mismatch")
                chosen = AnswerUtilityWrongImageDonor(
                    anchor_image_group_key=anchor.image_group_key,
                    anchor_image_sha256=anchor_image_sha256,
                    donor_sample_id=representative.sample_id,
                    donor_sample_content_sha256=representative.content_sha256,
                    donor_image_group_key=representative.image_group_key,
                    donor_image=representative.image,
                    donor_image_sha256=donor_image_sha256,
                    image_grid_thw=donor_grid,
                    match_tier=match_tier,
                )
                break
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError(
                "no exact-grid, answer-disjoint, byte-distinct wrong image for "
                f"anchor group {anchor.image_group_key}"
            )
        mapping[anchor.image_group_key] = chosen
    if len(mapping) != len(tuple(groups)):
        raise ValueError("wrong-image mapping contains duplicate anchor groups")
    return mapping


def _load_qwen_image_grid_contract(
    training: RepresentationTrainingConfig,
) -> _QwenImageGridContract:
    path = training.model.local_path / "preprocessor_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Qwen preprocessor config is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    size = payload.get("size")
    if not isinstance(size, Mapping):
        raise ValueError("Qwen preprocessor size contract is missing")
    patch_size = payload.get("patch_size")
    merge_size = payload.get("merge_size")
    min_pixels = size.get("shortest_edge")
    configured_max = training.model.image_max_pixels
    max_pixels = size.get("longest_edge") if configured_max is None else configured_max
    return _QwenImageGridContract(
        patch_size=patch_size,
        merge_size=merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def _qwen_image_grid_thw(
    image_path: str, contract: _QwenImageGridContract
) -> tuple[int, int, int]:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - production dependency
        raise RuntimeError("Pillow is required for wrong-image matching") from error
    with Image.open(image_path) as image:
        width, height = image.size
    if height <= 0 or width <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError(f"unsupported Qwen image geometry: {image_path}")
    factor = contract.patch_size * contract.merge_size
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > contract.max_pixels:
        scale = math.sqrt((height * width) / contract.max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < contract.min_pixels:
        scale = math.sqrt(contract.min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return (
        1,
        resized_height // contract.patch_size,
        resized_width // contract.patch_size,
    )


def _same_target_wrong_image_model_inputs(
    rows: Sequence[Any], donor: AnswerUtilityWrongImageDonor
) -> tuple[Any, ...]:
    """Change only the D-source image while retaining anchor Q and target."""

    materialized = tuple(rows)
    if not materialized:
        raise ValueError("wrong-image D materialization requires anchor rows")
    if {row.image_group_key for row in materialized} != {donor.anchor_image_group_key}:
        raise ValueError("wrong-image donor does not bind the anchor group")
    return tuple(
        replace(
            row,
            image=donor.donor_image,
            image_group_key=donor.donor_image_group_key,
        )
        for row in materialized
    )


def _load_private_inputs(
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
) -> AnswerUtilityEvaluationInputs:
    selected_arms = _normalize_evaluation_arms(arms)
    run = load_answer_utility_run_config(run_config_path)
    experiment = load_answer_utility_experiment_config(run.experiment_config_path)
    if run.run_id != experiment.run_id:
        raise ValueError("run and experiment run_id values differ")
    if run.experiment_config_sha256 != experiment.source_toml_sha256:
        raise ValueError("run sidecar points at another experiment config")
    training = load_representation_training_config(experiment.base_training_config_path)
    _require_instruct_training(training)
    source_evaluation = load_representation_internal_evaluation_run_config(
        source_evaluation_config_path
    )
    _validate_private_source_bindings(run, experiment, source_evaluation)
    _load_validated_production_export(training, source_evaluation)
    artifact = load_answer_utility_adapter_artifact(run.final_artifact_path)
    _audit_completed_training_metrics(run, experiment.variant.value, artifact)
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
        production_source_artifact_sha256=(source_evaluation.artifact_file_sha256),
        production_source_manifest_sha256=(source_evaluation.artifact_manifest_sha256),
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
    return _materialize_common_inputs(
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


def _load_production_source_inputs(
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
) -> AnswerUtilityEvaluationInputs:
    selected_arms = _normalize_evaluation_arms(arms)
    source_evaluation = load_representation_internal_evaluation_run_config(
        source_evaluation_config_path
    )
    training = load_representation_training_config(
        source_evaluation.training_config_path,
        allow_existing_post_training_report=True,
    )
    _require_instruct_training(training)
    export = _load_validated_production_export(training, source_evaluation)
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
        training_run_identity_sha256=(source_evaluation.expected_run_identity_sha256),
        production_source_artifact_path=source_evaluation.artifact_path,
        production_source_artifact_sha256=(source_evaluation.artifact_file_sha256),
        production_source_manifest_sha256=(source_evaluation.artifact_manifest_sha256),
        production_source_run_identity_sha256=(
            source_evaluation.expected_run_identity_sha256
        ),
        production_source_global_step=source_evaluation.expected_global_step,
        protected_paths=(source_evaluation.artifact_path.parent.resolve(),),
    )
    return _materialize_common_inputs(
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


def _materialize_common_inputs(
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


def _validate_private_source_bindings(
    run: AnswerUtilityRunConfig,
    experiment: Any,
    source_evaluation: Any,
) -> None:
    if source_evaluation.training_config_path != experiment.base_training_config_path:
        raise ValueError("evaluation source identifies another base training config")
    if (
        source_evaluation.training_config_sha256
        != experiment.base_training_config_sha256
    ):
        raise ValueError("evaluation/base training config SHA256 mismatch")
    if source_evaluation.artifact_path != run.source_artifact.path:
        raise ValueError("evaluation source identifies another production Adapter")
    if source_evaluation.artifact_file_sha256 != run.source_artifact.file_sha256:
        raise ValueError("evaluation/source production artifact file SHA256 mismatch")
    if (
        source_evaluation.artifact_manifest_sha256
        != run.source_artifact.manifest_sha256
    ):
        raise ValueError("evaluation/source production artifact manifest mismatch")
    if (
        source_evaluation.expected_run_identity_sha256
        != run.source_artifact.expected_run_identity_sha256
    ):
        raise ValueError("evaluation/source production run identity mismatch")
    if (
        source_evaluation.expected_global_step
        != run.source_artifact.expected_global_step
    ):
        raise ValueError("evaluation/source production global step mismatch")


def _load_validated_production_export(
    training: RepresentationTrainingConfig,
    source_evaluation: Any,
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
        if file_sha256(path) != expected:
            raise ValueError(f"{name} SHA256 mismatch")
    export = load_rank_zero_adapter_owned_state_export(source_evaluation.artifact_path)
    if state_digest(export.manifest) != source_evaluation.artifact_manifest_sha256:
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
    _validate_training_artifact_binding(training, export.manifest.run_identity)
    return export


def _require_instruct_training(training: RepresentationTrainingConfig) -> None:
    if training.model.model_name != "Qwen3-VL-8B-Instruct" or (
        training.model.local_path.name != "Qwen3-VL-8B-Instruct"
    ):
        raise ValueError(
            "answer-utility held-out evaluation is pinned to Qwen3-VL-8B-Instruct"
        )


def _audit_completed_training_metrics(
    run: AnswerUtilityRunConfig,
    expected_variant: str,
    artifact: AnswerUtilityAdapterArtifact,
) -> None:
    if not run.metrics_path.is_file():
        raise FileNotFoundError("formal answer-utility metrics ledger is missing")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        run.metrics_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid formal metrics JSON at line {line_number}"
            ) from error
        if not isinstance(row, Mapping) or (
            row.get("schema_version") != ANSWER_UTILITY_METRICS_SCHEMA_VERSION
        ):
            raise ValueError(f"formal metrics schema mismatch at line {line_number}")
        rows.append(row)
    if not rows or rows[0].get("event") != "start":
        raise ValueError("formal metrics ledger has no start record")
    identity = rows[0].get("run_identity_sha256")
    _require_sha256(identity, name="metrics run identity")
    if (
        rows[0].get("run_id") != run.run_id
        or rows[0].get("run_config") != run.identity_payload()
    ):
        raise ValueError("formal metrics start record differs from run sidecar")
    active_step = 0
    completed = False
    for line_number, row in enumerate(rows[1:], 2):
        event = row.get("event")
        if completed:
            raise ValueError("formal metrics contain records after completion")
        if event == "step":
            if row.get("run_identity_sha256") != identity:
                raise ValueError(
                    f"formal metrics identity mismatch at line {line_number}"
                )
            step = row.get("global_step")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step <= active_step
            ):
                raise ValueError(
                    f"formal metrics step order mismatch at line {line_number}"
                )
            active_step = step
        elif event == "resume":
            if row.get("run_identity_sha256") != identity or (
                row.get("from_global_step") != active_step
            ):
                raise ValueError(
                    f"formal metrics resume mismatch at line {line_number}"
                )
        elif event == "stop":
            result = row.get("result")
            if not isinstance(result, Mapping) or (
                result.get("run_identity_sha256") != identity
                or result.get("global_step") != active_step
            ):
                raise ValueError(f"formal metrics stop mismatch at line {line_number}")
        elif event == "complete":
            result = row.get("result")
            if not isinstance(result, Mapping):
                raise ValueError("formal metrics completion result is missing")
            expected = {
                "status": "complete",
                "run_id": run.run_id,
                "variant": expected_variant,
                "run_identity_sha256": identity,
                "global_step": run.target_optimizer_steps,
                "planned_target_optimizer_steps": run.target_optimizer_steps,
                "artifact_path": str(run.final_artifact_path),
                "metrics_path": str(run.metrics_path),
            }
            if any(result.get(name) != value for name, value in expected.items()):
                raise ValueError("formal metrics completion differs from run/artifact")
            if active_step != run.target_optimizer_steps:
                raise ValueError("formal metrics did not reach the target step")
            completed = True
        else:
            raise ValueError(
                f"unknown formal metrics event at line {line_number}: {event!r}"
            )
    if not completed or rows[-1].get("event") != "complete":
        raise ValueError("formal answer-utility training is not complete")
    if identity != artifact.run_identity_sha256:
        raise ValueError("private artifact and formal metrics run identities differ")


def _generate_pending_answers(
    *,
    contexts: Sequence[OracleArmContext],
    runtime: Any,
    family_adapter: Any,
    eos_token_ids: tuple[int, ...],
    max_new_tokens: int,
    decode_mode: Literal["cached", "no_cache"],
    arm_batch_size: int,
) -> tuple[OracleGeneratedAnswer, ...]:
    """Decode same-row arms in bounded compatible batches with exact fallback."""

    pending = tuple(contexts)
    selected_batch_size = _normalize_arm_batch_size(arm_batch_size)
    generated: list[OracleGeneratedAnswer] = []
    for start in range(0, len(pending), selected_batch_size):
        batch = pending[start : start + selected_batch_size]
        if decode_mode == "cached" and len(batch) > 1:
            try:
                generated.extend(
                    greedy_oracle_answers_batched(
                        contexts=batch,
                        runtime=runtime,
                        family_adapter=family_adapter,
                        eos_token_ids=eos_token_ids,
                        max_new_tokens=max_new_tokens,
                    )
                )
                continue
            except OracleBatchCompatibilityError:
                pass
        generated.extend(
            greedy_oracle_answer(
                context=context,
                runtime=runtime,
                family_adapter=family_adapter,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                decode_mode=decode_mode,
            )
            for context in batch
        )
    return tuple(generated)


def _prepare_context(
    *,
    arm: AnswerUtilityEvaluationArm,
    model_input: Any,
    runtime: Any,
    source: Any,
    correct_d: Any,
    wrong_d: Any,
    same_target_wrong_image_d: Any | None,
    image_grid_thw: tuple[int, int, int],
) -> OracleArmContext:
    oracle_arm = _oracle_arm_for(arm)
    context = prepare_oracle_arm_context(
        model_input=model_input,
        arm=oracle_arm,
        runtime=runtime,
        source=source,
        correct_d=correct_d,
        matched_wrong_d=wrong_d,
        image_grid_thw=image_grid_thw,
    )
    if arm not in {
        AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG,
        AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE,
    }:
        return context
    if not context.d_positions or len(context.visual_blocks) != 2:
        raise RuntimeError("image-plus-correct template did not produce source plus D")
    selected_d = (
        same_target_wrong_image_d
        if arm is AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE
        else wrong_d
    )
    if selected_d is None:
        raise ValueError("same-target wrong-image arm requires its bound D")
    blocks = (
        *context.visual_blocks[:-1],
        _injected_block("focused_d", context.d_positions, selected_d),
    )
    return replace(context, visual_blocks=blocks)


def _oracle_arm_for(arm: AnswerUtilityEvaluationArm) -> OracleDUtilityArm:
    if arm in {
        AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG,
        AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE,
    }:
        return OracleDUtilityArm.IMAGE_CORRECT_D
    return OracleDUtilityArm(arm.value)


def _reader_model_input(model_input: Any) -> Any:
    """Change only the answer reader prompt, never D materialization inputs."""

    return replace(model_input, question=reader_question(model_input.question))


def _evaluation_identity_payload(
    inputs: AnswerUtilityEvaluationInputs, model_inputs: Sequence[Any]
) -> dict[str, Any]:
    implementation_files = _implementation_file_manifest()
    candidate = inputs.candidate
    arm_batch_size = _normalize_arm_batch_size(getattr(inputs, "arm_batch_size", 1))
    payload: dict[str, Any] = {
        "schema_version": ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION,
        "claim_scope": "oracle_target_conditioned_D_utility_not_end_to_end_tool_selection",
        "candidate_kind": candidate.kind,
        "candidate_id": candidate.candidate_id,
        "candidate_artifact_path": str(candidate.adapter_path),
        "candidate_artifact_file_sha256": candidate.adapter_file_sha256,
        "candidate_adapter_state_sha256": candidate.adapter_state_sha256,
        "candidate_training_run_identity_sha256": (
            candidate.training_run_identity_sha256
        ),
        "candidate_global_step": candidate.global_step,
        "private_run_id": candidate.private_run_id,
        "private_run_config_path": (
            None
            if candidate.private_run_config_path is None
            else str(candidate.private_run_config_path)
        ),
        "private_run_config_sha256": candidate.private_run_config_sha256,
        "private_experiment_config_sha256": (
            candidate.private_experiment_config_sha256
        ),
        "production_source_artifact_path": str(
            candidate.production_source_artifact_path
        ),
        "production_source_artifact_sha256": (
            candidate.production_source_artifact_sha256
        ),
        "production_source_manifest_sha256": (
            candidate.production_source_manifest_sha256
        ),
        "production_source_run_identity_sha256": (
            candidate.production_source_run_identity_sha256
        ),
        "production_source_global_step": candidate.production_source_global_step,
        "source_evaluation_config_path": str(inputs.source_evaluation.source_path),
        "source_evaluation_config_sha256": inputs.source_evaluation.source_sha256,
        "model_name": inputs.training.model.model_name,
        "model_path": str(inputs.training.model.local_path),
        "assistant_dialect": NativeAssistantDialect.QWEN3_VL_INSTRUCT.value,
        "data_manifest_sha256": inputs.data_manifest_sha256,
        "ordered_group_manifest_identity": inputs.ordered_group_manifest_identity,
        "ordered_group_manifest_sha256": inputs.source_evaluation.evaluation.ordered_group_manifest_sha256,
        "ordered_selected_samples": [
            _selected_sample_identity(inputs, row) for row in model_inputs
        ],
        "arms": [arm.value for arm in inputs.arms],
        "arm_contracts": {
            arm.value: _evaluation_arm_contract(arm) for arm in inputs.arms
        },
        "max_new_tokens": inputs.max_new_tokens,
        "eos_token_ids": list(inputs.eos_token_ids),
        "decode_mode": inputs.decode_mode,
        "arm_batch_size": arm_batch_size,
        "decoder_implementation": _decoder_implementation(inputs),
        "greedy": True,
        "random_seed": inputs.source_evaluation.evaluation.random_seed,
        "group_start": inputs.group_start,
        "group_limit": inputs.group_limit,
        "shard_index": inputs.shard_index,
        "shard_count": inputs.shard_count,
        "wrong_d_rule": "same_image_different_normalized_target_and_answer_cyclic_v1",
        "d_materialization_question": "original_unmodified_question",
        "reader_contract_version": INSTRUCT_READER_CONTRACT_VERSION,
        "reader_instruction": INSTRUCT_READER_INSTRUCTION,
        "reader_instruction_sha256": sha256(
            INSTRUCT_READER_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "reader_instruction_applies_to_all_arms": True,
        "scoring_contract_version": INSTRUCT_SCORING_CONTRACT_VERSION,
        "scoring_choices_model_visible": False,
        "scoring_label_route_enabled": False,
        "choice_text_reference_mapping_enabled": True,
        "ground_truth_model_input": False,
        "post_focus_transcript_model_input": False,
        "implementation_file_manifest": implementation_files,
        "implementation_sha256": sha256(
            json.dumps(
                implementation_files,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    if _has_same_target_wrong_image_arm(inputs.arms):
        if inputs.wrong_image_pool_manifest_sha256 is None:
            raise RuntimeError("wrong-image arm lost its donor-pool identity")
        ordered_donors = tuple(
            inputs.same_target_wrong_image_by_group_key[group[0].image_group_key]
            for _ordinal, group in inputs.selected_groups
        )
        payload.update(
            {
                "wrong_image_negative_pool_path": str(
                    inputs.training.data.train.jsonl_path
                ),
                "wrong_image_negative_pool_manifest_sha256": (
                    inputs.wrong_image_pool_manifest_sha256
                ),
                "wrong_image_negative_pool_source_sha256": (
                    inputs.training.data.train.source_sha256
                ),
                "wrong_image_mapping_rule": (
                    "rp66_train_exact_qwen_grid_answer_disjoint_distinct_bytes_"
                    "tiered_source_profile_domain_sha256_rank_v1"
                ),
                "wrong_image_mapping_random_seed": (
                    inputs.source_evaluation.evaluation.random_seed
                ),
                "wrong_image_match_tiers": dict(
                    _wrong_image_match_tier_counts(
                        inputs.same_target_wrong_image_by_group_key
                    )
                ),
                "same_target_wrong_image_mapping": [
                    {
                        **asdict(donor),
                        "image_grid_thw": list(donor.image_grid_thw),
                    }
                    for donor in ordered_donors
                ],
                "wrong_image_d_materialization_question": (
                    "anchor_original_unmodified_question"
                ),
                "wrong_image_d_materialization_target": "anchor_oracle_target",
                "wrong_image_contextual_hidden_state": (
                    "recomputed_by_frozen_qwen_on_wrong_image_anchor_question_target"
                ),
                "wrong_image_reader_source": (
                    "anchor_image_anchor_question_anchor_target"
                ),
            }
        )
    return payload


def _selected_sample_identity(
    inputs: AnswerUtilityEvaluationInputs, row: Any
) -> dict[str, Any]:
    identity = {
        "sample_id": row.sample_id,
        "sample_content_sha256": row.sample_content_sha256,
        "image_group_key": row.image_group_key,
        "reader_question_sha256": sha256(
            reader_question(row.question).encode("utf-8")
        ).hexdigest(),
        "wrong_d_source_sample_id": inputs.wrong_source_by_sample_id[row.sample_id],
    }
    if _has_same_target_wrong_image_arm(inputs.arms):
        donor = inputs.same_target_wrong_image_by_group_key[row.image_group_key]
        identity.update(
            {
                "wrong_image_donor_sample_id": donor.donor_sample_id,
                "wrong_image_donor_sample_content_sha256": (
                    donor.donor_sample_content_sha256
                ),
                "wrong_image_donor_group_key": donor.donor_image_group_key,
                "wrong_image_donor_file_sha256": donor.donor_image_sha256,
                "wrong_image_match_tier": donor.match_tier,
                "wrong_image_grid_thw": list(donor.image_grid_thw),
                "d_materialization_question_sha256": sha256(
                    row.question.encode("utf-8")
                ).hexdigest(),
                "d_materialization_target_sha256": sha256(
                    row.target.encode("utf-8")
                ).hexdigest(),
            }
        )
    return identity


def _implementation_file_manifest() -> dict[str, str]:
    """Bind every local implementation family used by generation/scoring."""

    source_root = Path(__file__).resolve().parents[5]
    patterns = (
        "tgvf_rl/checkpoint/*.py",
        "tgvf_rl/conditioning/*.py",
        "tgvf_rl/protocol/*.py",
        "tgvf_rl/qwen/*.py",
        "tgvf_rl/representation/adapter.py",
        "tgvf_rl/representation/training/*.py",
        "tgvf_rl/representation/experiments/answer_utility/*.py",
        "tgvf_rl/representation/experiments/answer_utility/evaluation/*.py",
    )
    files = tuple(
        sorted(
            {path for pattern in patterns for path in source_root.glob(pattern)},
            key=lambda path: str(path.relative_to(source_root)),
        )
    )
    if not files or any(not path.is_file() for path in files):
        raise RuntimeError("answer-utility evaluation implementation files are missing")
    return {
        str(path.relative_to(source_root)): sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _evaluation_arm_contract(arm: AnswerUtilityEvaluationArm) -> dict[str, Any]:
    if arm is AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE:
        return {
            "prompt": "image_question_plus_oracle_target_tool_transcript",
            "source_image": True,
            "oracle_target_transcript": True,
            "d": (
                "same_anchor_question_target_recomputed_on_exact_grid_"
                "answer_disjoint_distinct_wrong_image_stage1_main_and_all_deepstack"
            ),
            "contextual_target_hidden_state": "recomputed_on_wrong_image",
        }
    if arm is AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG:
        return {
            "prompt": "image_question_plus_oracle_target_tool_transcript",
            "source_image": True,
            "oracle_target_transcript": True,
            "d": "answer_safe_same_image_different_target_and_answer_stage1_main_and_all_deepstack",
        }
    contract = dict(_arm_contract(_oracle_arm_for(arm)))
    if _uses_wrong_d(arm):
        contract["d"] = (
            "answer_safe_same_image_different_target_and_answer_stage1_main_and_all_deepstack"
        )
    return contract


def _extended_summary(ledger: _OracleRunLedger) -> dict[str, object]:
    summary = ledger.summary()
    records = tuple(ledger._completed[key] for key in ledger.expected_keys)  # noqa: SLF001
    paired = dict(summary["paired_effects"])
    comparisons = (
        (
            "D_only_specificity",
            AnswerUtilityEvaluationArm.D_ONLY_CORRECT.value,
            AnswerUtilityEvaluationArm.D_ONLY_WRONG.value,
        ),
        (
            "image_plus_D_specificity",
            AnswerUtilityEvaluationArm.IMAGE_PLUS_CORRECT.value,
            AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG.value,
        ),
        (
            "image_plus_wrong_vs_zero",
            AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG.value,
            AnswerUtilityEvaluationArm.IMAGE_PLUS_ZERO.value,
        ),
        (
            "image_plus_D_image_grounding",
            AnswerUtilityEvaluationArm.IMAGE_PLUS_CORRECT.value,
            AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE.value,
        ),
    )
    available = set(summary["arms"])
    for name, treatment, control in comparisons:
        if treatment in available and control in available:
            paired[name] = _paired_summary(records, treatment, control)
    summary["paired_effects"] = paired
    summary["evaluation_schema_version"] = ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION
    summary["arm_batch_size"] = ledger.identity_payload.get("arm_batch_size", 1)
    summary["decoder_implementation"] = ledger.identity_payload.get(
        "decoder_implementation",
        "scalar_greedy_v1",
    )
    interpretation = (
        "Correct-vs-zero measures D content utility; correct-vs-answer-safe-wrong "
        "measures target specificity. All D arms condition on an oracle target and do "
        "not measure autonomous tool selection."
    )
    if AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE.value in available:
        interpretation += (
            " Correct-vs-same-target-wrong-image measures image dependence while "
            "holding the reader image, question, target, and D geometry fixed."
        )
    summary["interpretation"] = interpretation
    _atomic_write_json(ledger.summary_path, summary)
    return summary


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


def _decoder_implementation(inputs: AnswerUtilityEvaluationInputs) -> str:
    arm_batch_size = _normalize_arm_batch_size(getattr(inputs, "arm_batch_size", 1))
    if inputs.decode_mode == "cached" and arm_batch_size > 1:
        return "same_sample_compatible_arm_batched_cached_greedy_v1"
    return "scalar_greedy_v1"


def _uses_wrong_d(arm: AnswerUtilityEvaluationArm) -> bool:
    return arm in {
        AnswerUtilityEvaluationArm.D_ONLY_WRONG,
        AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG,
        AnswerUtilityEvaluationArm.DIRECT_WRONG_REPLACEMENT,
    }


def _uses_same_target_wrong_image_d(arm: AnswerUtilityEvaluationArm) -> bool:
    return arm is AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE


def _has_same_target_wrong_image_arm(
    arms: Sequence[AnswerUtilityEvaluationArm],
) -> bool:
    return AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE in arms


def _wrong_image_match_tier_counts(
    mapping: Mapping[str, AnswerUtilityWrongImageDonor],
) -> dict[str, int]:
    counts = {
        "exact_grid_same_source_dataset": 0,
        "exact_grid_same_source_profile": 0,
        "exact_grid_cross_domain": 0,
    }
    for donor in mapping.values():
        counts[donor.match_tier] += 1
    return counts


def _normalized_target_identity(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("target identity requires non-empty text")
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _assert_evaluation_output_isolated(
    output: Path, inputs: AnswerUtilityEvaluationInputs
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


def _require_sha256(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256")


__all__ = [
    "ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION",
    "DEFAULT_INSTRUCT_EOS_TOKEN_IDS",
    "DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS",
    "AnswerUtilityAdapterArtifact",
    "AnswerUtilityEvaluationArm",
    "AnswerUtilityEvaluationCandidate",
    "AnswerUtilityWrongImageDonor",
    "build_answer_safe_wrong_mapping",
    "build_same_target_wrong_image_mapping",
    "load_answer_utility_adapter_artifact",
    "run_answer_utility_evaluation",
    "run_production_source_answer_utility_evaluation",
    "validate_answer_utility_evaluation",
    "validate_production_source_answer_utility_evaluation",
]
