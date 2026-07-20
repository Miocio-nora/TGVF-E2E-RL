"""Once-after-completion orchestration for representation internal evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn

from tgvf_rl.qwen.base import QwenVLMFamilyAdapter

from .config import RepresentationPostTrainingInternalEvaluationConfig
from .internal_evaluation import (
    RepresentationInternalEvaluationArtifact,
    RepresentationInternalEvaluationIdentity,
    create_injected_native_counterfactual_evaluator,
    run_representation_internal_evaluation,
    save_representation_internal_evaluation_report_atomic,
)
from .qwen3_counterfactual import (
    Qwen3CounterfactualCaseBuilder,
    load_qwen3_counterfactual_manifest,
)
from .native_pipeline import Qwen3NativeRepresentationGroupBuilder
from .readout import RepresentationCandidateObservation
from .runtime import Qwen3RepresentationRuntime
from .schema import RepresentationTrainingSample


REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_SCHEMA_VERSION = (
    "representation_internal_evaluation_group_manifest_v2"
)
REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_LEGACY_SCHEMA_VERSION = (
    "representation_internal_evaluation_group_manifest_v1"
)
_SHA256_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class RepresentationInternalEvaluationSampleRef:
    sample_id: str
    content_sha256: str

    def __post_init__(self) -> None:
        _text(self.sample_id, name="sample_id")
        _sha256(self.content_sha256, name="content_sha256")


@dataclass(frozen=True, slots=True)
class RepresentationInternalEvaluationGroupRef:
    image_group_key: str
    samples: tuple[RepresentationInternalEvaluationSampleRef, ...]

    def __post_init__(self) -> None:
        _text(self.image_group_key, name="image_group_key")
        if (
            not isinstance(self.samples, tuple)
            or len(self.samples) < 2
            or any(
                not isinstance(item, RepresentationInternalEvaluationSampleRef)
                for item in self.samples
            )
        ):
            raise ValueError("each internal-evaluation group requires K>=2 samples")
        ids = tuple(item.sample_id for item in self.samples)
        if len(set(ids)) != len(ids):
            raise ValueError("one internal-evaluation group repeats a sample")


@dataclass(frozen=True, slots=True)
class RepresentationInternalEvaluationGroupManifest:
    identity: str
    source_data_manifest_sha256: str
    groups: tuple[RepresentationInternalEvaluationGroupRef, ...]
    schema_version: str = (
        REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _text(self.identity, name="identity")
        _sha256(
            self.source_data_manifest_sha256,
            name="source_data_manifest_sha256",
        )
        if self.schema_version not in {
            REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_SCHEMA_VERSION,
            REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_LEGACY_SCHEMA_VERSION,
        }:
            raise ValueError("internal-evaluation group manifest schema mismatch")
        if (
            not isinstance(self.groups, tuple)
            or len(self.groups) < 2
            or any(
                not isinstance(item, RepresentationInternalEvaluationGroupRef)
                for item in self.groups
            )
        ):
            raise ValueError("internal evaluation requires at least two groups")
        sizes = {len(group.samples) for group in self.groups}
        if (
            self.schema_version
            == REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_LEGACY_SCHEMA_VERSION
            and len(sizes) != 1
        ):
            raise ValueError("internal-evaluation groups must have equal K")
        keys = tuple(group.image_group_key for group in self.groups)
        if len(set(keys)) != len(keys):
            raise ValueError("internal-evaluation image groups must be unique")
        sample_ids = tuple(
            item.sample_id for group in self.groups for item in group.samples
        )
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("internal-evaluation samples must be globally unique")


def load_internal_evaluation_group_manifest(
    path: str | Path,
) -> RepresentationInternalEvaluationGroupManifest:
    """Load an exact ordered sample-group manifest without inferred selection."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "identity",
        "source_data_manifest_sha256",
        "groups",
    }:
        raise ValueError("internal-evaluation group manifest fields differ")
    raw_groups = payload["groups"]
    if isinstance(raw_groups, (str, bytes)) or not isinstance(raw_groups, Sequence):
        raise TypeError("internal-evaluation groups must be an array")
    groups: list[RepresentationInternalEvaluationGroupRef] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, Mapping) or set(raw_group) != {
            "image_group_key",
            "samples",
        }:
            raise ValueError("internal-evaluation group fields differ")
        raw_samples = raw_group["samples"]
        if isinstance(raw_samples, (str, bytes)) or not isinstance(
            raw_samples, Sequence
        ):
            raise TypeError("internal-evaluation group samples must be an array")
        refs: list[RepresentationInternalEvaluationSampleRef] = []
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, Mapping) or set(raw_sample) != {
                "sample_id",
                "content_sha256",
            }:
                raise ValueError("internal-evaluation sample fields differ")
            refs.append(
                RepresentationInternalEvaluationSampleRef(
                    sample_id=raw_sample["sample_id"],
                    content_sha256=raw_sample["content_sha256"],
                )
            )
        groups.append(
            RepresentationInternalEvaluationGroupRef(
                image_group_key=raw_group["image_group_key"],
                samples=tuple(refs),
            )
        )
    return RepresentationInternalEvaluationGroupManifest(
        identity=payload["identity"],
        source_data_manifest_sha256=payload["source_data_manifest_sha256"],
        groups=tuple(groups),
        schema_version=payload["schema_version"],
    )


def materialize_internal_evaluation_groups(
    manifest: RepresentationInternalEvaluationGroupManifest,
    *,
    data_manifest_sha256: str,
    samples: Sequence[RepresentationTrainingSample],
) -> tuple[tuple[RepresentationTrainingSample, ...], ...]:
    """Resolve exact ordered IDs and reject content/group drift."""

    if manifest.source_data_manifest_sha256 != data_manifest_sha256:
        raise ValueError("internal-evaluation manifest identifies another dataset")
    sample_map = {sample.sample_id: sample for sample in samples}
    if len(sample_map) != len(samples):
        raise ValueError("evaluation dataset contains duplicate sample IDs")
    groups: list[tuple[RepresentationTrainingSample, ...]] = []
    for group_ref in manifest.groups:
        group: list[RepresentationTrainingSample] = []
        for sample_ref in group_ref.samples:
            try:
                sample = sample_map[sample_ref.sample_id]
            except KeyError as error:
                raise ValueError(
                    f"internal-evaluation sample is unavailable: {sample_ref.sample_id}"
                ) from error
            if sample.content_sha256 != sample_ref.content_sha256:
                raise ValueError("internal-evaluation sample content SHA256 drifted")
            if sample.image_group_key != group_ref.image_group_key:
                raise ValueError("internal-evaluation image-group identity drifted")
            group.append(sample)
        groups.append(tuple(group))
    return tuple(groups)


def run_post_training_internal_evaluation(
    *,
    config: RepresentationPostTrainingInternalEvaluationConfig,
    runtime: Qwen3RepresentationRuntime,
    qwen_model: nn.Module,
    family_adapter: QwenVLMFamilyAdapter,
    validation_samples: Sequence[RepresentationTrainingSample],
    validation_manifest_sha256: str,
    group_builder: Qwen3NativeRepresentationGroupBuilder,
    model_identity: str,
    checkpoint_identity: str,
    prompt_identity: str,
) -> RepresentationInternalEvaluationArtifact:
    """Execute the enabled suite on every rank and publish only from rank zero."""

    if not config.enabled:
        raise ValueError("post-training internal evaluation is disabled")
    assert config.ordered_group_manifest_path is not None
    assert config.counterfactual_manifest_path is not None
    assert config.report_path is not None
    assert config.evaluation_id is not None
    assert config.random_seed is not None
    assert config.max_new_tokens is not None
    assert config.eos_token_ids is not None
    assert config.ordered_group_manifest_sha256 is not None
    assert config.counterfactual_manifest_sha256 is not None
    for name, path, expected_sha256 in (
        (
            "ordered-group",
            config.ordered_group_manifest_path,
            config.ordered_group_manifest_sha256,
        ),
        (
            "counterfactual",
            config.counterfactual_manifest_path,
            config.counterfactual_manifest_sha256,
        ),
    ):
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"post-training {name} manifest changed during training")
    group_manifest = load_internal_evaluation_group_manifest(
        config.ordered_group_manifest_path
    )
    sample_groups = materialize_internal_evaluation_groups(
        group_manifest,
        data_manifest_sha256=validation_manifest_sha256,
        samples=validation_samples,
    )
    _validate_eos_ids(runtime, config.eos_token_ids)
    observations = _build_observations(
        sample_groups,
        runtime=runtime,
        group_builder=group_builder,
    )
    counterfactual_manifest = load_qwen3_counterfactual_manifest(
        config.counterfactual_manifest_path
    )
    selected_samples = tuple(sample for group in sample_groups for sample in group)
    counterfactual_build = Qwen3CounterfactualCaseBuilder(
        runtime=runtime,
        prompt=group_builder.prompt,
        image_max_pixels=group_builder.image_max_pixels,
    ).build(
        manifest=counterfactual_manifest,
        data_manifest_sha256=validation_manifest_sha256,
        samples=selected_samples,
        observations=observations,
    )
    evaluator = create_injected_native_counterfactual_evaluator(
        model=qwen_model,
        family_adapter=family_adapter,
        materializer=counterfactual_build.materializer,
        eos_token_ids=config.eos_token_ids,
        max_new_tokens=config.max_new_tokens,
    )
    report = run_representation_internal_evaluation(
        identity=RepresentationInternalEvaluationIdentity(
            evaluation_id=config.evaluation_id,
            model_identity=model_identity,
            checkpoint_identity=checkpoint_identity,
            data_manifest_sha256=validation_manifest_sha256,
            prompt_identity=prompt_identity,
            target_conditioning_provider=runtime.conditioning_config.provider,
            random_seed=config.random_seed,
        ),
        adapter=runtime.adapter,
        qwen_model=qwen_model,
        family_adapter=family_adapter,
        sample_groups=sample_groups,
        group_builder=group_builder,
        native_counterfactual_cases=counterfactual_build.cases,
        causal_value_flip_evaluator=evaluator.causal_value_flip,
        free_continuation_evaluator=evaluator.free_continuation,
    )
    artifact: RepresentationInternalEvaluationArtifact | None = None
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        artifact = save_representation_internal_evaluation_report_atomic(
            report, config.report_path
        )
    if torch.distributed.is_initialized():
        payload: list[object | None] = [artifact]
        torch.distributed.broadcast_object_list(payload, src=0)
        artifact = payload[0]
    if not isinstance(artifact, RepresentationInternalEvaluationArtifact):
        raise RuntimeError("rank zero did not publish internal-evaluation artifact")
    return artifact


def _build_observations(
    sample_groups: tuple[tuple[RepresentationTrainingSample, ...], ...],
    *,
    runtime: Qwen3RepresentationRuntime,
    group_builder: Qwen3NativeRepresentationGroupBuilder,
) -> dict[str, RepresentationCandidateObservation]:
    modes = tuple((module, module.training) for module in runtime.adapter.modules())
    observations: dict[str, RepresentationCandidateObservation] = {}
    try:
        runtime.adapter.eval()
        with torch.no_grad():
            for samples in sample_groups:
                group = group_builder(
                    samples,
                    runtime.adapter,
                    collective_candidate_count=len(samples),
                )
                for candidate in group.candidates:
                    observations[candidate.sample_id] = candidate
    finally:
        for module, was_training in modes:
            module.training = was_training
    return observations


def _validate_eos_ids(
    runtime: Qwen3RepresentationRuntime,
    eos_token_ids: tuple[int, ...],
) -> None:
    if any(token_id >= len(runtime.tokenizer) for token_id in eos_token_ids):
        raise ValueError("internal-evaluation EOS token ID lies outside tokenizer")
    configured = runtime.tokenizer.eos_token_id
    configured_ids = (
        tuple(configured)
        if isinstance(configured, (list, tuple))
        else (configured,)
    )
    if any(not isinstance(token_id, int) for token_id in configured_ids):
        raise TypeError("Qwen tokenizer has no integer EOS token identity")
    if not set(configured_ids).issubset(eos_token_ids):
        raise ValueError("internal-evaluation EOS IDs omit tokenizer EOS")


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _text(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _sha256(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _SHA256_CHARS for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


__all__ = [
    "REPRESENTATION_INTERNAL_EVALUATION_GROUP_MANIFEST_SCHEMA_VERSION",
    "RepresentationInternalEvaluationGroupManifest",
    "RepresentationInternalEvaluationGroupRef",
    "RepresentationInternalEvaluationSampleRef",
    "load_internal_evaluation_group_manifest",
    "materialize_internal_evaluation_groups",
    "run_post_training_internal_evaluation",
]
