"""Audited Qwen3 native-D grounding diagnostics.

The manifest binds every probe to retained source rows and raw image bytes.
Cross-image probes reuse the established causal value-flip evaluator.  Target-
presence probes construct two observations from one image and audited present /
absent target strings, then expose exact-layout zero-D controls downstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

from .internal_evaluation_contract import NativeDOnlyContext, NativeTargetPresenceCase
from .native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
    _qwen3_position_ids,
)
from .qwen3_counterfactual import (
    Qwen3CounterfactualBuild,
    Qwen3CounterfactualCaseBuilder,
    Qwen3CounterfactualManifest,
    Qwen3CounterfactualPairSpec,
    Qwen3NativeDOnlyContextRecord,
    Qwen3NativeInjectedRequestMaterializer,
    _canonical_sha256,
    _detached_visual,
    _observation_identity,
    _qwen3_geometry_carrier,
    _runtime_language_device,
    materialize_qwen3_d_only_processor_prefix,
)
from .readout import RepresentationCandidateObservation
from .runtime import Qwen3RepresentationRuntime
from .schema import RepresentationTrainingSample


QWEN3_GROUNDING_MANIFEST_SCHEMA_VERSION = "qwen3_grounding_manifest_v1"
QWEN3_CROSS_IMAGE_PROBE_SCHEMA_VERSION = "qwen3_cross_image_probe_v1"
QWEN3_TARGET_PRESENCE_PROBE_SCHEMA_VERSION = "qwen3_target_presence_probe_v1"
QWEN3_GROUNDING_BUILD_SCHEMA_VERSION = "qwen3_grounding_build_v1"
TARGET_PRESENCE_QUESTION = (
    "Does the requested visual target occur in the image? "
    "Answer PRESENT or NOT_PRESENT."
)
_AUXILIARY_TARGET = "entire source image used only as a diagnostic batching companion"
_SHA256_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class Qwen3CrossImageProbeSpec:
    pair_id: str
    source_sample_a_id: str
    source_sample_a_content_sha256: str
    source_image_a_sha256: str
    source_sample_b_id: str
    source_sample_b_content_sha256: str
    source_image_b_sha256: str
    question: str
    target: str
    expected_value_a: str
    expected_value_b: str
    audit_rationale_a: str
    audit_rationale_b: str
    pair_audit_identity: str
    schema_version: str = QWEN3_CROSS_IMAGE_PROBE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "source_sample_a_id",
            "source_sample_b_id",
            "question",
            "target",
            "expected_value_a",
            "expected_value_b",
            "audit_rationale_a",
            "audit_rationale_b",
            "pair_audit_identity",
        ):
            _text(getattr(self, name), name=name)
        for name in (
            "source_sample_a_content_sha256",
            "source_image_a_sha256",
            "source_sample_b_content_sha256",
            "source_image_b_sha256",
        ):
            _digest(getattr(self, name), name=name)
        if self.source_sample_a_id == self.source_sample_b_id:
            raise ValueError("cross-image probe requires distinct source samples")
        if self.source_image_a_sha256 == self.source_image_b_sha256:
            raise ValueError("cross-image probe requires distinct image bytes")
        if self.expected_value_a == self.expected_value_b:
            raise ValueError("cross-image probe values must be distinct")
        if self.schema_version != QWEN3_CROSS_IMAGE_PROBE_SCHEMA_VERSION:
            raise ValueError("cross-image probe schema mismatch")

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(_dataclass_payload(self))


@dataclass(frozen=True, slots=True)
class Qwen3TargetPresenceProbeSpec:
    pair_id: str
    source_sample_id: str
    source_sample_content_sha256: str
    source_image_sha256: str
    positive_target: str
    negative_target: str
    positive_audit_rationale: str
    negative_audit_rationale: str
    pair_audit_identity: str
    question: str = TARGET_PRESENCE_QUESTION
    present_value: str = "PRESENT"
    not_present_value: str = "NOT_PRESENT"
    schema_version: str = QWEN3_TARGET_PRESENCE_PROBE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "source_sample_id",
            "positive_target",
            "negative_target",
            "positive_audit_rationale",
            "negative_audit_rationale",
            "pair_audit_identity",
            "question",
        ):
            _text(getattr(self, name), name=name)
        _digest(self.source_sample_content_sha256, name="source_sample_content_sha256")
        _digest(self.source_image_sha256, name="source_image_sha256")
        if self.positive_target == self.negative_target:
            raise ValueError("target-presence probe targets must be distinct")
        if self.question != TARGET_PRESENCE_QUESTION:
            raise ValueError("target-presence question is fixed")
        if self.present_value != "PRESENT" or self.not_present_value != "NOT_PRESENT":
            raise ValueError("target-presence labels are fixed")
        if self.schema_version != QWEN3_TARGET_PRESENCE_PROBE_SCHEMA_VERSION:
            raise ValueError("target-presence probe schema mismatch")

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(_dataclass_payload(self))


@dataclass(frozen=True, slots=True)
class Qwen3GroundingManifest:
    identity: str
    source_data_manifest_sha256: str
    cross_image_probes: tuple[Qwen3CrossImageProbeSpec, ...]
    target_presence_probes: tuple[Qwen3TargetPresenceProbeSpec, ...]
    schema_version: str = QWEN3_GROUNDING_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.identity, name="grounding manifest identity")
        _digest(self.source_data_manifest_sha256, name="source_data_manifest_sha256")
        if not self.cross_image_probes or not self.target_presence_probes:
            raise ValueError("grounding manifest requires both diagnostic families")
        if len({probe.pair_id for probe in self.cross_image_probes}) != len(
            self.cross_image_probes
        ):
            raise ValueError("cross-image probe IDs must be unique")
        if len({probe.pair_id for probe in self.target_presence_probes}) != len(
            self.target_presence_probes
        ):
            raise ValueError("target-presence probe IDs must be unique")
        if self.schema_version != QWEN3_GROUNDING_MANIFEST_SCHEMA_VERSION:
            raise ValueError("grounding manifest schema mismatch")

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "identity": self.identity,
                "source_data_manifest_sha256": self.source_data_manifest_sha256,
                "cross_image_probes": [
                    _dataclass_payload(probe) for probe in self.cross_image_probes
                ],
                "target_presence_probes": [
                    _dataclass_payload(probe) for probe in self.target_presence_probes
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class Qwen3GroundingBuild:
    manifest_identity: str
    manifest_sha256: str
    cross_image: Qwen3CounterfactualBuild
    target_presence_cases: tuple[NativeTargetPresenceCase, ...]
    target_presence_materializer: Qwen3NativeInjectedRequestMaterializer
    schema_version: str = QWEN3_GROUNDING_BUILD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.manifest_identity, name="grounding build identity")
        _digest(self.manifest_sha256, name="grounding build SHA256")
        if not isinstance(self.cross_image, Qwen3CounterfactualBuild):
            raise TypeError("grounding build requires cross-image build")
        if not self.target_presence_cases:
            raise ValueError("grounding build requires target-presence cases")
        if self.schema_version != QWEN3_GROUNDING_BUILD_SCHEMA_VERSION:
            raise ValueError("grounding build schema mismatch")


def load_qwen3_grounding_manifest(path: str | Path) -> Qwen3GroundingManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "identity",
        "source_data_manifest_sha256",
        "cross_image_probes",
        "target_presence_probes",
    }:
        raise ValueError("Qwen3 grounding manifest fields differ")
    cross_fields = set(Qwen3CrossImageProbeSpec.__dataclass_fields__)
    presence_fields = set(Qwen3TargetPresenceProbeSpec.__dataclass_fields__)
    cross_raw = _array(payload["cross_image_probes"], name="cross_image_probes")
    presence_raw = _array(
        payload["target_presence_probes"], name="target_presence_probes"
    )
    cross: list[Qwen3CrossImageProbeSpec] = []
    for raw in cross_raw:
        if not isinstance(raw, Mapping) or set(raw) != cross_fields:
            raise ValueError("cross-image probe fields differ")
        cross.append(Qwen3CrossImageProbeSpec(**dict(raw)))
    presence: list[Qwen3TargetPresenceProbeSpec] = []
    for raw in presence_raw:
        if not isinstance(raw, Mapping) or set(raw) != presence_fields:
            raise ValueError("target-presence probe fields differ")
        presence.append(Qwen3TargetPresenceProbeSpec(**dict(raw)))
    return Qwen3GroundingManifest(
        identity=payload["identity"],
        source_data_manifest_sha256=payload["source_data_manifest_sha256"],
        cross_image_probes=tuple(cross),
        target_presence_probes=tuple(presence),
        schema_version=payload["schema_version"],
    )


class Qwen3GroundingDiagnosticBuilder:
    def __init__(
        self,
        *,
        runtime: Qwen3RepresentationRuntime,
        group_builder: Qwen3NativeRepresentationGroupBuilder,
    ) -> None:
        if not isinstance(runtime, Qwen3RepresentationRuntime):
            raise TypeError("grounding builder requires Qwen3 runtime")
        if not isinstance(group_builder, Qwen3NativeRepresentationGroupBuilder):
            raise TypeError("grounding builder requires native group builder")
        if group_builder.runtime is not runtime:
            raise ValueError("grounding builder runtime differs from group builder")
        self.runtime = runtime
        self.group_builder = group_builder
        self.prompt = group_builder.prompt

    def build(
        self,
        *,
        manifest: Qwen3GroundingManifest,
        data_manifest_sha256: str,
        samples: Sequence[RepresentationTrainingSample],
    ) -> Qwen3GroundingBuild:
        if not isinstance(manifest, Qwen3GroundingManifest):
            raise TypeError("manifest must be Qwen3GroundingManifest")
        if manifest.source_data_manifest_sha256 != data_manifest_sha256:
            raise ValueError("grounding manifest identifies another dataset")
        source_by_id = {sample.sample_id: sample for sample in samples}
        if len(source_by_id) != len(samples):
            raise ValueError("grounding source dataset repeats sample IDs")

        cross_samples: list[RepresentationTrainingSample] = []
        cross_observations: dict[str, RepresentationCandidateObservation] = {}
        cross_specs: list[Qwen3CounterfactualPairSpec] = []
        target_cases: list[NativeTargetPresenceCase] = []
        target_context_records: list[Qwen3NativeDOnlyContextRecord] = []

        modes = tuple(
            (module, module.training) for module in self.runtime.adapter.modules()
        )
        try:
            self.runtime.adapter.eval()
            with torch.no_grad():
                for probe in manifest.cross_image_probes:
                    source_a = _bound_source(
                        source_by_id,
                        probe.source_sample_a_id,
                        probe.source_sample_a_content_sha256,
                        probe.source_image_a_sha256,
                    )
                    source_b = _bound_source(
                        source_by_id,
                        probe.source_sample_b_id,
                        probe.source_sample_b_content_sha256,
                        probe.source_image_b_sha256,
                    )
                    sample_a = _probe_sample(
                        source_a,
                        sample_id=f"{manifest.identity}:{probe.pair_id}:a",
                        question=probe.question,
                        target=probe.target,
                        value=probe.expected_value_a,
                    )
                    sample_b = _probe_sample(
                        source_b,
                        sample_id=f"{manifest.identity}:{probe.pair_id}:b",
                        question=probe.question,
                        target=probe.target,
                        value=probe.expected_value_b,
                    )
                    candidate_a = self._candidate(sample_a, source_a)
                    candidate_b = self._candidate(sample_b, source_b)
                    cross_samples.extend((sample_a, sample_b))
                    cross_observations[sample_a.sample_id] = candidate_a
                    cross_observations[sample_b.sample_id] = candidate_b
                    cross_specs.append(
                        Qwen3CounterfactualPairSpec(
                            pair_id=probe.pair_id,
                            sample_a_id=sample_a.sample_id,
                            sample_b_id=sample_b.sample_id,
                            expected_value_a=probe.expected_value_a,
                            expected_value_b=probe.expected_value_b,
                            pair_audit_identity=probe.pair_audit_identity,
                        )
                    )

                for probe in manifest.target_presence_probes:
                    source = _bound_source(
                        source_by_id,
                        probe.source_sample_id,
                        probe.source_sample_content_sha256,
                        probe.source_image_sha256,
                    )
                    positive = _probe_sample(
                        source,
                        sample_id=f"{manifest.identity}:{probe.pair_id}:positive",
                        question=probe.question,
                        target=probe.positive_target,
                        value=probe.present_value,
                    )
                    negative = _probe_sample(
                        source,
                        sample_id=f"{manifest.identity}:{probe.pair_id}:negative",
                        question=probe.question,
                        target=probe.negative_target,
                        value=probe.not_present_value,
                    )
                    group = self.group_builder(
                        (positive, negative),
                        self.runtime.adapter,
                        collective_candidate_count=2,
                    )
                    positive_candidate, negative_candidate = group.candidates
                    positive_record = self._context_record(
                        probe=probe,
                        variant="positive",
                        sample=positive,
                        candidate=positive_candidate,
                    )
                    negative_record = self._context_record(
                        probe=probe,
                        variant="negative",
                        sample=negative,
                        candidate=negative_candidate,
                    )
                    target_context_records.extend((positive_record, negative_record))
                    target_cases.append(
                        NativeTargetPresenceCase(
                            case_id=f"{manifest.identity}:{probe.pair_id}",
                            pair_identity=probe.content_sha256,
                            source_sample_id=probe.source_sample_id,
                            source_image_sha256=probe.source_image_sha256,
                            positive_target=probe.positive_target,
                            negative_target=probe.negative_target,
                            positive_context=positive_record.context,
                            negative_context=negative_record.context,
                            positive_observation_identity=_observation_identity(
                                positive_candidate
                            ),
                            negative_observation_identity=_observation_identity(
                                negative_candidate
                            ),
                            positive_observation=_detached_visual(
                                positive_candidate.visual
                            ),
                            negative_observation=_detached_visual(
                                negative_candidate.visual
                            ),
                        )
                    )
        finally:
            for module, was_training in modes:
                module.training = was_training

        cross_manifest = Qwen3CounterfactualManifest(
            identity=f"{manifest.identity}:cross-image",
            source_data_manifest_sha256=data_manifest_sha256,
            pairs=tuple(cross_specs),
        )
        cross_build = Qwen3CounterfactualCaseBuilder(
            runtime=self.runtime,
            prompt=self.prompt,
            image_max_pixels=self.group_builder.image_max_pixels,
        ).build(
            manifest=cross_manifest,
            data_manifest_sha256=data_manifest_sha256,
            samples=tuple(cross_samples),
            observations=cross_observations,
        )
        target_materializer = Qwen3NativeInjectedRequestMaterializer(
            runtime=self.runtime,
            contexts=tuple(target_context_records),
        )
        self.runtime.assert_bound_invariants()
        return Qwen3GroundingBuild(
            manifest_identity=manifest.identity,
            manifest_sha256=manifest.content_sha256,
            cross_image=cross_build,
            target_presence_cases=tuple(target_cases),
            target_presence_materializer=target_materializer,
        )

    def _candidate(
        self,
        probe: RepresentationTrainingSample,
        source: RepresentationTrainingSample,
    ) -> RepresentationCandidateObservation:
        auxiliary_target = source.target
        if auxiliary_target == probe.target:
            auxiliary_target = _AUXILIARY_TARGET
        auxiliary = replace(
            source,
            sample_id=f"{probe.sample_id}:auxiliary",
            question=probe.question,
            target=auxiliary_target,
            evidence_description="Auxiliary diagnostic batching row.",
            short_answer="AUXILIARY",
        )
        group = self.group_builder(
            (probe, auxiliary),
            self.runtime.adapter,
            collective_candidate_count=2,
        )
        return group.candidates[0]

    def _context_record(
        self,
        *,
        probe: Qwen3TargetPresenceProbeSpec,
        variant: str,
        sample: RepresentationTrainingSample,
        candidate: RepresentationCandidateObservation,
    ) -> Qwen3NativeDOnlyContextRecord:
        geometry = _qwen3_geometry_carrier(
            self.runtime.processor, candidate.image_grid_thw
        )
        prefix = materialize_qwen3_d_only_processor_prefix(
            processor=self.runtime.processor,
            renderer=self.runtime.renderer,
            sample=sample,
            prompt=self.prompt,
            geometry_image=geometry,
            image_max_pixels=self.group_builder.image_max_pixels,
        )
        if len(prefix.d_positions) != candidate.visual.main.shape[1]:
            raise ValueError("target-presence D shape differs from native geometry")
        device = _runtime_language_device(self.runtime)
        input_ids = prefix.input_ids.to(device=device)
        attention_mask = prefix.attention_mask.to(device=device)
        grid = prefix.image_grid_thw.to(device=device)
        position_ids = _qwen3_position_ids(
            self.runtime.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=grid,
        )
        context_id = _canonical_sha256(
            {
                "schema": "qwen3_target_presence_context_identity_v1",
                "probe_sha256": probe.content_sha256,
                "variant": variant,
                "prompt_identity": self.prompt.identity,
                "transcript_identity": prefix.transcript.token_ids_sha256,
                "input_ids": tuple(int(value) for value in input_ids[0].tolist()),
                "position_ids_shape": tuple(int(value) for value in position_ids.shape),
                "position_ids_sha256": sha256(
                    position_ids.detach().cpu().to(torch.int64).numpy().tobytes()
                ).hexdigest(),
                "image_grid_thw": tuple(int(value) for value in grid[0].tolist()),
                "d_positions": prefix.d_positions,
            }
        )
        context = NativeDOnlyContext(
            context_id=context_id,
            transcript_identity=prefix.transcript.token_ids_sha256,
            family="qwen3_vl",
            input_ids=input_ids.detach().clone(),
            attention_mask=attention_mask.detach().clone(),
            position_ids=position_ids.detach().clone(),
            d_positions=prefix.d_positions,
            image_grid_thw=(tuple(int(value) for value in grid[0].tolist()),),
        )
        return Qwen3NativeDOnlyContextRecord(
            pair_id=f"{probe.pair_id}:{variant}",
            prompt_identity=self.prompt.identity,
            transcript=prefix.transcript,
            context=context,
        )


def _probe_sample(
    source: RepresentationTrainingSample,
    *,
    sample_id: str,
    question: str,
    target: str,
    value: str,
) -> RepresentationTrainingSample:
    return replace(
        source,
        sample_id=sample_id,
        question=question,
        target=target,
        evidence_description=f"The audited diagnostic value is {value}.",
        short_answer=value,
        choices=(),
    )


def _bound_source(
    sources: Mapping[str, RepresentationTrainingSample],
    sample_id: str,
    content_sha256: str,
    image_sha256: str,
) -> RepresentationTrainingSample:
    try:
        source = sources[sample_id]
    except KeyError as error:
        raise ValueError(
            f"grounding source sample is unavailable: {sample_id}"
        ) from error
    if source.content_sha256 != content_sha256:
        raise ValueError("grounding source sample content SHA256 drifted")
    image_path = Path(source.image).resolve(strict=True)
    if (
        not image_path.is_file()
        or sha256(image_path.read_bytes()).hexdigest() != image_sha256
    ):
        raise ValueError("grounding source image SHA256 drifted")
    return source


def _dataclass_payload(value: object) -> dict[str, Any]:
    fields = value.__dataclass_fields__  # type: ignore[attr-defined]
    return {name: getattr(value, name) for name in fields}


def _array(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TypeError(f"{name} must be a non-empty array")
    return value


def _text(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _digest(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


__all__ = [
    "QWEN3_CROSS_IMAGE_PROBE_SCHEMA_VERSION",
    "QWEN3_GROUNDING_BUILD_SCHEMA_VERSION",
    "QWEN3_GROUNDING_MANIFEST_SCHEMA_VERSION",
    "QWEN3_TARGET_PRESENCE_PROBE_SCHEMA_VERSION",
    "TARGET_PRESENCE_QUESTION",
    "Qwen3CrossImageProbeSpec",
    "Qwen3GroundingBuild",
    "Qwen3GroundingDiagnosticBuilder",
    "Qwen3GroundingManifest",
    "Qwen3TargetPresenceProbeSpec",
    "load_qwen3_grounding_manifest",
]
