"""Executable scheduled T1 dataset for the PRL13 native DeepEyes control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from tgvf_rl.data.deepeyes_official_schedule import (
    DEEPEYES_CANDIDATE_SHA256,
    DEEPEYES_CANDIDATE_SIDECAR,
    DEEPEYES_PROBE_SEED,
    DEEPEYES_T1_CONTENT_SHA256,
    DEEPEYES_T1_MANIFEST_FILE_SHA256,
    DEEPEYES_T1_ROOT,
    DEEPEYES_T1_SAMPLE_COUNT,
    DEEPEYES_T1_SAMPLES_SHA256,
    DEEPEYES_TRAIN_SEED,
)
from tgvf_rl.data.deepeyes_official_schedule_index import (
    DEEPEYES_SCHEDULE_INDEX_FILE_SHA256,
    DEEPEYES_SCHEDULE_INDEX_IDENTITY_SHA256,
    DEEPEYES_SCHEDULE_INDEX_PATH,
    DeepEyesScheduleIndex,
    load_deepeyes_schedule_index,
)
from tgvf_rl.policy.deepeyes_official_protocol import (
    THINKLITE_PROMPT_IDENTITY,
    VISUAL_PROMPT_IDENTITY,
    build_thinklite_messages,
    build_visual_messages,
)


DEEPEYES_OFFICIAL_DATASET_SCHEMA = "tgvf.verl-deepeyes-official-dataset.v1"
DEEPEYES_OFFICIAL_DATASET_CLASS = (
    "tgvf_rl.framework.verl.deepeyes_official_dataset."
    "TGVFDeepEyesOfficialDataset"
)
DEEPEYES_TRAIN_SENTINEL = DEEPEYES_T1_ROOT / "prl13-train.schedule"
DEEPEYES_PROBE_SENTINEL = DEEPEYES_T1_ROOT / "prl13-probe.schedule"
DEEPEYES_SMOKE_SENTINEL = DEEPEYES_T1_ROOT / "prl13-smoke.schedule"

DatasetSplit = Literal["train", "probe", "smoke"]


def _plain_mapping(value: object, name: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    try:
        from omegaconf import OmegaConf

        plain = OmegaConf.to_container(value, resolve=True)
    except (ImportError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a mapping") from error
    if not isinstance(plain, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return plain


def _config_value(config: object, name: str) -> object:
    if isinstance(config, Mapping):
        return config.get(name)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(name)
    return getattr(config, name)


@dataclass(frozen=True, slots=True)
class DeepEyesOfficialDatasetBinding:
    root: Path
    candidate_sidecar_path: Path
    manifest_file_sha256: str
    content_sha256: str
    samples_sha256: str
    candidate_sidecar_sha256: str
    expected_sample_count: int
    schedule_mode: Literal["stratified", "natural"]
    schedule_seed: int
    probe_seed: int
    visual_prompt_bundle_sha256: str
    thinklite_prompt_bundle_sha256: str
    schema_version: str = DEEPEYES_OFFICIAL_DATASET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(
            self, "candidate_sidecar_path", Path(self.candidate_sidecar_path)
        )
        expected = {
            "root": DEEPEYES_T1_ROOT,
            "candidate_sidecar_path": DEEPEYES_CANDIDATE_SIDECAR,
            "manifest_file_sha256": DEEPEYES_T1_MANIFEST_FILE_SHA256,
            "content_sha256": DEEPEYES_T1_CONTENT_SHA256,
            "samples_sha256": DEEPEYES_T1_SAMPLES_SHA256,
            "candidate_sidecar_sha256": DEEPEYES_CANDIDATE_SHA256,
            "expected_sample_count": DEEPEYES_T1_SAMPLE_COUNT,
            "schedule_seed": DEEPEYES_TRAIN_SEED,
            "probe_seed": DEEPEYES_PROBE_SEED,
            "visual_prompt_bundle_sha256": VISUAL_PROMPT_IDENTITY.bundle_sha256,
            "thinklite_prompt_bundle_sha256": THINKLITE_PROMPT_IDENTITY.bundle_sha256,
            "schema_version": DEEPEYES_OFFICIAL_DATASET_SCHEMA,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"official dataset {name} differs")
        if self.schedule_mode not in {"stratified", "natural"}:
            raise ValueError("official dataset schedule_mode differs")

    @classmethod
    def from_config(cls, value: object) -> "DeepEyesOfficialDatasetBinding":
        mapping = dict(_plain_mapping(value, "data.deepeyes_official"))
        required = {
            "root",
            "candidate_sidecar_path",
            "manifest_file_sha256",
            "content_sha256",
            "samples_sha256",
            "candidate_sidecar_sha256",
            "expected_sample_count",
            "schedule_mode",
            "schedule_seed",
            "probe_seed",
            "visual_prompt_bundle_sha256",
            "thinklite_prompt_bundle_sha256",
            "schema_version",
        }
        if set(mapping) != required:
            raise ValueError("data.deepeyes_official fields differ")
        return cls(**mapping)


@lru_cache(maxsize=1)
def _verified_schedule_index() -> DeepEyesScheduleIndex:
    return load_deepeyes_schedule_index(
        DEEPEYES_SCHEDULE_INDEX_PATH,
        expected_file_sha256=DEEPEYES_SCHEDULE_INDEX_FILE_SHA256,
        expected_identity_sha256=DEEPEYES_SCHEDULE_INDEX_IDENTITY_SHA256,
    )


@lru_cache(maxsize=None)
def _observed_image_sha256(path_value: str) -> str:
    """Hash one consumed image exactly once in this Dataset process."""

    image_path = Path(path_value)
    if (
        not image_path.is_absolute()
        or image_path.is_symlink()
        or not image_path.is_file()
        or image_path.resolve(strict=True) != image_path
    ):
        raise ValueError("schedule-index source image is unsafe or missing")
    digest = hashlib.sha256()
    with image_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_from_files(data_files: str | Sequence[str]) -> DatasetSplit:
    paths = (
        (data_files,)
        if isinstance(data_files, (str, Path))
        else tuple(data_files)
    )
    normalized = tuple(str(path) for path in paths)
    if normalized == (str(DEEPEYES_TRAIN_SENTINEL),):
        return "train"
    if normalized == (str(DEEPEYES_PROBE_SENTINEL),):
        return "probe"
    if normalized == (str(DEEPEYES_SMOKE_SENTINEL),):
        return "smoke"
    raise ValueError("official dataset files do not select train/probe/smoke schedule")


class TGVFDeepEyesOfficialDataset(Dataset):
    """Return exact scheduled rows; no Parquet conversion or new filtering."""

    def __init__(
        self,
        data_files: str | Sequence[str],
        tokenizer: object,
        config: object,
        processor: object | None = None,
        max_samples: int = -1,
    ) -> None:
        del tokenizer
        if processor is None:
            raise TypeError("official visual dataset requires a processor")
        if max_samples != -1:
            raise ValueError("official schedule forbids max_samples filtering")
        binding = DeepEyesOfficialDatasetBinding.from_config(
            _config_value(config, "deepeyes_official")
        )
        if binding.schedule_mode != "stratified":
            raise ValueError(
                "official schedule index currently binds only the stratified arm"
            )
        split = _split_from_files(data_files)
        schedule = _verified_schedule_index()
        if split == "train":
            selected = schedule.train
        elif split == "probe":
            selected = schedule.probe
        else:
            selected = schedule.smoke
        self.binding = binding
        self.split = split
        self.schedule_index_file_sha256 = schedule.file_sha256
        self.schedule_index_identity_sha256 = schedule.identity_sha256
        self.schedule_identity_sha256 = schedule.schedule_identity_sha256
        self.probe_manifest = schedule.probe_manifest
        self.samples = selected
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        sample = self.samples[index]
        observed_image_sha256 = _observed_image_sha256(str(sample.image_path))
        if observed_image_sha256 != sample.image_sha256:
            raise ValueError(
                f"schedule-index source image SHA-256 differs for {sample.sample_id}"
            )
        if sample.data_source == "thinklite":
            raw_prompt: list[dict[str, Any]] = [
                _copy_message(message)
                for message in build_thinklite_messages(
                    sample.question,
                    image=str(sample.image_path),
                    task_kind=sample.task_kind,
                )
            ]
            need_tools_kwargs = False
        else:
            raw_prompt = [
                _copy_message(message)
                for message in build_visual_messages(
                    sample.question, image=str(sample.image_path)
                )
            ]
            need_tools_kwargs = True
        tools_kwargs = dict(sample.tools_kwargs)
        extra_info = {
            "index": index,
            "sample_id": sample.sample_id,
            "question": sample.question,
            "task_kind": sample.task_kind,
            "tools_kwargs": tools_kwargs,
            "need_tools_kwargs": need_tools_kwargs,
            "schedule_index_file_sha256": self.schedule_index_file_sha256,
            "schedule_index_identity_sha256": self.schedule_index_identity_sha256,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "dataset_split": self.split,
        }
        if self.split == "smoke":
            extra_info["smoke_expectation"] = (
                "crop_possible"
                if sample.data_source == "vstar"
                else "direct_no_call"
                if sample.data_source == "arxivqa"
                else "no_tool"
            )
        return {
            "raw_prompt": raw_prompt,
            "data_source": sample.data_source,
            "reward_model": {"ground_truth": sample.ground_truth},
            "extra_info": extra_info,
            "sample_id": sample.sample_id,
            "question": sample.question,
            "task_kind": sample.task_kind,
            "source_image_path": str(sample.image_path),
            "source_image_sha256": sample.image_sha256,
            "candidate_sha256": sample.candidate_sha256,
            "agent_name": sample.agent_name,
            "tools_kwargs": tools_kwargs,
            "index": index,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size: int,
        config: object,
    ) -> object:
        """Delegate to the pinned veRL multimodal processor implementation."""

        try:
            from verl.utils.dataset.rl_dataset import RLHFDataset
        except ImportError as error:  # pragma: no cover - formal environment gate
            raise RuntimeError("official dataset requires pinned veRL") from error
        return await RLHFDataset.process_vision_info(
            messages, image_patch_size, config
        )


def _copy_message(message: Mapping[str, Any]) -> dict[str, Any]:
    content = message["content"]
    return {
        **message,
        "content": (
            [dict(item) for item in content]
            if not isinstance(content, str)
            else content
        ),
    }


__all__ = [
    "DEEPEYES_OFFICIAL_DATASET_CLASS",
    "DEEPEYES_OFFICIAL_DATASET_SCHEMA",
    "DEEPEYES_PROBE_SENTINEL",
    "DEEPEYES_SMOKE_SENTINEL",
    "DEEPEYES_TRAIN_SENTINEL",
    "DeepEyesOfficialDatasetBinding",
    "TGVFDeepEyesOfficialDataset",
]
