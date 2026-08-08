from __future__ import annotations

from collections import Counter
from pathlib import Path

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
    build_deepeyes_schedule,
)
from tgvf_rl.data.deepeyes_official_schedule_index import (
    DeepEyesScheduleIndex,
)
from tgvf_rl.framework.verl import deepeyes_official_dataset as dataset_module
from tgvf_rl.framework.verl.deepeyes_official_dataset import (
    DEEPEYES_OFFICIAL_DATASET_SCHEMA,
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_SMOKE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
    TGVFDeepEyesOfficialDataset,
)
from tgvf_rl.policy.deepeyes_official_protocol import (
    DEEPEYES_THINKLITE_AGENT_NAME,
    DEEPEYES_VISUAL_AGENT_NAME,
    SYSTEM_PROMPT_V2,
    THINKLITE_PROMPT_IDENTITY,
    USER_PROMPT_V2,
    VISUAL_PROMPT_IDENTITY,
)

from tests.data.test_deepeyes_official_schedule import synthetic_official_pool


def _config(mode: str) -> dict[str, object]:
    return {
        "deepeyes_official": {
            "schema_version": DEEPEYES_OFFICIAL_DATASET_SCHEMA,
            "root": str(DEEPEYES_T1_ROOT),
            "candidate_sidecar_path": str(DEEPEYES_CANDIDATE_SIDECAR),
            "manifest_file_sha256": DEEPEYES_T1_MANIFEST_FILE_SHA256,
            "content_sha256": DEEPEYES_T1_CONTENT_SHA256,
            "samples_sha256": DEEPEYES_T1_SAMPLES_SHA256,
            "candidate_sidecar_sha256": DEEPEYES_CANDIDATE_SHA256,
            "expected_sample_count": DEEPEYES_T1_SAMPLE_COUNT,
            "schedule_mode": mode,
            "schedule_seed": DEEPEYES_TRAIN_SEED,
            "probe_seed": DEEPEYES_PROBE_SEED,
            "visual_prompt_bundle_sha256": VISUAL_PROMPT_IDENTITY.bundle_sha256,
            "thinklite_prompt_bundle_sha256": THINKLITE_PROMPT_IDENTITY.bundle_sha256,
        }
    }


def _synthetic_index() -> DeepEyesScheduleIndex:
    schedule = build_deepeyes_schedule(
        synthetic_official_pool(), mode="stratified"
    )
    train = tuple(
        schedule.samples[index] for batch in schedule.batches for index in batch
    )
    excluded = set(schedule.probe_indices)
    excluded.update(index for batch in schedule.batches for index in batch)
    available = {
        source: sorted(
            (
                sample
                for index, sample in enumerate(schedule.samples)
                if index not in excluded and sample.data_source == source
            ),
            key=lambda sample: sample.sample_id,
        )
        for source in ("vstar", "arxivqa", "thinklite")
    }
    smoke = (
        available["vstar"][0],
        available["arxivqa"][0],
        available["thinklite"][0],
        available["vstar"][1],
    )
    return DeepEyesScheduleIndex(
        path=Path("/synthetic-index.json"),
        file_sha256="c" * 64,
        identity_sha256="d" * 64,
        schedule_identity_sha256=schedule.identity_sha256,
        probe_manifest=schedule.probe_manifest,
        train=train,
        probe=schedule.probe,
        smoke=smoke,
    )


def _bind_synthetic_index(monkeypatch) -> None:
    index = _synthetic_index()
    monkeypatch.setattr(dataset_module, "_verified_schedule_index", lambda: index)
    monkeypatch.setattr(
        dataset_module, "_observed_image_sha256", lambda _path: "b" * 64
    )


def test_custom_dataset_materializes_exact_80_step_schedule_and_probe(
    monkeypatch,
) -> None:
    _bind_synthetic_index(monkeypatch)
    train = TGVFDeepEyesOfficialDataset(
        [str(DEEPEYES_TRAIN_SENTINEL)],
        tokenizer=object(),
        processor=object(),
        config=_config("stratified"),
    )
    probe = TGVFDeepEyesOfficialDataset(
        [str(DEEPEYES_PROBE_SENTINEL)],
        tokenizer=object(),
        processor=object(),
        config=_config("stratified"),
    )
    assert len(train) == 20_480
    assert len(probe) == 256
    assert not {sample.sample_id for sample in train.samples}.intersection(
        sample.sample_id for sample in probe.samples
    )
    assert Counter(sample.data_source for sample in train.samples[:256]) == Counter(
        {"vstar": 120, "arxivqa": 77, "thinklite": 59}
    )


def test_rows_use_official_prompt_agent_and_nested_tool_route(monkeypatch) -> None:
    _bind_synthetic_index(monkeypatch)
    dataset = TGVFDeepEyesOfficialDataset(
        str(DEEPEYES_TRAIN_SENTINEL),
        tokenizer=object(),
        processor=object(),
        config=_config("stratified"),
    )
    visual_index = next(
        index
        for index, sample in enumerate(dataset.samples)
        if sample.data_source == "vstar"
    )
    visual = dataset[visual_index]
    assert visual["agent_name"] == DEEPEYES_VISUAL_AGENT_NAME
    assert visual["raw_prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT_V2}
    assert visual["raw_prompt"][1]["content"][1]["text"].endswith(USER_PROMPT_V2)
    assert visual["tools_kwargs"] == visual["extra_info"]["tools_kwargs"]
    assert visual["extra_info"]["question"] == visual["question"]

    thinklite_index = next(
        index
        for index, sample in enumerate(dataset.samples)
        if sample.data_source == "thinklite"
    )
    thinklite = dataset[thinklite_index]
    assert thinklite["agent_name"] == DEEPEYES_THINKLITE_AGENT_NAME
    assert thinklite["tools_kwargs"] == {}
    assert thinklite["extra_info"]["need_tools_kwargs"] is False
    content = thinklite["raw_prompt"][0]["content"]
    assert content[0] == {"type": "image", "image": thinklite["source_image_path"]}
    assert ("\\boxed{}" in content[1]["text"]) == (
        thinklite["task_kind"] == "math"
    )


def test_smoke_split_is_four_source_covering_rows_disjoint_from_formal(monkeypatch) -> None:
    _bind_synthetic_index(monkeypatch)
    train = TGVFDeepEyesOfficialDataset(
        str(DEEPEYES_TRAIN_SENTINEL),
        tokenizer=object(),
        processor=object(),
        config=_config("stratified"),
    )
    probe = TGVFDeepEyesOfficialDataset(
        str(DEEPEYES_PROBE_SENTINEL),
        tokenizer=object(),
        processor=object(),
        config=_config("stratified"),
    )
    smoke = TGVFDeepEyesOfficialDataset(
        str(DEEPEYES_SMOKE_SENTINEL),
        tokenizer=object(),
        processor=object(),
        config=_config("stratified"),
    )
    assert [sample.data_source for sample in smoke.samples] == [
        "vstar",
        "arxivqa",
        "thinklite",
        "vstar",
    ]
    formal_ids = {sample.sample_id for sample in (*train.samples, *probe.samples)}
    assert not formal_ids.intersection(sample.sample_id for sample in smoke.samples)
    assert smoke[0]["extra_info"]["smoke_expectation"] == "crop_possible"
    assert smoke[1]["extra_info"]["smoke_expectation"] == "direct_no_call"
    assert smoke[2]["extra_info"]["smoke_expectation"] == "no_tool"
