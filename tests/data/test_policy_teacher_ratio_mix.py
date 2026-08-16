from __future__ import annotations

from types import MappingProxyType

import pytest

from tgvf_rl.data import policy_teacher_quarter_mix as legacy
from tgvf_rl.data import policy_teacher_ratio_mix as module
from tgvf_rl.data.policy_teacher_ratio_mix import (
    PolicyTeacherRatioMixRuntimeBinding,
    materialize_policy_teacher_ratio_mix,
    policy_teacher_ratio_mix_iteration_identity_sha256,
    policy_teacher_ratio_mix_profile,
)


@pytest.mark.parametrize(
    (
        "teacher_percentage",
        "base_macro_cycle",
        "teacher_per_macro",
        "teacher_sources",
        "cadence_unit",
    ),
    (
        (
            25,
            ({"vstar": 90, "arxivqa": 58, "thinklite": 44},),
            64,
            {
                "chartqa": 455,
                "docvqa": 864,
                "textocr": 816,
                "textvqa": 840,
                "visual_genome": 2_145,
            },
            ("base", "base", "base", "teacher"),
        ),
        (
            50,
            (
                {"vstar": 60, "arxivqa": 39, "thinklite": 29},
                {"vstar": 60, "arxivqa": 38, "thinklite": 30},
            ),
            128,
            {
                "chartqa": 909,
                "docvqa": 1_728,
                "textocr": 1_633,
                "textvqa": 1_680,
                "visual_genome": 4_290,
            },
            ("base", "teacher"),
        ),
        (
            100,
            ({"vstar": 0, "arxivqa": 0, "thinklite": 0},),
            256,
            {
                "chartqa": 1_818,
                "docvqa": 3_456,
                "textocr": 3_266,
                "textvqa": 3_360,
                "visual_genome": 8_580,
            },
            ("teacher",),
        ),
    ),
)
def test_ratio_profiles_bind_exact_source_quotas_and_bs16_cadence(
    teacher_percentage: int,
    base_macro_cycle: tuple[dict[str, int], ...],
    teacher_per_macro: int,
    teacher_sources: dict[str, int],
    cadence_unit: tuple[str, ...],
) -> None:
    profile = policy_teacher_ratio_mix_profile(teacher_percentage)

    assert tuple(
        dict(counts) for counts in profile.base_macro_source_counts_cycle
    ) == base_macro_cycle
    assert profile.teacher_per_macro == teacher_per_macro
    assert dict(profile.teacher_source_counts) == teacher_sources
    assert profile.role_cadence == cadence_unit * (16 // len(cadence_unit))
    assert profile.teacher_count == 20_480 * teacher_percentage // 100
    assert profile.base_count + profile.teacher_count == 20_480


def test_ratio_profiles_are_nested_by_each_parent_source() -> None:
    teacher25 = policy_teacher_ratio_mix_profile(25)
    teacher50 = policy_teacher_ratio_mix_profile(50)
    teacher100 = policy_teacher_ratio_mix_profile(100)

    for source in teacher25.teacher_source_counts:
        assert (
            teacher25.teacher_source_counts[source]
            <= teacher50.teacher_source_counts[source]
            <= teacher100.teacher_source_counts[source]
        )
    for source in teacher25.base_source_counts:
        assert (
            teacher25.base_source_counts[source]
            >= teacher50.base_source_counts[source]
            >= teacher100.base_source_counts[source]
        )


def test_teacher50_alternates_tied_macro_quota_and_preserves_global_half() -> None:
    profile = policy_teacher_ratio_mix_profile(50)

    assert dict(profile.macro_source_counts_for(0)) == {
        "vstar": 60,
        "arxivqa": 39,
        "thinklite": 29,
        "teacher": 128,
    }
    assert dict(profile.macro_source_counts_for(1)) == {
        "vstar": 60,
        "arxivqa": 38,
        "thinklite": 30,
        "teacher": 128,
    }
    assert dict(profile.macro_source_counts_for(2)) == dict(
        profile.macro_source_counts_for(0)
    )
    assert dict(profile.base_source_counts) == {
        "vstar": 4_800,
        "arxivqa": 3_080,
        "thinklite": 2_360,
    }


@pytest.mark.parametrize("teacher_percentage", (-1, 0, 75, 101, True))
def test_ratio_profile_rejects_unapproved_percentages(
    teacher_percentage: int,
) -> None:
    with pytest.raises(ValueError, match="25, 50, or 100"):
        policy_teacher_ratio_mix_profile(teacher_percentage)


def test_teacher_selector_retains_the_prl22_hash_namespace() -> None:
    record = {
        "sample_id": "teacher:fixture:1",
        "extra_info": {"source_dataset": "chartqa"},
    }

    assert module._stable_teacher_key(record, 42) == legacy._stable_teacher_key(
        record, 42
    )


def test_teacher25_materializer_is_the_legacy_byte_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    output = (tmp_path / "teacher25").resolve()
    observed: dict[str, object] = {}
    legacy_result = legacy.PolicyTeacherQuarterMixMaterializationResult(
        output_root=output,
        sample_count=20_480,
        samples_sha256="1" * 64,
        content_sha256="2" * 64,
        manifest_file_sha256="3" * 64,
        iteration_identity_sha256="4" * 64,
        schedule_seed=42,
        source_counts=MappingProxyType(
            {"vstar": 7_200, "arxivqa": 4_640, "thinklite": 3_520, "teacher": 5_120}
        ),
        teacher_source_counts=MappingProxyType(
            {
                "chartqa": 455,
                "docvqa": 864,
                "textocr": 816,
                "textvqa": 840,
                "visual_genome": 2_145,
            }
        ),
    )

    def fake_legacy(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return legacy_result

    monkeypatch.setattr(legacy, "materialize_policy_teacher_quarter_mix", fake_legacy)
    result = materialize_policy_teacher_ratio_mix(
        output,
        teacher_percentage=25,
        teacher_root="/teacher",
        schedule_index_path="/schedule",
        schedule_seed=42,
    )

    assert observed == {
        "args": (output,),
        "kwargs": {
            "teacher_root": "/teacher",
            "schedule_index_path": "/schedule",
            "schedule_seed": 42,
        },
    }
    assert result.manifest_file_sha256 == legacy_result.manifest_file_sha256
    assert result.content_sha256 == legacy_result.content_sha256
    assert result.samples_sha256 == legacy_result.samples_sha256
    assert result.iteration_identity_sha256 == (
        legacy_result.iteration_identity_sha256
    )


def test_teacher25_iteration_identity_is_byte_compatible_with_prl22() -> None:
    ratio_binding = PolicyTeacherRatioMixRuntimeBinding(
        manifest_file_sha256="1" * 64,
        content_sha256="2" * 64,
        schedule_seed=42,
        expected_sample_count=20_480,
        teacher_percentage=25,
    )
    legacy_binding = legacy.PolicyTeacherQuarterMixRuntimeBinding(
        manifest_file_sha256="1" * 64,
        content_sha256="2" * 64,
        schedule_seed=42,
        expected_sample_count=20_480,
    )

    assert policy_teacher_ratio_mix_iteration_identity_sha256(
        ratio_binding, samples_sha256="3" * 64
    ) == legacy.policy_teacher_quarter_mix_iteration_identity_sha256(
        legacy_binding, samples_sha256="3" * 64
    )


def test_ratio_iteration_identity_binds_percentage() -> None:
    common = {
        "manifest_file_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "schedule_seed": 42,
        "expected_sample_count": 20_480,
    }
    teacher50 = PolicyTeacherRatioMixRuntimeBinding(
        **common, teacher_percentage=50
    )
    teacher100 = PolicyTeacherRatioMixRuntimeBinding(
        **common, teacher_percentage=100
    )

    assert policy_teacher_ratio_mix_iteration_identity_sha256(
        teacher50, samples_sha256="3" * 64
    ) != policy_teacher_ratio_mix_iteration_identity_sha256(
        teacher100, samples_sha256="3" * 64
    )
