from __future__ import annotations

import hashlib

import pytest

from tgvf_rl.artifact_contracts import canonical_json_bytes, canonical_json_sha256
from tgvf_rl.data import (
    deepeyes_official_schedule,
    deepeyes_official_schedule_index,
    policy_selection_screening,
    policy_selection_t1_judge,
    policy_selection_t1_resume_smoke,
    policy_selection_t1_replay_audit,
    policy_selection_t1_scoring,
    policy_selection_vllm,
)


def test_canonical_json_bytes_is_compact_sorted_utf8_without_newline() -> None:
    value = {"z": 1, "a": "汉字", "nested": {"β": True, "α": None}}

    encoded = canonical_json_bytes(value)

    assert encoded == (
        b'{"a":"\xe6\xb1\x89\xe5\xad\x97","nested":'
        b'{"\xce\xb1":null,"\xce\xb2":true},"z":1}'
    )
    assert not encoded.endswith(b"\n")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_bytes_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": value})


def test_canonical_json_bytes_digest_is_pinned() -> None:
    value = {"z": 1, "a": "汉字", "nested": {"β": True, "α": None}}
    encoded = canonical_json_bytes(value)

    assert hashlib.sha256(encoded).hexdigest() == (
        "9baa18a6055780ea925c6c1774929f5d9c097753014f4efd7c0f3563b13a0cc0"
    )
    assert canonical_json_sha256(value) == (
        "9baa18a6055780ea925c6c1774929f5d9c097753014f4efd7c0f3563b13a0cc0"
    )


def test_migrated_data_helpers_are_exact_aliases() -> None:
    migrated_modules = (
        deepeyes_official_schedule_index,
        policy_selection_screening,
        policy_selection_vllm,
        policy_selection_t1_resume_smoke,
        policy_selection_t1_scoring,
        policy_selection_t1_judge,
    )

    assert all(
        module._canonical_json_bytes is canonical_json_bytes  # noqa: SLF001
        for module in migrated_modules
    )


def test_migrated_data_digest_helpers_are_exact_aliases() -> None:
    migrated_modules = (
        deepeyes_official_schedule,
        deepeyes_official_schedule_index,
        policy_selection_t1_replay_audit,
    )

    assert all(
        module._sha256_json is canonical_json_sha256  # noqa: SLF001
        for module in migrated_modules
    )
