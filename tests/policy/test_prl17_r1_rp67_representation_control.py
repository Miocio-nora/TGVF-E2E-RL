from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from tgvf_rl.policy.run_config import (
    RP66AdapterUpdateMode,
    load_policy_e2e_smoke_run_config,
)


_ROOT = Path(__file__).resolve().parents[2]
_RUNS = _ROOT / "configs/policy/runs"
_R0 = (
    _RUNS
    / "prl_17_r0_qwen3_instruct_full_frozen_rp66_bs16_n16_t1_"
    "shaped_novisual_8step_ws8.toml"
)
_R1 = (
    _RUNS
    / "prl_17_r1_qwen3_instruct_full_frozen_rp67_bs16_n16_t1_"
    "shaped_novisual_8step_ws8.toml"
)
_RP67_ARTIFACT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/representation/"
    "RP-67-qwen3-instruct-balanced-t1-image-axis-grounded-2000-gpu01/"
    "adapter.pt"
)


def _different_leaf_paths(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
) -> set[str]:
    """Return public canonical-record leaves whose values differ."""

    paths: set[str] = set()
    for key in left.keys() | right.keys():
        left_value = left.get(key)
        right_value = right.get(key)
        path = (*prefix, key)
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            paths.update(_different_leaf_paths(left_value, right_value, path))
        elif left_value != right_value:
            paths.add(".".join(path))
    return paths


def test_prl17_r1_is_a_strict_frozen_rp67_representation_control() -> None:
    r0 = load_policy_e2e_smoke_run_config(
        _R0.resolve(), allow_external_agent_loop_config=True
    )
    r1 = load_policy_e2e_smoke_run_config(
        _R1.resolve(), allow_external_agent_loop_config=True
    )

    assert r1.representation.artifact_path == _RP67_ARTIFACT
    assert r1.representation.artifact_file_sha256 == (
        "13332865eb30a2b04ce2ee90a9228e490c718e87fa57bc758078cdd28b6f0f68"
    )
    assert r1.representation.artifact.namespace == "tgvf-representation"
    assert r1.representation.artifact.name == (
        "qwen3-instruct-balanced-t1-image-axis-grounded"
    )
    assert r1.representation.artifact.version == "rp67-step2000"
    assert r1.representation.artifact.sha256 == (
        "2ea098967ba36671d6975a17e3830778d441149c27f5f80e43e78daf818933b1"
    )
    assert r1.representation.expected_run_id == (
        "RP-67-QWEN3-INSTRUCT-REP-BALANCED-T1-IMAGE-AXIS-GROUNDED-2000-GPU01"
    )
    assert r1.representation.expected_run_identity_sha256 == (
        "0b53d04cf8e4c8b665e76279da1df8d1e6ebabee63318c644a3bff5bad099b44"
    )
    assert (
        r1.representation.adapter_update_mode
        is RP66AdapterUpdateMode.FROZEN_ADAPTER
    )
    assert r1.representation.adapter_trainable is False

    # Utility labels remain intentionally fixed: re-materializing them with
    # RP67 would change both representation and reward in the same arm.
    assert r1.reward.tool_utility == r0.reward.tool_utility

    assert _different_leaf_paths(r0.as_record(), r1.as_record()) == {
        "framework.agent_loop_config_path",
        "output.checkpoint_directory",
        "output.metrics_path",
        "output.root",
        "representation.artifact_file_sha256",
        "representation.artifact_manifest_sha256",
        "representation.artifact_name",
        "representation.artifact_path",
        "representation.artifact_version",
        "representation.expected_run_id",
        "representation.expected_run_identity_sha256",
        "reward.judge_config_path",
        "reward.judge_reason",
        "run_id",
    }

    # Source paths and source hashes are expected to differ because these are
    # two separately addressable run documents; each still binds its bytes.
    assert r0.source_path == _R0.resolve()
    assert r1.source_path == _R1.resolve()
    assert r0.source_sha256 == sha256(_R0.read_bytes()).hexdigest()
    assert r1.source_sha256 == sha256(_R1.read_bytes()).hexdigest()
    assert r0.source_sha256 != r1.source_sha256
    assert r0.identity_sha256 != r1.identity_sha256
