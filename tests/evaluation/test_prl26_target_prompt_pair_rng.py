from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tgvf_rl.evaluation import policy_coredev as coredev
from tgvf_rl.evaluation.policy_paired_tgvf_snapshot import (
    PairedTGVFEvaluationSnapshot,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).parents[2]
_TASK_SHA256 = "3f69119d24867c3f3210c8b01eb71304247725ddaf9ca983d2b41c2885403cbc"
_NAMESPACE = (
    "coredev2511/prl26/short-vs-target-guide-v2/s32/train512-eval512/seed42/v1"
)


def _run(name: str):
    return load_policy_e2e_smoke_run_config(
        (_ROOT / "configs/policy/runs" / name).resolve(),
        allow_external_agent_loop_config=True,
    )


def _snapshot(run, *, step: int = 32, weights: str = "a" * 64):
    snapshot = PairedTGVFEvaluationSnapshot.__new__(PairedTGVFEvaluationSnapshot)
    object.__setattr__(snapshot, "run", run)
    object.__setattr__(
        snapshot,
        "receipt",
        SimpleNamespace(optimizer_step=step, combined_weights_sha256=weights),
    )
    object.__setattr__(snapshot, "rp66_tensors", {})
    return snapshot


def _config(*, cap: int = 262144):
    return SimpleNamespace(
        evaluation_protocol=coredev.TRAINING_RUN_EVALUATION_PROTOCOL,
        paired_seed_namespace=_NAMESPACE,
        paired_rng_protocol_projection=coredev.TARGET_PROMPT_PAIR_PROJECTION,
        evaluation_image_max_pixels=cap,
    )


def _contracts():
    short = _snapshot(
        _run(
            "prl_26_c_qwen3_instruct_short_tgvf_train512_parity_"
            "s32_bs16_n16_teacher25_ws8.toml"
        ),
        weights="a" * 64,
    )
    full = _snapshot(
        _run(
            "prl_26_d_qwen3_instruct_target_guide_v2_tgvf_train512_parity_"
            "s32_bs16_n16_teacher25_ws8.toml"
        ),
        weights="b" * 64,
    )
    contracts = tuple(
        coredev.paired_evaluation_rng_contract(
            _config(), snapshot, task_manifest_sha256=_TASK_SHA256
        )
        for snapshot in (short, full)
    )
    assert all(contract is not None for contract in contracts)
    return (short, full), contracts


def test_target_prompt_projection_retains_arm_protocol_and_shares_rng_stream() -> None:
    snapshots, (short, full) = _contracts()

    assert short["schema_version"] == coredev.RESOLUTION_PAIRED_POLICY_EVALUATION_RNG_SCHEMA
    assert short["arm_protocol_sha256"] != full["arm_protocol_sha256"]
    assert short["seed_protocol_sha256"] == full["seed_protocol_sha256"]
    assert short["protocol_projection"] == full["protocol_projection"] == {
        "kind": coredev.TARGET_PROMPT_PAIR_PROJECTION,
        "excluded_protocol_field": "prompt_sha256",
        "axis_values": list(coredev.TARGET_PROMPT_PAIR_VALUES),
    }

    identities = []
    for snapshot, contract in zip(snapshots, (short, full), strict=True):
        rng = coredev.paired_evaluation_rng_for_task(
            {"sampling_rng": contract}, sample_id="sample-7", rollout_index=0
        )
        identities.append(
            rng.for_turn(
                (1, 2, 3),
                turn_index=2,
                behavior_policy=snapshot.policy_version,
            )
        )
    assert identities[0] == identities[1]


@pytest.mark.parametrize(
    ("cap", "step", "error"),
    (
        (1003520, 32, "exact Short/Target-v2"),
        (262144, 31, "exact Short/Target-v2"),
    ),
)
def test_target_prompt_projection_rejects_cap_or_step_drift(
    cap: int, step: int, error: str
) -> None:
    run = _run(
        "prl_26_c_qwen3_instruct_short_tgvf_train512_parity_"
        "s32_bs16_n16_teacher25_ws8.toml"
    )

    with pytest.raises(ValueError, match=error):
        coredev.paired_evaluation_rng_contract(
            _config(cap=cap),
            _snapshot(run, step=step),
            task_manifest_sha256=_TASK_SHA256,
        )

