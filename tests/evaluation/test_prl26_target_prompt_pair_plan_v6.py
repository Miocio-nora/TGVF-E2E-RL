from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).parents[2]
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_BASE_PLAN = (
    _ROOT
    / "configs/evaluation/"
    "prl25_c_frozen_rp67_tfree_teacher25_s64_matched_pixel512_"
    "coredev2511_plan.json"
)
_FORMAL_PLAN = (
    _ROOT
    / "configs/evaluation/"
    "prl26_cd_tgvf_target_prompt_pair_s32_pixel512_coredev2511_plan.json"
)
_CONFIGS = {
    "short": (
        _ROOT
        / "configs/policy/runs/"
        "prl_26_c_qwen3_instruct_short_tgvf_train512_parity_"
        "s32_bs16_n16_teacher25_ws8.toml"
    ),
    "full": (
        _ROOT
        / "configs/policy/runs/"
        "prl_26_d_qwen3_instruct_target_guide_v2_tgvf_train512_parity_"
        "s32_bs16_n16_teacher25_ws8.toml"
    ),
}
_SPEC = importlib.util.spec_from_file_location("target_prompt_pair_runner", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(run) -> dict[str, object]:
    return {
        "evaluation_protocol": "training_run",
        "prompt_sha256": run.protocol.prompt_sha256,
        "tool_profile": run.protocol.tool_profile.value,
        "tool_schema_sha256": run.protocol.tool_schema_sha256,
        "maximum_tool_calls": run.protocol.maximum_tool_calls,
        "sampling_source": "bound_policy_run_config",
        "same_tasks_and_rank_partition": True,
    }


def _payload(tmp_path: Path) -> dict[str, object]:
    base = json.loads(_BASE_PLAN.read_text(encoding="utf-8"))
    runs = {
        arm: load_policy_e2e_smoke_run_config(
            path.resolve(), allow_external_agent_loop_config=True
        )
        for arm, path in _CONFIGS.items()
    }
    projection = _RUNNER._target_prompt_pair_projection_identity()
    runtime_protocols = {
        arm: _RUNNER._training_run_rng_protocol(run)
        for arm, run in runs.items()
    }
    arm_protocols = {
        arm: _RUNNER._canonical_json_sha256(protocol)
        for arm, protocol in runtime_protocols.items()
    }
    projected = copy.deepcopy(runtime_protocols["short"])
    projected.pop("prompt_sha256")
    seed_protocol = _RUNNER._canonical_json_sha256(
        {"projection": projection, "projected_protocol": projected}
    )
    return {
        "schema_version": _RUNNER.PLAN_SCHEMA_V6,
        "evaluation_id": "PRL26-CD-TARGET-PROMPT-PAIR-S32-PIXEL512-V1",
        "status": "ready",
        "evaluation_image_max_pixels": 262144,
        "output_root": str((tmp_path / "evaluation").resolve()),
        "policy_arms": {
            arm: {
                "policy_config": str(path.resolve()),
                "policy_config_sha256": _sha256(path),
            }
            for arm, path in _CONFIGS.items()
        },
        "task_manifest_path": base["task_manifest_path"],
        "task_manifest_sha256": base["task_manifest_sha256"],
        "expected_task_count": 2511,
        "expected_single_image_count": 2240,
        "unsupported_multi_image_count": 271,
        "executor": {
            "path": "tools/run_prl15_paired_evaluation.py",
            "supervisor": "tools/supervise_prl26_tgvf_prompt_s32_evaluation.sh",
            "snapshot_backend": "paired_tgvf",
            "supports_wait_for_final_arm": True,
            "waits_for_gpu_release": True,
            "supports_resume": True,
            "eight_gpu_schedule": (
                "short_gpu0_3_full_gpu4_7_parallel_inference_sequential_scoring"
            ),
        },
        "protocols": {arm: _protocol(run) for arm, run in runs.items()},
        "paired_rng": {
            "schema_version": _RUNNER.RESOLUTION_PAIRED_RNG_PLAN_SCHEMA,
            "mode": "common_random_numbers_per_task_turn",
            "seed_namespace": (
                "coredev2511/prl26/short-vs-target-guide-v2/"
                "s32/train512-eval512/temp1/seed42/v1"
            ),
            "master_seed": 42,
            "task_manifest_sha256": base["task_manifest_sha256"],
            "seed_protocol_sha256": seed_protocol,
            "arm_protocol_sha256": arm_protocols,
            "protocol_projection": projection,
            "temperature": 1.0,
            "do_sample": True,
            "excluded_arm_components": list(_RUNNER._PAIRED_RNG_EXCLUSIONS),
        },
        "arms": [
            {
                "name": arm,
                "optimizer_step": 32,
                "qwen_source": (
                    "output.root/permanent-checkpoints/global_step_32"
                ),
                "rp66_source": (
                    "output.root/runtime-policy-state/lora-manifests/"
                    "step-00000032-*.json"
                ),
                "evaluation_id": f"PRL26-{arm.upper()}-S32-PIXEL512-V1",
            }
            for arm in ("short", "full")
        ],
        "required_pairing": base["required_pairing"],
        "scoring": {
            **base["scoring"],
            "run_id_prefix": "T20260829-PRL26-CD-TARGET-PROMPT-PAIR-S32-PIXEL512",
        },
    }


def _write_plan(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_v6_plan_loads_two_distinct_training_owners_with_one_seed_protocol(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    plan = _RUNNER._load_plan(_write_plan(tmp_path, payload))
    runtime = _RUNNER._load_evaluation_runtime(plan)

    assert plan["schema_version"] == _RUNNER.PLAN_SCHEMA_V6
    assert runtime.checkpoint_owner_for_arm("short").run_id != (
        runtime.checkpoint_owner_for_arm("full").run_id
    )
    assert _RUNNER._arm_rng_protocol_projection(plan, "short") == (
        _RUNNER.TARGET_PROMPT_PAIR_PROJECTION
    )
    assert plan["paired_rng"]["arm_protocol_sha256"]["short"] != (
        plan["paired_rng"]["arm_protocol_sha256"]["full"]
    )
    assert plan["paired_rng"]["protocol_projection"] == {
        "kind": _RUNNER.TARGET_PROMPT_PAIR_PROJECTION,
        "excluded_protocol_field": "prompt_sha256",
        "axis_values": list(_RUNNER.TARGET_PROMPT_PAIR_VALUES),
    }


def test_formal_prl26_cd_v6_plan_is_runner_loadable() -> None:
    plan = _RUNNER._load_plan(_FORMAL_PLAN)
    runtime = _RUNNER._load_evaluation_runtime(plan)

    assert plan["paired_rng"]["seed_protocol_sha256"] == (
        "4cbfd3cf698cb47b0c9594ca9f9e146ca09932d62bdb93d0877e59f9a85bee9c"
    )
    assert runtime.checkpoint_owner_for_arm("short").protocol.prompt_sha256 == (
        _RUNNER.TARGET_PROMPT_PAIR_VALUES[0]
    )
    assert runtime.checkpoint_owner_for_arm("full").protocol.prompt_sha256 == (
        _RUNNER.TARGET_PROMPT_PAIR_VALUES[1]
    )


def test_v6_arms_keep_two_disjoint_tp2_judge_endpoints() -> None:
    plan = _RUNNER._load_plan(_FORMAL_PLAN)
    judge = _RUNNER._load_judge_config(plan, require_local_model=False)

    assigned = _RUNNER._assign_arm_judges(
        judge,
        arm_names=("short", "full"),
        gpu_ids=(0, 1, 2, 3),
    )

    assert assigned["short"]["devices"]["physical"] == [0, 1]
    assert assigned["full"]["devices"]["physical"] == [2, 3]
    assert assigned["short"]["server"]["base_url"] == "http://127.0.0.1:8012/v1"
    assert assigned["full"]["server"]["base_url"] == "http://127.0.0.1:8013/v1"


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            lambda payload: payload["paired_rng"]["protocol_projection"].__setitem__(
                "excluded_protocol_field", "tool_schema_sha256"
            ),
            "RNG identity",
        ),
        (
            lambda payload: payload["protocols"]["full"].__setitem__(
                "maximum_tool_calls", 5
            ),
            "plan protocol differs",
        ),
        (
            lambda payload: payload["arms"][1].__setitem__(
                "optimizer_step", 31
            ),
            "S32 source/evaluation",
        ),
    ),
)
def test_v6_plan_rejects_projection_protocol_or_step_drift(
    tmp_path: Path, mutation, error: str
) -> None:
    payload = _payload(tmp_path)
    mutation(payload)

    with pytest.raises((ValueError, RuntimeError), match=error):
        _RUNNER._load_plan(_write_plan(tmp_path, payload))


def test_v6_scoring_drains_short_before_launching_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class Process:
        def __init__(self, arm: str) -> None:
            self.arm = arm
            self.pid = 10_000 + len(events)

        def wait(self) -> int:
            events.append(f"wait:{self.arm}")
            return 0

    judges = {"short": {"endpoint": "short"}, "full": {"endpoint": "full"}}

    def launch(_config, _plan, judge, *, arm, log_root):
        assert log_root == tmp_path
        assert judge is judges[arm]
        events.append(f"launch:{arm}")
        return [("dataset", Process(arm))]

    def summarize(config, _plan, judge):
        arm = config.name
        assert judge is judges[arm]
        events.append(f"summarize:{arm}")
        return {"arm": arm}

    monkeypatch.setattr(_RUNNER, "_launch_score_arm", launch)
    monkeypatch.setattr(_RUNNER, "_worker_group_exists", lambda _process: False)
    monkeypatch.setattr(_RUNNER, "_summarize_scored_arm", summarize)
    existing = _RUNNER._score_missing_arms(
        {"short": Path("short"), "full": Path("full")},
        {"short": None, "full": None},
        {
            "schema_version": _RUNNER.PLAN_SCHEMA_V6,
            "arms": [{"name": "short"}, {"name": "full"}],
        },
        judges,
        log_root=tmp_path,
    )

    assert events == [
        "launch:short",
        "wait:short",
        "summarize:short",
        "launch:full",
        "wait:full",
        "summarize:full",
    ]
    assert existing == {"short": {"arm": "short"}, "full": {"arm": "full"}}


def test_v6_scoring_fails_closed_before_launching_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class Process:
        pid = 20_000

        def wait(self) -> int:
            events.append("wait:short")
            return 1

    def launch(_config, _plan, _judge, *, arm, log_root):
        assert arm == "short"
        assert log_root == tmp_path
        events.append("launch:short")
        return [("dataset", Process())]

    monkeypatch.setattr(_RUNNER, "_launch_score_arm", launch)
    monkeypatch.setattr(_RUNNER, "_worker_group_exists", lambda _process: False)
    monkeypatch.setattr(
        _RUNNER,
        "_summarize_scored_arm",
        lambda *_args, **_kwargs: pytest.fail("failed arm must not be summarized"),
    )

    with pytest.raises(RuntimeError, match="short/dataset=1"):
        _RUNNER._score_missing_arms(
            {"short": Path("short"), "full": Path("full")},
            {"short": None, "full": None},
            {
                "schema_version": _RUNNER.PLAN_SCHEMA_V6,
                "arms": [{"name": "short"}, {"name": "full"}],
            },
            {"short": {}, "full": {}},
            log_root=tmp_path,
        )

    assert events == ["launch:short", "wait:short"]


def test_v6_resume_requires_exact_pinned_scorer_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    config = SimpleNamespace(evaluation_id="PRL26-SHORT-EVAL")

    monkeypatch.setattr(
        _RUNNER, "load_policy_coredev_config", lambda _path: config
    )
    monkeypatch.setattr(
        _RUNNER, "_scoring_root", lambda _path, _plan: tmp_path
    )

    def accepted(scoring_root, judge, **kwargs):
        captured.update(
            {"scoring_root": scoring_root, "judge": judge, **kwargs}
        )
        return {"status": "pass"}

    monkeypatch.setattr(_RUNNER, "_accepted_official_summary", accepted)
    judge = {"server": {"base_url": "http://127.0.0.1:8012/v1"}}
    plan = {
        "schema_version": _RUNNER.PLAN_SCHEMA_V6,
        "scoring": {"run_id_prefix": "T20260829-PRL26-CD"},
    }

    assert _RUNNER._accepted_scored_arm(Path("short.json"), plan, judge) == {
        "status": "pass"
    }
    assert captured["scoring_root"] == tmp_path
    assert captured["judge"] is judge
    assert captured["include_headline"] is True
    assert captured["source_evaluation_id"] == config.evaluation_id
    assert captured["source_run_id"] == _RUNNER._vlmevalkit_scoring_run_id(
        run_id_prefix="T20260829-PRL26-CD",
        arm_evaluation_id=config.evaluation_id,
    )
