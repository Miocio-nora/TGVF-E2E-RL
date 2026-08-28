from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tgvf_rl.evaluation.policy_coredev import (
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    TRAINING_RUN_EVALUATION_PROTOCOL,
)
from tgvf_rl.policy.deepeyes_native_contract import (
    load_deepeyes_native_run_contract,
)
from tgvf_rl.policy.run_config import (
    POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
)


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
CROP_CONFIG = ROOT / (
    "configs/policy/runs/"
    "prl_26_b_qwen3_instruct_full_crop_train512_parity_s32_bs16_n16_"
    "teacher25_ws8.toml"
)
PROTOCOL_CONFIG = ROOT / (
    "configs/policy/runs/"
    "prl_24_d_base_qwen3_instruct_full_crop_teacher25_native_prl13.toml"
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def binder() -> ModuleType:
    return _load_module(
        "prl26_train512_binder_under_test",
        TOOLS / "bind_prl26_train512_s32_evaluation.py",
    )


@pytest.fixture(scope="module")
def paired_runner() -> ModuleType:
    return _load_module(
        "prl26_paired_runner_under_test",
        TOOLS / "run_prl15_paired_evaluation.py",
    )


@pytest.fixture(scope="module")
def proof_validator() -> ModuleType:
    return _load_module(
        "prl26_processor_proof_under_test",
        TOOLS / "validate_prl26_train512_processor_proof.py",
    )


def _pixel512_geometry() -> dict[str, object]:
    return {
        "configured_image_max_pixels": 262_144,
        "processor_image_size": {
            "shortest_edge": 65_536,
            "longest_edge": 16_777_216,
        },
        "effective_processor_image_size": {
            "shortest_edge": 65_536,
            "longest_edge": 262_144,
        },
        "runtime_mm_processor_kwargs": {
            "size": {"shortest_edge": 65_536, "longest_edge": 262_144}
        },
        "runtime_override_path": "mm_processor_kwargs.size.longest_edge",
        "vllm_012_shallow_hashable": True,
        "nested_images_kwargs_present": False,
        "max_pixels_kwarg_present": False,
    }


def test_real_processor_proof_rejects_longest_edge_drift(
    proof_validator: ModuleType,
) -> None:
    proof = _pixel512_geometry()
    proof_validator._common_geometry(proof)

    drifted = deepcopy(proof)
    drifted["runtime_mm_processor_kwargs"]["size"]["longest_edge"] = 1_003_520
    with pytest.raises(RuntimeError, match="runtime override proof differs"):
        proof_validator._common_geometry(drifted)


def test_crop_plan_binds_post_completion_receipt_bytes_fail_closed(
    binder: ModuleType,
    paired_runner: ModuleType,
    tmp_path: Path,
) -> None:
    source_run = load_policy_e2e_smoke_run_config(
        CROP_CONFIG.resolve(), allow_external_agent_loop_config=True
    )
    protocol = load_deepeyes_native_run_contract(PROTOCOL_CONFIG.resolve())
    owner_root = tmp_path / "owner"
    completion = owner_root / (
        "permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"
    )
    completion.parent.mkdir(parents=True)
    completion.write_text('{"receipt":"first"}\n', encoding="utf-8")
    run = SimpleNamespace(
        run_id=source_run.run_id,
        identity_sha256=source_run.identity_sha256,
        output=SimpleNamespace(root=owner_root),
        distributed=source_run.distributed,
        model=source_run.model,
        policy=source_run.policy,
        rollout_rng=source_run.rollout_rng,
    )
    plan = binder._crop_plan(
        run=run, completion_path=completion.resolve(), protocol=protocol
    )
    assert plan["checkpoint_owner"]["completion_sha256"] == binder._sha256(completion)
    assert plan["evaluation_image_max_pixels"] == 262_144
    assert plan["checkpoint_owner"]["contract_type"] == (
        "policy_e2e_crop_exact_pixel512_parity_run_config_v1"
    )
    plan_path = tmp_path / "bound-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    paired_runner._load_plan(plan_path)

    completion.write_text('{"receipt":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="completion identity differs"):
        paired_runner._load_plan(plan_path)


def _crop_runtime_plan(owner: object, protocol: object) -> dict[str, object]:
    protocol_payload = protocol.payload["protocol"]
    return {
        "schema_version": "tgvf.paired-policy-benchmark-plan.v3",
        "evaluation_image_max_pixels": 262_144,
        "checkpoint_owner": {
            "contract_type": ("policy_e2e_crop_exact_pixel512_parity_run_config_v1"),
            "config_sha256": owner.source_sha256,
            "run_id": owner.run_id,
            "run_identity_sha256": owner.identity_sha256,
            "output_root": str(owner.output.root),
            "checkpoint_world_size": owner.distributed.world_size,
            "completion_path": str(
                owner.output.root / "permanent-checkpoints/global_step_32/"
                "tgvf_permanent_checkpoint_receipt.json"
            ),
        },
        "protocol_contract": {
            "config_sha256": protocol.source_sha256,
            "run_id": protocol.run_id,
            "run_identity_sha256": protocol.identity_sha256,
        },
        "protocol": {
            "evaluation_protocol": DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
            "visual_prompt_bundle_sha256": protocol_payload[
                "visual_prompt_bundle_sha256"
            ],
            "tool_name": protocol_payload["tool_name"],
            "tool_parser": protocol_payload["tool_parser"],
            "maximum_tool_calls": protocol_payload["max_active_perception"],
            "native_pixels": True,
            "sampling_source": "bound_protocol_contract",
            "same_tasks_and_rank_partition": True,
        },
        "paired_rng": {
            "master_seed": 42,
            "temperature": 1.0,
            "do_sample": True,
            "task_manifest_sha256": (
                "3f69119d24867c3f3210c8b01eb71304247725ddaf9ca983d2b41c2885403cbc"
            ),
            "seed_namespace": "negative-test",
            "protocol_sha256": (
                "cadbff473bbc55650a5ac3b1b99fe8b4c5645ddc4dfb0cc0a0be75fd86183920"
            ),
        },
        "arms": [
            {
                "name": "step32",
                "optimizer_step": 32,
                "source": {
                    "kind": "owner_checkpoint",
                    "relative_path": "permanent-checkpoints/global_step_32",
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evaluation_image_max_pixels", 1_003_520),
        (
            "contract_type",
            "policy_e2e_crop_exact_run_config_v1",
        ),
    ),
)
def test_pixel512_crop_owner_rejects_cap_or_schema_aliasing(
    paired_runner: ModuleType,
    field: str,
    value: object,
) -> None:
    owner = load_policy_e2e_smoke_run_config(
        CROP_CONFIG.resolve(), allow_external_agent_loop_config=True
    )
    assert (
        owner.schema_version
        == POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
    )
    protocol = load_deepeyes_native_run_contract(PROTOCOL_CONFIG.resolve())
    plan = _crop_runtime_plan(owner, protocol)
    if field == "contract_type":
        plan["checkpoint_owner"][field] = value
    else:
        plan[field] = value
    with pytest.raises(RuntimeError, match="checkpoint owner|owner differs"):
        paired_runner._validate_v3_policy_run_runtime(plan, owner, protocol)


@pytest.mark.parametrize(
    "schema",
    (
        POLICY_E2E_NO_TOOL_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
        POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    ),
)
def test_no_tool_static_validate_emits_processor_proof_for_old_and_new_schema(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    schema: str,
) -> None:
    module = _load_module(
        f"prl26_policy_benchmark_validate_{schema}",
        TOOLS / "run_policy_benchmark.py",
    )

    class FakeFullModelSnapshot:
        def __init__(self) -> None:
            self.run = SimpleNamespace(
                schema_version=schema,
                model=SimpleNamespace(
                    revision_or_path="/tmp/model", tokenizer_length=151_669
                ),
            )
            self.policy_version = SimpleNamespace(
                optimizer_step=32, weights_sha256="a" * 64
            )

    snapshot = FakeFullModelSnapshot()
    config = SimpleNamespace(
        evaluation_id="TEST-NOTOOL-PIXEL512",
        evaluation_protocol=TRAINING_RUN_EVALUATION_PROTOCOL,
    )
    monkeypatch.setattr(module, "FullModelEvaluationSnapshot", FakeFullModelSnapshot)
    monkeypatch.setattr(
        module,
        "load_bound_policy_benchmark_tasks",
        lambda _config: [SimpleNamespace(single_image=True)],
    )
    monkeypatch.setattr(module, "_world_size", lambda _config, _world: 4)
    monkeypatch.setattr(
        module, "load_frozen_policy_evaluation_snapshot", lambda _config: snapshot
    )
    monkeypatch.setattr(
        module,
        "write_policy_evaluation_identity",
        lambda _config, _snapshot: {"identity_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        module,
        "validate_policy_benchmark_runtime_interfaces",
        lambda _run, image_max_pixels: {"image_max_pixels": image_max_pixels},
    )
    monkeypatch.setattr(module, "evaluation_image_max_pixels", lambda *_args: 262_144)
    monkeypatch.setattr(
        module,
        "validate_no_tool_matched_processor",
        lambda *_args, **_kwargs: {"configured_image_max_pixels": 262_144},
    )
    import transformers

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    assert module._validate(config, 4) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["no_tool_matched_processor_proof"] == {
        "configured_image_max_pixels": 262_144
    }
