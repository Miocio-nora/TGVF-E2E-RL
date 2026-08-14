from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.evaluation import policy_coredev as implementation
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    build_crop_tgvf_visual_messages,
)
from tgvf_rl.policy.run_config import (
    POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
    POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    build_tgvf_visual_messages,
)
from tgvf_rl.protocol import NativeAssistantDialect, NativeToolCapabilityProfile


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


class _Processor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[object],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.calls.append((messages, tools))
        return "matched-body<|im_start|>assistant\n"


class _Renderer:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.assistant_dialect = NativeAssistantDialect.QWEN3_VL_INSTRUCT
        self.render_calls: list[object] = []

    def assert_tokenizer_length(self) -> None:
        return None

    def assert_chat_template_identity(self) -> None:
        return None

    def assert_tool_schema_identity(self) -> None:
        return None

    def assert_generation_prefill(self, rendered: object, tokenizer: object) -> None:
        assert tokenizer is self.tokenizer
        assert rendered.text.endswith("<|im_start|>assistant\n")

    def render(self, messages: object, *, add_generation_prompt: bool) -> object:
        assert add_generation_prompt is True
        self.render_calls.append(messages)
        return SimpleNamespace(
            text="legacy-body<|im_start|>assistant\n",
            token_ids=(1, 2, 3),
        )


def _run(schema: str, profile: NativeToolCapabilityProfile) -> object:
    tool_name = {
        NativeToolCapabilityProfile.TGVF_ONLY: "tgvf_focus_tool",
        NativeToolCapabilityProfile.CROP_ONLY: "image_zoom_in_tool",
        NativeToolCapabilityProfile.CROP_TGVF: "tgvf_crop_tool",
    }[profile]
    return SimpleNamespace(
        schema_version=schema,
        model=SimpleNamespace(
            model_name="Qwen3-VL-8B-Instruct",
            revision_or_path="fixture",
        ),
        policy=SimpleNamespace(image_max_pixels=1_003_520),
        protocol=SimpleNamespace(
            tool_profile=profile,
            enabled_tool_names=(tool_name,),
        ),
    )


@pytest.mark.parametrize(
    ("run", "expected_messages", "expected_bundle"),
    (
        (
            _run(
                POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
                NativeToolCapabilityProfile.TGVF_ONLY,
            ),
            build_tgvf_visual_messages("question"),
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        ),
        (
            _run(
                POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
                NativeToolCapabilityProfile.CROP_TGVF,
            ),
            build_crop_tgvf_visual_messages("question"),
            CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        ),
    ),
)
def test_training_matched_prompt_routes_use_exact_builder_and_empty_tools(
    run: object,
    expected_messages: tuple[dict[str, object], ...],
    expected_bundle: str,
) -> None:
    processor = _Processor()
    renderer = _Renderer()

    text, token_ids = implementation._render_training_run_visual_prompt(
        run=run,
        processor=processor,
        renderer=renderer,
        question="question",
    )

    assert text == "matched-body<|im_start|>assistant\n"
    assert token_ids == tuple(ord(character) for character in text)
    assert processor.calls == [(list(expected_messages), [])]
    assert renderer.render_calls == []
    identity = implementation._matched_prompt_materializer_identity(run)
    assert identity is not None
    assert identity["template_tools_argument"] == []
    assert identity["prompt_bundle_sha256"] == expected_bundle


def test_legacy_generic_crop_prompt_route_is_unchanged() -> None:
    run = _run(
        POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
        NativeToolCapabilityProfile.CROP_ONLY,
    )
    processor = _Processor()
    renderer = _Renderer()

    text, token_ids = implementation._render_training_run_visual_prompt(
        run=run,
        processor=processor,
        renderer=renderer,
        question="question",
    )

    assert text == "legacy-body<|im_start|>assistant\n"
    assert token_ids == (1, 2, 3)
    assert not processor.calls
    assert len(renderer.render_calls) == 1
    assert implementation._matched_prompt_materializer_identity(run) is None


def test_official_visible_identity_does_not_admit_matched_materializer() -> None:
    matched_run = _run(
        POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
        NativeToolCapabilityProfile.TGVF_ONLY,
    )
    official_config = SimpleNamespace(
        evaluation_protocol=(
            implementation.DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
        )
    )

    assert (
        implementation._evaluation_prompt_materializer_identity(
            official_config, matched_run
        )
        is None
    )


def test_prompt_materializer_is_outside_paired_rng_protocol_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = {"profile": "training_run", "prompt_sha256": "a" * 64}
    monkeypatch.setattr(
        implementation, "_evaluation_protocol_identity", lambda *_args: protocol
    )
    config = SimpleNamespace(paired_seed_namespace="paired/temp1/v1")
    snapshot = SimpleNamespace(run=SimpleNamespace(rollout_rng=SimpleNamespace(master_seed=7)))

    contract = implementation._paired_evaluation_rng_contract(
        config,
        snapshot,
        task_manifest_sha256="b" * 64,
    )

    assert contract is not None
    assert contract["protocol_sha256"] == implementation._canonical_json_sha256(
        protocol
    )
    assert "prompt_materializer" not in contract["seed_components"]


@pytest.mark.parametrize(
    ("run", "runtime_name", "manager_method"),
    (
        (
            _run(
                POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
                NativeToolCapabilityProfile.TGVF_ONLY,
            ),
            "_RemoteTGVFFocusToolRuntime",
            "materialize_focus",
        ),
        (
            _run(
                POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
                NativeToolCapabilityProfile.CROP_ONLY,
            ),
            "_RemoteCropVisualMaterializer",
            "materialize_crop",
        ),
        (
            _run(
                POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
                NativeToolCapabilityProfile.CROP_TGVF,
            ),
            "_RemoteAtomicCropTGVFToolRuntime",
            "materialize_crop_tgvf",
        ),
    ),
)
def test_cpu_preflight_is_tool_profile_aware(
    run: object, runtime_name: str, manager_method: str
) -> None:
    proof = implementation.validate_policy_benchmark_runtime_interfaces(run)

    assert proof["tool_profile"] == run.protocol.tool_profile.value
    assert proof["remote_tool_runtime"] == runtime_name
    assert proof["standalone_manager_method"] == manager_method


def test_hidden_capture_includes_atomic_crop_tgvf_but_not_crop() -> None:
    config = SimpleNamespace(
        evaluation_protocol=implementation.TRAINING_RUN_EVALUATION_PROTOCOL
    )
    combo = _run(
        POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
        NativeToolCapabilityProfile.CROP_TGVF,
    )
    crop = _run(
        POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
        NativeToolCapabilityProfile.CROP_ONLY,
    )

    assert implementation._evaluation_requires_hidden_capture(config, combo) is True
    assert implementation._evaluation_requires_hidden_capture(config, crop) is False


@dataclass
class _FakeCropTGVFResult:
    source_image_sha256: str
    crop_sha256: str
    preprocessed_visual_sha256: str
    image_grid_thw: tuple[int, int, int]
    call_index: int
    model_bbox_2d: tuple[int, int, int, int]
    target_start: int
    target_end: int
    target_token_ids: tuple[int, ...]
    provider: str


class _Engine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def collective_rpc(
        self, method: str, *, kwargs: dict[str, object]
    ) -> list[dict[str, object]]:
        self.calls.append((method, kwargs))
        return [{}]


def _manager_result() -> _FakeCropTGVFResult:
    return _FakeCropTGVFResult(
        source_image_sha256="a" * 64,
        crop_sha256="c" * 64,
        preprocessed_visual_sha256="d" * 64,
        image_grid_thw=(1, 2, 4),
        call_index=2,
        model_bbox_2d=(10, 20, 800, 900),
        target_start=1,
        target_end=3,
        target_token_ids=(11, 12),
        provider="contextual_hidden_state",
    )


def _call_manager(
    manager: implementation.StandaloneTGVFVLLMManager,
    *,
    expected_target_token_ids: tuple[int, ...] = (11, 12),
):
    return manager.materialize_crop_tgvf(
        request_id="request",
        expected_step=8,
        sampled_output_ids=(10, 11, 12, 13),
        call_index=2,
        pixel_values=torch.ones((4, 6)),
        image_grid_thw=torch.tensor([[1, 2, 4]]),
        source_image_sha256="a" * 64,
        crop_sha256="c" * 64,
        preprocessed_visual_sha256="d" * 64,
        model_bbox_2d=(10, 20, 800, 900),
        target_start=1,
        target_end=3,
        expected_target_token_ids=expected_target_token_ids,
        provider="contextual_hidden_state",
    )


def test_standalone_combo_manager_binds_turn_target_and_rpc_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine()
    manager = implementation.StandaloneTGVFVLLMManager(
        engine, None, capture_hidden=True
    )
    manager.turns["request"] = implementation._TurnRoute(
        backend_request_id="backend",
        output_ids=(10, 11, 12, 13),
        optimizer_step=8,
    )
    result = _manager_result()
    monkeypatch.setattr(
        implementation, "TGVFCropMaterializationResult", _FakeCropTGVFResult
    )
    monkeypatch.setattr(
        implementation, "_crop_tgvf_from_utility_wire", lambda _wire: result
    )

    assert asyncio.run(_call_manager(manager)) is result
    assert engine.calls[0][0] == "tgvf_materialize_crop_tgvf"
    assert engine.calls[0][1]["backend_request_id"] == "backend"
    assert engine.calls[0][1]["expected_target_token_ids"] == (11, 12)

    mismatched = _manager_result()
    mismatched.crop_sha256 = "e" * 64
    monkeypatch.setattr(
        implementation, "_crop_tgvf_from_utility_wire", lambda _wire: mismatched
    )
    with pytest.raises(IdentityMismatchError, match="requested binding"):
        asyncio.run(_call_manager(manager))


def test_standalone_combo_manager_rejects_target_before_rpc() -> None:
    engine = _Engine()
    manager = implementation.StandaloneTGVFVLLMManager(
        engine, None, capture_hidden=True
    )
    manager.turns["request"] = implementation._TurnRoute(
        backend_request_id="backend",
        output_ids=(10, 11, 12, 13),
        optimizer_step=8,
    )

    with pytest.raises(RuntimeError, match="target differs"):
        asyncio.run(_call_manager(manager, expected_target_token_ids=(99, 12)))
    assert engine.calls == []
