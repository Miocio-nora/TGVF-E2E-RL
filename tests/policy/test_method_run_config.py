from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.framework.verl.policy_runtime import _policy_termination_contract
from tgvf_rl.framework.vllm import VLLMTerminationOutcome
from tgvf_rl.judges import load_openai_compatible_judge
from tgvf_rl.policy.config import (
    PilotSamplingConfig,
    PolicyMethodExperimentConfig,
    PolicyMethodProfile,
    PolicyMethodSamplingConfig,
    PolicyPilotV1Config,
)
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.deepeyes_official_protocol import VISUAL_PROMPT_IDENTITY
from tgvf_rl.policy.no_tool_rl_protocol import NO_TOOL_RL_PROMPT_IDENTITY
from tgvf_rl.policy.run_config import (
    POLICY_E2E_MIXED_ALTERNATE_ANSWER_VERIFIER,
    POLICY_E2E_MIXED_ALTERNATE_JUDGE_MODE,
    POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA,
    POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA,
    POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
    policy_e2e_mixed_alternate_answer_verifier_sha256,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.tgvf_target_guide_v2_protocol import (
    TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY,
)
from tgvf_rl.protocol import (
    NativeActionBoundaryProtocolId,
    NativeSuccessObservationProtocolId,
    NativeToolCapabilityProfile,
    StandardToolError,
    ToolErrorCode,
)
from tests.policy.test_run_config import (
    _prepare_external_inputs,
    _q,
    _teacher_quarter_config_text,
    _write_teacher_quarter_artifact,
)


_METHOD_CASES = (
    (
        PolicyMethodProfile.NO_TOOL,
        NativeToolCapabilityProfile.NO_TOOL,
        NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256,
        NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1,
    ),
    (
        PolicyMethodProfile.CROP,
        NativeToolCapabilityProfile.CROP_ONLY,
        VISUAL_PROMPT_IDENTITY.bundle_sha256,
        NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
    ),
    (
        PolicyMethodProfile.TGVF_SHORT,
        NativeToolCapabilityProfile.TGVF_ONLY,
        TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1,
    ),
    (
        PolicyMethodProfile.TGVF_TARGET_GUIDE_V2,
        NativeToolCapabilityProfile.TGVF_ONLY,
        TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256,
        NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1,
    ),
    (
        PolicyMethodProfile.ATOMIC,
        NativeToolCapabilityProfile.CROP_TGVF,
        CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        NativeSuccessObservationProtocolId.DEEPEYES_ATOMIC_MATCHED_V1,
    ),
)


def _cap_error_sha256(maximum_tool_calls: int) -> str:
    return StandardToolError(
        code=ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,
        message=(
            f"The maximum of {maximum_tool_calls} tool-call attempts has been "
            "reached; this call was not executed."
        ),
        attempt_index=maximum_tool_calls,
        recoverable=True,
        maximum_tool_calls=maximum_tool_calls,
    ).payload_sha256


@pytest.fixture
def method_config_factory(
    tmp_path: Path,
) -> Callable[..., tuple[Path, str]]:
    external = _prepare_external_inputs(tmp_path)
    teacher = _write_teacher_quarter_artifact(tmp_path)

    def build(
        *,
        profile: PolicyMethodProfile,
        tool_profile: NativeToolCapabilityProfile,
        prompt_sha256: str,
        observation_id: NativeSuccessObservationProtocolId,
        image_max_pixels: int = 345_678,
        trajectories_per_prompt: int = 3,
        max_response_length: int = 1_234,
        maximum_tool_calls: int = 5,
        rollout_seed: int = 77,
        maximum_optimizer_steps: int = 2,
        native_deepstack_enabled: bool = True,
        weight_sync_interval_optimizer_steps: int = 1,
        performance_v2: bool = False,
        dynamic_token_batching: bool = True,
        use_remove_padding: bool = True,
        enable_gradient_checkpointing: bool = True,
        vllm_enable_prefix_caching: bool = True,
        vllm_enable_chunked_prefill: bool = True,
        vllm_enable_cuda_graph: bool = True,
        vllm_cuda_graph_capture_sizes: tuple[int, ...] = (1, 2, 4, 8),
        judge_dispatch_mode: str = "dedicated_thread_pool",
        judge_max_concurrency_per_worker: int = 8,
        reference_replay_mode: str = "off",
    ) -> tuple[Path, str]:
        text = _teacher_quarter_config_text(tmp_path, external, teacher).replace(
            "policy-e2e-explicit-observation-run-config-v1",
            POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA,
        )
        text = text.replace(
            'run_id = "policy-smoke-test-001"\n',
            'run_id = "policy-smoke-test-001"\n\n'
            "[method]\n"
            'matrix_id = "config-owned-method-matrix"\n'
            f'profile = "{profile.value}"\n',
        )
        text = text.replace(
            "image_max_pixels = 262144",
            f"image_max_pixels = {image_max_pixels}",
        )
        text = text.replace(
            "native_deepstack_enabled = true",
            "native_deepstack_enabled = "
            + ("true" if native_deepstack_enabled else "false"),
        )
        text = text.replace(
            "weight_sync_interval_optimizer_steps = 1",
            "weight_sync_interval_optimizer_steps = "
            f"{weight_sync_interval_optimizer_steps}",
        )
        weight_sync_mode = (
            "nccl_full_qwen_v1"
            if profile in {PolicyMethodProfile.NO_TOOL, PolicyMethodProfile.CROP}
            else "nccl_full_qwen_plus_trainable_rp66_v1"
        )
        text = text.replace(
            'weight_sync_mode = "nccl_lora_state_v1"',
            f'weight_sync_mode = "{weight_sync_mode}"',
        )
        if maximum_optimizer_steps != 2:
            text = text.replace(
                "total_steps = 2",
                f"total_steps = {maximum_optimizer_steps}",
            ).replace(
                "maximum_optimizer_steps = 2",
                f"maximum_optimizer_steps = {maximum_optimizer_steps}",
            )
            checkpoint_steps = sorted(
                {0, maximum_optimizer_steps // 2, maximum_optimizer_steps}
            )
            text = text.replace(
                "checkpoint_steps = [0, 1, 2]",
                f"checkpoint_steps = {checkpoint_steps}",
            )
        representation_suffix = ""
        if profile in {
            PolicyMethodProfile.TGVF_SHORT,
            PolicyMethodProfile.TGVF_TARGET_GUIDE_V2,
            PolicyMethodProfile.ATOMIC,
        }:
            representation_suffix = 'adapter_update_mode = "frozen_adapter"\n\n'
        text = text.replace(
            "[representation.conditioning]",
            representation_suffix + "[representation.conditioning]",
        )
        protocol_start = text.index("[protocol]")
        sampling_start = text.index("[sampling]")
        protocol = f'''[protocol]
prompt_sha256 = "{prompt_sha256}"
cap_error_sha256 = "{_cap_error_sha256(maximum_tool_calls)}"
tool_profile = "{tool_profile.value}"
tool_schema_sha256 = "{tool_profile.tool_set_sha256}"
enabled_tool_names = {list(tool_profile.tool_names)!r}
maximum_tool_calls = {maximum_tool_calls}
success_observation_protocol_id = "{observation_id.value}"
action_boundary_protocol_id = "{NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2.value}"

'''.replace("'", '"')
        text = text[:protocol_start] + protocol + text[sampling_start:]
        sampling_start = text.index("[sampling]")
        stop_strings = (
            "[]" if profile is PolicyMethodProfile.NO_TOOL else '["</tool_call>"]'
        )
        sampling_end = text.index("[reward]")
        sampling = f"""[sampling]
trajectories_per_prompt = {trajectories_per_prompt}
temperature = 1.0
top_p = 1.0
top_k = -1
min_p = 0.0
repetition_penalty = 1.0
presence_penalty = 0.0
frequency_penalty = 0.0
max_response_length = {max_response_length}
asynchronous_staleness_steps = 0
do_sample = true
backend = "vllm"
backend_version = "0.12.0"
logit_processors = []
logprob_measurement = "after_sampling_transforms"
stop_token_ids = [151645]
stop_strings = {stop_strings}
include_stop_str_in_output = true
ignore_eos = false
rollout_master_seed = {rollout_seed}
seed_derivation_name = "content-addressed-vllm-turn-rng-v1"
seed_derivation_sha256 = "fe8d2da92471dcf97ff195f9c1e085fe422673e3b0a4b863ad28ca528ace867e"

"""
        text = text[:sampling_start] + sampling + text[sampling_end:]
        reward_start = text.index("[reward]")
        optimizer_start = text.index("[optimizer]")
        judge_path = (
            Path(__file__).parents[2]
            / "configs/policy/judges/qwen25_72b_rl_answer_judge_v1.json"
        ).resolve()
        judge_sha256 = hashlib.sha256(judge_path.read_bytes()).hexdigest()
        judge_model_route = (
            'judge_model_route = "qwen2.5_72b"\n' if performance_v2 else ""
        )
        reward = f'''[reward]
task_kind = "mixed"
answer_verifier = "rule_first_qwen25_72b"
answer_verifier_sha256 = "{POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256}"
judge_mode = "qwen25_72b_semantic_fallback"
judge_reason = "method config fixture"
judge_config_path = {_q(judge_path)}
judge_config_sha256 = "{judge_sha256}"
{judge_model_route}profile = "stage3-shaped-v1"
answer_reward_scale = 1.25
repeated_call_penalty = 0.0
protocol_error_penalty = 0.75
tool_utility_reward_enabled = false
focus_reward_enabled = false
grounding_reward_enabled = false
visual_quality_judge_mode = "disabled"

'''
        text = text[:reward_start] + reward + text[optimizer_start:]
        if performance_v2:
            text = text.replace(
                POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA,
                POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA,
                1,
            )
            migrated_lines = []
            moved_fields = {
                "vllm_tensor_parallel_size",
                "vllm_enable_chunked_prefill",
                "vllm_enforce_eager",
            }
            for line in text.splitlines(keepends=True):
                if line.split("=", 1)[0].strip() in moved_fields:
                    continue
                migrated_lines.append(line)
            text = "".join(migrated_lines)
            capture_sizes = ", ".join(
                str(value) for value in vllm_cuda_graph_capture_sizes
            )
            performance = f'''[performance]
dynamic_token_batching = {str(dynamic_token_batching).lower()}
use_remove_padding = {str(use_remove_padding).lower()}
enable_gradient_checkpointing = {str(enable_gradient_checkpointing).lower()}
vllm_enable_prefix_caching = {str(vllm_enable_prefix_caching).lower()}
vllm_enable_chunked_prefill = {str(vllm_enable_chunked_prefill).lower()}
vllm_enable_cuda_graph = {str(vllm_enable_cuda_graph).lower()}
vllm_cuda_graph_capture_sizes = [{capture_sizes}]
vllm_tensor_parallel_size = 1
rollout_logprob_bypass = true
reference_replay_mode = "{reference_replay_mode}"
judge_dispatch_mode = "{judge_dispatch_mode}"
judge_max_concurrency_per_worker = {judge_max_concurrency_per_worker}

'''
            text = text.replace("[capacity]\n", performance + "[capacity]\n", 1)
        path = tmp_path / f"method-{profile.value}.toml"
        path.write_text(text, encoding="utf-8")
        return path, text

    return build


@pytest.mark.parametrize(
    ("profile", "tool_profile", "prompt_sha256", "observation_id"),
    _METHOD_CASES,
)
def test_unified_method_schema_binds_protocol_and_config_owned_values(
    method_config_factory: Callable[..., tuple[Path, str]],
    profile: PolicyMethodProfile,
    tool_profile: NativeToolCapabilityProfile,
    prompt_sha256: str,
    observation_id: NativeSuccessObservationProtocolId,
) -> None:
    path, _ = method_config_factory(
        profile=profile,
        tool_profile=tool_profile,
        prompt_sha256=prompt_sha256,
        observation_id=observation_id,
    )

    config = load_policy_e2e_smoke_run_config(path)

    assert config.method is not None
    assert config.method.matrix_id == "config-owned-method-matrix"
    assert config.method.profile is profile
    assert isinstance(config.policy, PolicyMethodExperimentConfig)
    assert isinstance(config.policy.sampling, PolicyMethodSamplingConfig)
    assert config.policy.image_max_pixels == 345_678
    assert config.policy.sampling.trajectories_per_prompt == 3
    assert config.policy.sampling.max_response_length == 1_234
    assert config.rollout_rng.master_seed == 77
    assert config.protocol.maximum_tool_calls == 5
    assert config.protocol.success_observation_protocol_id is observation_id
    assert config.protocol.action_boundary_protocol_id is (
        NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
    )
    assert config.reward.answer_reward_scale == 1.25
    assert config.reward.repeated_call_penalty == 0.0
    assert config.reward.protocol_error_penalty == 0.75
    assert config.reward.visual_quality_judge_mode == "disabled"


def test_method_v2_binds_explicit_performance_without_two_tp_truths(
    method_config_factory: Callable[..., tuple[Path, str]],
) -> None:
    path, _ = method_config_factory(
        profile=PolicyMethodProfile.CROP,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        prompt_sha256=VISUAL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
        performance_v2=True,
    )

    config = load_policy_e2e_smoke_run_config(path)

    assert config.schema_version == POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA
    assert config.performance is not None
    assert config.performance.dynamic_token_batching is True
    assert config.performance.use_remove_padding is True
    assert config.performance.enable_gradient_checkpointing is True
    assert config.performance.vllm_enable_prefix_caching is True
    assert config.performance.vllm_enable_chunked_prefill is True
    assert config.performance.vllm_enable_cuda_graph is True
    assert config.performance.vllm_cuda_graph_capture_sizes == (1, 2, 4, 8)
    assert config.performance.vllm_tensor_parallel_size == 1
    assert config.performance.rollout_logprob_bypass is True
    assert config.performance.reference_replay_mode == "off"
    assert config.performance.judge_dispatch_mode == "dedicated_thread_pool"
    assert config.performance.judge_max_concurrency_per_worker == 8
    assert config.distributed.vllm_tensor_parallel_size == 1
    assert config.capacity.vllm_enable_chunked_prefill is True
    assert config.capacity.vllm_enforce_eager is False
    assert "vllm_tensor_parallel_size" not in config.as_record()["distributed"]
    assert "vllm_enforce_eager" not in config.as_record()["capacity"]


def test_method_v2_explicit_alternate_binds_effective_verifier_identity(
    tmp_path: Path,
    method_config_factory: Callable[..., tuple[Path, str]],
) -> None:
    path, text = method_config_factory(
        profile=PolicyMethodProfile.CROP,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        prompt_sha256=VISUAL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
        performance_v2=True,
    )
    default = load_policy_e2e_smoke_run_config(path)
    source_judge_path = (
        Path(__file__).parents[2]
        / "configs/policy/judges/qwen25_72b_rl_answer_judge_v1.json"
    )
    judge_payload = json.loads(source_judge_path.read_text(encoding="utf-8"))
    judge_payload["identity"] = "test-small-alternate-judge-v1"
    judge_payload["model"].update(
        {
            "repository": "tests/Small-Alternate-Judge",
            "revision": "small-alternate-v1",
            "local_path": "/tests/models/small-alternate-v1",
            "served_name": "test-small-alternate-judge",
        }
    )
    alternate_path = tmp_path / "alternate-judge.json"
    alternate_path.write_text(
        json.dumps(judge_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    alternate_file_sha256 = hashlib.sha256(alternate_path.read_bytes()).hexdigest()
    alternate = load_openai_compatible_judge(
        alternate_path,
        expected_file_sha256=alternate_file_sha256,
    )
    effective_verifier_sha256 = policy_e2e_mixed_alternate_answer_verifier_sha256(
        alternate.model_identity
    )
    replacements = {
        'answer_verifier = "rule_first_qwen25_72b"': (
            f'answer_verifier = "{POLICY_E2E_MIXED_ALTERNATE_ANSWER_VERIFIER}"'
        ),
        f'answer_verifier_sha256 = "{POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256}"': (
            f'answer_verifier_sha256 = "{effective_verifier_sha256}"'
        ),
        'judge_mode = "qwen25_72b_semantic_fallback"': (
            f'judge_mode = "{POLICY_E2E_MIXED_ALTERNATE_JUDGE_MODE}"'
        ),
        f"judge_config_path = {_q(default.reward.judge_config_path)}": (
            f"judge_config_path = {_q(alternate_path)}"
        ),
        f'judge_config_sha256 = "{default.reward.judge_config_sha256}"': (
            f'judge_config_sha256 = "{alternate_file_sha256}"'
        ),
        'judge_model_route = "qwen2.5_72b"\n': (
            'judge_model_route = "explicit_alternate"\n'
            'alternate_judge_model_name = "test-small-alternate-judge"\n'
            "alternate_judge_model_identity_sha256 = "
            f'"{alternate.model_identity.sha256}"\n'
            "alternate_semantics_acknowledged = true\n"
        ),
    }
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")

    config = load_policy_e2e_smoke_run_config(path)

    assert config.reward.judge_model_route == "explicit_alternate"
    assert config.reward.answer_verifier == POLICY_E2E_MIXED_ALTERNATE_ANSWER_VERIFIER
    assert config.reward.answer_verifier_sha256 == effective_verifier_sha256
    assert config.reward.answer_verifier_sha256 != (
        POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256
    )
    assert config.reward.judge_mode == POLICY_E2E_MIXED_ALTERNATE_JUDGE_MODE
    assert config.reward.alternate_judge_model_name == "test-small-alternate-judge"
    assert config.reward.alternate_judge_model_identity == alternate.model_identity
    assert config.reward.alternate_semantics_acknowledged is True

    path.write_text(
        text.replace(
            effective_verifier_sha256,
            POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256,
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="alternate model-bound contract"):
        load_policy_e2e_smoke_run_config(path)

    path.write_text(
        text.replace(
            "alternate_semantics_acknowledged = true",
            "alternate_semantics_acknowledged = false",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="semantic acknowledgement"):
        load_policy_e2e_smoke_run_config(path)

    path.write_text(
        text.replace(
            'alternate_judge_model_name = "test-small-alternate-judge"\n',
            "",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires model name, identity SHA256"):
        load_policy_e2e_smoke_run_config(path)


def test_method_v1_keeps_legacy_performance_out_of_source_identity(
    method_config_factory: Callable[..., tuple[Path, str]],
) -> None:
    path, _ = method_config_factory(
        profile=PolicyMethodProfile.CROP,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        prompt_sha256=VISUAL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
    )

    config = load_policy_e2e_smoke_run_config(path)

    assert config.schema_version == POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA
    assert config.performance is not None
    assert config.performance.dynamic_token_batching is False
    assert config.performance.use_remove_padding is False
    assert config.performance.enable_gradient_checkpointing is False
    assert config.performance.vllm_enable_prefix_caching is False
    assert config.performance.judge_dispatch_mode == "inherit"
    assert config.performance.reference_replay_mode == "full_diagnostic"
    assert "performance" not in config.as_record()


def test_method_v2_accepts_cuda_graph_capture_size_above_sequence_cap(
    method_config_factory: Callable[..., tuple[Path, str]],
) -> None:
    path, _ = method_config_factory(
        profile=PolicyMethodProfile.CROP,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        prompt_sha256=VISUAL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
        performance_v2=True,
        vllm_cuda_graph_capture_sizes=(1, 16),
    )

    config = load_policy_e2e_smoke_run_config(path)

    assert config.capacity.vllm_max_num_seqs == 8
    assert config.capacity.vllm_max_num_batched_tokens == 32_768
    assert config.performance is not None
    assert config.performance.vllm_cuda_graph_capture_sizes == (1, 16)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"vllm_cuda_graph_capture_sizes": ()},
            "enabled CUDA graph requires explicit",
        ),
        (
            {
                "vllm_enable_cuda_graph": False,
                "vllm_cuda_graph_capture_sizes": (1,),
            },
            "disabled CUDA graph requires empty",
        ),
        (
            {"vllm_cuda_graph_capture_sizes": (1, 4, 4)},
            "strictly increasing and unique",
        ),
        (
            {"vllm_cuda_graph_capture_sizes": (1, 32_769)},
            "cannot exceed capacity.vllm_max_num_batched_tokens",
        ),
        (
            {
                "judge_dispatch_mode": "inline",
                "judge_max_concurrency_per_worker": 2,
            },
            "requires maximum concurrency 1",
        ),
        (
            {"judge_max_concurrency_per_worker": 257},
            r"must be in \[1, 256\]",
        ),
    ),
)
def test_method_v2_rejects_invalid_performance_contract(
    method_config_factory: Callable[..., tuple[Path, str]],
    changes: dict[str, object],
    message: str,
) -> None:
    path, _ = method_config_factory(
        profile=PolicyMethodProfile.CROP,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        prompt_sha256=VISUAL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
        performance_v2=True,
        **changes,
    )

    with pytest.raises(ValueError, match=message):
        load_policy_e2e_smoke_run_config(path)


def test_no_tool_runtime_uses_configured_eos_without_tool_call_stop(
    method_config_factory: Callable[..., tuple[Path, str]],
) -> None:
    path, _ = method_config_factory(
        profile=PolicyMethodProfile.NO_TOOL,
        tool_profile=NativeToolCapabilityProfile.NO_TOOL,
        prompt_sha256=NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1,
    )
    config = load_policy_e2e_smoke_run_config(path)

    termination = _policy_termination_contract(config)

    expected_outcomes = (
        VLLMTerminationOutcome("stop", 151645),
        VLLMTerminationOutcome("length", None),
    )
    assert termination.required_request_stop_strings == ()
    assert termination.required_request_stop_token_ids == (151645,)
    assert termination.final_turn_outcomes == expected_outcomes
    assert termination.tool_call_outcomes == expected_outcomes


def test_tool_runtime_keeps_complete_tool_call_closer_contract(
    method_config_factory: Callable[..., tuple[Path, str]],
) -> None:
    path, _ = method_config_factory(
        profile=PolicyMethodProfile.CROP,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        prompt_sha256=VISUAL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
    )
    config = load_policy_e2e_smoke_run_config(path)

    termination = _policy_termination_contract(config)

    assert termination.required_request_stop_strings == ("</tool_call>",)
    assert termination.tool_call_outcomes == (
        VLLMTerminationOutcome("stop", "</tool_call>"),
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            lambda text: text.replace(
                NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2.value,
                NativeActionBoundaryProtocolId.LEGACY_ANSWER_OVER_ACTION_V1.value,
            ),
            "method action-boundary protocol",
        ),
        (
            lambda text: text.replace(
                NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1.value,
                NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1.value,
            ),
            "method success-observation protocol",
        ),
        (
            lambda text: text.replace(
                'stop_strings = ["</tool_call>"]', "stop_strings = []"
            ),
            "method complete stop contract",
        ),
        (
            lambda text: text.replace(
                "include_stop_str_in_output = true",
                "include_stop_str_in_output = false",
            ),
            "complete closing tag remains policy-sampled|method complete stop contract",
        ),
    ),
)
def test_unified_method_schema_rejects_protocol_drift(
    method_config_factory: Callable[..., tuple[Path, str]],
    mutation: Callable[[str], str],
    error: str,
) -> None:
    path, text = method_config_factory(
        profile=PolicyMethodProfile.TGVF_SHORT,
        tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
        prompt_sha256=TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1,
    )
    path.write_text(mutation(text), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_policy_e2e_smoke_run_config(path)


def test_method_policy_direct_construction_is_typed_but_scale_flexible() -> None:
    sampling = PolicyMethodSamplingConfig(
        trajectories_per_prompt=7,
        max_response_length=3_333,
    )
    policy = PolicyMethodExperimentConfig(
        method=PolicyMethodProfile.CROP,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        enabled_tool_names=NativeToolCapabilityProfile.CROP_ONLY.tool_names,
        max_tgvf_call_attempts=9,
        image_max_pixels=456_789,
        sampling=sampling,
    )

    assert policy.sampling is sampling
    assert policy.max_tgvf_call_attempts == 9
    assert policy.image_max_pixels == 456_789
    with pytest.raises(TypeError, match="PolicyMethodSamplingConfig"):
        replace(policy, sampling=PilotSamplingConfig())
    with pytest.raises(ValueError, match="positive integer"):
        replace(policy, max_tgvf_call_attempts=0)
    with pytest.raises(ValueError, match="method experiment requires tool_profile"):
        replace(policy, tool_profile=NativeToolCapabilityProfile.TGVF_ONLY)


@pytest.mark.parametrize("native_deepstack_enabled", (True, False))
def test_method_schema_exposes_native_deepstack_ablation(
    method_config_factory: Callable[..., tuple[Path, str]],
    native_deepstack_enabled: bool,
) -> None:
    path, _ = method_config_factory(
        profile=PolicyMethodProfile.CROP,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        prompt_sha256=VISUAL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
        native_deepstack_enabled=native_deepstack_enabled,
    )

    config = load_policy_e2e_smoke_run_config(path)

    assert config.policy.native_deepstack_enabled is native_deepstack_enabled
    direct = replace(config.policy, native_deepstack_enabled=native_deepstack_enabled)
    assert direct.native_deepstack_enabled is native_deepstack_enabled


def test_legacy_policy_schema_retains_native_deepstack_enabled_semantics() -> None:
    with pytest.raises(ValueError, match="native_deepstack_enabled"):
        PolicyPilotV1Config(native_deepstack_enabled=False)


@pytest.mark.parametrize("interval", (2, 7))
def test_method_schema_rejects_unsupported_weight_sync_intervals(
    method_config_factory: Callable[..., tuple[Path, str]],
    interval: int,
) -> None:
    path, _ = method_config_factory(
        profile=PolicyMethodProfile.NO_TOOL,
        tool_profile=NativeToolCapabilityProfile.NO_TOOL,
        prompt_sha256=NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1,
        weight_sync_interval_optimizer_steps=interval,
    )

    with pytest.raises(ValueError, match="pinned synchronous veRL trainer"):
        load_policy_e2e_smoke_run_config(path)


def test_legacy_resolution_schema_is_read_only_alias(
    method_config_factory: Callable[..., tuple[Path, str]],
) -> None:
    path, text = method_config_factory(
        profile=PolicyMethodProfile.TGVF_SHORT,
        tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
        prompt_sha256=TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1,
    )
    text = text.replace(
        POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA,
        POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    ).replace(
        '[method]\nmatrix_id = "config-owned-method-matrix"\nprofile = "tgvf_short"\n',
        "",
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="schemas are read-only"):
        load_policy_e2e_smoke_run_config(path)
    config = load_policy_e2e_smoke_run_config(
        path,
        allow_historical_read_only_contract=True,
    )

    assert config.method is not None
    assert config.method.profile is PolicyMethodProfile.TGVF_SHORT
    assert config.method.legacy_schema_alias == config.schema_version


def test_method_capacity_preserves_rollout_and_reference_actor_floor(
    method_config_factory: Callable[..., tuple[Path, str]],
) -> None:
    path, text = method_config_factory(
        profile=PolicyMethodProfile.NO_TOOL,
        tool_profile=NativeToolCapabilityProfile.NO_TOOL,
        prompt_sha256=NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1,
    )
    path.write_text(
        text.replace(
            "rollout_log_prob_max_token_len_per_gpu = 98304",
            "rollout_log_prob_max_token_len_per_gpu = 98303",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be smaller than the actor bound"):
        load_policy_e2e_smoke_run_config(path)
