"""Backend-neutral identity contract for post-training policy evaluation.

The LoRA evaluator and the immutable full-model snapshot loader both need to
produce the same evaluation identity.  Keeping that composition here makes
the dependency direction explicit: backend modules supply only their snapshot
record, policy-config path, and (when applicable) a step-zero equivalence
proof.  This module never imports either backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

from tgvf_rl.environment.native_appender import NativeSuccessObservationContract
from tgvf_rl.framework.vllm import (
    VLLMOutputDecodingContract,
    VLLMTerminationOutcome,
    VLLMTurnTerminationContract,
)
from tgvf_rl.public_api_compat import (
    rebind_public_class,
    rebind_public_function,
)
from tgvf_rl.protocol import (
    native_assistant_dialect_for_model,
    NativeActionBoundaryProtocolId,
    NativeAssistantDialect,
    NativeSuccessObservationProtocolId,
    NativeToolCapabilityProfile,
)
from tgvf_rl.protocol.state_machine import CapErrorBehavior

from .policy_evaluation_config import (
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    POLICY_EVALUATION_PROTOCOLS,
    TRAINING_RUN_EVALUATION_PROTOCOL,
    _require_sha256,
)


POLICY_EVALUATION_IDENTITY_SCHEMA = "tgvf-policy-evaluation-identity-v1"
POLICY_EVAL_CONTRACT_SCHEMA = "tgvf-policy-eval-contract-v1"


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_benchmark_task_path(config: object) -> Path:
    filename = (
        "coredev-official-tasks.jsonl"
        if bool(getattr(config, "uses_legacy_coredev_manifest"))
        else "policy-benchmark-tasks.jsonl"
    )
    return Path(getattr(config, "output_root")) / "runtime" / filename


@dataclass(frozen=True, slots=True)
class PolicyEvalContract:
    """Immutable identity for every behavior-relevant evaluation boundary."""

    evaluation_protocol: str
    training_image_max_pixels: int
    declared_image_max_pixels: int
    effective_image_max_pixels: int
    prompt_protocol_id: str
    prompt_identity_sha256: str
    parser_protocol_id: str
    parser_identity_sha256: str
    success_observation_protocol_id: NativeSuccessObservationProtocolId
    success_observation_identity_sha256: str
    action_boundary_protocol_id: NativeActionBoundaryProtocolId
    action_boundary_identity_sha256: str
    schema_version: str = POLICY_EVAL_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.evaluation_protocol not in POLICY_EVALUATION_PROTOCOLS:
            raise ValueError("evaluation contract protocol differs")
        for name in (
            "training_image_max_pixels",
            "declared_image_max_pixels",
            "effective_image_max_pixels",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("prompt_protocol_id", "parser_protocol_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "prompt_identity_sha256",
            "parser_identity_sha256",
            "success_observation_identity_sha256",
            "action_boundary_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if not isinstance(
            self.success_observation_protocol_id,
            NativeSuccessObservationProtocolId,
        ):
            raise TypeError(
                "success_observation_protocol_id must be an explicit protocol ID"
            )
        if not isinstance(
            self.action_boundary_protocol_id,
            NativeActionBoundaryProtocolId,
        ):
            raise TypeError(
                "action_boundary_protocol_id must be an explicit protocol ID"
            )

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_protocol": self.evaluation_protocol,
            "pixels": {
                "training_image_max_pixels": self.training_image_max_pixels,
                "declared_image_max_pixels": self.declared_image_max_pixels,
                "effective_image_max_pixels": self.effective_image_max_pixels,
            },
            "prompt": {
                "protocol_id": self.prompt_protocol_id,
                "identity_sha256": self.prompt_identity_sha256,
            },
            "parser": {
                "protocol_id": self.parser_protocol_id,
                "identity_sha256": self.parser_identity_sha256,
            },
            "success_observation": {
                "protocol_id": self.success_observation_protocol_id.value,
                "identity_sha256": self.success_observation_identity_sha256,
            },
            "action_boundary": {
                "protocol_id": self.action_boundary_protocol_id.value,
                "identity_sha256": self.action_boundary_identity_sha256,
            },
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.canonical_payload)


def _decoding_contract() -> VLLMOutputDecodingContract:
    return VLLMOutputDecodingContract(
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        output_kind="final_only",
    )


def _termination_contract(run: object) -> VLLMTurnTerminationContract:
    sampling = getattr(getattr(run, "policy"), "sampling")
    return VLLMTurnTerminationContract(
        required_request_stop_strings=tuple(sampling.stop_strings or ()),
        required_request_stop_token_ids=tuple(sampling.stop_token_ids or ()),
        include_stop_str_in_output=bool(sampling.include_stop_str_in_output),
        tool_call_terminal_suffixes=("",),
        tool_call_outcomes=(VLLMTerminationOutcome("stop", "</tool_call>"),),
        final_turn_outcomes=tuple(
            VLLMTerminationOutcome("stop", token_id)
            for token_id in tuple(sampling.stop_token_ids or ())
        )
        + (
            VLLMTerminationOutcome("stop", None),
            VLLMTerminationOutcome("length", None),
        ),
    )


def policy_eval_parser_identity(
    *, evaluation_protocol: str, run: object
) -> tuple[str, str]:
    if evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        from tgvf_rl.policy.deepeyes_official_protocol import (
            DEEPEYES_TOOL_NAME,
            DEEPEYES_TOOL_PARSER,
        )

        protocol_id = "deepeyes-hermes-last-complete-crop-call-v1"
        payload = {
            "implementation": (
                "tgvf_rl.policy.deepeyes_official_protocol.parse_hermes_crop_call"
            ),
            "upstream_parser": DEEPEYES_TOOL_PARSER,
            "enabled_tool_names": [DEEPEYES_TOOL_NAME],
            "multiple_complete_calls": "select_last",
        }
    else:
        protocol = getattr(run, "protocol")
        protocol_id = "strict-native-single-tool-call-v1"
        payload = {
            "implementation": "tgvf_rl.protocol.parser.StrictToolCallParser",
            "enabled_tool_names": list(protocol.enabled_tool_names),
            "tool_schema_sha256": protocol.tool_schema_sha256,
            "complete_call_count": 1,
            "trailing_assistant_text": "reject",
        }
    return protocol_id, canonical_json_sha256(payload)


def policy_eval_action_boundary_identity(
    *,
    evaluation_protocol: str,
    run: object,
    action_boundary_protocol_id: NativeActionBoundaryProtocolId,
) -> tuple[NativeActionBoundaryProtocolId, str]:
    sampling = getattr(getattr(run, "policy"), "sampling")
    if evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        payload: dict[str, object] = {
            "dispatcher": (
                "tgvf_rl.evaluation.policy_official_visible."
                "OfficialVisiblePolicyEvaluator"
            ),
            "boundary_classifier": (
                "tgvf_rl.protocol.action_boundary.classify_assistant_action_boundary"
            ),
            "tool_marker": "<tool_call>...</tool_call>",
            "required_request_stop_strings": list(
                getattr(sampling, "stop_strings", ()) or ()
            ),
            "required_request_stop_token_ids": list(
                getattr(sampling, "stop_token_ids", ()) or ()
            ),
            "include_stop_str_in_output": bool(
                getattr(sampling, "include_stop_str_in_output", False)
            ),
        }
        if (
            action_boundary_protocol_id
            is NativeActionBoundaryProtocolId.LEGACY_ANSWER_OVER_ACTION_V1
        ):
            payload.update(
                {
                    "trailing_final_answer_precedence": "final_answer",
                    "multiple_complete_calls": "execute_last",
                    "malformed_tool_call_tags": "reject",
                }
            )
        elif (
            action_boundary_protocol_id
            is NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
        ):
            payload.update(
                {
                    "complete_call_count": 1,
                    "terminal_tool_call_required": True,
                    "trailing_assistant_text": "reject",
                    "multiple_complete_calls": "reject",
                    "malformed_tool_call_tags": "reject",
                }
            )
        else:  # pragma: no cover - enum expansion requires an explicit contract
            raise ValueError("official-visible action-boundary protocol is unsupported")
    else:
        if (
            action_boundary_protocol_id
            is not NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
        ):
            raise ValueError("training-run evaluator requires strict boundary v2")
        payload = {
            "dispatcher": "tgvf_rl.environment.agent_loop.FrameworkNeutralAgentLoop",
            "tool_marker_precedence": "any_marker_routes_strict_parser",
            "multiple_complete_calls": "reject",
            "trailing_assistant_text": "reject",
            "cap_error_behavior": CapErrorBehavior.ONE_FINAL_ANSWER_TURN.value,
            "decoding": _decoding_contract().canonical_payload,
            "termination": _termination_contract(run).canonical_payload,
        }
    payload["protocol_id"] = action_boundary_protocol_id.value
    return action_boundary_protocol_id, canonical_json_sha256(payload)


def policy_eval_observation_identity(
    contract: NativeSuccessObservationContract,
) -> str:
    from tgvf_rl.environment.native_appender import (
        QWEN_NATIVE_IMAGE_PLACEHOLDER,
        QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT_SHA256,
        QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256,
        QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX,
        qwen_native_response_suffix,
    )
    from tgvf_rl.protocol.tool_prompts import (
        IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT_SHA256,
        QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER_SHA256,
        TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE_SHA256,
        TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE_SHA256,
    )

    if (
        contract.protocol_id
        is NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1
    ):
        return QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256
    if (
        contract.protocol_id
        is NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1
    ):
        return QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT_SHA256
    response_template_sha256 = {
        NativeToolCapabilityProfile.TGVF_ONLY: (
            TGVF_FOCUS_SUCCESS_RESPONSE_TEMPLATE_SHA256
        ),
        NativeToolCapabilityProfile.CROP_ONLY: (
            IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT_SHA256
        ),
        NativeToolCapabilityProfile.CROP_TGVF: (
            TGVF_CROP_SUCCESS_RESPONSE_TEMPLATE_SHA256
        ),
    }[contract.tool_profile]
    payload = {
        "protocol_id": contract.protocol_id.value,
        "tool_profile": contract.tool_profile.value,
        "assistant_dialect": contract.assistant_dialect.value,
        "prefix_sha256": hashlib.sha256(
            QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX.encode("utf-8")
        ).hexdigest(),
        "response_template_sha256": response_template_sha256,
        "image_placeholder_sha256": hashlib.sha256(
            QWEN_NATIVE_IMAGE_PLACEHOLDER.encode("utf-8")
        ).hexdigest(),
        "reasoning_reminder_sha256": (
            QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER_SHA256
            if contract.assistant_dialect is NativeAssistantDialect.QWEN3_VL_INSTRUCT
            else None
        ),
        "suffix_sha256": hashlib.sha256(
            qwen_native_response_suffix(contract.assistant_dialect).encode("utf-8")
        ).hexdigest(),
    }
    return canonical_json_sha256(payload)


def effective_evaluation_image_max_pixels(config: object, snapshot: object) -> int:
    """Return the pixel cap actually consumed by the selected evaluator."""

    if (
        getattr(config, "evaluation_protocol")
        == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
    ):
        effective = getattr(getattr(snapshot, "run"), "policy").image_max_pixels
        if getattr(config, "declared_image_max_pixels") != effective:
            raise ValueError(
                "official-visible declared pixels differ from its effective runtime"
            )
        return int(effective)
    return int(getattr(config, "declared_image_max_pixels"))


def build_policy_eval_contract(
    config: object,
    snapshot: object,
    *,
    contract_type: type[PolicyEvalContract] = PolicyEvalContract,
) -> PolicyEvalContract:
    """Bind pixels, prompt, parser, observation bytes, and action boundary."""

    run = getattr(snapshot, "run")
    dialect = native_assistant_dialect_for_model(run.model.model_name)
    if (
        getattr(config, "evaluation_protocol")
        == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
    ):
        from tgvf_rl.policy.deepeyes_official_protocol import (
            DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA,
            VISUAL_PROMPT_IDENTITY,
        )

        observation_profile = NativeToolCapabilityProfile.CROP_ONLY
        prompt_protocol_id = (
            f"{DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA}:{VISUAL_PROMPT_IDENTITY.version}"
        )
        prompt_identity_sha256 = VISUAL_PROMPT_IDENTITY.bundle_sha256
    else:
        observation_profile = run.protocol.tool_profile
        prompt_protocol_id = "tgvf-native-run-prompt-v1"
        prompt_identity_sha256 = run.protocol.prompt_sha256

    observation_contract = NativeSuccessObservationContract(
        protocol_id=getattr(config, "success_observation_protocol_id"),
        tool_profile=observation_profile,
        assistant_dialect=dialect,
    )
    parser_protocol_id, parser_identity_sha256 = policy_eval_parser_identity(
        evaluation_protocol=getattr(config, "evaluation_protocol"), run=run
    )
    action_protocol_id, action_identity_sha256 = policy_eval_action_boundary_identity(
        evaluation_protocol=getattr(config, "evaluation_protocol"),
        run=run,
        action_boundary_protocol_id=getattr(config, "action_boundary_protocol_id"),
    )
    effective_image_max_pixels = effective_evaluation_image_max_pixels(config, snapshot)
    return contract_type(
        evaluation_protocol=getattr(config, "evaluation_protocol"),
        training_image_max_pixels=run.policy.image_max_pixels,
        declared_image_max_pixels=getattr(config, "declared_image_max_pixels"),
        effective_image_max_pixels=effective_image_max_pixels,
        prompt_protocol_id=prompt_protocol_id,
        prompt_identity_sha256=prompt_identity_sha256,
        parser_protocol_id=parser_protocol_id,
        parser_identity_sha256=parser_identity_sha256,
        success_observation_protocol_id=observation_contract.protocol_id,
        success_observation_identity_sha256=policy_eval_observation_identity(
            observation_contract
        ),
        action_boundary_protocol_id=action_protocol_id,
        action_boundary_identity_sha256=action_identity_sha256,
    )


def evaluation_protocol_identity(
    config: object,
    snapshot: object,
    *,
    is_lora_snapshot: bool,
    step_zero_equivalence: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    evaluation_protocol = getattr(config, "evaluation_protocol")
    run = getattr(snapshot, "run")
    if evaluation_protocol == TRAINING_RUN_EVALUATION_PROTOCOL:
        if not is_lora_snapshot:
            raise ValueError("training-run evaluation requires a LoRA snapshot")
        protocol = run.protocol
        return {
            "profile": TRAINING_RUN_EVALUATION_PROTOCOL,
            "prompt_sha256": protocol.prompt_sha256,
            "tool_schema_sha256": protocol.tool_schema_sha256,
            "tool_profile": protocol.tool_profile.value,
            "enabled_tool_names": list(protocol.enabled_tool_names),
            "maximum_tool_calls": protocol.maximum_tool_calls,
            "native_pixels": False,
        }
    if evaluation_protocol != DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        raise ValueError("unsupported policy evaluation protocol")
    from tgvf_rl.policy.deepeyes_official_protocol import (
        DEEPEYES_MAX_ACTIVE_PERCEPTION,
        DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA,
        DEEPEYES_TOOL_NAME,
        DEEPEYES_TOOL_PARSER,
        SYSTEM_PROMPT_V2_SHA256,
        USER_PROMPT_V2_SHA256,
        VISUAL_PROMPT_IDENTITY,
    )
    from tgvf_rl.qwen.crop_coordinates import (
        QWEN3_CROP_CONVERSION_VERSION,
        QWEN3_CROP_COORDINATE_SPACE,
    )

    if run.model.model_name != "Qwen3-VL-8B-Instruct":
        raise ValueError(
            "official-visible base evaluation requires Qwen3-VL-8B-Instruct"
        )
    identity: dict[str, object] = {
        "profile": DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
        "protocol_schema_version": DEEPEYES_OFFICIAL_PROTOCOL_SCHEMA,
        "source_repository": "https://github.com/Visual-Agent/DeepEyes",
        "source_commit": "11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1",
        "prompt_source_path": "verl/workers/agent/envs/mm_process_engine/prompt.py",
        "prompt_source_file_sha256": (
            "35ef1bae8da550827bc53e23751e64d4c8eecc76d9170ea5673aa2493628cc23"
        ),
        "crop_source_path": (
            "verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py"
        ),
        "crop_source_file_sha256": (
            "0d56b2ff584fe56e68f20bbb4d25a9774ecbab605ad02cdaf1dac7cd6fa8bc60"
        ),
        "system_prompt_sha256": SYSTEM_PROMPT_V2_SHA256,
        "user_prompt_sha256": USER_PROMPT_V2_SHA256,
        "prompt_bundle_sha256": VISUAL_PROMPT_IDENTITY.bundle_sha256,
        "visible_system_tool_schema": True,
        "template_tools_argument": [],
        "tool_parser": DEEPEYES_TOOL_PARSER,
        "enabled_tool_names": [DEEPEYES_TOOL_NAME],
        "maximum_tool_calls": DEEPEYES_MAX_ACTIVE_PERCEPTION,
        "coordinate_mapper": "qwen_0_1000_to_source_v1",
        "crop_coordinate_space": QWEN3_CROP_COORDINATE_SPACE,
        "crop_coordinate_conversion_version": QWEN3_CROP_CONVERSION_VERSION,
        "crop_coordinate_reference_size": [1000, 1000],
        "crop_source": "immutable_original_image",
        "native_pixels": True,
        "precomputed_image_embeds": False,
        "image_max_pixels": effective_evaluation_image_max_pixels(config, snapshot),
        "native_image_limit_per_prompt": DEEPEYES_MAX_ACTIVE_PERCEPTION + 1,
        "observation_role": "user",
        "observation_envelope": (
            "<tool_response><image>USER_PROMPT_V2</tool_response>"
        ),
    }
    if snapshot.policy_version.optimizer_step == 0:
        identity["base_equivalence"] = dict(step_zero_equivalence())
    return identity


def build_policy_evaluation_identity(
    config: object,
    snapshot: object,
    *,
    is_lora_snapshot: bool,
    policy_snapshot: Mapping[str, object],
    policy_config_path: Path,
    step_zero_equivalence: Callable[[], Mapping[str, object]],
    contract_type: type[PolicyEvalContract] = PolicyEvalContract,
) -> dict[str, object]:
    """Compose one backend-independent evaluation identity payload."""

    task_path = policy_benchmark_task_path(config).resolve()
    task_sha256 = sha256_file(task_path)
    expected_task_sha256 = getattr(config, "task_manifest_sha256")
    if expected_task_sha256 is not None and task_sha256 != expected_task_sha256:
        raise ValueError("bound policy benchmark task manifest SHA256 changed")
    eval_contract = build_policy_eval_contract(
        config, snapshot, contract_type=contract_type
    )
    run = getattr(snapshot, "run")
    content: dict[str, object] = {
        "schema_version": POLICY_EVALUATION_IDENTITY_SCHEMA,
        "evaluation_id": getattr(config, "evaluation_id"),
        "evaluation_schema_version": getattr(config, "schema_version"),
        "policy_config_path": str(policy_config_path.resolve()),
        "policy_config_file_sha256": sha256_file(policy_config_path),
        "policy_run_config_identity_sha256": run.identity_sha256,
        "model_identity": asdict(run.model),
        "policy_snapshot": dict(policy_snapshot),
        "task_manifest": {
            "path": str(task_path),
            "sha256": task_sha256,
            "task_count": getattr(config, "expected_task_count"),
            "single_image_count": getattr(config, "expected_single_image_count"),
        },
        "eval_contract": {
            **eval_contract.canonical_payload,
            "identity_sha256": eval_contract.identity_sha256,
        },
        "execution": {
            "world_size": len(getattr(config, "gpu_ids")),
            "gpu_ids": list(getattr(config, "gpu_ids")),
            "max_model_len": getattr(config, "max_model_len"),
            "max_num_batched_tokens": getattr(config, "max_num_batched_tokens"),
            "enable_chunked_prefill": getattr(config, "enable_chunked_prefill"),
            "inference_concurrency_per_gpu": getattr(
                config, "inference_concurrency_per_gpu"
            ),
        },
    }
    if (
        getattr(config, "evaluation_protocol")
        == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
    ):
        content["protocol"] = evaluation_protocol_identity(
            config,
            snapshot,
            is_lora_snapshot=is_lora_snapshot,
            step_zero_equivalence=step_zero_equivalence,
        )
    return {**content, "identity_sha256": canonical_json_sha256(content)}


_LEGACY_MODULE = "tgvf_rl.evaluation.policy_coredev"


# These contracts were historically defined by policy_coredev.  The facade
# re-exports the exact same objects, including their pickle/import coordinates.
rebind_public_class(
    PolicyEvalContract,
    implementation_module=__name__,
    public_module=_LEGACY_MODULE,
)
for _function, _legacy_name in (
    (canonical_json_sha256, "_canonical_json_sha256"),
    (policy_benchmark_task_path, "policy_benchmark_task_path"),
    (policy_eval_parser_identity, "_policy_eval_parser_identity"),
    (
        policy_eval_action_boundary_identity,
        "_policy_eval_action_boundary_identity",
    ),
    (policy_eval_observation_identity, "_policy_eval_observation_identity"),
    (
        effective_evaluation_image_max_pixels,
        "effective_evaluation_image_max_pixels",
    ),
    (build_policy_eval_contract, "build_policy_eval_contract"),
    (_decoding_contract, "_decoding_contract"),
    (_termination_contract, "_termination_contract"),
):
    rebind_public_function(
        _function,
        implementation_module=__name__,
        public_module=_LEGACY_MODULE,
        public_name=_legacy_name,
        public_qualname=_legacy_name,
    )
del _function, _legacy_name


__all__ = [
    "POLICY_EVAL_CONTRACT_SCHEMA",
    "POLICY_EVALUATION_IDENTITY_SCHEMA",
    "PolicyEvalContract",
    "build_policy_eval_contract",
    "build_policy_evaluation_identity",
    "canonical_json_sha256",
    "effective_evaluation_image_max_pixels",
    "evaluation_protocol_identity",
    "policy_benchmark_task_path",
    "policy_eval_action_boundary_identity",
    "policy_eval_observation_identity",
    "policy_eval_parser_identity",
    "sha256_file",
]
