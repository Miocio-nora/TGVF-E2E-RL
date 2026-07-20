from __future__ import annotations

from dataclasses import replace
import json

import pytest

from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.errors import ContractUnsetError
from tgvf_rl.contracts.identity import ArtifactIdentity, CodeIdentity, ModelIdentity
from tgvf_rl.data import (
    DEEPEYES47K_DATASET_ID,
    DEEPEYES47K_SNAPSHOT,
    DEEPEYES47K_TOTAL_ROWS,
)
from tgvf_rl.policy import (
    POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_JUDGE_MODEL_NAME,
    POLICY_PILOT_V1_MODEL_FAMILY,
    POLICY_PILOT_V1_MODEL_NAME,
    POLICY_PILOT_V1_MODEL_PATH,
    POLICY_PILOT_V1_TOKENIZER_LENGTH,
    PilotExecutionBindings,
    PilotJudgeBindings,
    PilotObjectiveBindings,
    PilotSamplingConfig,
    PolicyPilotV1Config,
    PolicyPilotV1RunManifest,
)
from tgvf_rl.protocol import TGVF_FOCUS_TOOL_SCHEMA_SHA256


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64


def _artifact(name: str, sha256: str = SHA0) -> ArtifactIdentity:
    return ArtifactIdentity("pilot-test", name, "v1", sha256)


def _sampling() -> PilotSamplingConfig:
    return PilotSamplingConfig().bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(151645,),
        stop_strings=("</tool_call>",),
        include_stop_str_in_output=False,
        ignore_eos=False,
    )


def _manifest() -> PolicyPilotV1RunManifest:
    return PolicyPilotV1RunManifest(
        run_id="pilot-001",
        policy=PolicyPilotV1Config(sampling=_sampling()),
        base_model=ModelIdentity(
            family=POLICY_PILOT_V1_MODEL_FAMILY,
            model_name=POLICY_PILOT_V1_MODEL_NAME,
            revision_or_path=POLICY_PILOT_V1_MODEL_PATH,
            tokenizer_length=POLICY_PILOT_V1_TOKENIZER_LENGTH,
            chat_template_sha256=POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
        ),
        processor=_artifact("qwen3-processor"),
        tokenizer_fixture=_artifact("tokenizer-golden", SHA1),
        chat_template_fixture=_artifact("chat-template-golden", SHA2),
        native_transcript_fixture=_artifact("native-transcript-golden"),
        prompt=_artifact("pilot-prompt", SHA1),
        cap_error_and_recovery_fixture=_artifact("cap-error-recovery"),
        tgvf_adapter=_artifact("representation-adapter", SHA2),
        target_conditioning=TargetConditioningConfig(
            provider=TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE,
            hidden_layer=-1,
        ),
        dataset_manifest=_artifact("deepeyes47k-manifest", SHA0),
        dataset_shuffle_seed=42,
        objectives=PilotObjectiveBindings(
            grpo=_artifact("grpo"),
            reward_pipeline=_artifact("reward"),
            answer_verifier=_artifact("answer-verifier"),
            format_verifier=_artifact("format-verifier"),
            conditional_tool_verifier=_artifact("conditional-tool-verifier"),
            diagnostic_kl_estimator=_artifact("diagnostic-kl"),
        ),
        judge=PilotJudgeBindings(
            model=_artifact(POLICY_PILOT_V1_JUDGE_MODEL_NAME),
            prompt=_artifact("judge-prompt"),
            service=_artifact("judge-service"),
            sampling=_artifact("judge-sampling"),
            calibration=_artifact("judge-calibration"),
            failure_policy=_artifact("judge-failure-policy"),
        ),
        execution=PilotExecutionBindings(
            code=CodeIdentity("Miocio-nora/TGVF", "abc123", SHA2),
            dependencies=_artifact("dependency-lock"),
            hardware_topology=_artifact("hardware-topology"),
            optimizer=_artifact("optimizer"),
            scheduler=_artifact("scheduler"),
            precision_batching=_artifact("precision-batching"),
            weight_sync=_artifact("weight-sync"),
            sampler_rng=_artifact("sampler-rng"),
        ),
        maximum_optimizer_steps=80,
        checkpoint_and_evaluation_steps=(0, 10, 20, 45, 80),
    )


def test_manifest_binds_fixed_pilot_and_all_open_run_identities() -> None:
    manifest = _manifest()
    record = manifest.as_record()

    assert record["dataset_id"] == DEEPEYES47K_DATASET_ID
    assert record["dataset_snapshot"] == DEEPEYES47K_SNAPSHOT
    assert record["dataset_rows"] == DEEPEYES47K_TOTAL_ROWS
    assert record["tool_schema_sha256"] == TGVF_FOCUS_TOOL_SCHEMA_SHA256
    assert record["policy"]["max_tgvf_call_attempts"] == 4
    assert record["policy"]["image_max_pixels"] == 512 * 512
    assert record["policy"]["sampling"]["trajectories_per_prompt"] == 8
    assert record["policy"]["lora"]["rank"] == 64
    assert record["reward_weights"] == [0.8, 0.2, 1.2]
    assert record["base_model"] == {
        "chat_template_sha256": POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
        "family": POLICY_PILOT_V1_MODEL_FAMILY,
        "model_name": POLICY_PILOT_V1_MODEL_NAME,
        "revision_or_path": POLICY_PILOT_V1_MODEL_PATH,
        "tokenizer_length": POLICY_PILOT_V1_TOKENIZER_LENGTH,
    }
    assert record["execution"]["hardware_topology"]["name"] == "hardware-topology"
    assert json.loads(manifest.canonical_json) == record
    assert len(manifest.identity_sha256) == 64

    hashes = manifest.checkpoint_run_identity()
    by_name = {item.name: item.sha256 for item in hashes.hashes}
    assert by_name["pilot_manifest"] == manifest.identity_sha256
    assert by_name["policy_config"] == manifest.policy.identity_sha256
    assert by_name["chat_template"] == POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256
    assert by_name["chat_template_fixture"] == SHA2
    assert by_name["data_manifest"] == SHA0
    assert by_name["prompt"] == SHA1
    assert by_name["tgvf_adapter"] == SHA2
    assert by_name["tokenizer_fixture"] == SHA1
    assert by_name["native_transcript_fixture"] == SHA0


def test_manifest_is_deterministic_and_every_binding_changes_aggregate_identity() -> None:
    original = _manifest()
    assert _manifest().canonical_json == original.canonical_json
    changed = replace(original, dataset_shuffle_seed=43)
    assert changed.identity_sha256 != original.identity_sha256
    changed = replace(
        original,
        execution=replace(
            original.execution,
            hardware_topology=_artifact("hardware-topology", SHA1),
        ),
    )
    assert changed.identity_sha256 != original.identity_sha256


def test_manifest_fails_closed_on_unbound_or_incompatible_inputs() -> None:
    manifest = _manifest()
    with pytest.raises(ContractUnsetError, match="min-p"):
        replace(
            manifest,
            policy=PolicyPilotV1Config(sampling=PilotSamplingConfig()),
        )
    with pytest.raises(ValueError, match="base-model path"):
        replace(
            manifest,
            base_model=replace(manifest.base_model, revision_or_path="other"),
        )
    with pytest.raises(ValueError, match="dataset_snapshot"):
        replace(manifest, dataset_snapshot="moving-main")
    with pytest.raises(ValueError, match="judge model"):
        replace(
            manifest,
            judge=replace(manifest.judge, model=_artifact("some-other-judge")),
        )
    with pytest.raises(TypeError, match="target_conditioning"):
        replace(manifest, target_conditioning="contextual_hidden_state")


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("family", "qwen3-vl", "base-model family"),
        ("model_name", "Qwen3-VL-8B-Instruct", "base-model name"),
        ("tokenizer_length", 151_936, "base-model tokenizer length"),
        ("chat_template_sha256", SHA1, "base-model chat-template SHA256"),
    ],
)
def test_manifest_rejects_drift_from_fixed_qwen3_identity(
    field: str, value: object, error: str
) -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match=error):
        replace(manifest, base_model=replace(manifest.base_model, **{field: value}))


def test_protocol_fixture_artifacts_remain_independent_run_bindings() -> None:
    manifest = _manifest()
    changed_chat_template_fixture = replace(
        manifest,
        chat_template_fixture=replace(manifest.chat_template_fixture, sha256=SHA1),
    )
    changed_tokenizer = replace(
        manifest,
        tokenizer_fixture=replace(manifest.tokenizer_fixture, sha256=SHA2),
    )
    changed_transcript = replace(
        manifest,
        native_transcript_fixture=replace(
            manifest.native_transcript_fixture,
            sha256=SHA2,
        ),
    )

    assert manifest.chat_template_fixture.sha256 != (
        manifest.base_model.chat_template_sha256
    )
    assert changed_chat_template_fixture.identity_sha256 != manifest.identity_sha256
    assert changed_tokenizer.identity_sha256 != manifest.identity_sha256
    assert changed_transcript.identity_sha256 != manifest.identity_sha256


@pytest.mark.parametrize(
    "steps, error",
    [
        ((10, 20), "start at step 0"),
        ((0, 20, 10), "strictly increasing"),
        ((0, 10, 81), "exceeds maximum_optimizer_steps"),
    ],
)
def test_checkpoint_schedule_is_an_explicit_ordered_run_binding(
    steps: tuple[int, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        replace(_manifest(), checkpoint_and_evaluation_steps=steps)
