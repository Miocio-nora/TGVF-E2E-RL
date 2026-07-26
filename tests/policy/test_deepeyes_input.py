from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.data import (
    DeepEyes47KRuntimeBinding,
    DeepEyesSourceFileSpec,
    load_deepeyes47k_runtime,
    materialize_deepeyes47k_fixture,
)
from tgvf_rl.policy import (
    POLICY_EXECUTION_GROUP_UID_PREFIX,
    POLICY_PILOT_V1_GROUP_SIZE,
    DeepEyesRewardContextProvider,
    build_qwen_policy_user_prompt,
    derive_policy_execution_group,
)
from tgvf_rl.protocol import (
    NATIVE_SHARED_USER_TEXT_TEMPLATE,
    TGVF_FOCUS_TOOL_NAME,
    TGVF_ONLY_SYSTEM_PROMPT,
    NativeToolCapabilityProfile,
    build_visual_tool_prompt_messages,
)
from tgvf_rl.rewards import AnswerTaskKind
from tgvf_rl.trajectories.schema import TrajectoryRecord, TrajectoryStop


FORBIDDEN_HISTORICAL_PROMPT = "FORBIDDEN DEEPEYES ZOOM PROMPT"
SHA0 = "0" * 64


def _materialized_sample(root: Path):
    source_payload = b"policy-input-fixture"
    source = DeepEyesSourceFileSpec(
        filename="fixture.parquet",
        rows=1,
        lfs_sha256=hashlib.sha256(source_payload).hexdigest(),
        byte_size=len(source_payload),
    )
    result = materialize_deepeyes47k_fixture(
        {
            source.filename: [
                {
                    "images": [{"bytes": b"fixture-image-bytes"}],
                    "prompt": FORBIDDEN_HISTORICAL_PROMPT,
                    "extra_info": {
                        "question": "Which option is shown?\nA. red\nB. blue",
                        "ability": "multiple_choice",
                        "split": "train",
                    },
                    "reward_model": {"ground_truth": "B", "style": "mcq"},
                    "data_source": "fixture_mcq",
                }
            ]
        },
        (source,),
        root,
        shuffle_seed=42,
    )
    binding = DeepEyes47KRuntimeBinding.fixture_binding(
        manifest_file_sha256=result.manifest_file_sha256,
        content_sha256=result.content_sha256,
        shuffle_seed=42,
        expected_sample_count=1,
    )
    dataset = load_deepeyes47k_runtime(root, binding=binding)
    return dataset[0]


def _trajectory(identity, *, answer: str) -> TrajectoryRecord:
    return TrajectoryRecord(
        schema_version="policy-input-test-v1",
        identity=identity,
        model=ModelIdentity(
            family="qwen3_vl",
            model_name="fixture",
            revision_or_path="fixture",
            tokenizer_length=1,
            chat_template_sha256=SHA0,
        ),
        behavior_policy=PolicyVersion(
            run_id=identity.run_id,
            optimizer_step=0,
            weights_sha256=SHA0,
        ),
        assistant_turns=(),
        tool_calls=(),
        observations=(),
        final_answer=answer,
        stop=TrajectoryStop.DIRECT_ANSWER,
    )


def test_prompt_group_and_reward_vertical_slice_is_prompt_free(tmp_path: Path) -> None:
    sample = _materialized_sample(tmp_path / "materialized")

    prompt = build_qwen_policy_user_prompt(sample)
    serialized_messages = json.dumps(prompt.messages, ensure_ascii=False)
    assert prompt.messages == (
        {"role": "system", "content": TGVF_ONLY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                {"type": "image"},
                {
                    "type": "text",
                    "text": NATIVE_SHARED_USER_TEXT_TEMPLATE.format(
                        question=sample.question
                    ),
                },
            ),
        },
    )
    assert FORBIDDEN_HISTORICAL_PROMPT not in serialized_messages
    assert prompt.image_path == sample.image_path
    assert prompt.image_sha256 == sample.image_sha256
    assert prompt.tool_names == (TGVF_FOCUS_TOOL_NAME,)
    assert prompt.tool_profile is NativeToolCapabilityProfile.TGVF_ONLY
    assert prompt.prompt_version == prompt.prompt_identity.version
    assert prompt.system_prompt_sha256 == prompt.prompt_identity.system_prompt_sha256
    assert (
        prompt.shared_user_prompt_template_sha256
        == prompt.prompt_identity.shared_user_prompt_template_sha256
    )
    assert prompt.prompt_bundle_sha256 == prompt.prompt_identity.bundle_sha256
    assert prompt.response_version == prompt.prompt_identity.response_version
    assert (
        prompt.success_response_template_sha256
        == prompt.prompt_identity.success_response_template_sha256
    )
    assert len(prompt.messages_sha256) == 64
    assert tuple(schema["function"]["name"] for schema in prompt.tool_schemas) == (
        TGVF_FOCUS_TOOL_NAME,
    )
    for profile in NativeToolCapabilityProfile:
        selected = build_qwen_policy_user_prompt(sample, tool_profile=profile)
        assert selected.messages == build_visual_tool_prompt_messages(
            sample.question,
            tool_profile=profile,
        )
        assert selected.tool_profile is profile
        assert selected.tool_names == profile.tool_names
        assert (
            tuple(schema["function"]["name"] for schema in selected.tool_schemas)
            == profile.tool_names
        )
        assert selected.tool_schema_sha256 == profile.tool_set_sha256

    group = derive_policy_execution_group(
        sample,
        run_id="pilot-run",
        optimizer_step=3,
        data_cursor=17,
        execution_nonce="attempt-001",
    )
    assert group.group_uid.startswith(POLICY_EXECUTION_GROUP_UID_PREFIX)
    assert len(group.trajectory_identities) == POLICY_PILOT_V1_GROUP_SIZE == 8
    assert tuple(
        identity.rollout_index for identity in group.trajectory_identities
    ) == tuple(range(8))
    assert len({identity.canonical_id for identity in group.trajectory_identities}) == 8
    assert all(
        identity.group_id == group.group_uid for identity in group.trajectory_identities
    )
    assert group.group_uid != sample.prompt_group_uid
    assert (
        derive_policy_execution_group(
            sample,
            run_id="pilot-run",
            optimizer_step=3,
            data_cursor=17,
            execution_nonce="attempt-002",
        ).group_uid
        != group.group_uid
    )

    provider = DeepEyesRewardContextProvider((sample,))
    identity = group.trajectory_identities[0]
    trajectory = _trajectory(identity, answer="B")
    context = provider.build(
        request=SimpleNamespace(identity=identity),
        trajectory=trajectory,
    )
    source = provider.source_for_sample(sample.sample_id)
    assert source.expected_answer == sample.ground_truth == "B"
    assert source.task_kind is AnswerTaskKind.MULTIPLE_CHOICE
    assert source.data_source == sample.data_source == "fixture_mcq"
    assert context.expected_answer == source.expected_answer
    assert context.task_kind is source.task_kind
    assert context.data_source == source.data_source
    assert context.question == sample.question


def test_execution_group_identity_binds_step_cursor_and_nonce(tmp_path: Path) -> None:
    sample = _materialized_sample(tmp_path / "materialized")
    baseline = derive_policy_execution_group(
        sample,
        run_id="pilot-run",
        optimizer_step=0,
        data_cursor=0,
        execution_nonce="nonce",
    )
    assert (
        derive_policy_execution_group(
            sample,
            run_id="pilot-run",
            optimizer_step=1,
            data_cursor=0,
            execution_nonce="nonce",
        ).group_uid
        != baseline.group_uid
    )
    assert (
        derive_policy_execution_group(
            sample,
            run_id="pilot-run",
            optimizer_step=0,
            data_cursor=1,
            execution_nonce="nonce",
        ).group_uid
        != baseline.group_uid
    )
    assert (
        derive_policy_execution_group(
            sample,
            run_id="other-run",
            optimizer_step=0,
            data_cursor=0,
            execution_nonce="nonce",
        ).group_uid
        != baseline.group_uid
    )
