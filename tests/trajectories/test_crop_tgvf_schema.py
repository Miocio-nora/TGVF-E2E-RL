from __future__ import annotations

from dataclasses import replace

import pytest

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
    TokenSpan,
)
from tgvf_rl.environment.crop_tgvf_tool import AtomicCropTGVFTool
from tgvf_rl.trajectories.schema import (
    AssistantTurnRecord,
    CropTGVFToolCallRecord,
    CropToolCallRecord,
    ToolObservationRecord,
    ToolCallRecord,
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)
from tgvf_rl.trajectories import BehaviorTraceStore, VLLMBehaviorRecorder
from tgvf_rl.trajectories.validation import TrajectoryValidator
from tgvf_rl.qwen.crop_coordinates import (
    CanonicalSourcePixelCropCoordinateMapper,
)
from tests.environment.test_crop_tgvf_tool import SHA0, TRAJECTORY_ID, _fixture


def _call() -> CropTGVFToolCallRecord:
    return CropTGVFToolCallRecord(
        call_index=0,
        assistant_turn_index=1,
        function_name="tgvf_crop_tool",
        bbox_2d=(1, 2, 9, 10),
        target="serial number",
        target_token_span=TokenSpan(5, 7),
        target_char_span=(42, 55),
        raw_call_text="raw",
    )


def test_atomic_crop_tgvf_call_is_not_a_sequential_tool_alias() -> None:
    call = _call()
    assert call.attempt_index == 0
    assert not isinstance(call, (CropToolCallRecord, ToolCallRecord))
    assert call.bbox_2d == (1, 2, 9, 10)
    assert call.target == "serial number"


def test_plain_crop_call_record_preserves_optional_label() -> None:
    call = CropToolCallRecord(
        call_index=0,
        assistant_turn_index=1,
        function_name="image_zoom_in_tool",
        bbox_2d=(1, 2, 9, 10),
        raw_call_text="raw",
        label="serial-number plate",
    )
    assert call.label == "serial-number plate"
    with pytest.raises(ValueError, match="label must be a string"):
        replace(call, label=3)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("function_name", "image_zoom_in_tool", "unexpected atomic"),
        ("bbox_2d", (1, 2, 1, 10), "bbox must be non-empty"),
        ("target", "  ", "target must be non-empty"),
        ("target_char_span", (7, 7), "char span must be non-empty"),
    ),
)
def test_atomic_crop_tgvf_call_fails_closed(field, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_call(), **{field: value})


def test_trajectory_validator_binds_atomic_bbox_target_and_observation() -> None:
    store, _, _, materializer, _, adapter, request = _fixture()
    observation = AtomicCropTGVFTool(
        materializer=materializer,
        adapter=adapter,
        store=store,
        coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
    ).execute(request)
    parsed = request.parsed_call
    tokens = OwnedTokenSequence(
        token_ids=parsed.sampled_token_ids,
        ownership=tuple(
            TokenOwnership.POLICY_SAMPLED for _ in parsed.sampled_token_ids
        ),
    )
    sampling = SamplingIdentity(
        policy_version=request.policy_version,
        backend="vllm",
        backend_version="fixture",
        seed=7,
        rng_state_sha256=SHA0,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    behavior_store = BehaviorTraceStore()
    trace = VLLMBehaviorRecorder(behavior_store).record(
        trajectory_id=TRAJECTORY_ID,
        assistant_turn_index=0,
        tokens=tokens,
        actual_sampled_logprobs=tuple(-0.1 for _ in parsed.sampled_token_ids),
        sampling=sampling,
        behavior_policy=request.policy_version,
        backend_request_sha256=SHA0,
        backend_response_sha256=SHA0,
    )
    turn = AssistantTurnRecord(
        turn_index=0,
        raw_text=parsed.sampled_text,
        tokens=tokens,
        behavior_trace=trace,
        think_span=None,
        is_tool_call=True,
    )
    call = CropTGVFToolCallRecord(
        call_index=0,
        assistant_turn_index=0,
        function_name="tgvf_crop_tool",
        bbox_2d=parsed.bbox_2d,
        target=parsed.target,
        target_token_span=TokenSpan(
            parsed.target_span.token_start,
            parsed.target_span.token_end,
        ),
        target_char_span=(
            parsed.target_span.offsets.char_start,
            parsed.target_span.offsets.char_end,
        ),
        raw_call_text=parsed.raw_tool_call,
    )
    trajectory = TrajectoryRecord(
        schema_version="trajectory-v1",
        identity=TrajectoryIdentity("run", "sample", 0, "group"),
        model=request.model,
        behavior_policy=request.policy_version,
        assistant_turns=(turn,),
        tool_calls=(call,),
        observations=(ToolObservationRecord(0, observation.handle, (1, 2)),),
        final_answer=None,
        stop=TrajectoryStop.MAX_TOKENS,
    )
    validator = TrajectoryValidator(store, behavior_store)
    validator.validate(trajectory)

    bad_call = replace(call, bbox_2d=(0, 0, 1, 1))
    with pytest.raises(IdentityMismatchError, match="bbox differs"):
        validator.validate(replace(trajectory, tool_calls=(bad_call,)))
