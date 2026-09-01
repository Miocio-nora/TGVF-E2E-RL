from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from hashlib import sha256
import pickle
import threading
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from tgvf_rl.framework.verl import native_agent_client as client_leaf
from tgvf_rl.framework.verl import native_agent_loop as native_agent_loop_facade
from tgvf_rl.contracts.errors import ContractUnsetError, ReplayMismatchError
from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.environment.agent_loop import FrameworkNeutralAgentLoop, RolloutRequest
from tgvf_rl.environment.native_appender import (
    NativeSuccessObservationContract,
    QwenNativeToolObservationAppender,
)
from tgvf_rl.framework.verl.native_agent_loop import (
    BoundVerlNativeAgentLoopInvocationFactory,
    VerlFrameworkNeutralAgentLoop,
    VerlNativeAgentLoopInvocation,
    VerlNativeTrajectoryComponents,
    _recover_termination,
)
from tgvf_rl.framework.vllm.live_client import VLLMLivePromptInputsPort
from tgvf_rl.framework.vllm import (
    ContentAddressedVLLMTurnRNG,
    LiveVLLMTurnContextRegistry,
    VLLMLivePromptInputs,
    VLLMOutputDecodingContract,
    VLLMPolicySampler,
    VLLMPolicyTurnRequest,
    VLLMPolicyTurnResponse,
    VLLMResolvedObservationPayload,
    VLLMTerminationOutcome,
    VLLMTurnRNGIdentity,
    VLLMTurnTerminationContract,
    bind_preexpanded_prompt_contract,
    split_preexpanded_prompt_contract,
)
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.policy.config import PilotSamplingConfig, PolicyMethodSamplingConfig
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.protocol.observation_contract import (
    NativeSuccessObservationProtocolId,
)
from tgvf_rl.protocol.schema import NativeToolCapabilityProfile
from tgvf_rl.trajectories.behavior import BehaviorTraceStore, VLLMBehaviorRecorder
from tgvf_rl.trajectories.schema import (
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)
from tests.support import populated_observation_store, trajectory_source_visual


SHA0 = "0" * 64
SHA1 = "1" * 64
POLICY = PolicyVersion("policy-pilot", 7, SHA0)
DECODING = VLLMOutputDecodingContract(True, False, False, "final_only")
TERMINATION = VLLMTurnTerminationContract(
    required_request_stop_strings=("</tool_call>",),
    required_request_stop_token_ids=(ord("!"),),
    include_stop_str_in_output=True,
    tool_call_terminal_suffixes=("",),
    tool_call_outcomes=(VLLMTerminationOutcome("stop", "</tool_call>"),),
    final_turn_outcomes=(
        VLLMTerminationOutcome("stop", ord("!")),
        VLLMTerminationOutcome("length", None),
    ),
)
_SOURCE_STORE, _SOURCE_HANDLE = populated_observation_store()
SOURCE_VISUAL = trajectory_source_visual(_SOURCE_STORE.resolve_record(_SOURCE_HANDLE))
IMAGE_TOKEN_ID = 9876
IMAGE_TOKEN = "<|image_pad|>"


def test_async_server_client_leaf_preserves_exact_facade_identity() -> None:
    public_client = native_agent_loop_facade.VerlAsyncServerPolicyTurnClient

    assert public_client is client_leaf.VerlAsyncServerPolicyTurnClient
    assert public_client.__module__ == native_agent_loop_facade.__name__
    assert public_client.__qualname__ == "VerlAsyncServerPolicyTurnClient"
    assert public_client.__init__.__module__ == native_agent_loop_facade.__name__
    assert public_client.generate.__module__ == native_agent_loop_facade.__name__
    assert public_client.generate_async.__module__ == native_agent_loop_facade.__name__
    assert pickle.loads(pickle.dumps(public_client)) is public_client
    assert get_type_hints(public_client.__init__)["prompt_inputs"] is (
        VLLMLivePromptInputsPort
    )
    assert get_type_hints(public_client.generate) == {
        "request": VLLMPolicyTurnRequest,
        "return": VLLMPolicyTurnResponse,
    }
    assert get_type_hints(public_client.generate_async) == {
        "request": VLLMPolicyTurnRequest,
        "return": VLLMPolicyTurnResponse,
    }

    for helper_name in (
        "_verl_server_sampling_parameters",
        "_token_ids",
        "_selected_processed_logprobs",
        "_validate_policy_step_evidence",
        "_recover_termination",
    ):
        public_helper = getattr(native_agent_loop_facade, helper_name)
        assert public_helper is getattr(client_leaf, helper_name)
        assert public_helper.__module__ == native_agent_loop_facade.__name__
        assert pickle.loads(pickle.dumps(public_helper)) is public_helper


class _CharacterTokenizer:
    is_fast = True
    all_special_ids = [IMAGE_TOKEN_ID]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        result: list[int] = []
        cursor = 0
        while True:
            start = text.find(IMAGE_TOKEN, cursor)
            if start < 0:
                result.extend(ord(character) for character in text[cursor:])
                return result
            result.extend(ord(character) for character in text[cursor:start])
            result.append(IMAGE_TOKEN_ID)
            cursor = start + len(IMAGE_TOKEN)

    @staticmethod
    def convert_tokens_to_ids(token):
        assert token == IMAGE_TOKEN
        return IMAGE_TOKEN_ID

    @staticmethod
    def get_added_vocab():
        return {
            **{chr(token_id): token_id for token_id in range(128)},
            IMAGE_TOKEN: IMAGE_TOKEN_ID,
        }

    @staticmethod
    def convert_ids_to_tokens(token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return [
            IMAGE_TOKEN if token_id == IMAGE_TOKEN_ID else chr(token_id)
            for token_id in token_ids
        ]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
        spaces_between_special_tokens,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        assert spaces_between_special_tokens is False
        return "".join(
            IMAGE_TOKEN if token_id == IMAGE_TOKEN_ID else chr(token_id)
            for token_id in token_ids
        )

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_offsets_mapping,
        truncation,
    ):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        assert truncation is False
        return {
            "input_ids": self.encode(text, add_special_tokens=False),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class _ObservationResolver:
    def __init__(self) -> None:
        self.items: list[object] = []

    def resolve(self, observation, *, call_index):
        item = {
            "image_embeds": object(),
            "image_grid_thw": object(),
            "deepstack_embeds": object(),
        }
        self.items.append(item)
        return VLLMResolvedObservationPayload(
            observation=observation,
            call_index=call_index,
            modality="image",
            multi_modal_data_item=item,
            payload_sha256=sha256(
                f"{observation.observation_id}:{call_index}".encode()
            ).hexdigest(),
            multi_modal_uuid=None,
        )

    @staticmethod
    def resolve_visual_token_count(observation):
        assert isinstance(observation, ObservationHandle)
        return 3


class _ToolRuntime:
    def __init__(self) -> None:
        self.contexts = []

    def execute(self, parsed_call, context):
        del parsed_call
        self.contexts.append(context)
        return ObservationHandle(
            f"observation-{context.call_index}",
            sha256(f"record-{context.call_index}".encode()).hexdigest(),
        )


class _FakeServerManager:
    outputs = (
        "inspect</think><tool_call>"
        '{"name":"tgvf_focus_tool","arguments":{"target":"red label"}}'
        "</tool_call>",
        "answer reasoning</think>blue!",
    )

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def generate(self, **kwargs):
        await asyncio.sleep(0)
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        text = self.outputs[index]
        token_ids = tuple(ord(character) for character in text)
        logprob = -0.125 if index == 0 else -0.75
        return SimpleNamespace(
            token_ids=token_ids,
            log_probs=tuple(logprob for _ in token_ids),
            stop_reason="completed",
            num_preempted=0,
            extra_fields={
                "min_global_steps": POLICY.optimizer_step,
                "max_global_steps": POLICY.optimizer_step,
                "logprobs_mode": "processed_logprobs",
            },
        )


class _ConcurrentFinalServerManager:
    """Require two native policy awaits to overlap on the owner event loop."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.both_started = asyncio.Event()

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 2:
            self.both_started.set()
        await self.both_started.wait()
        text = "answer reasoning</think>blue!"
        token_ids = tuple(ord(character) for character in text)
        return SimpleNamespace(
            token_ids=token_ids,
            log_probs=tuple(-0.25 for _ in token_ids),
            stop_reason="completed",
            num_preempted=0,
            extra_fields={
                "min_global_steps": POLICY.optimizer_step,
                "max_global_steps": POLICY.optimizer_step,
                "logprobs_mode": "processed_logprobs",
            },
        )


class _InvocationFactory:
    def __init__(self, tokenizer: _CharacterTokenizer) -> None:
        self.tokenizer = tokenizer
        self.behavior_store = BehaviorTraceStore()
        self.runtime = _ToolRuntime()
        self.resolver = _ObservationResolver()
        self.registry = LiveVLLMTurnContextRegistry(observation_resolver=self.resolver)
        self.initial_image_item = {"source_image": object()}
        self.initial_prompt = (10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 30)
        self.registry.register_initial_prompt(
            self.initial_prompt,
            VLLMLivePromptInputs(
                backend_prompt_payload_sha256=SHA1,
                multi_modal_data={"image": [self.initial_image_item]},
                mm_processor_kwargs=bind_preexpanded_prompt_contract(
                    {"do_resize": False},
                    prompt_token_ids=self.initial_prompt,
                    image_token_id=IMAGE_TOKEN_ID,
                    expected_image_items=1,
                ),
                multi_modal_uuids=None,
            ),
        )
        self.trajectory: TrajectoryRecord | None = None
        self.received_sampling_params = None
        self.received_sample_fields = None

    def build(self, *, sampling_params, sample_fields):
        self.received_sampling_params = sampling_params
        self.received_sample_fields = sample_fields
        request = RolloutRequest(
            schema_version="trajectory-v1",
            identity=TrajectoryIdentity("pilot", "sample-0", 0, "group-0"),
            model=ModelIdentity("qwen3_vl", "fixture", "/fixture", 151_669, SHA0),
            behavior_policy=POLICY,
            trajectory_source_visual=SOURCE_VISUAL,
            initial_prompt_token_ids=self.initial_prompt,
            sampling_parameters=_sampling_parameters(),
        )

        def native_loop_factory(sampler):
            return FrameworkNeutralAgentLoop(
                sampler=sampler,
                tool_runtime=self.runtime,
                appender=QwenNativeToolObservationAppender(
                    tokenizer=self.tokenizer,
                    registrar=self.registry,
                    observation_contract=NativeSuccessObservationContract(
                        protocol_id=(
                            NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1
                        ),
                        tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
                        assistant_dialect=(NativeAssistantDialect.QWEN3_VL_THINKING),
                    ),
                    visual_token_count_resolver=self.resolver,
                ),
                parser=StrictToolCallParser(),
                behavior_recorder=VLLMBehaviorRecorder(self.behavior_store),
                max_tool_calls=4,
            )

        def output_builder(trajectory):
            self.trajectory = trajectory
            response_ids: list[int] = []
            response_mask: list[int] = []
            response_logprobs: list[float] = []
            observations_by_turn = {
                call.assistant_turn_index: observation
                for call, observation in zip(
                    trajectory.tool_calls, trajectory.observations, strict=True
                )
            }
            for turn in trajectory.assistant_turns:
                record = self.behavior_store.resolve(turn.behavior_trace)
                response_ids.extend(turn.tokens.token_ids)
                response_mask.extend([1] * len(turn.tokens.token_ids))
                response_logprobs.extend(record.behavior.logprobs)
                observation = observations_by_turn.get(turn.turn_index)
                if observation is not None:
                    response_ids.extend(observation.template_token_ids)
                    response_mask.extend([0] * len(observation.template_token_ids))
                    response_logprobs.extend(
                        [0.0] * len(observation.template_token_ids)
                    )
            return SimpleNamespace(
                prompt_ids=list(request.initial_prompt_token_ids),
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                num_turns=len(trajectory.assistant_turns),
                metrics={},
                extra_fields={},
            )

        return VerlNativeAgentLoopInvocation(
            request=request,
            native_loop_factory=native_loop_factory,
            prompt_context=self.registry,
            rng=ContentAddressedVLLMTurnRNG(
                master_seed=42,
                stream_identity=request.identity.canonical_id,
            ),
            decoding=DECODING,
            termination=TERMINATION,
            sticky_request_id="sticky-sample-0",
            max_model_len=4096,
            output_builder=output_builder,
        )


class _HydraPartialFactory:
    def __init__(
        self,
        *,
        run_config_path,
        expected_run_identity_sha256,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        data_config,
    ) -> None:
        self.arguments = {
            "run_config_path": run_config_path,
            "expected_run_identity_sha256": expected_run_identity_sha256,
            "trainer_config": trainer_config,
            "server_manager": server_manager,
            "tokenizer": tokenizer,
            "processor": processor,
            "dataset_cls": dataset_cls,
            "data_config": data_config,
        }

    def build(self, *, sampling_params, sample_fields):
        del sampling_params, sample_fields
        raise AssertionError("constructor-only fixture must not build a rollout")


class DictConfigWrap:
    """Minimal fixture matching veRL's Hydra-protection wrapper."""

    def __init__(self, config):
        self.config = config


def test_agent_loop_completes_hydra_runtime_factory_partial() -> None:
    trainer_config = object()
    server_manager = object()
    tokenizer = _CharacterTokenizer()
    processor = object()
    dataset_cls = object
    data_config = object()
    factory = partial(
        _HydraPartialFactory,
        run_config_path="/fixture/policy.toml",
        expected_run_identity_sha256=SHA0,
    )

    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=trainer_config,
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=dataset_cls,
        data_config=data_config,
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
    )

    assert isinstance(bridge.invocation_factory, _HydraPartialFactory)
    assert bridge.invocation_factory.arguments == {
        "run_config_path": "/fixture/policy.toml",
        "expected_run_identity_sha256": SHA0,
        "trainer_config": trainer_config,
        "server_manager": server_manager,
        "tokenizer": tokenizer,
        "processor": processor,
        "dataset_cls": dataset_cls,
        "data_config": data_config,
    }


def test_agent_loop_unwraps_verl_dict_configs_before_runtime_factory() -> None:
    trainer_config = object()
    data_config = object()
    factory = partial(
        _HydraPartialFactory,
        run_config_path="/fixture/policy.toml",
        expected_run_identity_sha256=SHA0,
    )

    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=DictConfigWrap(trainer_config),
        server_manager=object(),
        tokenizer=_CharacterTokenizer(),
        processor=object(),
        dataset_cls=object,
        data_config=DictConfigWrap(data_config),
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
    )

    assert bridge.invocation_factory.arguments["trainer_config"] is trainer_config
    assert bridge.invocation_factory.arguments["data_config"] is data_config


def _sampling_parameters(**updates):
    values = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stop_token_ids": [ord("!")],
        "stop": ["</tool_call>"],
        "include_stop_str_in_output": True,
        "ignore_eos": False,
        "max_tokens": 1024,
        "logprobs": True,
    }
    values.update(updates)
    return values


def test_upstream_agent_loop_bridge_preserves_multiturn_inputs_and_logprobs() -> None:
    tokenizer = _CharacterTokenizer()
    factory = _InvocationFactory(tokenizer)
    server_manager = _FakeServerManager()
    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=object(),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=object(),
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
        server_timeout_seconds=5.0,
    )

    output = asyncio.run(bridge.run({"temperature": 1.0}, data_source="fixture"))

    assert factory.received_sampling_params == {"temperature": 1.0}
    assert factory.received_sample_fields == {"data_source": "fixture"}
    assert len(server_manager.calls) == 2
    first, second = server_manager.calls
    assert first["request_id"] == second["request_id"] == "sticky-sample-0"
    assert first["prompt_ids"] == list(factory.initial_prompt)
    assert second["prompt_ids"][: len(factory.initial_prompt)] == list(
        factory.initial_prompt
    )
    assert len(second["prompt_ids"]) > len(first["prompt_ids"])
    assert first["image_data"][0] is factory.initial_image_item
    assert second["image_data"][0] is factory.initial_image_item
    assert second["image_data"][1] is factory.resolver.items[0]
    first_contract, first_clean_kwargs = split_preexpanded_prompt_contract(
        first["mm_processor_kwargs"]
    )
    second_contract, second_clean_kwargs = split_preexpanded_prompt_contract(
        second["mm_processor_kwargs"]
    )
    assert first_clean_kwargs == second_clean_kwargs == {"do_resize": False}
    assert len(first_contract.ordered_visual_placeholder_ranges) == 1
    assert len(second_contract.ordered_visual_placeholder_ranges) == 2
    assert first["sampling_params"]["logprobs"] is True
    assert "output_kind" not in first["sampling_params"]
    assert first["sampling_params"]["stop"] == ["</tool_call>"]

    trajectory = factory.trajectory
    assert trajectory is not None
    assert trajectory.final_answer == "blue!"
    assert len(trajectory.assistant_turns) == 2
    assert len(trajectory.observations) == 1
    assert len(factory.runtime.contexts) == 1
    first_record = factory.behavior_store.resolve(
        trajectory.assistant_turns[0].behavior_trace
    )
    second_record = factory.behavior_store.resolve(
        trajectory.assistant_turns[1].behavior_trace
    )
    assert set(first_record.behavior.logprobs) == {-0.125}
    assert set(second_record.behavior.logprobs) == {-0.75}
    assert tuple(output.prompt_ids) == factory.initial_prompt
    assert tuple(output.response_ids) == (
        trajectory.assistant_turns[0].tokens.token_ids
        + trajectory.observations[0].template_token_ids
        + trajectory.assistant_turns[1].tokens.token_ids
    )
    environment_start = len(trajectory.assistant_turns[0].tokens.token_ids)
    environment_end = environment_start + len(
        trajectory.observations[0].template_token_ids
    )
    assert all(
        mask == 0 and logprob == 0.0
        for mask, logprob in zip(
            output.response_mask[environment_start:environment_end],
            output.response_logprobs[environment_start:environment_end],
            strict=True,
        )
    )


def test_async_bridge_never_invokes_whole_sync_trajectory_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _CharacterTokenizer()
    factory = _InvocationFactory(tokenizer)

    def reject_sync_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("async bridge must not invoke the whole sync trajectory")

    monkeypatch.setattr(FrameworkNeutralAgentLoop, "run", reject_sync_run)
    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=object(),
        server_manager=_FakeServerManager(),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=object(),
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
        server_timeout_seconds=5.0,
    )

    output = asyncio.run(bridge.run({}, data_source="fixture"))

    assert factory.trajectory is not None
    assert factory.trajectory.stop is TrajectoryStop.FINAL_ANSWER
    assert output.num_turns == 2


@pytest.mark.parametrize(
    ("target", "method_name"),
    (
        (
            native_agent_loop_facade.VerlAsyncServerPolicyTurnClient,
            "_response_from_output",
        ),
        (
            native_agent_loop_facade.VerlAsyncServerPolicyTurnClient,
            "_prepare_generation",
        ),
        (VLLMPolicySampler, "_build_request"),
        (VLLMPolicySampler, "_sampled_turn"),
    ),
)
def test_async_long_response_cpu_stages_do_not_block_owner_loop(
    monkeypatch: pytest.MonkeyPatch,
    target: type[object],
    method_name: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    worker_thread_ids: list[int] = []
    original = getattr(target, method_name)

    def blocking_first_call(*args: object, **kwargs: object) -> object:
        if not worker_thread_ids:
            worker_thread_ids.append(threading.get_ident())
            entered.set()
            if not release.wait(timeout=1.0):
                raise AssertionError(
                    f"{method_name} blocked the server-manager owner event loop"
                )
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method_name, blocking_first_call)
    tokenizer = _CharacterTokenizer()
    factory = _InvocationFactory(tokenizer)
    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=object(),
        server_manager=_FakeServerManager(),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=object(),
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
        server_timeout_seconds=5.0,
    )

    async def exercise() -> object:
        owner_thread_id = threading.get_ident()
        task = asyncio.create_task(bridge.run({}, data_source="fixture"))
        deadline = asyncio.get_running_loop().time() + 1.0
        while not entered.is_set():
            if task.done():
                return await task
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"{method_name} did not begin")
            await asyncio.sleep(0)
        assert threading.get_ident() == owner_thread_id
        release.set()
        output = await asyncio.wait_for(task, timeout=2.0)
        assert len(worker_thread_ids) == 1
        assert worker_thread_ids[0] != owner_thread_id
        return output

    output = asyncio.run(exercise())

    assert output.num_turns == 2


def test_async_server_manager_generate_stays_on_owner_loop() -> None:
    class LoopAffinityServerManager(_FakeServerManager):
        def __init__(self) -> None:
            super().__init__()
            self.thread_ids: list[int] = []
            self.event_loops: list[asyncio.AbstractEventLoop] = []

        async def generate(self, **kwargs):
            self.thread_ids.append(threading.get_ident())
            self.event_loops.append(asyncio.get_running_loop())
            return await super().generate(**kwargs)

    async def exercise() -> tuple[int, asyncio.AbstractEventLoop, object]:
        tokenizer = _CharacterTokenizer()
        server_manager = LoopAffinityServerManager()
        bridge = VerlFrameworkNeutralAgentLoop(
            trainer_config=object(),
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=object,
            data_config=object(),
            invocation_factory=_InvocationFactory(tokenizer),
            logprobs_mode="processed_logprobs",
            server_timeout_seconds=5.0,
        )
        owner_loop = asyncio.get_running_loop()
        owner_thread_id = threading.get_ident()
        await bridge.run({}, data_source="fixture")
        return owner_thread_id, owner_loop, server_manager

    owner_thread_id, owner_loop, server_manager = asyncio.run(exercise())

    assert server_manager.thread_ids == [owner_thread_id, owner_thread_id]
    assert server_manager.event_loops == [owner_loop, owner_loop]


def test_async_bridge_offloads_sync_invocation_factory_fallback() -> None:
    tokenizer = _CharacterTokenizer()
    factory = _InvocationFactory(tokenizer)
    original_build = factory.build
    entered = threading.Event()
    release = threading.Event()
    worker_thread_ids: list[int] = []

    def blocking_build(**kwargs):
        worker_thread_ids.append(threading.get_ident())
        entered.set()
        if not release.wait(timeout=1.0):
            raise AssertionError("sync invocation factory blocked the owner loop")
        return original_build(**kwargs)

    factory.build = blocking_build
    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=object(),
        server_manager=_FakeServerManager(),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=object(),
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
        server_timeout_seconds=5.0,
    )

    async def exercise() -> object:
        owner_thread_id = threading.get_ident()
        task = asyncio.create_task(bridge.run({}, data_source="fixture"))
        deadline = asyncio.get_running_loop().time() + 1.0
        while not entered.is_set():
            if task.done():
                return await task
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("sync invocation factory did not begin")
            await asyncio.sleep(0)
        assert threading.get_ident() == owner_thread_id
        release.set()
        output = await asyncio.wait_for(task, timeout=2.0)
        assert len(worker_thread_ids) == 1
        assert worker_thread_ids[0] != owner_thread_id
        return output

    output = asyncio.run(exercise())

    assert output.num_turns == 2


def test_async_policy_turns_overlap_with_one_sync_worker() -> None:
    async def exercise() -> tuple[object, object]:
        tokenizer = _CharacterTokenizer()
        server_manager = _ConcurrentFinalServerManager()
        factories = (_InvocationFactory(tokenizer), _InvocationFactory(tokenizer))
        bridges = tuple(
            VerlFrameworkNeutralAgentLoop(
                trainer_config=object(),
                server_manager=server_manager,
                tokenizer=tokenizer,
                processor=None,
                dataset_cls=object,
                data_config=object(),
                invocation_factory=factory,
                logprobs_mode="processed_logprobs",
                server_timeout_seconds=5.0,
            )
            for factory in factories
        )
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        outputs = await asyncio.wait_for(
            asyncio.gather(
                *(bridge.run({}, data_source="fixture") for bridge in bridges)
            ),
            timeout=2.0,
        )
        assert len(server_manager.calls) == 2
        assert all(factory.trajectory is not None for factory in factories)
        return outputs

    first, second = asyncio.run(exercise())

    assert first.num_turns == second.num_turns == 1


def test_async_cancellation_drains_tool_segment_before_sticky_release() -> None:
    class BlockingFailingToolRuntime:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.unblock = threading.Event()
            self.finished = threading.Event()

        def execute(self, parsed_call, context):
            del parsed_call, context
            self.entered.set()
            try:
                if not self.unblock.wait(timeout=2.0):
                    raise TimeoutError("test tool worker was not unblocked")
                raise RuntimeError("worker fault after caller cancellation")
            finally:
                self.finished.set()

    class ReleaseTrackingServerManager(_FakeServerManager):
        def __init__(self, runtime: BlockingFailingToolRuntime) -> None:
            super().__init__()
            self.runtime = runtime
            self.release_seen = asyncio.Event()
            self.released_before_tool_finished: bool | None = None

        async def release_trajectory(self, request_id: str) -> None:
            assert request_id == "sticky-sample-0"
            self.released_before_tool_finished = not self.runtime.finished.is_set()
            self.release_seen.set()

    async def exercise() -> tuple[bool, bool | None]:
        tokenizer = _CharacterTokenizer()
        factory = _InvocationFactory(tokenizer)
        runtime = BlockingFailingToolRuntime()
        factory.runtime = runtime
        server_manager = ReleaseTrackingServerManager(runtime)
        bridge = VerlFrameworkNeutralAgentLoop(
            trainer_config=object(),
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=object,
            data_config=object(),
            invocation_factory=factory,
            logprobs_mode="processed_logprobs",
            server_timeout_seconds=5.0,
        )
        task = asyncio.create_task(bridge.run({}, data_source="fixture"))
        deadline = asyncio.get_running_loop().time() + 1.0
        while not runtime.entered.is_set():
            if task.done():
                await task
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("tool worker did not begin")
            await asyncio.sleep(0)

        task.cancel()
        await asyncio.sleep(0)
        released_while_blocked = server_manager.release_seen.is_set()
        runtime.unblock.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime.finished.is_set()
        assert server_manager.release_seen.is_set()
        return released_while_blocked, server_manager.released_before_tool_finished

    released_while_blocked, released_before_tool_finished = asyncio.run(exercise())

    assert released_while_blocked is False
    assert released_before_tool_finished is False


def test_async_cancellation_remains_prompt_during_server_sampling() -> None:
    class CancellableServerManager:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.generation_cancelled = asyncio.Event()
            self.release_seen = asyncio.Event()

        async def generate(self, **kwargs):
            del kwargs
            self.started.set()
            try:
                await asyncio.Future()
            finally:
                self.generation_cancelled.set()

        async def release_trajectory(self, request_id: str) -> None:
            assert request_id == "sticky-sample-0"
            self.release_seen.set()

    async def exercise() -> tuple[bool, bool]:
        tokenizer = _CharacterTokenizer()
        server_manager = CancellableServerManager()
        bridge = VerlFrameworkNeutralAgentLoop(
            trainer_config=object(),
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=object,
            data_config=object(),
            invocation_factory=_InvocationFactory(tokenizer),
            logprobs_mode="processed_logprobs",
            server_timeout_seconds=5.0,
        )
        task = asyncio.create_task(bridge.run({}, data_source="fixture"))
        await asyncio.wait_for(server_manager.started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)
        return (
            server_manager.generation_cancelled.is_set(),
            server_manager.release_seen.is_set(),
        )

    generation_cancelled, release_seen = asyncio.run(exercise())

    assert generation_cancelled is True
    assert release_seen is True


def test_async_bridge_prefers_async_output_builder() -> None:
    class AsyncOutputBuilder:
        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.called = False

        def __call__(self, trajectory):
            del trajectory
            raise AssertionError("sync output builder must not be selected")

        async def build_async(self, trajectory):
            await asyncio.sleep(0)
            self.called = True
            return self.delegate(trajectory)

    class AsyncOutputInvocationFactory(_InvocationFactory):
        def __init__(self, tokenizer: _CharacterTokenizer) -> None:
            super().__init__(tokenizer)
            self.async_builder = None

        def build(self, *, sampling_params, sample_fields):
            invocation = super().build(
                sampling_params=sampling_params,
                sample_fields=sample_fields,
            )
            self.async_builder = AsyncOutputBuilder(invocation.output_builder)
            return replace(invocation, output_builder=self.async_builder)

    tokenizer = _CharacterTokenizer()
    factory = AsyncOutputInvocationFactory(tokenizer)
    server_manager = _ConcurrentFinalServerManager()
    server_manager.both_started.set()
    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=object(),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=object(),
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
        server_timeout_seconds=5.0,
    )

    output = asyncio.run(bridge.run({}, data_source="fixture"))

    assert factory.async_builder is not None
    assert factory.async_builder.called is True
    assert output.num_turns == 1


def test_agent_loop_keeps_public_client_facade_hook_late_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _CharacterTokenizer()
    factory = _InvocationFactory(tokenizer)
    original_client = native_agent_loop_facade.VerlAsyncServerPolicyTurnClient
    client_calls: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> object:
        client_calls.append(dict(kwargs))
        return original_client(**kwargs)

    monkeypatch.setattr(
        native_agent_loop_facade,
        "VerlAsyncServerPolicyTurnClient",
        client_factory,
    )
    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=object(),
        server_manager=_FakeServerManager(),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=object(),
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
        server_timeout_seconds=5.0,
    )

    asyncio.run(bridge.run({"temperature": 1.0}, data_source="fixture"))

    assert len(client_calls) == 1


def test_upstream_length_truncation_is_retained_as_invalid_trajectory() -> None:
    tokenizer = _CharacterTokenizer()
    factory = _InvocationFactory(tokenizer)
    server_manager = _FakeServerManager()
    server_manager.outputs = ("x" * 1024,)
    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=object(),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=object(),
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
        server_timeout_seconds=5.0,
    )

    output = asyncio.run(bridge.run({}, data_source="fixture"))

    trajectory = factory.trajectory
    assert trajectory is not None
    assert trajectory.stop is TrajectoryStop.MAX_TOKENS
    assert trajectory.final_answer is None
    assert len(trajectory.assistant_turns) == 1
    assert trajectory.assistant_turns[0].think_span is None
    assert len(output.response_ids) == 1024
    assert output.response_mask == [1] * 1024
    assert output.response_logprobs == [-0.125] * 1024


def test_upstream_malformed_tool_turn_recovers_without_executing_tool() -> None:
    tokenizer = _CharacterTokenizer()
    factory = _InvocationFactory(tokenizer)
    server_manager = _FakeServerManager()
    server_manager.outputs = (
        '<tool_call>{"name":"tgvf_focus_tool","arguments":'
        '{"target":"red label"}}</tool_call>',
        "answer reasoning</think>blue!",
    )
    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=object(),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=object(),
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
        server_timeout_seconds=5.0,
    )

    output = asyncio.run(bridge.run({}, data_source="fixture"))

    trajectory = factory.trajectory
    assert trajectory is not None
    assert trajectory.stop is TrajectoryStop.FINAL_ANSWER
    assert trajectory.final_answer == "blue!"
    assert len(trajectory.assistant_turns) == 2
    assert trajectory.tool_calls == ()
    assert trajectory.observations == ()
    assert len(trajectory.tool_errors) == 1
    assert trajectory.tool_errors[0].code == "tool_parse.missing_think_closer"
    assert factory.runtime.contexts == []
    assert len(output.response_ids) == len(output.response_mask)
    assert all(
        mask == 1 for mask in output.response_mask[: len(server_manager.outputs[0])]
    )


def _turn_request(text: str, *, max_tokens: int) -> VLLMPolicyTurnRequest:
    del text
    return VLLMPolicyTurnRequest(
        request_id="request-0",
        prompt_token_ids=(1, 2),
        sampling_parameters={
            **_sampling_parameters(max_tokens=max_tokens),
            "seed": 42,
            "n": 1,
            "min_tokens": 0,
            "prompt_logprobs": None,
            "flat_logprobs": False,
            "detokenize": True,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "output_kind": "final_only",
            "logits_processors": None,
        },
        turn_index=0,
        behavior_policy=POLICY,
        rng=VLLMTurnRNGIdentity(42, SHA0),
        backend_prompt_payload_sha256=SHA1,
        backend_version="0.12.0",
        logprobs_mode="processed_logprobs",
        decoding=DECODING,
        termination_contract_sha256=TERMINATION.sha256,
    )


def test_collapsed_upstream_stop_takes_precedence_at_exact_length_boundary() -> None:
    text = "reason</think><tool_call>{}</tool_call>"
    token_ids = tuple(ord(character) for character in text)
    request = _turn_request(text, max_tokens=len(token_ids))

    assert _recover_termination(
        request=request,
        token_ids=token_ids,
        text=text,
        upstream_stop_reason="completed",
    ) == ("stop", "</tool_call>")

    non_ambiguous = _turn_request(text, max_tokens=len(token_ids) + 1)
    assert _recover_termination(
        request=non_ambiguous,
        token_ids=token_ids,
        text=text,
        upstream_stop_reason="completed",
    ) == ("stop", "</tool_call>")

    text_without_terminal_evidence = "reason</think>answer"
    ids_without_terminal_evidence = tuple(
        ord(character) for character in text_without_terminal_evidence
    )
    with pytest.raises(ReplayMismatchError, match="no recoverable"):
        _recover_termination(
            request=_turn_request(
                text_without_terminal_evidence,
                max_tokens=len(ids_without_terminal_evidence) + 1,
            ),
            token_ids=ids_without_terminal_evidence,
            text=text_without_terminal_evidence,
            upstream_stop_reason="completed",
        )


def test_exact_standalone_vllm_stop_survives_collapsed_verl_output() -> None:
    text = "reason</think>answer"
    token_ids = tuple(ord(character) for character in text)

    assert _recover_termination(
        request=_turn_request(text, max_tokens=len(token_ids) + 1),
        token_ids=token_ids,
        text=text,
        upstream_stop_reason="completed",
        exact_finish_reason="stop",
        exact_stop_reason=151645,
    ) == ("stop", 151645)


def test_bridge_rejects_upstream_silent_max_token_clamp() -> None:
    tokenizer = _CharacterTokenizer()
    factory = _InvocationFactory(tokenizer)
    original_build = factory.build

    def build(**kwargs):
        invocation = original_build(**kwargs)
        return VerlNativeAgentLoopInvocation(
            request=invocation.request,
            native_loop_factory=invocation.native_loop_factory,
            prompt_context=invocation.prompt_context,
            rng=invocation.rng,
            decoding=invocation.decoding,
            termination=invocation.termination,
            sticky_request_id=invocation.sticky_request_id,
            max_model_len=100,
            output_builder=invocation.output_builder,
        )

    factory.build = build
    bridge = VerlFrameworkNeutralAgentLoop(
        trainer_config=object(),
        server_manager=_FakeServerManager(),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=object(),
        invocation_factory=factory,
        logprobs_mode="processed_logprobs",
    )

    with pytest.raises(ContractUnsetError, match="silent max_tokens clamp"):
        asyncio.run(bridge.run({}))


class _CurrentPolicy:
    def current_policy_version(self):
        return POLICY


class _TrajectoryComponents:
    def __init__(self) -> None:
        self.registry = LiveVLLMTurnContextRegistry(
            observation_resolver=_ObservationResolver()
        )

    def build_trajectory_components(self, **kwargs):
        del kwargs
        return VerlNativeTrajectoryComponents(
            source_visual=SOURCE_VISUAL,
            native_loop_factory=lambda sampler: None,
            prompt_context=self.registry,
            output_builder=lambda trajectory: trajectory,
        )


class _AsyncTrajectoryComponents:
    def __init__(self) -> None:
        self.registry = LiveVLLMTurnContextRegistry(
            observation_resolver=_ObservationResolver()
        )
        self.calls: list[dict[str, object]] = []

    async def build_trajectory_components_async(self, **kwargs):
        self.calls.append(kwargs)
        return VerlNativeTrajectoryComponents(
            source_visual=SOURCE_VISUAL,
            native_loop_factory=lambda sampler: None,
            prompt_context=self.registry,
            output_builder=lambda trajectory: trajectory,
        )


def test_bound_invocation_factory_accepts_async_only_trajectory_builder() -> None:
    sampling = PilotSamplingConfig().bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(ord("!"),),
        stop_strings=("</tool_call>",),
        include_stop_str_in_output=True,
        ignore_eos=False,
    )
    components = _AsyncTrajectoryComponents()
    factory = BoundVerlNativeAgentLoopInvocationFactory(
        run_id=POLICY.run_id,
        model=ModelIdentity("qwen3_vl", "fixture", "/fixture", 151_669, SHA0),
        sampling_contract=sampling,
        policy_version=_CurrentPolicy(),
        trajectory_components=components,
        decoding=DECODING,
        termination=TERMINATION,
        rollout_master_seed=42,
        max_model_len=16_384,
    )

    invocation = asyncio.run(
        factory.build_async(
            sampling_params={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
                "logprobs": True,
            },
            sample_fields={
                "sample_id": "sample-async",
                "uid": "upstream-group-async",
                "index": 0,
                "initial_prompt_token_ids": (10, 20, 30),
            },
        )
    )

    assert invocation.request.trajectory_source_visual == SOURCE_VISUAL
    assert invocation.request.behavior_policy == POLICY
    assert invocation.request.identity.rollout_index == 0
    assert len(components.calls) == 1


def test_bound_invocation_factory_offloads_sync_components_fallback() -> None:
    class BlockingTrajectoryComponents(_TrajectoryComponents):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.worker_thread_ids: list[int] = []

        def build_trajectory_components(self, **kwargs):
            self.worker_thread_ids.append(threading.get_ident())
            self.entered.set()
            if not self.release.wait(timeout=1.0):
                raise AssertionError("sync components builder blocked the owner loop")
            return super().build_trajectory_components(**kwargs)

    sampling = PilotSamplingConfig().bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(ord("!"),),
        stop_strings=("</tool_call>",),
        include_stop_str_in_output=True,
        ignore_eos=False,
    )
    components = BlockingTrajectoryComponents()
    factory = BoundVerlNativeAgentLoopInvocationFactory(
        run_id=POLICY.run_id,
        model=ModelIdentity("qwen3_vl", "fixture", "/fixture", 151_669, SHA0),
        sampling_contract=sampling,
        policy_version=_CurrentPolicy(),
        trajectory_components=components,
        decoding=DECODING,
        termination=TERMINATION,
        rollout_master_seed=42,
        max_model_len=16_384,
    )

    async def exercise() -> object:
        owner_thread_id = threading.get_ident()
        task = asyncio.create_task(
            factory.build_async(
                sampling_params={
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "repetition_penalty": 1.0,
                    "logprobs": True,
                },
                sample_fields={
                    "sample_id": "sample-sync-fallback",
                    "uid": "upstream-group-sync-fallback",
                    "index": 0,
                    "initial_prompt_token_ids": (10, 20, 30),
                },
            )
        )
        deadline = asyncio.get_running_loop().time() + 1.0
        while not components.entered.is_set():
            if task.done():
                return await task
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("sync components builder did not begin")
            await asyncio.sleep(0)
        assert threading.get_ident() == owner_thread_id
        components.release.set()
        invocation = await asyncio.wait_for(task, timeout=2.0)
        assert len(components.worker_thread_ids) == 1
        assert components.worker_thread_ids[0] != owner_thread_id
        return invocation

    invocation = asyncio.run(exercise())

    assert invocation.request.identity.rollout_index == 0


def test_bound_invocation_factory_assigns_exactly_eight_group_indices() -> None:
    sampling = PilotSamplingConfig().bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(ord("!"),),
        stop_strings=("</tool_call>",),
        include_stop_str_in_output=True,
        ignore_eos=False,
    )
    factory = BoundVerlNativeAgentLoopInvocationFactory(
        run_id=POLICY.run_id,
        model=ModelIdentity("qwen3_vl", "fixture", "/fixture", 151_669, SHA0),
        sampling_contract=sampling,
        policy_version=_CurrentPolicy(),
        trajectory_components=_TrajectoryComponents(),
        decoding=DECODING,
        termination=TERMINATION,
        rollout_master_seed=42,
        max_model_len=16_384,
    )
    upstream_sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "logprobs": True,
    }
    sample = {
        "sample_id": "sample-0",
        "uid": "upstream-group-0",
        "index": 0,
        "initial_prompt_token_ids": (10, 20, 30),
    }

    invocations = tuple(
        factory.build(sampling_params=upstream_sampling, sample_fields=sample)
        for _ in range(8)
    )

    assert tuple(item.request.identity.rollout_index for item in invocations) == tuple(
        range(8)
    )
    assert len({item.request.identity.group_id for item in invocations}) == 1
    assert all(item.request.behavior_policy == POLICY for item in invocations)
    with pytest.raises(ReplayMismatchError, match="reused a completed n=8 group"):
        factory.build(sampling_params=upstream_sampling, sample_fields=sample)


def test_bound_invocation_factory_assigns_deepeyes_n16_group_indices() -> None:
    sampling = PilotSamplingConfig(
        trajectories_per_prompt=16,
        max_response_length=20480,
    ).bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(ord("!"),),
        stop_strings=("</tool_call>",),
        include_stop_str_in_output=True,
        ignore_eos=False,
    )
    factory = BoundVerlNativeAgentLoopInvocationFactory(
        run_id=POLICY.run_id,
        model=ModelIdentity("qwen3_vl", "fixture", "/fixture", 151_669, SHA0),
        sampling_contract=sampling,
        policy_version=_CurrentPolicy(),
        trajectory_components=_TrajectoryComponents(),
        decoding=DECODING,
        termination=TERMINATION,
        rollout_master_seed=42,
        max_model_len=32_768,
        rollouts_per_prompt=16,
    )
    upstream_sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "logprobs": True,
    }
    sample = {
        "sample_id": "sample-n16",
        "uid": "upstream-group-n16",
        "index": 0,
        "initial_prompt_token_ids": (10, 20, 30),
    }

    invocations = tuple(
        factory.build(sampling_params=upstream_sampling, sample_fields=sample)
        for _ in range(16)
    )
    assert tuple(item.request.identity.rollout_index for item in invocations) == tuple(
        range(16)
    )
    with pytest.raises(ReplayMismatchError, match="completed n=16 group"):
        factory.build(sampling_params=upstream_sampling, sample_fields=sample)


def test_bound_invocation_factory_accepts_config_owned_n3_group() -> None:
    sampling = PolicyMethodSamplingConfig(
        trajectories_per_prompt=3,
        max_response_length=1_234,
    ).bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(ord("!"),),
        stop_strings=("</tool_call>",),
        include_stop_str_in_output=True,
        ignore_eos=False,
    )
    factory = BoundVerlNativeAgentLoopInvocationFactory(
        run_id=POLICY.run_id,
        model=ModelIdentity("qwen3_vl", "fixture", "/fixture", 151_669, SHA0),
        sampling_contract=sampling,
        policy_version=_CurrentPolicy(),
        trajectory_components=_TrajectoryComponents(),
        decoding=DECODING,
        termination=TERMINATION,
        rollout_master_seed=42,
        max_model_len=4_096,
        rollouts_per_prompt=3,
    )
    upstream_sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "logprobs": True,
    }
    sample = {
        "sample_id": "sample-n3",
        "uid": "upstream-group-n3",
        "index": 0,
        "initial_prompt_token_ids": (10, 20, 30),
    }

    invocations = tuple(
        factory.build(sampling_params=upstream_sampling, sample_fields=sample)
        for _ in range(3)
    )

    assert tuple(item.request.identity.rollout_index for item in invocations) == (
        0,
        1,
        2,
    )
    with pytest.raises(ReplayMismatchError, match="completed n=3 group"):
        factory.build(sampling_params=upstream_sampling, sample_fields=sample)


def test_bound_invocation_factory_keeps_legacy_group_size_allowlist() -> None:
    sampling = PilotSamplingConfig().bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(ord("!"),),
        stop_strings=("</tool_call>",),
        include_stop_str_in_output=True,
        ignore_eos=False,
    )

    with pytest.raises(ValueError, match="group size is unsupported"):
        BoundVerlNativeAgentLoopInvocationFactory(
            run_id=POLICY.run_id,
            model=ModelIdentity("qwen3_vl", "fixture", "/fixture", 151_669, SHA0),
            sampling_contract=sampling,
            policy_version=_CurrentPolicy(),
            trajectory_components=_TrajectoryComponents(),
            decoding=DECODING,
            termination=TERMINATION,
            rollout_master_seed=42,
            max_model_len=4_096,
            rollouts_per_prompt=3,
        )
