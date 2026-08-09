from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningDependencies,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import (
    ArtifactIdentity,
    CodeIdentity,
    ModelIdentity,
    PolicyVersion,
)
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.adapter_runtime import (
    BehaviorHiddenStateMaterialization,
    BranchMergerRuntimeBinding,
    RepresentationArtifactRuntimeBinding,
    build_policy_pilot_focus_runtime,
    load_frozen_tgvf_adapter,
)
from tgvf_rl.environment.agent_loop import SampledPolicyTurn, ToolExecutionContext
from tgvf_rl.environment.focus_runtime import FocusExecutionLedger
from tgvf_rl.environment.focus_tool import (
    ReplayLayoutTensors,
    SourceVisualTensorBundle,
)
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.observations.schema import VisualLayout
from tgvf_rl.observations.store import ObservationStore, tensor_checksum
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import TokenByteSpan
from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter
from tgvf_rl.representation.training.checkpoint import (
    RepresentationAccumulationIdentity,
    RepresentationAdapterContractIdentity,
    RepresentationInitializationIdentity,
    RepresentationOptimizerIdentity,
    RepresentationRunIdentity,
    RepresentationSamplerContractIdentity,
    RepresentationTrainerExecutionIdentity,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    RankZeroAdapterOwnedStateExport,
    RankZeroAdapterOwnedStateManifest,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveKind,
)
from tgvf_rl.trajectories.schema import TrajectoryIdentity


# Synthetic CPU contract fixtures only; these are not live-model hashes or parity data.
SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
SHA3 = "3" * 64
BRANCH_LAYERS = (8, 16, 24)


class _Merger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(16, 8, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens.reshape(-1, 16))


def _projection(identity: str) -> FrozenProjectionPort:
    return FrozenProjectionPort(
        _Merger(),
        identity=identity,
        input_dim=4,
        output_dim=8,
        spatial_merge_size=2,
    )


def _adapter(seed: int, *, identity_prefix: str = "qwen") -> TGVFAdapter:
    torch.manual_seed(seed)
    return TGVFAdapter(
        d_lm=8,
        d_v=4,
        attn_dim=6,
        main_projection=_projection(f"{identity_prefix}.main@fixture"),
        deepstack_projections=tuple(
            _projection(f"{identity_prefix}.branch.{layer}@fixture")
            for layer in BRANCH_LAYERS
        ),
        branch_layers=BRANCH_LAYERS,
    )


def _model(*, family: str = "qwen3_vl") -> ModelIdentity:
    return ModelIdentity(
        family=family,
        model_name=(
            "Qwen3-VL-8B-Thinking"
            if family == "qwen3_vl"
            else "Qwen/Qwen2.5-VL-7B-Instruct"
        ),
        revision_or_path=f"/{family}/fixture",
        tokenizer_length=256,
        chat_template_sha256=SHA0,
    )


def _conditioning(provider: str) -> TargetConditioningConfig:
    if provider == "contextual_hidden_state":
        return TargetConditioningConfig(
            TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE,
            hidden_layer=-1,
        )
    return TargetConditioningConfig(
        TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING,
        embedding_identity="language_model.input_embeddings@fixture",
    )


def _run_identity(
    adapter: TGVFAdapter,
    *,
    model: ModelIdentity,
    conditioning: TargetConditioningConfig,
) -> RepresentationRunIdentity:
    optimizer = torch.optim.AdamW(
        tuple(
            parameter for parameter in adapter.parameters() if parameter.requires_grad
        ),
        lr=1e-4,
    )
    return RepresentationRunIdentity(
        run_id="representation-runtime-fixture",
        code=CodeIdentity("Miocio-nora/TGVF-E2E-RL", "fixture-commit"),
        model=model,
        provider=conditioning,
        data_manifest_sha256=SHA1,
        prompt_sha256=SHA2,
        objective=RepresentationObjectiveConfig(
            identity="runtime-fixture-objective",
            kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
            matrix_ce_weight=1.0,
            l_gen_weight=1.0,
        ),
        adapter_contract=RepresentationAdapterContractIdentity.from_adapter(adapter),
        accumulation=RepresentationAccumulationIdentity(1, 1),
        optimizer=RepresentationOptimizerIdentity.from_optimizer(optimizer),
        scheduler=None,
        trainer_execution=RepresentationTrainerExecutionIdentity(
            precision="fp32",
            max_grad_norm=1.0,
            require_all_adapter_gradients=True,
        ),
        initialization=RepresentationInitializationIdentity.from_adapter(
            adapter,
            kind="fresh_random",
            seed=42,
            source_artifact_sha256=None,
        ),
        sampler_contract=RepresentationSamplerContractIdentity(
            batch_size=4,
            seed=42,
            world_size=1,
            data_manifest_sha256=SHA1,
        ),
    )


def _write_artifact(
    path: Path,
    *,
    provider: str,
) -> tuple[RepresentationArtifactRuntimeBinding, TGVFAdapter]:
    source = _adapter(7)
    model = _model()
    conditioning = _conditioning(provider)
    run_identity = _run_identity(source, model=model, conditioning=conditioning)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in source.artifact_state_dict().items()
    }
    names = tuple(sorted(state))
    manifest = RankZeroAdapterOwnedStateManifest(
        run_identity=run_identity,
        run_identity_sha256=run_identity.identity_sha256,
        global_step=20,
        tensor_names=names,
        tensor_shapes=tuple(tuple(state[name].shape) for name in names),
        tensor_dtypes=tuple(str(state[name].dtype) for name in names),
        tensor_sha256=tuple(tensor_checksum(state[name]) for name in names),
    )
    torch.save(RankZeroAdapterOwnedStateExport(manifest, state), path)
    artifact = ArtifactIdentity(
        "tgvf",
        "native-representation-adapter",
        "fixture",
        state_digest(manifest),
    )
    binding = RepresentationArtifactRuntimeBinding(
        artifact_path=path,
        artifact=artifact,
        expected_run_id=run_identity.run_id,
        expected_run_identity_sha256=run_identity.identity_sha256,
        model=model,
        conditioning=conditioning,
        adapter_contract=run_identity.adapter_contract,
    )
    return binding, _adapter(99)


class _HiddenStateDependency:
    def __init__(self, forward_identity: ArtifactIdentity) -> None:
        self.forward_identity = forward_identity
        self.policy_override: PolicyVersion | None = None
        self.forward_override: ArtifactIdentity | None = None
        self.calls = 0

    def capture_hidden_states(self, request):
        self.calls += 1
        length = int(request.input_ids.shape[0])
        hidden = torch.arange(length * 8, dtype=torch.float32).reshape(length, 8)
        hidden.requires_grad_(True)
        return BehaviorHiddenStateMaterialization(
            policy_version=self.policy_override
            or request.call.identity.behavior_policy,
            forward_identity=self.forward_override or self.forward_identity,
            hidden_layer=request.hidden_layer,
            hidden_states=hidden,
            deterministic_forward=True,
            policy_adapter_dropout=0.0,
        )


class _LayoutDependency:
    def __init__(self) -> None:
        self.calls = 0

    def build_replay_layout(self, request, source_visual):
        self.calls += 1
        del source_visual
        sequence = 12
        visible = torch.ones(1, sequence, dtype=torch.bool)
        return ReplayLayoutTensors(
            position_ids=torch.arange(sequence).view(1, sequence),
            attention_mask=visible,
            policy_visible_mask=visible.clone(),
            reference_visible_mask=visible.clone(),
            teacher_visible_mask=visible.clone(),
            token_type_ids=None,
            original_image_key_block_mask=None,
            cache_position=None,
            rope_delta=None,
            visual_layout=VisualLayout(
                sequence_length=sequence,
                original_image_positions=request.trajectory_source_visual.positions,
                d_positions=(7,),
                deepstack_branch_layers=BRANCH_LAYERS,
                deepstack_injection_positions=((7,), (7,), (7,)),
            ),
        )


def _source() -> SourceVisualTensorBundle:
    torch.manual_seed(123)
    return SourceVisualTensorBundle(
        image_sha256=SHA2,
        premerge_main=torch.randn(4, 4),
        premerge_deepstack=tuple(torch.randn(4, 4) for _ in BRANCH_LAYERS),
        merged_main=torch.randn(1, 8),
        merged_deepstack=tuple(torch.randn(1, 8) for _ in BRANCH_LAYERS),
        image_grid_thw=(1, 2, 2),
        spatial_merge_size=2,
    )


def _policy() -> PolicyVersion:
    return PolicyVersion("pilot", 4, SHA1)


def _sampled_turn():
    text = (
        "inspect</think>\n<tool_call>"
        '{"name":"tgvf_focus_tool","arguments":{"target":"red label"}}'
        "</tool_call>"
    )
    token_ids = tuple(ord(character) for character in text)
    spans = tuple(
        TokenByteSpan(index, token_id, index, index + 1)
        for index, token_id in enumerate(token_ids)
    )
    sampled = SampledPolicyTurn(
        text=text,
        token_ids=token_ids,
        token_byte_spans=spans,
        behavior_logprobs=tuple(-0.2 for _ in token_ids),
        sampling=SamplingIdentity(
            policy_version=_policy(),
            backend="vllm",
            backend_version="fixture",
            seed=42,
            rng_state_sha256=SHA2,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            logit_processors=(),
            measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
            asynchronous_staleness_steps=0,
        ),
        think_token_span=TokenSpan(0, text.index("</think>") + len("</think>")),
        stop_reason="stop",
        backend_request_sha256=SHA2,
        backend_response_sha256=SHA3,
    )
    return sampled, StrictToolCallParser().parse(sampled.parser_turn())


def _context(identity: TrajectoryIdentity, source_visual) -> ToolExecutionContext:
    sampled, _ = _sampled_turn()
    return ToolExecutionContext(
        trajectory_identity=identity,
        model=_model(),
        behavior_policy=_policy(),
        trajectory_source_visual=source_visual,
        prior_observation_handles=(),
        prompt_token_ids_before_turn=(7, 8, 9),
        sampled_turn=sampled,
        assistant_turn_index=0,
        attempt_index=0,
        call_index=0,
    )


def _branch_bindings(
    contract: RepresentationAdapterContractIdentity,
) -> tuple[BranchMergerRuntimeBinding, ...]:
    return tuple(
        BranchMergerRuntimeBinding(
            projection_identity=projection_identity,
            artifact=ArtifactIdentity(
                "qwen",
                f"deepstack-merger-{layer}",
                "fixture",
                str(index + 4) * 64,
            ),
        )
        for index, (layer, projection_identity) in enumerate(
            zip(
                contract.deepstack_branch_layers,
                contract.deepstack_projection_identities,
                strict=True,
            )
        )
    )


def _build_bridge(tmp_path: Path, *, provider: str):
    binding, adapter = _write_artifact(
        tmp_path / f"{provider}.pt",
        provider=provider,
    )
    store = ObservationStore()
    forward_identity = ArtifactIdentity(
        "policy",
        "behavior-hidden-forward",
        "fixture",
        "8" * 64,
    )
    hidden = _HiddenStateDependency(forward_identity)
    if provider == "contextual_hidden_state":
        dependencies = TargetConditioningDependencies()
        hidden_dependency = hidden
        selected_forward_identity = forward_identity
    else:
        embedding = nn.Embedding(binding.model.tokenizer_length, 8)
        embedding.requires_grad_(False)
        dependencies = TargetConditioningDependencies(base_embedding=embedding)
        hidden_dependency = None
        selected_forward_identity = None
    layout = _LayoutDependency()
    bridge = build_policy_pilot_focus_runtime(
        artifact_binding=binding,
        adapter=adapter,
        conditioning_dependencies=dependencies,
        contextual_hidden_state_dependency=hidden_dependency,
        contextual_forward_identity=selected_forward_identity,
        replay_layout_dependency=layout,
        branch_mergers=_branch_bindings(binding.adapter_contract),
        observation_store=store,
        execution_ledger=FocusExecutionLedger(),
        runtime_device=torch.device("cpu"),
    )
    return bridge, store, binding, hidden, layout


@pytest.mark.parametrize(
    ("provider", "expected_hidden_calls"),
    (("contextual_hidden_state", 2), ("target_token_embedding", 0)),
)
def test_bridge_freezes_adapter_runs_both_providers_and_records_all_branches(
    tmp_path: Path,
    provider: str,
    expected_hidden_calls: int,
) -> None:
    bridge, store, _binding, hidden, layout = _build_bridge(tmp_path, provider=provider)
    adapter = bridge.loaded_adapter.adapter
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    assert all(not module.training for module in adapter.modules())

    source = _source()
    first_identity = TrajectoryIdentity("pilot", "sample-a", 0, "group")
    second_identity = TrajectoryIdentity("pilot", "sample-b", 1, "group")
    first_source = record_trajectory_source_visual(
        trajectory_id=first_identity.canonical_id,
        source_visual=source,
        source_positions=(1,),
        deepstack_branch_layers=BRANCH_LAYERS,
        deepstack_injection_positions=((1,), (1,), (1,)),
        observation_store=store,
        preprocessed_pixel_values=torch.ones((4, 3), dtype=torch.float32),
    )
    second_source = record_trajectory_source_visual(
        trajectory_id=second_identity.canonical_id,
        source_visual=source,
        source_positions=(1,),
        deepstack_branch_layers=BRANCH_LAYERS,
        deepstack_injection_positions=((1,), (1,), (1,)),
        observation_store=store,
        preprocessed_pixel_values=torch.ones((4, 3), dtype=torch.float32),
    )
    sampled, parsed = _sampled_turn()
    first_context = _context(first_identity, first_source)
    assert first_context.sampled_turn == sampled
    first_handle = bridge.runtime.execute(parsed, first_context)
    assert bridge.runtime.execute(parsed, first_context) == first_handle
    second_handle = bridge.runtime.execute(
        parsed,
        _context(second_identity, second_source),
    )

    first = store.resolve_record(first_handle)
    second = store.resolve_record(second_handle)
    expected_forward = (
        hidden.forward_identity if provider == "contextual_hidden_state" else None
    )
    assert first.condition.contextual_forward_identity == expected_forward
    assert second.condition.contextual_forward_identity == expected_forward
    assert len(first.payload.deepstack) == len(first.branches) == 3
    assert first.layout.deepstack_branch_layers == BRANCH_LAYERS
    assert tuple(branch.layer for branch in first.branches) == BRANCH_LAYERS
    assert first.payload.main_d.address.digest == second.payload.main_d.address.digest
    assert tuple(ref.address.digest for ref in first.payload.deepstack) == tuple(
        ref.address.digest for ref in second.payload.deepstack
    )
    for ref in (first.payload.main_d, *first.payload.deepstack):
        value = store.resolve_verified_for_trajectory(
            ref, trajectory_id=first_identity.canonical_id
        )
        assert not value.requires_grad and value.grad_fn is None
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert hidden.calls == expected_hidden_calls
    assert layout.calls == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("artifact", "manifest digest"),
        ("run_id", "run ID"),
        ("run_identity", "run identity"),
        ("model", "cross-family"),
        ("provider", "provider"),
        ("architecture", "architecture"),
    ),
)
def test_artifact_identity_mismatch_fails_before_mutating_adapter(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    binding, adapter = _write_artifact(
        tmp_path / "adapter.pt", provider="contextual_hidden_state"
    )
    if mutation == "artifact":
        binding = replace(
            binding,
            artifact=replace(binding.artifact, sha256="f" * 64),
        )
    elif mutation == "run_id":
        binding = replace(binding, expected_run_id="another-run")
    elif mutation == "run_identity":
        binding = replace(binding, expected_run_identity_sha256="f" * 64)
    elif mutation == "model":
        binding = replace(binding, model=_model(family="qwen2_5_vl"))
    elif mutation == "provider":
        binding = replace(binding, conditioning=_conditioning("target_token_embedding"))
    else:
        binding = replace(
            binding,
            adapter_contract=replace(
                binding.adapter_contract,
                attention_dim=binding.adapter_contract.attention_dim + 1,
            ),
        )

    assert adapter.training
    assert any(parameter.requires_grad for parameter in adapter.parameters())
    with pytest.raises(IdentityMismatchError, match=message):
        load_frozen_tgvf_adapter(binding=binding, adapter=adapter)
    assert adapter.training
    assert any(parameter.requires_grad for parameter in adapter.parameters())


def test_runtime_adapter_architecture_mismatch_fails_closed(tmp_path: Path) -> None:
    binding, _ = _write_artifact(
        tmp_path / "adapter.pt", provider="contextual_hidden_state"
    )
    mismatched = _adapter(99, identity_prefix="other-qwen")
    with pytest.raises(IdentityMismatchError, match="architecture/projection"):
        load_frozen_tgvf_adapter(binding=binding, adapter=mismatched)


def test_composer_preflights_layout_dependency_before_mutating_adapter(
    tmp_path: Path,
) -> None:
    binding, adapter = _write_artifact(
        tmp_path / "adapter.pt", provider="contextual_hidden_state"
    )
    forward_identity = ArtifactIdentity(
        "policy", "behavior-hidden-forward", "fixture", "8" * 64
    )
    before_state = {
        name: tensor.detach().clone() for name, tensor in adapter.state_dict().items()
    }
    before_grad_flags = tuple(
        (name, parameter.requires_grad)
        for name, parameter in adapter.named_parameters()
    )
    before_training_flags = tuple(
        (name, module.training) for name, module in adapter.named_modules()
    )

    with pytest.raises(TypeError, match="build_replay_layout"):
        build_policy_pilot_focus_runtime(
            artifact_binding=binding,
            adapter=adapter,
            conditioning_dependencies=TargetConditioningDependencies(),
            contextual_hidden_state_dependency=_HiddenStateDependency(forward_identity),
            contextual_forward_identity=forward_identity,
            replay_layout_dependency=object(),
            branch_mergers=_branch_bindings(binding.adapter_contract),
            observation_store=ObservationStore(),
            execution_ledger=FocusExecutionLedger(),
            runtime_device=torch.device("cpu"),
        )

    assert (
        tuple(
            (name, parameter.requires_grad)
            for name, parameter in adapter.named_parameters()
        )
        == before_grad_flags
    )
    assert (
        tuple((name, module.training) for name, module in adapter.named_modules())
        == before_training_flags
    )
    for name, tensor in adapter.state_dict().items():
        torch.testing.assert_close(tensor, before_state[name], rtol=0, atol=0)


def test_contextual_behavior_policy_and_forward_identity_must_match(
    tmp_path: Path,
) -> None:
    bridge, store, _binding, hidden, _layout = _build_bridge(
        tmp_path, provider="contextual_hidden_state"
    )
    identity = TrajectoryIdentity("pilot", "sample", 0, "group")
    source = record_trajectory_source_visual(
        trajectory_id=identity.canonical_id,
        source_visual=_source(),
        source_positions=(1,),
        deepstack_branch_layers=BRANCH_LAYERS,
        deepstack_injection_positions=((1,), (1,), (1,)),
        observation_store=store,
        preprocessed_pixel_values=torch.ones((4, 3), dtype=torch.float32),
    )
    context = _context(identity, source)
    _, parsed = _sampled_turn()
    hidden.policy_override = PolicyVersion("pilot", 5, "9" * 64)
    with pytest.raises(IdentityMismatchError, match="different behavior policy"):
        bridge.runtime.execute(parsed, context)

    hidden.policy_override = None
    hidden.forward_override = ArtifactIdentity(
        "policy", "other-forward", "fixture", "9" * 64
    )
    with pytest.raises(IdentityMismatchError, match="forward identity"):
        bridge.runtime.execute(parsed, context)


def test_behavior_hidden_state_contract_rejects_nondeterminism_and_dropout() -> None:
    base = dict(
        policy_version=_policy(),
        forward_identity=ArtifactIdentity("policy", "forward", "fixture", "8" * 64),
        hidden_layer=-1,
        hidden_states=torch.zeros(2, 8),
        deterministic_forward=True,
        policy_adapter_dropout=0.0,
    )
    with pytest.raises(ValueError, match="deterministic"):
        BehaviorHiddenStateMaterialization(**(base | {"deterministic_forward": False}))
    with pytest.raises(ValueError, match="zero policy dropout"):
        BehaviorHiddenStateMaterialization(**(base | {"policy_adapter_dropout": 0.1}))


def test_source_port_rejects_released_trajectory_when_digest_is_still_shared(
    tmp_path: Path,
) -> None:
    bridge, store, _binding, _hidden, _layout = _build_bridge(
        tmp_path, provider="target_token_embedding"
    )
    source = _source()
    released_identity = TrajectoryIdentity("pilot", "released", 0, "group")
    live_identity = TrajectoryIdentity("pilot", "live", 1, "group")
    released_source = record_trajectory_source_visual(
        trajectory_id=released_identity.canonical_id,
        source_visual=source,
        source_positions=(1,),
        deepstack_branch_layers=BRANCH_LAYERS,
        deepstack_injection_positions=((1,), (1,), (1,)),
        observation_store=store,
        preprocessed_pixel_values=torch.ones((4, 3), dtype=torch.float32),
    )
    live_source = record_trajectory_source_visual(
        trajectory_id=live_identity.canonical_id,
        source_visual=source,
        source_positions=(1,),
        deepstack_branch_layers=BRANCH_LAYERS,
        deepstack_injection_positions=((1,), (1,), (1,)),
        observation_store=store,
        preprocessed_pixel_values=torch.ones((4, 3), dtype=torch.float32),
    )
    store.release_trajectories((released_identity.canonical_id,))
    assert store.resource_counts().tensors > 0
    _, parsed = _sampled_turn()
    with pytest.raises(ReplayMismatchError, match="released"):
        bridge.runtime.execute(
            parsed,
            _context(released_identity, released_source),
        )
    # The same content remains valid for its still-live owner.
    bridge.runtime.execute(parsed, _context(live_identity, live_source))
