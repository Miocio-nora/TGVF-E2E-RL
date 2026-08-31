from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningProviderKind,
    TargetConditioningRequest,
)
from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.contracts.tokens import TokenSpan
from tgvf_rl.representation import TGVFAdapterVariant
from tgvf_rl.representation.training.runtime import (
    QWEN3_PATCH_EMBED_LINEAR_FAST_PATH,
    QWEN3_REPRESENTATION_BRANCH_LAYERS,
    Qwen3ContextualHiddenStateStack,
    Qwen3VisionPreMergeRequest,
    create_qwen3_representation_runtime,
    install_qwen3_patch_embed_linear_fast_path,
    qwen3_input_embedding_identity,
)


class _Tokenizer:
    def __init__(self, *, length: int, template: str, name_or_path: str) -> None:
        self.length = length
        self.chat_template = template
        self.name_or_path = name_or_path

    def __len__(self) -> int:
        return self.length

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(char) % self.length for char in text]


class _Processor:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template

    def apply_chat_template(self, *args, **kwargs) -> str:
        return "rendered"


class _Merger(nn.Module):
    def __init__(self, vision_width: int, language_width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(vision_width * 4, language_width, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states.reshape(-1, hidden_states.shape[-1] * 4))


class _VisionTower(nn.Module):
    def __init__(
        self,
        *,
        patch_width: int,
        vision_width: int,
        language_width: int,
        skip_branch: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(patch_width, vision_width, bias=False)
        self.merger = _Merger(vision_width, language_width)
        self.deepstack_merger_list = nn.ModuleList(
            [_Merger(vision_width, language_width) for _ in range(3)]
        )
        self.skip_branch = skip_branch

    def forward(
        self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        assert grid_thw.shape == (1, 3)
        hidden = self.patch_projection(pixel_values)
        deepstack = []
        for index, merger in enumerate(self.deepstack_merger_list):
            if index != self.skip_branch:
                deepstack.append(merger(hidden + index + 1))
        return self.merger(hidden), deepstack


class _ProductionGeometryPatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 3
        self.temporal_patch_size = 2
        self.patch_size = 16
        self.embed_dim = 1152
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(values.to(dtype=self.proj.weight.dtype)).view(
            -1, self.embed_dim
        )


class _ProductionGeometryVisionShell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _ProductionGeometryPatchEmbed()


class _LanguageModel(nn.Module):
    def __init__(self, tokenizer_length: int, language_width: int) -> None:
        super().__init__()
        # Qwen is allowed to have padded rows beyond len(tokenizer).
        self.embed_tokens = nn.Embedding(tokenizer_length + 4, language_width)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class _QwenCore(nn.Module):
    def __init__(self, visual: _VisionTower, language_model: _LanguageModel) -> None:
        super().__init__()
        self.visual = visual
        self.language_model = language_model


class _TinyQwen3(nn.Module):
    def __init__(
        self,
        *,
        name_or_path: str,
        tokenizer_length: int = 16,
        vision_width: int = 4,
        language_width: int = 6,
        skip_branch: int | None = None,
    ) -> None:
        super().__init__()
        self.model = _QwenCore(
            _VisionTower(
                patch_width=3,
                vision_width=vision_width,
                language_width=language_width,
                skip_branch=skip_branch,
            ),
            _LanguageModel(tokenizer_length, language_width),
        )
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            _name_or_path=name_or_path,
            vision_config=SimpleNamespace(
                hidden_size=vision_width,
                out_hidden_size=language_width,
                spatial_merge_size=2,
                deepstack_visual_indexes=QWEN3_REPRESENTATION_BRANCH_LAYERS,
            ),
            text_config=SimpleNamespace(hidden_size=language_width),
        )


def _identity(*, family: str = "qwen3_vl") -> ModelIdentity:
    template = "tiny-qwen3-template"
    return ModelIdentity(
        family=family,
        model_name="tiny-qwen3",
        revision_or_path="/tiny-qwen3",
        tokenizer_length=16,
        chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
    )


def _processor() -> _Processor:
    return _Processor(
        _Tokenizer(
            length=16,
            template="tiny-qwen3-template",
            name_or_path="/tiny-qwen3",
        )
    )


def _context_config() -> TargetConditioningConfig:
    return TargetConditioningConfig(
        provider=TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE,
        hidden_layer=-1,
    )


def _embedding_config(identity: ModelIdentity) -> TargetConditioningConfig:
    return TargetConditioningConfig(
        provider=TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING,
        embedding_identity=qwen3_input_embedding_identity(identity),
    )


def _target_request(identity: ModelIdentity) -> TargetConditioningRequest:
    return TargetConditioningRequest(
        input_ids=torch.tensor([1, 5, 6, 2]),
        target_span=TokenSpan(1, 3),
        expected_target_token_ids=(5, 6),
        trajectory_id="trajectory-a",
        call_index=0,
        model_identity=identity,
    )


def _vision_request() -> Qwen3VisionPreMergeRequest:
    return Qwen3VisionPreMergeRequest(
        pixel_values=torch.arange(12, dtype=torch.float32).view(4, 3),
        image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),
    )


def test_factory_freezes_qwen_and_borrows_exact_four_mergers() -> None:
    identity = _identity()
    model = _TinyQwen3(name_or_path=identity.revision_or_path)
    runtime = create_qwen3_representation_runtime(
        model=model,
        processor=_processor(),
        model_identity=identity,
        conditioning_config=_context_config(),
        adapter_dtype=torch.float32,
        fixture_mode=True,
    )

    runtime.assert_bound_invariants()
    assert not model.training
    assert all(not module.training for module in model.modules())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert runtime.adapter.main_projection.projection is model.model.visual.merger
    assert tuple(
        port.projection for port in runtime.adapter.d_deepstack_projections.projections
    ) == tuple(model.model.visual.deepstack_merger_list)
    owned = runtime.adapter.artifact_state_dict(keep_vars=True)
    assert owned
    assert all(tensor.requires_grad for tensor in owned.values())
    assert runtime.patch_embed_fast_path is None


def test_factory_builds_isolated_full_vision_routing_variant() -> None:
    identity = _identity()
    model = _TinyQwen3(name_or_path=identity.revision_or_path)

    runtime = create_qwen3_representation_runtime(
        model=model,
        processor=_processor(),
        model_identity=identity,
        conditioning_config=_context_config(),
        adapter_dtype=torch.float32,
        adapter_variant=(TGVFAdapterVariant.FULL_D_DEEPSTACK_VISION_ROUTING),
        fixture_mode=True,
    )

    assert runtime.adapter.variant is (
        TGVFAdapterVariant.FULL_D_DEEPSTACK_VISION_ROUTING
    )
    assert runtime.adapter.vision_routing_only
    assert len(runtime.adapter.d_deepstack_branch_adapters) == 3
    assert len(runtime.adapter.artifact_state_dict()) == 104


def test_factory_builds_isolated_visual_barycentric_variant() -> None:
    identity = _identity()
    model = _TinyQwen3(name_or_path=identity.revision_or_path)

    runtime = create_qwen3_representation_runtime(
        model=model,
        processor=_processor(),
        model_identity=identity,
        conditioning_config=_context_config(),
        adapter_dtype=torch.float32,
        adapter_variant=(TGVFAdapterVariant.FULL_D_DEEPSTACK_VISUAL_BARYCENTRIC),
        fixture_mode=True,
    )

    assert runtime.adapter.variant is (
        TGVFAdapterVariant.FULL_D_DEEPSTACK_VISUAL_BARYCENTRIC
    )
    assert runtime.adapter.vision_routing_only
    assert runtime.adapter.visual_barycentric_writer
    assert len(runtime.adapter.d_deepstack_branch_adapters) == 3
    assert len(runtime.adapter.artifact_state_dict()) == 104


def test_patch_embed_linear_fast_path_preserves_state_and_parameter_identity() -> None:
    torch.manual_seed(803)
    vision = _ProductionGeometryVisionShell().eval()
    patches = torch.randn(7, 3 * 2 * 16 * 16)
    native = vision.patch_embed(patches)
    parameters_before = tuple(
        (name, id(parameter)) for name, parameter in vision.named_parameters()
    )
    state_before = {
        name: value.detach().clone() for name, value in vision.state_dict().items()
    }

    installed = install_qwen3_patch_embed_linear_fast_path(vision)
    linear = vision.patch_embed(patches)

    assert installed is vision.patch_embed
    assert installed._tgvf_fast_path_identity == (  # type: ignore[attr-defined]
        QWEN3_PATCH_EMBED_LINEAR_FAST_PATH
    )
    assert (
        tuple((name, id(parameter)) for name, parameter in vision.named_parameters())
        == parameters_before
    )
    assert tuple(vision.state_dict()) == tuple(state_before)
    for name, expected in state_before.items():
        assert torch.equal(vision.state_dict()[name], expected)
    torch.testing.assert_close(linear, native, atol=1.0e-5, rtol=1.0e-5)
    assert linear.shape == (7, 1152)
    assert install_qwen3_patch_embed_linear_fast_path(vision) is installed


def test_patch_embed_linear_fast_path_fails_closed_on_geometry_or_input_drift() -> None:
    vision = _ProductionGeometryVisionShell().eval()
    vision.patch_embed.patch_size = 8
    with pytest.raises(ValueError, match="attributes changed"):
        install_qwen3_patch_embed_linear_fast_path(vision)

    vision.patch_embed.patch_size = 16
    install_qwen3_patch_embed_linear_fast_path(vision)
    with pytest.raises(ValueError, match=r"shape \[N,1536\]"):
        vision.patch_embed(torch.randn(2, 64))


def test_contextual_hq_and_vision_features_form_one_typed_adapter_input() -> None:
    identity = _identity()
    model = _TinyQwen3(name_or_path=identity.revision_or_path)
    runtime = create_qwen3_representation_runtime(
        model=model,
        processor=_processor(),
        model_identity=identity,
        conditioning_config=_context_config(),
        adapter_dtype=torch.float32,
        fixture_mode=True,
    )
    first = torch.randn(4, 6)
    final = torch.randn(4, 6)
    stack = Qwen3ContextualHiddenStateStack((first, final))

    condition = runtime.build_target_condition(
        _target_request(identity), contextual_hidden_states=stack
    )
    vision = runtime.extract_vision_features(_vision_request())
    adapter_input = runtime.make_adapter_input(condition, vision)
    output = runtime.adapter(adapter_input)

    torch.testing.assert_close(condition.hq, final[1:3])
    assert condition.provenance.hidden_layer == -1
    assert vision.pre_merge_main.shape == (4, 4)
    assert all(branch.shape == (4, 4) for branch in vision.pre_merge_deepstack)
    assert vision.merged_main.shape == (1, 6)
    assert all(branch.shape == (1, 6) for branch in vision.merged_deepstack)
    assert output.main_d.shape == (1, 6)
    assert len(output.deepstack_visual_embeds) == 3
    assert not any(tensor.requires_grad for tensor in vision.pre_merge_deepstack)
    assert not model.model.visual.merger._forward_hooks
    assert all(
        not merger._forward_hooks for merger in model.model.visual.deepstack_merger_list
    )


def test_target_embedding_provider_uses_only_valid_rows_of_padded_qwen_embedding() -> (
    None
):
    identity = _identity()
    model = _TinyQwen3(name_or_path=identity.revision_or_path)
    runtime = create_qwen3_representation_runtime(
        model=model,
        processor=_processor(),
        model_identity=identity,
        conditioning_config=_embedding_config(identity),
        adapter_dtype=torch.float32,
        fixture_mode=True,
    )

    condition = runtime.build_target_condition(_target_request(identity))
    expected = model.model.language_model.embed_tokens(torch.tensor([5, 6]))
    torch.testing.assert_close(condition.hq, expected)
    assert condition.provenance.embedding_identity == qwen3_input_embedding_identity(
        identity
    )
    assert not condition.hq.requires_grad
    assert model.model.language_model.embed_tokens.num_embeddings == 20
    assert len(runtime.tokenizer) == 16


def test_runtime_rejects_qwen25_and_tiny_model_outside_fixture_mode() -> None:
    qwen25 = _identity(family="qwen25_vl")
    with pytest.raises(ValueError, match="only qwen3_vl"):
        create_qwen3_representation_runtime(
            model=_TinyQwen3(name_or_path=qwen25.revision_or_path),
            processor=_processor(),
            model_identity=qwen25,
            conditioning_config=_context_config(),
            adapter_dtype=torch.float32,
            fixture_mode=True,
        )

    tiny = _identity()
    with pytest.raises(
        ValueError, match="production runtime requires a pinned Qwen3 edition"
    ):
        create_qwen3_representation_runtime(
            model=_TinyQwen3(name_or_path=tiny.revision_or_path),
            processor=_processor(),
            model_identity=tiny,
            conditioning_config=_context_config(),
            adapter_dtype=torch.float32,
        )


def test_runtime_rejects_identity_provider_and_hidden_state_drift() -> None:
    identity = _identity()
    processor = _processor()
    runtime = create_qwen3_representation_runtime(
        model=_TinyQwen3(name_or_path=identity.revision_or_path),
        processor=processor,
        model_identity=identity,
        conditioning_config=_context_config(),
        adapter_dtype=torch.float32,
        fixture_mode=True,
    )
    request = _target_request(identity)
    direct = TargetConditioningRequest(
        input_ids=request.input_ids,
        target_span=request.target_span,
        expected_target_token_ids=request.expected_target_token_ids,
        trajectory_id=request.trajectory_id,
        call_index=request.call_index,
        model_identity=request.model_identity,
        contextual_hidden_states=torch.randn(4, 6),
    )
    with pytest.raises(ValueError, match="must not bypass"):
        runtime.build_target_condition(direct)
    with pytest.raises(ValueError, match="state stack"):
        runtime.build_target_condition(request)

    processor.chat_template = "mutated-template"
    with pytest.raises(ValueError, match="chat template changed"):
        runtime.assert_bound_invariants()


def test_public_runtime_entry_still_rejects_invariant_drift() -> None:
    identity = _identity()
    model = _TinyQwen3(name_or_path=identity.revision_or_path)
    runtime = create_qwen3_representation_runtime(
        model=model,
        processor=_processor(),
        model_identity=identity,
        conditioning_config=_embedding_config(identity),
        adapter_dtype=torch.float32,
        fixture_mode=True,
    )
    model.config._name_or_path = "/mutated-before-public-call"

    with pytest.raises(ValueError, match="model path differs"):
        runtime.build_target_condition(_target_request(identity))


def test_vision_capture_fails_closed_when_a_deepstack_merger_is_not_executed() -> None:
    identity = _identity()
    model = _TinyQwen3(name_or_path=identity.revision_or_path, skip_branch=1)
    runtime = create_qwen3_representation_runtime(
        model=model,
        processor=_processor(),
        model_identity=identity,
        conditioning_config=_context_config(),
        adapter_dtype=torch.float32,
        fixture_mode=True,
    )

    with pytest.raises(RuntimeError, match="execute exactly once"):
        runtime.extract_vision_features(_vision_request())
    assert not model.model.visual.merger._forward_hooks
    assert all(
        not merger._forward_hooks for merger in model.model.visual.deepstack_merger_list
    )
